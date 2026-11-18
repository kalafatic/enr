package ai.enr.cli;

import ai.enr.client.EnrClient;
import ai.enr.core.json.Json;
import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonObject;
import ai.enr.runtime.EnrRuntime;
import ai.enr.runtime.RuntimeConfig;
import ai.enr.server.EnrServer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.logging.ConsoleHandler;
import java.util.logging.Formatter;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

/**
 * ENR command line interface.
 *
 * Commands:
 *   enr start [-c config.yaml] [-p port] [-H host] [--cuda|--cpu]
 *   enr stop
 *   enr status
 *   enr models
 *   enr load <path> [--backend cpu|cuda]
 *   enr unload <modelId> [--force]
 *   enr infer <modelId> "<prompt>" [--max-tokens N] [--temperature F] [--stream] [--session S]
 *   enr devices
 *   enr diagnostics
 */
public final class EnrCli {

    private static final Logger LOG = Logger.getLogger("ai.enr");

    public static void main(String[] args) {
        configureLogging();
        int exit = new EnrCli().run(args);
        System.exit(exit);
    }

    private int run(String[] args) {
        if (args.length == 0) {
            printUsage();
            return 1;
        }
        String command = args[0];
        String[] rest = java.util.Arrays.copyOfRange(args, 1, args.length);
        try {
            return switch (command) {
                case "start" -> start(rest);
                case "stop" -> stop(rest);
                case "status" -> client(rest).map(c -> {
                    System.out.println(Json.write(c.status(), true));
                    return 0;
                }).orElse(1);
                case "models" -> client(rest).map(c -> {
                    var models = c.models();
                    if (models.isEmpty()) {
                        System.out.println("No models loaded.");
                    }
                    for (var model : models) {
                        System.out.printf("%-20s %-8s %-10s params=%-10d backend=%s sessions=%d%n",
                                model.id(), model.state(), model.architecture(),
                                model.parameters(), model.backend(), model.activeSessions());
                    }
                    return 0;
                }).orElse(1);
                case "load" -> load(rest);
                case "unload" -> unload(rest);
                case "infer" -> infer(rest);
                case "devices" -> devices(rest);
                case "diagnostics" -> diagnostics(rest);
                case "help", "--help", "-h" -> {
                    printUsage();
                    yield 0;
                }
                default -> {
                    System.err.println("Unknown command: " + command);
                    printUsage();
                    yield 1;
                }
            };
        } catch (Exception e) {
            System.err.println("ERROR: " + e.getMessage());
            return 1;
        }
    }

    private int start(String[] args) throws Exception {
        String host = null;
        int port = -1;
        Path configPath = null;
        String backend = null;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-H", "--host" -> host = args[++i];
                case "-p", "--port" -> port = Integer.parseInt(args[++i]);
                case "-c", "--config" -> configPath = Path.of(args[++i]);
                case "--cuda" -> backend = "cuda";
                case "--cpu" -> backend = "cpu";
                default -> throw new IllegalArgumentException("Unknown start option: " + args[i]);
            }
        }
        JsonObject config = CliConfig.load(configPath);
        RuntimeConfig.Builder builder = RuntimeConfig.builder();
        JsonObject server = config.getObject("server");
        if (server != null) {
            if (host == null) {
                host = server.getString("host");
            }
            if (port < 0) {
                port = server.getInt("port", -1);
            }
        }
        JsonObject runtime = config.getObject("runtime");
        if (runtime != null) {
            if (backend == null) {
                backend = runtime.getString("preferred_backend");
            }
            builder.preferredBackend(backend != null ? backend : "cpu");
            builder.modelDirectories(CliConfig.modelDirectories(config));
            builder.maxConcurrentSessions(runtime.getInt("max_concurrent_sessions",
                    builder.build().maxConcurrentSessions()));
            builder.defaultMaxTokens(runtime.getInt("default_max_tokens", builder.build().defaultMaxTokens()));
        }
        RuntimeConfig runtimeConfig = builder.host(host != null ? host : "127.0.0.1")
                .port(port > 0 ? port : 8111)
                .build();

        EnrRuntime runtimeInstance = new EnrRuntime(runtimeConfig);
        for (Path dir : runtimeConfig.modelDirectories()) {
            for (Path modelFile : runtimeInstance.models().discoverModelFiles()) {
                System.out.println("Auto-loading model: " + modelFile);
                runtimeInstance.loadModel(modelFile);
            }
        }
        EnrServer serverInstance = new EnrServer(runtimeInstance, runtimeConfig.host(), runtimeConfig.port());
        serverInstance.start();
        System.out.println("ENR started on " + runtimeConfig.host() + ":" + serverInstance.port());

        Path pidFile = Path.of("/tmp/enr.pid");
        try {
            Files.writeString(pidFile, String.valueOf(ProcessHandle.current().pid()));
        } catch (IOException e) {
            System.err.println("Warning: could not write pid file: " + e.getMessage());
        }

        CountDownLatch latch = new CountDownLatch(1);
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("Shutting down ENR...");
            serverInstance.close();
            runtimeInstance.shutdown();
            try {
                Files.deleteIfExists(pidFile);
            } catch (IOException ignored) {
                // best effort
            }
            latch.countDown();
        }));
        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        return 0;
    }

    private int stop(String[] args) throws IOException, InterruptedException {
        String pidFile = "/tmp/enr.pid";
        Path pidPath = Path.of(pidFile);
        if (Files.exists(pidPath)) {
            String pid = Files.readString(pidPath).strip();
            new ProcessBuilder("kill", pid).inheritIO().start().waitFor();
            Files.deleteIfExists(pidPath);
            System.out.println("ENR stop signal sent (pid " + pid + ")");
            return 0;
        }
        System.err.println("No ENR pid file found; ENR may not be running or was started without pid tracking.");
        return 1;
    }

    private int load(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: enr load <path> [--backend cpu|cuda]");
            return 1;
        }
        String path = args[0];
        String backend = null;
        for (int i = 1; i < args.length; i++) {
            if ("--backend".equals(args[i])) {
                backend = args[++i];
            }
        }
        var client = client(new String[0]).orElseThrow();
        var model = client.loadModel(path, backend);
        System.out.println("Loaded model: " + model.id() + " (" + model.parameters() + " params, backend "
                + model.backend() + ")");
        return 0;
    }

    private int unload(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: enr unload <modelId> [--force]");
            return 1;
        }
        String id = args[0];
        boolean force = false;
        for (int i = 1; i < args.length; i++) {
            if ("--force".equals(args[i])) {
                force = true;
            }
        }
        var client = client(new String[0]).orElseThrow();
        client.unloadModel(id, force);
        System.out.println("Unloaded model: " + id);
        return 0;
    }

    private int infer(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: enr infer <modelId> \"<prompt>\" [--max-tokens N] [--temperature F] [--stream]");
            return 1;
        }
        String modelId = args[0];
        String prompt = args[1];
        var request = new EnrClient.InferenceRequest();
        boolean stream = false;
        for (int i = 2; i < args.length; i++) {
            switch (args[i]) {
                case "--max-tokens" -> request.maxTokens(Integer.parseInt(args[++i]));
                case "--temperature" -> request.temperature(Float.parseFloat(args[++i]));
                case "--top-p" -> request.topP(Float.parseFloat(args[++i]));
                case "--top-k" -> request.topK(Integer.parseInt(args[++i]));
                case "--seed" -> request.seed(Long.parseLong(args[++i]));
                case "--session" -> request.session(args[++i]);
                case "--stream" -> stream = true;
                default -> throw new IllegalArgumentException("Unknown infer option: " + args[i]);
            }
        }
        var client = client(new String[0]).orElseThrow();
        if (stream) {
            client.inferStream(modelId, prompt, request,
                    token -> System.out.print(token.text()),
                    result -> System.out.println("\n[reason=" + result.reason() + ", "
                            + result.generatedTokens() + " tokens, " + result.elapsedMs() + " ms]"));
        } else {
            var result = client.infer(modelId, prompt, request);
            System.out.println(result.text());
            System.out.println("[reason=" + result.reason() + ", prompt=" + result.promptTokens()
                    + ", generated=" + result.generatedTokens() + ", " + result.elapsedMs() + " ms]");
        }
        return 0;
    }

    private int devices(String[] args) {
        var client = client(args).orElseThrow();
        for (var device : client.devices()) {
            System.out.printf("%s:%d %s available=%s total=%dMB free=%dMB driver=%s%n",
                    device.type(), device.index(), device.name(), device.available(),
                    device.totalMemoryBytes() >> 20, device.freeMemoryBytes() >> 20,
                    device.driverVersion());
        }
        return 0;
    }

    private int diagnostics(String[] args) {
        var client = client(args).orElseThrow();
        System.out.println("=== ENR Diagnostics ===");
        System.out.println(Json.write(client.status(), true));
        System.out.println("=== Models ===");
        for (var model : client.models()) {
            System.out.printf("  %s: %s, %d params, backend=%s, state=%s, active=%d%n",
                    model.id(), model.architecture(), model.parameters(), model.backend(),
                    model.state(), model.activeSessions());
        }
        System.out.println("=== Devices ===");
        for (var device : client.devices()) {
            System.out.printf("  %s:%d %s available=%s%n", device.type(), device.index(),
                    device.name(), device.available());
        }
        System.out.println("=== Metrics ===");
        System.out.println(Json.write(client.metrics(), true));
        return 0;
    }

    private java.util.Optional<EnrClient> client(String[] args) {
        String url = null;
        for (int i = 0; i < args.length; i++) {
            if ("--url".equals(args[i])) {
                url = args[++i];
            }
        }
        String base = url != null ? url : "http://127.0.0.1:8111";
        return java.util.Optional.of(new EnrClient(base));
    }

    private static void printUsage() {
        System.out.println("""
                EVO Native Runtime (ENR) CLI

                Usage:
                  enr start [-c config.yaml] [-p port] [-H host] [--cuda|--cpu]
                  enr stop
                  enr status [--url http://host:port]
                  enr models
                  enr load <path> [--backend cpu|cuda]
                  enr unload <modelId> [--force]
                  enr infer <modelId> "<prompt>" [--max-tokens N] [--temperature F] [--stream] [--session S]
                  enr devices
                  enr diagnostics
                """);
    }

    private static void configureLogging() {
        Logger root = Logger.getLogger("");
        for (java.util.logging.Handler handler : root.getHandlers()) {
            root.removeHandler(handler);
        }
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.INFO);
        handler.setFormatter(new Formatter() {
            @Override
            public String format(LogRecord record) {
                return java.time.Instant.ofEpochMilli(record.getMillis())
                        + " [" + record.getLevel().getName() + "] " + record.getMessage() + "\n";
            }
        });
        root.addHandler(handler);
        root.setLevel(Level.INFO);
    }
}
