package ai.enr.cli;

import ai.enr.client.EnrClient;
import ai.enr.core.json.Json;
import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonObject;
import ai.enr.manager.EnrConfig;
import ai.enr.manager.EnrConfigLoader;
import ai.enr.manager.EnrManager;
import ai.enr.manager.logs.LoggingConfig;
import ai.enr.manager.service.SystemdSupport;
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
 *   enr serve [-c config.yaml] [-p port] [-H host] [--cuda|--cpu]   foreground server (systemd)
 *   enr start [-c config.yaml] [-p port] [-H host] [--cuda|--cpu]   server with pid file
 *   enr stop
 *   enr status | models | devices | diagnostics
 *   enr list                              installed models
 *   enr pull <ref> [--no-stream]          download from Ollama Library / Hugging Face
 *   enr remove <id>
 *   enr import <path> [--id id]
 *   enr load <path> | unload <id> | infer <model> "<prompt>"
 *   enr chat <model> ["prompt"] [--max-tokens N] [--temperature F] [--multi]
 *   enr provider add|list|remove|test|models <...>
 *   enr service install|uninstall|status|unit [--config PATH] [--jar PATH] [--user U]
 */
public final class EnrCli {

    private static final Logger LOG = Logger.getLogger("ai.enr");

    public static void main(String[] args) {
        if (LoggingConfig.conflicting(args)) {
            System.err.println("ERROR: --logging and --no-logging are mutually exclusive");
            System.exit(2);
            return;
        }
        // --no-logging suppresses routine application logging but keeps errors
        // (SEVERE) on stderr. --logging (and the default) enable INFO logging.
        configureLogging(LoggingConfig.level(args));
        int exit = new EnrCli().run(LoggingConfig.strip(args));
        System.exit(exit);
    }

    private int run(String[] args) {
        if (args.length == 0) {
            return launchMenu();
        }
        String command = args[0];
        String[] rest = java.util.Arrays.copyOfRange(args, 1, args.length);
        try {
            return switch (command) {
                case "serve" -> serve(rest, false);
                case "start" -> start(rest);
                case "stop" -> stop(rest);
                case "restart" -> restart(rest);
                case "status" -> status(rest);
                case "logs" -> logs(rest);
                case "models" -> modelsCommand(rest);
                case "list" -> list(rest);
                case "ps" -> ps(rest);
                case "show" -> show(rest);
                case "run" -> runModel(rest);
                case "pull" -> pull(rest);
                case "remove", "rm" -> remove(rest);
                case "copy" -> copy(rest);
                case "import" -> importModel(rest);
                case "load" -> load(rest);
                case "unload" -> unload(rest);
                case "infer" -> infer(rest);
                case "chat" -> chat(rest);
                case "version" -> version(rest);
                case "devices" -> devices(rest);
                case "diagnostics" -> diagnostics(rest);
                case "provider", "providers" -> providers(rest);
                case "service" -> service(rest);
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

    // ------------------------------------------------------------ server

    /**
     * Interactive first-launch menu shown when the JAR is run with no arguments
     * (double-click / {@code java -jar enr.jar}). Defaults to option 1 on a
     * blank line. When stdin is not a terminal the usage text is printed instead.
     */
    private int launchMenu() {
        if (System.console() == null) {
            printUsage();
            return 1;
        }
        System.out.println("ENR - EVO Native Runtime");
        System.out.println();
        System.out.println("Welcome! Choose how to run ENR:");
        System.out.println();
        System.out.println("[1] Install as system service");
        System.out.println("    Permanent installation, automatic OS startup.");
        System.out.println();
        System.out.println("[2] Run directly in this terminal");
        System.out.println("    No installation, no administrator privileges.");
        System.out.println();
        System.out.println("[0] Exit");
        System.out.println();
        String line = System.console().readLine("Selection [1]: ");
        String choice = line == null ? "" : line.strip();
        if (choice.isEmpty()) {
            choice = "1";
        }
        return switch (choice) {
            case "1" -> installServiceInteractive();
            case "2" -> {
                try {
                    yield serve(new String[0], false);
                } catch (Exception e) {
                    System.err.println("ERROR: " + e.getMessage());
                    yield 1;
                }
            }
            case "0" -> 0;
            default -> {
                System.out.println("Invalid selection: " + choice);
                yield 1;
            }
        };
    }

    /** Installs ENR as a service (menu option 1), then starts and verifies it. */
    private int installServiceInteractive() {
        try {
            Path jar = currentJarPath();
            if (jar == null) {
                System.err.println("Could not determine the ENR JAR path.");
                return 1;
            }
            System.out.println("Installing ENR as a service...");
            ServiceManager.install(jar, "127.0.0.1", 8111);
            return 0;
        } catch (Exception e) {
            System.err.println("Installation failed: " + e.getMessage());
            return 1;
        }
    }

    /** Resolves the running JAR path (for {@code java -jar enr.jar}). */
    private static Path currentJarPath() {
        String cp = System.getProperty("java.class.path", "");
        if (cp != null && cp.endsWith(".jar") && Files.exists(Path.of(cp))) {
            return Path.of(cp).toAbsolutePath();
        }
        String cmd = ProcessHandle.current().info().commandLine().orElse("");
        for (String token : cmd.split(" ")) {
            if (token.endsWith(".jar") && Files.exists(Path.of(token))) {
                return Path.of(token).toAbsolutePath();
            }
        }
        return SystemdSupport.defaultJarPath();
    }

    /** enr models [providers|installed <p>|available <p>|pull <p> <m>|status <job>|cancel <job>|remove <p> <m>] */
    private int modelsCommand(String[] args) {
        String sub = args.length > 0 && !args[0].startsWith("-") ? args[0] : "";
        switch (sub) {
            case "providers" -> {
                return modelProviderList(java.util.Arrays.copyOfRange(args, 1, args.length));
            }
            case "installed" -> {
                return modelProviderModels("installed", java.util.Arrays.copyOfRange(args, 1, args.length));
            }
            case "available" -> {
                return modelProviderModels("available", java.util.Arrays.copyOfRange(args, 1, args.length));
            }
            case "pull" -> {
                return modelPull(java.util.Arrays.copyOfRange(args, 1, args.length));
            }
            case "status" -> {
                return modelJobStatus(java.util.Arrays.copyOfRange(args, 1, args.length));
            }
            case "cancel" -> {
                return modelJobCancel(java.util.Arrays.copyOfRange(args, 1, args.length));
            }
            case "remove" -> {
                return modelProviderRemove(java.util.Arrays.copyOfRange(args, 1, args.length));
            }
            default -> {
            }
        }
        return client(args).map(c -> {
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
    }

    private int modelProviderList(String[] args) {
        var client = client(args).orElseThrow();
        JsonObject body = client.modelProviders();
        JsonArray arr = body.getArray("providers");
        if (arr == null || arr.size() == 0) {
            System.out.println("No model providers registered.");
            return 0;
        }
        for (int i = 0; i < arr.size(); i++) {
            JsonObject provider = arr.getObject(i);
            JsonArray capabilities = provider.getArray("capabilities");
            List<String> caps = new ArrayList<>();
            if (capabilities != null) {
                for (int j = 0; j < capabilities.size(); j++) {
                    caps.add(capabilities.getString(j));
                }
            }
            System.out.printf("%-16s %-24s reachable=%s capabilities=%s%n",
                    provider.getString("id", ""), provider.getString("display_name", ""),
                    provider.getBoolean("reachable", false), String.join(",", caps));
        }
        return 0;
    }

    private int modelProviderModels(String kind, String[] args) {
        if (args.length < 1 || args[0].startsWith("-")) {
            System.err.println("Usage: enr models " + kind + " <provider> [--url URL]");
            return 1;
        }
        var client = client(args).orElseThrow();
        JsonObject body = "installed".equals(kind)
                ? client.modelInstalled(args[0]) : client.modelAvailable(args[0]);
        JsonArray arr = body.getArray("models");
        if (arr == null || arr.size() == 0) {
            System.out.println("No models.");
            return 0;
        }
        for (int i = 0; i < arr.size(); i++) {
            JsonObject model = arr.getObject(i);
            System.out.printf("%-32s %-10s %s%n", model.getString("id", ""),
                    model.getLong("size_bytes", -1) >= 0
                            ? humanBytes(model.getLong("size_bytes", -1)) : "-",
                    model.getString("digest", ""));
        }
        return 0;
    }

    private int modelPull(String[] args) {
        if (args.length < 2 || args[0].startsWith("-") || args[1].startsWith("-")) {
            System.err.println("Usage: enr models pull <provider> <model> [--url URL]");
            return 1;
        }
        var client = client(args).orElseThrow();
        EnrClient.ModelJob job = client.pullModel(args[0], args[1]);
        System.out.println("Pull accepted: provider=" + job.provider() + " model=" + job.model()
                + " job=" + job.jobId() + " status=" + job.status());
        while (true) {
            JsonObject status = client.downloadStatus(job.jobId());
            String state = status.getString("status", "");
            long percent = status.getInt("percent", 0);
            long total = status.getLong("total_bytes", -1);
            long completed = status.getLong("completed_bytes", 0);
            String operation = status.getString("operation", "");
            System.out.printf("\r%-12s %s%s %3d%%", state,
                    operation.isBlank() ? "" : "(" + operation + ") ",
                    total > 0 ? humanBytes(completed) + "/" + humanBytes(total) : "", percent);
            if ("COMPLETED".equals(state)) {
                System.out.println();
                System.out.println("Pull complete: " + args[1]);
                return 0;
            }
            if ("FAILED".equals(state)) {
                System.out.println();
                System.err.println("Pull failed: " + status.getString("error", ""));
                return 1;
            }
            if ("CANCELLED".equals(state) || "CANCEL_REQUESTED".equals(state)) {
                System.out.println();
                System.out.println("Pull cancelled: " + args[1]);
                return 0;
            }
            try {
                Thread.sleep(400);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return 1;
            }
        }
    }

    private int modelJobStatus(String[] args) {
        if (args.length < 1 || args[0].startsWith("-")) {
            System.err.println("Usage: enr models status <jobId> [--url URL]");
            return 1;
        }
        var client = client(args).orElseThrow();
        System.out.println(Json.write(client.downloadStatus(args[0]), true));
        return 0;
    }

    private int modelJobCancel(String[] args) {
        if (args.length < 1 || args[0].startsWith("-")) {
            System.err.println("Usage: enr models cancel <jobId> [--url URL]");
            return 1;
        }
        var client = client(args).orElseThrow();
        JsonObject response = client.cancelDownload(args[0]);
        System.out.println("Cancelled: " + response.getBoolean("cancelled", false));
        return response.getBoolean("cancelled", false) ? 0 : 1;
    }

    private int modelProviderRemove(String[] args) {
        if (args.length < 2 || args[0].startsWith("-") || args[1].startsWith("-")) {
            System.err.println("Usage: enr models remove <provider> <model> [--url URL]");
            return 1;
        }
        var client = client(args).orElseThrow();
        client.removeModel(args[0], args[1]);
        System.out.println("Removed model: " + args[1] + " (provider " + args[0] + ")");
        return 0;
    }

    private int serve(String[] args, boolean writePidFile) throws Exception {
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
                default -> throw new IllegalArgumentException("Unknown serve option: " + args[i]);
            }
        }
        EnrConfig config = EnrConfigLoader.load(configPath);
        config = rebase(config, host, port, backend);

        EnrManager manager = new EnrManager(config);
        autoLoad(manager);
        EnrServer server = new EnrServer(manager, config.host(), config.port());
        server.start();
        System.out.println("ENR server started on " + config.host() + ":" + server.port());
        System.out.println("  Web Control Center: http://" + displayHost(config.host()) + ":" + server.port() + "/");
        System.out.println("  Ollama API:        http://" + displayHost(config.host()) + ":" + server.port() + "/api");
        System.out.println("  OpenAI API:        http://" + displayHost(config.host()) + ":" + server.port() + "/v1");

        Path pidFile = Path.of("/tmp/enr.pid");
        if (writePidFile) {
            try {
                Files.writeString(pidFile, String.valueOf(ProcessHandle.current().pid()));
            } catch (IOException e) {
                System.err.println("Warning: could not write pid file: " + e.getMessage());
            }
        }

        CountDownLatch latch = new CountDownLatch(1);
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("Shutting down ENR (unloading models)...");
            server.close();
            manager.close();
            if (writePidFile) {
                try {
                    Files.deleteIfExists(pidFile);
                } catch (IOException ignored) {
                    // best effort
                }
            }
            latch.countDown();
        }, "enr-shutdown"));
        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        return 0;
    }

    private void autoLoad(EnrManager manager) {
        for (Path dir : manager.config().modelDirectories()) {
            try {
                for (Path modelFile : manager.runtime().models().discoverModelFiles()) {
                    System.out.println("Auto-loading model: " + modelFile);
                    manager.runtime().loadModel(modelFile);
                }
            } catch (Exception e) {
                System.err.println("Warning: auto-load failed for " + dir + ": " + e.getMessage());
            }
        }
    }

    private static String displayHost(String host) {
        return "0.0.0.0".equals(host) || host == null ? "127.0.0.1" : host;
    }

    private static EnrConfig rebase(EnrConfig config, String host, int port, String backend) {
        return EnrConfig.builder()
                .host(host != null ? host : config.host())
                .port(port > 0 ? port : config.port())
                .dataDir(config.dataDir())
                .modelsDir(config.modelsDir())
                .logDir(config.logDir())
                .modelDirectories(config.modelDirectories())
                .preferredBackend(backend != null ? backend : config.preferredBackend())
                .allowCudaFallback(config.allowCudaFallback())
                .maxConcurrentSessions(config.maxConcurrentSessions())
                .defaultMaxTokens(config.defaultMaxTokens())
                .sessionIdleTimeoutSeconds(config.sessionIdleTimeoutSeconds())
                .requestTimeoutSeconds(config.requestTimeoutSeconds())
                .enableMetrics(config.enableMetrics())
                .authToken(config.authToken())
                .ollamaRegistryUrl(config.ollamaRegistryUrl())
                .ollamaSearchUrl(config.ollamaSearchUrl())
                .huggingFaceApiUrl(config.huggingFaceApiUrl())
                .downloadTimeoutMs(config.downloadTimeoutMs())
                .build();
    }

    private int stop(String[] args) throws IOException, InterruptedException {
        // enr stop <model> — unload a model (Ollama-compatible)
        if (args.length >= 1 && !args[0].startsWith("-")) {
            var client = client(args).orElseThrow();
            client.unloadModel(args[0], false);
            System.out.println("Stopped model: " + args[0]);
            return 0;
        }
        if (ServiceManager.installed()) {
            try {
                ServiceManager.stop();
                System.out.println("ENR service stopped.");
                return 0;
            } catch (IOException e) {
                System.err.println("Failed to stop service: " + e.getMessage());
                return 1;
            }
        }
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

    private int start(String[] args) throws Exception {
        if (ServiceManager.installed()) {
            try {
                ServiceManager.start();
                System.out.println("ENR service started.");
                return 0;
            } catch (IOException e) {
                System.err.println("Failed to start service: " + e.getMessage());
                return 1;
            }
        }
        return serve(args, true);
    }

    private int restart(String[] args) {
        if (!ServiceManager.installed()) {
            System.err.println("ENR is not installed as a service. Run 'enr service install' first.");
            return 1;
        }
        try {
            ServiceManager.restart();
            System.out.println("ENR service restarted.");
            return 0;
        } catch (IOException e) {
            System.err.println("Failed to restart service: " + e.getMessage());
            return 1;
        }
    }

    private int status(String[] args) {
        if (ServiceManager.installed()) {
            System.out.println(ServiceManager.status());
            return 0;
        }
        return client(args).map(c -> {
            System.out.println(Json.write(c.status(), true));
            return 0;
        }).orElse(1);
    }

    private int logs(String[] args) {
        if (!ServiceManager.installed()) {
            System.err.println("ENR is not installed as a service. Run 'enr service install' first.");
            return 1;
        }
        int lines = 100;
        for (int i = 0; i < args.length; i++) {
            if ("-n".equals(args[i]) || "--lines".equals(args[i])) {
                lines = Integer.parseInt(args[++i]);
            }
        }
        System.out.println(ServiceManager.logs(lines));
        return 0;
    }

    // ------------------------------------------------------------ model management

    private int list(String[] args) {
        var client = client(args).orElseThrow();
        JsonObject body = client.getJson("/api/v1/catalog/models");
        JsonArray models = body.getArray("models");
        if (models == null || models.size() == 0) {
            System.out.println("No models installed.");
            return 0;
        }
        System.out.printf("%-24s %-10s %-10s %-8s %-9s %s%n", "NAME", "STATE", "BACKEND", "SIZE", "VERIFIED", "SOURCE");
        for (int i = 0; i < models.size(); i++) {
            JsonObject m = models.getObject(i);
            String state = m.getBoolean("loaded", false) ? "loaded" : "stopped";
            String backend = m.getString("backend", "");
            long size = m.getLong("size_bytes", -1);
            System.out.printf("%-24s %-10s %-10s %-8s %-9s %s%n",
                    m.getString("id", ""), state,
                    backend.isBlank() ? "-" : backend,
                    size >= 0 ? humanBytes(size) : "-",
                    m.getBoolean("verified", false) ? "yes" : "no",
                    m.getString("source", "-"));
        }
        return 0;
    }

    private int ps(String[] args) {
        var client = client(args).orElseThrow();
        JsonObject body = client.getJson("/api/ps");
        JsonArray models = body.getArray("models");
        if (models == null || models.size() == 0) {
            System.out.println("No models loaded.");
            return 0;
        }
        System.out.printf("%-24s %-18s %-10s%n", "NAME", "PARAMETERS", "SIZE");
        for (int i = 0; i < models.size(); i++) {
            JsonObject m = models.getObject(i);
            JsonObject details = m.getObject("details");
            System.out.printf("%-24s %-18s %-10s%n",
                    m.getString("name", ""),
                    details == null ? "-" : details.getString("parameter_size", "-"),
                    humanBytes(m.getLong("size", 0)));
        }
        return 0;
    }

    private int show(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: enr show <model>");
            return 1;
        }
        var client = client(args).orElseThrow();
        JsonObject body = new JsonObject();
        body.put("name", args[0]);
        JsonObject info = client.postJson("/api/show", body);
        JsonObject details = info.getObject("details");
        JsonArray capabilities = info.getArray("capabilities");
        StringBuilder caps = new StringBuilder();
        if (capabilities != null) {
            for (int i = 0; i < capabilities.size(); i++) {
                if (caps.length() > 0) {
                    caps.append(',');
                }
                caps.append(capabilities.getString(i));
            }
        }
        System.out.println("Model:     " + args[0]);
        System.out.println("Format:    " + (details == null ? "-" : details.getString("format", "-")));
        System.out.println("Parameters:" + (details == null ? "-" : details.getString("parameter_size", "-")));
        System.out.println("Context:   " + (details == null ? "-" : details.getString("context_length", "-")));
        System.out.println("Caps:      " + (caps.isEmpty() ? "-" : caps.toString()));
        JsonObject modelInfo = info.getObject("model_info");
        if (modelInfo != null) {
            for (var entry : modelInfo.entries()) {
                System.out.println("  " + entry.getKey() + " = " + Json.write(entry.getValue(), false));
            }
        }
        String modelfile = info.getString("modelfile", "");
        if (!modelfile.isBlank()) {
            System.out.println("Modelfile:");
            System.out.println(modelfile);
        }
        return 0;
    }

    private int version(String[] args) {
        var client = client(args).orElseThrow();
        JsonObject v = client.getJson("/api/version");
        System.out.println("ENR " + v.getString("version", "unknown"));
        return 0;
    }

    private int pull(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: enr pull <ref> [--no-stream]   (ref: NAME[:TAG] or hf:REPO/FILE.gguf)");
            return 1;
        }
        String ref = args[0];
        boolean stream = !java.util.Arrays.asList(args).contains("--no-stream");
        var client = client(args).orElseThrow();
        if (stream) {
            try {
                var done = streamPull(client, ref);
                if (done.has("error")) {
                    System.err.println("Pull failed: " + done.getString("error", "unknown error"));
                    return 1;
                }
                System.out.println("Pull complete: " + ref);
                return 0;
            } catch (Exception e) {
                System.err.println("Pull failed: " + e.getMessage());
                return 1;
            }
        }
        JsonObject body = new JsonObject();
        body.put("name", ref);
        JsonObject response = client.postJson("/api/v1/catalog/pull", body);
        String downloadId = response.getString("download_id", "");
        while (true) {
            JsonObject downloads = client.getJson("/api/v1/downloads");
            JsonArray arr = downloads.getArray("downloads");
            JsonObject job = null;
            if (arr != null) {
                for (int i = 0; i < arr.size(); i++) {
                    JsonObject entry = arr.getObject(i);
                    if (downloadId.equals(entry.getString("id", ""))) {
                        job = entry;
                    }
                }
            }
            if (job == null) {
                System.err.println("Download job vanished.");
                return 1;
            }
            String status = job.getString("status", "");
            if ("completed".equals(status)) {
                System.out.println("Pull complete: " + ref);
                return 0;
            }
            if ("failed".equals(status) || "cancelled".equals(status)) {
                System.err.println("Pull " + status + ": " + job.getString("error", ""));
                return 1;
            }
            try {
                Thread.sleep(400);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return 1;
            }
        }
    }

    private JsonObject streamPull(EnrClient client, String ref) throws Exception {
        JsonObject request = new JsonObject();
        request.put("name", ref);
        request.put("stream", true);
        java.net.http.HttpRequest httpRequest = java.net.http.HttpRequest.newBuilder()
                .uri(java.net.URI.create(client.baseUrl() + "/api/pull"))
                .header("Content-Type", "application/json")
                .POST(java.net.http.HttpRequest.BodyPublishers.ofString(Json.write(request, false)))
                .build();
        java.net.http.HttpResponse<java.io.InputStream> response = java.net.http.HttpClient.newBuilder()
                .connectTimeout(java.time.Duration.ofSeconds(30))
                .build()
                .send(httpRequest, java.net.http.HttpResponse.BodyHandlers.ofInputStream());
        if (response.statusCode() != 200) {
            throw new IOException("HTTP " + response.statusCode());
        }
        JsonObject last = new JsonObject();
        try (java.io.BufferedReader reader = new java.io.BufferedReader(
                new java.io.InputStreamReader(response.body()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                JsonObject event = Json.parseObject(line);
                last = event;
                String status = event.getString("status", "");
                long total = event.getLong("total", 0);
                long completed = event.getLong("completed", 0);
                String message = status.isBlank() ? event.getString("error", "") : status;
                if (total > 0 && completed >= 0) {
                    System.out.printf("\r%s %s/%s (%d%%)", message, humanBytes(completed),
                            humanBytes(total), completed * 100 / total);
                } else {
                    System.out.println(message);
                }
                if (event.has("error")) {
                    return event;
                }
            }
        }
        System.out.println();
        return last;
    }

    private int remove(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: enr remove <modelId>");
            return 1;
        }
        var client = client(args).orElseThrow();
        client.deleteJson("/api/v1/catalog/models/" + java.net.URLEncoder.encode(args[0]));
        System.out.println("Removed model: " + args[0]);
        return 0;
    }

    private int copy(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: enr copy <source> <destination>");
            return 1;
        }
        var client = client(args).orElseThrow();
        JsonObject body = new JsonObject();
        body.put("source", args[0]);
        body.put("destination", args[1]);
        client.postJson("/api/v1/catalog/copy", body);
        System.out.println("Copied " + args[0] + " -> " + args[1]);
        return 0;
    }

    private int importModel(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: enr import <path> [--id modelId]");
            return 1;
        }
        String path = args[0];
        String id = null;
        for (int i = 1; i < args.length; i++) {
            if ("--id".equals(args[i])) {
                id = args[++i];
            }
        }
        var client = client(args).orElseThrow();
        JsonObject body = new JsonObject();
        body.put("path", path);
        if (id != null) {
            body.put("id", id);
        }
        JsonObject response = client.postJson("/api/v1/catalog/import", body);
        JsonObject model = response.getObject("model");
        System.out.println("Imported model: " + (model == null ? path : model.getString("id", path)));
        return 0;
    }

    private int chat(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: enr chat <model> [\"prompt\"] [--max-tokens N] [--temperature F] [--multi]");
            return 1;
        }
        String model = args[0];
        Integer maxTokens = null;
        Float temperature = null;
        boolean multi = false;
        String initialPrompt = null;
        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--max-tokens" -> maxTokens = Integer.parseInt(args[++i]);
                case "--temperature" -> temperature = Float.parseFloat(args[++i]);
                case "--multi" -> multi = true;
                default -> initialPrompt = args[i];
            }
        }
        var client = client(args).orElseThrow();
        return interactiveLoop(client, model, maxTokens, temperature, initialPrompt, multi);
    }

    /**
     * {@code enr run <model>}: resolves the reference, auto-pulls if missing,
     * loads and starts interactive inference (Ollama-style).
     */
    private int runModel(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: enr run <model> [\"prompt\"] [--max-tokens N] [--temperature F]");
            return 1;
        }
        String ref = args[0];
        Integer maxTokens = null;
        Float temperature = null;
        String initialPrompt = null;
        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--max-tokens" -> maxTokens = Integer.parseInt(args[++i]);
                case "--temperature" -> temperature = Float.parseFloat(args[++i]);
                default -> initialPrompt = args[i];
            }
        }
        var client = client(args).orElseThrow();
        String id = ensureAvailable(client, ref);
        if (id == null) {
            return 1;
        }
        System.out.println("Loading model...");
        try {
            JsonObject start = new JsonObject();
            start.put("id", id);
            client.postJson("/api/v1/models/start", start);
        } catch (RuntimeException e) {
            System.err.println("Model loading failed: " + e.getMessage());
            return 1;
        }
        System.out.println("Model ready.");
        return interactiveLoop(client, id, maxTokens, temperature, initialPrompt, true);
    }

    /** Resolves a reference, pulling it if it is not already registered. */
    private String ensureAvailable(EnrClient client, String ref) {
        JsonObject body = new JsonObject();
        body.put("name", ref);
        JsonObject response = client.postJson("/api/v1/catalog/ensure", body);
        String id = response.getString("id", "");
        if (response.getBoolean("installed", false)) {
            return id;
        }
        String downloadId = response.getString("download_id", "");
        System.out.println("Resolving model...");
        while (true) {
            JsonObject downloads = client.getJson("/api/v1/downloads");
            JsonArray arr = downloads.getArray("downloads");
            JsonObject job = null;
            if (arr != null) {
                for (int i = 0; i < arr.size(); i++) {
                    JsonObject entry = arr.getObject(i);
                    if (downloadId.equals(entry.getString("id", ""))) {
                        job = entry;
                    }
                }
            }
            if (job == null) {
                System.err.println("Download job vanished.");
                return null;
            }
            String status = job.getString("status", "");
            if ("completed".equals(status)) {
                System.out.println();
                System.out.println("Success.");
                return id;
            }
            if ("failed".equals(status) || "cancelled".equals(status)) {
                System.err.println("Download " + status + ": " + job.getString("error", ""));
                return null;
            }
            long total = job.getLong("total_bytes", -1);
            long completed = job.getLong("completed_bytes", 0);
            if (total > 0) {
                System.out.printf("\rDownloading... %d%%", completed * 100 / total);
            } else {
                System.out.print("\rDownloading...");
            }
            try {
                Thread.sleep(400);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return null;
            }
        }
    }

    private int interactiveLoop(EnrClient client, String model, Integer maxTokens, Float temperature,
                                String initialPrompt, boolean multi) {
        List<JsonObject> messages = new ArrayList<>();
        if (initialPrompt != null) {
            messages.add(chatMessage("user", initialPrompt));
        }
        try (java.io.BufferedReader console = new java.io.BufferedReader(
                new java.io.InputStreamReader(System.in))) {
            while (true) {
                String prompt = initialPrompt;
                initialPrompt = null;
                if (prompt == null) {
                    System.out.print(">>> ");
                    prompt = console.readLine();
                    if (prompt == null || "/exit".equals(prompt.strip())) {
                        System.out.println("Bye.");
                        return 0;
                    }
                    if (prompt.isBlank()) {
                        continue;
                    }
                }
                messages.add(chatMessage("user", prompt));
                final boolean lastTurn = !multi;
                client.chatStream(model, messages, maxTokens, temperature,
                        token -> System.out.print(token),
                        done -> {
                            JsonObject message = done.getObject("message");
                            if (done.has("error")) {
                                System.out.println("\n[error: " + done.getString("error", "") + "]");
                            } else {
                                System.out.println();
                            }
                            messages.add(chatMessage("assistant",
                                    message == null ? "" : message.getString("content", "")));
                        });
                if (lastTurn) {
                    break;
                }
            }
            return 0;
        } catch (IOException e) {
            System.err.println("Chat failed: " + e.getMessage());
            return 1;
        }
    }

    private static JsonObject chatMessage(String role, String content) {
        JsonObject message = new JsonObject();
        message.put("role", role);
        message.put("content", content);
        return message;
    }

    // ------------------------------------------------------------ providers

    private int providers(String[] args) {
        if (args.length == 0) {
            System.err.println("Usage: enr provider list|add|remove|test|models <...>");
            return 1;
        }
        return switch (args[0]) {
            case "list" -> providerList(java.util.Arrays.copyOfRange(args, 1, args.length));
            case "add" -> providerAdd(java.util.Arrays.copyOfRange(args, 1, args.length));
            case "remove" -> providerRemove(java.util.Arrays.copyOfRange(args, 1, args.length));
            case "test" -> providerTest(java.util.Arrays.copyOfRange(args, 1, args.length));
            case "models" -> providerModels(java.util.Arrays.copyOfRange(args, 1, args.length));
            default -> {
                System.err.println("Unknown provider subcommand: " + args[0]);
                yield 1;
            }
        };
    }

    private int providerList(String[] args) {
        var client = client(args).orElseThrow();
        JsonObject body = client.getJson("/api/v1/providers");
        JsonArray arr = body.getArray("providers");
        if (arr == null || arr.size() == 0) {
            System.out.println("No providers configured.");
            return 0;
        }
        System.out.printf("%-16s %-18s %-8s %-10s %s%n", "ID", "TYPE", "ENABLED", "STATUS", "BASE URL");
        for (int i = 0; i < arr.size(); i++) {
            JsonObject p = arr.getObject(i);
            String status = p.getBoolean("last_test_ok", false) ? "ok" : "untested";
            System.out.printf("%-16s %-18s %-8s %-10s %s%n",
                    p.getString("id", ""), p.getString("type", ""),
                    p.getBoolean("enabled", true) ? "yes" : "no",
                    status, p.getString("base_url", ""));
        }
        return 0;
    }

    private int providerAdd(String[] args) {
        String id = null, type = null, name = null, baseUrl = null, apiKey = "";
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--id" -> id = args[++i];
                case "--type" -> type = args[++i];
                case "--name" -> name = args[++i];
                case "--base-url" -> baseUrl = args[++i];
                case "--api-key" -> apiKey = args[++i];
                default -> throw new IllegalArgumentException("Unknown provider add option: " + args[i]);
            }
        }
        if (type == null) {
            System.err.println("Usage: enr provider add --type deepseek|openai|anthropic|gemini|"
                    + "openai-compatible|ollama|llama.cpp --api-key KEY [--id ID] [--name N] [--base-url URL]");
            return 1;
        }
        var client = client(new String[0]).orElseThrow();
        JsonObject body = new JsonObject();
        body.put("id", id == null ? type : id);
        body.put("type", type);
        body.put("name", name == null ? type : name);
        if (baseUrl != null) {
            body.put("base_url", baseUrl);
        }
        body.put("api_key", apiKey);
        JsonObject provider = client.postJson("/api/v1/providers", body);
        System.out.println("Provider added: " + provider.getString("id", "")
                + " (" + provider.getString("type", "") + ")");
        return 0;
    }

    private int providerRemove(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: enr provider remove <id>");
            return 1;
        }
        var client = client(new String[0]).orElseThrow();
        client.deleteJson("/api/v1/providers/" + java.net.URLEncoder.encode(args[0]));
        System.out.println("Provider removed: " + args[0]);
        return 0;
    }

    private int providerTest(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: enr provider test <id>");
            return 1;
        }
        var client = client(new String[0]).orElseThrow();
        JsonObject result = client.postJson("/api/v1/providers/"
                + java.net.URLEncoder.encode(args[0]) + "/test", new JsonObject());
        System.out.println(result.getBoolean("ok", false)
                ? "OK: " + result.getString("message", "") + " (" + result.getLong("elapsed_ms", 0) + " ms)"
                : "FAILED: " + result.getString("message", ""));
        JsonArray models = result.getArray("models");
        if (models != null && models.size() > 0) {
            System.out.println("Models (" + models.size() + "):");
            for (int i = 0; i < models.size(); i++) {
                System.out.println("  " + models.getString(i));
            }
        }
        return result.getBoolean("ok", false) ? 0 : 1;
    }

    private int providerModels(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: enr provider models <id>");
            return 1;
        }
        var client = client(new String[0]).orElseThrow();
        JsonObject result = client.getJson("/api/v1/providers/"
                + java.net.URLEncoder.encode(args[0]) + "/models");
        JsonArray models = result.getArray("models");
        if (models == null || models.size() == 0) {
            System.out.println("No models reported.");
            return 0;
        }
        for (int i = 0; i < models.size(); i++) {
            System.out.println(models.getString(i));
        }
        return 0;
    }

    // ------------------------------------------------------------ system service

    private int service(String[] args) throws IOException {
        if (args.length == 0) {
            System.err.println("Usage: enr service install|uninstall|status|logs|unit [--jar PATH]");
            return 1;
        }
        Path jarPath = currentJarPath();
        for (int i = 1; i < args.length; i++) {
            if ("--jar".equals(args[i])) {
                jarPath = Path.of(args[++i]);
            }
        }
        return switch (args[0]) {
            case "unit" -> {
                String java = SystemdSupport.currentJavaBinary().toString();
                String logDir = Path.of(System.getProperty("user.home"), ".enr", "logs").toString();
                System.out.println(ServiceManager.detectScope() == ServiceManager.Scope.SYSTEM
                        ? SystemdSupport.unitFile(java, jarPath.toString(),
                                Path.of(System.getProperty("user.home"), ".enr", "enr.yaml").toString(),
                                System.getProperty("user.name"), System.getProperty("user.name"), logDir)
                        : SystemdSupport.unitFileUser(java, jarPath.toString(), logDir));
                yield 0;
            }
            case "install" -> {
                if (!Files.exists(jarPath)) {
                    System.err.println("JAR not found: " + jarPath + " (run mvn package first, or pass --jar)");
                    yield 1;
                }
                ServiceManager.install(jarPath, "127.0.0.1", 8111);
                yield 0;
            }
            case "uninstall" -> {
                ServiceManager.uninstall();
                yield 0;
            }
            case "status" -> {
                System.out.println(ServiceManager.installed()
                        ? ServiceManager.status() : "ENR service is not installed.");
                yield 0;
            }
            case "logs" -> {
                if (!ServiceManager.installed()) {
                    System.err.println("ENR service is not installed.");
                    yield 1;
                }
                System.out.println(ServiceManager.logs(100));
                yield 0;
            }
            default -> {
                System.err.println("Unknown service subcommand: " + args[0]);
                yield 1;
            }
        };
    }

    // ------------------------------------------------------------ legacy commands

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

    private static String humanBytes(long bytes) {
        if (bytes < 1024) {
            return bytes + " B";
        }
        double value = bytes;
        String[] units = {"KB", "MB", "GB", "TB"};
        int unit = -1;
        while (value >= 1024 && unit < units.length - 1) {
            value /= 1024;
            unit++;
        }
        return String.format(java.util.Locale.ROOT, "%.1f %s", value, units[unit]);
    }

    private static void printUsage() {
        System.out.println("""
                ENR — Native AI Model Inference Runtime & Control Center

                Global options:
                  --logging                                          enable application logging (INFO)
                  --no-logging                                       disable routine logging (errors only)

                Server:
                  enr serve [-c config.yaml] [-p port] [-H host] [--cuda|--cpu]    foreground server
                  enr start [same options]                                          start (service, else pid file)
                  enr stop                                                          stop service / server
                  enr restart                                                       restart the service
                  enr status | devices | diagnostics
                  enr logs [-n N]                                                  service logs (journalctl)

                Models:
                  enr list                                             installed models
                  enr ps                                               loaded models
                  enr show <model>                                     model information
                  enr pull <NAME[:TAG] | hf.co/OWNER/REPO:QUANT | hf:OWNER/REPO/FILE.gguf>
                  enr remove <modelId>       (alias: enr rm <modelId>)
                  enr copy <source> <destination>
                  enr import <path.gguf> [--id modelId]
                  enr load <path> [--backend cpu|cuda]
                  enr unload <modelId> [--force]
                  enr stop <modelId>                                  unload a model
                  enr infer <modelId> "<prompt>" [--max-tokens N] [--temperature F] [--stream]

                Chat:
                  enr chat <model> ["prompt"] [--max-tokens N] [--temperature F] [--multi]
                  enr run <model> ["prompt"] [--max-tokens N] [--temperature F]   (auto-pulls if missing)

                Other:
                  enr version                                         runtime version

                Providers:
                  enr provider list
                  enr provider add --type deepseek|openai|anthropic|gemini|openai-compatible|ollama|llama.cpp --api-key KEY [--id ID] [--base-url URL]
                  enr provider remove <id>
                  enr provider test <id>
                  enr provider models <id>

                System service:
                  enr service install [--jar PATH]                                  install + start + verify
                  enr service uninstall
                  enr service status
                  enr service logs
                  enr service unit [--jar PATH]                                     print the unit file
                """);
    }

    private static void configureLogging(Level level) {
        Logger root = Logger.getLogger("");
        for (java.util.logging.Handler handler : root.getHandlers()) {
            root.removeHandler(handler);
        }
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(level);
        handler.setFormatter(new Formatter() {
            @Override
            public String format(LogRecord record) {
                return java.time.Instant.ofEpochMilli(record.getMillis())
                        + " [" + record.getLevel().getName() + "] " + record.getMessage() + "\n";
            }
        });
        root.addHandler(handler);
        root.setLevel(level);
    }
}
