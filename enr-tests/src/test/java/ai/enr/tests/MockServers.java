package ai.enr.tests;

import ai.enr.core.io.Hashing;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;

/**
 * Hermetic mock servers for integration tests: a minimal Ollama registry
 * (manifest + blobs) and an OpenAI-compatible cloud provider.
 */
final class MockServers {

    private MockServers() {
    }

    /** Serves a GGUF file as an Ollama registry model under the given name. */
    static HttpServer ollamaRegistry(Path modelFile, String modelName, String tag) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 8);
        byte[] bytes = Files.readAllBytes(modelFile);
        String digest = Hashing.sha256Hex(bytes);
        server.createContext("/", exchange -> {
            String path = exchange.getRequestURI().getPath();
            if ("/library".equals(path)) {
                send(exchange, 200,
                        "<html><body><a href=\"/library/" + modelName + "\">" + modelName + "</a></body></html>",
                        "text/html");
                return;
            }
            if (path.equals("/v2/library/" + modelName + "/manifests/" + tag)) {
                String manifest = """
                        {"schemaVersion":2,"mediaType":"application/vnd.docker.distribution.manifest.v2+json",
                        "config":{"mediaType":"application/vnd.docker.container.image.v1+json","digest":"sha256:%s","size":2},
                        "layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:%s","size":%d}]}
                        """.formatted(digest, digest, bytes.length);
                send(exchange, 200, manifest, "application/json");
                return;
            }
            if (path.equals("/v2/library/" + modelName + "/blobs/sha256:" + digest)) {
                exchange.getResponseHeaders().set("Content-Type", "application/octet-stream");
                exchange.sendResponseHeaders(200, bytes.length);
                try (OutputStream out = exchange.getResponseBody()) {
                    out.write(bytes);
                }
                return;
            }
            send(exchange, 404, "{\"error\":\"not found\"}", "application/json");
        });
        server.setExecutor(Executors.newVirtualThreadPerTaskExecutor());
        server.start();
        return server;
    }

    /**
     * An OpenAI-compatible provider. Verifies the Bearer key, exposes the
     * given models and echoes a fixed completion. Supports a slow endpoint
     * for download tests: GET /slow streams bytes with delays.
     */
    static HttpServer openAiProvider(String apiKey, List<String> models) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 8);
        server.createContext("/", exchange -> {
            String path = exchange.getRequestURI().getPath();
            String auth = exchange.getRequestHeaders().getFirst("Authorization");
            String expected = "Bearer " + apiKey;
            if (!expected.equals(auth)) {
                send(exchange, 401, "{\"error\":{\"message\":\"invalid api key\"}}", "application/json");
                return;
            }
            if ("/v1/models".equals(path)) {
                StringBuilder data = new StringBuilder();
                for (String model : models) {
                    if (data.length() > 0) {
                        data.append(',');
                    }
                    data.append("{\"id\":\"").append(model)
                            .append("\",\"object\":\"model\",\"created\":1,\"owned_by\":\"mock\"}");
                }
                send(exchange, 200, "{\"object\":\"list\",\"data\":[" + data + "]}", "application/json");
                return;
            }
            if ("/v1/chat/completions".equals(path)) {
                String body = new String(exchange.getRequestBody().readAllBytes());
                if (body.contains("\"stream\":true")) {
                    exchange.getResponseHeaders().set("Content-Type", "text/event-stream");
                    exchange.sendResponseHeaders(200, 0);
                    try (OutputStream out = exchange.getResponseBody()) {
                        out.write(("data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\","
                                + "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"mock \"},\"finish_reason\":null}]}\n\n"
                                + "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\","
                                + "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"reply\"},\"finish_reason\":\"stop\"}]}\n\n"
                                + "data: [DONE]\n\n").getBytes());
                    }
                    return;
                }
                send(exchange, 200,
                        "{\"id\":\"c\",\"object\":\"chat.completion\",\"created\":1,"
                                + "\"model\":\"mock-model\","
                                + "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"mock reply\"},"
                                + "\"finish_reason\":\"stop\"}],"
                                + "\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}",
                        "application/json");
                return;
            }
            if ("/v1/embeddings".equals(path)) {
                send(exchange, 200,
                        "{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"embedding\":[0.1,0.2,0.3],\"index\":0}],"
                                + "\"model\":\"mock-model\",\"usage\":{\"prompt_tokens\":1,\"total_tokens\":1}}",
                        "application/json");
                return;
            }
            send(exchange, 404, "{\"error\":{\"message\":\"not found\"}}", "application/json");
        });
        server.setExecutor(Executors.newVirtualThreadPerTaskExecutor());
        server.start();
        return server;
    }

    /** Simple HTTP endpoint streaming bytes with a delay (download tests). */
    static HttpServer slowFileServer(byte[] payload, long chunkDelayMs) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 8);
        server.createContext("/slow", exchange -> {
            exchange.getResponseHeaders().set("Content-Type", "application/octet-stream");
            exchange.sendResponseHeaders(200, payload.length);
            try (OutputStream out = exchange.getResponseBody()) {
                int chunk = Math.max(1, payload.length / 20);
                for (int off = 0; off < payload.length; off += chunk) {
                    out.write(payload, off, Math.min(chunk, payload.length - off));
                    out.flush();
                    Thread.sleep(chunkDelayMs);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        server.setExecutor(Executors.newVirtualThreadPerTaskExecutor());
        server.start();
        return server;
    }

    /**
     * Configurable mock Ollama API server (/api/tags, /api/show, /api/pull
     * NDJSON stream, /api/delete, /api/version) for model management tests.
     */
    static final class OllamaMock implements AutoCloseable {

        final HttpServer server;
        final List<String> installed = new java.util.ArrayList<>();
        volatile List<String> pullScript;
        volatile long lineDelayMs = 0;
        volatile boolean failPullRequest = false;
        volatile long pulledRequests = 0;

        OllamaMock() throws IOException {
            server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 8);
            server.createContext("/", exchange -> {
                String path = exchange.getRequestURI().getPath();
                switch (path) {
                    case "/api/version" -> send(exchange, 200, "{\"version\":\"0.6.0-mock\"}", "application/json");
                    case "/api/tags" -> {
                        StringBuilder models = new StringBuilder();
                        synchronized (installed) {
                            for (int i = 0; i < installed.size(); i++) {
                                if (models.length() > 0) {
                                    models.append(',');
                                }
                                models.append("{\"name\":\"").append(installed.get(i))
                                        .append("\",\"model\":\"").append(installed.get(i))
                                        .append("\",\"modified_at\":\"2026-11-21T10:00:00Z\",")
                                        .append("\"size\":").append(1_000_000 + i * 10_000)
                                        .append(",\"digest\":\"sha256:mock").append(i).append("\",")
                                        .append("\"details\":{\"family\":\"llama\",\"parameter_size\":\"7B\",")
                                        .append("\"quantization_level\":\"Q4_K_M\",\"format\":\"gguf\",")
                                        .append("\"context_length\":8192}}");
                            }
                        }
                        send(exchange, 200, "{\"models\":[" + models + "]}", "application/json");
                    }
                    case "/api/show" -> {
                        String body = new String(exchange.getRequestBody().readAllBytes());
                        String name = body.replaceAll(".*\"name\"\\s*:\\s*\"([^\"]*)\".*", "$1");
                        send(exchange, 200, "{\"model\":\"" + name + "\",\"digest\":\"sha256:show\","
                                + "\"size\":123456,\"modified_at\":1769486400,"
                                + "\"model_info\":{\"general.architecture\":\"llama\","
                                + "\"general.parameter_count\":\"7000000000\","
                                + "\"general.context_length\":8192,\"general.file_type\":\"Q4_K_M\"}}",
                                "application/json");
                    }
                    case "/api/pull" -> {
                        pulledRequests++;
                        if (failPullRequest) {
                            send(exchange, 404, "{\"error\":\"pull model manifest: file does not exist\"}",
                                    "application/json");
                            break;
                        }
                        exchange.getResponseHeaders().set("Content-Type", "application/x-ndjson");
                        exchange.sendResponseHeaders(200, 0);
                        try (OutputStream out = exchange.getResponseBody()) {
                            for (String line : pullScript == null ? List.<String>of() : pullScript) {
                                out.write((line + "\n").getBytes());
                                out.flush();
                                if (lineDelayMs > 0) {
                                    Thread.sleep(lineDelayMs);
                                }
                            }
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                    }
                    case "/api/delete" -> {
                        String body = new String(exchange.getRequestBody().readAllBytes());
                        String name = body.replaceAll(".*\"name\"\\s*:\\s*\"([^\"]*)\".*", "$1");
                        synchronized (installed) {
                            if (installed.remove(name)) {
                                send(exchange, 200, "{}", "application/json");
                            } else {
                                send(exchange, 404, "{\"error\":\"model '" + name + "' not found\"}",
                                        "application/json");
                            }
                        }
                    }
                    default -> send(exchange, 404, "{\"error\":\"not found\"}", "application/json");
                }
            });
            server.setExecutor(Executors.newVirtualThreadPerTaskExecutor());
            server.start();
        }

        String baseUrl() {
            return "http://127.0.0.1:" + server.getAddress().getPort();
        }

        /** Standard two-layer pull script ending in success. */
        static List<String> standardPullScript(String model, String layerA, String layerB) {
            List<String> lines = new ArrayList<>();
            lines.add("{\"status\":\"pulling manifest\"}");
            lines.add("{\"status\":\"pulling " + layerA.substring(0, 8) + "\",\"digest\":\""
                    + layerA + "\",\"total\":400,\"completed\":0}");
            lines.add("{\"status\":\"pulling " + layerA.substring(0, 8) + "\",\"digest\":\""
                    + layerA + "\",\"total\":400,\"completed\":200}");
            lines.add("{\"status\":\"pulling " + layerB.substring(0, 8) + "\",\"digest\":\""
                    + layerB + "\",\"total\":600,\"completed\":100}");
            lines.add("{\"status\":\"pulling " + layerB.substring(0, 8) + "\",\"digest\":\""
                    + layerB + "\",\"total\":600,\"completed\":600}");
            lines.add("{\"status\":\"pulling " + layerA.substring(0, 8) + "\",\"digest\":\""
                    + layerA + "\",\"total\":400,\"completed\":400}");
            lines.add("{\"status\":\"verifying sha256 digest\"}");
            lines.add("{\"status\":\"writing manifest\"}");
            lines.add("{\"status\":\"success\"}");
            return lines;
        }

        @Override
        public void close() {
            server.stop(0);
        }
    }

    private static void send(HttpExchange exchange, int status, String body, String contentType)
            throws IOException {
        byte[] bytes = body.getBytes();
        exchange.getResponseHeaders().set("Content-Type", contentType);
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }
}
