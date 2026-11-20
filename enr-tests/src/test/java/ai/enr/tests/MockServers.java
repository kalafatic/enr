package ai.enr.tests;

import ai.enr.core.io.Hashing;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
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
