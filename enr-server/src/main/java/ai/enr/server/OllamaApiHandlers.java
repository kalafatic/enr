package ai.enr.server;

import ai.enr.core.json.Json;
import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonObject;
import ai.enr.core.json.JsonValue;
import ai.enr.manager.EnrManager;
import ai.enr.manager.provider.ChatMessage;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import java.io.IOException;
import java.io.OutputStream;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Ollama-compatible API surface. Endpoints: /api/tags, /api/show, /api/pull,
 * /api/delete, /api/chat, /api/generate, /api/ps, /api/version. Streaming
 * responses use Ollama's newline-delimited JSON format.
 */
final class OllamaApiHandlers {

    private OllamaApiHandlers() {
    }

    static HttpHandler tags(EnrManager manager) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only GET allowed");
                return;
            }
            JsonObject body = new JsonObject();
            body.put("models", manager.ollamaTags());
            HttpSupport.sendJson(exchange, 200, body);
        };
    }

    static HttpHandler show(EnrManager manager) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only POST allowed");
                return;
            }
            try {
                JsonObject request = HttpSupport.parseBodyObject(exchange);
                String name = request.getString("name", "");
                HttpSupport.sendJson(exchange, 200, manager.ollamaShow(name));
            } catch (IllegalArgumentException e) {
                HttpSupport.sendError(exchange, 404, "MODEL_NOT_FOUND", e.getMessage());
            }
        };
    }

    static HttpHandler pull(EnrManager manager) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only POST allowed");
                return;
            }
            JsonObject request = HttpSupport.parseBodyObject(exchange);
            String name = request.getString("name", "");
            boolean stream = request.getBoolean("stream", true);
            if (name.isBlank()) {
                HttpSupport.sendError(exchange, 400, "INVALID_REQUEST", "Missing 'name' field");
                return;
            }
            if (!stream) {
                pullBlocking(exchange, manager, name);
                return;
            }
            exchange.getResponseHeaders().set("Content-Type", "application/x-ndjson; charset=utf-8");
            exchange.sendResponseHeaders(200, 0);
            try (OutputStream out = exchange.getResponseBody()) {
                status(out, "pulling manifest");
                String jobId;
                try {
                    jobId = manager.pullModel(name);
                } catch (Exception e) {
                    HttpSupport.sendNdJson(out, errorLine(e.getMessage() == null
                            ? e.getClass().getSimpleName() : e.getMessage()));
                    return;
                }
                if (jobId == null) {
                    HttpSupport.sendNdJson(out, errorLine("Download could not be started"));
                    return;
                }
                long lastUpdate = 0;
                while (true) {
                    var job = manager.downloads().get(jobId);
                    if (job.isEmpty()) {
                        break;
                    }
                    var current = job.get();
                    long now = System.currentTimeMillis();
                    if (now - lastUpdate > 200) {
                        lastUpdate = now;
                        JsonObject progress = new JsonObject();
                        progress.put("status", "downloading " + current.digest());
                        progress.put("digest", "sha256:" + current.digest());
                        progress.put("total", current.totalBytes());
                        progress.put("completed", current.completedBytes());
                        HttpSupport.sendNdJson(out, progress);
                    }
                    switch (current.status()) {
                        case COMPLETED -> {
                            JsonObject verify = new JsonObject();
                            verify.put("status", current.digest().isBlank()
                                    ? "writing manifest" : "verifying sha256 digest");
                            HttpSupport.sendNdJson(out, verify);
                            JsonObject done = new JsonObject();
                            done.put("status", "success");
                            HttpSupport.sendNdJson(out, done);
                            return;
                        }
                        case FAILED -> {
                            HttpSupport.sendNdJson(out, errorLine(current.error()));
                            return;
                        }
                        case CANCELLED -> {
                            HttpSupport.sendNdJson(out, errorLine("download cancelled"));
                            return;
                        }
                        default -> {
                            Thread.sleep(200);
                        }
                    }
                }
                HttpSupport.sendNdJson(out, errorLine("download job disappeared"));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        };
    }

    private static void pullBlocking(HttpExchange exchange, EnrManager manager, String name)
            throws IOException {
        String jobId;
        try {
            jobId = manager.pullModel(name);
        } catch (Exception e) {
            HttpSupport.sendError(exchange, 500, "PULL_FAILED", e.getMessage() == null
                    ? e.getClass().getSimpleName() : e.getMessage());
            return;
        }
        while (true) {
            var job = manager.downloads().get(jobId);
            if (job.isEmpty()) {
                HttpSupport.sendError(exchange, 500, "PULL_FAILED", "download job disappeared");
                return;
            }
            var current = job.get();
            switch (current.status()) {
                case COMPLETED -> {
                    JsonObject done = new JsonObject();
                    done.put("status", "success");
                    HttpSupport.sendJson(exchange, 200, done);
                    return;
                }
                case FAILED -> {
                    HttpSupport.sendError(exchange, 500, "PULL_FAILED", current.error());
                    return;
                }
                case CANCELLED -> {
                    HttpSupport.sendError(exchange, 409, "PULL_CANCELLED", "download cancelled");
                    return;
                }
                default -> {
                    try {
                        Thread.sleep(200);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return;
                    }
                }
            }
        }
    }

    static HttpHandler delete(EnrManager manager) {
        return exchange -> {
            if (!"DELETE".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only DELETE allowed");
                return;
            }
            try {
                JsonObject request = HttpSupport.parseBodyObject(exchange);
                String name = request.getString("name", "");
                if (name.isBlank()) {
                    HttpSupport.sendError(exchange, 400, "INVALID_REQUEST", "Missing 'name' field");
                    return;
                }
                manager.removeModel(name);
                exchange.sendResponseHeaders(200, -1);
                exchange.close();
            } catch (IllegalArgumentException e) {
                JsonObject error = new JsonObject();
                error.put("error", e.getMessage());
                HttpSupport.sendJson(exchange, 404, error);
            } catch (Exception e) {
                HttpSupport.sendError(exchange, 500, "DELETE_FAILED", e.getMessage());
            }
        };
    }

    static HttpHandler chat(EnrManager manager) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only POST allowed");
                return;
            }
            try {
                JsonObject request = HttpSupport.parseBodyObject(exchange);
                String model = request.getString("model", "");
                List<ChatMessage> messages = messagesFrom(request);
                boolean stream = request.getBoolean("stream", false);
                EnrManager.ChatOptions options = optionsFrom(request, stream);
                if (model.isBlank()) {
                    HttpSupport.sendError(exchange, 400, "INVALID_REQUEST", "Missing 'model' field");
                    return;
                }
                if (!stream) {
                    JsonObject result = manager.chat(model, messages, options);
                    if (result.has("error")) {
                        HttpSupport.sendError(exchange, 500, "INFERENCE_FAILED",
                                result.getString("error", "inference failed"));
                        return;
                    }
                    JsonObject response = new JsonObject();
                    response.put("model", model);
                    response.put("created_at", java.time.Instant.now().toString());
                    JsonObject message = new JsonObject();
                    message.put("role", "assistant");
                    message.put("content", result.getString("content", ""));
                    response.put("message", message);
                    response.put("done", true);
                    response.put("done_reason", result.getString("finish_reason", "stop"));
                    response.put("total_duration", result.getLong("elapsed_ms", 0) * 1_000_000);
                    response.put("prompt_eval_count", result.getInt("prompt_tokens", 0));
                    response.put("eval_count", result.getInt("completion_tokens", 0));
                    HttpSupport.sendJson(exchange, 200, response);
                    return;
                }
                exchange.getResponseHeaders().set("Content-Type", "application/x-ndjson; charset=utf-8");
                exchange.sendResponseHeaders(200, 0);
                try (OutputStream out = exchange.getResponseBody()) {
                    manager.chatStream(model, messages, options, chunk -> {
                        try {
                            JsonObject line = new JsonObject();
                            line.put("model", model);
                            line.put("created_at", java.time.Instant.now().toString());
                            JsonObject message = new JsonObject();
                            message.put("role", "assistant");
                            message.put("content", chunk.getString("content", ""));
                            line.put("message", message);
                            line.put("done", chunk.getBoolean("done", false));
                            if (chunk.getBoolean("done", false)) {
                                line.put("done_reason", chunk.getString("finish_reason", "stop"));
                                line.put("total_duration", chunk.getLong("elapsed_ms", 0) * 1_000_000);
                                line.put("prompt_eval_count", chunk.getInt("prompt_tokens", 0));
                                line.put("eval_count", chunk.getInt("completion_tokens", 0));
                            }
                            HttpSupport.sendNdJson(out, line);
                        } catch (IOException e) {
                            throw new RuntimeException("client disconnected", e);
                        }
                    });
                } catch (RuntimeException ignored) {
                    // client disconnected
                }
            } catch (IllegalArgumentException e) {
                HttpSupport.sendError(exchange, 400, "INVALID_REQUEST", e.getMessage());
            } catch (Exception e) {
                HttpSupport.sendError(exchange, 500, "INFERENCE_FAILED", e.getMessage());
            }
        };
    }

    static HttpHandler generate(EnrManager manager) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only POST allowed");
                return;
            }
            try {
                JsonObject request = HttpSupport.parseBodyObject(exchange);
                String model = request.getString("model", "");
                String prompt = request.getString("prompt", "");
                boolean stream = request.getBoolean("stream", false);
                if (model.isBlank()) {
                    HttpSupport.sendError(exchange, 400, "INVALID_REQUEST", "Missing 'model' field");
                    return;
                }
                List<ChatMessage> messages = List.of(ChatMessage.of("user", prompt));
                EnrManager.ChatOptions options = optionsFrom(request, stream);
                if (!stream) {
                    JsonObject result = manager.chat(model, messages, options);
                    JsonObject response = new JsonObject();
                    response.put("model", model);
                    response.put("created_at", java.time.Instant.now().toString());
                    response.put("response", result.getString("content", ""));
                    response.put("done", true);
                    response.put("done_reason", result.getString("finish_reason", "stop"));
                    response.put("total_duration", result.getLong("elapsed_ms", 0) * 1_000_000);
                    response.put("prompt_eval_count", result.getInt("prompt_tokens", 0));
                    response.put("eval_count", result.getInt("completion_tokens", 0));
                    HttpSupport.sendJson(exchange, 200, response);
                    return;
                }
                exchange.getResponseHeaders().set("Content-Type", "application/x-ndjson; charset=utf-8");
                exchange.sendResponseHeaders(200, 0);
                try (OutputStream out = exchange.getResponseBody()) {
                    manager.chatStream(model, messages, options, chunk -> {
                        try {
                            JsonObject line = new JsonObject();
                            line.put("model", model);
                            line.put("created_at", java.time.Instant.now().toString());
                            line.put("response", chunk.getString("content", ""));
                            line.put("done", chunk.getBoolean("done", false));
                            if (chunk.getBoolean("done", false)) {
                                line.put("done_reason", chunk.getString("finish_reason", "stop"));
                                line.put("total_duration", chunk.getLong("elapsed_ms", 0) * 1_000_000);
                                line.put("prompt_eval_count", chunk.getInt("prompt_tokens", 0));
                                line.put("eval_count", chunk.getInt("completion_tokens", 0));
                            }
                            HttpSupport.sendNdJson(out, line);
                        } catch (IOException e) {
                            throw new RuntimeException("client disconnected", e);
                        }
                    });
                } catch (RuntimeException ignored) {
                    // client disconnected
                }
            } catch (IllegalArgumentException e) {
                HttpSupport.sendError(exchange, 400, "INVALID_REQUEST", e.getMessage());
            } catch (Exception e) {
                HttpSupport.sendError(exchange, 500, "INFERENCE_FAILED", e.getMessage());
            }
        };
    }

    static HttpHandler ps(EnrManager manager) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only GET allowed");
                return;
            }
            HttpSupport.sendJson(exchange, 200, manager.ollamaPs());
        };
    }

    static HttpHandler version(EnrManager manager) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only GET allowed");
                return;
            }
            HttpSupport.sendJson(exchange, 200, manager.version());
        };
    }

    private static List<ChatMessage> messagesFrom(JsonObject request) {
        List<ChatMessage> messages = new ArrayList<>();
        JsonArray arr = request.getArray("messages");
        if (arr == null) {
            return messages;
        }
        for (int i = 0; i < arr.size(); i++) {
            JsonObject entry = arr.getObject(i);
            if (entry == null) {
                continue;
            }
            String content = entry.getString("content", "");
            JsonArray images = entry.getArray("images");
            if (images != null && images.size() > 0 && content.isEmpty()) {
                content = "[image input]";
            }
            messages.add(ChatMessage.of(entry.getString("role", "user"), content));
        }
        return messages;
    }

    private static EnrManager.ChatOptions optionsFrom(JsonObject request, boolean stream) {
        JsonObject options = request.getObject("options");
        EnrManager.ChatOptions defaults = EnrManager.ChatOptions.defaults();
        Float temperature = null;
        Float topP = null;
        Integer topK = null;
        Integer maxTokens = null;
        Long seed = null;
        List<String> stop = List.of();
        if (options != null) {
            temperature = options.has("temperature") ? options.getFloat("temperature", 0.7f) : null;
            topP = options.has("top_p") ? options.getFloat("top_p", 0.9f) : null;
            topK = options.has("top_k") ? options.getInt("top_k", 40) : null;
            maxTokens = options.has("num_predict") ? options.getInt("num_predict", -1) : null;
            seed = options.has("seed") ? options.getLong("seed", 0) : null;
            if (options.has("stop")) {
                JsonArray stopArr = options.getArray("stop");
                List<String> stops = new ArrayList<>();
                if (stopArr != null) {
                    for (int i = 0; i < stopArr.size(); i++) {
                        stops.add(stopArr.getString(i));
                    }
                }
                stop = stops;
            }
        }
        if (maxTokens == null && request.has("num_predict")) {
            maxTokens = request.getInt("num_predict", -1);
        }
        int effectiveMaxTokens = maxTokens != null ? maxTokens : defaults.maxTokens();
        return new EnrManager.ChatOptions(effectiveMaxTokens, temperature, topP, topK, seed, stop, stream);
    }

    private static JsonObject errorLine(String message) {
        JsonObject line = new JsonObject();
        line.put("error", message == null ? "unknown error" : message);
        return line;
    }

    private static void status(OutputStream out, String status) throws IOException {
        JsonObject line = new JsonObject();
        line.put("status", status);
        HttpSupport.sendNdJson(out, line);
    }
}
