package ai.enr.server;

import ai.enr.core.json.JsonObject;
import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonValue;
import ai.enr.inference.GenerationEngine;
import ai.enr.runtime.EnrRuntime;
import ai.enr.runtime.ModelInstance;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import java.io.IOException;
import java.io.OutputStream;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Versioned REST API handlers (see ENR-API-SPECIFICATION.md).
 */
final class ApiHandlers {

    private ApiHandlers() {
    }

    static HttpHandler health(EnrRuntime runtime) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only GET allowed");
                return;
            }
            Set<String> failures = runtime.healthChecks();
            JsonObject body = new JsonObject();
            body.put("status", failures.isEmpty() ? "healthy" : "unhealthy");
            body.put("checks", failures.isEmpty() ? new JsonArray() : arrayOf(failures));
            HttpSupport.sendJson(exchange, failures.isEmpty() ? 200 : 503, body);
        };
    }

    static HttpHandler status(EnrRuntime runtime) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only GET allowed");
                return;
            }
            HttpSupport.sendJson(exchange, 200, runtime.status());
        };
    }

    static HttpHandler listModels(EnrRuntime runtime) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only GET allowed");
                return;
            }
            JsonArray models = new JsonArray();
            for (ModelInstance model : runtime.models().list()) {
                models.add(modelJson(model));
            }
            JsonObject body = new JsonObject();
            body.put("models", models);
            HttpSupport.sendJson(exchange, 200, body);
        };
    }

    static HttpHandler loadModel(EnrRuntime runtime) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only POST allowed");
                return;
            }
            try {
                JsonValue body = HttpSupport.parseBody(exchange);
                String path = field(body, "path");
                String backend = field(body, "backend");
                if (path == null || path.isBlank()) {
                    HttpSupport.sendError(exchange, 400, "INVALID_REQUEST", "Missing 'path' field");
                    return;
                }
                ModelInstance model = runtime.loadModel(java.nio.file.Path.of(path),
                        backend == null || backend.isBlank() ? null : backend);
                JsonObject response = new JsonObject();
                response.put("model", modelJson(model));
                HttpSupport.sendJson(exchange, 200, response);
            } catch (IllegalArgumentException e) {
                HttpSupport.sendError(exchange, 400, "INVALID_REQUEST", e.getMessage());
            } catch (Exception e) {
                HttpSupport.sendError(exchange, 422, "LOAD_FAILED", String.valueOf(e.getMessage()));
            }
        };
    }

    static HttpHandler unloadModel(EnrRuntime runtime) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only POST allowed");
                return;
            }
            try {
                JsonValue body = HttpSupport.parseBody(exchange);
                String id = field(body, "id");
                boolean force = "true".equalsIgnoreCase(field(body, "force"));
                if (id == null || id.isBlank()) {
                    HttpSupport.sendError(exchange, 400, "INVALID_REQUEST", "Missing 'id' field");
                    return;
                }
                runtime.unloadModel(id, force);
                JsonObject response = new JsonObject();
                response.put("unloaded", id);
                HttpSupport.sendJson(exchange, 200, response);
            } catch (IllegalArgumentException e) {
                HttpSupport.sendError(exchange, 404, "MODEL_NOT_FOUND", e.getMessage());
            } catch (IllegalStateException e) {
                HttpSupport.sendError(exchange, 409, "MODEL_IN_USE", e.getMessage());
            }
        };
    }

    static HttpHandler getModel(EnrRuntime runtime) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only GET allowed");
                return;
            }
            String id = exchange.getRequestURI().getPath().substring("/api/v1/models/".length());
            id = URLDecoder.decode(id, StandardCharsets.UTF_8);
            ModelInstance model = runtime.models().get(id);
            if (model == null) {
                HttpSupport.sendError(exchange, 404, "MODEL_NOT_FOUND", "Model not loaded: " + id);
                return;
            }
            JsonObject response = new JsonObject();
            response.put("model", modelJson(model));
            HttpSupport.sendJson(exchange, 200, response);
        };
    }

    static HttpHandler infer(EnrRuntime runtime) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only POST allowed");
                return;
            }
            try {
                JsonValue body = HttpSupport.parseBody(exchange);
                String modelId = field(body, "model");
                String prompt = field(body, "prompt");
                if (modelId == null || prompt == null) {
                    HttpSupport.sendError(exchange, 400, "INVALID_REQUEST", "Missing 'model' or 'prompt'");
                    return;
                }
                EnrRuntime.InferenceOptions options = parseOptions(body);
                var result = runtime.infer(modelId, prompt, options, null);
                JsonObject response = new JsonObject();
                response.put("request_id", java.util.UUID.randomUUID().toString());
                response.put("model", modelId);
                response.put("text", result.text());
                JsonArray tokens = new JsonArray();
                for (int token : result.tokenIds()) {
                    tokens.add(token);
                }
                response.put("tokens", tokens);
                response.put("prompt_tokens", result.promptTokens());
                response.put("generated_tokens", result.generatedTokens());
                response.put("reason", result.reason().name());
                response.put("elapsed_ms", result.elapsedMs());
                HttpSupport.sendJson(exchange, 200, response);
            } catch (IllegalArgumentException e) {
                HttpSupport.sendError(exchange, 400, "INVALID_REQUEST", e.getMessage());
            } catch (Exception e) {
                HttpSupport.sendError(exchange, 500, "INFERENCE_FAILED", String.valueOf(e.getMessage()));
            }
        };
    }

    static HttpHandler inferStream(EnrRuntime runtime) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only POST allowed");
                return;
            }
            JsonValue body = HttpSupport.parseBody(exchange);
            String modelId = field(body, "model");
            String prompt = field(body, "prompt");
            if (modelId == null || prompt == null) {
                HttpSupport.sendError(exchange, 400, "INVALID_REQUEST", "Missing 'model' or 'prompt'");
                return;
            }
            EnrRuntime.InferenceOptions options = parseOptions(body);

            exchange.getResponseHeaders().set("Content-Type", "text/event-stream; charset=utf-8");
            exchange.getResponseHeaders().set("Cache-Control", "no-cache");
            exchange.getResponseHeaders().set("Connection", "keep-alive");
            exchange.sendResponseHeaders(200, 0);

            try (OutputStream out = exchange.getResponseBody()) {
                String requestId = java.util.UUID.randomUUID().toString();
                JsonObject start = new JsonObject();
                start.put("request_id", requestId);
                start.put("model", modelId);
                HttpSupport.sendSseEvent(out, "start", start);

                var handle = runtime.inferStream(modelId, prompt, options, (tokenId, piece) -> {
                    try {
                        JsonObject event = new JsonObject();
                        event.put("token_id", tokenId);
                        event.put("text", piece);
                        HttpSupport.sendSseEvent(out, "token", event);
                    } catch (IOException e) {
                        throw new RuntimeException("SSE stream closed", e);
                    }
                });

                var result = handle.future().join();
                JsonObject done = new JsonObject();
                done.put("request_id", requestId);
                done.put("reason", result.reason().name());
                done.put("prompt_tokens", result.promptTokens());
                done.put("generated_tokens", result.generatedTokens());
                done.put("elapsed_ms", result.elapsedMs());
                done.put("full_text", result.text());
                HttpSupport.sendSseEvent(out, "done", done);
            } catch (RuntimeException ignored) {
                // client disconnected
            }
        };
    }

    static HttpHandler cancel(EnrRuntime runtime) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only POST allowed");
                return;
            }
            JsonValue body = HttpSupport.parseBody(exchange);
            String requestId = field(body, "request_id");
            String sessionId = field(body, "session_id");
            if (requestId == null && sessionId == null) {
                HttpSupport.sendError(exchange, 400, "INVALID_REQUEST", "Provide 'request_id' or 'session_id'");
                return;
            }
            runtime.cancelStream(requestId, sessionId);
            JsonObject response = new JsonObject();
            response.put("cancelled", true);
            HttpSupport.sendJson(exchange, 200, response);
        };
    }

    static HttpHandler sessions(EnrRuntime runtime) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only GET allowed");
                return;
            }
            JsonArray arr = new JsonArray();
            for (var session : runtime.sessions().list()) {
                JsonObject s = new JsonObject();
                s.put("id", session.id());
                s.put("model", session.modelId());
                s.put("context_length", session.contextLength());
                s.put("created_at", session.createdAt());
                s.put("last_active_at", session.lastActiveAt());
                arr.add(s);
            }
            JsonObject body = new JsonObject();
            body.put("sessions", arr);
            HttpSupport.sendJson(exchange, 200, body);
        };
    }

    static HttpHandler metrics(EnrRuntime runtime) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only GET allowed");
                return;
            }
            HttpSupport.sendJson(exchange, 200, runtime.metrics().toJson());
        };
    }

    static HttpHandler devices(EnrRuntime runtime) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only GET allowed");
                return;
            }
            JsonArray arr = new JsonArray();
            for (var device : runtime.devices()) {
                JsonObject d = new JsonObject();
                d.put("index", device.index());
                d.put("type", device.type().id());
                d.put("name", device.name());
                d.put("total_memory_bytes", device.totalMemoryBytes());
                d.put("free_memory_bytes", device.freeMemoryBytes());
                d.put("driver_version", device.driverVersion());
                d.put("available", device.available());
                arr.add(d);
            }
            JsonObject body = new JsonObject();
            body.put("devices", arr);
            HttpSupport.sendJson(exchange, 200, body);
        };
    }

    private static EnrRuntime.InferenceOptions parseOptions(JsonValue body) {
        var defaults = EnrRuntime.InferenceOptions.defaults();
        if (!(body instanceof JsonObject obj)) {
            return defaults;
        }
        int maxTokens = obj.getInt("max_tokens", defaults.maxTokens());
        var sampling = defaults.sampling();
        if (obj.has("temperature")) {
            sampling = sampling.withTemperature(obj.getFloat("temperature", sampling.temperature()));
        }
        if (obj.has("seed") && obj.get("seed") instanceof ai.enr.core.json.JsonNumber n) {
            sampling = sampling.withSeed(n.asLong());
        }
        String sessionId = obj.getString("session_id");
        Set<Integer> stopTokens = defaults.stopTokenIds();
        var stopArr = obj.getArray("stop_token_ids");
        if (stopArr != null) {
            stopTokens = new java.util.HashSet<>();
            for (int i = 0; i < stopArr.size(); i++) {
                stopTokens.add(stopArr.getInt(i));
            }
        }
        Integer timeout = obj.has("timeout_seconds") ? obj.getInt("timeout_seconds", 0) : null;
        return new EnrRuntime.InferenceOptions(maxTokens, sampling, stopTokens, sessionId, timeout);
    }

    private static JsonObject modelJson(ModelInstance model) {
        JsonObject m = new JsonObject();
        m.put("id", model.id());
        m.put("path", model.path().toString());
        m.put("state", model.state().name().toLowerCase());
        m.put("architecture", model.loadedModel().architecture().family());
        m.put("vocab_size", model.loadedModel().architecture().vocabSize());
        m.put("d_model", model.loadedModel().architecture().dModel());
        m.put("num_heads", model.loadedModel().architecture().numHeads());
        m.put("num_kv_heads", model.loadedModel().architecture().numKvHeads());
        m.put("num_blocks", model.loadedModel().architecture().numBlocks());
        m.put("dff", model.loadedModel().architecture().dff());
        m.put("max_seq_len", model.loadedModel().architecture().maxSeqLen());
        m.put("positional_encoding", model.loadedModel().architecture().positionalEncoding());
        m.put("parameters", model.loadedModel().totalParameters());
        m.put("weight_bytes", model.loadedModel().weightBytes());
        m.put("backend", model.backend().name());
        m.put("active_sessions", model.activeSessions());
        m.put("total_requests", model.totalRequests());
        m.put("loaded_at", model.loadedAt());
        return m;
    }

    private static String field(JsonValue body, String key) {
        if (body instanceof JsonObject obj) {
            JsonValue v = obj.get(key);
            if (v instanceof ai.enr.core.json.JsonString s) {
                return s.value();
            }
            if (v instanceof ai.enr.core.json.JsonNumber n) {
                return String.valueOf((long) n.asDouble());
            }
        }
        return null;
    }

    private static JsonArray arrayOf(Iterable<String> values) {
        JsonArray arr = new JsonArray();
        for (String value : values) {
            arr.add(value);
        }
        return arr;
    }
}
