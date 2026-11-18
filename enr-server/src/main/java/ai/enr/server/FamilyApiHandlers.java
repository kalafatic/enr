package ai.enr.server;

import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonObject;
import ai.enr.core.json.JsonString;
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
import java.nio.file.Path;

/**
 * Family ENR 1.0 wire surface (base {@code /enr/v1}), as consumed by DCode,
 * TRYST and DStudio. Maps the family contract (see the normative ENR-1.0.md in
 * the EVO repository) onto the standalone ENR runtime. Errors use the family
 * TEIP error envelope {@code {code,message,retryable,timestamp}}.
 */
final class FamilyApiHandlers {

    private FamilyApiHandlers() {
    }

    static HttpHandler familyHealth(EnrRuntime runtime) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendFamilyError(exchange, 405, "VALIDATION_FAILED", "Only GET allowed");
                return;
            }
            boolean healthy = runtime.healthChecks().isEmpty();
            JsonObject body = new JsonObject();
            body.put("application", "ENR");
            body.put("version", "1.0");
            body.put("status", healthy ? "UP" : "DEGRADED");
            body.put("ready", healthy && runtime.isRunning());
            body.put("timestamp", System.currentTimeMillis());
            HttpSupport.sendJson(exchange, healthy ? 200 : 503, body);
        };
    }

    static HttpHandler familyListModels(EnrRuntime runtime) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendFamilyError(exchange, 405, "VALIDATION_FAILED", "Only GET allowed");
                return;
            }
            JsonArray models = new JsonArray();
            for (ModelInstance model : runtime.models().list()) {
                models.add(familyModelJson(model));
            }
            HttpSupport.sendJson(exchange, 200, models);
        };
    }

    static HttpHandler familyModel(EnrRuntime runtime) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendFamilyError(exchange, 405, "VALIDATION_FAILED", "Only GET allowed");
                return;
            }
            String id = familyPathId(exchange);
            ModelInstance model = runtime.models().get(id);
            if (model == null) {
                sendFamilyError(exchange, 404, "NOT_FOUND", "Model not loaded: " + id);
                return;
            }
            HttpSupport.sendJson(exchange, 200, familyModelJson(model));
        };
    }

    /** Routes {@code /enr/v1/models/{id}}, {@code .../{id}/load} and {@code .../{id}/unload}. */
    static HttpHandler familyModelRoute(EnrRuntime runtime) {
        return exchange -> {
            String path = exchange.getRequestURI().getPath();
            if (path.endsWith("/load")) {
                familyLoadModel(runtime).handle(exchange);
            } else if (path.endsWith("/unload")) {
                familyUnloadModel(runtime).handle(exchange);
            } else {
                familyModel(runtime).handle(exchange);
            }
        };
    }

    static HttpHandler familyLoadModel(EnrRuntime runtime) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendFamilyError(exchange, 405, "VALIDATION_FAILED", "Only POST allowed");
                return;
            }
            String id = familyPathId(exchange);
            try {
                JsonValue body = HttpSupport.parseBody(exchange);
                String backend = stringField(body, "backend");
                Path path = runtime.models().resolveModelPath(id);
                if (path == null) {
                    sendFamilyError(exchange, 404, "NOT_FOUND", "Model not found: " + id);
                    return;
                }
                ModelInstance model = runtime.loadModel(path, backend);
                JsonObject response = new JsonObject();
                response.put("model", id);
                response.put("success", true);
                response.put("state", model.state().name().toLowerCase());
                HttpSupport.sendJson(exchange, 200, response);
            } catch (IllegalArgumentException e) {
                sendFamilyError(exchange, 400, "VALIDATION_FAILED", e.getMessage());
            } catch (Exception e) {
                sendFamilyError(exchange, 422, "INTERNAL_ERROR", String.valueOf(e.getMessage()));
            }
        };
    }

    static HttpHandler familyUnloadModel(EnrRuntime runtime) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendFamilyError(exchange, 405, "VALIDATION_FAILED", "Only POST allowed");
                return;
            }
            String id = familyPathId(exchange);
            try {
                JsonValue body = HttpSupport.parseBody(exchange);
                boolean force = "true".equalsIgnoreCase(stringField(body, "force"));
                runtime.unloadModel(id, force);
                JsonObject response = new JsonObject();
                response.put("model", id);
                response.put("success", true);
                HttpSupport.sendJson(exchange, 200, response);
            } catch (IllegalArgumentException e) {
                sendFamilyError(exchange, 404, "NOT_FOUND", e.getMessage());
            } catch (IllegalStateException e) {
                sendFamilyError(exchange, 409, "CONFLICT", e.getMessage());
            }
        };
    }

    static HttpHandler familyChat(EnrRuntime runtime) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendFamilyError(exchange, 405, "VALIDATION_FAILED", "Only POST allowed");
                return;
            }
            try {
                JsonValue body = HttpSupport.parseBody(exchange);
                String modelId = stringField(body, "model");
                String prompt = messagesToPrompt(body);
                if (modelId == null || modelId.isBlank()) {
                    sendFamilyError(exchange, 400, "VALIDATION_FAILED", "Missing 'model' field");
                    return;
                }
                EnrRuntime.InferenceOptions options = familyOptions(body);
                long started = System.currentTimeMillis();
                EnrRuntime.TextResult result = runtime.infer(modelId, prompt, options, null);
                JsonObject response = new JsonObject();
                response.put("content", result.text());
                response.put("finishReason", familyFinishReason(result.reason()));
                response.put("engine", "enr-native");
                response.put("model", modelId);
                response.put("durationMs", result.elapsedMs());
                response.put("promptTokens", result.promptTokens());
                response.put("completionTokens", result.generatedTokens());
                response.put("durationMsTotal", System.currentTimeMillis() - started);
                HttpSupport.sendJson(exchange, 200, response);
            } catch (IllegalArgumentException e) {
                sendFamilyError(exchange, 400, "VALIDATION_FAILED", e.getMessage());
            } catch (Exception e) {
                e.printStackTrace();
                java.util.logging.Logger.getLogger(FamilyApiHandlers.class.getName())
                        .log(java.util.logging.Level.SEVERE, "family chat failed", e);
                sendFamilyError(exchange, 500, "INTERNAL_ERROR", String.valueOf(e.getMessage()));
            }
        };
    }

    static HttpHandler familyChatStream(EnrRuntime runtime) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendFamilyError(exchange, 405, "VALIDATION_FAILED", "Only POST allowed");
                return;
            }
            JsonValue body = HttpSupport.parseBody(exchange);
            String modelId = stringField(body, "model");
            String prompt = messagesToPrompt(body);
            if (modelId == null || modelId.isBlank()) {
                sendFamilyError(exchange, 400, "VALIDATION_FAILED", "Missing 'model' field");
                return;
            }
            EnrRuntime.InferenceOptions options = familyOptions(body);

            exchange.getResponseHeaders().set("Content-Type", "text/event-stream; charset=utf-8");
            exchange.getResponseHeaders().set("Cache-Control", "no-cache");
            exchange.getResponseHeaders().set("Connection", "keep-alive");
            exchange.sendResponseHeaders(200, 0);

            try (OutputStream out = exchange.getResponseBody()) {
                var handle = runtime.inferStream(modelId, prompt, options, (tokenId, piece) -> {
                    try {
                        JsonObject event = new JsonObject();
                        event.put("chunk", piece);
                        event.put("engine", "enr-native");
                        event.put("done", false);
                        HttpSupport.sendSseEvent(out, null, event);
                    } catch (IOException e) {
                        throw new RuntimeException("SSE stream closed", e);
                    }
                });

                EnrRuntime.TextResult result = handle.future().join();
                JsonObject chunk = new JsonObject();
                chunk.put("chunk", "");
                chunk.put("engine", "enr-native");
                chunk.put("done", true);
                HttpSupport.sendSseEvent(out, "done", chunk);
                JsonObject done = new JsonObject();
                done.put("model", modelId);
                done.put("finishReason", familyFinishReason(result.reason()));
                done.put("durationMs", result.elapsedMs());
                done.put("promptTokens", result.promptTokens());
                done.put("completionTokens", result.generatedTokens());
                HttpSupport.sendSseEvent(out, null, done);
                exchange.close();
            } catch (RuntimeException ignored) {
                // client disconnected
            } catch (Exception e) {
                sendFamilyError(exchange, 500, "INTERNAL_ERROR", String.valueOf(e.getMessage()));
            }
        };
    }

    static HttpHandler familyRuntime(EnrRuntime runtime) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendFamilyError(exchange, 405, "VALIDATION_FAILED", "Only GET allowed");
                return;
            }
            Runtime jvm = Runtime.getRuntime();
            JsonObject body = new JsonObject();
            body.put("running", runtime.isRunning());
            JsonObject cpu = new JsonObject();
            cpu.put("processors", jvm.availableProcessors());
            cpu.put("loadAverage", jvm.availableProcessors());
            body.put("cpu", cpu);
            JsonObject heap = new JsonObject();
            heap.put("usedBytes", jvm.totalMemory() - jvm.freeMemory());
            heap.put("maxBytes", jvm.maxMemory());
            body.put("heap", heap);
            JsonArray devices = new JsonArray();
            for (var device : runtime.devices()) {
                JsonObject d = new JsonObject();
                d.put("index", device.index());
                d.put("type", device.type().id());
                d.put("name", device.name());
                d.put("totalMemoryBytes", device.totalMemoryBytes());
                d.put("freeMemoryBytes", device.freeMemoryBytes());
                d.put("available", device.available());
                devices.add(d);
            }
            body.put("devices", devices);
            JsonArray engines = new JsonArray();
            engines.add("enr-native");
            body.put("activeEngines", engines);
            HttpSupport.sendJson(exchange, 200, body);
        };
    }

    static HttpHandler familyProviders(EnrRuntime runtime) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendFamilyError(exchange, 405, "VALIDATION_FAILED", "Only GET allowed");
                return;
            }
            JsonArray providers = new JsonArray();
            JsonObject nativeProvider = new JsonObject();
            nativeProvider.put("id", "enr-native");
            nativeProvider.put("name", "ENR Native");
            nativeProvider.put("type", "ENR");
            nativeProvider.put("baseUrl", "http://" + runtime.config().host() + ":" + runtime.config().port());
            nativeProvider.put("local", true);
            nativeProvider.put("available", runtime.isRunning());
            providers.add(nativeProvider);
            HttpSupport.sendJson(exchange, 200, providers);
        };
    }

    static HttpHandler familyCapabilities(EnrRuntime runtime) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendFamilyError(exchange, 405, "VALIDATION_FAILED", "Only GET allowed");
                return;
            }
            JsonArray capabilities = new JsonArray();
            capabilities.add(capability("ENR_INFERENCE", "ENR Inference",
                    "Blocking chat inference through the ENR runtime", "inference.generation"));
            capabilities.add(capability("ENR_STREAMING", "ENR Streaming",
                    "Token-level streaming inference (SSE)", "inference.streaming"));
            capabilities.add(capability("ENR_MODEL_MANAGEMENT", "ENR Model Management",
                    "Model discovery, loading and unloading", "inference.model.lifecycle"));
            capabilities.add(capability("ENR_RUNTIME_MONITORING", "ENR Runtime Monitoring",
                    "Runtime health, devices and resource reporting", "runtime.monitoring"));
            HttpSupport.sendJson(exchange, 200, capabilities);
        };
    }

    private static JsonObject capability(String id, String name, String description, String feature) {
        JsonObject capability = new JsonObject();
        capability.put("id", id);
        capability.put("name", name);
        capability.put("description", description);
        JsonArray features = new JsonArray();
        features.add(feature);
        capability.put("features", features);
        return capability;
    }

    private static JsonObject familyModelJson(ModelInstance model) {
        JsonObject m = new JsonObject();
        m.put("id", model.id());
        m.put("name", model.id());
        m.put("provider", "enr-native");
        m.put("engine", "enr-native");
        m.put("sizeBytes", model.totalWeightBytes());
        m.put("available", model.state() == ModelInstance.State.READY);
        m.put("loaded", true);
        m.put("contextLength", model.loadedModel().architecture().maxSeqLen());
        m.put("backend", model.backend().name());
        m.put("state", model.state().name().toLowerCase());
        return m;
    }

    private static String familyFinishReason(GenerationEngine.TerminationReason reason) {
        return switch (reason) {
            case EOS_REACHED, STOP_TOKEN, MAX_SEQ_LEN_REACHED, MAX_TOKENS_REACHED -> "stop";
            case CANCELLED -> "cancelled";
            case ERROR -> "error";
        };
    }

    private static String messagesToPrompt(JsonValue body) {
        if (!(body instanceof JsonObject obj)) {
            return "";
        }
        JsonArray messages = obj.getArray("messages");
        if (messages == null) {
            return "";
        }
        StringBuilder prompt = new StringBuilder();
        for (int i = 0; i < messages.size(); i++) {
            JsonValue message = messages.get(i);
            if (!(message instanceof JsonObject entry)) {
                continue;
            }
            String content = entry.getString("content");
            if (content == null || content.isBlank()) {
                continue;
            }
            prompt.append(content).append('\n');
        }
        return prompt.toString();
    }

    private static EnrRuntime.InferenceOptions familyOptions(JsonValue body) {
        var defaults = EnrRuntime.InferenceOptions.defaults();
        if (!(body instanceof JsonObject obj)) {
            return defaults;
        }
        int maxTokens = obj.getInt("maxTokens", defaults.maxTokens());
        if (!obj.has("maxTokens")) {
            maxTokens = obj.getInt("max_tokens", defaults.maxTokens());
        }
        var sampling = defaults.sampling();
        if (obj.has("temperature")) {
            sampling = sampling.withTemperature(obj.getFloat("temperature", sampling.temperature()));
        }
        Integer timeout = null;
        if (obj.has("timeoutSeconds")) {
            timeout = obj.getInt("timeoutSeconds", 0);
        } else if (obj.has("timeout_seconds")) {
            timeout = obj.getInt("timeout_seconds", 0);
        }
        return new EnrRuntime.InferenceOptions(maxTokens, sampling, defaults.stopTokenIds(),
                defaults.sessionId(), timeout);
    }

    private static String stringField(JsonValue body, String key) {
        if (body instanceof JsonObject obj) {
            JsonValue v = obj.get(key);
            if (v instanceof JsonString s) {
                return s.value();
            }
        }
        return null;
    }

    private static String familyPathId(HttpExchange exchange) {
        String path = exchange.getRequestURI().getPath();
        int prefix = path.indexOf("/enr/v1/models/");
        String id = prefix >= 0 ? path.substring(prefix + "/enr/v1/models/".length()) : "";
        int action = id.indexOf('/');
        if (action >= 0) {
            id = id.substring(0, action);
        }
        return URLDecoder.decode(id, StandardCharsets.UTF_8);
    }

    private static void sendFamilyError(HttpExchange exchange, int status, String code,
                                        String message) throws IOException {
        JsonObject body = new JsonObject();
        body.put("code", code);
        body.put("message", message == null ? "" : message);
        body.put("retryable", false);
        body.put("timestamp", System.currentTimeMillis());
        HttpSupport.sendJson(exchange, status, body);
    }
}
