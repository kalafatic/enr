package ai.enr.client;

import ai.enr.core.json.Json;
import ai.enr.core.json.JsonObject;
import ai.enr.core.json.JsonValue;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;

/**
 * Java 21 client for ENR. Depends only on enr-core; synchronous,
 * asynchronous and streaming invocation styles.
 */
public final class EnrClient implements AutoCloseable {

    public record ModelInfo(
            String id, String path, String state, String architecture,
            int vocabSize, int dModel, int numHeads, int numKvHeads, int numBlocks,
            int dff, int maxSeqLen, String positionalEncoding, long parameters,
            long weightBytes, String backend, long activeSessions, long totalRequests) {
    }

    public record Device(
            int index, String type, String name, long totalMemoryBytes,
            long freeMemoryBytes, String driverVersion, boolean available) {
    }

    public record InferenceResult(
            String text, List<Integer> tokens, int promptTokens, int generatedTokens,
            String reason, long elapsedMs) {
    }

    public record StreamedToken(int tokenId, String text) {
    }

    public static final class EnrException extends RuntimeException {
        private final int status;
        private final String code;

        public EnrException(int status, String code, String message) {
            super(message);
            this.status = status;
            this.code = code;
        }

        public int status() {
            return status;
        }

        public String code() {
            return code;
        }
    }

    private final String baseUrl;
    private final HttpClient http;

    public EnrClient(String baseUrl) {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.http = HttpClient.newBuilder().build();
    }

    public EnrClient(String baseUrl, HttpClient http) {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.http = http;
    }

    public String baseUrl() {
        return baseUrl;
    }

    public boolean health() {
        try {
            HttpResponse<String> response = send(HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/api/v1/health"))
                    .GET().build());
            return response.statusCode() == 200;
        } catch (RuntimeException e) {
            return false;
        }
    }

    public JsonObject status() {
        return asObject(request("GET", "/api/v1/status", null));
    }

    public List<ModelInfo> models() {
        JsonObject body = asObject(request("GET", "/api/v1/models", null));
        List<ModelInfo> models = new ArrayList<>();
        var arr = body.getArray("models");
        if (arr != null) {
            for (int i = 0; i < arr.size(); i++) {
                models.add(parseModel(arr.getObject(i)));
            }
        }
        return models;
    }

    public ModelInfo model(String id) {
        return parseModel(asObject(request("GET", "/api/v1/models/" + encode(id), null)).getObject("model"));
    }

    public ModelInfo loadModel(String path, String backend) {
        JsonObject req = new JsonObject();
        req.put("path", path);
        if (backend != null) {
            req.put("backend", backend);
        }
        return parseModel(asObject(request("POST", "/api/v1/models/load", req)).getObject("model"));
    }

    public void unloadModel(String id, boolean force) {
        JsonObject req = new JsonObject();
        req.put("id", id);
        req.put("force", force);
        request("POST", "/api/v1/models/unload", req);
    }

    public InferenceResult infer(String model, String prompt) {
        return infer(model, prompt, null);
    }

    public InferenceResult infer(String model, String prompt, InferenceRequest request) {
        JsonObject req = buildInferenceRequest(model, prompt, request);
        return parseInference(asObject(request("POST", "/api/v1/inference", req)));
    }

    public CompletableFuture<InferenceResult> inferAsync(String model, String prompt, InferenceRequest request) {
        JsonObject req = buildInferenceRequest(model, prompt, request);
        return requestAsync("POST", "/api/v1/inference", req).thenApply(this::asObject)
                .thenApply(this::parseInference);
    }

    public void inferStream(String model, String prompt, InferenceRequest request,
                            Consumer<StreamedToken> onToken, Consumer<InferenceResult> onDone) {
        JsonObject req = buildInferenceRequest(model, prompt, request);
        HttpRequest httpRequest = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/api/v1/inference/stream"))
                .header("Accept", "text/event-stream")
                .POST(HttpRequest.BodyPublishers.ofString(Json.write(req, false)))
                .build();
        try {
            HttpResponse<java.io.InputStream> response = http.send(httpRequest,
                    HttpResponse.BodyHandlers.ofInputStream());
            if (response.statusCode() != 200) {
                throw new EnrException(response.statusCode(), "HTTP_ERROR",
                        "Stream request failed: " + response.statusCode());
            }
            try (java.io.InputStream in = response.body()) {
                parseSse(in, onToken, onDone);
            }
        } catch (java.io.IOException e) {
            throw new EnrException(-1, "IO_ERROR", e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new EnrException(-1, "INTERRUPTED", "Stream interrupted");
        }
    }

    public void cancel(String requestId, String sessionId) {
        JsonObject req = new JsonObject();
        if (requestId != null) {
            req.put("request_id", requestId);
        }
        if (sessionId != null) {
            req.put("session_id", sessionId);
        }
        request("POST", "/api/v1/inference/cancel", req);
    }

    public List<JsonObject> sessions() {
        JsonObject body = asObject(request("GET", "/api/v1/sessions", null));
        List<JsonObject> sessions = new ArrayList<>();
        var arr = body.getArray("sessions");
        if (arr != null) {
            for (int i = 0; i < arr.size(); i++) {
                sessions.add(arr.getObject(i));
            }
        }
        return sessions;
    }

    public JsonObject metrics() {
        return asObject(request("GET", "/api/v1/metrics", null));
    }

    public List<Device> devices() {
        JsonObject body = asObject(request("GET", "/api/v1/devices", null));
        List<Device> devices = new ArrayList<>();
        var arr = body.getArray("devices");
        if (arr != null) {
            for (int i = 0; i < arr.size(); i++) {
                JsonObject d = arr.getObject(i);
                devices.add(new Device(
                        d.getInt("index", -1),
                        d.getString("type", "unknown"),
                        d.getString("name", ""),
                        d.getLong("total_memory_bytes", 0),
                        d.getLong("free_memory_bytes", 0),
                        d.getString("driver_version", ""),
                        d.getBoolean("available", false)));
            }
        }
        return devices;
    }

    public static final class InferenceRequest {
        private Integer maxTokens;
        private Float temperature;
        private Float topP;
        private Integer topK;
        private Float repeatPenalty;
        private Long seed;
        private String sessionId;
        private List<Integer> stopTokenIds;

        public InferenceRequest maxTokens(int maxTokens) {
            this.maxTokens = maxTokens;
            return this;
        }

        public InferenceRequest temperature(float temperature) {
            this.temperature = temperature;
            return this;
        }

        public InferenceRequest topP(float topP) {
            this.topP = topP;
            return this;
        }

        public InferenceRequest topK(int topK) {
            this.topK = topK;
            return this;
        }

        public InferenceRequest repeatPenalty(float repeatPenalty) {
            this.repeatPenalty = repeatPenalty;
            return this;
        }

        public InferenceRequest seed(long seed) {
            this.seed = seed;
            return this;
        }

        public InferenceRequest session(String sessionId) {
            this.sessionId = sessionId;
            return this;
        }

        public InferenceRequest addStopToken(int tokenId) {
            if (stopTokenIds == null) {
                stopTokenIds = new ArrayList<>();
            }
            stopTokenIds.add(tokenId);
            return this;
        }
    }

    private JsonObject buildInferenceRequest(String model, String prompt, InferenceRequest request) {
        JsonObject req = new JsonObject();
        req.put("model", model);
        req.put("prompt", prompt);
        if (request != null) {
            if (request.maxTokens != null) {
                req.put("max_tokens", request.maxTokens);
            }
            if (request.temperature != null) {
                req.put("temperature", request.temperature);
            }
            if (request.topP != null) {
                req.put("top_p", request.topP);
            }
            if (request.topK != null) {
                req.put("top_k", request.topK);
            }
            if (request.repeatPenalty != null) {
                req.put("repeat_penalty", request.repeatPenalty);
            }
            if (request.seed != null) {
                req.put("seed", request.seed);
            }
            if (request.sessionId != null) {
                req.put("session_id", request.sessionId);
            }
            if (request.stopTokenIds != null) {
                var arr = new ai.enr.core.json.JsonArray();
                for (int token : request.stopTokenIds) {
                    arr.add(token);
                }
                req.put("stop_token_ids", arr);
            }
        }
        return req;
    }

    private ModelInfo parseModel(JsonObject m) {
        return new ModelInfo(
                m.getString("id", ""),
                m.getString("path", ""),
                m.getString("state", ""),
                m.getString("architecture", ""),
                m.getInt("vocab_size", 0),
                m.getInt("d_model", 0),
                m.getInt("num_heads", 0),
                m.getInt("num_kv_heads", 0),
                m.getInt("num_blocks", 0),
                m.getInt("dff", 0),
                m.getInt("max_seq_len", 0),
                m.getString("positional_encoding", ""),
                m.getLong("parameters", 0),
                m.getLong("weight_bytes", 0),
                m.getString("backend", ""),
                m.getLong("active_sessions", 0),
                m.getLong("total_requests", 0));
    }

    private InferenceResult parseInference(JsonObject body) {
        List<Integer> tokens = new ArrayList<>();
        var arr = body.getArray("tokens");
        if (arr != null) {
            for (int i = 0; i < arr.size(); i++) {
                tokens.add(arr.getInt(i));
            }
        }
        return new InferenceResult(
                body.getString("text", ""),
                tokens,
                body.getInt("prompt_tokens", 0),
                body.getInt("generated_tokens", 0),
                body.getString("reason", ""),
                body.getLong("elapsed_ms", 0));
    }

    private JsonObject asObject(HttpResponse<String> response) {
        if (response.statusCode() >= 400) {
            String code = "HTTP_" + response.statusCode();
            String message = response.body();
            try {
                JsonValue parsed = Json.parse(response.body());
                if (parsed instanceof JsonObject obj) {
                    JsonObject error = obj.getObject("error");
                    if (error != null) {
                        code = error.getString("code", code);
                        message = error.getString("message", message);
                    }
                }
            } catch (RuntimeException ignored) {
                // keep raw body as message
            }
            throw new EnrException(response.statusCode(), code, message);
        }
        return Json.parseObject(response.body());
    }

    private HttpResponse<String> request(String method, String path, JsonObject body) {
        HttpRequest.Builder builder = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .header("Content-Type", "application/json");
        if (body != null) {
            builder.method(method, HttpRequest.BodyPublishers.ofString(Json.write(body, false)));
        } else {
            builder.method(method, HttpRequest.BodyPublishers.noBody());
        }
        return send(builder.build());
    }

    private CompletableFuture<HttpResponse<String>> requestAsync(String method, String path, JsonObject body) {
        HttpRequest.Builder builder = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .header("Content-Type", "application/json");
        if (body != null) {
            builder.method(method, HttpRequest.BodyPublishers.ofString(Json.write(body, false)));
        } else {
            builder.method(method, HttpRequest.BodyPublishers.noBody());
        }
        return http.sendAsync(builder.build(), HttpResponse.BodyHandlers.ofString());
    }

    private HttpResponse<String> send(HttpRequest request) {
        try {
            return http.send(request, HttpResponse.BodyHandlers.ofString());
        } catch (java.io.IOException e) {
            throw new EnrException(-1, "IO_ERROR", e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new EnrException(-1, "INTERRUPTED", "Request interrupted");
        }
    }

    private void parseSse(java.io.InputStream in, Consumer<StreamedToken> onToken,
                          Consumer<InferenceResult> onDone) throws java.io.IOException {
        java.io.BufferedReader reader = new java.io.BufferedReader(
                new java.io.InputStreamReader(in, StandardCharsets.UTF_8));
        String line;
        StringBuilder data = new StringBuilder();
        while ((line = reader.readLine()) != null) {
            if (line.startsWith("data: ")) {
                data.append(line.substring(6)).append('\n');
            } else if (line.isEmpty()) {
                if (data.length() > 0) {
                    JsonValue parsed = Json.parse(data.toString());
                    if (parsed instanceof JsonObject obj) {
                        if (obj.has("token_id")) {
                            onToken.accept(new StreamedToken(
                                    obj.getInt("token_id", -1), obj.getString("text", "")));
                        } else if (obj.has("reason")) {
                            onDone.accept(new InferenceResult(
                                    obj.getString("full_text", ""), List.of(),
                                    obj.getInt("prompt_tokens", 0), obj.getInt("generated_tokens", 0),
                                    obj.getString("reason", ""), obj.getLong("elapsed_ms", 0)));
                        }
                    }
                    data.setLength(0);
                }
            }
        }
    }

    private static String encode(String value) {
        return java.net.URLEncoder.encode(value, StandardCharsets.UTF_8);
    }

    @Override
    public void close() {
    }
}
