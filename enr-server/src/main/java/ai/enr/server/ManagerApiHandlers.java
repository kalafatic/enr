package ai.enr.server;

import ai.enr.core.json.Json;
import ai.enr.core.json.JsonObject;
import ai.enr.manager.EnrManager;
import ai.enr.manager.provider.ChatMessage;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import java.io.IOException;
import java.io.OutputStream;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Management REST API: model catalog (discover/pull/import/verify/remove),
 * providers (CRUD/test/models/health), downloads, unified chat, diagnostics,
 * logs and hardware. All endpoints delegate to EnrManager — the single source
 * of truth.
 */
final class ManagerApiHandlers {

    private ManagerApiHandlers() {
    }

    // ------------------------------------------------------------ catalog

    static HttpHandler catalogModels(EnrManager manager) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only GET allowed");
                return;
            }
            HttpSupport.sendJson(exchange, 200, manager.listModels());
        };
    }

    static HttpHandler catalogSearch(EnrManager manager) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only GET allowed");
                return;
            }
            String query = queryParam(exchange, "q", "");
            int limit = intQueryParam(exchange, "limit", 20);
            HttpSupport.sendJson(exchange, 200, manager.searchCatalog(query, limit));
        };
    }

    static HttpHandler catalogPull(EnrManager manager) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only POST allowed");
                return;
            }
            if (!authorize(exchange, manager)) {
                return;
            }
            JsonObject request = HttpSupport.parseBodyObject(exchange);
            String ref = request.getString("name", request.getString("ref", ""));
            if (ref.isBlank()) {
                HttpSupport.sendError(exchange, 400, "INVALID_REQUEST", "Missing 'name' field");
                return;
            }
            try {
                String jobId = manager.pullModel(ref);
                JsonObject response = new JsonObject();
                response.put("download_id", jobId);
                response.put("name", ref);
                HttpSupport.sendJson(exchange, 202, response);
            } catch (Exception e) {
                HttpSupport.sendError(exchange, 422, "PULL_FAILED", e.getMessage() == null
                        ? e.getClass().getSimpleName() : e.getMessage());
            }
        };
    }

    static HttpHandler catalogImport(EnrManager manager) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only POST allowed");
                return;
            }
            if (!authorize(exchange, manager)) {
                return;
            }
            JsonObject request = HttpSupport.parseBodyObject(exchange);
            String path = request.getString("path", "");
            if (path.isBlank()) {
                HttpSupport.sendError(exchange, 400, "INVALID_REQUEST", "Missing 'path' field");
                return;
            }
            try {
                JsonObject model = manager.importModel(Path.of(path), request.getString("id"));
                JsonObject response = new JsonObject();
                response.put("model", model);
                HttpSupport.sendJson(exchange, 200, response);
            } catch (Exception e) {
                HttpSupport.sendError(exchange, 422, "IMPORT_FAILED", e.getMessage());
            }
        };
    }

    static HttpHandler catalogVerify(EnrManager manager) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only POST allowed");
                return;
            }
            if (!authorize(exchange, manager)) {
                return;
            }
            JsonObject request = HttpSupport.parseBodyObject(exchange);
            String id = request.getString("id", "");
            try {
                JsonObject response = manager.verifyModel(id);
                HttpSupport.sendJson(exchange, response.getBoolean("verified", false) ? 200 : 409, response);
            } catch (IllegalArgumentException e) {
                HttpSupport.sendError(exchange, 404, "MODEL_NOT_FOUND", e.getMessage());
            }
        };
    }

    /** Routes /api/v1/catalog/models/{id}: GET details, DELETE remove. */
    static HttpHandler catalogModelRoute(EnrManager manager) {
        return exchange -> {
            String id = pathId(exchange, "/api/v1/catalog/models/");
            switch (exchange.getRequestMethod().toUpperCase()) {
                case "GET" -> HttpSupport.sendJson(exchange, 200, manager.modelDetails(id));
                case "DELETE" -> {
                    if (!authorize(exchange, manager)) {
                        return;
                    }
                    try {
                        HttpSupport.sendJson(exchange, 200, manager.removeModel(id));
                    } catch (IllegalArgumentException e) {
                        HttpSupport.sendError(exchange, 404, "MODEL_NOT_FOUND", e.getMessage());
                    } catch (Exception e) {
                        HttpSupport.sendError(exchange, 500, "REMOVE_FAILED", e.getMessage());
                    }
                }
                default -> HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "GET or DELETE");
            }
        };
    }

    // ------------------------------------------------------------ model lifecycle

    static HttpHandler modelStart(EnrManager manager) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only POST allowed");
                return;
            }
            if (!authorize(exchange, manager)) {
                return;
            }
            JsonObject request = HttpSupport.parseBodyObject(exchange);
            String id = request.getString("id", "");
            try {
                JsonObject model = manager.startModel(id, request.getString("backend"));
                JsonObject response = new JsonObject();
                response.put("model", model);
                HttpSupport.sendJson(exchange, 200, response);
            } catch (IllegalArgumentException e) {
                HttpSupport.sendError(exchange, 404, "MODEL_NOT_FOUND", e.getMessage());
            } catch (Exception e) {
                HttpSupport.sendError(exchange, 422, "START_FAILED", e.getMessage());
            }
        };
    }

    static HttpHandler modelStop(EnrManager manager) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only POST allowed");
                return;
            }
            if (!authorize(exchange, manager)) {
                return;
            }
            JsonObject request = HttpSupport.parseBodyObject(exchange);
            String id = request.getString("id", "");
            boolean force = request.getBoolean("force", false);
            try {
                manager.stopModel(id, force);
                JsonObject response = new JsonObject();
                response.put("stopped", id);
                HttpSupport.sendJson(exchange, 200, response);
            } catch (IllegalArgumentException e) {
                HttpSupport.sendError(exchange, 404, "MODEL_NOT_FOUND", e.getMessage());
            } catch (IllegalStateException e) {
                HttpSupport.sendError(exchange, 409, "MODEL_IN_USE", e.getMessage());
            }
        };
    }

    // ------------------------------------------------------------ downloads

    static HttpHandler downloads(EnrManager manager) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only GET allowed");
                return;
            }
            HttpSupport.sendJson(exchange, 200, manager.downloads().toJson());
        };
    }

    static HttpHandler downloadCancel(EnrManager manager) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only POST allowed");
                return;
            }
            if (!authorize(exchange, manager)) {
                return;
            }
            JsonObject request = HttpSupport.parseBodyObject(exchange);
            String id = request.getString("id", "");
            boolean cancelled = manager.downloads().cancel(id);
            JsonObject response = new JsonObject();
            response.put("id", id);
            response.put("cancelled", cancelled);
            HttpSupport.sendJson(exchange, cancelled ? 200 : 404, response);
        };
    }

    // ------------------------------------------------------------ providers

    static HttpHandler providers(EnrManager manager) {
        return exchange -> {
            if (!authorize(exchange, manager)) {
                return;
            }
            switch (exchange.getRequestMethod().toUpperCase()) {
                case "GET" -> HttpSupport.sendJson(exchange, 200, manager.providers().toJson());
                case "POST" -> {
                    JsonObject request = HttpSupport.parseBodyObject(exchange);
                    String id = request.getString("id", "");
                    String type = request.getString("type", "");
                    try {
                        JsonObject provider = manager.addProvider(id, type,
                                request.getString("name"),
                                request.getString("base_url"),
                                request.getString("api_key"),
                                request.getBoolean("enabled", true));
                        HttpSupport.sendJson(exchange, 200, provider);
                    } catch (IllegalArgumentException e) {
                        HttpSupport.sendError(exchange, 400, "INVALID_REQUEST", e.getMessage());
                    }
                }
                default -> HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "GET or POST");
            }
        };
    }

    /** Routes /api/v1/providers/{id}[/action]. */
    static HttpHandler providerRoute(EnrManager manager) {
        return exchange -> {
            if (!authorize(exchange, manager)) {
                return;
            }
            String path = exchange.getRequestURI().getPath();
            String id = pathId(exchange, "/api/v1/providers/");
            int actionIdx = id.indexOf('/');
            String action = "";
            if (actionIdx >= 0) {
                action = id.substring(actionIdx + 1);
                id = id.substring(0, actionIdx);
            }
            id = URLDecoder.decode(id, StandardCharsets.UTF_8);
            switch (action) {
                case "test" -> {
                    try {
                        HttpSupport.sendJson(exchange, 200, manager.testProvider(id));
                    } catch (IllegalArgumentException e) {
                        HttpSupport.sendError(exchange, 404, "PROVIDER_NOT_FOUND", e.getMessage());
                    }
                }
                case "models" -> {
                    try {
                        HttpSupport.sendJson(exchange, 200, manager.providerModels(id));
                    } catch (IllegalArgumentException e) {
                        HttpSupport.sendError(exchange, 404, "PROVIDER_NOT_FOUND", e.getMessage());
                    }
                }
                case "health" -> {
                    try {
                        HttpSupport.sendJson(exchange, 200, manager.providerHealth(id));
                    } catch (IllegalArgumentException e) {
                        HttpSupport.sendError(exchange, 404, "PROVIDER_NOT_FOUND", e.getMessage());
                    }
                }
                default -> {
                    switch (exchange.getRequestMethod().toUpperCase()) {
                        case "PUT" -> {
                            JsonObject request = HttpSupport.parseBodyObject(exchange);
                            try {
                                JsonObject provider = manager.updateProvider(id,
                                        request.getString("name"),
                                        request.getString("base_url"),
                                        request.getString("api_key"),
                                        request.has("enabled") ? request.getBoolean("enabled", true) : null);
                                HttpSupport.sendJson(exchange, 200, provider);
                            } catch (IllegalArgumentException e) {
                                HttpSupport.sendError(exchange, 404, "PROVIDER_NOT_FOUND", e.getMessage());
                            }
                        }
                        case "DELETE" -> {
                            try {
                                HttpSupport.sendJson(exchange, 200, manager.deleteProvider(id));
                            } catch (IllegalArgumentException e) {
                                HttpSupport.sendError(exchange, 404, "PROVIDER_NOT_FOUND", e.getMessage());
                            }
                        }
                        default -> HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED",
                                "PUT, DELETE, or /test /models /health");
                    }
                }
            }
        };
    }

    // ------------------------------------------------------------ chat

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
                if (model.isBlank()) {
                    HttpSupport.sendError(exchange, 400, "INVALID_REQUEST", "Missing 'model' field");
                    return;
                }
                EnrManager.ChatOptions options = optionsFrom(request, stream);
                manager.validateTools(model, options);
                if (!stream) {
                    JsonObject response = manager.chat(model, messages, options);
                    HttpSupport.sendJson(exchange, response.has("error") ? 500 : 200, response);
                    return;
                }
                exchange.getResponseHeaders().set("Content-Type", "text/event-stream; charset=utf-8");
                exchange.getResponseHeaders().set("Cache-Control", "no-cache");
                exchange.sendResponseHeaders(200, 0);
                try (OutputStream out = exchange.getResponseBody()) {
                    manager.chatStream(model, messages, options, chunk -> {
                        try {
                            HttpSupport.sendSseEvent(out, "chunk", chunk);
                        } catch (IOException e) {
                            throw new RuntimeException("client disconnected", e);
                        }
                    });
                } catch (RuntimeException ignored) {
                    // client disconnected
                }
            } catch (EnrManager.ToolCallingUnsupportedException e) {
                HttpSupport.sendError(exchange, 400, "INVALID_REQUEST", e.getMessage());
            } catch (IllegalArgumentException e) {
                HttpSupport.sendError(exchange, 404, "MODEL_NOT_FOUND", e.getMessage());
            } catch (Exception e) {
                HttpSupport.sendError(exchange, 500, "INFERENCE_FAILED", e.getMessage());
            }
        };
    }

    // ------------------------------------------------------------ diagnostics

    static HttpHandler diagnostics(EnrManager manager) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only GET allowed");
                return;
            }
            if (!authorize(exchange, manager)) {
                return;
            }
            HttpSupport.sendJson(exchange, 200, manager.diagnostics());
        };
    }

    static HttpHandler dashboard(EnrManager manager) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only GET allowed");
                return;
            }
            HttpSupport.sendJson(exchange, 200, manager.dashboard());
        };
    }

    static HttpHandler logs(EnrManager manager) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only GET allowed");
                return;
            }
            if (!authorize(exchange, manager)) {
                return;
            }
            String source = queryParam(exchange, "source", "all");
            int tail = intQueryParam(exchange, "tail", 200);
            HttpSupport.sendJson(exchange, 200, manager.logs(source, tail));
        };
    }

    static HttpHandler hardware(EnrManager manager) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only GET allowed");
                return;
            }
            HttpSupport.sendJson(exchange, 200, manager.diagnostics().getObject("hardware"));
        };
    }

    // ------------------------------------------------------------ model management

    /** GET /api/v1/models/providers: registered model providers with capabilities. */
    static HttpHandler modelProviders(EnrManager manager) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only GET allowed");
                return;
            }
            HttpSupport.sendJson(exchange, 200, manager.modelManagement().providersToJson());
        };
    }

    /**
     * Routes /api/v1/models/providers/{provider}[/action]:
     * GET available|installed, POST pull, DELETE {modelId}.
     */
    static HttpHandler modelProviderRoute(EnrManager manager) {
        return exchange -> {
            String rest = pathId(exchange, "/api/v1/models/providers/");
            int slash = rest.indexOf('/');
            String providerId = URLDecoder.decode(
                    slash >= 0 ? rest.substring(0, slash) : rest, StandardCharsets.UTF_8);
            String action = slash >= 0 ? rest.substring(slash + 1) : "";
            var models = manager.modelManagement();
            switch (exchange.getRequestMethod().toUpperCase()) {
                case "GET" -> {
                    switch (action) {
                        case "available" -> {
                            try {
                                HttpSupport.sendJson(exchange, 200, models.modelsToJson(providerId));
                            } catch (IllegalArgumentException e) {
                                HttpSupport.sendError(exchange, 404, "PROVIDER_NOT_FOUND", e.getMessage());
                            }
                        }
                        case "installed" -> {
                            try {
                                HttpSupport.sendJson(exchange, 200, models.installedToJson(providerId));
                            } catch (IllegalArgumentException e) {
                                HttpSupport.sendError(exchange, 404, "PROVIDER_NOT_FOUND", e.getMessage());
                            }
                        }
                        default -> HttpSupport.sendError(exchange, 404, "NOT_FOUND",
                                "Unknown model provider action: " + action);
                    }
                }
                case "POST" -> {
                    if (!"pull".equals(action)) {
                        HttpSupport.sendError(exchange, 404, "NOT_FOUND",
                                "Unknown model provider action: " + action);
                        return;
                    }
                    if (!authorize(exchange, manager)) {
                        return;
                    }
                    JsonObject request = HttpSupport.parseBodyObject(exchange);
                    String model = request.getString("model", "");
                    if (model.isBlank()) {
                        HttpSupport.sendError(exchange, 400, "INVALID_REQUEST", "Missing 'model' field");
                        return;
                    }
                    try {
                        HttpSupport.sendJson(exchange, 202, models.pullModel(providerId, model).toJson());
                    } catch (IllegalArgumentException e) {
                        HttpSupport.sendError(exchange, 404, "PROVIDER_NOT_FOUND", e.getMessage());
                    } catch (IllegalStateException e) {
                        HttpSupport.sendError(exchange, 409, "PULL_UNSUPPORTED", e.getMessage());
                    }
                }
                case "DELETE" -> {
                    if (!authorize(exchange, manager)) {
                        return;
                    }
                    if (action.isBlank()) {
                        HttpSupport.sendError(exchange, 400, "INVALID_REQUEST", "Missing model identifier");
                        return;
                    }
                    String modelId = URLDecoder.decode(action, StandardCharsets.UTF_8);
                    try {
                        models.removeModel(providerId, modelId);
                        JsonObject response = new JsonObject();
                        response.put("provider", providerId);
                        response.put("model", modelId);
                        response.put("removed", true);
                        HttpSupport.sendJson(exchange, 200, response);
                    } catch (IllegalArgumentException e) {
                        HttpSupport.sendError(exchange,
                                e.getMessage() != null && e.getMessage().contains("not installed")
                                        ? 404 : 400,
                                "REMOVE_FAILED", e.getMessage());
                    } catch (IllegalStateException e) {
                        HttpSupport.sendError(exchange, 409, "REMOVE_UNSUPPORTED", e.getMessage());
                    }
                }
                default -> HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "GET, POST or DELETE");
            }
        };
    }

    /** GET /api/v1/models/downloads: all model download jobs (newest first). */
    static HttpHandler modelDownloads(EnrManager manager) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only GET allowed");
                return;
            }
            HttpSupport.sendJson(exchange, 200, manager.modelManagement().downloadsToJson());
        };
    }

    /** Routes /api/v1/models/downloads/{jobId}: GET status, DELETE cancel. */
    static HttpHandler modelDownloadRoute(EnrManager manager) {
        return exchange -> {
            String jobId = URLDecoder.decode(pathId(exchange, "/api/v1/models/downloads/"),
                    StandardCharsets.UTF_8);
            var models = manager.modelManagement();
            switch (exchange.getRequestMethod().toUpperCase()) {
                case "GET" -> {
                    var status = models.getDownloadStatus(jobId);
                    if (status.isEmpty()) {
                        HttpSupport.sendError(exchange, 404, "DOWNLOAD_NOT_FOUND",
                                "Unknown download job: " + jobId);
                        return;
                    }
                    HttpSupport.sendJson(exchange, 200, status.get().toJson());
                }
                case "DELETE" -> {
                    if (!authorize(exchange, manager)) {
                        return;
                    }
                    boolean cancelled = models.cancelDownload(jobId);
                    JsonObject response = new JsonObject();
                    response.put("job_id", jobId);
                    response.put("cancelled", cancelled);
                    HttpSupport.sendJson(exchange, cancelled ? 200 : 404, response);
                }
                default -> HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "GET or DELETE");
            }
        };
    }

    // ------------------------------------------------------------ shared

    private static boolean authorize(HttpExchange exchange, EnrManager manager) throws IOException {
        if (!HttpSupport.authorized(exchange, manager.config().authToken())) {
            HttpSupport.sendUnauthorized(exchange);
            return false;
        }
        return true;
    }

    private static List<ChatMessage> messagesFrom(JsonObject request) {
        List<ChatMessage> messages = new ArrayList<>();
        var arr = request.getArray("messages");
        if (arr == null) {
            return messages;
        }
        for (int i = 0; i < arr.size(); i++) {
            JsonObject entry = arr.getObject(i);
            if (entry == null) {
                continue;
            }
            messages.add(ChatMessage.of(entry.getString("role", "user"),
                    entry.getString("content", "")));
        }
        return messages;
    }

    private static EnrManager.ChatOptions optionsFrom(JsonObject request, boolean stream) {
        var defaults = EnrManager.ChatOptions.defaults();
        Float temperature = request.has("temperature") ? request.getFloat("temperature", 0.7f) : null;
        Float topP = request.has("top_p") ? request.getFloat("top_p", 0.9f) : null;
        Integer topK = request.has("top_k") ? request.getInt("top_k", 40) : null;
        Integer maxTokens = request.has("max_tokens") ? request.getInt("max_tokens", 256) : null;
        Long seed = request.has("seed") ? request.getLong("seed", 0) : null;
        List<String> stop = new ArrayList<>();
        var stopArr = request.getArray("stop");
        if (stopArr != null) {
            for (int i = 0; i < stopArr.size(); i++) {
                stop.add(stopArr.getString(i));
            }
        }
        return new EnrManager.ChatOptions(
                maxTokens == null ? defaults.maxTokens() : maxTokens,
                temperature, topP, topK, seed, stop, stream, request.getArray("tools"));
    }

    private static String queryParam(HttpExchange exchange, String name, String defaultValue) {
        String query = exchange.getRequestURI().getQuery();
        if (query == null) {
            return defaultValue;
        }
        for (String pair : query.split("&")) {
            int eq = pair.indexOf('=');
            if (eq < 0) {
                continue;
            }
            String key = URLDecoder.decode(pair.substring(0, eq), StandardCharsets.UTF_8);
            if (name.equals(key)) {
                return URLDecoder.decode(pair.substring(eq + 1), StandardCharsets.UTF_8);
            }
        }
        return defaultValue;
    }

    private static int intQueryParam(HttpExchange exchange, String name, int defaultValue) {
        try {
            return Integer.parseInt(queryParam(exchange, name, String.valueOf(defaultValue)));
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    private static String pathId(HttpExchange exchange, String prefix) {
        String path = exchange.getRequestURI().getPath();
        return path.length() > prefix.length() ? path.substring(prefix.length()) : "";
    }
}
