package ai.enr.server;

import ai.enr.core.json.Json;
import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonObject;
import ai.enr.manager.EnrManager;
import ai.enr.manager.provider.ChatMessage;
import ai.enr.manager.provider.Provider;
import ai.enr.manager.provider.ProviderClient;
import ai.enr.manager.router.ModelRouter;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * OpenAI-compatible API surface. Endpoints: /v1/models, /v1/chat/completions,
 * /v1/completions, /v1/embeddings. Streaming uses OpenAI's SSE chunk format
 * terminated by {@code data: [DONE]}.
 */
final class OpenAiApiHandlers {

    private OpenAiApiHandlers() {
    }

    static HttpHandler models(EnrManager manager) {
        return exchange -> {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only GET allowed");
                return;
            }
            JsonObject body = new JsonObject();
            body.put("object", "list");
            JsonArray data = new JsonArray();
            long created = System.currentTimeMillis() / 1000;
            for (String name : manager.router().routableNames()) {
                JsonObject entry = new JsonObject();
                entry.put("id", name);
                entry.put("object", "model");
                entry.put("created", created);
                entry.put("owned_by", "enr");
                data.add(entry);
            }
            body.put("data", data);
            HttpSupport.sendJson(exchange, 200, body);
        };
    }

    static HttpHandler chatCompletions(EnrManager manager) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only POST allowed");
                return;
            }
            try {
                JsonObject request = HttpSupport.parseBodyObject(exchange);
                String model = request.getString("model", "");
                boolean stream = request.getBoolean("stream", false);
                List<ChatMessage> messages = messagesFrom(request);
                EnrManager.ChatOptions options = chatOptionsFrom(request, stream);
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
                    HttpSupport.sendJson(exchange, 200, completionResponse(model, result));
                    return;
                }
                exchange.getResponseHeaders().set("Content-Type", "text/event-stream; charset=utf-8");
                exchange.getResponseHeaders().set("Cache-Control", "no-cache");
                exchange.sendResponseHeaders(200, 0);
                try (OutputStream out = exchange.getResponseBody()) {
                    String id = "chatcmpl-" + UUID.randomUUID();
                    long created = System.currentTimeMillis() / 1000;
                    final int[] promptTokens = {0};
                    final int[] completionTokens = {0};
                    final String[] finishReason = {"stop"};
                    manager.chatStream(model, messages, options, chunk -> {
                        try {
                            JsonObject line = new JsonObject();
                            line.put("id", id);
                            line.put("object", "chat.completion.chunk");
                            line.put("created", created);
                            line.put("model", model);
                            JsonArray choices = new JsonArray();
                            JsonObject choice = new JsonObject();
                            choice.put("index", 0);
                            JsonObject delta = new JsonObject();
                            if (chunk.getBoolean("done", false)) {
                                delta.put("content", "");
                                String finish = chunk.getString("finish_reason", "stop");
                                if (finish.isBlank()) {
                                    finish = "stop";
                                }
                                choice.put("finish_reason", finish);
                                finishReason[0] = finish;
                                promptTokens[0] = chunk.getInt("prompt_tokens", 0);
                                completionTokens[0] = chunk.getInt("completion_tokens", 0);
                            } else {
                                delta.put("content", chunk.getString("content", ""));
                                choice.put("finish_reason", (String) null);
                            }
                            choice.put("delta", delta);
                            choices.add(choice);
                            line.put("choices", choices);
                            if (chunk.getBoolean("done", false)) {
                                JsonObject usage = new JsonObject();
                                usage.put("prompt_tokens", promptTokens[0]);
                                usage.put("completion_tokens", completionTokens[0]);
                                usage.put("total_tokens", promptTokens[0] + completionTokens[0]);
                                line.put("usage", usage);
                            }
                            HttpSupport.sendSseData(out, line);
                            if (chunk.getBoolean("done", false)) {
                                out.write("data: [DONE]\n\n".getBytes(java.nio.charset.StandardCharsets.UTF_8));
                                out.flush();
                            }
                        } catch (IOException e) {
                            throw new RuntimeException("client disconnected", e);
                        }
                    });
                } catch (RuntimeException ignored) {
                    // client disconnected
                }
            } catch (IllegalArgumentException e) {
                HttpSupport.sendError(exchange, 404, "MODEL_NOT_FOUND", e.getMessage());
            } catch (Exception e) {
                HttpSupport.sendError(exchange, 500, "INFERENCE_FAILED", e.getMessage());
            }
        };
    }

    static HttpHandler completions(EnrManager manager) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only POST allowed");
                return;
            }
            try {
                JsonObject request = HttpSupport.parseBodyObject(exchange);
                String model = request.getString("model", "");
                String prompt = request.getString("prompt", "");
                if (model.isBlank()) {
                    HttpSupport.sendError(exchange, 400, "INVALID_REQUEST", "Missing 'model' field");
                    return;
                }
                List<ChatMessage> messages = List.of(ChatMessage.of("user", prompt));
                JsonObject result = manager.chat(model, messages,
                        chatOptionsFrom(request, false));
                if (result.has("error")) {
                    HttpSupport.sendError(exchange, 500, "INFERENCE_FAILED",
                            result.getString("error", "inference failed"));
                    return;
                }
                JsonObject response = new JsonObject();
                response.put("id", "cmpl-" + UUID.randomUUID());
                response.put("object", "text_completion");
                response.put("created", System.currentTimeMillis() / 1000);
                response.put("model", model);
                JsonArray choices = new JsonArray();
                JsonObject choice = new JsonObject();
                choice.put("text", result.getString("content", ""));
                choice.put("index", 0);
                choice.put("finish_reason", result.getString("finish_reason", "stop"));
                choices.add(choice);
                response.put("choices", choices);
                JsonObject usage = new JsonObject();
                usage.put("prompt_tokens", result.getInt("prompt_tokens", 0));
                usage.put("completion_tokens", result.getInt("completion_tokens", 0));
                usage.put("total_tokens", result.getInt("prompt_tokens", 0)
                        + result.getInt("completion_tokens", 0));
                response.put("usage", usage);
                HttpSupport.sendJson(exchange, 200, response);
            } catch (IllegalArgumentException e) {
                HttpSupport.sendError(exchange, 404, "MODEL_NOT_FOUND", e.getMessage());
            } catch (Exception e) {
                HttpSupport.sendError(exchange, 500, "INFERENCE_FAILED", e.getMessage());
            }
        };
    }

    /**
     * Embeddings are only supported through providers that implement the
     * OpenAI embeddings wire format. Native ENR models do not expose hidden
     * states, so they return an explicit 501 — documented limitation.
     */
    static HttpHandler embeddings(EnrManager manager) {
        return exchange -> {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only POST allowed");
                return;
            }
            JsonObject request = HttpSupport.parseBodyObject(exchange);
            String model = request.getString("model", "");
            try {
                ModelRouter.Route route = manager.router().resolve(model);
                if (route.target() == ModelRouter.Route.Target.LOCAL) {
                    HttpSupport.sendError(exchange, 501, "UNSUPPORTED",
                            "Embeddings are not supported for local ENR models (the native runtime "
                                    + "exposes logits only). Use a cloud or OpenAI-compatible provider.");
                    return;
                }
                Provider provider = route.provider();
                if (provider.type() == ai.enr.manager.provider.ProviderType.ANTHROPIC
                        || provider.type() == ai.enr.manager.provider.ProviderType.GEMINI
                        || provider.type() == ai.enr.manager.provider.ProviderType.OLLAMA
                        || provider.type() == ai.enr.manager.provider.ProviderType.LLAMACPP
                        || provider.type() == ai.enr.manager.provider.ProviderType.NATIVE) {
                    HttpSupport.sendError(exchange, 501, "UNSUPPORTED",
                            "Provider '" + provider.id() + "' (" + provider.type().id()
                                    + ") does not support embeddings");
                    return;
                }
                JsonObject forwarded = new JsonObject();
                forwarded.put("model", route.modelId());
                forwarded.put("input", request.get("input"));
                byte[] body = sendProviderEmbeddings(provider, forwarded);
                if (body == null) {
                    HttpSupport.sendError(exchange, 502, "PROVIDER_ERROR", "Provider rejected the request");
                    return;
                }
                byte[] bytes = Json.write(Json.parse(new String(body, java.nio.charset.StandardCharsets.UTF_8)), false)
                        .getBytes(java.nio.charset.StandardCharsets.UTF_8);
                exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
                exchange.sendResponseHeaders(200, bytes.length);
                try (OutputStream out = exchange.getResponseBody()) {
                    out.write(bytes);
                }
            } catch (IllegalArgumentException e) {
                HttpSupport.sendError(exchange, 404, "MODEL_NOT_FOUND", e.getMessage());
            } catch (Exception e) {
                HttpSupport.sendError(exchange, 500, "INFERENCE_FAILED", e.getMessage());
            }
        };
    }

    private static byte[] sendProviderEmbeddings(Provider provider, JsonObject body) throws Exception {
        try (ProviderClient client = new ProviderClient()) {
            return client.embeddings(provider, body);
        }
    }

    private static JsonObject completionResponse(String model, JsonObject result) {
        JsonObject response = new JsonObject();
        response.put("id", "chatcmpl-" + UUID.randomUUID());
        response.put("object", "chat.completion");
        response.put("created", System.currentTimeMillis() / 1000);
        response.put("model", model);
        JsonArray choices = new JsonArray();
        JsonObject choice = new JsonObject();
        choice.put("index", 0);
        JsonObject message = new JsonObject();
        message.put("role", "assistant");
        message.put("content", result.getString("content", ""));
        choice.put("message", message);
        choice.put("finish_reason", result.getString("finish_reason", "stop"));
        choices.add(choice);
        response.put("choices", choices);
        JsonObject usage = new JsonObject();
        usage.put("prompt_tokens", result.getInt("prompt_tokens", 0));
        usage.put("completion_tokens", result.getInt("completion_tokens", 0));
        usage.put("total_tokens", result.getInt("prompt_tokens", 0)
                + result.getInt("completion_tokens", 0));
        response.put("usage", usage);
        return response;
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
            if (content == null || content.isEmpty()) {
                JsonArray contentArr = entry.getArray("content");
                if (contentArr != null) {
                    StringBuilder text = new StringBuilder();
                    for (int j = 0; j < contentArr.size(); j++) {
                        JsonObject part = contentArr.getObject(j);
                        if (part != null && "text".equals(part.getString("type"))) {
                            text.append(part.getString("text", ""));
                        }
                    }
                    content = text.toString();
                }
            }
            messages.add(ChatMessage.of(entry.getString("role", "user"), content));
        }
        return messages;
    }

    private static EnrManager.ChatOptions chatOptionsFrom(JsonObject request, boolean stream) {
        EnrManager.ChatOptions defaults = EnrManager.ChatOptions.defaults();
        Float temperature = request.has("temperature") ? request.getFloat("temperature", 0.7f) : null;
        Float topP = request.has("top_p") ? request.getFloat("top_p", 0.9f) : null;
        Integer topK = request.has("top_k") ? request.getInt("top_k", 40) : null;
        Integer maxTokens = request.has("max_tokens") ? request.getInt("max_tokens", 256) : null;
        if (maxTokens == null && request.has("max_completion_tokens")) {
            maxTokens = request.getInt("max_completion_tokens", 256);
        }
        Long seed = request.has("seed") ? request.getLong("seed", 0) : null;
        List<String> stop = new ArrayList<>();
        JsonArray stopArr = request.getArray("stop");
        if (stopArr != null) {
            for (int i = 0; i < stopArr.size(); i++) {
                stop.add(stopArr.getString(i));
            }
        }
        return new EnrManager.ChatOptions(
                maxTokens == null ? defaults.maxTokens() : maxTokens,
                temperature, topP, topK, seed, stop, stream);
    }
}
