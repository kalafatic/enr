package ai.enr.manager.provider;

import ai.enr.core.json.Json;
import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonObject;
import ai.enr.core.json.JsonString;
import ai.enr.core.json.JsonValue;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * Wire-protocol client for all non-native provider types: OpenAI-compatible
 * chat (OpenAI, DeepSeek, OpenAI-compatible servers, Ollama /v1, llama.cpp
 * /v1), Anthropic Messages and Google Gemini generateContent. API keys are
 * sent as headers only and never appear in messages, errors or logs.
 */
public final class ProviderClient implements AutoCloseable {

    public record ChatOptions(int maxTokens, Float temperature, Float topP, Integer topK,
                              Long seed, List<String> stop, boolean stream, JsonArray tools) {

        public static ChatOptions defaults() {
            return new ChatOptions(256, null, null, null, null, List.of(), false, null);
        }
    }

    public record ChatResult(String content, int promptTokens, int completionTokens,
                             String finishReason, long elapsedMs, boolean ok, String error,
                             JsonValue toolCalls, String reasoningContent, int status) {

        public static ChatResult error(String message) {
            return error(502, message);
        }

        public static ChatResult error(int status, String message) {
            return new ChatResult("", 0, 0, "error", 0, false, message, null, null, status);
        }

        /**
         * Maps an upstream provider HTTP status to the client-facing status ENR
         * should return. Preserves meaningful status codes (rate limiting,
         * gateway timeout, overload) while translating provider-side auth
         * failures and other upstream errors into 502 Bad Gateway so the
         * harness does not mistake them for its own request errors.
         */
        public static int gatewayStatus(int upstream) {
            return switch (upstream) {
                case 429 -> 429;                 // rate limit: propagate for backoff
                case 408 -> 504;                 // upstream timeout -> gateway timeout
                case 502, 503, 504 -> upstream;  // upstream gateway/overload codes
                default -> 502;                  // auth (401/403) and other errors -> bad gateway
            };
        }
    }

    public record StreamChunk(String content, boolean done, String finishReason,
                              int promptTokens, int completionTokens, JsonValue toolCalls,
                              String reasoningContent) {
    }

    public record TestResult(boolean ok, String message, long elapsedMs, List<String> models) {
    }

    private final HttpClient http;

    public ProviderClient() {
        this(HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(15))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build());
    }

    public ProviderClient(HttpClient http) {
        this.http = http;
    }

    /** Lists remote model names exposed by the provider. */
    public List<String> listModels(Provider provider) throws Exception {
        String base = normalizeBase(provider.baseUrl());
        return switch (provider.type()) {
            case NATIVE -> List.of();
            case OLLAMA -> listOllamaModels(base);
            case ANTHROPIC -> listAnthropicModels(base, provider.apiKey());
            case GEMINI -> listGeminiModels(base, provider.apiKey());
            case DEEPSEEK, OPENAI, OPENAI_COMPAT, LLAMACPP -> listOpenAiModels(base, provider.apiKey());
        };
    }

    public ChatResult chat(Provider provider, String model, List<ChatMessage> messages,
                           ChatOptions options) {
        long started = System.currentTimeMillis();
        try {
            return switch (provider.type()) {
                case NATIVE -> ChatResult.error("Native provider is handled by the ENR runtime");
                case ANTHROPIC -> anthropicChat(provider, model, messages, options);
                case GEMINI -> geminiChat(provider, model, messages, options);
                case DEEPSEEK, OPENAI, OPENAI_COMPAT, OLLAMA, LLAMACPP ->
                        openAiChat(provider, model, messages, options);
            };
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return ChatResult.error("Interrupted");
        } catch (Exception e) {
            String message = e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
            return ChatResult.error("Provider request failed (" + (System.currentTimeMillis() - started)
                    + " ms): " + message);
        }
    }

    public void chatStream(Provider provider, String model, List<ChatMessage> messages,
                           ChatOptions options, Consumer<StreamChunk> onChunk) throws Exception {
        switch (provider.type()) {
            case NATIVE -> throw new IllegalArgumentException("Native provider is handled by the ENR runtime");
            case ANTHROPIC -> anthropicStream(provider, model, messages, options, onChunk);
            case GEMINI -> geminiStream(provider, model, messages, options, onChunk);
            case DEEPSEEK, OPENAI, OPENAI_COMPAT, OLLAMA, LLAMACPP ->
                    openAiStream(provider, model, messages, options, onChunk);
        }
    }

    /**
     * Forwards an embeddings request to an OpenAI-compatible provider and
     * returns the raw response body (or null when the provider rejects it).
     */
    public byte[] embeddings(Provider provider, JsonObject body) throws Exception {
        String base = normalizeBase(provider.baseUrl());
        HttpResponse<String> response = send(auth(provider, HttpRequest.newBuilder()
                        .uri(URI.create(base + "/embeddings"))
                        .POST(HttpRequest.BodyPublishers.ofString(Json.write(body, false))))
                .header("Content-Type", "application/json")
                .build());
        if (response.statusCode() != 200) {
            return null;
        }
        return response.body().getBytes(StandardCharsets.UTF_8);
    }

    // ---------------------------------------------------------------- OpenAI-compatible

    private ChatResult openAiChat(Provider provider, String model, List<ChatMessage> messages,
                                  ChatOptions options) throws Exception {
        JsonObject body = openAiBody(model, messages, options, false);
        HttpResponse<String> response = send(auth(provider, HttpRequest.newBuilder()
                        .uri(URI.create(normalizeBase(provider.baseUrl()) + "/chat/completions"))
                        .POST(HttpRequest.BodyPublishers.ofString(Json.write(body, false))))
                .header("Content-Type", "application/json")
                .build());
        if (response.statusCode() != 200) {
            return ChatResult.error(ChatResult.gatewayStatus(response.statusCode()),
                    "HTTP " + response.statusCode() + ": " + shortBody(response.body()));
        }
        JsonObject obj = Json.parseObject(response.body());
        JsonArray choices = obj.getArray("choices");
        String content = "";
        String finish = "stop";
        JsonValue toolCalls = null;
        String reasoning = null;
        if (choices != null && choices.size() > 0) {
            JsonObject first = choices.getObject(0);
            JsonObject message = first.getObject("message");
            content = message == null ? "" : message.getString("content", "");
            if (message != null) {
                toolCalls = parsedToolCalls(message.get("tool_calls"));
                if (message.get("reasoning_content") != null) {
                    reasoning = message.getString("reasoning_content", "");
                }
            }
            finish = first.getString("finish_reason", "stop");
        }
        JsonObject usage = obj.getObject("usage");
        return new ChatResult(content,
                usage == null ? 0 : usage.getInt("prompt_tokens", 0),
                usage == null ? 0 : usage.getInt("completion_tokens", 0),
                finish, System.currentTimeMillis(), true, "", toolCalls, reasoning, 200);
    }

    private void openAiStream(Provider provider, String model, List<ChatMessage> messages,
                              ChatOptions options, Consumer<StreamChunk> onChunk) throws Exception {
        JsonObject body = openAiBody(model, messages, options, true);
        HttpResponse<InputStream> response = http.send(auth(provider, HttpRequest.newBuilder()
                        .uri(URI.create(normalizeBase(provider.baseUrl()) + "/chat/completions"))
                        .POST(HttpRequest.BodyPublishers.ofString(Json.write(body, false))))
                .header("Content-Type", "application/json")
                .header("Accept", "text/event-stream")
                .build(), HttpResponse.BodyHandlers.ofInputStream());
        if (response.statusCode() != 200) {
            String err = new String(response.body().readAllBytes(), StandardCharsets.UTF_8);
            throw new IllegalStateException("HTTP " + response.statusCode() + ": " + shortBody(err));
        }
        java.util.Map<Integer, String> callIds = new java.util.LinkedHashMap<>();
        java.util.Map<Integer, StringBuilder> callNames = new java.util.LinkedHashMap<>();
        java.util.Map<Integer, StringBuilder> callArguments = new java.util.LinkedHashMap<>();
        StringBuilder reasoning = new StringBuilder();
        int[] usageTokens = {0, 0};
        boolean[] finished = {false};
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(response.body(),
                StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.startsWith("data:")) {
                    continue;
                }
                String payload = line.substring(5).strip();
                if (payload.isEmpty() || "[DONE]".equals(payload)) {
                    continue;
                }
                JsonObject chunk = Json.parseObject(payload);
                JsonArray choices = chunk.getArray("choices");
                if (choices == null || choices.size() == 0) {
                    JsonObject usage = chunk.getObject("usage");
                    if (usage != null) {
                        usageTokens[0] = usage.getInt("prompt_tokens", 0);
                        usageTokens[1] = usage.getInt("completion_tokens", 0);
                        if (!finished[0]) {
                            onChunk.accept(new StreamChunk("", true, "stop", usageTokens[0],
                                    usageTokens[1], aggregateToolCalls(callIds, callNames, callArguments),
                                    reasoning.length() > 0 ? reasoning.toString() : null));
                            finished[0] = true;
                        }
                    }
                    continue;
                }
                if (finished[0]) {
                    continue;
                }
                JsonObject first = choices.getObject(0);
                JsonObject delta = first == null ? null : first.getObject("delta");
                if (delta != null) {
                    String reasoningDelta = delta.getString("reasoning_content");
                    if (reasoningDelta != null && !reasoningDelta.isEmpty()) {
                        reasoning.append(reasoningDelta);
                    }
                    JsonArray toolDeltas = delta.getArray("tool_calls");
                    if (toolDeltas != null) {
                        for (int i = 0; i < toolDeltas.size(); i++) {
                            JsonObject toolDelta = toolDeltas.getObject(i);
                            if (toolDelta == null) {
                                continue;
                            }
                            int index = toolDelta.getInt("index", 0);
                            String id = toolDelta.getString("id");
                            if (id != null && !id.isBlank()) {
                                callIds.put(index, id);
                            }
                            JsonObject function = toolDelta.getObject("function");
                            if (function != null) {
                                String name = function.getString("name");
                                String arguments = function.getString("arguments");
                                if (name != null && !name.isEmpty()) {
                                    callNames.computeIfAbsent(index, k -> new StringBuilder()).append(name);
                                }
                                if (arguments != null && !arguments.isEmpty()) {
                                    callArguments.computeIfAbsent(index, k -> new StringBuilder())
                                            .append(arguments);
                                }
                            }
                        }
                    }
                }
                String piece = delta == null ? "" : delta.getString("content", "");
                String finish = first == null ? null : first.getString("finish_reason");
                if (finish != null && !finish.isBlank()) {
                    JsonValue toolCalls = aggregateToolCalls(callIds, callNames, callArguments);
                    String effectiveFinish = toolCalls != null ? "tool_calls" : finish;
                    onChunk.accept(new StreamChunk(piece, true, effectiveFinish, usageTokens[0],
                            usageTokens[1], toolCalls,
                            reasoning.length() > 0 ? reasoning.toString() : null));
                    finished[0] = true;
                } else if (piece != null && !piece.isEmpty()) {
                    onChunk.accept(new StreamChunk(piece, false, null, 0, 0, null, null));
                }
            }
        }
        if (!finished[0]) {
            JsonValue toolCalls = aggregateToolCalls(callIds, callNames, callArguments);
            String finish = toolCalls != null ? "tool_calls" : "stop";
            onChunk.accept(new StreamChunk("", true, finish, usageTokens[0], usageTokens[1], toolCalls,
                    reasoning.length() > 0 ? reasoning.toString() : null));
        }
    }

    /**
     * Aggregates streamed tool-call deltas into a canonical tool_calls array:
     * {@code [{"id","type":"function","function":{"name","arguments":<object>}}]}.
     * Arguments are parsed from their JSON string form when possible; invalid
     * JSON is preserved as a string. Returns null when nothing accumulated.
     */
    private static JsonValue aggregateToolCalls(java.util.Map<Integer, String> ids,
                                                java.util.Map<Integer, StringBuilder> names,
                                                java.util.Map<Integer, StringBuilder> arguments) {
        if (names.isEmpty() && arguments.isEmpty()) {
            return null;
        }
        JsonArray toolCalls = new JsonArray();
        java.util.Set<Integer> indexes = new java.util.TreeSet<>();
        indexes.addAll(names.keySet());
        indexes.addAll(arguments.keySet());
        for (int index : indexes) {
            JsonObject call = new JsonObject();
            String id = ids.get(index);
            call.put("id", id == null ? "call_" + index : id);
            call.put("type", "function");
            JsonObject function = new JsonObject();
            function.put("name", names.containsKey(index) ? names.get(index).toString() : "");
            StringBuilder args = arguments.get(index);
            if (args != null) {
                function.put("arguments", parseArguments(args.toString()));
            } else {
                function.put("arguments", new JsonObject());
            }
            call.put("function", function);
            toolCalls.add(call);
        }
        return toolCalls;
    }

    /**
     * Serializes canonical tool calls into the OpenAI request wire format:
     * {@code arguments} must be a JSON string (OpenAI contract), even when the
     * canonical form carries a parsed object (e.g. from Ollama-format input).
     */
    public static JsonArray openAiWireToolCalls(JsonValue toolCalls) {
        JsonArray result = new JsonArray();
        if (!(toolCalls instanceof JsonArray arr)) {
            return result;
        }
        for (int i = 0; i < arr.size(); i++) {
            JsonObject entry = arr.getObject(i);
            if (entry == null) {
                continue;
            }
            JsonObject call = new JsonObject();
            String id = entry.getString("id");
            call.put("id", id == null || id.isBlank() ? "call_" + i : id);
            String type = entry.getString("type");
            call.put("type", type == null || type.isBlank() ? "function" : type);
            JsonObject function = entry.getObject("function");
            JsonObject normalized = new JsonObject();
            if (function != null) {
                normalized.put("name", function.getString("name", ""));
                JsonValue arguments = function.get("arguments");
                if (arguments instanceof JsonString s) {
                    normalized.put("arguments", s.value());
                } else if (arguments != null) {
                    normalized.put("arguments", Json.write(arguments, false));
                } else {
                    normalized.put("arguments", "{}");
                }
            } else {
                normalized.put("name", "");
                normalized.put("arguments", "{}");
            }
            call.put("function", normalized);
            result.add(call);
        }
        return result;
    }

    /** Parses a JSON object string; falls back to the raw string when invalid. */
    public static JsonValue parseArguments(String text) {
        try {
            return Json.parse(text.strip());
        } catch (RuntimeException e) {
            return new ai.enr.core.json.JsonString(text);
        }
    }

    /** Normalizes provider tool_calls into the canonical form; returns null when absent. */
    static JsonValue parsedToolCalls(JsonValue toolCalls) {
        if (!(toolCalls instanceof JsonArray arr)) {
            return null;
        }
        JsonArray result = new JsonArray();
        for (int i = 0; i < arr.size(); i++) {
            JsonObject entry = arr.getObject(i);
            if (entry == null) {
                continue;
            }
            JsonObject call = new JsonObject();
            String id = entry.getString("id");
            call.put("id", id == null || id.isBlank() ? "call_" + i : id);
            String type = entry.getString("type");
            call.put("type", type == null || type.isBlank() ? "function" : type);
            JsonObject function = entry.getObject("function");
            JsonObject normalizedFunction = new JsonObject();
            if (function != null) {
                normalizedFunction.put("name", function.getString("name", ""));
                JsonValue arguments = function.get("arguments");
                if (arguments instanceof JsonString s) {
                    normalizedFunction.put("arguments", parseArguments(s.value()));
                } else if (arguments != null) {
                    normalizedFunction.put("arguments", arguments);
                } else {
                    normalizedFunction.put("arguments", new JsonObject());
                }
            } else {
                normalizedFunction.put("name", "");
                normalizedFunction.put("arguments", new JsonObject());
            }
            call.put("function", normalizedFunction);
            result.add(call);
        }
        return result.size() > 0 ? result : null;
    }

    private JsonObject openAiBody(String model, List<ChatMessage> messages, ChatOptions options,
                                  boolean stream) {
        JsonObject body = new JsonObject();
        body.put("model", model);
        JsonArray arr = new JsonArray();
        java.util.ArrayDeque<String> pendingToolIds = new java.util.ArrayDeque<>();
        for (ChatMessage message : messages) {
            JsonObject entry = new JsonObject();
            entry.put("role", message.role());
            JsonObject payload = message.payload();
            if ("assistant".equals(message.role()) && payload != null
                    && payload.get("tool_calls") != null) {
                JsonArray wire = openAiWireToolCalls(payload.get("tool_calls"));
                pendingToolIds.clear();
                for (int i = 0; i < wire.size(); i++) {
                    pendingToolIds.add(wire.getObject(i).getString("id", ""));
                }
                entry.put("tool_calls", wire);
                if (payload.get("reasoning_content") != null) {
                    entry.put("reasoning_content", payload.get("reasoning_content"));
                }
            } else if ("assistant".equals(message.role()) && payload != null
                    && payload.get("reasoning_content") != null) {
                entry.put("reasoning_content", payload.get("reasoning_content"));
            } else if ("tool".equals(message.role())) {
                JsonValue toolCallId = payload == null ? null : payload.get("tool_call_id");
                if (toolCallId != null) {
                    entry.put("tool_call_id", toolCallId);
                } else if (!pendingToolIds.isEmpty()) {
                    entry.put("tool_call_id", pendingToolIds.poll());
                }
            } else if (payload != null) {
                JsonValue toolCallId = payload.get("tool_call_id");
                if (toolCallId != null) {
                    entry.put("tool_call_id", toolCallId);
                }
            }
            entry.put("content", message.content());
            arr.add(entry);
        }
        body.put("messages", arr);
        body.put("stream", stream);
        if (stream) {
            JsonObject streamOptions = new JsonObject();
            streamOptions.put("include_usage", true);
            body.put("stream_options", streamOptions);
        }
        if (options.tools() != null && options.tools().size() > 0) {
            body.put("tools", options.tools());
        }
        if (options.maxTokens() > 0) {
            body.put("max_tokens", options.maxTokens());
        }
        if (options.temperature() != null) {
            body.put("temperature", options.temperature());
        }
        if (options.topP() != null) {
            body.put("top_p", options.topP());
        }
        if (options.seed() != null) {
            body.put("seed", options.seed());
        }
        if (options.stop() != null && !options.stop().isEmpty()) {
            JsonArray stop = new JsonArray();
            for (String s : options.stop()) {
                stop.add(s);
            }
            body.put("stop", stop);
        }
        return body;
    }

    private List<String> listOpenAiModels(String base, String apiKey) throws Exception {
        HttpResponse<String> response = send(auth(apiKey, HttpRequest.newBuilder()
                        .uri(URI.create(base + "/models"))
                        .GET())
                .header("Accept", "application/json")
                .build());
        if (response.statusCode() != 200) {
            throw new IllegalStateException("HTTP " + response.statusCode());
        }
        return modelIds(Json.parseObject(response.body()));
    }

    private static List<String> modelIds(JsonObject obj) {
        List<String> ids = new ArrayList<>();
        JsonArray data = obj.getArray("data");
        if (data != null) {
            for (int i = 0; i < data.size(); i++) {
                JsonObject entry = data.getObject(i);
                String id = entry == null ? null : entry.getString("id");
                if (id != null) {
                    ids.add(id);
                }
            }
        }
        return ids;
    }

    private List<String> listOllamaModels(String base) throws Exception {
        HttpResponse<String> response = send(HttpRequest.newBuilder()
                .uri(URI.create(base + "/tags"))
                .GET()
                .build());
        if (response.statusCode() != 200) {
            throw new IllegalStateException("HTTP " + response.statusCode());
        }
        List<String> ids = new ArrayList<>();
        JsonObject obj = Json.parseObject(response.body());
        JsonArray models = obj.getArray("models");
        if (models != null) {
            for (int i = 0; i < models.size(); i++) {
                JsonObject entry = models.getObject(i);
                if (entry != null) {
                    String name = entry.getString("name");
                    if (name != null) {
                        ids.add(name);
                    }
                }
            }
        }
        return ids;
    }

    // ---------------------------------------------------------------- Anthropic

    private ChatResult anthropicChat(Provider provider, String model, List<ChatMessage> messages,
                                     ChatOptions options) throws Exception {
        JsonObject body = anthropicBody(model, messages, options, false);
        HttpResponse<String> response = send(anthropicAuth(provider, HttpRequest.newBuilder()
                        .uri(URI.create(normalizeBase(provider.baseUrl()) + "/v1/messages"))
                        .POST(HttpRequest.BodyPublishers.ofString(Json.write(body, false))))
                .header("Content-Type", "application/json")
                .build());
        if (response.statusCode() != 200) {
            return ChatResult.error(ChatResult.gatewayStatus(response.statusCode()),
                    "HTTP " + response.statusCode() + ": " + shortBody(response.body()));
        }
        JsonObject obj = Json.parseObject(response.body());
        StringBuilder content = new StringBuilder();
        JsonArray blocks = obj.getArray("content");
        if (blocks != null) {
            for (int i = 0; i < blocks.size(); i++) {
                JsonObject block = blocks.getObject(i);
                if (block != null && "text".equals(block.getString("type"))) {
                    content.append(block.getString("text", ""));
                }
            }
        }
        JsonObject usage = obj.getObject("usage");
        return new ChatResult(content.toString(),
                usage == null ? 0 : usage.getInt("input_tokens", 0),
                usage == null ? 0 : usage.getInt("output_tokens", 0),
                obj.getString("stop_reason", "end_turn"), System.currentTimeMillis(), true, "", null, null, 200);
    }

    private void anthropicStream(Provider provider, String model, List<ChatMessage> messages,
                                 ChatOptions options, Consumer<StreamChunk> onChunk) throws Exception {
        JsonObject body = anthropicBody(model, messages, options, true);
        HttpResponse<InputStream> response = http.send(anthropicAuth(provider, HttpRequest.newBuilder()
                        .uri(URI.create(normalizeBase(provider.baseUrl()) + "/v1/messages"))
                        .POST(HttpRequest.BodyPublishers.ofString(Json.write(body, false))))
                .header("Content-Type", "application/json")
                .header("Accept", "text/event-stream")
                .build(), HttpResponse.BodyHandlers.ofInputStream());
        if (response.statusCode() != 200) {
            String err = new String(response.body().readAllBytes(), StandardCharsets.UTF_8);
            throw new IllegalStateException("HTTP " + response.statusCode() + ": " + shortBody(err));
        }
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(response.body(),
                StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.startsWith("data:")) {
                    continue;
                }
                String payload = line.substring(5).strip();
                if (payload.isBlank()) {
                    continue;
                }
                JsonObject event = Json.parseObject(payload);
                String type = event.getString("type", "");
                switch (type) {
                    case "content_block_delta" -> {
                        JsonObject delta = event.getObject("delta");
                        if (delta != null) {
                            String text = delta.getString("text", "");
                            if (!text.isEmpty()) {
                                onChunk.accept(new StreamChunk(text, false, null, 0, 0, null, null));
                            }
                        }
                    }
                    case "message_delta" -> {
                        JsonObject delta = event.getObject("delta");
                        String stop = delta == null ? "end_turn" : delta.getString("stop_reason", "end_turn");
                        JsonObject usage = event.getObject("usage");
                        onChunk.accept(new StreamChunk("", true, stop,
                                usage == null ? 0 : usage.getInt("input_tokens", 0),
                                usage == null ? 0 : usage.getInt("output_tokens", 0), null, null));
                    }
                    case "message_stop" -> onChunk.accept(new StreamChunk("", true, "end_turn", 0, 0, null, null));
                    case "error" -> throw new IllegalStateException(
                            event.getObject("error") == null ? "stream error"
                                    : event.getObject("error").getString("message", "stream error"));
                    default -> {
                        // ping / start events carry no content
                    }
                }
            }
        }
    }

    private JsonObject anthropicBody(String model, List<ChatMessage> messages, ChatOptions options,
                                     boolean stream) {
        JsonObject body = new JsonObject();
        body.put("model", model);
        body.put("max_tokens", options.maxTokens() > 0 ? options.maxTokens() : 1024);
        if (options.temperature() != null) {
            body.put("temperature", options.temperature());
        }
        if (options.topP() != null) {
            body.put("top_p", options.topP());
        }
        JsonArray arr = new JsonArray();
        for (ChatMessage message : messages) {
            if ("system".equals(message.role())) {
                body.put("system", message.content());
                continue;
            }
            JsonObject entry = new JsonObject();
            entry.put("role", "assistant".equals(message.role()) ? "assistant" : "user");
            entry.put("content", message.content());
            arr.add(entry);
        }
        body.put("messages", arr);
        body.put("stream", stream);
        return body;
    }

    private List<String> listAnthropicModels(String base, String apiKey) throws Exception {
        HttpResponse<String> response = send(HttpRequest.newBuilder()
                .uri(URI.create(base + "/v1/models"))
                .header("x-api-key", apiKey == null ? "" : apiKey)
                .header("anthropic-version", "2023-06-01")
                .GET()
                .build());
        if (response.statusCode() != 200) {
            throw new IllegalStateException("HTTP " + response.statusCode());
        }
        return modelIds(Json.parseObject(response.body()));
    }

    private static HttpRequest.Builder anthropicAuth(Provider provider, HttpRequest.Builder builder) {
        return builder
                .header("x-api-key", provider.apiKey() == null ? "" : provider.apiKey())
                .header("anthropic-version", "2023-06-01");
    }

    // ---------------------------------------------------------------- Gemini

    private ChatResult geminiChat(Provider provider, String model, List<ChatMessage> messages,
                                  ChatOptions options) throws Exception {
        JsonObject body = geminiBody(messages, options);
        URI uri = geminiUri(provider, model);
        HttpResponse<String> response = send(geminiAuth(provider, HttpRequest.newBuilder()
                        .uri(uri)
                        .POST(HttpRequest.BodyPublishers.ofString(Json.write(body, false))))
                .header("Content-Type", "application/json")
                .build());
        if (response.statusCode() != 200) {
            return ChatResult.error(ChatResult.gatewayStatus(response.statusCode()),
                    "HTTP " + response.statusCode() + ": " + shortBody(response.body()));
        }
        JsonObject obj = Json.parseObject(response.body());
        JsonArray candidates = obj.getArray("candidates");
        String content = "";
        String finish = "STOP";
        if (candidates != null && candidates.size() > 0) {
            JsonObject candidate = candidates.getObject(0);
            finish = candidate == null ? "STOP" : candidate.getString("finishReason", "STOP");
            JsonObject candContent = candidate == null ? null : candidate.getObject("content");
            JsonArray parts = candContent == null ? null : candContent.getArray("parts");
            if (parts != null) {
                StringBuilder text = new StringBuilder();
                for (int i = 0; i < parts.size(); i++) {
                    JsonObject part = parts.getObject(i);
                    if (part != null && part.has("text")) {
                        text.append(part.getString("text", ""));
                    }
                }
                content = text.toString();
            }
        }
        JsonObject usage = obj.getObject("usageMetadata");
        return new ChatResult(content,
                usage == null ? 0 : usage.getInt("promptTokenCount", 0),
                usage == null ? 0 : usage.getInt("candidatesTokenCount", 0),
                finish, System.currentTimeMillis(), true, "", null, null, 200);
    }

    private void geminiStream(Provider provider, String model, List<ChatMessage> messages,
                              ChatOptions options, Consumer<StreamChunk> onChunk) throws Exception {
        JsonObject body = geminiBody(messages, options);
        URI uri = geminiUri(provider, model);
        HttpResponse<InputStream> response = http.send(geminiAuth(provider, HttpRequest.newBuilder()
                        .uri(uri)
                        .POST(HttpRequest.BodyPublishers.ofString(Json.write(body, false))))
                .header("Content-Type", "application/json")
                .header("Accept", "text/event-stream")
                .build(), HttpResponse.BodyHandlers.ofInputStream());
        if (response.statusCode() != 200) {
            String err = new String(response.body().readAllBytes(), StandardCharsets.UTF_8);
            throw new IllegalStateException("HTTP " + response.statusCode() + ": " + shortBody(err));
        }
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(response.body(),
                StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.startsWith("data:")) {
                    continue;
                }
                String payload = line.substring(5).strip();
                if (payload.isBlank()) {
                    continue;
                }
                JsonObject chunk = Json.parseObject(payload);
                JsonArray candidates = chunk.getArray("candidates");
                if (candidates == null || candidates.size() == 0) {
                    continue;
                }
                JsonObject candidate = candidates.getObject(0);
                JsonObject candContent = candidate == null ? null : candidate.getObject("content");
                JsonArray parts = candContent == null ? null : candContent.getArray("parts");
                if (parts != null) {
                    StringBuilder text = new StringBuilder();
                    for (int i = 0; i < parts.size(); i++) {
                        JsonObject part = parts.getObject(i);
                        if (part != null && part.has("text")) {
                            text.append(part.getString("text", ""));
                        }
                    }
                    if (text.length() > 0) {
                        onChunk.accept(new StreamChunk(text.toString(), false, null, 0, 0, null, null));
                    }
                }
                String finish = candidate == null ? null : candidate.getString("finishReason");
                if (finish != null && !finish.isBlank()) {
                    JsonObject usage = chunk.getObject("usageMetadata");
                    onChunk.accept(new StreamChunk("", true, finish,
                            usage == null ? 0 : usage.getInt("promptTokenCount", 0),
                            usage == null ? 0 : usage.getInt("candidatesTokenCount", 0), null, null));
                }
            }
        }
    }

    private JsonObject geminiBody(List<ChatMessage> messages, ChatOptions options) {
        JsonObject body = new JsonObject();
        JsonArray contents = new JsonArray();
        StringBuilder systemText = new StringBuilder();
        for (ChatMessage message : messages) {
            if ("system".equals(message.role())) {
                if (systemText.length() > 0) {
                    systemText.append('\n');
                }
                systemText.append(message.content());
                continue;
            }
            JsonObject content = new JsonObject();
            content.put("role", "assistant".equals(message.role()) ? "model" : "user");
            JsonArray parts = new JsonArray();
            JsonObject part = new JsonObject();
            part.put("text", message.content());
            parts.add(part);
            content.put("parts", parts);
            contents.add(content);
        }
        body.put("contents", contents);
        if (systemText.length() > 0) {
            JsonObject system = new JsonObject();
            JsonArray parts = new JsonArray();
            JsonObject part = new JsonObject();
            part.put("text", systemText.toString());
            parts.add(part);
            system.put("parts", parts);
            body.put("systemInstruction", system);
        }
        JsonObject generationConfig = new JsonObject();
        generationConfig.put("maxOutputTokens", options.maxTokens() > 0 ? options.maxTokens() : 1024);
        if (options.temperature() != null) {
            generationConfig.put("temperature", options.temperature());
        }
        if (options.topP() != null) {
            generationConfig.put("topP", options.topP());
        }
        body.put("generationConfig", generationConfig);
        return body;
    }

    private URI geminiUri(Provider provider, String model) {
        return URI.create(normalizeBase(provider.baseUrl()) + "/v1beta/models/"
                + java.net.URLEncoder.encode(model, StandardCharsets.UTF_8)
                + ":generateContent?alt=sse");
    }

    private static HttpRequest.Builder geminiAuth(Provider provider, HttpRequest.Builder builder) {
        return builder.header("x-goog-api-key", provider.apiKey() == null ? "" : provider.apiKey());
    }

    private List<String> listGeminiModels(String base, String apiKey) throws Exception {
        HttpResponse<String> response = send(geminiAuth(new Provider("x", ProviderType.GEMINI,
                                "x", base, apiKey, false), HttpRequest.newBuilder()
                        .uri(URI.create(base + "/v1beta/models"))
                        .GET())
                .build());
        if (response.statusCode() != 200) {
            throw new IllegalStateException("HTTP " + response.statusCode());
        }
        List<String> ids = new ArrayList<>();
        JsonArray models = Json.parseObject(response.body()).getArray("models");
        if (models != null) {
            for (int i = 0; i < models.size(); i++) {
                JsonObject entry = models.getObject(i);
                if (entry == null) {
                    continue;
                }
                String name = entry.getString("name", "");
                if (name.startsWith("models/")) {
                    ids.add(name.substring("models/".length()));
                }
            }
        }
        return ids;
    }

    // ---------------------------------------------------------------- shared

    private HttpResponse<String> send(HttpRequest request) throws Exception {
        return http.send(request, HttpResponse.BodyHandlers.ofString());
    }

    private static HttpRequest.Builder auth(Provider provider, HttpRequest.Builder builder) {
        return auth(provider.apiKey(), builder);
    }

    private static HttpRequest.Builder auth(String apiKey, HttpRequest.Builder builder) {
        if (apiKey != null && !apiKey.isBlank()) {
            builder.header("Authorization", "Bearer " + apiKey);
        }
        return builder;
    }

    private static String normalizeBase(String baseUrl) {
        String base = baseUrl == null ? "" : baseUrl.strip();
        while (base.endsWith("/")) {
            base = base.substring(0, base.length() - 1);
        }
        return base;
    }

    private static String shortBody(String body) {
        if (body == null) {
            return "";
        }
        String cleaned = body.replaceAll("\\s+", " ").strip();
        return cleaned.length() > 400 ? cleaned.substring(0, 400) + "..." : cleaned;
    }

    @Override
    public void close() {
    }
}
