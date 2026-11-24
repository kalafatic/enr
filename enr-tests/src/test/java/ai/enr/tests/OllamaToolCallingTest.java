package ai.enr.tests;

import ai.enr.core.json.Json;
import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonObject;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.List;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression tests for Ollama protocol compatibility fixes and tool calling:
 * /api/show model-field fallback + capabilities, plain-string Ollama errors,
 * 404 for unknown models, tool-call passthrough for OpenAI-compatible
 * providers (Ollama and OpenAI wire formats) and explicit 400 rejection for
 * models/providers without structured tool support.
 */
class OllamaToolCallingTest extends ManagerTestBase {

    private static final String WEATHER_TOOL =
            "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
                    + "\"description\":\"Get current weather for a city\","
                    + "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},"
                    + "\"required\":[\"city\"]}}}";

    private final HttpClient http = HttpClient.newBuilder().build();

    // ------------------------------------------------------------ show / tags

    @Test
    void showAcceptsModelFieldLikeRealOllama() throws Exception {
        importTinyEvo("tiny-evo");
        post("/api/v1/models/start", "{\"id\":\"tiny-evo\"}");
        HttpResponse<String> response = post("/api/show", "{\"model\":\"tiny-evo\"}");
        assertEquals(200, response.statusCode(), response.body());
        JsonObject body = Json.parseObject(response.body());
        assertTrue(body.has("modelfile"));
        JsonArray capabilities = body.getArray("capabilities");
        assertNotNull(capabilities, "/api/show must advertise capabilities for OpenCode discovery");
        List<String> caps = new ArrayList<>();
        for (int i = 0; i < capabilities.size(); i++) {
            caps.add(capabilities.getString(i));
        }
        assertTrue(caps.contains("completion"));
        JsonObject modelInfo = body.getObject("model_info");
        assertNotNull(modelInfo);
        long context = -1;
        for (var entry : modelInfo.entries()) {
            if (entry.getKey().endsWith(".context_length")
                    && entry.getValue() instanceof ai.enr.core.json.JsonNumber n) {
                context = n.asLong();
            }
        }
        assertTrue(context > 0, "model_info must expose a numeric *.context_length");
    }

    @Test
    void showWithoutLoadedModelStillAdvertisesCapabilities() throws Exception {
        importTinyEvo("tiny-evo");
        HttpResponse<String> response = post("/api/show", "{\"name\":\"tiny-evo\"}");
        assertEquals(200, response.statusCode());
        JsonObject body = Json.parseObject(response.body());
        JsonArray capabilities = body.getArray("capabilities");
        assertNotNull(capabilities, "capabilities must not depend on runtime load state");
        List<String> caps = new ArrayList<>();
        for (int i = 0; i < capabilities.size(); i++) {
            caps.add(capabilities.getString(i));
        }
        assertTrue(caps.contains("completion"));
    }

    @Test
    void showByNameStillWorks() throws Exception {
        importTinyEvo("tiny-evo");
        HttpResponse<String> response = post("/api/show", "{\"name\":\"tiny-evo\"}");
        assertEquals(200, response.statusCode());
        assertTrue(Json.parseObject(response.body()).has("modelfile"));
    }

    @Test
    void tagsAdvertiseCapabilitiesPerModel() throws Exception {
        importTinyEvo("tiny-evo");
        JsonObject body = json(get("/api/tags"));
        JsonArray models = body.getArray("models");
        assertEquals(1, models.size());
        JsonObject model = models.getObject(0);
        JsonArray capabilities = model.getArray("capabilities");
        assertNotNull(capabilities);
        assertTrue(capabilities.size() > 0);
    }

    // ------------------------------------------------------------ error format

    @Test
    void chatUnknownModelReturns404WithPlainError() throws Exception {
        HttpResponse<String> response = post("/api/chat",
                "{\"model\":\"nope\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
        assertEquals(404, response.statusCode());
        JsonObject body = Json.parseObject(response.body());
        assertTrue(body.get("error") instanceof ai.enr.core.json.JsonString,
                "Ollama errors must be a plain string: " + response.body());
    }

    @Test
    void generateUnknownModelReturns404WithPlainError() throws Exception {
        HttpResponse<String> response = post("/api/generate", "{\"model\":\"nope\",\"prompt\":\"hi\"}");
        assertEquals(404, response.statusCode());
        assertTrue(Json.parseObject(response.body()).get("error")
                instanceof ai.enr.core.json.JsonString);
    }

    @Test
    void showUnknownModelReturns404WithPlainError() throws Exception {
        HttpResponse<String> response = post("/api/show", "{\"name\":\"nope\"}");
        assertEquals(404, response.statusCode());
        assertTrue(Json.parseObject(response.body()).get("error")
                instanceof ai.enr.core.json.JsonString);
    }

    @Test
    void chatMalformedJsonReturns400() throws Exception {
        HttpResponse<String> response = post("/api/chat", "not json");
        assertEquals(400, response.statusCode(), response.body());
        assertTrue(Json.parseObject(response.body()).get("error")
                instanceof ai.enr.core.json.JsonString);
    }

    @Test
    void generateMalformedJsonReturns400() throws Exception {
        HttpResponse<String> response = post("/api/generate", "not json");
        assertEquals(400, response.statusCode(), response.body());
        assertTrue(Json.parseObject(response.body()).get("error")
                instanceof ai.enr.core.json.JsonString);
    }

    @Test
    void openAiChatMalformedJsonReturns400() throws Exception {
        HttpResponse<String> response = post("/v1/chat/completions", "not json");
        assertEquals(400, response.statusCode(), response.body());
        JsonObject body = Json.parseObject(response.body());
        assertEquals("INVALID_REQUEST", body.getObject("error").getString("code", ""));
    }

    // ------------------------------------------------------------ local tools rejected

    @Test
    void chatWithToolsOnLocalModelReturns400() throws Exception {
        importTinyEvo("tiny-evo");
        HttpResponse<String> response = post("/api/chat",
                "{\"model\":\"tiny-evo\",\"messages\":[{\"role\":\"user\",\"content\":\"weather?\"}],"
                        + "\"tools\":[" + WEATHER_TOOL + "]}");
        assertEquals(400, response.statusCode(), response.body());
        JsonObject body = Json.parseObject(response.body());
        assertTrue(body.getString("error", "").contains("does not support tools"), response.body());
    }

    @Test
    void streamingChatWithToolsOnLocalModelRejectsBeforeHeaders() throws Exception {
        importTinyEvo("tiny-evo");
        HttpResponse<java.io.InputStream> response = postStream("/api/chat",
                "{\"model\":\"tiny-evo\",\"stream\":true,"
                        + "\"messages\":[{\"role\":\"user\",\"content\":\"weather?\"}],"
                        + "\"tools\":[" + WEATHER_TOOL + "]}");
        assertEquals(400, response.statusCode());
        try (var in = response.body()) {
            JsonObject body = Json.parseObject(new String(in.readAllBytes()));
            assertTrue(body.getString("error", "").contains("does not support tools"));
        }
    }

    @Test
    void openAiChatWithToolsOnLocalModelReturns400() throws Exception {
        importTinyEvo("tiny-evo");
        HttpResponse<String> response = post("/v1/chat/completions",
                "{\"model\":\"tiny-evo\",\"messages\":[{\"role\":\"user\",\"content\":\"weather?\"}],"
                        + "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
                        + "\"parameters\":{\"type\":\"object\"}}}]}");
        assertEquals(400, response.statusCode(), response.body());
    }

    @Test
    void toolsRejectedForNonOpenAiProviderTypes() throws Exception {
        manager.addProvider("gem", "gemini", "Gemini", "https://example.invalid", "sk-key", true);
        HttpResponse<String> response = post("/api/chat",
                "{\"model\":\"gem:some-model\",\"messages\":[{\"role\":\"user\",\"content\":\"weather?\"}],"
                        + "\"tools\":[" + WEATHER_TOOL + "]}");
        assertEquals(400, response.statusCode(), response.body());
        assertTrue(Json.parseObject(response.body()).getString("error", "")
                .contains("not supported for provider type"));
    }

    // ------------------------------------------------------------ provider passthrough

    @Test
    void chatToolsForwardedToProviderAndReturnedInOllamaFormat() throws Exception {
        AtomicReference<String> upstream = new AtomicReference<>();
        HttpServer provider = MockServers.toolCallingProvider("sk-test-key", upstream);
        try {
            manager.addProvider("mock", "openai-compatible", "Mock",
                    "http://127.0.0.1:" + provider.getAddress().getPort() + "/v1", "sk-test-key", true);
            JsonObject body = json(post("/api/chat",
                    "{\"model\":\"mock:mock-model\","
                            + "\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris?\"}],"
                            + "\"tools\":[" + WEATHER_TOOL + "]}"));
            JsonObject message = body.getObject("message");
            assertEquals("assistant", message.getString("role", ""));
            JsonArray toolCalls = message.getArray("tool_calls");
            assertNotNull(toolCalls);
            assertEquals(1, toolCalls.size());
            JsonObject function = toolCalls.getObject(0).getObject("function");
            assertEquals("get_weather", function.getString("name", ""));
            JsonObject arguments = function.getObject("arguments");
            assertNotNull(arguments);
            assertEquals("Paris", arguments.getString("city", ""));

            JsonObject upstreamBody = Json.parseObject(upstream.get());
            assertTrue(upstreamBody.has("tools"), "tools must be forwarded upstream");
            JsonArray upstreamTools = upstreamBody.getArray("tools");
            assertEquals(1, upstreamTools.size());
            assertEquals("get_weather",
                    upstreamTools.getObject(0).getObject("function").getString("name", ""));
        } finally {
            provider.stop(0);
        }
    }

    @Test
    void chatStreamingToolsReturnedInFinalChunk() throws Exception {
        HttpServer provider = MockServers.toolCallingProvider("sk-test-key", null);
        try {
            manager.addProvider("mock", "openai-compatible", "Mock",
                    "http://127.0.0.1:" + provider.getAddress().getPort() + "/v1", "sk-test-key", true);
            List<JsonObject> lines = ndjson(postStream("/api/chat",
                    "{\"model\":\"mock:mock-model\",\"stream\":true,"
                            + "\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris?\"}],"
                            + "\"tools\":[" + WEATHER_TOOL + "]}"));
            assertFalse(lines.isEmpty());
            JsonObject last = lines.get(lines.size() - 1);
            assertTrue(last.getBoolean("done", false));
            JsonArray toolCalls = last.getObject("message").getArray("tool_calls");
            assertNotNull(toolCalls, "final stream chunk must carry tool_calls");
            assertEquals(1, toolCalls.size());
            assertEquals("get_weather",
                    toolCalls.getObject(0).getObject("function").getString("name", ""));
            assertEquals("Paris",
                    toolCalls.getObject(0).getObject("function").getObject("arguments")
                            .getString("city", ""));
            assertEquals("tool_calls", last.getString("done_reason", ""));
        } finally {
            provider.stop(0);
        }
    }

    @Test
    void chatRoundTripsAssistantToolCallsAndToolResults() throws Exception {
        AtomicReference<String> upstream = new AtomicReference<>();
        HttpServer provider = MockServers.toolCallingProvider("sk-test-key", upstream);
        try {
            manager.addProvider("mock", "openai-compatible", "Mock",
                    "http://127.0.0.1:" + provider.getAddress().getPort() + "/v1", "sk-test-key", true);
            HttpResponse<String> response = post("/api/chat",
                    "{\"model\":\"mock:mock-model\","
                            + "\"messages\":["
                            + "{\"role\":\"user\",\"content\":\"What is the weather in Paris?\"},"
                            + "{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"function\":"
                            + "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}}]},"
                            + "{\"role\":\"tool\",\"content\":\"22 degrees\"}],"
                            + "\"tools\":[" + WEATHER_TOOL + "]}");
            assertEquals(200, response.statusCode(), response.body());

            JsonObject upstreamBody = Json.parseObject(upstream.get());
            JsonArray messages = upstreamBody.getArray("messages");
            assertEquals(3, messages.size());
            JsonObject assistant = messages.getObject(1);
            assertTrue(assistant.has("tool_calls"), "assistant tool_calls must round-trip");
            assertEquals("get_weather", assistant.getArray("tool_calls").getObject(0)
                    .getObject("function").getString("name", ""));
            JsonObject tool = messages.getObject(2);
            assertEquals("tool", tool.getString("role", ""));
            assertEquals("22 degrees", tool.getString("content", ""));
        } finally {
            provider.stop(0);
        }
    }

    @Test
    void openAiChatCompletionsToolCallingNonStream() throws Exception {
        HttpServer provider = MockServers.toolCallingProvider("sk-test-key", null);
        try {
            manager.addProvider("mock", "openai-compatible", "Mock",
                    "http://127.0.0.1:" + provider.getAddress().getPort() + "/v1", "sk-test-key", true);
            JsonObject body = json(post("/v1/chat/completions",
                    "{\"model\":\"mock:mock-model\","
                            + "\"messages\":[{\"role\":\"user\",\"content\":\"weather?\"}],"
                            + "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
                            + "\"parameters\":{\"type\":\"object\"}}}]}"));
            JsonObject choice = body.getArray("choices").getObject(0);
            assertEquals("tool_calls", choice.getString("finish_reason", ""));
            JsonArray toolCalls = choice.getObject("message").getArray("tool_calls");
            assertNotNull(toolCalls);
            assertEquals(1, toolCalls.size());
            assertEquals("call_abc", toolCalls.getObject(0).getString("id", ""));
            assertEquals("function", toolCalls.getObject(0).getString("type", ""));
            assertEquals("get_weather",
                    toolCalls.getObject(0).getObject("function").getString("name", ""));
            assertEquals("{\"city\":\"Paris\"}",
                    toolCalls.getObject(0).getObject("function").getString("arguments", ""));
        } finally {
            provider.stop(0);
        }
    }

    @Test
    void openAiChatCompletionsToolCallingStream() throws Exception {
        HttpServer provider = MockServers.toolCallingProvider("sk-test-key", null);
        try {
            manager.addProvider("mock", "openai-compatible", "Mock",
                    "http://127.0.0.1:" + provider.getAddress().getPort() + "/v1", "sk-test-key", true);
            HttpResponse<java.io.InputStream> response = postStream("/v1/chat/completions",
                    "{\"model\":\"mock:mock-model\",\"stream\":true,"
                            + "\"messages\":[{\"role\":\"user\",\"content\":\"weather?\"}],"
                            + "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
                            + "\"parameters\":{\"type\":\"object\"}}}]}");
            assertEquals(200, response.statusCode());
            List<JsonObject> chunks = new ArrayList<>();
            boolean sawDone = false;
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(response.body()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.startsWith("data: [DONE]")) {
                        sawDone = true;
                        continue;
                    }
                    if (!line.startsWith("data: ")) {
                        continue;
                    }
                    chunks.add(Json.parseObject(line.substring(6)));
                }
            }
            assertTrue(sawDone, "stream must terminate with data: [DONE]");
            JsonObject last = chunks.get(chunks.size() - 1);
            JsonObject choice = last.getArray("choices").getObject(0);
            assertEquals("tool_calls", choice.getString("finish_reason", ""));
            JsonArray toolCalls = choice.getObject("delta").getArray("tool_calls");
            assertNotNull(toolCalls, "final delta must carry tool_calls");
            assertEquals(1, toolCalls.size());
            assertEquals("get_weather",
                    toolCalls.getObject(0).getObject("function").getString("name", ""));
            JsonObject args = Json.parseObject(toolCalls.getObject(0).getObject("function")
                    .getString("arguments", ""));
            assertEquals("Paris", args.getString("city", ""));
            assertEquals("sunny", args.getString("mode", ""));
        } finally {
            provider.stop(0);
        }
    }

    @Test
    void storagePathsExpandTilde() {
        assertEquals(Path.of(System.getProperty("user.home"), ".enr").toAbsolutePath(),
                ai.enr.manager.EnrConfigLoader.expandHome("~/.enr"));
        assertEquals(Path.of(System.getProperty("user.home")).toAbsolutePath(),
                ai.enr.manager.EnrConfigLoader.expandHome("~"));
        assertEquals(Path.of("/opt/enr/data").toAbsolutePath(),
                ai.enr.manager.EnrConfigLoader.expandHome("/opt/enr/data"));
    }

    @Test
    void yamlInlineCommentsAreStripped() throws Exception {
        Path yaml = tempDir.resolve("enr-comment-test.yaml");
        java.nio.file.Files.writeString(yaml, """
                storage:
                  data_dir: ~/.enr            # provider state (0600), registry
                  models_dir: ~/.enr/models   # installed models
                  log_dir: ~/.enr/logs
                server:
                  port: 9123                  # test port
                """);
        JsonObject cfg = ai.enr.manager.EnrConfigLoader.parseYaml(yaml);
        JsonObject storage = cfg.getObject("storage");
        assertEquals("~/.enr", storage.getString("data_dir"));
        assertEquals("~/.enr/models", storage.getString("models_dir"));
        assertEquals(9123, cfg.getObject("server").getInt("port", 0));
    }

    // ------------------------------------------------------------ helpers

    private HttpResponse<String> get(String path) throws Exception {
        return http.send(HttpRequest.newBuilder().uri(URI.create(base + path)).GET().build(),
                HttpResponse.BodyHandlers.ofString());
    }

    private HttpResponse<String> post(String path, String body) throws Exception {
        return http.send(HttpRequest.newBuilder().uri(URI.create(base + path))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body)).build(),
                HttpResponse.BodyHandlers.ofString());
    }

    private HttpResponse<java.io.InputStream> postStream(String path, String body) throws Exception {
        return http.send(HttpRequest.newBuilder().uri(URI.create(base + path))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body)).build(),
                HttpResponse.BodyHandlers.ofInputStream());
    }

    private static JsonObject json(HttpResponse<String> response) {
        assertEquals(200, response.statusCode(), response.body());
        return Json.parseObject(response.body());
    }

    private static List<JsonObject> ndjson(HttpResponse<java.io.InputStream> response) throws Exception {
        assertEquals(200, response.statusCode());
        List<JsonObject> lines = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(response.body()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.isBlank()) {
                    lines.add(Json.parseObject(line));
                }
            }
        }
        return lines;
    }
}
