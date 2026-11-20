package ai.enr.tests;

import ai.enr.core.json.Json;
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

import static org.junit.jupiter.api.Assertions.*;

/**
 * OpenAI API compatibility: /v1/models, /v1/chat/completions (sync + SSE
 * streaming), /v1/completions, /v1/embeddings — local models via the native
 * runtime and cloud models via a mock OpenAI-compatible provider.
 */
class OpenAiCompatibilityTest extends ManagerTestBase {

    private final HttpClient http = HttpClient.newBuilder().build();

    @Test
    void modelsListsLocalAndProviderModels() throws Exception {
        importTinyEvo("tiny-evo");
        JsonObject body = json(get("/v1/models"));
        assertEquals("list", body.getString("object", ""));
        var data = body.getArray("data");
        assertTrue(data.size() >= 1);
        List<String> ids = new ArrayList<>();
        for (int i = 0; i < data.size(); i++) {
            ids.add(data.getObject(i).getString("id", ""));
        }
        assertTrue(ids.contains("tiny-evo"));
    }

    @Test
    void chatCompletionsAgainstLocalModel() throws Exception {
        importTinyEvo("tiny-evo");
        JsonObject body = json(post("/v1/chat/completions",
                "{\"model\":\"tiny-evo\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}"));
        assertEquals("chat.completion", body.getString("object", ""));
        assertTrue(body.getString("id", "").startsWith("chatcmpl-"));
        var choices = body.getArray("choices");
        assertEquals(1, choices.size());
        JsonObject message = choices.getObject(0).getObject("message");
        assertEquals("assistant", message.getString("role", ""));
        assertFalse(message.getString("content", "").isEmpty());
        assertEquals("stop", choices.getObject(0).getString("finish_reason", ""));
        JsonObject usage = body.getObject("usage");
        assertTrue(usage.getInt("total_tokens", 0) > 0);
        assertEquals(usage.getInt("total_tokens", -1),
                usage.getInt("prompt_tokens", 0) + usage.getInt("completion_tokens", 0));
    }

    @Test
    void chatCompletionsStreamingSse() throws Exception {
        importTinyEvo("tiny-evo");
        HttpResponse<java.io.InputStream> response = postStream("/v1/chat/completions",
                "{\"model\":\"tiny-evo\",\"stream\":true,"
                        + "\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}");
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
        assertFalse(chunks.isEmpty());
        assertTrue(chunks.stream().anyMatch(c -> !c.getArray("choices").getObject(0)
                .getObject("delta").getString("content", "").isEmpty()));
        JsonObject last = chunks.get(chunks.size() - 1);
        assertFalse(last.getArray("choices").getObject(0).getString("finish_reason", "").isEmpty());
    }

    @Test
    void completionsAgainstLocalModel() throws Exception {
        importTinyEvo("tiny-evo");
        JsonObject body = json(post("/v1/completions",
                "{\"model\":\"tiny-evo\",\"prompt\":\"hello\",\"max_tokens\":8}"));
        assertEquals("text_completion", body.getString("object", ""));
        var choices = body.getArray("choices");
        assertEquals(1, choices.size());
        assertFalse(choices.getObject(0).getString("text", "").isEmpty());
        assertEquals(0, choices.getObject(0).getInt("index", -1));
    }

    @Test
    void embeddingsForLocalModelsReturn501() throws Exception {
        importTinyEvo("tiny-evo");
        HttpResponse<String> response = post("/v1/embeddings",
                "{\"model\":\"tiny-evo\",\"input\":\"hello\"}");
        assertEquals(501, response.statusCode());
        assertTrue(response.body().toLowerCase().contains("embedding"));
    }

    @Test
    void chatCompletionsRoutedToCloudProvider() throws Exception {
        HttpServer provider = MockServers.openAiProvider("sk-test-key", List.of("mock-model"));
        try {
            manager.addProvider("mock", "openai-compatible", "Mock Cloud",
                    "http://127.0.0.1:" + provider.getAddress().getPort() + "/v1", "sk-test-key", true);
            JsonObject body = json(post("/v1/chat/completions",
                    "{\"model\":\"mock:mock-model\","
                            + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"));
            var choices = body.getArray("choices");
            JsonObject message = choices.getObject(0).getObject("message");
            assertEquals("mock reply", message.getString("content", ""));
            JsonObject usage = body.getObject("usage");
            assertEquals(3, usage.getInt("prompt_tokens", 0));
            assertEquals(2, usage.getInt("completion_tokens", 0));
        } finally {
            provider.stop(0);
        }
    }

    @Test
    void cloudProviderRejectsWrongKey() throws Exception {
        HttpServer provider = MockServers.openAiProvider("sk-real", List.of("mock-model"));
        try {
            manager.addProvider("mock", "openai-compatible", "Mock Cloud",
                    "http://127.0.0.1:" + provider.getAddress().getPort() + "/v1", "sk-wrong", true);
            JsonObject result = manager.testProvider("mock");
            assertFalse(result.getBoolean("ok", true));
        } finally {
            provider.stop(0);
        }
    }

    @Test
    void embeddingsForwardedToCloudProvider() throws Exception {
        HttpServer provider = MockServers.openAiProvider("sk-test-key", List.of("mock-model"));
        try {
            manager.addProvider("mock", "openai-compatible", "Mock Cloud",
                    "http://127.0.0.1:" + provider.getAddress().getPort() + "/v1", "sk-test-key", true);
            JsonObject body = json(post("/v1/embeddings",
                    "{\"model\":\"mock:mock-model\",\"input\":\"hello\"}"));
            assertEquals("list", body.getString("object", ""));
            var data = body.getArray("data");
            assertEquals(1, data.size());
            var embedding = data.getObject(0).getArray("embedding");
            assertEquals(3, embedding.size());
        } finally {
            provider.stop(0);
        }
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
}
