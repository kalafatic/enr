package ai.enr.tests;

import ai.enr.client.EnrClient;
import ai.enr.core.json.Json;
import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonObject;
import ai.enr.manager.EnrConfig;
import ai.enr.manager.EnrManager;
import ai.enr.server.EnrServer;
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
 * Ollama API compatibility: /api/tags, /api/show, /api/pull, /api/delete,
 * /api/chat, /api/generate, /api/ps, /api/version — exercised against a live
 * ENR server with a real local model and a mock registry.
 */
class OllamaCompatibilityTest extends ManagerTestBase {

    private final HttpClient http = HttpClient.newBuilder().build();

    @Test
    void versionEndpoint() throws Exception {
        JsonObject version = json(get("/api/version"));
        assertEquals("2.6.11", version.getString("version", ""));
    }

    @Test
    void tagsShowInstalledModels() throws Exception {
        importTinyEvo("tiny-evo");
        JsonObject body = json(get("/api/tags"));
        var models = body.getArray("models");
        assertNotNull(models);
        assertEquals(1, models.size());
        JsonObject model = models.getObject(0);
        assertEquals("tiny-evo", model.getString("name", ""));
        assertTrue(model.has("digest"));
        assertTrue(model.has("details"));
    }

    @Test
    void psInitiallyEmptyThenShowsRunningModel() throws Exception {
        JsonObject empty = json(get("/api/ps"));
        assertEquals(0, empty.getArray("models").size());
        importTinyEvo("tiny-evo");
        post("/api/v1/models/start", "{\"id\":\"tiny-evo\"}");
        JsonObject ps = json(get("/api/ps"));
        var models = ps.getArray("models");
        assertEquals(1, models.size());
        assertEquals("tiny-evo", models.getObject(0).getString("name", ""));
    }

    @Test
    void showReturnsModelInfo() throws Exception {
        importTinyEvo("tiny-evo");
        JsonObject body = json(post("/api/show", "{\"name\":\"tiny-evo\"}"));
        assertTrue(body.has("modelfile"));
        assertTrue(body.has("details"));
    }

    @Test
    void chatAgainstLocalModel() throws Exception {
        importTinyEvo("tiny-evo");
        JsonObject body = json(post("/api/chat",
                "{\"model\":\"tiny-evo\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}"));
        assertTrue(body.getBoolean("done", false));
        JsonObject message = body.getObject("message");
        assertNotNull(message);
        assertEquals("assistant", message.getString("role", ""));
        assertFalse(message.getString("content", "").isEmpty());
        assertTrue(body.has("eval_count"));
        assertTrue(body.has("prompt_eval_count"));
    }

    @Test
    void chatStreamingAgainstLocalModel() throws Exception {
        importTinyEvo("tiny-evo");
        List<JsonObject> lines = ndjson(postStream("/api/chat",
                "{\"model\":\"tiny-evo\",\"stream\":true,"
                        + "\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}"));
        assertFalse(lines.isEmpty());
        assertTrue(lines.get(0).has("message"));
        JsonObject last = lines.get(lines.size() - 1);
        assertTrue(last.getBoolean("done", false));
    }

    @Test
    void generateAgainstLocalModel() throws Exception {
        importTinyEvo("tiny-evo");
        JsonObject body = json(post("/api/generate",
                "{\"model\":\"tiny-evo\",\"prompt\":\"hello\"}"));
        assertTrue(body.getBoolean("done", false));
        assertFalse(body.getString("response", "").isEmpty());
    }

    @Test
    void pullFromRegistryStreamsProgressAndInstalls() throws Exception {
        HttpServer registry = MockServers.ollamaRegistry(
                FIXTURES.resolve("tiny-evo.gguf"), "catalog-model", "latest");
        try {
            var config = ai.enr.manager.EnrConfig.builder()
                    .dataDir(tempDir.resolve("data"))
                    .modelsDir(tempDir.resolve("models"))
                    .logDir(tempDir.resolve("logs"))
                    .ollamaRegistryUrl("http://127.0.0.1:" + registry.getAddress().getPort())
                    .build();
            restartManager(config);
            List<JsonObject> lines = ndjson(postStream("/api/pull",
                    "{\"name\":\"catalog-model:latest\",\"stream\":true}"));
            assertFalse(lines.isEmpty());
            JsonObject last = lines.get(lines.size() - 1);
            assertEquals("success", last.getString("status", ""));
            assertTrue(lines.stream().anyMatch(l -> l.getString("status", "")
                    .startsWith("pulling manifest")));
            assertTrue(lines.stream().anyMatch(l -> l.getString("status", "")
                    .startsWith("downloading")));
            assertTrue(lines.stream().anyMatch(l -> l.getString("status", "")
                    .startsWith("verifying sha256 digest")));
            JsonObject tags = json(get("/api/tags"));
            var models = tags.getArray("models");
            assertEquals(1, models.size());
            assertEquals("catalog-model", models.getObject(0).getString("name", ""));
            JsonObject installed = client.getJson("/api/v1/catalog/models");
            JsonObject entry = installed.getArray("models").getObject(0);
            assertEquals("catalog:catalog-model:latest", entry.getString("source", ""));
            assertTrue(entry.getBoolean("verified", false));
        } finally {
            registry.stop(0);
        }
    }

    @Test
    void deleteRemovesModel() throws Exception {
        importTinyEvo("tiny-evo");
        HttpResponse<String> response = delete("/api/delete", "{\"name\":\"tiny-evo\"}");
        assertEquals(200, response.statusCode());
        JsonObject tags = json(get("/api/tags"));
        assertEquals(0, tags.getArray("models").size());
    }

    @Test
    void deleteUnknownModelReturns404() throws Exception {
        HttpResponse<String> response = delete("/api/delete", "{\"name\":\"nope\"}");
        assertEquals(404, response.statusCode());
    }

    @Test
    void copyModelCreatesNewIdentifier() throws Exception {
        importTinyEvo("tiny-evo");
        HttpResponse<String> response = post("/api/copy",
                "{\"source\":\"tiny-evo\",\"destination\":\"tiny-evo-copy\"}");
        assertEquals(200, response.statusCode(), response.body());
        JsonObject tags = json(get("/api/tags"));
        var models = tags.getArray("models");
        assertEquals(2, models.size());
        List<String> names = new ArrayList<>();
        for (int i = 0; i < models.size(); i++) {
            names.add(models.getObject(i).getString("name", ""));
        }
        assertTrue(names.contains("tiny-evo"));
        assertTrue(names.contains("tiny-evo-copy"));
    }

    @Test
    void copyMissingSourceReturns404() throws Exception {
        HttpResponse<String> response = post("/api/copy",
                "{\"source\":\"nope\",\"destination\":\"x\"}");
        assertEquals(404, response.statusCode());
    }

    @Test
    void embedOnLocalModelReturns501() throws Exception {
        importTinyEvo("tiny-evo");
        HttpResponse<String> response = post("/api/embed",
                "{\"model\":\"tiny-evo\",\"input\":\"hello\"}");
        assertEquals(501, response.statusCode(), response.body());
    }

    @Test
    void embedViaProviderReturnsEmbeddings() throws Exception {
        HttpServer provider = MockServers.openAiProvider("sk-test-key", List.of("mock-model"));
        try {
            manager.addProvider("mock", "openai-compatible", "Mock",
                    "http://127.0.0.1:" + provider.getAddress().getPort() + "/v1", "sk-test-key", true);
            JsonObject body = json(post("/api/embed", "{\"model\":\"mock:mock-model\",\"input\":\"hello\"}"));
            JsonArray embeddings = body.getArray("embeddings");
            assertNotNull(embeddings);
            assertEquals(1, embeddings.size());
            assertEquals(1, body.getInt("prompt_eval_count", 0));
        } finally {
            provider.stop(0);
        }
    }

    // ------------------------------------------------------------ helpers

    private void restartManager(EnrConfig config) {
        server.close();
        manager.close();
        manager = new EnrManager(config);
        try {
            server = new EnrServer(manager, "127.0.0.1", 0);
            server.start();
            base = "http://127.0.0.1:" + server.port();
            client = new EnrClient(base);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

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

    private HttpResponse<String> delete(String path, String body) throws Exception {
        return http.send(HttpRequest.newBuilder().uri(URI.create(base + path))
                        .header("Content-Type", "application/json")
                        .method("DELETE", HttpRequest.BodyPublishers.ofString(body)).build(),
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
