package ai.enr.tests;

import ai.enr.runtime.EnrRuntime;
import ai.enr.runtime.RuntimeConfig;
import ai.enr.server.EnrServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * End-to-end tests for the family ENR 1.0 surface ({@code /enr/v1}) consumed by
 * DCode, TRYST and DStudio.
 */
class FamilyApiIntegrationTest {

    private static final Path MODELS = Path.of("../enr-model-loader/src/test/resources/models").toAbsolutePath();

    private EnrRuntime runtime;
    private EnrServer server;
    private HttpClient http;
    private String base;

    @BeforeEach
    void setUp() throws Exception {
        RuntimeConfig config = RuntimeConfig.builder()
                .addModelDirectory(MODELS)
                .maxConcurrentSessions(16)
                .defaultMaxTokens(16)
                .port(0)
                .build();
        runtime = new EnrRuntime(config);
        server = new EnrServer(runtime, "127.0.0.1", 0);
        server.start();
        base = "http://127.0.0.1:" + server.port();
        http = HttpClient.newHttpClient();
    }

    @AfterEach
    void tearDown() {
        server.close();
        runtime.shutdown();
    }

    @Test
    void familyHealthEndpoint() throws Exception {
        HttpResponse<String> response = get("/enr/v1/health");
        assertEquals(200, response.statusCode());
        assertTrue(response.body().contains("\"application\":\"ENR\""));
        assertTrue(response.body().contains("\"status\":\"UP\""));
        assertTrue(response.body().contains("\"ready\":true"));
    }

    @Test
    void familyModelDiscoveryLoadAndLookup() throws Exception {
        HttpResponse<String> response = get("/enr/v1/models");
        assertEquals(200, response.statusCode());
        assertEquals("[]", response.body().trim());

        String body = "{\"backend\":\"cpu\"}";
        response = post("/enr/v1/models/tiny-evo/load", body);
        assertEquals(200, response.statusCode(), response.body());
        assertTrue(response.body().contains("\"success\":true"));
        assertTrue(response.body().contains("\"tiny-evo\""));

        response = get("/enr/v1/models");
        assertEquals(200, response.statusCode());
        assertTrue(response.body().contains("\"id\":\"tiny-evo\""));
        assertTrue(response.body().contains("\"provider\":\"enr-native\""));

        response = get("/enr/v1/models/tiny-evo");
        assertEquals(200, response.statusCode());
        assertTrue(response.body().contains("\"id\":\"tiny-evo\""));

        response = get("/enr/v1/models/missing");
        assertEquals(404, response.statusCode());
        assertTrue(response.body().contains("\"code\":\"NOT_FOUND\""));

        response = post("/enr/v1/models/tiny-evo/unload", "{}");
        assertEquals(200, response.statusCode());
        assertTrue(response.body().contains("\"success\":true"));
    }

    @Test
    void familyChatAndStreaming() throws Exception {
        post("/enr/v1/models/tiny-evo/load", "{}");

        String chatBody = "{\"model\":\"tiny-evo\",\"messages\":[{\"role\":\"system\","
                + "\"content\":\"token_1 token_2\"},{\"role\":\"user\","
                + "\"content\":\"token_4 token_5 token_6\"}],\"maxTokens\":8}";
        HttpResponse<String> response = post("/enr/v1/chat", chatBody);
        assertEquals(200, response.statusCode(), response.body());
        assertTrue(response.body().contains("\"content\""));
        assertTrue(response.body().contains("\"engine\":\"enr-native\""));
        assertTrue(response.body().contains("\"finishReason\""));

        HttpResponse<String> stream = post("/enr/v1/chat/stream", chatBody);
        assertEquals(200, stream.statusCode());
        List<String> chunks = new ArrayList<>();
        boolean terminalEvent = false;
        for (String line : stream.body().split("\n")) {
            if (line.startsWith("data: ")) {
                chunks.add(line);
            }
            if ("event: done".equals(line.trim())) {
                terminalEvent = true;
            }
        }
        assertFalse(chunks.isEmpty(), "expected SSE data events");
        assertTrue(chunks.stream().anyMatch(l -> l.contains("\"done\":true")),
                "expected terminal done data event");
        assertTrue(terminalEvent, "expected terminal 'event: done' marker");
    }

    @Test
    void familyProvidersRuntimeAndCapabilities() throws Exception {
        HttpResponse<String> providers = get("/enr/v1/providers");
        assertEquals(200, providers.statusCode());
        assertTrue(providers.body().contains("\"id\":\"enr-native\""));
        assertTrue(providers.body().contains("\"type\":\"ENR\""));

        HttpResponse<String> runtimeStatus = get("/enr/v1/runtime");
        assertEquals(200, runtimeStatus.statusCode());
        assertTrue(runtimeStatus.body().contains("\"running\":true"));
        assertTrue(runtimeStatus.body().contains("\"devices\""));

        HttpResponse<String> capabilities = get("/enr/v1/capabilities");
        assertEquals(200, capabilities.statusCode());
        assertTrue(capabilities.body().contains("\"id\":\"ENR_INFERENCE\""));
        assertTrue(capabilities.body().contains("\"id\":\"ENR_MODEL_MANAGEMENT\""));
    }

    @Test
    void familyErrorsUseTeipEnvelope() throws Exception {
        HttpResponse<String> response = post("/enr/v1/chat", "{\"messages\":[]}");
        assertEquals(400, response.statusCode());
        assertTrue(response.body().contains("\"code\":\"VALIDATION_FAILED\""));
        assertTrue(response.body().contains("\"retryable\":false"));
    }

    private HttpResponse<String> get(String path) throws Exception {
        return http.send(HttpRequest.newBuilder()
                .uri(URI.create(base + path))
                .GET()
                .build(), HttpResponse.BodyHandlers.ofString());
    }

    private HttpResponse<String> post(String path, String body) throws Exception {
        return http.send(HttpRequest.newBuilder()
                .uri(URI.create(base + path))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build(), HttpResponse.BodyHandlers.ofString());
    }
}
