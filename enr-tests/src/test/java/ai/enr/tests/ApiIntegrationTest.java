package ai.enr.tests;

import ai.enr.client.EnrClient;
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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end API tests: real server + real Java client + real models.
 */
class ApiIntegrationTest {

    private static final Path MODELS = Path.of("../enr-model-loader/src/test/resources/models").toAbsolutePath();

    private EnrRuntime runtime;
    private EnrServer server;
    private EnrClient client;

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
        client = new EnrClient("http://127.0.0.1:" + server.port());
    }

    @AfterEach
    void tearDown() {
        server.close();
        runtime.shutdown();
    }

    @Test
    void healthEndpoint() {
        assertTrue(client.health());
    }

    @Test
    void statusEndpoint() {
        var status = client.status();
        assertTrue(status.getBoolean("running", false));
    }

    @Test
    void modelListInitiallyEmptyThenLoad() {
        assertTrue(client.models().isEmpty());
        var model = client.loadModel(MODELS.resolve("tiny-evo.gguf").toString(), "cpu");
        assertEquals("tiny-evo", model.id());
        assertEquals(1, client.models().size());
        assertEquals("tiny-evo", client.model("tiny-evo").id());
    }

    @Test
    void unloadModel() {
        client.loadModel(MODELS.resolve("tiny-evo.gguf").toString(), "cpu");
        client.unloadModel("tiny-evo", false);
        assertTrue(client.models().isEmpty());
    }

    @Test
    void inferenceEndpointProducesText() {
        client.loadModel(MODELS.resolve("tiny-evo.gguf").toString(), "cpu");
        var result = client.infer("tiny-evo", "token_4 token_5 token_6");
        assertNotNull(result.text());
        assertTrue(result.generatedTokens() > 0);
        assertFalse(result.reason().isEmpty());
    }

    @Test
    void inferenceWithMissingModelFails() {
        var ex = assertThrows(EnrClient.EnrException.class,
                () -> client.infer("nope", "hello"));
        assertEquals(400, ex.status());
    }

    @Test
    void streamingInferenceDeliversTokensAndDone() throws Exception {
        client.loadModel(MODELS.resolve("tiny-evo.gguf").toString(), "cpu");
        List<String> pieces = new ArrayList<>();
        CountDownLatch done = new CountDownLatch(1);
        List<EnrClient.InferenceResult> results = new ArrayList<>();
        client.inferStream("tiny-evo", "token_4 token_5 token_6",
                new EnrClient.InferenceRequest().maxTokens(8),
                token -> pieces.add(token.text()),
                result -> {
                    results.add(result);
                    done.countDown();
                });
        assertTrue(done.await(30, TimeUnit.SECONDS));
        assertFalse(results.isEmpty());
        assertTrue(results.get(0).generatedTokens() > 0);
    }

    @Test
    void concurrentRequestsViaHttp() throws Exception {
        client.loadModel(MODELS.resolve("tiny-evo.gguf").toString(), "cpu");
        client.loadModel(MODELS.resolve("evo.gguf").toString(), "cpu");
        assertEquals(2, client.models().size());

        HttpClient http = HttpClient.newHttpClient();
        int workers = 8;
        var futures = new ArrayList<java.util.concurrent.Future<HttpResponse<String>>>();
        try (var pool = java.util.concurrent.Executors.newFixedThreadPool(workers)) {
            for (int i = 0; i < workers; i++) {
                final String model = i % 2 == 0 ? "tiny-evo" : "evo";
                futures.add(pool.submit(() -> {
                    String body = "{\"model\":\"" + model + "\",\"prompt\":\"token_4 token_5\",\"max_tokens\":8}";
                    HttpRequest request = HttpRequest.newBuilder()
                            .uri(URI.create("http://127.0.0.1:" + server.port() + "/api/v1/inference"))
                            .header("Content-Type", "application/json")
                            .POST(HttpRequest.BodyPublishers.ofString(body))
                            .build();
                    return http.send(request, HttpResponse.BodyHandlers.ofString());
                }));
            }
            for (var future : futures) {
                HttpResponse<String> response = future.get(60, TimeUnit.SECONDS);
                assertEquals(200, response.statusCode(), response.body());
                assertTrue(response.body().contains("\"text\""));
            }
        }
    }

    @Test
    void invalidRequestsRejected() throws Exception {
        HttpClient http = HttpClient.newHttpClient();
        HttpResponse<String> response = http.send(HttpRequest.newBuilder()
                .uri(URI.create("http://127.0.0.1:" + server.port() + "/api/v1/inference"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString("{\"model\":\"x\"}"))
                .build(), HttpResponse.BodyHandlers.ofString());
        assertEquals(400, response.statusCode());
        assertTrue(response.body().contains("error"));

        response = http.send(HttpRequest.newBuilder()
                .uri(URI.create("http://127.0.0.1:" + server.port() + "/api/v1/health"))
                .POST(HttpRequest.BodyPublishers.noBody())
                .build(), HttpResponse.BodyHandlers.ofString());
        assertEquals(405, response.statusCode());
    }

    @Test
    void sessionsAndMetricsEndpoints() {
        client.loadModel(MODELS.resolve("tiny-evo.gguf").toString(), "cpu");
        client.infer("tiny-evo", "token_4 token_5");
        var metrics = client.metrics();
        assertTrue(metrics.getLong("total_requests", 0) >= 1);
        assertTrue(metrics.getLong("total_generated_tokens", 0) > 0);
        assertFalse(client.devices().isEmpty());
    }

    @Test
    void devicesEndpoint() {
        var devices = client.devices();
        assertFalse(devices.isEmpty());
        assertTrue(devices.stream().anyMatch(d -> "cpu".equals(d.type())));
    }
}
