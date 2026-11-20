package ai.enr.tests;

import ai.enr.client.EnrClient;
import ai.enr.core.json.Json;
import ai.enr.core.json.JsonObject;
import ai.enr.manager.EnrConfig;
import ai.enr.manager.EnrManager;
import ai.enr.manager.service.SystemdSupport;
import ai.enr.server.EnrServer;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Manager-level tests: model store lifecycle, provider registry + secret
 * redaction, router determinism, download cancellation, admin auth and the
 * systemd unit.
 */
class ManagerTest extends ManagerTestBase {

    private final HttpClient http = HttpClient.newBuilder().build();

    @Test
    void importListVerifyRemoveLifecycle() throws Exception {
        importTinyEvo("my-model");
        JsonObject list = client.getJson("/api/v1/catalog/models");
        assertEquals(1, list.getArray("models").size());
        JsonObject details = client.getJson("/api/v1/catalog/models/my-model");
        assertEquals("gguf", details.getString("format", ""));
        assertTrue(details.getBoolean("verified", false));

        JsonObject verified = client.postJson("/api/v1/catalog/verify", obj("id", "my-model"));
        assertTrue(verified.getBoolean("verified", false));

        client.deleteJson("/api/v1/catalog/models/my-model");
        JsonObject after = client.getJson("/api/v1/catalog/models");
        assertEquals(0, after.getArray("models").size());
    }

    @Test
    void startAndStopModelVerifiesRealState() throws Exception {
        importTinyEvo("tiny-evo");
        client.postJson("/api/v1/models/start", obj("id", "tiny-evo"));
        JsonObject list = client.getJson("/api/v1/catalog/models");
        JsonObject model = list.getArray("models").getObject(0);
        assertTrue(model.getBoolean("loaded", false));
        assertEquals("ready", model.getString("state", ""));

        client.postJson("/api/v1/models/stop", obj("id", "tiny-evo"));
        JsonObject after = client.getJson("/api/v1/catalog/models");
        assertFalse(after.getArray("models").getObject(0).getBoolean("loaded", true));
    }

    @Test
    void providerApiKeyNeverExposed() throws Exception {
        HttpServer provider = MockServers.openAiProvider("sk-secret-key", List.of("m1"));
        try {
            manager.addProvider("mock", "openai-compatible", "Mock",
                    "http://127.0.0.1:" + provider.getAddress().getPort() + "/v1",
                    "sk-secret-key", true);
            JsonObject providers = client.getJson("/api/v1/providers");
            String raw = Json.write(providers, false);
            assertFalse(raw.contains("sk-secret-key"), "API key leaked in provider listing");
            JsonObject entry = providers.getArray("providers").getObject(0);
            assertTrue(entry.getBoolean("api_key_configured", false));
            assertFalse(entry.has("api_key"), "api_key field must be absent from API responses");
        } finally {
            provider.stop(0);
        }
    }

    @Test
    void providerConnectionTest() throws Exception {
        HttpServer provider = MockServers.openAiProvider("sk-ok", List.of("alpha", "beta"));
        try {
            manager.addProvider("mock", "openai-compatible", "Mock",
                    "http://127.0.0.1:" + provider.getAddress().getPort() + "/v1", "sk-ok", true);
            JsonObject result = manager.testProvider("mock");
            assertTrue(result.getBoolean("ok", false), result.toString());
            assertEquals(2, result.getArray("models").size());
            JsonObject health = client.getJson("/api/v1/providers/mock/health");
            assertTrue(health.getBoolean("connected", false));
        } finally {
            provider.stop(0);
        }
    }

    @Test
    void routerPrefersLocalModelsAndNeverSwitchesSilently() throws Exception {
        importTinyEvo("tiny-evo");
        HttpServer provider = MockServers.openAiProvider("sk-ok", List.of("tiny-evo"));
        try {
            manager.addProvider("mock", "openai-compatible", "Mock",
                    "http://127.0.0.1:" + provider.getAddress().getPort() + "/v1", "sk-ok", true);
            // Local model wins even though a provider exposes the same name.
            var route = manager.router().resolve("tiny-evo");
            assertEquals(ai.enr.manager.router.ModelRouter.Route.Target.LOCAL, route.target());
            // Explicit prefix routes to the provider.
            var cloud = manager.router().resolve("mock:tiny-evo");
            assertEquals(ai.enr.manager.router.ModelRouter.Route.Target.PROVIDER, cloud.target());
            assertEquals("mock", cloud.provider().id());
        } finally {
            provider.stop(0);
        }
        // Unknown models fail loudly.
        assertThrows(IllegalArgumentException.class, () -> manager.router().resolve("no-such-model"));
    }

    @Test
    void ambiguousRoutingFailsLoudly() throws Exception {
        HttpServer a = MockServers.openAiProvider("sk-a", List.of("shared-model"));
        HttpServer b = MockServers.openAiProvider("sk-b", List.of("shared-model"));
        try {
            manager.addProvider("pa", "openai-compatible", "A",
                    "http://127.0.0.1:" + a.getAddress().getPort() + "/v1", "sk-a", true);
            manager.addProvider("pb", "openai-compatible", "B",
                    "http://127.0.0.1:" + b.getAddress().getPort() + "/v1", "sk-b", true);
            var error = assertThrows(IllegalArgumentException.class,
                    () -> manager.router().resolve("shared-model"));
            assertTrue(error.getMessage().contains("ambiguous"), error.getMessage());
        } finally {
            a.stop(0);
            b.stop(0);
        }
    }

    @Test
    void downloadCancelStopsTransfer() throws Exception {
        byte[] payload = new byte[2_000_000];
        java.util.Arrays.fill(payload, (byte) 7);
        HttpServer slow = MockServers.slowFileServer(payload, 40);
        try {
            String jobId = manager.downloads().start("slow-test",
                    "http://127.0.0.1:" + slow.getAddress().getPort() + "/slow",
                    tempDir.resolve("slow.bin"), "", null);
            Thread.sleep(300);
            assertTrue(manager.downloads().cancel(jobId));
            long deadline = System.currentTimeMillis() + 10_000;
            var job = manager.downloads().get(jobId);
            while (job.isPresent() && job.get().status() == ai.enr.manager.download.DownloadManager.Status.RUNNING
                    && System.currentTimeMillis() < deadline) {
                Thread.sleep(100);
                job = manager.downloads().get(jobId);
            }
            assertNotEquals(ai.enr.manager.download.DownloadManager.Status.RUNNING,
                    manager.downloads().get(jobId).orElseThrow().status());
        } finally {
            slow.stop(0);
        }
    }

    @Test
    void adminAuthProtectsManagementEndpoints() throws Exception {
        server.close();
        manager.close();
        EnrConfig config = EnrConfig.builder()
                .host("127.0.0.1").port(0)
                .dataDir(tempDir.resolve("data2"))
                .modelsDir(tempDir.resolve("models2"))
                .logDir(tempDir.resolve("logs2"))
                .authToken("secret-token")
                .build();
        manager = new EnrManager(config);
        server = new EnrServer(manager, "127.0.0.1", 0);
        server.start();
        String securedBase = "http://127.0.0.1:" + server.port();

        HttpResponse<String> denied = http.send(HttpRequest.newBuilder()
                .uri(URI.create(securedBase + "/api/v1/providers")).GET().build(),
                HttpResponse.BodyHandlers.ofString());
        assertEquals(401, denied.statusCode());

        HttpResponse<String> allowed = http.send(HttpRequest.newBuilder()
                .uri(URI.create(securedBase + "/api/v1/providers"))
                .header("Authorization", "Bearer secret-token").GET().build(),
                HttpResponse.BodyHandlers.ofString());
        assertEquals(200, allowed.statusCode());

        HttpResponse<String> publicHealth = http.send(HttpRequest.newBuilder()
                .uri(URI.create(securedBase + "/api/v1/health")).GET().build(),
                HttpResponse.BodyHandlers.ofString());
        assertEquals(200, publicHealth.statusCode());
    }

    @Test
    void systemdUnitIsProductionReady() {
        String unit = SystemdSupport.unitFile("/usr/bin/java", "/opt/enr/enr.jar",
                "/etc/enr/enr.yaml", "enr", "enr", "/var/log/enr");
        assertTrue(unit.contains("Description=ENR Server"));
        assertTrue(unit.contains("ExecStart=/usr/bin/java --enable-native-access=ALL-UNNAMED -jar /opt/enr/enr.jar serve -c /etc/enr/enr.yaml"));
        assertTrue(unit.contains("Restart=on-failure"));
        assertTrue(unit.contains("KillSignal=SIGTERM"));
        assertTrue(unit.contains("WantedBy=multi-user.target"));
        assertTrue(unit.contains("User=enr"));
    }

    @Test
    void systemConfigGeneration() throws Exception {
        Path configPath = tempDir.resolve("etc/enr.yaml");
        Path written = SystemdSupport.ensureSystemConfig(configPath,
                tempDir.resolve("var/models"), tempDir.resolve("var/log"));
        assertTrue(Files.exists(written));
        String content = Files.readString(written);
        assertTrue(content.contains("host: 0.0.0.0"));
        assertTrue(content.contains("models_dir"));
    }

    @Test
    void chatStreamUnifiedEndpoint() throws Exception {
        importTinyEvo("tiny-evo");
        HttpResponse<java.io.InputStream> response = http.send(HttpRequest.newBuilder()
                        .uri(URI.create(base + "/api/v1/chat"))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(
                                "{\"model\":\"tiny-evo\",\"stream\":true,"
                                        + "\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}"))
                        .build(),
                HttpResponse.BodyHandlers.ofInputStream());
        assertEquals(200, response.statusCode());
        boolean sawContent = false, sawDone = false;
        try (var reader = new java.io.BufferedReader(new java.io.InputStreamReader(response.body()))) {
            String line;
            StringBuilder data = new StringBuilder();
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("data: ")) {
                    data.append(line.substring(6)).append('\n');
                } else if (line.isEmpty() && data.length() > 0) {
                    JsonObject chunk = Json.parseObject(data.toString());
                    if (!chunk.getString("content", "").isEmpty()) {
                        sawContent = true;
                    }
                    if (chunk.getBoolean("done", false)) {
                        sawDone = true;
                    }
                    data.setLength(0);
                }
            }
        }
        assertTrue(sawContent, "no streamed content");
        assertTrue(sawDone, "no final chunk");
    }

    @Test
    void hardwareReportIsGenuine() {
        JsonObject diagnostics = manager.diagnostics();
        JsonObject hardware = diagnostics.getObject("hardware");
        assertNotNull(hardware);
        assertTrue(hardware.getObject("cpu").getInt("processors", 0) > 0);
        assertTrue(hardware.getObject("ram").has("jvm_heap_used_bytes"));
        assertTrue(hardware.has("gpus"));
    }

    @Test
    void searchCatalogReachesConfiguredRegistry() throws Exception {
        HttpServer registry = MockServers.ollamaRegistry(
                FIXTURES.resolve("tiny-evo.gguf"), "search-model", "latest");
        try {
            var config = EnrConfig.builder()
                    .dataDir(tempDir.resolve("data3"))
                    .modelsDir(tempDir.resolve("models3"))
                    .logDir(tempDir.resolve("logs3"))
                    .ollamaRegistryUrl("http://127.0.0.1:" + registry.getAddress().getPort())
                    .ollamaSearchUrl("http://127.0.0.1:" + registry.getAddress().getPort())
                    .huggingFaceApiUrl("http://127.0.0.1:" + registry.getAddress().getPort())
                    .build();
            server.close();
            manager.close();
            manager = new EnrManager(config);
            server = new EnrServer(manager, "127.0.0.1", 0);
            server.start();
            base = "http://127.0.0.1:" + server.port();
            client = new EnrClient(base);
            JsonObject search = client.getJson("/api/v1/catalog/search?q=search");
            var models = search.getArray("models");
            assertNotNull(models);
            List<String> ids = new ArrayList<>();
            for (int i = 0; i < models.size(); i++) {
                ids.add(models.getObject(i).getString("id", ""));
            }
            assertTrue(ids.contains("search-model"), "search must find the mock Ollama model: " + ids);
        } finally {
            registry.stop(0);
        }
    }

    private static JsonObject obj(String key, String value) {
        JsonObject body = new JsonObject();
        body.put(key, value);
        return body;
    }
}
