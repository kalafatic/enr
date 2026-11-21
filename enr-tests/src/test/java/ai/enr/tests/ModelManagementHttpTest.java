package ai.enr.tests;

import ai.enr.client.EnrClient;
import ai.enr.core.json.Json;
import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonObject;
import ai.enr.manager.EnrConfig;
import ai.enr.manager.EnrManager;
import ai.enr.server.EnrServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Model management HTTP API (scenario A/C wire): providers, pull with job
 * monitoring, installed models, removal, cancellation — all against a mock
 * Ollama API so no real Ollama installation is required.
 */
class ModelManagementHttpTest {

    @TempDir
    Path tempDir;

    MockServers.OllamaMock ollama;
    EnrManager manager;
    EnrServer server;
    EnrClient client;

    @BeforeEach
    void setUp() throws Exception {
        ollama = new MockServers.OllamaMock();
        ollama.installed.add("llama3:8b");
        ollama.pullScript = MockServers.OllamaMock.standardPullScript(
                "qwen3-vl:30b", "sha256:aaaa1111aaaa1111aaaa", "sha256:bbbb2222bbbb2222bbbb");
        JsonObject providers = new JsonObject();
        providers.put("ollama.endpoint", ollama.baseUrl());
        providers.put("ollama.connect_timeout_ms", 2000);
        providers.put("ollama.read_timeout_ms", 2000);
        EnrConfig config = EnrConfig.builder()
                .host("127.0.0.1")
                .port(0)
                .dataDir(tempDir.resolve("data"))
                .modelsDir(tempDir.resolve("models"))
                .logDir(tempDir.resolve("logs"))
                .modelProviders(providers)
                .build();
        manager = new EnrManager(config);
        server = new EnrServer(manager, "127.0.0.1", 0);
        server.start();
        client = new EnrClient("http://127.0.0.1:" + server.port());
    }

    @AfterEach
    void tearDown() {
        server.close();
        manager.close();
        ollama.close();
    }

    private JsonObject awaitTerminal(EnrClient client, String jobId, long timeoutMs)
            throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (System.currentTimeMillis() < deadline) {
            JsonObject status = client.downloadStatus(jobId);
            String state = status.getString("status", "");
            if ("COMPLETED".equals(state) || "FAILED".equals(state) || "CANCELLED".equals(state)) {
                return status;
            }
            Thread.sleep(30);
        }
        return client.downloadStatus(jobId);
    }

    @Test
    void providers_endpointListsOllamaWithCapabilities() {
        JsonObject body = client.modelProviders();
        JsonArray providers = body.getArray("providers");
        assertNotNull(providers);
        JsonObject ollama = null;
        for (int i = 0; i < providers.size(); i++) {
            if ("ollama".equals(providers.getObject(i).getString("id", ""))) {
                ollama = providers.getObject(i);
            }
        }
        assertNotNull(ollama, "ollama provider must be registered");
        assertTrue(ollama.getBoolean("reachable", false), "mock Ollama must be reachable");
        JsonArray capabilities = ollama.getArray("capabilities");
        assertNotNull(capabilities);
        StringBuilder caps = new StringBuilder();
        for (int i = 0; i < capabilities.size(); i++) {
            caps.append(capabilities.getString(i)).append(' ');
        }
        assertTrue(caps.toString().contains("MODEL_PULL"));
        assertTrue(caps.toString().contains("MODEL_REMOVE"));
        assertTrue(caps.toString().contains("MODEL_LIST"));
    }

    @Test
    void pull_scenarioA_monitorUntilCompleted() throws InterruptedException {
        EnrClient.ModelJob job = client.pullModel("ollama", "qwen3-vl:30b");
        assertNotNull(job.jobId());
        assertEquals("ollama", job.provider());
        assertEquals("qwen3-vl:30b", job.model());
        assertEquals("QUEUED", job.status());
        JsonObject status = awaitTerminal(client, job.jobId(), 8000);
        assertEquals("COMPLETED", status.getString("status", ""));
        assertEquals(100, status.getInt("percent", 0));
        assertEquals(1000, status.getLong("total_bytes", -1));
        assertEquals(1000, status.getLong("completed_bytes", 0));
        assertEquals("qwen3-vl:30b", status.getString("model", ""));
        JsonArray layers = status.getArray("layers");
        assertNotNull(layers);
        assertEquals(2, layers.size());
    }

    @Test
    void downloads_listContainsActiveJob() throws InterruptedException {
        EnrClient.ModelJob job = client.pullModel("ollama", "qwen3-vl:30b");
        JsonObject list = client.getJson("/api/v1/models/downloads");
        JsonArray downloads = list.getArray("downloads");
        assertNotNull(downloads);
        boolean found = false;
        for (int i = 0; i < downloads.size(); i++) {
            if (job.jobId().equals(downloads.getObject(i).getString("job_id", ""))) {
                found = true;
            }
        }
        assertTrue(found, "job must be listed in GET /api/v1/models/downloads");
        awaitTerminal(client, job.jobId(), 8000);
    }

    @Test
    void installedModels_reflectProviderStore() {
        JsonObject body = client.modelInstalled("ollama");
        JsonArray models = body.getArray("models");
        assertNotNull(models);
        assertEquals(1, models.size());
        assertEquals("llama3:8b", models.getObject(0).getString("id", ""));
    }

    @Test
    void removeModel_scenarioD_removesFromProvider() {
        JsonObject removed = client.removeModel("ollama", "llama3:8b");
        assertTrue(removed.getBoolean("removed", false));
        JsonObject body = client.modelInstalled("ollama");
        assertEquals(0, body.getArray("models").size());
    }

    @Test
    void cancelDownload_abortsActivePull() throws Exception {
        ollama.pullScript = MockServers.OllamaMock.standardPullScript(
                "slow-model", "sha256:cccc3333cccc3333cccc", "sha256:dddd4444dddd4444dddd");
        ollama.lineDelayMs = 100;
        EnrClient.ModelJob job = client.pullModel("ollama", "slow-model");
        Thread.sleep(150);
        JsonObject cancel = client.cancelDownload(job.jobId());
        assertTrue(cancel.getBoolean("cancelled", false));
        JsonObject status = awaitTerminal(client, job.jobId(), 8000);
        assertTrue("CANCELLED".equals(status.getString("status", ""))
                || "CANCEL_REQUESTED".equals(status.getString("status", "")),
                "job must end cancelled, was " + status.getString("status", ""));
    }

    @Test
    void unknownProvider_returns404() {
        try {
            client.modelInstalled("nope");
            org.junit.jupiter.api.Assertions.fail("unknown provider must fail");
        } catch (EnrClient.EnrException e) {
            assertEquals(404, e.status());
            assertEquals("PROVIDER_NOT_FOUND", e.code());
        }
    }

    @Test
    void unknownJob_returns404() {
        try {
            client.downloadStatus("no-such-job");
            org.junit.jupiter.api.Assertions.fail("unknown job must fail");
        } catch (EnrClient.EnrException e) {
            assertEquals(404, e.status());
            assertEquals("DOWNLOAD_NOT_FOUND", e.code());
        }
    }

    @Test
    void pullMissingModel_returns400() {
        try {
            JsonObject req = new JsonObject();
            client.postJson("/api/v1/models/providers/ollama/pull", req);
            org.junit.jupiter.api.Assertions.fail("missing model must fail");
        } catch (EnrClient.EnrException e) {
            assertEquals(400, e.status());
            assertEquals("INVALID_REQUEST", e.code());
        }
    }

    @Test
    void availableModels_returnsEmptyForOllama() {
        JsonObject body = client.modelAvailable("ollama");
        assertNotNull(body.getArray("models"));
        assertEquals(0, body.getArray("models").size(),
                "Ollama has no discovery capability; availability must be empty");
    }

    @Test
    void rawErrorEnvelope_isStructured() {
        try {
            client.getJson("/api/v1/models/providers/nope/installed");
            org.junit.jupiter.api.Assertions.fail("must fail");
        } catch (EnrClient.EnrException e) {
            assertEquals("PROVIDER_NOT_FOUND", e.code());
            assertFalse(e.getMessage().isBlank());
        }
    }

    @Test
    void describe_metadataAvailable() {
        JsonObject model = client.getJson(
                "/api/v1/models/providers/ollama/installed");
        assertNotNull(model.getArray("models"));
    }

    @Test
    void persistedJobs_surviveInManagerState() throws Exception {
        EnrClient.ModelJob job = client.pullModel("ollama", "qwen3-vl:30b");
        awaitTerminal(client, job.jobId(), 8000);
        assertTrue(java.nio.file.Files.isRegularFile(
                        tempDir.resolve("data").resolve("model-download-jobs.json")),
                "job status must be persisted to disk");
        JsonObject parsed = Json.parse(java.nio.file.Files.readString(
                tempDir.resolve("data").resolve("model-download-jobs.json")))
                instanceof JsonObject obj ? obj : new JsonObject();
        assertNotNull(parsed.getArray("downloads"));
    }
}
