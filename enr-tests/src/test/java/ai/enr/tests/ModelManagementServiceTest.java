package ai.enr.tests;

import ai.enr.core.json.Json;
import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonObject;
import ai.enr.manager.models.DownloadJob;
import ai.enr.manager.models.DownloadLayer;
import ai.enr.manager.models.DownloadProgress;
import ai.enr.manager.models.DownloadState;
import ai.enr.manager.models.DownloadStatus;
import ai.enr.manager.models.ModelDescriptor;
import ai.enr.manager.models.ModelManagementService;
import ai.enr.manager.models.ModelProviderConfig;
import ai.enr.tests.modelmgmt.StubModelProvider;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * ModelManagementService lifecycle: provider discovery, job execution,
 * duplicate prevention, cancellation, failures, concurrency, removal and
 * durable job status. Uses the scripted stub provider — no network involved.
 */
class ModelManagementServiceTest {

    @TempDir
    Path tempDir;

    ModelManagementService service;
    StubModelProvider stub;

    @BeforeEach
    void setUp() {
        stub = StubModelProvider.instance("stub");
        stub.reset();
        service = new ModelManagementService(
                Map.of("stub", new ModelProviderConfig("stub", "", 0, 0, 0, Map.of())),
                tempDir.resolve("model-jobs.json"), null, getClass().getClassLoader());
    }

    @AfterEach
    void tearDown() {
        service.close();
    }

    private static DownloadStatus awaitTerminal(ModelManagementService service, String jobId,
                                                long timeoutMs) throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (System.currentTimeMillis() < deadline) {
            Optional<DownloadStatus> status = service.getDownloadStatus(jobId);
            if (status.isPresent() && status.get().terminal()) {
                return status.get();
            }
            Thread.sleep(20);
        }
        return service.getDownloadStatus(jobId).orElse(null);
    }

    @Test
    void providerDiscovery_registersConfiguredProviders() {
        var providers = service.listProviders();
        assertEquals(1, providers.size());
        var provider = providers.get(0);
        assertEquals("stub", provider.id());
        assertEquals("Stub Provider stub", provider.displayName());
        assertTrue(provider.capabilities().containsAll(Set.of(
                ai.enr.manager.models.ProviderCapability.MODEL_PULL,
                ai.enr.manager.models.ProviderCapability.MODEL_LIST,
                ai.enr.manager.models.ProviderCapability.MODEL_REMOVE)));
    }

    @Test
    void pullLifecycle_completesWithProgress() throws Exception {
        stub.script(List.of(
                DownloadProgress.of(DownloadState.DOWNLOADING, 1000, 250, List.of(
                        new DownloadLayer("sha256:a", 400, 250, "pulling"))),
                DownloadProgress.of(DownloadState.DOWNLOADING, 1000, 600, List.of(
                        new DownloadLayer("sha256:a", 400, 400, "pulling"),
                        new DownloadLayer("sha256:b", 600, 200, "pulling"))),
                new DownloadProgress(DownloadState.COMPLETED, 1000, 1000, "", List.of(
                        new DownloadLayer("sha256:a", 400, 400, "success"),
                        new DownloadLayer("sha256:b", 600, 600, "success")))));
        DownloadJob job = service.pullModel("stub", "qwen3-vl:30b");
        assertNotNull(job.jobId());
        assertEquals("stub", job.providerId());
        assertEquals("qwen3-vl:30b", job.modelId());
        assertEquals(DownloadState.QUEUED, job.state());
        DownloadStatus status = awaitTerminal(service, job.jobId(), 5000);
        assertNotNull(status, "job must reach a terminal state");
        assertEquals(DownloadState.COMPLETED, status.state());
        assertEquals(100, status.percent());
        assertEquals(1000, status.totalBytes());
        assertEquals(1000, status.completedBytes());
        assertEquals(2, status.layers().size());
        assertTrue(status.completedAt() > 0);
    }

    @Test
    void duplicatePull_returnsExistingJob() throws Exception {
        stub.endlessLoop(20);
        DownloadJob first = service.pullModel("stub", "dup-model");
        stub.awaitPullStarted();
        DownloadJob second = service.pullModel("stub", "dup-model");
        assertEquals(first.jobId(), second.jobId(), "duplicate pull must reuse the active job");
        assertTrue(service.cancelDownload(first.jobId()));
        awaitTerminal(service, first.jobId(), 5000);
    }

    @Test
    void cancelDownload_marksCancelled() throws Exception {
        stub.endlessLoop(20);
        DownloadJob job = service.pullModel("stub", "cancel-model");
        stub.awaitPullStarted();
        assertTrue(service.cancelDownload(job.jobId()));
        DownloadStatus status = awaitTerminal(service, job.jobId(), 5000);
        assertNotNull(status);
        assertEquals(DownloadState.CANCELLED, status.state());
    }

    @Test
    void pullFailure_marksFailedWithError() throws Exception {
        stub.failWith("registry exploded");
        DownloadJob job = service.pullModel("stub", "bad-model");
        DownloadStatus status = awaitTerminal(service, job.jobId(), 5000);
        assertNotNull(status);
        assertEquals(DownloadState.FAILED, status.state());
        assertEquals("registry exploded", status.error());
    }

    @Test
    void concurrentDownloads_runIndependently() throws Exception {
        stub.script(List.of(
                DownloadProgress.of(DownloadState.DOWNLOADING, 500, 100, List.of()),
                new DownloadProgress(DownloadState.COMPLETED, 500, 500, "", List.of())));
        DownloadJob a = service.pullModel("stub", "model-a");
        DownloadJob b = service.pullModel("stub", "model-b");
        assertFalse(a.jobId().equals(b.jobId()), "different models get different jobs");
        DownloadStatus sa = awaitTerminal(service, a.jobId(), 5000);
        DownloadStatus sb = awaitTerminal(service, b.jobId(), 5000);
        assertNotNull(sa);
        assertNotNull(sb);
        assertEquals(DownloadState.COMPLETED, sa.state());
        assertEquals(DownloadState.COMPLETED, sb.state());
    }

    @Test
    void alreadyInstalledModel_completesImmediately() throws Exception {
        stub.addInstalled("qwen3-vl:30b");
        DownloadJob job = service.pullModel("stub", "qwen3-vl:30b");
        DownloadStatus status = awaitTerminal(service, job.jobId(), 5000);
        assertNotNull(status);
        assertEquals(DownloadState.COMPLETED, status.state());
        assertEquals("already installed", status.operation());
        assertEquals(0, stub.pullCount, "no native pull must be started for an installed model");
    }

    @Test
    void listInstalledModels_returnsProviderModels() {
        stub.addInstalled("llama3:8b");
        List<ModelDescriptor> installed = service.listInstalledModels("stub");
        assertEquals(1, installed.size());
        assertEquals("llama3:8b", installed.get(0).id());
    }

    @Test
    void removeModel_removesFromProvider() {
        stub.addInstalled("llama3:8b");
        service.removeModel("stub", "llama3:8b");
        assertTrue(service.listInstalledModels("stub").isEmpty());
        assertThrows(IllegalArgumentException.class,
                () -> service.removeModel("stub", "llama3:8b"));
    }

    @Test
    void unknownProvider_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> service.pullModel("missing", "model"));
        assertThrows(IllegalArgumentException.class,
                () -> service.listInstalledModels("missing"));
    }

    @Test
    void blankModelId_throws() {
        assertThrows(IllegalArgumentException.class, () -> service.pullModel("stub", "  "));
    }

    @Test
    void persistence_recoversUnfinishedJobsAsFailed() throws Exception {
        Path stateFile = tempDir.resolve("recover-jobs.json");
        JsonObject root = new JsonObject();
        JsonArray arr = new JsonArray();
        JsonObject job = new JsonObject();
        job.put("job_id", "old-job");
        job.put("provider", "stub");
        job.put("model", "model");
        job.put("status", "DOWNLOADING");
        job.put("operation", "");
        job.put("total_bytes", 1000);
        job.put("completed_bytes", 250);
        job.put("percent", 25);
        job.put("started_at", System.currentTimeMillis() - 60_000);
        job.put("updated_at", System.currentTimeMillis() - 30_000);
        job.put("completed_at", 0);
        job.put("error", "");
        job.put("layers", new JsonArray());
        arr.add(job);
        root.put("downloads", arr);
        Files.writeString(stateFile, Json.write(root, false));
        try (ModelManagementService recovered = new ModelManagementService(
                Map.of("stub", new ModelProviderConfig("stub", "", 0, 0, 0, Map.of())),
                stateFile, null, getClass().getClassLoader())) {
            DownloadStatus status = recovered.getDownloadStatus("old-job").orElseThrow();
            assertEquals(DownloadState.FAILED, status.state());
            assertEquals("interrupted by restart", status.error());
        }
    }

    @Test
    void persistence_savesCompletedJobs() throws Exception {
        stub.script(List.of(
                new DownloadProgress(DownloadState.COMPLETED, 10, 10, "", List.of())));
        DownloadJob job = service.pullModel("stub", "persist-model");
        awaitTerminal(service, job.jobId(), 5000);
        assertTrue(Files.isRegularFile(tempDir.resolve("model-jobs.json")),
                "job state must be persisted");
        try (ModelManagementService recovered = new ModelManagementService(
                Map.of("stub", new ModelProviderConfig("stub", "", 0, 0, 0, Map.of())),
                tempDir.resolve("model-jobs.json"), null, getClass().getClassLoader())) {
            DownloadStatus status = recovered.getDownloadStatus(job.jobId()).orElseThrow();
            assertEquals(DownloadState.COMPLETED, status.state());
        }
    }

    @Test
    void listeners_areNotifiedOfTransitions() throws Exception {
        java.util.concurrent.atomic.AtomicInteger terminalCalls = new java.util.concurrent.atomic.AtomicInteger();
        service.addListener(new ModelManagementService.Listener() {
            @Override
            public void onChanged(DownloadStatus status) {
                assertSame(service.getDownloadStatus(status.jobId()).orElse(null), status);
            }

            @Override
            public void onTerminal(DownloadStatus status) {
                terminalCalls.incrementAndGet();
                assertTrue(status.terminal());
            }
        });
        stub.script(List.of(
                new DownloadProgress(DownloadState.COMPLETED, 1, 1, "", List.of())));
        DownloadJob job = service.pullModel("stub", "listener-model");
        awaitTerminal(service, job.jobId(), 5000);
        assertEquals(1, terminalCalls.get(), "terminal listener must fire exactly once");
    }
}
