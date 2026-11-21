package ai.enr.tests;

import ai.enr.manager.models.DownloadProgress;
import ai.enr.manager.models.DownloadState;
import ai.enr.manager.models.ModelDescriptor;
import ai.enr.manager.models.ollama.OllamaModelProvider;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Ollama model provider against a hermetic mock Ollama HTTP API: tags, show,
 * pull streaming (layers, changing totals, errors, truncation, cancellation,
 * read timeout), delete and connectivity.
 */
class OllamaModelProviderTest {

    MockServers.OllamaMock mock;
    OllamaModelProvider provider;
    List<DownloadProgress> events;

    @BeforeEach
    void setUp() throws Exception {
        mock = new MockServers.OllamaMock();
        provider = new OllamaModelProvider("ollama", mock.baseUrl(), 2000, 2000, 0);
        events = new ArrayList<>();
    }

    @AfterEach
    void tearDown() {
        provider.close();
        mock.close();
    }

    private void pull(String model) throws Exception {
        provider.pull(model, () -> false, events::add);
    }

    @Test
    void ping_reportsConnectivity() {
        assertTrue(provider.ping());
        try (OllamaModelProvider dead = new OllamaModelProvider("ollama",
                "http://127.0.0.1:1", 200, 200, 0)) {
            assertFalse(dead.ping(), "unreachable endpoint must report false");
        }
    }

    @Test
    void listInstalled_parsesModelsAndDetails() throws Exception {
        mock.installed.add("llama3:8b");
        mock.installed.add("qwen3-vl:30b");
        List<ModelDescriptor> models = provider.listInstalled();
        assertEquals(2, models.size());
        ModelDescriptor first = models.get(0);
        assertEquals("llama3:8b", first.id());
        assertTrue(first.installed());
        assertTrue(first.sizeBytes() > 0);
        assertTrue(first.digest().startsWith("sha256:"));
        assertEquals("llama", first.details().get("family"));
        assertEquals("Q4_K_M", first.details().get("quantization_level"));
        assertTrue(first.modifiedAt() > 0);
    }

    @Test
    void describe_returnsModelMetadata() throws Exception {
        ModelDescriptor descriptor = provider.describe("llama3:8b");
        assertEquals("llama3:8b", descriptor.id());
        assertEquals("llama", descriptor.details().get("architecture"));
        assertEquals("8192", descriptor.details().get("context_length"));
    }

    @Test
    void pull_streamsLayersAndCompletes() throws Exception {
        mock.pullScript = MockServers.OllamaMock.standardPullScript(
                "qwen3-vl:30b", "sha256:aaaaaaaaaaaaaaaaaaaa", "sha256:bbbbbbbbbbbbbbbbbbbb");
        pull("qwen3-vl:30b");
        DownloadProgress last = events.get(events.size() - 1);
        assertEquals(DownloadState.COMPLETED, last.state());
        assertEquals(1000, last.totalBytes(), "aggregate total = sum of layer totals");
        assertEquals(1000, last.completedBytes(), "aggregate completed = sum of layer completions");
        assertEquals(2, last.layers().size());
        assertTrue(events.stream().anyMatch(e -> e.state() == DownloadState.CONNECTING),
                "manifest phase must be reported as CONNECTING");
        assertTrue(events.stream().anyMatch(e -> e.state() == DownloadState.VERIFYING),
                "verify/write phase must be reported as VERIFYING");
        assertEquals(1, mock.pulledRequests);
    }

    @Test
    void pull_changingTotals_growAggregate() throws Exception {
        mock.pullScript = MockServers.OllamaMock.standardPullScript(
                "model", "sha256:cccccccccccccccccccc", "sha256:dddddddddddddddddddd");
        pull("model");
        List<Long> totals = events.stream().map(DownloadProgress::totalBytes)
                .filter(total -> total > 0).toList();
        assertEquals(400, totals.get(0), "first reported total covers the first layer only");
        assertEquals(1000, totals.get(totals.size() - 1), "totals must grow as layers appear");
    }

    @Test
    void pull_errorLine_failsWithOllamaError() {
        mock.pullScript = List.of("{\"status\":\"pulling manifest\"}",
                "{\"error\":\"pull model manifest: file does not exist\"}");
        Exception error = assertThrows(Exception.class, () -> pull("missing-model"));
        assertTrue(error.getMessage().contains("file does not exist"), error.getMessage());
    }

    @Test
    void pull_httpFailure_failsWithStatus() {
        mock.failPullRequest = true;
        Exception error = assertThrows(Exception.class, () -> pull("missing-model"));
        assertTrue(error.getMessage().contains("404"), error.getMessage());
    }

    @Test
    void pull_truncatedStream_fails() {
        mock.pullScript = List.of("{\"status\":\"pulling manifest\"}",
                "{\"status\":\"pulling 3f4c8abc\",\"digest\":\"sha256:ee\",\"total\":10,\"completed\":4}");
        Exception error = assertThrows(Exception.class, () -> pull("broken-model"));
        assertTrue(error.getMessage().contains("success"), error.getMessage());
    }

    @Test
    void pull_cancellation_returnsEarly() throws Exception {
        mock.pullScript = MockServers.OllamaMock.standardPullScript(
                "cancel-model", "sha256:ffffffffffffffffffff", "sha256:eeeeeeeeeeeeeeeeeeee");
        mock.lineDelayMs = 30;
        AtomicBoolean cancel = new AtomicBoolean(false);
        AtomicReference<Exception> failure = new AtomicReference<>();
        Thread worker = new Thread(() -> {
            try {
                provider.pull("cancel-model", cancel::get, events::add);
            } catch (Exception e) {
                failure.set(e);
            }
        });
        worker.start();
        Thread.sleep(100);
        cancel.set(true);
        worker.join(5000);
        assertFalse(worker.isAlive(), "pull must return promptly after cancellation");
        if (failure.get() != null) {
            fail("cancelled pull must not report an error: " + failure.get().getMessage());
        }
        assertTrue(events.stream().anyMatch(
                e -> e.state() == DownloadState.CANCEL_REQUESTED), "cancellation must be reported");
    }

    @Test
    void pull_readTimeout_abortsStalledStream() throws Exception {
        mock.pullScript = List.of("{\"status\":\"pulling manifest\"}");
        mock.lineDelayMs = 30_000;
        try (OllamaModelProvider impatient = new OllamaModelProvider(
                "ollama", mock.baseUrl(), 2000, 300, 0)) {
            long start = System.currentTimeMillis();
            assertThrows(Exception.class, () -> impatient.pull("stalled-model", () -> false, events::add));
            assertTrue(System.currentTimeMillis() - start < 5000,
                    "read timeout must abort the stalled stream quickly");
        }
    }

    @Test
    void pull_downloadTimeout_abortsLongPull() throws Exception {
        mock.pullScript = MockServers.OllamaMock.standardPullScript(
                "slow-model", "sha256:11111111111111111111", "sha256:22222222222222222222");
        mock.lineDelayMs = 200;
        try (OllamaModelProvider limited = new OllamaModelProvider(
                "ollama", mock.baseUrl(), 2000, 5000, 400)) {
            assertThrows(Exception.class, () -> limited.pull("slow-model", () -> false, events::add));
        }
    }

    @Test
    void remove_removesAndReportsMissingModels() throws Exception {
        mock.installed.add("doomed-model");
        provider.remove("doomed-model");
        assertTrue(mock.installed.isEmpty());
        assertThrows(IllegalArgumentException.class, () -> provider.remove("doomed-model"));
    }
}
