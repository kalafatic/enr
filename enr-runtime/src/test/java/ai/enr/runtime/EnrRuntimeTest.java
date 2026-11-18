package ai.enr.runtime;

import ai.enr.inference.GenerationEngine;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class EnrRuntimeTest {

    private static RuntimeConfig config(Path modelsDir) {
        return RuntimeConfig.builder()
                .addModelDirectory(modelsDir)
                .maxConcurrentSessions(8)
                .defaultMaxTokens(16)
                .build();
    }

    @Test
    void loadsTwoModelsAndRunsConcurrentInferenceWithoutContamination() throws Exception {
        Path modelsDir = Path.of("src/test/resources/models").toAbsolutePath();
        EnrRuntime runtime = new EnrRuntime(config(modelsDir));

        ModelInstance tiny = runtime.loadModel(modelsDir.resolve("tiny-evo.gguf"));
        ModelInstance evo = runtime.loadModel(modelsDir.resolve("evo.gguf"));
        assertEquals(2, runtime.models().list().size());

        int workers = 8;
        ExecutorService pool = Executors.newFixedThreadPool(workers);
        CountDownLatch start = new CountDownLatch(1);
        List<Future<String>> futures = new ArrayList<>();
        for (int i = 0; i < workers; i++) {
            final int n = i;
            futures.add(pool.submit(() -> {
                start.await();
                String modelId = n % 2 == 0 ? "tiny-evo" : "evo";
                var result = runtime.infer(modelId, "token_4 token_5 token_6",
                        EnrRuntime.InferenceOptions.defaults(), null);
                return result.text() + "|" + result.reason();
            }));
        }
        start.countDown();
        List<String> results = new ArrayList<>();
        for (Future<String> future : futures) {
            results.add(future.get(60, TimeUnit.SECONDS));
        }
        pool.shutdown();

        assertEquals(workers, results.size());
        for (String result : results) {
            assertFalse(result.endsWith("ERROR"), "request failed: " + result);
        }
        assertTrue(runtime.metrics().totalRequests() >= workers);
        assertTrue(runtime.metrics().totalGeneratedTokens() > 0);
        assertEquals(2, runtime.models().list().size());
        runtime.shutdown();
    }

    @Test
    void sessionsAreIndependentAndIsolated() throws Exception {
        Path modelsDir = Path.of("src/test/resources/models").toAbsolutePath();
        EnrRuntime runtime = new EnrRuntime(config(modelsDir));
        runtime.loadModel(modelsDir.resolve("tiny-evo.gguf"));

        var session1 = runtime.createSession("tiny-evo");
        var session2 = runtime.createSession("tiny-evo");
        assertNotEquals(session1.id(), session2.id());
        assertNotSame(session1.kvCache(), session2.kvCache());

        String id1 = String.valueOf(session1.id());
        String id2 = String.valueOf(session2.id());
        var result1 = runtime.infer("tiny-evo", "token_4 token_5", sessionOptions(id1), null);
        var result2 = runtime.infer("tiny-evo", "token_7 token_8", sessionOptions(id2), null);
        assertNotNull(result1.text());
        assertNotNull(result2.text());
        assertEquals(2, runtime.sessions().liveCount());

        runtime.deleteSession(session1.id());
        runtime.deleteSession(session2.id());
        assertEquals(0, runtime.sessions().liveCount());
        runtime.shutdown();
    }

    @Test
    void refusesToUnloadModelWithActiveSessions() throws Exception {
        Path modelsDir = Path.of("src/test/resources/models").toAbsolutePath();
        EnrRuntime runtime = new EnrRuntime(config(modelsDir));
        runtime.loadModel(modelsDir.resolve("tiny-evo.gguf"));
        var session = runtime.createSession("tiny-evo");

        assertThrows(IllegalStateException.class, () -> runtime.unloadModel("tiny-evo", false));
        runtime.deleteSession(session.id());
        runtime.unloadModel("tiny-evo", false);
        assertEquals(0, runtime.models().list().size());
        runtime.shutdown();
    }

    @Test
    void unknownModelRejected() throws Exception {
        Path modelsDir = Path.of("src/test/resources/models").toAbsolutePath();
        EnrRuntime runtime = new EnrRuntime(config(modelsDir));
        assertThrows(IllegalArgumentException.class,
                () -> runtime.infer("no-such-model", "hi", EnrRuntime.InferenceOptions.defaults(), null));
        runtime.shutdown();
    }

    @Test
    void sessionLimitEnforced() throws Exception {
        Path modelsDir = Path.of("src/test/resources/models").toAbsolutePath();
        RuntimeConfig cfg = RuntimeConfig.builder()
                .addModelDirectory(modelsDir)
                .maxConcurrentSessions(2)
                .build();
        EnrRuntime runtime = new EnrRuntime(cfg);
        runtime.loadModel(modelsDir.resolve("tiny-evo.gguf"));
        runtime.createSession("tiny-evo");
        runtime.createSession("tiny-evo");
        assertThrows(IllegalStateException.class, () -> runtime.createSession("tiny-evo"));
        runtime.shutdown();
    }

    @Test
    void statusAndHealthReportRealValues() throws Exception {
        Path modelsDir = Path.of("src/test/resources/models").toAbsolutePath();
        EnrRuntime runtime = new EnrRuntime(config(modelsDir));
        runtime.loadModel(modelsDir.resolve("tiny-evo.gguf"));
        runtime.infer("tiny-evo", "token_4", EnrRuntime.InferenceOptions.defaults(), null);

        var status = runtime.status();
        assertTrue(status.getBoolean("running", false));
        assertEquals(1, status.getInt("models_loaded", 0));
        assertTrue(status.getInt("total_requests", 0) >= 1);
        assertTrue(runtime.healthChecks().isEmpty(), () -> String.join("; ", runtime.healthChecks()));
        assertFalse(runtime.devices().isEmpty());
        runtime.shutdown();
    }

    @Test
    void requestTimeoutCancelsSlowGeneration() throws Exception {
        Path modelsDir = Path.of("src/test/resources/models").toAbsolutePath();
        EnrRuntime runtime = new EnrRuntime(config(modelsDir));
        runtime.loadModel(modelsDir.resolve("tiny-evo.gguf"));

        var defaults = EnrRuntime.InferenceOptions.defaults();
        var options = new EnrRuntime.InferenceOptions(16, defaults.sampling(),
                defaults.stopTokenIds(), null, 1);
        var handle = runtime.inferStream("tiny-evo", "token_4 token_5", options, (tokenId, piece) -> {
            try {
                Thread.sleep(1500);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        var result = handle.future().get(20, TimeUnit.SECONDS);
        assertEquals(GenerationEngine.TerminationReason.CANCELLED, result.reason(),
                "generation should be cancelled by the 1s request timeout, got " + result.reason());
        assertEquals(0, runtime.sessions().liveCount(),
                "own session must be cleaned up after timeout cancellation");
        runtime.shutdown();
    }

    @Test
    void idleSessionsAreEvicted() throws Exception {
        Path modelsDir = Path.of("src/test/resources/models").toAbsolutePath();
        RuntimeConfig cfg = RuntimeConfig.builder()
                .addModelDirectory(modelsDir)
                .sessionIdleTimeoutSeconds(1)
                .build();
        EnrRuntime runtime = new EnrRuntime(cfg);
        runtime.loadModel(modelsDir.resolve("tiny-evo.gguf"));
        var session = runtime.createSession("tiny-evo");
        assertEquals(1, runtime.sessions().liveCount());

        Thread.sleep(1300);
        var evicted = runtime.sessions().evictIdle(runtime.models());
        assertTrue(evicted.contains(session.id()), "idle session should be evicted");
        assertEquals(0, runtime.sessions().liveCount());
        runtime.shutdown();
    }

    private static EnrRuntime.InferenceOptions sessionOptions(String sessionId) {
        var defaults = EnrRuntime.InferenceOptions.defaults();
        return new EnrRuntime.InferenceOptions(defaults.maxTokens(), defaults.sampling(),
                defaults.stopTokenIds(), sessionId, null);
    }
}
