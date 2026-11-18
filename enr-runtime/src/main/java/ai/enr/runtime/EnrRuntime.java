package ai.enr.runtime;

import ai.enr.compute.Backends;
import ai.enr.compute.ComputeBackend;
import ai.enr.compute.DeviceInfo;
import ai.enr.inference.GenerationEngine;
import ai.enr.inference.sample.SamplingConfig;
import ai.enr.inference.session.InferenceSession;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

/**
 * Central runtime facade: models, sessions, generation, cancellation,
 * devices and metrics.
 */
public final class EnrRuntime {

    public record InferenceOptions(
            int maxTokens,
            SamplingConfig sampling,
            Set<Integer> stopTokenIds,
            String sessionId,
            Integer timeoutSeconds) {

        public static InferenceOptions defaults() {
            return new InferenceOptions(256, SamplingConfig.defaults(), Set.of(), null, null);
        }
    }

    public record TextResult(
            String text,
            List<Integer> tokenIds,
            int promptTokens,
            int generatedTokens,
            GenerationEngine.TerminationReason reason,
            long elapsedMs) {
    }

    public record StreamHandle(
            String requestId,
            CompletableFuture<TextResult> future,
            InferenceSession session) {
    }

    private final RuntimeConfig config;
    private final ModelManager modelManager;
    private final SessionManager sessionManager;
    private final Metrics metrics = new Metrics();
    private final java.util.concurrent.ConcurrentHashMap<String, GenerationEngine.GenerationRequest> activeStreams =
            new java.util.concurrent.ConcurrentHashMap<>();
    private final java.util.concurrent.ScheduledExecutorService housekeeper;
    private volatile boolean running = true;

    public EnrRuntime(RuntimeConfig config) {
        this.config = config;
        this.modelManager = new ModelManager(config);
        this.sessionManager = new SessionManager(config);
        this.housekeeper = java.util.concurrent.Executors.newSingleThreadScheduledExecutor(r -> {
            Thread thread = new Thread(r, "enr-housekeeper");
            thread.setDaemon(true);
            return thread;
        });
        this.housekeeper.scheduleAtFixedRate(this::evictIdleResources, 30, 30,
                java.util.concurrent.TimeUnit.SECONDS);
    }

    private void evictIdleResources() {
        try {
            sessionManager.evictIdle(modelManager);
        } catch (RuntimeException e) {
            // housekeeping must never take the runtime down
        }
    }

    private java.util.concurrent.ScheduledFuture<?> armTimeout(InferenceSession session,
                                                               Integer timeoutSeconds) {
        int effective = timeoutSeconds != null && timeoutSeconds > 0
                ? timeoutSeconds : config.requestTimeoutSeconds();
        if (effective <= 0) {
            return null;
        }
        return housekeeper.schedule(() -> {
            try {
                session.cancel();
            } catch (RuntimeException e) {
                // session may already be gone
            }
        }, effective, java.util.concurrent.TimeUnit.SECONDS);
    }

    public RuntimeConfig config() {
        return config;
    }

    public ModelManager models() {
        return modelManager;
    }

    public SessionManager sessions() {
        return sessionManager;
    }

    public Metrics metrics() {
        return metrics;
    }

    public boolean isRunning() {
        return running;
    }

    public ModelInstance loadModel(Path path, String backend) throws Exception {
        return modelManager.load(path, backend);
    }

    public ModelInstance loadModel(Path path) throws Exception {
        return modelManager.load(path, config.preferredBackend());
    }

    public void unloadModel(String id, boolean force) {
        modelManager.unload(id, force);
    }

    public InferenceSession createSession(String modelId) {
        ModelInstance model = modelManager.require(modelId);
        return sessionManager.create(modelId, model);
    }

    public void deleteSession(String sessionId) {
        sessionManager.remove(sessionId, modelManager);
    }

    public TextResult infer(String modelId, String prompt, InferenceOptions options,
                            GenerationEngine.TokenListener listener) {
        ModelInstance model = modelManager.require(modelId);
        long start = System.currentTimeMillis();
        InferenceSession session = null;
        boolean ownSession = false;
        if (options.sessionId() != null) {
            session = sessionManager.require(options.sessionId());
            if (!session.modelId().equals(modelId)) {
                throw new IllegalArgumentException("Session " + options.sessionId()
                        + " belongs to model " + session.modelId() + ", not " + modelId);
            }
        } else {
            session = createSession(modelId);
            ownSession = true;
        }
        session.touch();

        int[] promptIds = model.tokenizer().encodeToArray(prompt);
        int maxTokens = options.maxTokens() > 0 ? options.maxTokens() : config.defaultMaxTokens();

        var request = new GenerationEngine.GenerationRequest(promptIds, maxTokens,
                options.sampling(), options.stopTokenIds());
        var timeoutTask = armTimeout(session, options.timeoutSeconds());
        try {
            var result = model.newGenerationEngine().generate(session, request, model.tokenizer(), listener);
            boolean failed = result.reason() == GenerationEngine.TerminationReason.ERROR;
            boolean cancelled = result.reason() == GenerationEngine.TerminationReason.CANCELLED;
            metrics.recordRequest(promptIds.length, result.generatedTokenIds().size(),
                    result.elapsedMs(), failed, cancelled);
            return new TextResult(result.text(), result.generatedTokenIds(), promptIds.length,
                    result.generatedTokenIds().size(), result.reason(), result.elapsedMs());
        } finally {
            if (timeoutTask != null) {
                timeoutTask.cancel(false);
            }
            if (ownSession) {
                sessionManager.remove(session.id(), modelManager);
            }
        }
    }

    public StreamHandle inferStream(String modelId, String prompt, InferenceOptions options,
                                    GenerationEngine.TokenListener listener) {
        ModelInstance model = modelManager.require(modelId);
        InferenceSession session = options.sessionId() != null
                ? sessionManager.require(options.sessionId())
                : createSession(modelId);
        if (options.sessionId() != null && !session.modelId().equals(modelId)) {
            throw new IllegalArgumentException("Session belongs to model " + session.modelId());
        }
        session.touch();

        String requestId = UUID.randomUUID().toString();
        int[] promptIds = model.tokenizer().encodeToArray(prompt);
        int maxTokens = options.maxTokens() > 0 ? options.maxTokens() : config.defaultMaxTokens();
        var request = new GenerationEngine.GenerationRequest(promptIds, maxTokens,
                options.sampling(), options.stopTokenIds());
        activeStreams.put(requestId, request);
        var timeoutTask = armTimeout(session, options.timeoutSeconds());

        CompletableFuture<TextResult> future = CompletableFuture.supplyAsync(() -> {
            try {
                var result = model.newGenerationEngine().generate(session, request, model.tokenizer(), listener);
                boolean failed = result.reason() == GenerationEngine.TerminationReason.ERROR;
                boolean cancelled = result.reason() == GenerationEngine.TerminationReason.CANCELLED;
                metrics.recordRequest(promptIds.length, result.generatedTokenIds().size(),
                        result.elapsedMs(), failed, cancelled);
                return new TextResult(result.text(), result.generatedTokenIds(), promptIds.length,
                        result.generatedTokenIds().size(), result.reason(), result.elapsedMs());
            } finally {
                if (timeoutTask != null) {
                    timeoutTask.cancel(false);
                }
                activeStreams.remove(requestId);
                if (options.sessionId() == null) {
                    sessionManager.remove(session.id(), modelManager);
                }
            }
        });
        return new StreamHandle(requestId, future, session);
    }

    public void cancelStream(String requestId, String sessionId) {
        GenerationEngine.GenerationRequest request = activeStreams.get(requestId);
        if (request != null) {
            InferenceSession session = sessionId != null ? sessionManager.find(sessionId).orElse(null) : null;
            if (session != null) {
                session.cancel();
            }
        }
        if (sessionId != null) {
            sessionManager.find(sessionId).ifPresent(InferenceSession::cancel);
        }
    }

    public List<DeviceInfo> devices() {
        List<DeviceInfo> devices = new ArrayList<>();
        for (ComputeBackend backend : Backends.discover()) {
            devices.add(backend.deviceInfo());
        }
        if (devices.isEmpty()) {
            devices.add(DeviceInfo.cpu());
        }
        return devices;
    }

    public ai.enr.core.json.JsonObject status() {
        ai.enr.core.json.JsonObject obj = new ai.enr.core.json.JsonObject();
        obj.put("running", running);
        obj.put("version", "1.0.0-SNAPSHOT");
        obj.put("uptime_ms", metrics.uptimeMs());
        obj.put("models_loaded", modelManager.list().size());
        obj.put("sessions_active", sessionManager.liveCount());
        obj.put("total_requests", metrics.totalRequests());
        obj.put("total_generated_tokens", metrics.totalGeneratedTokens());
        obj.put("tokens_per_second", (double) Math.round(metrics.tokensPerSecond() * 100) / 100);
        return obj;
    }

    public Set<String> healthChecks() {
        Set<String> failures = new LinkedHashSet<>();
        if (!running) {
            failures.add("runtime not running");
        }
        for (ModelInstance model : modelManager.list()) {
            if (model.state() != ModelInstance.State.READY) {
                failures.add("model " + model.id() + " in state " + model.state());
            }
        }
        if (devices().stream().noneMatch(d -> d.type() == ai.enr.compute.DeviceType.CPU)) {
            failures.add("no CPU backend");
        }
        return failures;
    }

    public void shutdown() {
        running = false;
        housekeeper.shutdownNow();
        sessionManager.all().keySet().forEach(id -> sessionManager.remove(id, modelManager));
        modelManager.shutdown();
    }
}
