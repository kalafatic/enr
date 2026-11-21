package ai.enr.manager;

import ai.enr.core.json.Json;
import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonObject;
import ai.enr.inference.GenerationEngine;
import ai.enr.manager.catalog.Catalogs;
import ai.enr.manager.download.DownloadManager;
import ai.enr.manager.logs.LogBuffer;
import ai.enr.manager.models.DownloadState;
import ai.enr.manager.models.DownloadStatus;
import ai.enr.manager.models.ModelManagementService;
import ai.enr.manager.monitor.HardwareMonitor;
import ai.enr.manager.provider.ChatMessage;
import ai.enr.manager.provider.Provider;
import ai.enr.manager.provider.ProviderClient;
import ai.enr.manager.provider.ProviderRegistry;
import ai.enr.manager.provider.ProviderType;
import ai.enr.manager.router.ModelRouter;
import ai.enr.manager.store.ModelManifest;
import ai.enr.manager.store.ModelStore;
import ai.enr.runtime.EnrRuntime;
import ai.enr.runtime.ModelInstance;
import ai.enr.runtime.RuntimeConfig;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * ENR Manager: the single authoritative facade for model management, provider
 * management, downloads, routing and diagnostics. The web UI, the CLI, the
 * REST API and the system service all delegate to this class — there is no
 * second registry, router or lifecycle implementation.
 */
public final class EnrManager implements AutoCloseable {

    public record ChatOptions(int maxTokens, Float temperature, Float topP, Integer topK,
                              Long seed, List<String> stop, boolean stream) {

        public static ChatOptions defaults() {
            return new ChatOptions(256, null, null, null, null, List.of(), false);
        }
    }

    private final EnrConfig config;
    private final LogBuffer log = new LogBuffer(2000);
    private final EnrRuntime runtime;
    private final ModelStore store;
    private final DownloadManager downloads = new DownloadManager();
    private final ProviderRegistry providers;
    private final ProviderClient providerClient = new ProviderClient();
    private final Catalogs catalogs;
    private final ModelRouter router;
    private final ModelManagementService modelManagement;
    private final HardwareMonitor hardware = new HardwareMonitor();
    private final long startedAt = System.currentTimeMillis();
    private final AtomicReference<FileHandler> logFileHandler = new AtomicReference<>();
    private volatile boolean closed = false;

    public EnrManager(EnrConfig config) {
        this.config = config;
        try {
            Files.createDirectories(config.dataDir());
            Files.createDirectories(config.modelsDir());
            Files.createDirectories(config.logDir());
        } catch (IOException e) {
            throw new IllegalStateException("Failed to create ENR directories under "
                    + config.dataDir() + ": " + e.getMessage(), e);
        }
        installLogHandlers();
        this.store = new ModelStore(config, log);
        this.providers = new ProviderRegistry(config.dataDir(), log);
        this.catalogs = new Catalogs(config, log);
        this.router = new ModelRouter(store, providers, log);
        RuntimeConfig runtimeConfig = RuntimeConfig.builder()
                .host(config.host())
                .port(config.port())
                .modelDirectories(config.modelDirectories())
                .preferredBackend(config.preferredBackend())
                .allowCudaFallback(config.allowCudaFallback())
                .maxConcurrentSessions(config.maxConcurrentSessions())
                .defaultMaxTokens(config.defaultMaxTokens())
                .sessionIdleTimeoutSeconds(config.sessionIdleTimeoutSeconds())
                .requestTimeoutSeconds(config.requestTimeoutSeconds())
                .enableMetrics(config.enableMetrics())
                .build();
        this.runtime = new EnrRuntime(runtimeConfig);
        downloads.addListener(new DownloadManager.Listener() {
            @Override
            public void onChanged(DownloadManager.Job job) {
                log.info("downloads", "Download " + job.name() + ": " + job.status().name().toLowerCase()
                        + progress(job));
            }

            @Override
            public void onCompleted(DownloadManager.Job job) {
                log.info("downloads", "Download finished: " + job.name() + " -> " + job.targetFile());
            }
        });
        this.modelManagement = new ModelManagementService(
                ModelManagementService.configsFromJson(config.modelProviders()),
                config.dataDir().resolve("model-download-jobs.json"), log);
        modelManagement.addListener(new ModelManagementService.Listener() {
            private final java.util.Map<String, DownloadState> lastStates = new java.util.concurrent.ConcurrentHashMap<>();

            @Override
            public void onChanged(DownloadStatus status) {
                DownloadState previous = lastStates.put(status.jobId(), status.state());
                if (previous != status.state()) {
                    log.info("models", "Model pull " + status.state().name().toLowerCase()
                            + ": " + status.providerId() + "/" + status.modelId()
                            + progress(status.totalBytes(), status.completedBytes()));
                }
            }

            @Override
            public void onTerminal(DownloadStatus status) {
                lastStates.put(status.jobId(), status.state());
                log.info("models", "Model pull finished (" + status.state().name().toLowerCase()
                        + "): " + status.providerId() + "/" + status.modelId()
                        + (status.error().isBlank() ? "" : " - " + status.error()));
            }
        });
        log.info("enr", "ENR Manager started (data dir " + config.dataDir() + ", models dir "
                + config.modelsDir() + ")");
    }

    private static String progress(DownloadManager.Job job) {
        if (job.totalBytes() > 0) {
            long percent = job.completedBytes() * 100 / job.totalBytes();
            return " (" + percent + "%, " + job.completedBytes() + "/" + job.totalBytes() + " bytes)";
        }
        return " (" + job.completedBytes() + " bytes)";
    }

    private static String progress(long totalBytes, long completedBytes) {
        if (totalBytes > 0) {
            long percent = completedBytes * 100 / totalBytes;
            return " (" + percent + "%, " + completedBytes + "/" + totalBytes + " bytes)";
        }
        return " (" + completedBytes + " bytes)";
    }

    private void installLogHandlers() {
        Logger root = Logger.getLogger("");
        root.addHandler(log.handler());
        try {
            FileHandler handler = new FileHandler(config.logDir().resolve("enr.log").toString(),
                    5 * 1024 * 1024, 3, true);
            handler.setFormatter(new java.util.logging.SimpleFormatter());
            handler.setLevel(Level.INFO);
            root.addHandler(handler);
            logFileHandler.set(handler);
        } catch (IOException e) {
            log.warn("enr", "Failed to open log file in " + config.logDir() + ": " + e.getMessage());
        }
    }

    // ------------------------------------------------------------ accessors

    public EnrConfig config() {
        return config;
    }

    public EnrRuntime runtime() {
        return runtime;
    }

    public ModelStore store() {
        return store;
    }

    public DownloadManager downloads() {
        return downloads;
    }

    public ModelManagementService modelManagement() {
        return modelManagement;
    }

    public ProviderRegistry providers() {
        return providers;
    }

    public ModelRouter router() {
        return router;
    }

    public LogBuffer logBuffer() {
        return log;
    }

    public long startedAt() {
        return startedAt;
    }

    // ------------------------------------------------------------ models

    /** Installed models merged with live runtime state (loaded/backend). */
    public JsonObject listModels() {
        JsonObject root = new JsonObject();
        JsonArray arr = new JsonArray();
        for (ModelStore.StoredModel stored : store.list()) {
            JsonObject obj = stored.manifest().toJson();
            obj.put("artifact_present", stored.artifactPresent());
            ModelInstance loaded = runtime.models().get(stored.id());
            obj.put("loaded", loaded != null);
            obj.put("state", loaded == null ? "not_loaded" : loaded.state().name().toLowerCase());
            obj.put("backend", loaded == null ? "" : loaded.backend().name());
            obj.put("active_sessions", loaded == null ? 0 : loaded.activeSessions());
            arr.add(obj);
        }
        root.put("models", arr);
        root.put("models_dir", config.modelsDir().toString());
        return root;
    }

    public JsonObject modelDetails(String id) {
        ModelStore.StoredModel stored = store.find(id);
        if (stored == null) {
            throw new IllegalArgumentException("Model not installed: " + id);
        }
        JsonObject obj = stored.manifest().toJson();
        obj.put("artifact_present", stored.artifactPresent());
        ModelInstance loaded = runtime.models().get(id);
        obj.put("loaded", loaded != null);
        if (loaded != null) {
            obj.put("state", loaded.state().name().toLowerCase());
            obj.put("backend", loaded.backend().name());
            obj.put("active_sessions", loaded.activeSessions());
            obj.put("total_requests", loaded.totalRequests());
            obj.put("parameters", loaded.loadedModel().totalParameters());
            obj.put("architecture", loaded.loadedModel().architecture().family());
            obj.put("max_seq_len", loaded.loadedModel().architecture().maxSeqLen());
        }
        return obj;
    }

    /** Loads (starts) an installed model; verifies the real runtime state. */
    public JsonObject startModel(String id, String backend) throws Exception {
        Path file = store.modelFile(id);
        if (file == null) {
            throw new IllegalArgumentException("Model not installed: " + id);
        }
        if (!Files.isRegularFile(file)) {
            throw new IllegalStateException("Model artifact missing for " + id + ": " + file);
        }
        ModelInstance model = runtime.loadModel(file,
                backend == null || backend.isBlank() ? null : backend);
        if (model.state() != ModelInstance.State.READY) {
            throw new IllegalStateException("Model " + id + " did not reach READY state: " + model.state());
        }
        log.info("enr", "Model started: " + id + " (backend " + model.backend().name() + ")");
        return modelDetails(id);
    }

    public void stopModel(String id, boolean force) {
        if (runtime.models().get(id) == null) {
            throw new IllegalArgumentException("Model not loaded: " + id);
        }
        runtime.unloadModel(id, force);
        if (runtime.models().get(id) != null) {
            throw new IllegalStateException("Model " + id + " still loaded after unload");
        }
        log.info("enr", "Model stopped: " + id);
    }

    public JsonObject removeModel(String id) throws IOException {
        if (runtime.models().get(id) != null) {
            runtime.unloadModel(id, true);
        }
        store.remove(id);
        return listModels();
    }

    public JsonObject importModel(Path source, String id) throws IOException {
        ModelManifest manifest = store.importModel(source, id);
        return modelDetails(manifest.id());
    }

    public JsonObject verifyModel(String id) {
        boolean ok = store.verify(id);
        log.info("enr", "Verification for " + id + ": " + (ok ? "OK" : "FAILED"));
        JsonObject response = new JsonObject();
        response.put("id", id);
        response.put("verified", ok);
        return response;
    }

    /** Resolves and starts a catalog download; returns the download job id. */
    public String pullModel(String ref) throws IOException {
        Catalogs.ResolvedDownload resolved = catalogs.resolveDownload(ref);
        String name = ModelStore.sanitizeId(modelName(ref));
        Path targetDir = config.modelsDir().resolve(name);
        String targetFile = targetDir.resolve(name + ".gguf").toString();
        log.info("enr", "Pull requested: " + ref + " -> " + resolved.url()
                + (resolved.digest().isBlank() ? " (no digest)" : " (digest " + resolved.digest() + ")"));
        return downloads.start(resolved.name(), resolved.url(), Path.of(targetFile), resolved.digest(), job -> {
            try {
                store.registerDownload(name, Path.of(job.targetFile()), job.digest(),
                        job.totalBytes(), "catalog:" + resolved.name());
                log.info("enr", "Pull completed and registered: " + name);
            } catch (Exception e) {
                log.error("enr", "Pull completed but registration failed for " + name + ": " + e.getMessage());
            }
        });
    }

    private static String modelName(String ref) {
        String name = ref.startsWith("hf:") ? ref.substring(3) : ref;
        if (!ref.startsWith("hf:") && name.startsWith("ollama:")) {
            name = name.substring("ollama:".length());
        }
        int colon = name.lastIndexOf(':');
        if (colon > 0 && name.indexOf('/') < 0) {
            name = name.substring(0, colon);
        }
        int slash = name.lastIndexOf('/');
        if (slash >= 0) {
            name = name.substring(slash + 1);
        }
        int dot = name.lastIndexOf('.');
        if (dot > 0) {
            name = name.substring(0, dot);
        }
        return name;
    }

    public JsonObject searchCatalog(String query, int limit) {
        Catalogs.SearchResult result = catalogs.search(query, limit);
        JsonObject root = new JsonObject();
        JsonArray models = new JsonArray();
        for (Catalogs.CatalogModel model : result.models()) {
            JsonObject entry = new JsonObject();
            entry.put("id", model.id());
            entry.put("source", model.source());
            entry.put("description", model.description());
            entry.put("size_bytes", model.sizeBytes());
            models.add(entry);
        }
        root.put("models", models);
        JsonArray errors = new JsonArray();
        for (String error : result.errors()) {
            errors.add(error);
        }
        root.put("errors", errors);
        return root;
    }

    // ------------------------------------------------------------ providers

    public JsonObject addProvider(String id, String typeId, String name, String baseUrl,
                                  String apiKey, boolean enabled) {
        ProviderType type = ProviderType.fromId(typeId);
        Provider provider = new Provider(id == null || id.isBlank() ? type.id() : id,
                type, name, baseUrl == null || baseUrl.isBlank() ? ProviderType.defaultBaseUrl(type) : baseUrl,
                apiKey, enabled);
        providers.put(provider);
        return provider.toJson();
    }

    public JsonObject updateProvider(String id, String name, String baseUrl, String apiKey,
                                     Boolean enabled) {
        Provider existing = providers.require(id);
        Provider updated = existing.withConfig(name, baseUrl, apiKey,
                enabled == null ? existing.enabled() : enabled);
        providers.put(updated);
        return updated.toJson();
    }

    public JsonObject deleteProvider(String id) {
        providers.remove(id);
        return providers.toJson();
    }

    public JsonObject testProvider(String id) {
        Provider provider = providers.require(id);
        ProviderClient.TestResult result = providers.test(provider);
        JsonObject response = new JsonObject();
        response.put("provider", id);
        response.put("ok", result.ok());
        response.put("message", result.message());
        response.put("elapsed_ms", result.elapsedMs());
        JsonArray models = new JsonArray();
        for (String model : result.models()) {
            models.add(model);
        }
        response.put("models", models);
        return response;
    }

    public JsonObject providerModels(String id) {
        Provider provider = providers.require(id);
        JsonObject response = new JsonObject();
        response.put("provider", id);
        JsonArray models = new JsonArray();
        for (String model : providers.models(provider)) {
            models.add(model);
        }
        response.put("models", models);
        return response;
    }

    public JsonObject providerHealth(String id) {
        Provider provider = providers.require(id);
        JsonObject response = new JsonObject();
        response.put("provider", id);
        response.put("enabled", provider.enabled());
        response.put("last_test_ok", provider.lastTestOk());
        response.put("last_test_at", provider.lastTestAt());
        response.put("last_test_message", provider.lastTestMessage());
        response.put("connected", provider.lastTestOk()
                && System.currentTimeMillis() - provider.lastTestAt() < 300_000);
        return response;
    }

    // ------------------------------------------------------------ chat

    public JsonObject chat(String modelName, List<ChatMessage> messages, ChatOptions options) {
        long started = System.currentTimeMillis();
        ModelRouter.Route route = router.resolve(modelName);
        if (route.target() == ModelRouter.Route.Target.LOCAL) {
            return chatLocal(route.modelId(), messages, options, started);
        }
        return chatProvider(route, messages, options, started, null);
    }

    /** Streams chat output; final chunk has done=true. */
    public void chatStream(String modelName, List<ChatMessage> messages, ChatOptions options,
                           Consumer<JsonObject> onChunk) throws IOException {
        long started = System.currentTimeMillis();
        ModelRouter.Route route = router.resolve(modelName);
        if (route.target() == ModelRouter.Route.Target.LOCAL) {
            chatLocalStream(route.modelId(), messages, options, started, onChunk);
        } else {
            try {
                providerClient.chatStream(route.provider(), route.modelId(), messages,
                        new ProviderClient.ChatOptions(options.maxTokens(), options.temperature(),
                                options.topP(), options.topK(), options.seed(), options.stop(),
                                options.stream()), chunk -> {
                            JsonObject obj = new JsonObject();
                            obj.put("content", chunk.content());
                            obj.put("done", chunk.done());
                            obj.put("finish_reason", chunk.finishReason() == null ? "" : chunk.finishReason());
                            obj.put("prompt_tokens", chunk.promptTokens());
                            obj.put("completion_tokens", chunk.completionTokens());
                            obj.put("elapsed_ms", System.currentTimeMillis() - started);
                            onChunk.accept(obj);
                        });
            } catch (Exception e) {
                String message = e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
                JsonObject obj = new JsonObject();
                obj.put("content", "");
                obj.put("done", true);
                obj.put("finish_reason", "error");
                obj.put("error", message);
                onChunk.accept(obj);
            }
        }
    }

    private JsonObject chatLocal(String modelId, List<ChatMessage> messages, ChatOptions options,
                                 long started) {
        ensureLoaded(modelId);
        String prompt = buildPrompt(messages);
        EnrRuntime.InferenceOptions inferenceOptions = new EnrRuntime.InferenceOptions(
                options.maxTokens() > 0 ? options.maxTokens() : config.defaultMaxTokens(),
                samplingFrom(options), stopTokensFrom(options), null, null);
        EnrRuntime.TextResult result = runtime.infer(modelId, prompt, inferenceOptions, null);
        JsonObject response = new JsonObject();
        response.put("content", result.text());
        response.put("model", modelId);
        response.put("local", true);
        response.put("finish_reason", finishReason(result.reason()));
        response.put("prompt_tokens", result.promptTokens());
        response.put("completion_tokens", result.generatedTokens());
        response.put("elapsed_ms", System.currentTimeMillis() - started);
        return response;
    }

    private void chatLocalStream(String modelId, List<ChatMessage> messages, ChatOptions options,
                                 long started, Consumer<JsonObject> onChunk) {
        ensureLoaded(modelId);
        String prompt = buildPrompt(messages);
        EnrRuntime.InferenceOptions inferenceOptions = new EnrRuntime.InferenceOptions(
                options.maxTokens() > 0 ? options.maxTokens() : config.defaultMaxTokens(),
                samplingFrom(options), stopTokensFrom(options), null, null);
        EnrRuntime.TextResult result = runtime.inferStream(modelId, prompt, inferenceOptions,
                (tokenId, piece) -> {
                    JsonObject obj = new JsonObject();
                    obj.put("content", piece);
                    obj.put("done", false);
                    obj.put("finish_reason", "");
                    obj.put("elapsed_ms", System.currentTimeMillis() - started);
                    onChunk.accept(obj);
                }).future().join();
        JsonObject obj = new JsonObject();
        obj.put("content", "");
        obj.put("done", true);
        obj.put("finish_reason", finishReason(result.reason()));
        obj.put("prompt_tokens", result.promptTokens());
        obj.put("completion_tokens", result.generatedTokens());
        obj.put("elapsed_ms", System.currentTimeMillis() - started);
        onChunk.accept(obj);
    }

    private JsonObject chatProvider(ModelRouter.Route route, List<ChatMessage> messages,
                                    ChatOptions options, long started, Object unused) {
        ProviderClient.ChatResult result = providerClient.chat(route.provider(), route.modelId(),
                messages, new ProviderClient.ChatOptions(options.maxTokens(), options.temperature(),
                        options.topP(), options.topK(), options.seed(), options.stop(), options.stream()));
        JsonObject response = new JsonObject();
        response.put("content", result.content());
        response.put("model", route.modelId());
        response.put("provider", route.provider().id());
        response.put("local", false);
        response.put("finish_reason", result.finishReason());
        response.put("prompt_tokens", result.promptTokens());
        response.put("completion_tokens", result.completionTokens());
        response.put("elapsed_ms", System.currentTimeMillis() - started);
        if (!result.ok()) {
            response.put("error", result.error());
        }
        return response;
    }

    static String buildPrompt(List<ChatMessage> messages) {
        StringBuilder prompt = new StringBuilder();
        for (ChatMessage message : messages) {
            switch (message.role()) {
                case "system" -> prompt.append("System: ").append(message.content()).append('\n');
                case "assistant" -> prompt.append("Assistant: ").append(message.content()).append('\n');
                default -> prompt.append("User: ").append(message.content()).append('\n');
            }
        }
        prompt.append("Assistant: ");
        return prompt.toString();
    }

    /** Lazily loads a local model when a chat request targets it. */
    private void ensureLoaded(String modelId) {
        if (runtime.models().get(modelId) != null) {
            return;
        }
        Path file = store.modelFile(modelId);
        if (file == null) {
            throw new IllegalArgumentException("Model not installed: " + modelId);
        }
        try {
            runtime.loadModel(file);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to auto-load model " + modelId + ": " + e.getMessage(), e);
        }
        if (runtime.models().get(modelId) == null) {
            throw new IllegalStateException("Model " + modelId + " failed to load");
        }
    }

    private static ai.enr.inference.sample.SamplingConfig samplingFrom(ChatOptions options) {
        var sampling = ai.enr.inference.sample.SamplingConfig.defaults();
        if (options.temperature() != null) {
            sampling = sampling.withTemperature(options.temperature());
        }
        if (options.seed() != null) {
            sampling = sampling.withSeed(options.seed());
        }
        return sampling;
    }

    private static java.util.Set<Integer> stopTokensFrom(ChatOptions options) {
        return java.util.Set.of();
    }

    private static String finishReason(GenerationEngine.TerminationReason reason) {
        return switch (reason) {
            case EOS_REACHED, STOP_TOKEN, MAX_TOKENS_REACHED, MAX_SEQ_LEN_REACHED -> "stop";
            case CANCELLED -> "cancelled";
            case ERROR -> "error";
        };
    }

    // ------------------------------------------------------------ diagnostics

    public JsonObject dashboard() {
        JsonObject root = new JsonObject();
        root.put("server", serverInfo());
        root.put("hardware", hardware.toJson(runtime));
        root.put("active_requests", activeRequestsJson());
        root.put("models_installed", store.list().size());
        root.put("models_loaded", runtime.models().list().size());
        root.put("providers", providers.list().size());
        root.put("downloads_active", downloads.list().stream()
                .filter(j -> j.status() == DownloadManager.Status.RUNNING).count());
        return root;
    }

    private JsonObject serverInfo() {
        JsonObject server = new JsonObject();
        server.put("name", "ENR");
        server.put("version", "2.6.11");
        server.put("host", config.host());
        server.put("port", config.port());
        server.put("uptime_ms", System.currentTimeMillis() - startedAt);
        server.put("status", runtime.isRunning() && runtime.healthChecks().isEmpty() ? "healthy" : "degraded");
        server.put("api", "http://" + config.host() + ":" + config.port() + "/api/v1");
        server.put("ollama_api", "http://" + config.host() + ":" + config.port() + "/api");
        server.put("openai_api", "http://" + config.host() + ":" + config.port() + "/v1");
        server.put("web_ui", "http://" + config.host() + ":" + config.port() + "/");
        return server;
    }

    private JsonArray activeRequestsJson() {
        JsonArray arr = new JsonArray();
        for (EnrRuntime.ActiveStreamInfo stream : runtime.activeStreams()) {
            JsonObject entry = new JsonObject();
            entry.put("request_id", stream.requestId());
            entry.put("model", stream.modelId());
            entry.put("started_at", stream.startedAt());
            entry.put("elapsed_ms", System.currentTimeMillis() - stream.startedAt());
            entry.put("max_tokens", stream.maxTokens());
            arr.add(entry);
        }
        return arr;
    }

    public JsonObject diagnostics() {
        JsonObject root = new JsonObject();
        root.put("status", runtime.status());
        root.put("health", healthJson());
        root.put("hardware", hardware.toJson(runtime));
        root.put("metrics", runtime.metrics().toJson());
        root.put("active_requests", activeRequestsJson());
        JsonArray checks = new JsonArray();
        for (String failure : runtime.healthChecks()) {
            checks.add(failure);
        }
        root.put("health_checks", checks);
        return root;
    }

    private JsonObject healthJson() {
        boolean healthy = runtime.isRunning() && runtime.healthChecks().isEmpty();
        JsonObject health = new JsonObject();
        health.put("status", healthy ? "healthy" : "unhealthy");
        health.put("uptime_ms", System.currentTimeMillis() - startedAt);
        return health;
    }

    public JsonObject logs(String source, int tail) {
        List<LogBuffer.Entry> entries = log.recent(tail);
        JsonObject root = new JsonObject();
        JsonArray arr = new JsonArray();
        for (LogBuffer.Entry entry : entries) {
            if (source != null && !source.isBlank() && !"all".equals(source)
                    && !entry.logger().startsWith(source)) {
                continue;
            }
            JsonObject obj = new JsonObject();
            obj.put("timestamp_ms", entry.timestampMs());
            obj.put("level", entry.level());
            obj.put("logger", entry.logger());
            obj.put("message", entry.message());
            arr.add(obj);
        }
        root.put("logs", arr);
        root.put("total", log.size());
        return root;
    }

    /** Pulls all local artifacts into one view for UI + Ollama /api/tags. */
    public JsonArray ollamaTags() {
        JsonArray models = new JsonArray();
        for (ModelStore.StoredModel stored : store.list()) {
            JsonObject entry = new JsonObject();
            entry.put("name", stored.id());
            entry.put("model", stored.id());
            entry.put("modified_at", stored.manifest().installedAt() == 0
                    ? java.time.Instant.now().toString() : java.time.Instant.ofEpochMilli(
                    stored.manifest().installedAt()).toString());
            entry.put("size", Math.max(stored.manifest().sizeBytes(), 0));
            entry.put("digest", "sha256:" + stored.manifest().digest());
            JsonObject details = new JsonObject();
            details.put("format", stored.manifest().format());
            details.put("family", "gguf");
            details.put("parameter_size", "");
            details.put("quantization_level", "");
            entry.put("details", details);
            models.add(entry);
        }
        return models;
    }

    public JsonObject ollamaShow(String name) {
        ModelStore.StoredModel stored = store.find(name);
        if (stored == null) {
            throw new IllegalArgumentException("Model not installed: " + name);
        }
        JsonObject response = new JsonObject();
        response.put("license", "");
        response.put("modelfile", "# ENR model " + name + "\nFROM " + stored.manifest().fileName());
        response.put("parameters", "");
        response.put("template", "");
        JsonObject details = new JsonObject();
        details.put("format", stored.manifest().format());
        details.put("family", "gguf");
        details.put("parameter_size", "");
        details.put("quantization_level", "");
        response.put("details", details);
        ModelInstance loaded = runtime.models().get(name);
        if (loaded != null) {
            details.put("parameter_size", loaded.loadedModel().totalParameters() + " parameters");
            details.put("context_length", loaded.loadedModel().architecture().maxSeqLen());
        }
        return response;
    }

    public JsonObject ollamaPs() {
        JsonObject root = new JsonObject();
        JsonArray models = new JsonArray();
        for (ModelInstance model : runtime.models().list()) {
            JsonObject entry = new JsonObject();
            entry.put("name", model.id());
            entry.put("model", model.id());
            entry.put("size", model.totalWeightBytes());
            entry.put("digest", "");
            JsonObject details = new JsonObject();
            details.put("format", "gguf");
            details.put("family", "gguf");
            details.put("parameter_size", model.loadedModel().totalParameters() + " parameters");
            details.put("quantization_level", "");
            entry.put("details", details);
            entry.put("expires_at", java.time.Instant.now().plusSeconds(300).toString());
            entry.put("size_vram", model.backend().deviceInfo().type()
                    == ai.enr.compute.DeviceType.CUDA ? model.totalWeightBytes() : 0);
            models.add(entry);
        }
        root.put("models", models);
        return root;
    }

    public JsonObject version() {
        JsonObject response = new JsonObject();
        response.put("version", "2.6.11");
        response.put("name", "ENR");
        return response;
    }

    public boolean isClosed() {
        return closed;
    }

    @Override
    public void close() {
        closed = true;
        modelManagement.close();
        downloads.close();
        providerClient.close();
        catalogs.close();
        runtime.shutdown();
        log.info("enr", "ENR Manager shut down");
        FileHandler handler = logFileHandler.get();
        if (handler != null) {
            handler.close();
        }
    }
}
