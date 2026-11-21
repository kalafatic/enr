package ai.enr.manager.models;

import ai.enr.core.json.Json;
import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonNumber;
import ai.enr.core.json.JsonObject;
import ai.enr.core.json.JsonString;
import ai.enr.core.json.JsonValue;
import ai.enr.manager.logs.LogBuffer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFilePermission;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Central model management service: provider registry, asynchronous download
 * jobs, duplicate prevention, cancellation, progress reporting and durable
 * job status. This is the single owner of model download state; providers
 * only translate their native protocol into {@link DownloadProgress} updates.
 *
 * <p>Providers are discovered through {@link ServiceLoader} of
 * {@link ModelProviderFactory} and configured through ENR configuration
 * (enr.yaml {@code model_providers} section). A provider is registered for
 * every discovered factory unless it is explicitly disabled; adding a
 * provider therefore never requires modifying this service.</p>
 *
 * <p>Thread-safety: all state transitions happen through immutable
 * {@link DownloadStatus} snapshots in a concurrent map; listeners are
 * notified outside locks and must never throw.</p>
 */
public final class ModelManagementService implements AutoCloseable {

    /** Provider-side job listener: changed = any snapshot, terminal = one of COMPLETED/FAILED/CANCELLED. */
    public interface Listener {
        void onChanged(DownloadStatus status);

        void onTerminal(DownloadStatus status);
    }

    private static final int MAX_PERSISTED_JOBS = 100;
    private static final String SOURCE = "models";

    private final Map<String, ModelProvider> providers = new LinkedHashMap<>();
    private final Map<String, DownloadStatus> jobs = new ConcurrentHashMap<>();
    private final Map<String, String> activePulls = new ConcurrentHashMap<>();
    private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
    private final List<Listener> listeners = new ArrayList<>();
    private final Path stateFile;
    private final LogBuffer log;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    public ModelManagementService(Map<String, ModelProviderConfig> providerConfigs,
                                  Path stateFile, LogBuffer log) {
        this(providerConfigs, stateFile, log, ModelManagementService.class.getClassLoader());
    }

    public ModelManagementService(Map<String, ModelProviderConfig> providerConfigs,
                                  Path stateFile, LogBuffer log, ClassLoader loader) {
        this.stateFile = stateFile;
        this.log = log;
        registerProviders(providerConfigs, loader);
        recoverJobs();
    }

    /**
     * Builds provider configs from the enr.yaml {@code model_providers} section.
     * Flat keys of the form {@code <providerId>.<option>} are grouped per
     * provider. The Ollama provider is enabled by default unless
     * {@code ollama.enabled: false} is set.
     */
    public static Map<String, ModelProviderConfig> configsFromJson(JsonObject section) {
        Map<String, Map<String, String>> grouped = new LinkedHashMap<>();
        if (section != null) {
            for (Map.Entry<String, JsonValue> entry : section.entries()) {
                String key = entry.getKey();
                int dot = key.indexOf('.');
                if (dot <= 0) {
                    continue;
                }
                String providerId = key.substring(0, dot);
                String option = key.substring(dot + 1);
                String value;
                if (entry.getValue() instanceof JsonString s) {
                    value = s.value();
                } else if (entry.getValue() instanceof JsonNumber n) {
                    value = String.valueOf(n.asLong());
                } else {
                    value = entry.getValue().toString();
                }
                grouped.computeIfAbsent(providerId, k -> new LinkedHashMap<>())
                        .put(option, value);
            }
        }
        Map<String, ModelProviderConfig> configs = new LinkedHashMap<>();
        if (!grouped.containsKey("ollama")
                || !"false".equalsIgnoreCase(grouped.get("ollama").get("enabled"))) {
            configs.put("ollama", fromOptions("ollama", grouped.getOrDefault("ollama", Map.of())));
        }
        for (Map.Entry<String, Map<String, String>> entry : grouped.entrySet()) {
            if ("ollama".equals(entry.getKey())) {
                continue;
            }
            if ("false".equalsIgnoreCase(entry.getValue().get("enabled"))) {
                continue;
            }
            configs.put(entry.getKey(), fromOptions(entry.getKey(), entry.getValue()));
        }
        return configs;
    }

    private static ModelProviderConfig fromOptions(String providerId, Map<String, String> options) {
        String endpoint = options.getOrDefault("endpoint", "");
        long connectTimeout = longValue(options, "connect_timeout_ms", 10_000);
        long readTimeout = longValue(options, "read_timeout_ms", 60_000);
        long downloadTimeout = longValue(options, "download_timeout_ms", 0);
        return new ModelProviderConfig(providerId, endpoint, connectTimeout, readTimeout,
                downloadTimeout, options);
    }

    private static long longValue(Map<String, String> options, String key, long defaultValue) {
        try {
            String value = options.get(key);
            return value == null || value.isBlank() ? defaultValue : Long.parseLong(value.trim());
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    private void registerProviders(Map<String, ModelProviderConfig> providerConfigs,
                                   ClassLoader loader) {
        Map<String, ModelProviderFactory> factories = new LinkedHashMap<>();
        for (ModelProviderFactory factory : ServiceLoader.load(ModelProviderFactory.class, loader)) {
            factories.put(factory.id(), factory);
        }
        Map<String, ModelProviderConfig> configs = providerConfigs == null
                ? configsFromJson(new JsonObject()) : providerConfigs;
        for (ModelProviderConfig config : configs.values()) {
            ModelProviderFactory factory = factories.get(config.providerId());
            if (factory == null) {
                warn("No ModelProviderFactory found for provider '" + config.providerId()
                        + "' (register one via META-INF/services)");
                continue;
            }
            try {
                ModelProvider provider = factory.create(config);
                providers.put(provider.id(), provider);
                info("Registered model provider '" + provider.id() + "' ("
                        + provider.displayName() + ")");
            } catch (RuntimeException e) {
                warn("Failed to create model provider '" + config.providerId() + "': "
                        + (e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage()));
            }
        }
    }

    // ------------------------------------------------------------ queries

    public List<ModelProvider> listProviders() {
        return List.copyOf(providers.values());
    }

    public Optional<ModelProvider> provider(String providerId) {
        return Optional.ofNullable(providers.get(providerId));
    }

    /** Models known to be available for download from the provider (empty when unsupported). */
    public List<ModelDescriptor> listModels(String providerId) {
        try {
            return List.copyOf(requireProvider(providerId).listAvailable());
        } catch (Exception e) {
            throw new IllegalArgumentException("Provider '" + providerId
                    + "' cannot list available models: " + message(e), e);
        }
    }

    /** Models currently installed in the provider's store. */
    public List<ModelDescriptor> listInstalledModels(String providerId) {
        try {
            return List.copyOf(requireProvider(providerId).listInstalled());
        } catch (Exception e) {
            throw new IllegalArgumentException("Provider '" + providerId
                    + "' cannot list installed models: " + message(e), e);
        }
    }

    /** Optional metadata for one model of a provider. */
    public Optional<ModelDescriptor> describe(String providerId, String modelId) {
        try {
            return Optional.of(requireProvider(providerId).describe(modelId));
        } catch (UnsupportedOperationException e) {
            return Optional.empty();
        } catch (Exception e) {
            throw new IllegalArgumentException("Provider '" + providerId
                    + "' cannot describe model '" + modelId + "': " + message(e), e);
        }
    }

    public Optional<DownloadStatus> getDownloadStatus(String jobId) {
        return Optional.ofNullable(jobs.get(jobId));
    }

    public List<DownloadStatus> listDownloads() {
        return jobs.values().stream()
                .sorted((a, b) -> Long.compare(b.startedAt(), a.startedAt()))
                .toList();
    }

    public JsonObject providersToJson() {
        JsonObject root = new JsonObject();
        JsonArray arr = new JsonArray();
        for (ModelProvider provider : listProviders()) {
            JsonObject obj = new JsonObject();
            obj.put("id", provider.id());
            obj.put("display_name", provider.displayName());
            JsonArray capabilities = new JsonArray();
            for (ProviderCapability capability : provider.capabilities()) {
                capabilities.add(capability.name());
            }
            obj.put("capabilities", capabilities);
            boolean reachable;
            try {
                reachable = provider.ping();
            } catch (Exception e) {
                reachable = false;
            }
            obj.put("reachable", reachable);
            arr.add(obj);
        }
        root.put("providers", arr);
        return root;
    }

    public JsonObject modelsToJson(String providerId) {
        JsonObject root = new JsonObject();
        JsonArray arr = new JsonArray();
        for (ModelDescriptor descriptor : listModels(providerId)) {
            arr.add(descriptor.toJson());
        }
        root.put("provider", providerId);
        root.put("models", arr);
        return root;
    }

    public JsonObject installedToJson(String providerId) {
        JsonObject root = new JsonObject();
        JsonArray arr = new JsonArray();
        for (ModelDescriptor descriptor : listInstalledModels(providerId)) {
            arr.add(descriptor.toJson());
        }
        root.put("provider", providerId);
        root.put("models", arr);
        return root;
    }

    public JsonObject downloadsToJson() {
        JsonObject root = new JsonObject();
        JsonArray arr = new JsonArray();
        for (DownloadStatus status : listDownloads()) {
            arr.add(status.toJson());
        }
        root.put("downloads", arr);
        return root;
    }

    // ------------------------------------------------------------ operations

    /**
     * Starts an asynchronous model download. Returns the job handle
     * immediately; a second pull of the same provider/model while a download
     * is active returns the existing job instead of starting a duplicate.
     */
    public DownloadJob pullModel(String providerId, String modelId) {
        ModelProvider provider = requireProvider(providerId);
        if (!provider.capabilities().contains(ProviderCapability.MODEL_PULL)) {
            throw new IllegalStateException("Provider '" + providerId + "' does not support model pull");
        }
        String model = normalizeModelId(modelId);
        String key = providerId + "\n" + model;
        String jobId = UUID.randomUUID().toString();
        String existing = activePulls.putIfAbsent(key, jobId);
        if (existing != null) {
            DownloadStatus current = jobs.get(existing);
            if (current != null) {
                return new DownloadJob(current.jobId(), current.providerId(), current.modelId(),
                        current.state(), current.startedAt());
            }
        }
        DownloadStatus initial = new DownloadStatus(jobId, providerId, model,
                DownloadState.QUEUED, "", -1, 0, 0,
                System.currentTimeMillis(), System.currentTimeMillis(), 0, "", List.of());
        jobs.put(jobId, initial);
        persist();
        notifyChanged(initial);
        info("Pull queued: " + providerId + "/" + model + " (job " + jobId + ")");
        executor.submit(() -> runPull(jobId, provider, model, key));
        return new DownloadJob(jobId, providerId, model, DownloadState.QUEUED,
                initial.startedAt());
    }

    public boolean cancelDownload(String jobId) {
        DownloadStatus current = jobs.get(jobId);
        if (current == null || current.terminal()) {
            return false;
        }
        DownloadStatus requested = update(jobId, new DownloadStatus(jobId, current.providerId(),
                current.modelId(), DownloadState.CANCEL_REQUESTED, "cancelling",
                current.totalBytes(), current.completedBytes(),
                DownloadStatus.percent(current.totalBytes(), current.completedBytes()),
                current.startedAt(), System.currentTimeMillis(), 0, "", current.layers()));
        info("Cancel requested for job " + jobId + " (" + requested.modelId() + ")");
        ModelProvider provider = providers.get(current.providerId());
        if (provider != null) {
            try {
                provider.cancel(current.modelId());
            } catch (Exception e) {
                warn("Provider '" + current.providerId() + "' cancel hook failed for "
                        + current.modelId() + ": " + message(e));
            }
        }
        return true;
    }

    public void removeModel(String providerId, String modelId) {
        ModelProvider provider = requireProvider(providerId);
        if (!provider.capabilities().contains(ProviderCapability.MODEL_REMOVE)) {
            throw new IllegalStateException("Provider '" + providerId + "' does not support model removal");
        }
        try {
            provider.remove(normalizeModelId(modelId));
            info("Model removed: " + providerId + "/" + normalizeModelId(modelId));
        } catch (Exception e) {
            throw new IllegalArgumentException("Provider '" + providerId
                    + "' failed to remove model '" + modelId + "': " + message(e), e);
        }
    }

    // ------------------------------------------------------------ job execution

    private void runPull(String jobId, ModelProvider provider, String model, String key) {
        try {
            if (isCancellationRequested(jobId)) {
                return;
            }
            if (provider.capabilities().contains(ProviderCapability.MODEL_LIST)) {
                boolean alreadyInstalled = false;
                try {
                    alreadyInstalled = provider.listInstalled().stream()
                            .anyMatch(descriptor -> model.equals(normalizeModelId(descriptor.id())));
                } catch (Exception e) {
                    warn("Pre-pull check for " + model + " failed: " + message(e));
                }
                if (alreadyInstalled) {
                    update(jobId, snapshot(jobs.get(jobId), DownloadState.COMPLETED,
                            "already installed", -1, 0, "", List.of()));
                    return;
                }
            }
            update(jobId, snapshot(jobs.get(jobId), DownloadState.CONNECTING, "", -1, 0, "", List.of()));
            provider.pull(model, () -> isCancellationRequested(jobId), progress -> applyProgress(jobId, progress));
            DownloadStatus current = jobs.get(jobId);
            if (current == null || current.terminal()) {
                return;
            }
            if (isCancellationRequested(jobId)) {
                update(jobId, snapshot(current, DownloadState.CANCELLED, "cancelled by user",
                        current.totalBytes(), current.completedBytes(), "", current.layers()));
            } else {
                update(jobId, snapshot(current, DownloadState.COMPLETED, "",
                        current.totalBytes(), current.completedBytes(), "", current.layers()));
            }
        } catch (Exception e) {
            DownloadStatus current = jobs.get(jobId);
            if (current == null || current.terminal()) {
                return;
            }
            if (isCancellationRequested(jobId) || current.state() == DownloadState.CANCEL_REQUESTED) {
                update(jobId, snapshot(current, DownloadState.CANCELLED, "cancelled by user",
                        current.totalBytes(), current.completedBytes(), "", current.layers()));
            } else {
                update(jobId, snapshot(current, DownloadState.FAILED, "",
                        current.totalBytes(), current.completedBytes(), message(e), current.layers()));
            }
        } finally {
            activePulls.remove(key, jobId);
        }
    }

    private void applyProgress(String jobId, DownloadProgress progress) {
        DownloadStatus current = jobs.get(jobId);
        if (current == null || current.terminal()
                || current.state() == DownloadState.CANCEL_REQUESTED) {
            return;
        }
        if (progress.state() == DownloadState.COMPLETED || progress.state() == DownloadState.FAILED
                || progress.state() == DownloadState.CANCELLED) {
            update(jobId, snapshot(current, progress.state(), "",
                    progress.totalBytes(), progress.completedBytes(), progress.error(),
                    progress.layers()));
            return;
        }
        update(jobId, snapshot(current, progress.state(), progress.state().name().toLowerCase(),
                progress.totalBytes(), progress.completedBytes(), progress.error(),
                progress.layers()));
    }

    private boolean isCancellationRequested(String jobId) {
        DownloadStatus current = jobs.get(jobId);
        return current != null && (current.state() == DownloadState.CANCEL_REQUESTED
                || current.state() == DownloadState.CANCELLED);
    }

    private DownloadStatus snapshot(DownloadStatus current, DownloadState state, String operation,
                                    long totalBytes, long completedBytes, String error,
                                    List<DownloadLayer> layers) {
        boolean terminal = state.terminal();
        long total = totalBytes >= 0 ? totalBytes
                : (current != null ? current.totalBytes() : -1);
        long completed = completedBytes > 0 ? completedBytes
                : (current != null ? current.completedBytes() : 0);
        if (state == DownloadState.COMPLETED && total > 0) {
            completed = total;
        }
        String jobId = current.jobId();
        String providerId = current.providerId();
        String modelId = current.modelId();
        long startedAt = current.startedAt();
        long now = System.currentTimeMillis();
        return new DownloadStatus(jobId, providerId, modelId, state, operation, total, completed,
                DownloadStatus.percent(total, completed), startedAt, now,
                terminal ? (current.completedAt() > 0 ? current.completedAt() : now) : 0,
                error, layers == null ? List.of() : layers);
    }

    private DownloadStatus update(String jobId, DownloadStatus next) {
        jobs.put(jobId, next);
        persist();
        if (next.terminal()) {
            info("Pull " + next.state().name().toLowerCase() + ": " + next.providerId() + "/"
                    + next.modelId() + (next.error().isBlank() ? "" : " - " + next.error()));
            notifyTerminal(next);
        } else {
            notifyChanged(next);
        }
        return next;
    }

    // ------------------------------------------------------------ persistence

    private void recoverJobs() {
        if (stateFile == null || !Files.isRegularFile(stateFile)) {
            return;
        }
        try {
            JsonValue parsed = Json.parse(Files.readString(stateFile));
            if (!(parsed instanceof JsonObject root)) {
                return;
            }
            JsonArray arr = root.getArray("downloads");
            if (arr == null) {
                return;
            }
            for (int i = 0; i < arr.size(); i++) {
                JsonObject obj = arr.getObject(i);
                if (obj == null) {
                    continue;
                }
                DownloadStatus status = DownloadStatus.fromJson(obj);
                if (!status.terminal()) {
                    status = new DownloadStatus(status.jobId(), status.providerId(),
                            status.modelId(), DownloadState.FAILED, "",
                            status.totalBytes(), status.completedBytes(),
                            DownloadStatus.percent(status.totalBytes(), status.completedBytes()),
                            status.startedAt(), System.currentTimeMillis(),
                            System.currentTimeMillis(), "interrupted by restart",
                            status.layers());
                }
                jobs.put(status.jobId(), status);
            }
            info("Recovered " + jobs.size() + " download job(s) from " + stateFile);
        } catch (Exception e) {
            warn("Failed to recover download jobs from " + stateFile + ": " + message(e));
        }
    }

    private void persist() {
        if (stateFile == null) {
            return;
        }
        try {
            Files.createDirectories(stateFile.getParent());
            JsonObject root = new JsonObject();
            JsonArray arr = new JsonArray();
            List<DownloadStatus> all = listDownloads();
            for (DownloadStatus status : all.subList(0, Math.min(MAX_PERSISTED_JOBS, all.size()))) {
                arr.add(status.toJson());
            }
            root.put("downloads", arr);
            Path tmp = stateFile.resolveSibling(stateFile.getFileName() + ".tmp");
            Files.writeString(tmp, Json.write(root, true));
            try {
                Files.setPosixFilePermissions(tmp, Set.of(
                        PosixFilePermission.OWNER_READ, PosixFilePermission.OWNER_WRITE));
            } catch (UnsupportedOperationException ignored) {
                // non-POSIX file system
            }
            Files.move(tmp, stateFile, StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            warn("Failed to persist download jobs: " + message(e));
        }
    }

    // ------------------------------------------------------------ listeners

    public void addListener(Listener listener) {
        synchronized (listeners) {
            listeners.add(listener);
        }
    }

    public void removeListener(Listener listener) {
        synchronized (listeners) {
            listeners.remove(listener);
        }
    }

    private void notifyChanged(DownloadStatus status) {
        for (Listener listener : listenerSnapshot()) {
            try {
                listener.onChanged(status);
            } catch (RuntimeException ignored) {
                // listeners must never break downloads
            }
        }
    }

    private void notifyTerminal(DownloadStatus status) {
        for (Listener listener : listenerSnapshot()) {
            try {
                listener.onTerminal(status);
            } catch (RuntimeException ignored) {
                // listeners must never break downloads
            }
        }
    }

    private List<Listener> listenerSnapshot() {
        synchronized (listeners) {
            return List.copyOf(listeners);
        }
    }

    // ------------------------------------------------------------ shared

    private ModelProvider requireProvider(String providerId) {
        ModelProvider provider = providers.get(providerId);
        if (provider == null) {
            throw new IllegalArgumentException("Unknown model provider: " + providerId);
        }
        return provider;
    }

    private static String normalizeModelId(String modelId) {
        String model = modelId == null ? "" : modelId.trim();
        if (model.isBlank()) {
            throw new IllegalArgumentException("Model identifier must not be blank");
        }
        return model;
    }

    private static String message(Throwable throwable) {
        String message = throwable.getMessage();
        return message == null || message.isBlank()
                ? throwable.getClass().getSimpleName() : message;
    }

    private void info(String message) {
        if (log != null) {
            log.info(SOURCE, message);
        }
    }

    private void warn(String message) {
        if (log != null) {
            log.warn(SOURCE, message);
        }
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        for (DownloadStatus status : List.copyOf(jobs.values())) {
            if (status.terminal()) {
                continue;
            }
            ModelProvider provider = providers.get(status.providerId());
            if (provider != null) {
                try {
                    provider.cancel(status.modelId());
                } catch (Exception ignored) {
                    // best effort during shutdown
                }
            }
            update(status.jobId(), snapshot(status, DownloadState.FAILED, "",
                    status.totalBytes(), status.completedBytes(), "service shut down", status.layers()));
        }
        executor.close();
        activePulls.clear();
    }

    public boolean isClosed() {
        return closed.get();
    }
}
