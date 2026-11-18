package ai.enr.runtime;

import ai.enr.compute.Backends;
import ai.enr.compute.ComputeBackend;
import ai.enr.loader.ModelSources;
import ai.enr.loader.api.LoadedModel;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;
import java.util.stream.Stream;

/**
 * Multi-model registry and lifecycle manager: discovery, load, unload,
 * residency tracking, backend selection and controlled eviction.
 * A model is never unloaded while it has active sessions.
 */
public final class ModelManager {

    private static final Logger LOG = Logger.getLogger(ModelManager.class.getName());

    private final RuntimeConfig config;
    private final ConcurrentHashMap<String, ModelInstance> instances = new ConcurrentHashMap<>();

    public ModelManager(RuntimeConfig config) {
        this.config = config;
    }

    public List<Path> discoverModelFiles() throws IOException {
        List<Path> found = new ArrayList<>();
        for (Path dir : config.modelDirectories()) {
            if (!Files.isDirectory(dir)) {
                continue;
            }
            try (Stream<Path> stream = Files.list(dir)) {
                stream.filter(Files::isRegularFile)
                        .filter(p -> {
                            String name = p.getFileName().toString().toLowerCase();
                            return name.endsWith(".evo") || name.endsWith(".gguf");
                        })
                        .sorted()
                        .forEach(found::add);
            }
        }
        return found;
    }

    public ModelInstance load(Path path, String preferredBackend) throws Exception {
        Path absolute = path.toAbsolutePath();
        String id = modelIdFromPath(absolute);

        ModelInstance existing = instances.get(id);
        if (existing != null) {
            return existing;
        }

        LoadedModel loadedModel = ModelSources.open(absolute);
        ComputeBackend backend = selectBackend(preferredBackend);

        ModelInstance instance = new ModelInstance(id, absolute, loadedModel, backend);
        ModelInstance raced = instances.putIfAbsent(id, instance);
        if (raced != null) {
            return raced;
        }
        LOG.info(() -> "Loaded model " + id + " from " + absolute + " on backend " + backend.name()
                + " (" + loadedModel.totalParameters() + " params, " + (loadedModel.weightBytes() >> 20) + " MB)");
        return instance;
    }

    public ComputeBackend selectBackend(String preferred) {
        String preference = preferred != null ? preferred : config.preferredBackend();
        if ("cuda".equalsIgnoreCase(preference)) {
            var cuda = Backends.cuda();
            if (cuda.isPresent()) {
                return cuda.get();
            }
            if (!config.allowCudaFallback()) {
                throw new IllegalStateException("CUDA backend requested but unavailable");
            }
            LOG.warning("CUDA backend unavailable, falling back to CPU");
        }
        return Backends.cpu().orElseGet(() -> {
            for (ComputeBackend backend : Backends.discover()) {
                return backend;
            }
            throw new IllegalStateException("No compute backend available");
        });
    }

    public void unload(String id, boolean force) {
        ModelInstance instance = instances.get(id);
        if (instance == null) {
            throw new IllegalArgumentException("Model not loaded: " + id);
        }
        if (instance.activeSessions() > 0 && !force) {
            throw new IllegalStateException("Model " + id + " has " + instance.activeSessions()
                    + " active session(s); cannot unload without force");
        }
        instance.state(ModelInstance.State.UNLOADING);
        instances.remove(id);
        LOG.info(() -> "Unloaded model " + id);
    }

    public ModelInstance require(String id) {
        ModelInstance instance = instances.get(id);
        if (instance == null) {
            throw new IllegalArgumentException("Model not loaded: " + id);
        }
        return instance;
    }

    /**
     * Resolves a model id (file name without extension, as used by the family
     * {@code /enr/v1} surface) to a model file in the configured model
     * directories. Returns {@code null} when no matching file exists.
     */
    public Path resolveModelPath(String id) throws IOException {
        for (Path candidate : discoverModelFiles()) {
            if (id.equalsIgnoreCase(modelIdFromPath(candidate))) {
                return candidate;
            }
        }
        return null;
    }

    public ModelInstance get(String id) {
        return instances.get(id);
    }

    public List<ModelInstance> list() {
        return instances.values().stream()
                .sorted(Comparator.comparing(ModelInstance::id))
                .toList();
    }

    public Map<String, ModelInstance> instances() {
        return new LinkedHashMap<>(instances);
    }

    /**
     * Evicts idle models until at most {@code maxResident} remain or until
     * free weight bytes reaches {@code targetFreeBytes}. Only idle models
     * (no active sessions) are candidates.
     */
    public List<String> evictIdle(int maxResident, long targetFreeBytes) {
        List<String> evicted = new ArrayList<>();
        List<ModelInstance> candidates = list().stream()
                .filter(m -> m.activeSessions() == 0)
                .sorted(Comparator.comparingLong(ModelInstance::loadedAt).reversed())
                .toList();
        long totalBytes = totalWeightBytes();
        for (ModelInstance candidate : candidates) {
            if (instances.size() - evicted.size() <= maxResident && totalBytes <= targetFreeBytes) {
                break;
            }
            unload(candidate.id(), false);
            totalBytes -= candidate.totalWeightBytes();
            evicted.add(candidate.id());
        }
        return evicted;
    }

    public long totalWeightBytes() {
        long total = 0;
        for (ModelInstance instance : instances.values()) {
            total += instance.totalWeightBytes();
        }
        return total;
    }

    public long totalLoadedWeightBytes() {
        long total = 0;
        for (ModelInstance instance : instances.values()) {
            total += instance.weightBytesLoaded();
        }
        return total;
    }

    public void shutdown() {
        for (ModelInstance instance : instances.values()) {
            instance.state(ModelInstance.State.UNLOADING);
        }
        instances.clear();
    }

    private static String modelIdFromPath(Path path) {
        String fileName = path.getFileName().toString();
        int dot = fileName.lastIndexOf('.');
        return dot > 0 ? fileName.substring(0, dot) : fileName;
    }
}
