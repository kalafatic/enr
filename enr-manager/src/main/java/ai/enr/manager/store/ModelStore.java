package ai.enr.manager.store;

import ai.enr.core.io.Hashing;
import ai.enr.core.json.Json;
import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonObject;
import ai.enr.manager.EnrConfig;
import ai.enr.manager.logs.LogBuffer;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

/**
 * Persistent installed-model registry. Each installed model lives in
 * {@code <modelsDir>/<id>/} with its artifact file and {@code manifest.json}.
 * Files present in the configured model directories (bare .gguf/.evo) are
 * discovered as installed models as well. This is the single source of truth
 * for what is installed locally; runtime load state is merged by EnrManager.
 */
public final class ModelStore {

    public record StoredModel(ModelManifest manifest, boolean artifactPresent) {

        public String id() {
            return manifest.id();
        }
    }

    private final EnrConfig config;
    private final LogBuffer log;

    public ModelStore(EnrConfig config, LogBuffer log) {
        this.config = config;
        this.log = log;
    }

    /** All installed models: managed model directories plus discovered bare files. */
    public List<StoredModel> list() {
        List<StoredModel> models = new ArrayList<>();
        Path modelsDir = config.modelsDir();
        if (Files.isDirectory(modelsDir)) {
            try (Stream<Path> dirs = Files.list(modelsDir)) {
                for (Path dir : dirs.filter(Files::isDirectory).sorted().toList()) {
                    ModelManifest manifest = ModelManifest.read(dir);
                    if (manifest == null) {
                        continue;
                    }
                    models.add(new StoredModel(manifest, Files.isRegularFile(manifest.modelFile())));
                }
            } catch (IOException e) {
                log.warn("model-store", "Failed to list model directory " + modelsDir + ": " + e.getMessage());
            }
        }
        for (Path file : discoverBareFiles()) {
            String id = idFromFileName(file.getFileName().toString());
            if (models.stream().anyMatch(m -> m.id().equals(id))) {
                continue;
            }
            ModelManifest manifest = new ModelManifest(id, file.getFileName().toString(),
                    formatOf(file), "", safeSize(file), "local-scan", safeMtime(file), false,
                    file.getParent());
            models.add(new StoredModel(manifest, Files.isRegularFile(file)));
        }
        // Ollama installation models: manifest-registered models plus any bare
        // .gguf/.evo artifacts placed directly in the Ollama models directory.
        // They are referenced in place (never copied) and share the same
        // registry, so ENR and Ollama see the same model identifiers.
        for (StoredModel ollama : discoverOllamaModels()) {
            if (models.stream().noneMatch(m -> m.id().equals(ollama.id()))) {
                models.add(ollama);
            }
        }
        models.sort(Comparator.comparing(StoredModel::id));
        return models;
    }

    /**
     * Discovers the models installed by the local Ollama installation from the
     * configured Ollama models directory: manifest-registered models
     * ({@code manifests/<registry>/<namespace>/<model>/<tag>} → model blob) plus
     * bare {@code .gguf}/{@code .evo} artifacts at the directory root. Both the
     * user-level ({@code ~/.ollama/models}) and the standard system-level
     * ({@code /usr/share/ollama/.ollama/models}, Debian/Ubuntu Ollama package)
     * locations are scanned so ENR sees the same models as the configured Ollama.
     */
    private List<StoredModel> discoverOllamaModels() {
        List<StoredModel> models = new ArrayList<>();
        for (Path ollamaDir : candidateOllamaDirs()) {
            collectOllamaDir(ollamaDir, models);
        }
        return models;
    }

    /** Candidate Ollama model-storage directories: configured + standard system location. */
    private List<Path> candidateOllamaDirs() {
        List<Path> dirs = new ArrayList<>();
        Path configured = config.ollamaModelsDir();
        if (configured != null) {
            dirs.add(configured);
        }
        // The Debian/Ubuntu Ollama package stores models under /usr/share/ollama;
        // include that location as a fallback only when the configured directory
        // is the default user location, so an explicit custom configuration (for
        // example test isolation) is honoured exactly.
        if (configured != null && configured.equals(defaultUserOllamaDir())) {
            Path system = Path.of("/usr/share/ollama/.ollama/models");
            if (Files.isDirectory(system) && !dirs.contains(system)) {
                dirs.add(system);
            }
        }
        return dirs;
    }

    private static Path defaultUserOllamaDir() {
        return Path.of(System.getProperty("user.home"), ".ollama", "models");
    }

    private void collectOllamaDir(Path ollamaDir, List<StoredModel> models) {
        if (ollamaDir == null || !Files.isDirectory(ollamaDir)) {
            return;
        }
        Path manifestsDir = ollamaDir.resolve("manifests");
        if (Files.isDirectory(manifestsDir)) {
            try (Stream<Path> walk = Files.walk(manifestsDir)) {
                for (Path manifest : walk.filter(Files::isRegularFile).toList()) {
                    StoredModel model = ollamaManifestModel(manifestsDir, manifest, ollamaDir);
                    if (model != null && models.stream().noneMatch(m -> m.id().equals(model.id()))) {
                        models.add(model);
                    }
                }
            } catch (IOException e) {
                log.warn("model-store", "Failed to scan Ollama manifests " + manifestsDir + ": "
                        + e.getMessage());
            }
        }
        try (Stream<Path> stream = Files.list(ollamaDir)) {
            stream.filter(Files::isRegularFile)
                    .filter(p -> {
                        String name = p.getFileName().toString().toLowerCase(Locale.ROOT);
                        return name.endsWith(".gguf") || name.endsWith(".evo");
                    })
                    .sorted()
                    .forEach(file -> {
                        String id = idFromFileName(file.getFileName().toString());
                        if (models.stream().anyMatch(m -> m.id().equals(id))) {
                            return;
                        }
                        models.add(new StoredModel(ModelManifest.of(id,
                                file.getFileName().toString(), formatOf(file), "", safeSize(file),
                                "ollama-local", safeMtime(file), file.getParent()),
                                Files.isRegularFile(file)));
                    });
        } catch (IOException e) {
            log.warn("model-store", "Failed to scan Ollama models directory " + ollamaDir + ": "
                    + e.getMessage());
        }
    }

    /** Resolves one Ollama manifest file into a model entry pointing at its model blob. */
    private static StoredModel ollamaManifestModel(Path manifestsDir, Path manifest, Path ollamaDir) {
        Path relative = manifestsDir.relativize(manifest);
        int count = relative.getNameCount();
        if (count < 3) {
            return null;
        }
        String namespace = relative.getName(1).toString();
        String model = relative.getName(count - 2).toString();
        String tag = relative.getName(count - 1).toString();
        String modelName = "library".equals(namespace)
                ? model + ":" + tag
                : namespace + "/" + model + ":" + tag;
        String digest = modelLayerDigest(manifest);
        if (digest == null || digest.isBlank()) {
            return null;
        }
        Path blob = ollamaDir.resolve("blobs").resolve("sha256-" + digest);
        if (!Files.isRegularFile(blob)) {
            return null;
        }
        String id = sanitizeId(modelName);
        ModelManifest manifestObj = ModelManifest.of(id, blob.getFileName().toString(), "gguf",
                digest, safeSize(blob), "ollama-registry", safeMtime(blob), blob.getParent());
        return new StoredModel(manifestObj, true);
    }

    /** The SHA-256 hex digest of a manifest's model layer (GGUF blob). */
    private static String modelLayerDigest(Path manifestFile) {
        try {
            JsonObject obj = Json.parseObject(Files.readString(manifestFile));
            JsonArray layers = obj.getArray("layers");
            if (layers == null) {
                return null;
            }
            for (int i = 0; i < layers.size(); i++) {
                JsonObject layer = layers.getObject(i);
                if (layer == null) {
                    continue;
                }
                if (layer.getString("mediaType", "").contains("model")) {
                    String digest = layer.getString("digest", "");
                    return digest.startsWith("sha256:") ? digest.substring("sha256:".length())
                            : digest;
                }
            }
        } catch (Exception ignored) {
            // unreadable or malformed manifest: skip this model
        }
        return null;
    }

    public StoredModel find(String id) {
        return list().stream().filter(m -> m.id().equals(id)).findFirst().orElse(null);
    }

    public Path modelFile(String id) {
        StoredModel model = find(id);
        return model == null ? null : model.manifest().modelFile();
    }

    /**
     * Imports an existing local GGUF/EVO file into the model store. The file
     * is copied under the models directory and a manifest with a recorded
     * SHA-256 digest is written.
     */
    public ModelManifest importModel(Path source, String requestedId) throws IOException {
        if (source == null || !Files.isRegularFile(source)) {
            throw new IllegalArgumentException("Import source is not a regular file: " + source);
        }
        String fileName = source.getFileName().toString();
        String format = formatOf(source);
        if (!("gguf".equals(format) || "evo".equals(format))) {
            throw new IllegalArgumentException("Unsupported model format: " + fileName
                    + " (expected .gguf or .evo)");
        }
        String id = requestedId != null && !requestedId.isBlank()
                ? requestedId : idFromFileName(fileName);
        id = sanitizeId(id);
        Path targetDir = config.modelsDir().resolve(id);
        Path targetFile = targetDir.resolve(id + "." + format);
        if (Files.exists(targetFile)) {
            throw new IllegalStateException("Model already installed: " + id);
        }
        Files.createDirectories(targetDir);
        Files.copy(source, targetFile, StandardCopyOption.REPLACE_EXISTING);
        String digest = Hashing.sha256Hex(Files.readAllBytes(targetFile));
        ModelManifest manifest = ModelManifest.of(id, targetFile.getFileName().toString(), format,
                digest, Files.size(targetFile), "import", System.currentTimeMillis(), targetDir);
        manifest.write();
        log.info("model-store", "Imported model " + id + " (" + digest + ", "
                + manifest.sizeBytes() + " bytes)");
        return manifest;
    }

    /** Finalizes a catalog download that landed in {@code finalFile}. */
    public ModelManifest registerDownload(String id, Path finalFile, String digest, long sizeBytes,
                                          String source) throws IOException {
        id = sanitizeId(id);
        Path targetDir = config.modelsDir().resolve(id);
        String fileName = id + "." + formatOf(finalFile);
        Path targetFile = targetDir.resolve(fileName);
        Files.createDirectories(targetDir);
        if (!Files.isSameFile(finalFile, targetFile)) {
            Files.move(finalFile, targetFile, StandardCopyOption.REPLACE_EXISTING);
        }
        ModelManifest manifest = ModelManifest.of(id, fileName, formatOf(finalFile), digest,
                sizeBytes > 0 ? sizeBytes : Files.size(targetFile), source,
                System.currentTimeMillis(), targetDir);
        manifest.write();
        log.info("model-store", "Registered downloaded model " + id + " (" + digest + ")");
        return manifest;
    }

    /** Removes an installed model (files + manifest). */
    public void remove(String id) throws IOException {
        StoredModel model = find(id);
        if (model == null) {
            throw new IllegalArgumentException("Model not installed: " + id);
        }
        Path dir = model.manifest().directory();
        if (dir == null || dir.getParent() == null || !dir.startsWith(config.modelsDir())) {
            throw new IllegalStateException("Refusing to remove externally discovered model: " + id
                    + " (remove the file manually from its source directory)");
        }
        try (Stream<Path> walk = Files.walk(dir)) {
            for (Path path : walk.sorted(Comparator.reverseOrder()).toList()) {
                Files.deleteIfExists(path);
            }
        }
        log.info("model-store", "Removed model " + id);
    }

    /** Verifies the artifact against the recorded digest. */
    public boolean verify(String id) {
        StoredModel model = find(id);
        if (model == null) {
            throw new IllegalArgumentException("Model not installed: " + id);
        }
        Path file = model.manifest().modelFile();
        if (!Files.isRegularFile(file)) {
            return false;
        }
        String digest = model.manifest().digest();
        if (digest == null || digest.isBlank()) {
            return false;
        }
        try {
            return digest.equals(Hashing.sha256Hex(Files.readAllBytes(file)));
        } catch (IOException e) {
            log.warn("model-store", "Verification read failed for " + id + ": " + e.getMessage());
            return false;
        }
    }

    public JsonObject toJson() {
        JsonObject root = new JsonObject();
        JsonArray models = new JsonArray();
        for (StoredModel model : list()) {
            JsonObject obj = model.manifest().toJson();
            obj.put("artifact_present", model.artifactPresent());
            models.add(obj);
        }
        root.put("models", models);
        root.put("models_dir", config.modelsDir().toString());
        return root;
    }

    private List<Path> discoverBareFiles() {
        List<Path> found = new ArrayList<>();
        for (Path dir : config.modelDirectories()) {
            if (!Files.isDirectory(dir)) {
                continue;
            }
            try (Stream<Path> stream = Files.list(dir)) {
                stream.filter(Files::isRegularFile)
                        .filter(p -> {
                            String name = p.getFileName().toString().toLowerCase(Locale.ROOT);
                            return name.endsWith(".evo") || name.endsWith(".gguf");
                        })
                        .sorted()
                        .forEach(found::add);
            } catch (IOException e) {
                log.warn("model-store", "Failed to scan model directory " + dir + ": " + e.getMessage());
            }
        }
        return found;
    }

    public static String idFromFileName(String fileName) {
        int dot = fileName.lastIndexOf('.');
        return dot > 0 ? fileName.substring(0, dot) : fileName;
    }

    public static String formatOf(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.endsWith(".gguf")) {
            return "gguf";
        }
        if (name.endsWith(".evo")) {
            return "evo";
        }
        return "";
    }

    public static String sanitizeId(String id) {
        String clean = id.replaceAll("[^A-Za-z0-9._:-]", "-");
        return clean.isBlank() ? "model" : clean;
    }

    private static long safeSize(Path file) {
        try {
            return Files.size(file);
        } catch (IOException e) {
            return -1;
        }
    }

    private static long safeMtime(Path file) {
        try {
            return Files.getLastModifiedTime(file).toMillis();
        } catch (IOException e) {
            return 0;
        }
    }
}
