package ai.enr.manager.store;

import ai.enr.core.io.Hashing;
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
        models.sort(Comparator.comparing(StoredModel::id));
        return models;
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
