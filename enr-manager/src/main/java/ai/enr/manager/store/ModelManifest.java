package ai.enr.manager.store;

import ai.enr.core.json.Json;
import ai.enr.core.json.JsonObject;

import java.nio.file.Path;

/**
 * Persistent metadata for one installed local model.
 */
public record ModelManifest(
        String id,
        String fileName,
        String format,
        String digest,
        long sizeBytes,
        String source,
        long installedAt,
        boolean verified,
        Path directory) {

    public static ModelManifest of(String id, String fileName, String format, String digest,
                                   long sizeBytes, String source, long installedAt, Path directory) {
        return new ModelManifest(id, fileName, format, digest, sizeBytes, source, installedAt,
                digest != null && !digest.isBlank(), directory);
    }

    public Path modelFile() {
        return directory.resolve(fileName);
    }

    public JsonObject toJson() {
        JsonObject obj = new JsonObject();
        obj.put("id", id);
        obj.put("file_name", fileName);
        obj.put("format", format);
        obj.put("digest", digest == null ? "" : digest);
        obj.put("size_bytes", sizeBytes);
        obj.put("source", source == null ? "" : source);
        obj.put("installed_at", installedAt);
        obj.put("verified", verified);
        return obj;
    }

    public static ModelManifest fromJson(JsonObject obj, Path directory) {
        return new ModelManifest(
                obj.getString("id", ""),
                obj.getString("file_name", "model.gguf"),
                obj.getString("format", "gguf"),
                obj.getString("digest", ""),
                obj.getLong("size_bytes", 0),
                obj.getString("source", ""),
                obj.getLong("installed_at", 0),
                obj.getBoolean("verified", false),
                directory);
    }

    public static ModelManifest read(Path directory) {
        Path manifestPath = directory.resolve("manifest.json");
        try {
            JsonObject obj = Json.parseObject(java.nio.file.Files.readString(manifestPath));
            return fromJson(obj, directory);
        } catch (Exception e) {
            return null;
        }
    }

    public void write() {
        Path manifestPath = directory.resolve("manifest.json");
        try {
            java.nio.file.Files.createDirectories(directory);
            java.nio.file.Files.writeString(manifestPath, Json.write(toJson(), true));
        } catch (Exception e) {
            throw new IllegalStateException("Failed to write model manifest " + manifestPath, e);
        }
    }
}
