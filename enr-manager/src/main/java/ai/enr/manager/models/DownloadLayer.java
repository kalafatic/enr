package ai.enr.manager.models;

import ai.enr.core.json.JsonObject;

/**
 * One downloadable layer reported by a provider (for Ollama this is one blob
 * digest). Totals are provider-reported and may change during a pull.
 */
public record DownloadLayer(
        String digest,
        long totalBytes,
        long completedBytes,
        String status) {

    public DownloadLayer {
        digest = digest == null ? "" : digest;
        status = status == null ? "" : status;
    }

    public JsonObject toJson() {
        JsonObject obj = new JsonObject();
        obj.put("digest", digest);
        obj.put("total_bytes", totalBytes);
        obj.put("completed_bytes", completedBytes);
        obj.put("status", status);
        return obj;
    }

    public static DownloadLayer fromJson(JsonObject obj) {
        return new DownloadLayer(
                obj.getString("digest", ""),
                obj.getLong("total_bytes", -1),
                obj.getLong("completed_bytes", 0),
                obj.getString("status", ""));
    }
}
