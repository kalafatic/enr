package ai.enr.manager.models;

import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonObject;

import java.util.ArrayList;
import java.util.List;

/**
 * Immutable snapshot of a download job. Progress is computed from the actual
 * provider-reported byte counters (sum of layer totals and completions), never
 * by averaging per-layer percentages.
 */
public record DownloadStatus(
        String jobId,
        String providerId,
        String modelId,
        DownloadState state,
        String operation,
        long totalBytes,
        long completedBytes,
        int percent,
        long startedAt,
        long updatedAt,
        long completedAt,
        String error,
        List<DownloadLayer> layers) {

    public DownloadStatus {
        operation = operation == null ? "" : operation;
        error = error == null ? "" : error;
        layers = layers == null ? List.of() : List.copyOf(layers);
    }

    public boolean terminal() {
        return state != null && state.terminal();
    }

    public static int percent(long totalBytes, long completedBytes) {
        if (totalBytes <= 0 || completedBytes <= 0) {
            return 0;
        }
        long value = completedBytes * 100L / totalBytes;
        return (int) Math.max(0, Math.min(100, value));
    }

    public JsonObject toJson() {
        JsonObject obj = new JsonObject();
        obj.put("job_id", jobId);
        obj.put("provider", providerId);
        obj.put("model", modelId);
        obj.put("status", state == null ? "" : state.name());
        obj.put("operation", operation);
        obj.put("total_bytes", totalBytes);
        obj.put("completed_bytes", completedBytes);
        obj.put("percent", percent);
        obj.put("started_at", startedAt);
        obj.put("updated_at", updatedAt);
        obj.put("completed_at", completedAt);
        obj.put("error", error);
        JsonArray layerArr = new JsonArray();
        for (DownloadLayer layer : layers) {
            layerArr.add(layer.toJson());
        }
        obj.put("layers", layerArr);
        return obj;
    }

    public static DownloadStatus fromJson(JsonObject obj) {
        List<DownloadLayer> layers = new ArrayList<>();
        JsonArray arr = obj.getArray("layers");
        if (arr != null) {
            for (int i = 0; i < arr.size(); i++) {
                JsonObject layer = arr.getObject(i);
                if (layer != null) {
                    layers.add(DownloadLayer.fromJson(layer));
                }
            }
        }
        DownloadState state;
        try {
            state = DownloadState.valueOf(obj.getString("status", "FAILED"));
        } catch (IllegalArgumentException e) {
            state = DownloadState.FAILED;
        }
        return new DownloadStatus(
                obj.getString("job_id", ""),
                obj.getString("provider", ""),
                obj.getString("model", ""),
                state,
                obj.getString("operation", ""),
                obj.getLong("total_bytes", -1),
                obj.getLong("completed_bytes", 0),
                obj.getInt("percent", 0),
                obj.getLong("started_at", 0),
                obj.getLong("updated_at", 0),
                obj.getLong("completed_at", 0),
                obj.getString("error", ""),
                layers);
    }
}
