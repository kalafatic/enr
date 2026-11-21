package ai.enr.manager.models;

import ai.enr.core.json.JsonObject;

/**
 * Handle returned when a model pull is accepted. The job id is stable for the
 * lifetime of the download and is the key used by status and cancellation APIs.
 */
public record DownloadJob(
        String jobId,
        String providerId,
        String modelId,
        DownloadState state,
        long createdAt) {

    public JsonObject toJson() {
        JsonObject obj = new JsonObject();
        obj.put("jobId", jobId);
        obj.put("job_id", jobId);
        obj.put("provider", providerId);
        obj.put("model", modelId);
        obj.put("status", state == null ? "" : state.name());
        obj.put("created_at", createdAt);
        return obj;
    }
}
