package ai.enr.manager.models;

import ai.enr.core.json.JsonObject;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Provider-neutral model metadata. Fields that a provider cannot supply remain
 * empty or -1 rather than being guessed.
 */
public record ModelDescriptor(
        String providerId,
        String id,
        String name,
        String digest,
        long sizeBytes,
        long modifiedAt,
        boolean installed,
        Map<String, String> details) {

    public ModelDescriptor {
        providerId = providerId == null ? "" : providerId;
        id = id == null ? "" : id;
        name = name == null || name.isBlank() ? id : name;
        digest = digest == null ? "" : digest;
        details = details == null ? Map.of() : Map.copyOf(details);
    }

    public JsonObject toJson() {
        JsonObject obj = new JsonObject();
        obj.put("provider", providerId);
        obj.put("id", id);
        obj.put("name", name);
        obj.put("digest", digest);
        obj.put("size_bytes", sizeBytes);
        obj.put("modified_at", modifiedAt);
        obj.put("installed", installed);
        JsonObject detailObj = new JsonObject();
        for (Map.Entry<String, String> entry : new LinkedHashMap<>(details).entrySet()) {
            detailObj.put(entry.getKey(), entry.getValue());
        }
        obj.put("details", detailObj);
        return obj;
    }

    public static ModelDescriptor fromJson(String providerId, JsonObject obj) {
        Map<String, String> details = new LinkedHashMap<>();
        JsonObject detailObj = obj.getObject("details");
        if (detailObj != null) {
            for (Map.Entry<String, ai.enr.core.json.JsonValue> entry : detailObj.entries()) {
                if (entry.getValue() instanceof ai.enr.core.json.JsonString value) {
                    details.put(entry.getKey(), value.value());
                }
            }
        }
        return new ModelDescriptor(
                obj.getString("provider", providerId),
                obj.getString("id", ""),
                obj.getString("name", ""),
                obj.getString("digest", ""),
                obj.getLong("size_bytes", -1),
                obj.getLong("modified_at", 0),
                obj.getBoolean("installed", false),
                details);
    }
}
