package ai.enr.manager.provider;

import ai.enr.core.json.JsonObject;

import java.time.Instant;
import java.util.List;

/**
 * One configured provider. The API key is a secret: it is stored in the
 * state file (0600) but never returned by {@link #toJson()} and never logged.
 */
public record Provider(
        String id,
        ProviderType type,
        String name,
        String baseUrl,
        String apiKey,
        boolean enabled,
        List<String> models,
        long modelsRefreshedAt,
        long lastTestAt,
        boolean lastTestOk,
        String lastTestMessage,
        long createdAt) {

    public Provider(String id, ProviderType type, String name, String baseUrl, String apiKey,
                    boolean enabled) {
        this(id, type, name, baseUrl, apiKey, enabled, List.of(), 0, 0, false, "",
                System.currentTimeMillis());
    }

    public Provider withModels(List<String> models) {
        return new Provider(id, type, name, baseUrl, apiKey, enabled, models,
                System.currentTimeMillis(), lastTestAt, lastTestOk, lastTestMessage, createdAt);
    }

    public Provider withTestResult(boolean ok, String message) {
        return new Provider(id, type, name, baseUrl, apiKey, enabled, models, modelsRefreshedAt,
                System.currentTimeMillis(), ok, message, createdAt);
    }

    public Provider withConfig(String newName, String newBaseUrl, String newApiKey, boolean newEnabled) {
        return new Provider(id, type,
                newName == null ? name : newName,
                newBaseUrl == null ? baseUrl : newBaseUrl,
                newApiKey == null ? apiKey : newApiKey,
                newEnabled, models, modelsRefreshedAt, lastTestAt, lastTestOk, lastTestMessage,
                createdAt);
    }

    /** JSON for API responses and persistence. API keys are redacted. */
    public JsonObject toJson() {
        JsonObject obj = new JsonObject();
        obj.put("id", id);
        obj.put("type", type.id());
        obj.put("name", name);
        obj.put("base_url", baseUrl);
        obj.put("api_key_configured", apiKey != null && !apiKey.isBlank());
        obj.put("enabled", enabled);
        obj.put("local", type.local());
        ai.enr.core.json.JsonArray modelArr = new ai.enr.core.json.JsonArray();
        for (String model : models) {
            modelArr.add(model);
        }
        obj.put("models", modelArr);
        obj.put("models_refreshed_at", modelsRefreshedAt);
        obj.put("last_test_at", lastTestAt);
        obj.put("last_test_ok", lastTestOk);
        obj.put("last_test_message", lastTestMessage == null ? "" : lastTestMessage);
        obj.put("created_at", createdAt);
        obj.put("connected", lastTestOk && System.currentTimeMillis() - lastTestAt < 300_000);
        return obj;
    }

    public JsonObject toJsonWithSecret() {
        JsonObject obj = toJson();
        obj.put("api_key", apiKey == null ? "" : apiKey);
        return obj;
    }

    public static Provider fromJson(JsonObject obj) {
        String typeId = obj.getString("type", "");
        if (typeId.isBlank()) {
            throw new IllegalArgumentException("Provider entry missing 'type': " + obj);
        }
        String id = obj.getString("id", typeId);
        return new Provider(id, ProviderType.fromId(typeId),
                obj.getString("name", id),
                obj.getString("base_url", ""),
                obj.getString("api_key", ""),
                obj.getBoolean("enabled", true));
    }
}
