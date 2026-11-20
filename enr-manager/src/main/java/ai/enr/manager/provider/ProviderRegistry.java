package ai.enr.manager.provider;

import ai.enr.core.json.Json;
import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonObject;
import ai.enr.manager.logs.LogBuffer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermission;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Provider registry: the single source of truth for all provider
 * configuration. Persists to {@code <dataDir>/state.json} with owner-only
 * permissions. API keys are stored in that file but redacted from every API
 * response and log line.
 */
public final class ProviderRegistry {

    private static final long MODEL_CACHE_TTL_MS = 60_000;

    private final Path stateFile;
    private final LogBuffer log;
    private final Map<String, Provider> providers = new ConcurrentHashMap<>();

    public ProviderRegistry(Path dataDir, LogBuffer log) {
        this.stateFile = dataDir.resolve("state.json");
        this.log = log;
        load();
    }

    public synchronized Provider put(Provider provider) {
        providers.put(provider.id(), provider);
        persist();
        log.info("providers", "Provider configured: " + provider.id() + " (" + provider.type().id() + ")");
        return provider;
    }

    public synchronized Provider remove(String id) {
        Provider removed = providers.remove(id);
        if (removed == null) {
            throw new IllegalArgumentException("Provider not found: " + id);
        }
        persist();
        log.info("providers", "Provider removed: " + id);
        return removed;
    }

    public Provider get(String id) {
        return providers.get(id);
    }

    public Provider require(String id) {
        Provider provider = providers.get(id);
        if (provider == null) {
            throw new IllegalArgumentException("Provider not configured: " + id);
        }
        return provider;
    }

    public List<Provider> list() {
        return providers.values().stream().sorted(Comparator.comparing(Provider::id)).toList();
    }

    public List<Provider> enabled() {
        return list().stream().filter(Provider::enabled).toList();
    }

    /**
     * Tests connectivity by asking the provider for its model list.
     * Stores the result on the provider record.
     */
    public ProviderClient.TestResult test(Provider provider) {
        long started = System.currentTimeMillis();
        try (ProviderClient client = new ProviderClient()) {
            List<String> models = client.listModels(provider);
            Provider updated = provider.withModels(models).withTestResult(true,
                    "OK — " + models.size() + " model(s) discovered");
            put(updated);
            log.info("providers", "Provider " + provider.id() + " connection test OK ("
                    + models.size() + " models, " + (System.currentTimeMillis() - started) + " ms)");
            return new ProviderClient.TestResult(true, updated.lastTestMessage(),
                    System.currentTimeMillis() - started, models);
        } catch (Exception e) {
            String message = e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
            Provider updated = provider.withTestResult(false, message);
            put(updated);
            log.warn("providers", "Provider " + provider.id() + " connection test FAILED: " + message);
            return new ProviderClient.TestResult(false, message,
                    System.currentTimeMillis() - started, List.of());
        }
    }

    /** Refreshes the provider's model list; cached for 60 seconds. */
    public synchronized List<String> models(Provider provider) {
        long now = System.currentTimeMillis();
        if (!provider.models().isEmpty() && now - provider.modelsRefreshedAt() < MODEL_CACHE_TTL_MS) {
            return provider.models();
        }
        try (ProviderClient client = new ProviderClient()) {
            List<String> models = client.listModels(provider);
            put(provider.withModels(models).withTestResult(true, "OK"));
            return models;
        } catch (Exception e) {
            String message = e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
            put(provider.withTestResult(false, message));
            log.warn("providers", "Failed to list models for " + provider.id() + ": " + message);
            return provider.models();
        }
    }

    /** Redacted listing for API responses. */
    public JsonObject toJson() {
        JsonObject root = new JsonObject();
        JsonArray arr = new JsonArray();
        for (Provider provider : list()) {
            arr.add(provider.toJson());
        }
        root.put("providers", arr);
        return root;
    }

    private void load() {
        if (!Files.exists(stateFile)) {
            return;
        }
        try {
            JsonObject root = Json.parseObject(Files.readString(stateFile));
            JsonArray arr = root.getArray("providers");
            if (arr != null) {
                for (int i = 0; i < arr.size(); i++) {
                    JsonObject entry = arr.getObject(i);
                    if (entry == null) {
                        continue;
                    }
                    try {
                        Provider provider = Provider.fromJson(entry);
                        providers.put(provider.id(), provider);
                    } catch (IllegalArgumentException e) {
                        log.warn("providers", "Skipping invalid provider entry: " + e.getMessage());
                    }
                }
            }
        } catch (Exception e) {
            log.error("providers", "Failed to load provider state from " + stateFile + ": " + e.getMessage());
        }
    }

    private void persist() {
        try {
            Files.createDirectories(stateFile.getParent());
            JsonObject root = new JsonObject();
            JsonArray arr = new JsonArray();
            for (Provider provider : list()) {
                arr.add(provider.toJsonWithSecret());
            }
            root.put("providers", arr);
            Files.writeString(stateFile, Json.write(root, true));
            try {
                Files.setPosixFilePermissions(stateFile, Set.of(
                        PosixFilePermission.OWNER_READ, PosixFilePermission.OWNER_WRITE));
            } catch (UnsupportedOperationException ignored) {
                // non-POSIX filesystem
            }
        } catch (Exception e) {
            log.error("providers", "Failed to persist provider state: " + e.getMessage());
        }
    }
}
