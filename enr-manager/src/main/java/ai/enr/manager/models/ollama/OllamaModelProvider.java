package ai.enr.manager.models.ollama;

import ai.enr.core.json.Json;
import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonObject;
import ai.enr.core.json.JsonValue;
import ai.enr.manager.models.DownloadLayer;
import ai.enr.manager.models.DownloadProgress;
import ai.enr.manager.models.DownloadState;
import ai.enr.manager.models.ModelDescriptor;
import ai.enr.manager.models.ModelProvider;
import ai.enr.manager.models.ProviderCapability;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

/**
 * Ollama model provider speaking Ollama's native HTTP API:
 * {@code GET /api/tags} (installed models), {@code GET /api/show}
 * (metadata), {@code POST /api/pull} with {@code stream: true}
 * (registry downloads with per-layer NDJSON progress),
 * {@code DELETE /api/delete} (removal) and {@code GET /api/version}
 * (connectivity). The provider does not manage download lifecycle; it only
 * translates the Ollama wire protocol into {@link DownloadProgress} events.
 */
public final class OllamaModelProvider implements ModelProvider, AutoCloseable {

    public static final String DEFAULT_ENDPOINT = "http://localhost:11434";
    public static final String PROVIDER_TYPE = "ollama";

    private static final Set<ProviderCapability> CAPABILITIES = Set.of(
            ProviderCapability.MODEL_PULL,
            ProviderCapability.MODEL_LIST,
            ProviderCapability.MODEL_REMOVE,
            ProviderCapability.DOWNLOAD_PROGRESS,
            ProviderCapability.MODEL_METADATA);

    private final String id;
    private final String endpoint;
    private final long connectTimeoutMs;
    private final long readTimeoutMs;
    private final long downloadTimeoutMs;
    private final HttpClient http;
    private final ScheduledExecutorService watchdog =
            Executors.newSingleThreadScheduledExecutor(r -> {
                Thread thread = new Thread(r, "enr-ollama-watchdog");
                thread.setDaemon(true);
                return thread;
            });
    private final Map<String, InputStream> activeBodies = new ConcurrentHashMap<>();
    private volatile boolean closed = false;

    public OllamaModelProvider(String id, String endpoint, long connectTimeoutMs,
                               long readTimeoutMs, long downloadTimeoutMs) {
        this.id = id == null || id.isBlank() ? PROVIDER_TYPE : id;
        this.endpoint = endpoint == null || endpoint.isBlank() ? DEFAULT_ENDPOINT : endpoint;
        this.connectTimeoutMs = connectTimeoutMs > 0 ? connectTimeoutMs : 10_000;
        this.readTimeoutMs = readTimeoutMs > 0 ? readTimeoutMs : 60_000;
        this.downloadTimeoutMs = downloadTimeoutMs;
        this.http = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.ALWAYS)
                .connectTimeout(Duration.ofMillis(this.connectTimeoutMs))
                .build();
    }

    @Override
    public String id() {
        return id;
    }

    @Override
    public String displayName() {
        return "Ollama";
    }

    public String endpoint() {
        return endpoint;
    }

    @Override
    public Set<ProviderCapability> capabilities() {
        return CAPABILITIES;
    }

    @Override
    public boolean ping() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(endpoint + "/api/version"))
                    .timeout(Duration.ofMillis(Math.max(connectTimeoutMs, 5_000)))
                    .GET()
                    .build();
            HttpResponse<String> response = http.send(request,
                    HttpResponse.BodyHandlers.ofString());
            return response.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public List<ModelDescriptor> listInstalled() throws Exception {
        HttpResponse<String> response = send(HttpRequest.newBuilder()
                .uri(URI.create(endpoint + "/api/tags"))
                .GET()
                .build());
        if (response.statusCode() != 200) {
            throw error(response);
        }
        JsonValue parsed = Json.parse(response.body());
        if (!(parsed instanceof JsonObject root)) {
            throw new IOException("Ollama /api/tags returned no JSON object");
        }
        List<ModelDescriptor> models = new ArrayList<>();
        JsonArray arr = root.getArray("models");
        if (arr == null) {
            return models;
        }
        for (int i = 0; i < arr.size(); i++) {
            JsonObject model = arr.getObject(i);
            if (model == null) {
                continue;
            }
            String name = model.getString("name", "");
            if (name.isBlank()) {
                continue;
            }
            models.add(new ModelDescriptor(id, name, name,
                    model.getString("digest", ""),
                    model.getLong("size", -1),
                    parseModifiedAt(model.getString("modified_at", "")),
                    true, detailsFrom(model.getObject("details"))));
        }
        return models;
    }

    @Override
    public List<ModelDescriptor> listAvailable() {
        // The Ollama server itself has no availability catalog; discovery is
        // served by ENR's catalog subsystem (ollama.com search).
        return List.of();
    }

    @Override
    public ModelDescriptor describe(String modelId) throws Exception {
        JsonObject body = new JsonObject();
        body.put("name", modelId);
        HttpResponse<String> response = send(HttpRequest.newBuilder()
                .uri(URI.create(endpoint + "/api/show"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(Json.write(body, false)))
                .build());
        if (response.statusCode() != 200) {
            throw error(response);
        }
        JsonValue parsed = Json.parse(response.body());
        if (!(parsed instanceof JsonObject root)) {
            throw new IOException("Ollama /api/show returned no JSON object");
        }
        JsonObject modelInfo = root.getObject("model_info");
        Map<String, String> details = new LinkedHashMap<>();
        if (modelInfo != null) {
            details.put("architecture", modelInfoString(modelInfo, "general.architecture"));
            details.put("parameter_count", modelInfoString(modelInfo, "general.parameter_count"));
            details.put("context_length", modelInfoString(modelInfo, "general.context_length"));
            details.put("file_type", modelInfoString(modelInfo, "general.file_type"));
        }
        String model = root.getString("model", modelId);
        return new ModelDescriptor(id, modelId, model, root.getString("digest", ""),
                root.getLong("size", -1), root.getLong("modified_at", 0), true, details);
    }

    @Override
    public void pull(String modelId, BooleanSupplier cancellationRequested,
                     Consumer<DownloadProgress> progress) throws Exception {
        if (closed) {
            throw new IOException("Provider " + id + " is closed");
        }
        JsonObject body = new JsonObject();
        body.put("name", modelId);
        body.put("stream", true);
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(endpoint + "/api/pull"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(Json.write(body, false)))
                .build();
        HttpResponse<InputStream> response;
        try {
            response = http.send(request, HttpResponse.BodyHandlers.ofInputStream());
        } catch (Exception e) {
            if (cancellationRequested.getAsBoolean()) {
                return;
            }
            throw new IOException("Ollama pull request failed for " + modelId + ": "
                    + e.getMessage(), e);
        }
        if (response.statusCode() != 200) {
            try (InputStream in = response.body()) {
                String errorBody = new String(in.readAllBytes(), StandardCharsets.UTF_8);
                throw new IOException("Ollama pull failed with HTTP " + response.statusCode()
                        + ": " + extractError(errorBody));
            }
        }
        try (InputStream in = response.body()) {
            activeBodies.put(modelId, in);
            consumePullStream(modelId, in, cancellationRequested, progress);
        } finally {
            activeBodies.remove(modelId);
        }
    }

    private void consumePullStream(String modelId, InputStream in,
                                   BooleanSupplier cancellationRequested,
                                   Consumer<DownloadProgress> progress) throws Exception {
        BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
        Map<String, long[]> layers = new LinkedHashMap<>();
        long startedAt = System.currentTimeMillis();
        String line;
        ScheduledFuture<?> idleWatch = armIdleWatchdog(modelId, in);
        try {
            while ((line = reader.readLine()) != null) {
                if (cancellationRequested.getAsBoolean()) {
                    progress.accept(new DownloadProgress(DownloadState.CANCEL_REQUESTED,
                            -1, 0, "cancelled by user", List.of()));
                    return;
                }
                idleWatch.cancel(false);
                JsonValue parsed = Json.parse(line.strip());
                if (!(parsed instanceof JsonObject event)) {
                    continue;
                }
                String status = event.getString("status", "");
                String error = event.getString("error", "");
                if (!error.isBlank()) {
                    throw new IOException("Ollama pull error for " + modelId + ": " + error);
                }
                DownloadState state = mapStatus(status);
                String digest = event.getString("digest", "");
                if (!digest.isBlank() && event.has("total") && event.has("completed")) {
                    long total = event.getLong("total", -1);
                    long completed = event.getLong("completed", 0);
                    layers.put(digest, new long[]{total, completed});
                }
                long totalBytes = 0;
                long completedBytes = 0;
                boolean anyKnown = false;
                List<DownloadLayer> snapshots = new ArrayList<>();
                for (Map.Entry<String, long[]> entry : layers.entrySet()) {
                    long[] layer = entry.getValue();
                    if (layer[0] > 0) {
                        anyKnown = true;
                        totalBytes += layer[0];
                    }
                    completedBytes += layer[1];
                    snapshots.add(new DownloadLayer(entry.getKey(), layer[0], layer[1], status));
                }
                if (state == DownloadState.COMPLETED) {
                    long total = totalBytes > 0 ? totalBytes : completedBytes;
                    long completed = completedBytes > 0 ? completedBytes : total;
                    progress.accept(new DownloadProgress(DownloadState.COMPLETED,
                            total, completed, "", snapshots));
                    return;
                }
                progress.accept(new DownloadProgress(state,
                        anyKnown ? totalBytes : -1, completedBytes, "", snapshots));
                if (downloadTimeoutMs > 0
                        && System.currentTimeMillis() - startedAt > downloadTimeoutMs) {
                    throw new IOException("Download timeout for " + modelId + " after "
                            + downloadTimeoutMs + " ms");
                }
                idleWatch = armIdleWatchdog(modelId, in);
            }
            // Ollama normally terminates with a "success" event; an end of
            // stream without success is a truncated transfer.
            throw new IOException("Ollama pull stream ended without a success event for " + modelId);
        } finally {
            idleWatch.cancel(false);
        }
    }

    private ScheduledFuture<?> armIdleWatchdog(String modelId, InputStream body) {
        return watchdog.schedule(() -> {
            try {
                body.close();
            } catch (IOException ignored) {
                // closing the stream aborts the read loop
            }
        }, readTimeoutMs, TimeUnit.MILLISECONDS);
    }

    private static DownloadState mapStatus(String status) {
        if (status == null || status.isBlank()) {
            return DownloadState.DOWNLOADING;
        }
        if (status.startsWith("success")) {
            return DownloadState.COMPLETED;
        }
        if (status.startsWith("pulling manifest")) {
            return DownloadState.CONNECTING;
        }
        if (status.startsWith("verifying") || status.startsWith("writing")) {
            return DownloadState.VERIFYING;
        }
        return DownloadState.DOWNLOADING;
    }

    @Override
    public void cancel(String modelId) {
        InputStream body = activeBodies.get(modelId);
        if (body != null) {
            try {
                body.close();
            } catch (IOException ignored) {
                // closing the stream aborts the pull loop
            }
        }
    }

    @Override
    public void remove(String modelId) throws Exception {
        JsonObject body = new JsonObject();
        body.put("name", modelId);
        HttpResponse<String> response = send(HttpRequest.newBuilder()
                .uri(URI.create(endpoint + "/api/delete"))
                .header("Content-Type", "application/json")
                .method("DELETE", HttpRequest.BodyPublishers.ofString(Json.write(body, false)))
                .build());
        if (response.statusCode() == 404) {
            throw new IllegalArgumentException("Model not installed: " + modelId);
        }
        if (response.statusCode() != 200) {
            throw error(response);
        }
    }

    private HttpResponse<String> send(HttpRequest request) throws IOException, InterruptedException {
        return http.send(request, HttpResponse.BodyHandlers.ofString());
    }

    private IOException error(HttpResponse<String> response) {
        return new IOException("Ollama HTTP " + response.statusCode() + ": "
                + extractError(response.body()));
    }

    private static String extractError(String body) {
        try {
            JsonValue parsed = Json.parse(body == null ? "" : body);
            if (parsed instanceof JsonObject obj) {
                String error = obj.getString("error", "");
                if (!error.isBlank()) {
                    return error;
                }
            }
        } catch (RuntimeException ignored) {
            // not JSON: use the raw body
        }
        return body == null ? "" : body.strip();
    }

    private static Map<String, String> detailsFrom(JsonObject details) {
        Map<String, String> result = new LinkedHashMap<>();
        if (details == null) {
            return result;
        }
        putDetail(result, details, "family");
        putDetail(result, details, "parameter_size");
        putDetail(result, details, "quantization_level");
        putDetail(result, details, "format");
        putDetail(result, details, "context_length");
        return result;
    }

    private static void putDetail(Map<String, String> target, JsonObject source, String key) {
        String value = source.getString(key, "");
        if (!value.isBlank()) {
            target.put(key, value);
        }
    }

    private static String modelInfoString(JsonObject modelInfo, String key) {
        String value = modelInfo.getString(key, "");
        if (!value.isBlank()) {
            return value;
        }
        return modelInfo.has(key) ? String.valueOf(modelInfo.getLong(key, 0)) : "";
    }

    private static long parseModifiedAt(String value) {
        if (value == null || value.isBlank()) {
            return 0;
        }
        try {
            return Instant.parse(value).toEpochMilli();
        } catch (RuntimeException e) {
            return 0;
        }
    }

    @Override
    public void close() {
        closed = true;
        for (InputStream body : activeBodies.values()) {
            try {
                body.close();
            } catch (IOException ignored) {
                // best effort during shutdown
            }
        }
        activeBodies.clear();
        watchdog.shutdownNow();
    }
}
