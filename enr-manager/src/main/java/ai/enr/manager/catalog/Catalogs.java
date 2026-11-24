package ai.enr.manager.catalog;

import ai.enr.core.json.Json;
import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonObject;
import ai.enr.core.json.JsonValue;
import ai.enr.manager.EnrConfig;
import ai.enr.manager.logs.LogBuffer;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/**
 * Model catalogs: search and download resolution for the Ollama Library
 * (registry.ollama.ai) and Hugging Face. Both base URLs are configurable so
 * tests and air-gapped deployments can point at private registries.
 */
public final class Catalogs implements AutoCloseable {

    public record CatalogModel(String id, String source, String description, long sizeBytes) {
    }

    /** A resolved, downloadable artifact with a verifiable digest. */
    public record ResolvedDownload(String name, String url, String digest, long sizeBytes) {
    }

    public record SearchResult(List<CatalogModel> models, List<String> errors) {
    }

    private final EnrConfig config;
    private final LogBuffer log;
    private final HttpClient http;

    public Catalogs(EnrConfig config, LogBuffer log) {
        this(config, log, HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(15))
                .followRedirects(HttpClient.Redirect.ALWAYS)
                .build());
    }

    public Catalogs(EnrConfig config, LogBuffer log, HttpClient http) {
        this.config = config;
        this.log = log;
        this.http = http;
    }

    public SearchResult search(String query, int limit) {
        List<CatalogModel> models = new ArrayList<>();
        List<String> errors = new ArrayList<>();
        try {
            models.addAll(searchOllama(query, limit));
        } catch (Exception e) {
            errors.add("Ollama Library: " + shortMessage(e));
        }
        try {
            models.addAll(searchHuggingFace(query, limit));
        } catch (Exception e) {
            errors.add("Hugging Face: " + shortMessage(e));
        }
        if (models.isEmpty() && errors.isEmpty()) {
            errors.add("No models found for query: " + query);
        }
        return new SearchResult(models, errors);
    }

    /**
     * Resolves a pull reference to a concrete artifact URL + digest.
     * Forms: {@code ollama:NAME[:TAG]} or plain {@code NAME[:TAG]} for the
     * Ollama Library; {@code hf:OWNER/REPO:QUANT} (quantization discovery) and
     * {@code hf:OWNER/REPO/FILE.gguf} (explicit file) for Hugging Face. The
     * {@code hf.co/} and {@code huggingface.co/} prefixes are also accepted.
     */
    public ResolvedDownload resolveDownload(String ref) throws IOException {
        if (ref == null || ref.isBlank()) {
            throw new IllegalArgumentException("Empty model reference");
        }
        if (ref.startsWith("hf:")) {
            return resolveHuggingFace(ref.substring(3).strip());
        }
        if (ref.startsWith("hf.co/") || ref.startsWith("huggingface.co/")) {
            return resolveHuggingFace(ref.substring(ref.indexOf('/') + 1).strip());
        }
        String name = ref;
        if (name.startsWith("ollama:")) {
            name = name.substring("ollama:".length()).strip();
        }
        if (name.contains("/") && !name.startsWith("library/")) {
            throw new IllegalArgumentException("Invalid Ollama model reference '" + ref
                    + "'; use NAME[:TAG] (e.g. 'llama3.2:1b') or hf:OWNER/REPO:QUANT for Hugging Face");
        }
        return resolveOllama(name);
    }

    // ---------------------------------------------------------------- Ollama

    private List<CatalogModel> searchOllama(String query, int limit) throws Exception {
        String searchUrl = stripTrailingSlash(config.ollamaSearchUrl())
                + "/library?q=" + URLEncoder.encode(query == null ? "" : query, StandardCharsets.UTF_8);
        List<CatalogModel> models = new ArrayList<>();
        String html = fetchString(searchUrl);
        // The library page is server-rendered HTML; model links are /library/<name>.
        // A JSON response (mock registries, tests) is also accepted.
        String trimmed = html.strip();
        if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
            JsonValue parsed = Json.parse(trimmed);
            JsonArray arr;
            if (parsed instanceof JsonArray direct) {
                arr = direct;
            } else if (parsed instanceof JsonObject obj && obj.getArray("models") != null) {
                arr = obj.getArray("models");
            } else {
                return models;
            }
            for (int i = 0; i < arr.size() && models.size() < limit; i++) {
                JsonObject entry = arr.getObject(i);
                if (entry != null && entry.getString("name") != null) {
                    models.add(new CatalogModel(entry.getString("name"), "ollama",
                            entry.getString("description", ""), 0));
                }
            }
            return models;
        }
        java.util.LinkedHashSet<String> seen = new java.util.LinkedHashSet<>();
        for (String href : html.split("href=\"/library/")) {
            if (href.length() < 2) {
                continue;
            }
            String name = href.substring(0, Math.min(href.length(), href.indexOf('"') > 0
                    ? href.indexOf('"') : href.length()));
            name = name.strip().replaceAll("[\"'<>]", "");
            if (name.matches("[a-zA-Z0-9][a-zA-Z0-9._-]*") && seen.add(name) && models.size() < limit) {
                models.add(new CatalogModel(name, "ollama", "", 0));
            }
        }
        return models;
    }

    private ResolvedDownload resolveOllama(String ref) throws IOException {
        String name = ref;
        String tag = "latest";
        int colon = ref.lastIndexOf(':');
        if (colon > 0) {
            name = ref.substring(0, colon);
            tag = ref.substring(colon + 1);
        }
        String registry = stripTrailingSlash(config.ollamaRegistryUrl());
        String manifestUrl = registry + "/v2/library/" + URLEncoder.encode(name, StandardCharsets.UTF_8)
                + "/manifests/" + URLEncoder.encode(tag, StandardCharsets.UTF_8);
        JsonObject manifest;
        try {
            manifest = Json.parseObject(fetchString(manifestUrl));
        } catch (IOException e) {
            throw new IOException("Failed to fetch manifest for '" + ref + "' from " + registry
                    + ": " + shortMessage(e), e);
        }
        JsonArray layers = manifest.getArray("layers");
        if (layers == null) {
            throw new IOException("Manifest for '" + ref + "' has no layers");
        }
        for (int i = 0; i < layers.size(); i++) {
            JsonObject layer = layers.getObject(i);
            if (layer == null) {
                continue;
            }
            String mediaType = layer.getString("mediaType", "");
            if (mediaType.contains("model")) {
                String digest = layer.getString("digest", "");
                long size = layer.getLong("size", 0);
                String blobUrl = registry + "/v2/library/" + URLEncoder.encode(name, StandardCharsets.UTF_8)
                        + "/blobs/" + digest;
                return new ResolvedDownload(ref, blobUrl, digestSha(digest), size);
            }
        }
        throw new IOException("Manifest for '" + ref + "' contains no model layer");
    }

    private static String digestSha(String digest) {
        return digest.startsWith("sha256:") ? digest.substring("sha256:".length()) : "";
    }

    // ---------------------------------------------------------------- Hugging Face

    private List<CatalogModel> searchHuggingFace(String query, int limit) throws Exception {
        String url = config.huggingFaceApiUrl() + "/api/models?search="
                + URLEncoder.encode(query == null ? "" : query, StandardCharsets.UTF_8)
                + "&limit=" + limit + "&filter=gguf";
        JsonValue parsed = fetchJson(url);
        List<CatalogModel> models = new ArrayList<>();
        JsonArray arr = parsed instanceof JsonArray direct
                ? direct
                : parsed instanceof JsonObject obj && obj.getArray("models") != null
                ? obj.getArray("models") : null;
        if (arr == null) {
            return models;
        }
        for (int i = 0; i < arr.size() && models.size() < limit; i++) {
            JsonObject entry = arr.getObject(i);
            if (entry == null) {
                continue;
            }
            String id = entry.getString("id");
            if (id == null) {
                continue;
            }
            models.add(new CatalogModel(id, "huggingface",
                    entry.getString("pipeline_tag", ""), 0));
        }
        return models;
    }

    private ResolvedDownload resolveHuggingFace(String ref) throws IOException {
        // Accepted forms after the hf: prefix is stripped:
        //   OWNER/REPO:QUANT            -> discover the GGUF for QUANT
        //   OWNER/REPO/FILE.gguf        -> explicit file (subdirs allowed)
        int colon = ref.lastIndexOf(':');
        if (colon > 0 && ref.indexOf('/', colon) < 0) {
            String repo = ref.substring(0, colon);
            String quant = ref.substring(colon + 1);
            return resolveHuggingFaceQuant(repo, quant);
        }
        String[] parts = ref.split("/");
        if (parts.length < 3) {
            throw new IllegalArgumentException("Hugging Face reference must be OWNER/REPO:QUANT "
                    + "or OWNER/REPO/FILE.gguf, got: " + ref);
        }
        String repo = parts[0] + "/" + parts[1];
        String filePath = String.join("/", java.util.Arrays.copyOfRange(parts, 2, parts.length));
        return resolveHuggingFaceFile(repo, filePath);
    }

    /**
     * Discovers the GGUF file for a quantization tag within a repository. A
     * requested quantization must resolve to exactly one file; ambiguity or
     * absence produces a clear error listing the available options.
     */
    private ResolvedDownload resolveHuggingFaceQuant(String repo, String quant) throws IOException {
        if (quant == null || quant.isBlank()) {
            throw new IllegalArgumentException("Missing quantization tag for Hugging Face repo '" + repo
                    + "' (use OWNER/REPO:QUANT)");
        }
        List<JsonObject> ggufs = listGgufFiles(repo);
        if (ggufs.isEmpty()) {
            throw new IOException("No GGUF files found in Hugging Face repo '" + repo + "'");
        }
        List<JsonObject> matches = new ArrayList<>();
        for (JsonObject f : ggufs) {
            if (matchesQuant(f.getString("path", ""), quant)) {
                matches.add(f);
            }
        }
        if (matches.isEmpty()) {
            List<String> available = new ArrayList<>();
            for (JsonObject f : ggufs) {
                available.add(quantOf(f.getString("path", "")));
            }
            throw new IOException("Quantization '" + quant + "' not found in repo '" + repo
                    + "'. Available: " + String.join(", ", available));
        }
        if (matches.size() > 1) {
            List<String> files = new ArrayList<>();
            for (JsonObject f : matches) {
                files.add(f.getString("path", ""));
            }
            throw new IOException("Quantization '" + quant + "' is ambiguous in repo '" + repo
                    + "'; matches: " + String.join(", ", files));
        }
        JsonObject file = matches.get(0);
        String path = file.getString("path", "");
        String base = stripTrailingSlash(config.huggingFaceApiUrl());
        String downloadUrl = base + "/" + repo + "/resolve/main/" + path;
        JsonObject lfs = file.getObject("lfs");
        String digest = lfs == null ? "" : lfs.getString("oid", "");
        long size = lfs == null ? file.getLong("size", 0) : lfs.getLong("size", file.getLong("size", 0));
        return new ResolvedDownload(repo + ":" + quant, downloadUrl, digest, size);
    }

    private ResolvedDownload resolveHuggingFaceFile(String repo, String filePath) throws IOException {
        String base = stripTrailingSlash(config.huggingFaceApiUrl());
        String treeUrl = base + "/api/models/" + repo + "/tree/main/"
                + URLEncoder.encode(filePath, StandardCharsets.UTF_8);
        String digest = "";
        long size = 0;
        try {
            JsonValue parsed = fetchJson(treeUrl);
            JsonObject entry = parsed instanceof JsonArray arr && arr.size() > 0
                    ? arr.getObject(0) : null;
            if (entry != null) {
                JsonObject lfs = entry.getObject("lfs");
                if (lfs != null) {
                    digest = lfs.getString("oid", "");
                    size = lfs.getLong("size", 0);
                }
            }
        } catch (IOException e) {
            log.warn("catalogs", "Could not read Hugging Face file metadata for " + repo + "/" + filePath
                    + ": " + shortMessage(e) + " — downloading without digest verification");
        }
        String downloadUrl = base + "/" + repo + "/resolve/main/" + filePath;
        return new ResolvedDownload(repo + "/" + filePath, downloadUrl, digest, size);
    }

    /** Lists the GGUF files at the root of a repository via the HF tree API. */
    private List<JsonObject> listGgufFiles(String repo) throws IOException {
        String base = stripTrailingSlash(config.huggingFaceApiUrl());
        String treeUrl = base + "/api/models/" + repo + "/tree/main";
        JsonValue parsed = fetchJson(treeUrl);
        List<JsonObject> files = new ArrayList<>();
        if (parsed instanceof JsonArray arr) {
            for (int i = 0; i < arr.size(); i++) {
                JsonObject entry = arr.getObject(i);
                if (entry == null) {
                    continue;
                }
                String type = entry.getString("type", "");
                String path = entry.getString("path", "");
                if ("file".equals(type) && path != null && path.toLowerCase().endsWith(".gguf")
                        && !path.toLowerCase().contains("mmproj")) {
                    files.add(entry);
                }
            }
        }
        return files;
    }

    /** Whether a GGUF file path carries the requested quantization tag. */
    public static boolean matchesQuant(String path, String quant) {
        if (path == null || quant == null) {
            return false;
        }
        String base = path.toLowerCase();
        if (base.endsWith(".gguf")) {
            base = base.substring(0, base.length() - ".gguf".length());
        }
        String t = quant.toLowerCase().strip();
        if (t.isEmpty()) {
            return false;
        }
        return base.equals(t) || base.endsWith("-" + t) || base.endsWith("_" + t)
                || base.contains("-" + t + "-") || base.contains("-" + t + ".");
    }

    /** Extracts a display quantization from a GGUF file name for error messages. */
    private static String quantOf(String path) {
        String base = path == null ? "" : path;
        int slash = base.lastIndexOf('/');
        if (slash >= 0) {
            base = base.substring(slash + 1);
        }
        if (base.toLowerCase().endsWith(".gguf")) {
            base = base.substring(0, base.length() - 5);
        }
        return base;
    }

    // ---------------------------------------------------------------- shared

    private JsonValue fetchJson(String url) throws IOException {
        return Json.parse(fetchString(url));
    }

    private String fetchString(String url) throws IOException {
        try {
            HttpRequest.Builder builder = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .header("User-Agent", "enr/2.6")
                    .header("Accept", "application/json")
                    .GET();
            String token = config.huggingFaceToken();
            if (!token.isBlank() && url.startsWith(stripTrailingSlash(config.huggingFaceApiUrl()))) {
                builder.header("Authorization", "Bearer " + token);
            }
            HttpResponse<String> response = http.send(builder.build(), HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                throw new IOException("HTTP " + response.statusCode() + " for " + url);
            }
            return response.body();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while fetching " + url, e);
        }
    }

    private static String stripTrailingSlash(String url) {
        String stripped = url == null ? "" : url;
        while (stripped.endsWith("/")) {
            stripped = stripped.substring(0, stripped.length() - 1);
        }
        return stripped;
    }

    private static String shortMessage(Exception e) {
        String message = e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
        return message.length() > 200 ? message.substring(0, 200) : message;
    }

    @Override
    public void close() {
    }
}
