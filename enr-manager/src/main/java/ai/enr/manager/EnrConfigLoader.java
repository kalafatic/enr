package ai.enr.manager;

import ai.enr.core.json.Json;
import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonObject;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * YAML-subset configuration loader for enr.yaml (flat key: value, two-level
 * sections, lists). Full YAML is deliberately not implemented.
 */
public final class EnrConfigLoader {

    private EnrConfigLoader() {
    }

    public static EnrConfig load(Path path) throws IOException {
        return load(path == null ? new JsonObject() : parseYaml(path));
    }

    public static EnrConfig load(JsonObject config) {
        EnrConfig.Builder builder = EnrConfig.builder();
        JsonObject server = config.getObject("server");
        if (server != null) {
            builder.host(server.getString("host", builder.build().host()));
            builder.port(server.getInt("port", builder.build().port()));
            builder.authToken(server.getString("auth_token", ""));
        }
        JsonObject runtime = config.getObject("runtime");
        if (runtime != null) {
            builder.preferredBackend(runtime.getString("preferred_backend", "cpu"));
            builder.allowCudaFallback(runtime.getBoolean("allow_cuda_fallback", true));
            builder.maxConcurrentSessions(runtime.getInt("max_concurrent_sessions", 8));
            builder.defaultMaxTokens(runtime.getInt("default_max_tokens", 256));
            builder.sessionIdleTimeoutSeconds(runtime.getInt("session_idle_timeout_seconds", 1800));
            builder.requestTimeoutSeconds(runtime.getInt("request_timeout_seconds", 300));
            builder.enableMetrics(runtime.getBoolean("enable_metrics", true));
            builder.modelDirectories(pathList(runtime.getArray("model_directories")));
        }
        JsonObject storage = config.getObject("storage");
        if (storage != null) {
            if (storage.has("data_dir")) {
                builder.dataDir(Path.of(storage.getString("data_dir")).toAbsolutePath());
            }
            if (storage.has("models_dir")) {
                builder.modelsDir(Path.of(storage.getString("models_dir")).toAbsolutePath());
            }
            if (storage.has("log_dir")) {
                builder.logDir(Path.of(storage.getString("log_dir")).toAbsolutePath());
            }
        }
        JsonObject catalogs = config.getObject("catalogs");
        if (catalogs != null) {
            builder.ollamaRegistryUrl(catalogs.getString("ollama_registry", "https://registry.ollama.ai"));
            builder.ollamaSearchUrl(catalogs.getString("ollama_search", "https://ollama.com"));
            builder.huggingFaceApiUrl(catalogs.getString("huggingface_api", "https://huggingface.co"));
            builder.downloadTimeoutMs(catalogs.getLong("download_timeout_ms", 0));
        }
        return builder.build();
    }

    public static JsonObject parseYaml(Path path) throws IOException {
        if (path == null || !Files.exists(path)) {
            return new JsonObject();
        }
        JsonObject root = new JsonObject();
        JsonObject current = root;
        JsonArray pendingList = null;
        List<String> lines = Files.readAllLines(path);
        for (int lineIndex = 0; lineIndex < lines.size(); lineIndex++) {
            String line = lines.get(lineIndex).strip();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            if (line.startsWith("- ") && pendingList != null) {
                pendingList.add(stripQuotes(line.substring(2).strip()));
                continue;
            }
            pendingList = null;
            if (line.endsWith(":") && !line.contains(": ")) {
                String name = line.substring(0, line.length() - 1).strip();
                String next = nextNonEmpty(lines, lineIndex);
                if (next != null && next.strip().startsWith("- ")) {
                    JsonArray arr = new JsonArray();
                    current.put(name, arr);
                    pendingList = arr;
                    continue;
                }
                JsonObject sectionObj = new JsonObject();
                root.put(name, sectionObj);
                current = sectionObj;
                continue;
            }
            int colon = line.indexOf(':');
            if (colon < 0) {
                continue;
            }
            String key = line.substring(0, colon).strip();
            String value = line.substring(colon + 1).strip();
            if (value.isEmpty()) {
                JsonArray arr = new JsonArray();
                current.put(key, arr);
                pendingList = arr;
                continue;
            }
            if (value.startsWith("[") && value.endsWith("]")) {
                JsonArray arr = new JsonArray();
                String inner = value.substring(1, value.length() - 1).strip();
                if (!inner.isEmpty()) {
                    for (String item : inner.split(",")) {
                        arr.add(stripQuotes(item.strip()));
                    }
                }
                current.put(key, arr);
            } else {
                putScalar(current, key, value);
            }
        }
        return root;
    }

    private static List<Path> pathList(JsonArray arr) {
        List<Path> paths = new ArrayList<>();
        if (arr != null) {
            for (int i = 0; i < arr.size(); i++) {
                paths.add(Path.of(arr.getString(i)));
            }
        }
        return paths;
    }

    private static String stripQuotes(String value) {
        if ((value.startsWith("\"") && value.endsWith("\""))
                || (value.startsWith("'") && value.endsWith("'"))) {
            return value.substring(1, value.length() - 1);
        }
        return value;
    }

    private static String nextNonEmpty(List<String> lines, int from) {
        for (int i = from + 1; i < lines.size(); i++) {
            String line = lines.get(i).strip();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            return line;
        }
        return null;
    }

    private static void putScalar(JsonObject obj, String key, String value) {
        String unquoted = value;
        if ((unquoted.startsWith("\"") && unquoted.endsWith("\""))
                || (unquoted.startsWith("'") && unquoted.endsWith("'"))) {
            obj.put(key, unquoted.substring(1, unquoted.length() - 1));
            return;
        }
        if ("true".equalsIgnoreCase(unquoted)) {
            obj.put(key, true);
            return;
        }
        if ("false".equalsIgnoreCase(unquoted)) {
            obj.put(key, false);
            return;
        }
        try {
            obj.put(key, Long.parseLong(unquoted));
            return;
        } catch (NumberFormatException ignored) {
            // fall through
        }
        obj.put(key, unquoted);
    }
}
