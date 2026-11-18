package ai.enr.cli;

import ai.enr.core.json.Json;
import ai.enr.core.json.JsonObject;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Minimal YAML-subset configuration loader (flat key: value and two-level
 * sections). Full YAML deliberately not implemented (see technology decisions).
 */
public final class CliConfig {

    private CliConfig() {
    }

    public static JsonObject load(Path path) throws IOException {
        if (path == null || !Files.exists(path)) {
            return new JsonObject();
        }
        JsonObject root = new JsonObject();
        JsonObject current = root;
        ai.enr.core.json.JsonArray pendingList = null;
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
                    var arr = new ai.enr.core.json.JsonArray();
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
                var arr = new ai.enr.core.json.JsonArray();
                current.put(key, arr);
                pendingList = arr;
                continue;
            }
            if (value.startsWith("[") && value.endsWith("]")) {
                var arr = new ai.enr.core.json.JsonArray();
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

    public static List<Path> modelDirectories(JsonObject config) {
        List<Path> dirs = new ArrayList<>();
        JsonObject runtime = config.getObject("runtime");
        if (runtime != null) {
            var arr = runtime.getArray("model_directories");
            if (arr != null) {
                for (int i = 0; i < arr.size(); i++) {
                    dirs.add(Path.of(arr.getString(i)));
                }
            }
        }
        return dirs;
    }
}
