package ai.enr.core.json;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

public final class JsonObject implements JsonValue {

    private final LinkedHashMap<String, JsonValue> entries = new LinkedHashMap<>();

    @Override
    public Type type() {
        return Type.OBJECT;
    }

    public JsonObject put(String key, JsonValue value) {
        if (key == null) {
            throw new IllegalArgumentException("JSON key cannot be null");
        }
        if (value == null) {
            value = JsonNull.INSTANCE;
        }
        entries.put(key, value);
        return this;
    }

    public JsonObject put(String key, String value) {
        return put(key, value == null ? JsonNull.INSTANCE : new JsonString(value));
    }

    public JsonObject put(String key, long value) {
        return put(key, new JsonNumber(value));
    }

    public JsonObject put(String key, double value) {
        return put(key, new JsonNumber(value));
    }

    public JsonObject put(String key, float value) {
        return put(key, new JsonNumber(value));
    }

    public JsonObject put(String key, int value) {
        return put(key, new JsonNumber(value));
    }

    public JsonObject put(String key, boolean value) {
        return put(key, value ? JsonBoolean.TRUE : JsonBoolean.FALSE);
    }

    public boolean has(String key) {
        return entries.containsKey(key);
    }

    public JsonValue get(String key) {
        return entries.get(key);
    }

    public JsonValue require(String key) {
        JsonValue v = entries.get(key);
        if (v == null) {
            throw new IllegalArgumentException("Missing required JSON field: " + key);
        }
        return v;
    }

    public String getString(String key) {
        JsonValue v = entries.get(key);
        return v instanceof JsonString s ? s.value() : null;
    }

    public String getString(String key, String defaultValue) {
        String s = getString(key);
        return s != null ? s : defaultValue;
    }

    public int getInt(String key, int defaultValue) {
        JsonValue v = entries.get(key);
        if (v instanceof JsonNumber n) {
            return (int) n.asLong();
        }
        return defaultValue;
    }

    public long getLong(String key, long defaultValue) {
        JsonValue v = entries.get(key);
        if (v instanceof JsonNumber n) {
            return n.asLong();
        }
        return defaultValue;
    }

    public double getDouble(String key, double defaultValue) {
        JsonValue v = entries.get(key);
        if (v instanceof JsonNumber n) {
            return n.asDouble();
        }
        return defaultValue;
    }

    public float getFloat(String key, float defaultValue) {
        JsonValue v = entries.get(key);
        if (v instanceof JsonNumber n) {
            return (float) n.asDouble();
        }
        return defaultValue;
    }

    public boolean getBoolean(String key, boolean defaultValue) {
        JsonValue v = entries.get(key);
        if (v instanceof JsonBoolean b) {
            return b.value();
        }
        return defaultValue;
    }

    public JsonObject getObject(String key) {
        JsonValue v = entries.get(key);
        return v instanceof JsonObject o ? o : null;
    }

    public JsonArray getArray(String key) {
        JsonValue v = entries.get(key);
        return v instanceof JsonArray a ? a : null;
    }

    public Set<Map.Entry<String, JsonValue>> entries() {
        return entries.entrySet();
    }

    public int size() {
        return entries.size();
    }

    @Override
    public String toString() {
        return Json.write(this, false);
    }

    public String toPrettyString() {
        return Json.write(this, true);
    }
}
