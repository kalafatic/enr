package ai.enr.core.json;

import java.util.ArrayList;
import java.util.List;

public final class JsonArray implements JsonValue {

    private final ArrayList<JsonValue> items = new ArrayList<>();

    @Override
    public Type type() {
        return Type.ARRAY;
    }

    public JsonArray add(JsonValue value) {
        items.add(value == null ? JsonNull.INSTANCE : value);
        return this;
    }

    public JsonArray add(String value) {
        items.add(value == null ? JsonNull.INSTANCE : new JsonString(value));
        return this;
    }

    public JsonArray add(long value) {
        items.add(new JsonNumber(value));
        return this;
    }

    public JsonArray add(double value) {
        items.add(new JsonNumber(value));
        return this;
    }

    public JsonArray add(int value) {
        items.add(new JsonNumber(value));
        return this;
    }

    public JsonArray add(boolean value) {
        items.add(value ? JsonBoolean.TRUE : JsonBoolean.FALSE);
        return this;
    }

    public JsonValue get(int index) {
        return items.get(index);
    }

    public JsonObject getObject(int index) {
        JsonValue v = items.get(index);
        return v instanceof JsonObject o ? o : null;
    }

    public String getString(int index) {
        JsonValue v = items.get(index);
        return v instanceof JsonString s ? s.value() : null;
    }

    public long getLong(int index) {
        JsonValue v = items.get(index);
        if (v instanceof JsonNumber n) {
            return n.asLong();
        }
        throw new IllegalArgumentException("JSON array element " + index + " is not a number");
    }

    public int getInt(int index) {
        return (int) getLong(index);
    }

    public int size() {
        return items.size();
    }

    public List<JsonValue> items() {
        return items;
    }
}
