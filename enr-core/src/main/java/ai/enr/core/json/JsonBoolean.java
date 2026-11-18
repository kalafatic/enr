package ai.enr.core.json;

public record JsonBoolean(boolean value) implements JsonValue {

    public static final JsonBoolean TRUE = new JsonBoolean(true);
    public static final JsonBoolean FALSE = new JsonBoolean(false);

    @Override
    public Type type() {
        return Type.BOOLEAN;
    }
}
