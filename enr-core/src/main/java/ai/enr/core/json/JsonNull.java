package ai.enr.core.json;

public enum JsonNull implements JsonValue {
    INSTANCE;

    @Override
    public Type type() {
        return Type.NULL;
    }
}
