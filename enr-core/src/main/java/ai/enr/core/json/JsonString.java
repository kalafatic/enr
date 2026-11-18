package ai.enr.core.json;

public record JsonString(String value) implements JsonValue {
    @Override
    public Type type() {
        return Type.STRING;
    }
}
