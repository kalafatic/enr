package ai.enr.core.json;

public record JsonNumber(double value) implements JsonValue {

    public JsonNumber(long value) {
        this((double) value);
    }

    @Override
    public Type type() {
        return Type.NUMBER;
    }

    public long asLong() {
        return (long) value;
    }

    public double asDouble() {
        return value;
    }
}
