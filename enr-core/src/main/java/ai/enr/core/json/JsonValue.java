package ai.enr.core.json;

public interface JsonValue {

    Type type();

    enum Type {
        OBJECT, ARRAY, STRING, NUMBER, BOOLEAN, NULL
    }
}
