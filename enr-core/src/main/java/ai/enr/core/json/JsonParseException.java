package ai.enr.core.json;

/**
 * Thrown when a JSON document cannot be parsed (malformed input). Subclasses
 * {@link IllegalArgumentException} so existing catch blocks that treat JSON
 * parsing as a request error keep working, while protocol handlers can
 * distinguish a malformed body (HTTP 400) from a semantic "not found" error
 * (HTTP 404).
 */
public final class JsonParseException extends IllegalArgumentException {

    public JsonParseException(String message) {
        super(message);
    }
}
