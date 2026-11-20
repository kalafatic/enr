package ai.enr.manager.provider;

/**
 * Unified chat message used across local inference and every cloud provider
 * wire protocol.
 */
public record ChatMessage(String role, String content) {

    public static ChatMessage of(String role, String content) {
        return new ChatMessage(role == null ? "user" : role, content == null ? "" : content);
    }
}
