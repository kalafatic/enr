package ai.enr.manager.provider;

import ai.enr.core.json.JsonObject;

/**
 * Unified chat message used across local inference and every cloud provider
 * wire protocol. The optional payload carries protocol-specific extras
 * (e.g. {@code tool_calls} for assistant messages, {@code tool_call_id} for
 * tool results); it is null for plain text messages. Text-only consumers
 * (local prompt building) only read role/content.
 */
public record ChatMessage(String role, String content, JsonObject payload) {

    public static ChatMessage of(String role, String content) {
        return new ChatMessage(role == null ? "user" : role, content == null ? "" : content, null);
    }

    public static ChatMessage withPayload(String role, String content, JsonObject payload) {
        return new ChatMessage(role == null ? "user" : role, content == null ? "" : content, payload);
    }
}
