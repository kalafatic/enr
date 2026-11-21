package ai.enr.manager.models;

import java.util.Map;

/**
 * Connection settings handed to a {@link ModelProviderFactory}. Values are
 * resolved from ENR configuration and provider registry state.
 */
public record ModelProviderConfig(
        String providerId,
        String endpoint,
        long connectTimeoutMs,
        long readTimeoutMs,
        long downloadTimeoutMs,
        Map<String, String> options) {

    public ModelProviderConfig {
        providerId = providerId == null ? "" : providerId;
        endpoint = endpoint == null ? "" : endpoint;
        options = options == null ? Map.of() : Map.copyOf(options);
    }
}
