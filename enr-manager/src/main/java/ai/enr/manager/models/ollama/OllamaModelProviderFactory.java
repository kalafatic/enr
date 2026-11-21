package ai.enr.manager.models.ollama;

import ai.enr.manager.models.ModelProvider;
import ai.enr.manager.models.ModelProviderConfig;
import ai.enr.manager.models.ModelProviderFactory;

/**
 * Service-loader factory for the Ollama model provider. Registered through
 * {@code META-INF/services/ai.enr.manager.models.ModelProviderFactory};
 * enabled by default and configurable via enr.yaml {@code model_providers}.
 */
public final class OllamaModelProviderFactory implements ModelProviderFactory {

    @Override
    public String id() {
        return OllamaModelProvider.PROVIDER_TYPE;
    }

    @Override
    public ModelProvider create(ModelProviderConfig config) {
        return new OllamaModelProvider(
                config.providerId(),
                config.endpoint(),
                config.connectTimeoutMs(),
                config.readTimeoutMs(),
                config.downloadTimeoutMs());
    }
}
