package ai.enr.manager.models;

/**
 * Service-loader entry point for model providers. Implementations must have a
 * public no-argument constructor and be listed in
 * {@code META-INF/services/ai.enr.manager.models.ModelProviderFactory}.
 */
public interface ModelProviderFactory {

    String id();

    ModelProvider create(ModelProviderConfig config);
}
