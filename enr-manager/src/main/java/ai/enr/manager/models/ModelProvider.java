package ai.enr.manager.models;

import java.util.List;
import java.util.Set;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

/**
 * SPI for a model source (Ollama, LM Studio, a local store, ...). A provider is
 * responsible only for its native protocol and metadata translation; download
 * lifecycle, duplicate prevention, persistence and listeners belong to
 * {@link ModelManagementService}.
 */
public interface ModelProvider {

    String id();

    String displayName();

    Set<ProviderCapability> capabilities();

    /**
     * Models currently installed in this provider's store.
     */
    List<ModelDescriptor> listInstalled() throws Exception;

    /**
     * Models known to be available for download. Providers without discovery
     * support return an empty list.
     */
    List<ModelDescriptor> listAvailable() throws Exception;

    /**
     * Optional metadata for one model.
     */
    default ModelDescriptor describe(String modelId) throws Exception {
        for (ModelDescriptor descriptor : listInstalled()) {
            if (descriptor.id().equals(modelId)) {
                return descriptor;
            }
        }
        throw new UnsupportedOperationException("Provider " + id() + " cannot describe models");
    }

    /**
     * Downloads a model, blocking until it is complete, cancelled or failed.
     * Progress must be reported through the supplied consumer. The provider
     * must poll {@code cancellationRequested} regularly and abort its native
     * transfer when it returns true.
     */
    void pull(String modelId, BooleanSupplier cancellationRequested,
            Consumer<DownloadProgress> progress) throws Exception;

    /**
     * Best-effort request to abort an in-flight pull. Providers that abort
     * through {@code cancellationRequested} may leave this empty.
     */
    default void cancel(String modelId) {
    }

    /**
     * Best-effort connectivity check. Providers that have no connectivity
     * concept report {@code true}; remote providers should probe their
     * endpoint and report {@code false} when unreachable.
     */
    default boolean ping() {
        return true;
    }

    default void remove(String modelId) throws Exception {
        throw new UnsupportedOperationException("Provider " + id() + " cannot remove models");
    }
}
