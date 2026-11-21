package ai.enr.manager.models;

/**
 * Capabilities a model provider may expose. A provider must declare exactly what
 * it supports; the service never assumes that every provider can download,
 * remove or enumerate models.
 */
public enum ProviderCapability {
    MODEL_PULL,
    MODEL_LIST,
    MODEL_REMOVE,
    DOWNLOAD_PROGRESS,
    MODEL_METADATA,
    MODEL_DISCOVERY
}
