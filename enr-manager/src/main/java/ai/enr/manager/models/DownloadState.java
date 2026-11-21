package ai.enr.manager.models;

/**
 * Lifecycle of a model download job. Terminal states are {@link #COMPLETED},
 * {@link #FAILED} and {@link #CANCELLED}; all other states are transient.
 */
public enum DownloadState {
    QUEUED,
    CONNECTING,
    DOWNLOADING,
    VERIFYING,
    COMPLETED,
    FAILED,
    CANCEL_REQUESTED,
    CANCELLED;

    public boolean terminal() {
        return this == COMPLETED || this == FAILED || this == CANCELLED;
    }
}
