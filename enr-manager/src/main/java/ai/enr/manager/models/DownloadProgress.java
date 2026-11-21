package ai.enr.manager.models;

import java.util.List;

/**
 * Provider-reported progress for one pull operation. Providers translate their
 * native stream into these updates; job identity and persistence are owned by
 * the service, not by the provider.
 */
public record DownloadProgress(
        DownloadState state,
        long totalBytes,
        long completedBytes,
        String error,
        List<DownloadLayer> layers) {

    public DownloadProgress {
        error = error == null ? "" : error;
        layers = layers == null ? List.of() : List.copyOf(layers);
    }

    public static DownloadProgress of(DownloadState state, long totalBytes, long completedBytes,
            List<DownloadLayer> layers) {
        return new DownloadProgress(state, totalBytes, completedBytes, "", layers);
    }

    public static DownloadProgress failed(String error) {
        return new DownloadProgress(DownloadState.FAILED, -1, 0, error, List.of());
    }

    public static DownloadProgress cancelled(long totalBytes, long completedBytes,
            List<DownloadLayer> layers) {
        return new DownloadProgress(DownloadState.CANCELLED, totalBytes, completedBytes,
                "cancelled by user", layers);
    }
}
