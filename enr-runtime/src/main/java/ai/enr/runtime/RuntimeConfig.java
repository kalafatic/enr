package ai.enr.runtime;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * ENR runtime configuration (defaults shown). Flat, documented subset of YAML.
 */
public record RuntimeConfig(
        String host,
        int port,
        List<Path> modelDirectories,
        String preferredBackend,
        boolean allowCudaFallback,
        long cpuMemoryLimitBytes,
        long cudaMemoryLimitBytes,
        int maxConcurrentSessions,
        int defaultMaxTokens,
        int sessionIdleTimeoutSeconds,
        int requestTimeoutSeconds,
        boolean enableMetrics) {

    public static Builder builder() {
        return new Builder();
    }

    public static RuntimeConfig defaults() {
        return new Builder().build();
    }

    public static final class Builder {
        private String host = "127.0.0.1";
        private int port = 28780;
        private final List<Path> modelDirectories = new ArrayList<>();
        private String preferredBackend = "cpu";
        private boolean allowCudaFallback = true;
        private long cpuMemoryLimitBytes = 0;
        private long cudaMemoryLimitBytes = 0;
        private int maxConcurrentSessions = 8;
        private int defaultMaxTokens = 256;
        private int sessionIdleTimeoutSeconds = 1800;
        private int requestTimeoutSeconds = 300;
        private boolean enableMetrics = true;

        public Builder host(String host) {
            this.host = host;
            return this;
        }

        public Builder port(int port) {
            this.port = port;
            return this;
        }

        public Builder addModelDirectory(Path dir) {
            this.modelDirectories.add(dir);
            return this;
        }

        public Builder modelDirectories(List<Path> dirs) {
            this.modelDirectories.clear();
            this.modelDirectories.addAll(dirs);
            return this;
        }

        public Builder preferredBackend(String backend) {
            this.preferredBackend = backend;
            return this;
        }

        public Builder allowCudaFallback(boolean allow) {
            this.allowCudaFallback = allow;
            return this;
        }

        public Builder cpuMemoryLimitBytes(long bytes) {
            this.cpuMemoryLimitBytes = bytes;
            return this;
        }

        public Builder cudaMemoryLimitBytes(long bytes) {
            this.cudaMemoryLimitBytes = bytes;
            return this;
        }

        public Builder maxConcurrentSessions(int max) {
            this.maxConcurrentSessions = max;
            return this;
        }

        public Builder defaultMaxTokens(int maxTokens) {
            this.defaultMaxTokens = maxTokens;
            return this;
        }

        public Builder sessionIdleTimeoutSeconds(int seconds) {
            this.sessionIdleTimeoutSeconds = seconds;
            return this;
        }

        public Builder requestTimeoutSeconds(int seconds) {
            this.requestTimeoutSeconds = seconds;
            return this;
        }

        public Builder enableMetrics(boolean enable) {
            this.enableMetrics = enable;
            return this;
        }

        public RuntimeConfig build() {
            return new RuntimeConfig(host, port, List.copyOf(modelDirectories), preferredBackend,
                    allowCudaFallback, cpuMemoryLimitBytes, cudaMemoryLimitBytes,
                    maxConcurrentSessions, defaultMaxTokens, sessionIdleTimeoutSeconds,
                    requestTimeoutSeconds, enableMetrics);
        }
    }
}
