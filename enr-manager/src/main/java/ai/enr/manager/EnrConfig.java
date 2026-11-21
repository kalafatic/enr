package ai.enr.manager;

import ai.enr.core.json.JsonObject;

import java.nio.file.Path;
import java.util.List;

/**
 * Persistent ENR configuration: server binding, storage locations, model
 * directories and administration. Loaded from the YAML subset (enr.yaml);
 * defaults match a production system install.
 */
public record EnrConfig(
        String host,
        int port,
        Path dataDir,
        Path modelsDir,
        Path logDir,
        List<Path> modelDirectories,
        String preferredBackend,
        boolean allowCudaFallback,
        int maxConcurrentSessions,
        int defaultMaxTokens,
        int sessionIdleTimeoutSeconds,
        int requestTimeoutSeconds,
        boolean enableMetrics,
        String authToken,
        String ollamaRegistryUrl,
        String ollamaSearchUrl,
        String huggingFaceApiUrl,
        long downloadTimeoutMs,
        JsonObject modelProviders) {

    public static Builder builder() {
        return new Builder();
    }

    public static EnrConfig defaults() {
        return new Builder().build();
    }

    public static final class Builder {
        private String host = "127.0.0.1";
        private int port = 8111;
        private Path dataDir = Path.of(System.getProperty("user.home"), ".enr");
        private Path modelsDir = Path.of(System.getProperty("user.home"), ".enr", "models");
        private Path logDir = Path.of(System.getProperty("user.home"), ".enr", "logs");
        private List<Path> modelDirectories = List.of();
        private String preferredBackend = "cpu";
        private boolean allowCudaFallback = true;
        private int maxConcurrentSessions = 8;
        private int defaultMaxTokens = 256;
        private int sessionIdleTimeoutSeconds = 1800;
        private int requestTimeoutSeconds = 300;
        private boolean enableMetrics = true;
        private String authToken = "";
        private String ollamaRegistryUrl = "https://registry.ollama.ai";
        private String ollamaSearchUrl = "https://ollama.com";
        private String huggingFaceApiUrl = "https://huggingface.co";
        private long downloadTimeoutMs = 0;
        private JsonObject modelProviders = new JsonObject();

        public Builder host(String host) {
            this.host = host;
            return this;
        }

        public Builder port(int port) {
            this.port = port;
            return this;
        }

        public Builder dataDir(Path dataDir) {
            this.dataDir = dataDir;
            return this;
        }

        public Builder modelsDir(Path modelsDir) {
            this.modelsDir = modelsDir;
            return this;
        }

        public Builder logDir(Path logDir) {
            this.logDir = logDir;
            return this;
        }

        public Builder modelDirectories(List<Path> dirs) {
            this.modelDirectories = dirs == null ? List.of() : List.copyOf(dirs);
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

        public Builder authToken(String token) {
            this.authToken = token == null ? "" : token;
            return this;
        }

        public Builder ollamaRegistryUrl(String url) {
            this.ollamaRegistryUrl = url == null ? "https://registry.ollama.ai" : url;
            return this;
        }

        public Builder ollamaSearchUrl(String url) {
            this.ollamaSearchUrl = url == null ? "https://ollama.com" : url;
            return this;
        }

        public Builder huggingFaceApiUrl(String url) {
            this.huggingFaceApiUrl = url == null ? "https://huggingface.co" : url;
            return this;
        }

        public Builder downloadTimeoutMs(long ms) {
            this.downloadTimeoutMs = ms;
            return this;
        }

        public Builder modelProviders(JsonObject modelProviders) {
            this.modelProviders = modelProviders == null ? new JsonObject() : modelProviders;
            return this;
        }

        public EnrConfig build() {
            return new EnrConfig(host, port, dataDir, modelsDir, logDir, modelDirectories,
                    preferredBackend, allowCudaFallback, maxConcurrentSessions, defaultMaxTokens,
                    sessionIdleTimeoutSeconds, requestTimeoutSeconds, enableMetrics,
                    authToken, ollamaRegistryUrl, ollamaSearchUrl, huggingFaceApiUrl,
                    downloadTimeoutMs, modelProviders);
        }
    }
}
