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
        String huggingFaceToken,
        Path ollamaModelsDir,
        long downloadTimeoutMs,
        JsonObject modelProviders,
        boolean webUiEnabled,
        boolean webUiAutoStart,
        int webPort) {

    /** Authoritative ENR API port (headless runtime + all API surfaces). */
    public static final int DEFAULT_API_PORT = 28780;
    /** Authoritative ENR Web Control Center port (independent listener). */
    public static final int DEFAULT_WEB_PORT = 28880;

    public static Builder builder() {
        return new Builder();
    }

    public static EnrConfig defaults() {
        return new Builder().build();
    }

    public static final class Builder {
        private String host = "127.0.0.1";
        private int port = DEFAULT_API_PORT;
        private int webPort = DEFAULT_WEB_PORT;
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
        private String huggingFaceToken = "";
        private Path ollamaModelsDir = defaultOllamaModelsDir();
        private long downloadTimeoutMs = 0;
        private JsonObject modelProviders = new JsonObject();
        private boolean webUiEnabled = true;
        private boolean webUiAutoStart = true;

        public Builder host(String host) {
            this.host = host;
            return this;
        }

        public Builder port(int port) {
            this.port = port;
            return this;
        }

        public Builder webPort(int webPort) {
            this.webPort = webPort;
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

        public Builder huggingFaceToken(String token) {
            this.huggingFaceToken = token == null ? "" : token;
            return this;
        }

        public Builder ollamaModelsDir(Path dir) {
            this.ollamaModelsDir = dir == null ? defaultOllamaModelsDir() : dir;
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

        public Builder webUiEnabled(boolean enabled) {
            this.webUiEnabled = enabled;
            return this;
        }

        public Builder webUiAutoStart(boolean autoStart) {
            this.webUiAutoStart = autoStart;
            return this;
        }

        public EnrConfig build() {
            return new EnrConfig(host, port, dataDir, modelsDir, logDir, modelDirectories,
                    preferredBackend, allowCudaFallback, maxConcurrentSessions, defaultMaxTokens,
                    sessionIdleTimeoutSeconds, requestTimeoutSeconds, enableMetrics,
                    authToken, ollamaRegistryUrl, ollamaSearchUrl, huggingFaceApiUrl,
                    huggingFaceToken, ollamaModelsDir, downloadTimeoutMs, modelProviders,
                    webUiEnabled, webUiAutoStart, webPort);
        }

        /** The default Ollama model storage directory (respects {@code OLLAMA_MODELS}). */
        private static Path defaultOllamaModelsDir() {
            String env = System.getenv("OLLAMA_MODELS");
            if (env != null && !env.isBlank()) {
                return Path.of(env.trim());
            }
            return Path.of(System.getProperty("user.home"), ".ollama", "models");
        }
    }
}
