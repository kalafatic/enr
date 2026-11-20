package ai.enr.manager.provider;

/**
 * Provider kinds supported by the registry. LOCAL kinds talk to a locally
 * running inference server; CLOUD kinds talk to hosted APIs. NATIVE is the
 * built-in ENR runtime itself.
 */
public enum ProviderType {
    NATIVE("native", true),
    OLLAMA("ollama", true),
    LLAMACPP("llama.cpp", true),
    OPENAI_COMPAT("openai-compatible", true),
    DEEPSEEK("deepseek", false),
    OPENAI("openai", false),
    ANTHROPIC("anthropic", false),
    GEMINI("gemini", false);

    private final String id;
    private final boolean local;

    ProviderType(String id, boolean local) {
        this.id = id;
        this.local = local;
    }

    public String id() {
        return id;
    }

    public boolean local() {
        return local;
    }

    public static ProviderType fromId(String id) {
        for (ProviderType type : values()) {
            if (type.id.equalsIgnoreCase(id)) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown provider type: " + id
                + " (expected one of " + ids() + ")");
    }

    private static String ids() {
        StringBuilder sb = new StringBuilder();
        for (ProviderType type : values()) {
            if (sb.length() > 0) {
                sb.append(", ");
            }
            sb.append(type.id);
        }
        return sb.toString();
    }

    public static String defaultBaseUrl(ProviderType type) {
        return switch (type) {
            case DEEPSEEK -> "https://api.deepseek.com/v1";
            case OPENAI -> "https://api.openai.com/v1";
            case ANTHROPIC -> "https://api.anthropic.com";
            case GEMINI -> "https://generativelanguage.googleapis.com";
            case OLLAMA -> "http://127.0.0.1:11434";
            case LLAMACPP -> "http://127.0.0.1:8080";
            case OPENAI_COMPAT, NATIVE -> "";
        };
    }
}
