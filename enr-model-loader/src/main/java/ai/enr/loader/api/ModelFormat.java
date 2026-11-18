package ai.enr.loader.api;

public enum ModelFormat {
    EVO_V2("EVO_NAT2"),
    EVO_V1("EVO_ARTIFACT_V1"),
    GGUF_V3("GGUF v3"),
    GGUF_V2("GGUF v2");

    private final String label;

    ModelFormat(String label) {
        this.label = label;
    }

    public String label() {
        return label;
    }
}
