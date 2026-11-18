package ai.enr.compute;

public enum DeviceType {
    CPU("cpu"),
    CUDA("cuda");

    private final String id;

    DeviceType(String id) {
        this.id = id;
    }

    public String id() {
        return id;
    }
}
