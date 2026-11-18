package ai.enr.core.tensor;

public enum DType {
    F32(0, 4),
    F16(1, 2),
    BF16(2, 2),
    INT8(3, 1),
    INT4(4, 1);

    private final int code;
    private final int elementBytes;

    DType(int code, int elementBytes) {
        this.code = code;
        this.elementBytes = elementBytes;
    }

    public int code() {
        return code;
    }

    public int elementBytes() {
        return elementBytes;
    }

    public static DType fromCode(int code) {
        for (DType t : values()) {
            if (t.code == code) {
                return t;
            }
        }
        throw new IllegalArgumentException("Unsupported dtype code: " + code);
    }
}
