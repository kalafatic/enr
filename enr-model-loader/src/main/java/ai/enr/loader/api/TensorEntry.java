package ai.enr.loader.api;

import ai.enr.core.tensor.DType;

public final class TensorEntry {

    private final int id;
    private final String name;
    private final DType dtype;
    private final long[] shape;
    private final int layout;
    private final long offset;
    private final long compressedSize;
    private final long uncompressedSize;
    private final int compression;
    private final String checksum;
    private final TensorStore store;

    public TensorEntry(int id, String name, DType dtype, long[] shape, int layout, long offset,
                       long compressedSize, long uncompressedSize, int compression,
                       String checksum, TensorStore store) {
        this.id = id;
        this.name = name;
        this.dtype = dtype;
        this.shape = shape.clone();
        this.layout = layout;
        this.offset = offset;
        this.compressedSize = compressedSize;
        this.uncompressedSize = uncompressedSize;
        this.compression = compression;
        this.checksum = checksum != null ? checksum : "";
        this.store = store;
    }

    public int id() {
        return id;
    }

    public String name() {
        return name;
    }

    public DType dtype() {
        return dtype;
    }

    public long[] shape() {
        return shape.clone();
    }

    public int layout() {
        return layout;
    }

    public long offset() {
        return offset;
    }

    public long compressedSize() {
        return compressedSize;
    }

    public long uncompressedSize() {
        return uncompressedSize;
    }

    public int compression() {
        return compression;
    }

    public String checksum() {
        return checksum;
    }

    public long elementCount() {
        long count = 1;
        for (long dim : shape) {
            count *= dim;
        }
        return count;
    }

    public float[] loadFloats() {
        if (store == null) {
            throw new IllegalStateException("Tensor " + name + " has no backing store");
        }
        return store.loadFloats();
    }

    public boolean isLoaded() {
        return store != null && store.isLoaded();
    }

    @Override
    public String toString() {
        return "TensorEntry[" + name + ", dtype=" + dtype + ", shape=" + java.util.Arrays.toString(shape) + "]";
    }
}
