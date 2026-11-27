package ai.enr.core.tensor;

import ai.enr.core.gguf.GgufQuant;

/**
 * A weight tensor stored in its original GGUF quantized form, without expanding
 * to F32. Holds the raw block-quantized (or F16) bytes plus enough metadata for
 * block-wise GEMV computation.
 *
 * <p>Layout: the weight is laid out {@code [rows][cols]} (row-major), where
 * {@code rows} is the number of output features and {@code cols} the number of
 * input features. Each output row is {@code cols} contiguous elements, further
 * sub-divided into quantization blocks of {@link GgufQuant#blockSize(int)}.</p>
 */
public final class QuantizedWeight {

    private final int ggmlType;
    private final int rows;
    private final int cols;
    private final byte[] data;
    private final long elementCount;

    public QuantizedWeight(int ggmlType, int rows, int cols, byte[] data) {
        if (rows < 0 || cols < 0) {
            throw new IllegalArgumentException("Negative quantized weight dimension");
        }
        this.ggmlType = ggmlType;
        this.rows = rows;
        this.cols = cols;
        this.data = data;
        this.elementCount = (long) rows * cols;
        long expected = GgufQuant.rowSizeBytes(ggmlType, elementCount);
        if (expected < 0) {
            throw new IllegalArgumentException("Unsupported GGML quantized type " + ggmlType);
        }
        if (data != null && data.length != expected) {
            throw new IllegalArgumentException("Quantized data length " + data.length
                    + " does not match expected " + expected + " for type " + ggmlType);
        }
    }

    public int ggmlType() {
        return ggmlType;
    }

    /** Number of output features. */
    public int rows() {
        return rows;
    }

    /** Number of input features. */
    public int cols() {
        return cols;
    }

    public byte[] data() {
        return data;
    }

    public long elementCount() {
        return elementCount;
    }

    /** Number of quantization blocks per output row. */
    public int blocksPerRow() {
        int blockSize = GgufQuant.blockSize(ggmlType);
        return cols / blockSize;
    }

    /** Total number of quantization blocks. */
    public int blockCount() {
        int blockSize = GgufQuant.blockSize(ggmlType);
        return (int) (elementCount / blockSize);
    }

    /** Raw byte size of the stored quantized data. */
    public int rawBytes() {
        return data.length;
    }

    /** Dequantizes this weight into a full F32 array (reference path). */
    public float[] dequantize() {
        return GgufQuant.dequantize(ggmlType, data, elementCount);
    }

    /**
     * Dequantizes a single output row ({@code cols} contiguous elements) into
     * {@code out}, for embedding lookups and other row-wise access. Avoids
     * materializing the full weight.
     */
    public void getRow(int row, float[] out) {
        if (row < 0 || row >= rows) {
            throw new IllegalArgumentException("Row " + row + " out of range [0," + rows + ")");
        }
        if (out.length != cols) {
            throw new IllegalArgumentException("Output length " + out.length + " != " + cols);
        }
        int blockSize = GgufQuant.blockSize(ggmlType);
        int blockBytes = GgufQuant.blockBytes(ggmlType);
        int blocksPerRow = cols / blockSize;
        int rawOff = row * blocksPerRow * blockBytes;
        for (int b = 0; b < blocksPerRow; b++) {
            GgufQuant.dequantizeBlock(out, b * blockSize, data, rawOff + b * blockBytes, ggmlType);
        }
    }

    @Override
    public String toString() {
        return "QuantizedWeight[type=" + ggmlType + ", shape=" + rows + "x" + cols
                + ", bytes=" + data.length + "]";
    }
}
