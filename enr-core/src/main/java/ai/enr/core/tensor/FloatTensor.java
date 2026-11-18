package ai.enr.core.tensor;

import java.util.Arrays;

public final class FloatTensor {

    private final long[] shape;
    private final float[] data;
    private final int rank;
    private final long size;

    public FloatTensor(long[] shape) {
        this.shape = shape.clone();
        this.rank = shape.length;
        long total = 1;
        for (long dim : shape) {
            if (dim < 0) {
                throw new IllegalArgumentException("Negative dimension in shape: " + Arrays.toString(shape));
            }
            total *= dim;
        }
        this.size = total;
        if (total > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("Tensor too large for float[] storage: " + total + " elements");
        }
        this.data = new float[(int) total];
    }

    public FloatTensor(long[] shape, float[] data) {
        this.shape = shape.clone();
        this.rank = shape.length;
        long total = 1;
        for (long dim : shape) {
            total *= dim;
        }
        this.size = total;
        if (data.length != total) {
            throw new IllegalArgumentException("Data length " + data.length + " does not match shape " + Arrays.toString(shape));
        }
        this.data = data;
    }

    public static FloatTensor of(float[] data, long... shape) {
        return new FloatTensor(shape, data);
    }

    public long[] shape() {
        return shape.clone();
    }

    public int rank() {
        return rank;
    }

    public long size() {
        return size;
    }

    public float[] data() {
        return data;
    }

    public float get(long... indices) {
        int idx = 0;
        int mult = 1;
        for (int i = rank - 1; i >= 0; i--) {
            idx += (int) (indices[i] * mult);
            mult *= (int) shape[i];
        }
        return data[idx];
    }

    public void set(float value, long... indices) {
        int idx = 0;
        int mult = 1;
        for (int i = rank - 1; i >= 0; i--) {
            idx += (int) (indices[i] * mult);
            mult *= (int) shape[i];
        }
        data[idx] = value;
    }

    public FloatTensor copy() {
        return new FloatTensor(shape, data.clone());
    }

    public FloatTensor reshape(long... newShape) {
        long total = 1;
        for (long d : newShape) {
            total *= d;
        }
        if (total != size) {
            throw new IllegalArgumentException("Reshape element count mismatch");
        }
        return new FloatTensor(newShape, data.clone());
    }

    @Override
    public String toString() {
        return "FloatTensor" + Arrays.toString(shape);
    }
}
