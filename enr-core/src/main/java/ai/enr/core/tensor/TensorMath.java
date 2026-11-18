package ai.enr.core.tensor;

import java.util.Arrays;

public final class TensorMath {

    private TensorMath() {
    }

    public static FloatTensor matmul(FloatTensor a, FloatTensor b) {
        if (a.rank() != 2 || b.rank() != 2) {
            throw new IllegalArgumentException("Matmul requires rank-2 operands");
        }
        int m = (int) a.shape()[0];
        int k = (int) a.shape()[1];
        int k2 = (int) b.shape()[0];
        int n = (int) b.shape()[1];
        if (k != k2) {
            throw new IllegalArgumentException("Incompatible matmul shapes: " + Arrays.toString(a.shape()) + " x " + Arrays.toString(b.shape()));
        }
        float[] ad = a.data();
        float[] bd = b.data();
        float[] out = new float[m * n];
        for (int i = 0; i < m; i++) {
            int aRow = i * k;
            int outRow = i * n;
            for (int j = 0; j < k; j++) {
                float av = ad[aRow + j];
                if (av == 0.0f) {
                    continue;
                }
                int bRow = j * n;
                for (int c = 0; c < n; c++) {
                    out[outRow + c] += av * bd[bRow + c];
                }
            }
        }
        return new FloatTensor(new long[]{m, n}, out);
    }

    public static FloatTensor matmulVec(FloatTensor a, FloatTensor b) {
        if (a.rank() != 1 || b.rank() != 2) {
            throw new IllegalArgumentException("matmulVec requires (vec, matrix)");
        }
        int k = (int) a.shape()[0];
        int k2 = (int) b.shape()[0];
        int n = (int) b.shape()[1];
        if (k != k2) {
            throw new IllegalArgumentException("Incompatible matmulVec shapes");
        }
        float[] ad = a.data();
        float[] bd = b.data();
        float[] out = new float[n];
        for (int j = 0; j < k; j++) {
            float av = ad[j];
            if (av == 0.0f) {
                continue;
            }
            int bRow = j * n;
            for (int c = 0; c < n; c++) {
                out[c] += av * bd[bRow + c];
            }
        }
        return new FloatTensor(new long[]{n}, out);
    }

    public static FloatTensor add(FloatTensor a, FloatTensor b) {
        checkSameShape(a, b, "add");
        float[] ad = a.data();
        float[] bd = b.data();
        float[] out = new float[ad.length];
        for (int i = 0; i < ad.length; i++) {
            out[i] = ad[i] + bd[i];
        }
        return new FloatTensor(a.shape(), out);
    }

    public static FloatTensor mul(FloatTensor a, FloatTensor b) {
        checkSameShape(a, b, "mul");
        float[] ad = a.data();
        float[] bd = b.data();
        float[] out = new float[ad.length];
        for (int i = 0; i < ad.length; i++) {
            out[i] = ad[i] * bd[i];
        }
        return new FloatTensor(a.shape(), out);
    }

    public static FloatTensor transpose(FloatTensor a) {
        if (a.rank() != 2) {
            throw new IllegalArgumentException("Transpose requires rank-2 tensor");
        }
        int m = (int) a.shape()[0];
        int n = (int) a.shape()[1];
        float[] ad = a.data();
        float[] out = new float[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                out[j * m + i] = ad[i * n + j];
            }
        }
        return new FloatTensor(new long[]{n, m}, out);
    }

    public static FloatTensor softmaxRows(FloatTensor a) {
        if (a.rank() != 2) {
            throw new IllegalArgumentException("softmaxRows requires rank-2 tensor");
        }
        int rows = (int) a.shape()[0];
        int cols = (int) a.shape()[1];
        float[] ad = a.data();
        float[] out = new float[ad.length];
        for (int r = 0; r < rows; r++) {
            int off = r * cols;
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < cols; c++) {
                float v = ad[off + c];
                if (v > max) {
                    max = v;
                }
            }
            float sum = 0f;
            for (int c = 0; c < cols; c++) {
                float e = (float) Math.exp(ad[off + c] - max);
                out[off + c] = e;
                sum += e;
            }
            if (sum > 0f) {
                for (int c = 0; c < cols; c++) {
                    out[off + c] /= sum;
                }
            }
        }
        return new FloatTensor(a.shape(), out);
    }

    public static FloatTensor rmsNorm(FloatTensor x, float[] weight, float eps) {
        if (x.rank() != 2) {
            throw new IllegalArgumentException("rmsNorm requires rank-2 tensor");
        }
        int rows = (int) x.shape()[0];
        int dim = (int) x.shape()[1];
        if (weight.length != dim) {
            throw new IllegalArgumentException("rmsNorm weight length mismatch");
        }
        float[] xd = x.data();
        float[] out = new float[xd.length];
        for (int r = 0; r < rows; r++) {
            int off = r * dim;
            float sumSq = 0f;
            for (int i = 0; i < dim; i++) {
                float v = xd[off + i];
                sumSq += v * v;
            }
            float rms = (float) Math.sqrt(sumSq / dim + eps);
            for (int i = 0; i < dim; i++) {
                out[off + i] = (xd[off + i] / rms) * weight[i];
            }
        }
        return new FloatTensor(x.shape(), out);
    }

    public static void rmsNormInPlace(FloatTensor x, float[] weight, float eps, float[] out) {
        int rows = (int) x.shape()[0];
        int dim = (int) x.shape()[1];
        float[] xd = x.data();
        for (int r = 0; r < rows; r++) {
            int off = r * dim;
            float sumSq = 0f;
            for (int i = 0; i < dim; i++) {
                float v = xd[off + i];
                sumSq += v * v;
            }
            float rms = (float) Math.sqrt(sumSq / dim + eps);
            for (int i = 0; i < dim; i++) {
                out[off + i] = (xd[off + i] / rms) * weight[i];
            }
        }
    }

    public static FloatTensor silu(FloatTensor x) {
        float[] xd = x.data();
        float[] out = new float[xd.length];
        for (int i = 0; i < xd.length; i++) {
            float z = xd[i];
            float sig = 1.0f / (1.0f + (float) Math.exp(-z));
            out[i] = z * sig;
        }
        return new FloatTensor(x.shape(), out);
    }

    public static void sinusoidalPositionalEncoding(float[] out, int seqLen, int dModel, int offset) {
        for (int pos = 0; pos < seqLen; pos++) {
            int absPos = pos + offset;
            for (int i = 0; i < dModel; i += 2) {
                double divTerm = Math.pow(10000.0, (double) i / dModel);
                out[pos * dModel + i] = (float) Math.sin(absPos / divTerm);
                if (i + 1 < dModel) {
                    out[pos * dModel + i + 1] = (float) Math.cos(absPos / divTerm);
                }
            }
        }
    }

    public static void addInPlace(float[] target, float[] addend) {
        if (target.length != addend.length) {
            throw new IllegalArgumentException("Length mismatch in addInPlace");
        }
        for (int i = 0; i < target.length; i++) {
            target[i] += addend[i];
        }
    }

    public static void scaleInPlace(float[] target, float scalar) {
        for (int i = 0; i < target.length; i++) {
            target[i] *= scalar;
        }
    }

    public static float dot(float[] a, int aOff, float[] b, int bOff, int len) {
        float sum = 0f;
        for (int i = 0; i < len; i++) {
            sum += a[aOff + i] * b[bOff + i];
        }
        return sum;
    }

    private static void checkSameShape(FloatTensor a, FloatTensor b, String op) {
        if (!Arrays.equals(a.shape(), b.shape())) {
            throw new IllegalArgumentException(op + " requires equal shapes: " + Arrays.toString(a.shape()) + " vs " + Arrays.toString(b.shape()));
        }
    }
}
