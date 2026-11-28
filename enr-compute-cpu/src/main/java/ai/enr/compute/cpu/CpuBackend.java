package ai.enr.compute.cpu;

import ai.enr.compute.ComputeBackend;
import ai.enr.compute.DeviceInfo;
import ai.enr.core.gguf.GgufQuant;
import ai.enr.core.tensor.FloatTensor;
import ai.enr.core.tensor.QuantizedWeight;
import ai.enr.core.tensor.TensorMath;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;

/**
 * CPU compute backend. Provides a deterministic single-threaded reference
 * implementation plus a parallel quantized GEMV for large weights.
 *
 * <p>Parallelism: independent output rows (or row blocks) are distributed across
 * a bounded {@link ForkJoinPool} of daemon worker threads. Rows are statically
 * partitioned, so each output element is accumulated by exactly one worker in a
 * fixed order — the parallel result is bit-for-bit identical to the
 * single-threaded reference. No temporary buffer is shared between workers.</p>
 */
public final class CpuBackend implements ComputeBackend {

    private static final int PARALLEL_WORK_THRESHOLD = 4096;

    /**
     * True when the jdk.incubator.vector module is available (SIMD enabled).
     * Detected via reflection so that {@link SimdOps} (which hard-references the
     * Vector API) is never class-loaded on runtimes without the module.
     */
    private static final boolean SIMD = detectSimd();

    private static boolean detectSimd() {
        try {
            Class.forName("jdk.incubator.vector.FloatVector");
            return true;
        } catch (Throwable t) {
            return false;
        }
    }

    private final int workers;
    private volatile ForkJoinPool pool;

    public CpuBackend() {
        this(Runtime.getRuntime().availableProcessors());
    }

    public CpuBackend(int workers) {
        this.workers = Math.max(1, workers);
    }

    /** Configured worker count. */
    public int workers() {
        return workers;
    }

    private ForkJoinPool pool() {
        ForkJoinPool p = pool;
        if (p == null) {
            synchronized (this) {
                if (pool == null) {
                    pool = new ForkJoinPool(workers);
                }
                p = pool;
            }
        }
        return p;
    }

    @Override
    public String name() {
        return "cpu";
    }

    @Override
    public DeviceInfo deviceInfo() {
        return DeviceInfo.cpu();
    }

    @Override
    public FloatTensor matmul(FloatTensor a, FloatTensor b) {
        return TensorMath.matmul(a, b);
    }

    @Override
    public FloatTensor matmulQuantized(FloatTensor x, QuantizedWeight w) {
        int seqLen = (int) x.shape()[0];
        int in = (int) x.shape()[1];
        int outFeatures = w.rows();
        if (in != w.cols()) {
            throw new IllegalArgumentException("Quantized matmul shape mismatch: x="
                    + java.util.Arrays.toString(x.shape()) + " vs weight cols=" + w.cols());
        }
        float[] result = new float[seqLen * outFeatures];
        if (workers <= 1 || (long) seqLen * outFeatures < PARALLEL_WORK_THRESHOLD) {
            matmulQuantizedRows(x, w, result, 0, outFeatures);
        } else {
            matmulQuantizedRowsParallel(x, w, result, outFeatures);
        }
        return new FloatTensor(new long[]{seqLen, outFeatures}, result);
    }

    /**
     * Single-threaded scalar reference implementation (block-major iteration).
     * Preserved for correctness comparison against the SIMD/parallel path.
     */
    FloatTensor matmulQuantizedReference(FloatTensor x, QuantizedWeight w) {
        int seqLen = (int) x.shape()[0];
        int in = (int) x.shape()[1];
        int outFeatures = w.rows();
        int ggmlType = w.ggmlType();
        int blockSize = GgufQuant.blockSize(ggmlType);
        int blockBytes = GgufQuant.blockBytes(ggmlType);
        int blocksPerRow = w.blocksPerRow();
        float[] xd = x.data();
        byte[] wd = w.data();
        float[] result = new float[seqLen * outFeatures];
        float[] tmp = new float[blockSize];
        int rawOff = 0;
        for (int block = 0; block < w.blockCount(); block++, rawOff += blockBytes) {
            int row = block / blocksPerRow;
            int colStart = (block % blocksPerRow) * blockSize;
            GgufQuant.dequantizeBlock(tmp, 0, wd, rawOff, ggmlType);
            for (int s = 0; s < seqLen; s++) {
                float acc = 0f;
                int xOff = s * in + colStart;
                for (int k = 0; k < blockSize; k++) {
                    acc += xd[xOff + k] * tmp[k];
                }
                result[s * outFeatures + row] += acc;
            }
        }
        return new FloatTensor(new long[]{seqLen, outFeatures}, result);
    }

    private void matmulQuantizedRowsParallel(FloatTensor x, QuantizedWeight w, float[] result,
                                             int outFeatures) {
        int chunk = Math.max(1, (outFeatures + workers - 1) / workers);
        int numChunks = (outFeatures + chunk - 1) / chunk;
        CountDownLatch latch = new CountDownLatch(numChunks);
        AtomicReference<Throwable> error = new AtomicReference<>();
        for (int start = 0; start < outFeatures; start += chunk) {
            int rowStart = start;
            int rowEnd = Math.min(outFeatures, start + chunk);
            pool().execute(() -> {
                try {
                    matmulQuantizedRows(x, w, result, rowStart, rowEnd);
                } catch (Throwable t) {
                    error.compareAndSet(null, t);
                } finally {
                    latch.countDown();
                }
            });
        }
        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted during parallel quantized matmul", e);
        }
        Throwable t = error.get();
        if (t != null) {
            throw new IllegalStateException("Parallel quantized matmul failed", t);
        }
    }

    /** Computes rows {@code [rowStart, rowEnd)} into {@code result} (row-major, SIMD-accelerated). */
    private static void matmulQuantizedRows(FloatTensor x, QuantizedWeight w, float[] result,
                                            int rowStart, int rowEnd) {
        int seqLen = (int) x.shape()[0];
        int in = w.cols();
        int out = w.rows();
        int ggmlType = w.ggmlType();
        int blockSize = GgufQuant.blockSize(ggmlType);
        int blockBytes = GgufQuant.blockBytes(ggmlType);
        int blocksPerRow = w.blocksPerRow();
        float[] xd = x.data();
        byte[] wd = w.data();
        boolean fusedQ8_0 = SIMD && ggmlType == GgufQuant.GGML_TYPE_Q8_0;
        boolean fusedQ4_K = SIMD && ggmlType == GgufQuant.GGML_TYPE_Q4_K;
        float[] tmp = (fusedQ8_0 || fusedQ4_K) ? null : new float[blockSize];
        for (int row = rowStart; row < rowEnd; row++) {
            int rawOff = row * blocksPerRow * blockBytes;
            for (int b = 0; b < blocksPerRow; b++) {
                int blockRawOff = rawOff + b * blockBytes;
                int colStart = b * blockSize;
                if (fusedQ8_0) {
                    for (int s = 0; s < seqLen; s++) {
                        result[s * out + row] += SimdOps.dotQ8_0(xd, s * in + colStart, wd, blockRawOff);
                    }
                } else if (fusedQ4_K) {
                    for (int s = 0; s < seqLen; s++) {
                        result[s * out + row] += SimdOps.dotQ4_K(xd, s * in + colStart, wd, blockRawOff);
                    }
                } else {
                    GgufQuant.dequantizeBlock(tmp, 0, wd, blockRawOff, ggmlType);
                    if (SIMD) {
                        for (int s = 0; s < seqLen; s++) {
                            result[s * out + row] += SimdOps.dot(xd, s * in + colStart, tmp, 0, blockSize);
                        }
                    } else {
                        for (int s = 0; s < seqLen; s++) {
                            float acc = 0f;
                            int xOff = s * in + colStart;
                            for (int k = 0; k < blockSize; k++) {
                                acc += xd[xOff + k] * tmp[k];
                            }
                            result[s * out + row] += acc;
                        }
                    }
                }
            }
        }
    }

    @Override
    public FloatTensor rmsNorm(FloatTensor x, float[] weight, float eps) {
        return TensorMath.rmsNorm(x, weight, eps);
    }

    @Override
    public FloatTensor softmaxRows(FloatTensor x) {
        return TensorMath.softmaxRows(x);
    }

    @Override
    public FloatTensor silu(FloatTensor x) {
        return TensorMath.silu(x);
    }

    @Override
    public FloatTensor siluMul(FloatTensor gate, FloatTensor up) {
        return TensorMath.mul(TensorMath.silu(gate), up);
    }

    @Override
    public FloatTensor add(FloatTensor a, FloatTensor b) {
        return TensorMath.add(a, b);
    }

    @Override
    public FloatTensor mul(FloatTensor a, FloatTensor b) {
        return TensorMath.mul(a, b);
    }

    @Override
    public FloatTensor attention(FloatTensor q, FloatTensor k, FloatTensor v, float scale, int startPos) {
        int heads = (int) q.shape()[0];
        int seqLen = (int) q.shape()[1];
        int kvSeqLen = (int) k.shape()[1];
        int headDim = (int) q.shape()[2];

        float[] qd = q.data();
        float[] kd = k.data();
        float[] vd = v.data();
        float[] out = new float[heads * seqLen * headDim];

        for (int h = 0; h < heads; h++) {
            int qOff = h * seqLen * headDim;
            int kvOff = h * kvSeqLen * headDim;
            for (int r = 0; r < seqLen; r++) {
                int absolutePos = startPos + r;
                int attendUpTo = Math.min(absolutePos, kvSeqLen - 1);
                float maxVal = Float.NEGATIVE_INFINITY;
                float[] scores = new float[kvSeqLen];
                for (int c = 0; c <= attendUpTo; c++) {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) {
                        dot += qd[qOff + r * headDim + d] * kd[kvOff + c * headDim + d];
                    }
                    float score = dot * scale;
                    scores[c] = score;
                    if (score > maxVal) {
                        maxVal = score;
                    }
                }
                float sum = 0f;
                for (int c = 0; c <= attendUpTo; c++) {
                    float e = (float) Math.exp(scores[c] - maxVal);
                    scores[c] = e;
                    sum += e;
                }
                if (sum > 0f) {
                    for (int c = 0; c <= attendUpTo; c++) {
                        scores[c] /= sum;
                    }
                }
                int outOff = qOff + r * headDim;
                for (int c = 0; c <= attendUpTo; c++) {
                    float p = scores[c];
                    if (p == 0f) {
                        continue;
                    }
                    for (int d = 0; d < headDim; d++) {
                        out[outOff + d] += p * vd[kvOff + c * headDim + d];
                    }
                }
            }
        }
        return new FloatTensor(new long[]{heads, seqLen, headDim}, out);
    }

    @Override
    public void close() {
        ForkJoinPool p = pool;
        if (p != null) {
            pool = null;
            p.shutdown();
        }
    }
}
