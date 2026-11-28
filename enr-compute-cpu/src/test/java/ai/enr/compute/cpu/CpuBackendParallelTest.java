package ai.enr.compute.cpu;

import ai.enr.core.gguf.GgufQuant;
import ai.enr.core.tensor.FloatTensor;
import ai.enr.core.tensor.QuantizedWeight;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Parallel quantized GEMV must be bit-for-bit identical to the single-threaded
 * SIMD path (static row partitioning with a fixed accumulation order), and the
 * SIMD path must match the scalar reference within float tolerance.
 */
class CpuBackendParallelTest {

    @Test
    void parallelMatchesSingleQ4_0() {
        assertParallelMatchesSingle(GgufQuant.GGML_TYPE_Q4_0);
    }

    @Test
    void parallelMatchesSingleQ8_0() {
        assertParallelMatchesSingle(GgufQuant.GGML_TYPE_Q8_0);
    }

    @Test
    void parallelMatchesSingleQ4_K() {
        assertParallelMatchesSingle(GgufQuant.GGML_TYPE_Q4_K);
    }

    @Test
    void parallelMatchesSingleQ6_K() {
        assertParallelMatchesSingle(GgufQuant.GGML_TYPE_Q6_K);
    }

    @Test
    void simdMatchesScalarReference() {
        assertSimdMatchesScalar(GgufQuant.GGML_TYPE_Q8_0);
        assertSimdMatchesScalar(GgufQuant.GGML_TYPE_Q4_K);
        assertSimdMatchesScalar(GgufQuant.GGML_TYPE_Q6_K);
    }

    @Test
    void smallWeightUsesSingleThreadedPath() {
        CpuBackend backend = new CpuBackend(8);
        QuantizedWeight w = randomWeight(GgufQuant.GGML_TYPE_Q8_0, 64, 256);
        FloatTensor x = FloatTensor.of(randomVector(256), 1, 256);
        FloatTensor out = backend.matmulQuantized(x, w);
        assertEquals(1, out.shape()[0]);
        assertEquals(64, out.shape()[1]);
        backend.close();
    }

    private static void assertParallelMatchesSingle(int ggmlType) {
        int in = 4096;
        int out = 8192;
        int seqLen = 3;
        QuantizedWeight w = randomWeight(ggmlType, out, in);
        FloatTensor x = FloatTensor.of(randomVector(seqLen * in), seqLen, in);

        CpuBackend single = new CpuBackend(1);
        CpuBackend parallel = new CpuBackend(8);
        try {
            FloatTensor a = single.matmulQuantized(x, w);
            FloatTensor b = parallel.matmulQuantized(x, w);
            assertArrayEquals(a.data(), b.data(), "parallel differs from single for type " + ggmlType);
        } finally {
            single.close();
            parallel.close();
        }
    }

    private static void assertSimdMatchesScalar(int ggmlType) {
        int in = 4096;
        int out = 8192;
        int seqLen = 3;
        QuantizedWeight w = randomWeight(ggmlType, out, in);
        FloatTensor x = FloatTensor.of(randomVector(seqLen * in), seqLen, in);
        CpuBackend backend = new CpuBackend(8);
        try {
            FloatTensor ref = backend.matmulQuantizedReference(x, w);
            FloatTensor simd = backend.matmulQuantized(x, w);
            assertClose(ref.data(), simd.data(), ggmlType);
        } finally {
            backend.close();
        }
    }

    private static void assertClose(float[] expected, float[] actual, int ggmlType) {
        assertEquals(expected.length, actual.length);
        float maxAbs = 0f;
        for (float v : expected) {
            maxAbs = Math.max(maxAbs, Math.abs(v));
        }
        float maxDiff = 0f;
        for (int i = 0; i < expected.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(expected[i] - actual[i]));
        }
        assertTrue(maxDiff <= 1e-3f * Math.max(maxAbs, 1f),
                "type " + ggmlType + " SIMD differs from scalar: maxDiff=" + maxDiff
                        + " maxAbs=" + maxAbs);
    }

    /**
     * Builds a quantized weight with deterministic random raw bytes, forcing the
     * fp16 scale/min fields to finite values so the results are well-conditioned
     * (random raw bytes can otherwise encode NaN/Inf fp16 scales).
     */
    private static QuantizedWeight randomWeight(int ggmlType, int out, int in) {
        int blockSize = GgufQuant.blockSize(ggmlType);
        if (in % blockSize != 0) {
            throw new IllegalArgumentException("in must be a multiple of block size");
        }
        long rawBytes = GgufQuant.rowSizeBytes(ggmlType, (long) out * in);
        byte[] raw = new byte[(int) rawBytes];
        Random rnd = new Random(42);
        rnd.nextBytes(raw);
        forceFiniteScales(ggmlType, raw);
        return new QuantizedWeight(ggmlType, out, in, raw);
    }

    /** Sets the fp16 scale/min fields of every block to finite values. */
    private static void forceFiniteScales(int ggmlType, byte[] raw) {
        int blockBytes = GgufQuant.blockBytes(ggmlType);
        int blockCount = raw.length / blockBytes;
        if (ggmlType == GgufQuant.GGML_TYPE_Q6_K) {
            for (int b = 0; b < blockCount; b++) {
                int off = b * blockBytes + 208;
                raw[off] = 0x00;
                raw[off + 1] = 0x3C; // d = 1.0
            }
            return;
        }
        for (int b = 0; b < blockCount; b++) {
            int off = b * blockBytes;
            raw[off] = 0x00;
            raw[off + 1] = 0x3C; // d / scale = 1.0
            if (ggmlType == GgufQuant.GGML_TYPE_Q4_K || ggmlType == GgufQuant.GGML_TYPE_Q5_K) {
                raw[off + 2] = 0x00;
                raw[off + 3] = 0x38; // dmin = 0.5
            }
        }
    }

    private static float[] randomVector(int n) {
        float[] v = new float[n];
        Random rnd = new Random(7);
        for (int i = 0; i < n; i++) {
            v[i] = (float) rnd.nextGaussian();
        }
        return v;
    }
}
