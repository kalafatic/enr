package ai.enr.compute.cpu;

import ai.enr.compute.ComputeBackend;
import ai.enr.core.tensor.FloatTensor;
import ai.enr.core.tensor.TensorMath;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CpuBackendOpsTest {

    private final ComputeBackend backend = new CpuBackend();

    @Test
    void matmulDimensionsAndValues() {
        FloatTensor a = FloatTensor.of(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        FloatTensor b = FloatTensor.of(new float[]{7, 8, 9, 10, 11, 12}, 3, 2);
        FloatTensor c = backend.matmul(a, b);
        assertArrayEquals(new long[]{2, 2}, c.shape());
        assertEquals(58f, c.data()[0], 1e-6f);  // 1*7+2*9+3*11
        assertEquals(64f, c.data()[1], 1e-6f);  // 1*8+2*10+3*12
        assertEquals(139f, c.data()[2], 1e-6f); // 4*7+5*9+6*11
        assertEquals(154f, c.data()[3], 1e-6f); // 4*8+5*10+6*12
    }

    @Test
    void matmulRejectsIncompatibleShapes() {
        FloatTensor a = FloatTensor.of(new float[6], 2, 3);
        FloatTensor b = FloatTensor.of(new float[6], 2, 3);
        assertThrows(IllegalArgumentException.class, () -> backend.matmul(a, b));
    }

    @Test
    void rmsNormReference() {
        FloatTensor x = FloatTensor.of(new float[]{1, 2, 3, 4, -1, -2, -3, -4}, 2, 4);
        float[] w = {1, 1, 1, 1};
        FloatTensor out = backend.rmsNorm(x, w, 1e-5f);
        float rms0 = (float) Math.sqrt((1 + 4 + 9 + 16) / 4.0 + 1e-5);
        float rms1 = rms0;
        assertEquals(1f / rms0, out.data()[0], 1e-6f);
        assertEquals(2f / rms0, out.data()[1], 1e-6f);
        assertEquals(3f / rms0, out.data()[2], 1e-6f);
        assertEquals(4f / rms0, out.data()[3], 1e-6f);
        assertEquals(-1f / rms1, out.data()[4], 1e-6f);
        assertEquals(-4f / rms1, out.data()[7], 1e-6f);
    }

    @Test
    void rmsNormAppliesWeight() {
        FloatTensor x = FloatTensor.of(new float[]{1, 2, 3, 4}, 1, 4);
        float[] w = {2, 0.5f, 1, -1};
        FloatTensor out = backend.rmsNorm(x, w, 1e-5f);
        float rms = (float) Math.sqrt((1 + 4 + 9 + 16) / 4.0 + 1e-5);
        assertEquals(2f / rms, out.data()[0], 1e-6f);
        assertEquals(1f / rms, out.data()[1], 1e-6f);
        assertEquals(3f / rms, out.data()[2], 1e-6f);
        assertEquals(-4f / rms, out.data()[3], 1e-6f);
    }

    @Test
    void softmaxRowsSumToOneAndIsShiftInvariant() {
        FloatTensor x = FloatTensor.of(new float[]{1, 2, 3, 4, 100, 101, 102, 103}, 2, 4);
        FloatTensor out = backend.softmaxRows(x);
        assertEquals(1f, out.data()[0] + out.data()[1] + out.data()[2] + out.data()[3], 1e-5f);
        assertEquals(1f, out.data()[4] + out.data()[5] + out.data()[6] + out.data()[7], 1e-5f);
        for (int r = 0; r < 2; r++) {
            assertEquals(1d, out.data()[r * 4 + 1] / out.data()[r * 4] / Math.exp(1), 1e-4f);
        }
    }

    @Test
    void softmaxIsNumericallyStableForLargeInputs() {
        FloatTensor x = FloatTensor.of(new float[]{1e10f, 2e10f, 3e10f}, 1, 3);
        FloatTensor out = backend.softmaxRows(x);
        assertFalse(Float.isNaN(out.data()[0]));
        assertTrue(out.data()[2] > 0.999f);
    }

    @Test
    void attentionCausalMasking() {
        // single head, 3 query rows, 4 kv rows: query r may attend c <= r
        FloatTensor q = FloatTensor.of(new float[]{
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0
        }, 1, 3, 4);
        FloatTensor k = FloatTensor.of(new float[]{
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                1, 1, 1, 1
        }, 1, 4, 4);
        FloatTensor v = FloatTensor.of(new float[]{
                10, 0, 0, 0,
                20, 0, 0, 0,
                30, 0, 0, 0,
                99, 99, 99, 99
        }, 1, 4, 4);
        FloatTensor out = backend.attention(q, k, v, 1.0f, 0);
        // row 0 attends only position 0 -> 10
        assertEquals(10f, out.data()[0], 1e-4f);
        // row 1 attends 0,1: q=[0,1,0,0] -> dot k1 = 1 -> softmax over 2 entries
        float s0 = (float) Math.exp(0);
        float s1 = (float) Math.exp(1);
        float expected1 = (10 * s0 + 20 * s1) / (s0 + s1);
        assertEquals(expected1, out.data()[4], 1e-4f);
        // row 2 attends 0,1,2 with dots 0,0,1: softmax weights
        float expected2 = (10f + 20f + 30f * (float) Math.exp(1)) / (2f + (float) Math.exp(1));
        assertEquals(expected2, out.data()[8], 1e-4f);
    }

    @Test
    void attentionKVOffsetMasking() {
        // query starts at absolute position 2, attends kv positions 0..2 only
        FloatTensor q = FloatTensor.of(new float[]{0, 0, 1, 0}, 1, 1, 4);
        FloatTensor k = FloatTensor.of(new float[]{
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                1, 1, 1, 1
        }, 1, 4, 4);
        FloatTensor v = FloatTensor.of(new float[]{
                10, 0, 0, 0,
                20, 0, 0, 0,
                30, 0, 0, 0,
                99, 99, 99, 99
        }, 1, 4, 4);
        FloatTensor out = backend.attention(q, k, v, 1.0f, 2);
        float expected = (10f + 20f + 30f * (float) Math.exp(1)) / (2f + (float) Math.exp(1));
        assertEquals(expected, out.data()[0], 1e-4f);
    }

    @Test
    void siluAndSwiGLU() {
        FloatTensor gate = FloatTensor.of(new float[]{0, 2, -2, 10}, 1, 4);
        FloatTensor up = FloatTensor.of(new float[]{1, 3, 5, 7}, 1, 4);
        FloatTensor out = backend.siluMul(gate, up);
        assertEquals(0f, out.data()[0], 1e-6f);
        float sig2 = 1f / (1f + (float) Math.exp(-2));
        assertEquals(2f * sig2 * 3f, out.data()[1], 1e-5f);
        float sigNeg2 = 1f / (1f + (float) Math.exp(2));
        assertEquals(-2f * sigNeg2 * 5f, out.data()[2], 1e-5f);
        float sig10 = 1f / (1f + (float) Math.exp(-10));
        assertEquals(10f * sig10 * 7f, out.data()[3], 1e-4f);
    }

    @Test
    void sinusoidalPositionalEncodingReference() {
        int seqLen = 4;
        int dModel = 8;
        float[] pe = new float[seqLen * dModel];
        TensorMath.sinusoidalPositionalEncoding(pe, seqLen, dModel, 0);
        // position 0: sin(0)=0, cos(0)=1
        assertEquals(0f, pe[0], 1e-7f);
        assertEquals(1f, pe[1], 1e-7f);
        assertEquals(0f, pe[2], 1e-7f);
        assertEquals(1f, pe[3], 1e-7f);
        // position 1, dim 0: sin(1/1) = sin(1)
        assertEquals((float) Math.sin(1), pe[dModel], 1e-6f);
        assertEquals((float) Math.cos(1), pe[dModel + 1], 1e-6f);
        // dim 2: div = 10000^(2/8) = 10; sin(1/10)
        assertEquals((float) Math.sin(0.1), pe[dModel + 2], 1e-6f);
    }

    @Test
    void transposeReference() {
        FloatTensor a = FloatTensor.of(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        FloatTensor t = TensorMath.transpose(a);
        assertArrayEquals(new long[]{3, 2}, t.shape());
        assertArrayEquals(new float[]{1, 4, 2, 5, 3, 6}, t.data(), 1e-7f);
    }

    @Test
    void addAndMul() {
        FloatTensor a = FloatTensor.of(new float[]{1, 2, 3}, 1, 3);
        FloatTensor b = FloatTensor.of(new float[]{4, 5, 6}, 1, 3);
        assertArrayEquals(new float[]{5, 7, 9}, backend.add(a, b).data(), 1e-7f);
        assertArrayEquals(new float[]{4, 10, 18}, backend.mul(a, b).data(), 1e-7f);
    }
}
