package ai.enr.compute.cuda;

import ai.enr.compute.ComputeBackend;
import ai.enr.compute.cpu.CpuBackend;
import ai.enr.core.tensor.FloatTensor;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * CUDA backend tests: run against the real GPU when available (RTX 5070 Ti);
 * skipped when no CUDA device is present. CPU remains the reference.
 */
class CudaBackendTest {

    private final CudaBackend cuda = new CudaBackend();
    private final ComputeBackend cpu = new CpuBackend();

    private boolean requireCuda() {
        if (!cuda.deviceInfo().available()) {
            System.out.println("SKIP: CUDA unavailable (" + cuda.deviceInfo().name() + ")");
        }
        Assumptions.assumeTrue(cuda.deviceInfo().available());
        return true;
    }

    @Test
    void deviceDiscoveryReportsRealDevice() {
        if (!cuda.deviceInfo().available()) {
            return;
        }
        var info = cuda.deviceInfo();
        assertTrue(info.totalMemoryBytes() > 0);
        assertFalse(info.name().isBlank());
        System.out.println("CUDA device: " + info);
    }

    @Test
    void matmulMatchesCpu() {
        requireCuda();
        Random rnd = new Random(7);
        FloatTensor a = random(rnd, 64, 96);
        FloatTensor b = random(rnd, 96, 128);
        assertClose(cpu.matmul(a, b).data(), cuda.matmul(a, b).data(), 1e-3f);
    }

    @Test
    void matmulLargeMatchesCpu() {
        requireCuda();
        Random rnd = new Random(11);
        FloatTensor a = random(rnd, 512, 256);
        FloatTensor b = random(rnd, 256, 512);
        assertClose(cpu.matmul(a, b).data(), cuda.matmul(a, b).data(), 1e-3f);
    }

    @Test
    void rmsNormMatchesCpu() {
        requireCuda();
        Random rnd = new Random(3);
        FloatTensor x = random(rnd, 33, 64);
        float[] w = random(rnd, 64).data();
        assertClose(cpu.rmsNorm(x, w, 1e-5f).data(), cuda.rmsNorm(x, w, 1e-5f).data(), 1e-4f);
    }

    @Test
    void softmaxMatchesCpu() {
        requireCuda();
        Random rnd = new Random(5);
        FloatTensor x = random(rnd, 17, 40);
        assertClose(cpu.softmaxRows(x).data(), cuda.softmaxRows(x).data(), 1e-4f);
    }

    @Test
    void siluMulMatchesCpu() {
        requireCuda();
        Random rnd = new Random(9);
        FloatTensor gate = random(rnd, 8, 32);
        FloatTensor up = random(rnd, 8, 32);
        assertClose(cpu.siluMul(gate, up).data(), cuda.siluMul(gate, up).data(), 1e-4f);
    }

    @Test
    void addMulMatchCpu() {
        requireCuda();
        Random rnd = new Random(13);
        FloatTensor a = random(rnd, 7, 15);
        FloatTensor b = random(rnd, 7, 15);
        assertClose(cpu.add(a, b).data(), cuda.add(a, b).data(), 1e-6f);
        assertClose(cpu.mul(a, b).data(), cuda.mul(a, b).data(), 1e-5f);
    }

    @Test
    void attentionMatchesCpu() {
        requireCuda();
        Random rnd = new Random(21);
        int heads = 2, seqLen = 6, kvSeqLen = 10, headDim = 8;
        FloatTensor q = random(rnd, heads, seqLen, headDim);
        FloatTensor k = random(rnd, heads, kvSeqLen, headDim);
        FloatTensor v = random(rnd, heads, kvSeqLen, headDim);
        float scale = 1.0f / (float) Math.sqrt(headDim);
        float[] expected = cpu.attention(q, k, v, scale, 4).data();
        float[] actual = cuda.attention(q, k, v, scale, 4).data();
        assertClose(expected, actual, 1e-3f);
    }

    @Test
    void vramAccountingWorks() {
        requireCuda();
        Random rnd = new Random(1);
        FloatTensor a = random(rnd, 128, 128);
        FloatTensor b = random(rnd, 128, 128);
        cuda.matmul(a, b);
        assertTrue(cuda.allocatedVramBytes() == 0,
                "VRAM must be fully released after ops, allocated=" + cuda.allocatedVramBytes());
    }

    private static FloatTensor random(Random rnd, long... shape) {
        long size = 1;
        for (long dim : shape) {
            size *= dim;
        }
        float[] data = new float[(int) size];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) rnd.nextGaussian();
        }
        return new FloatTensor(shape, data);
    }

    private static void assertClose(float[] expected, float[] actual, float tolerance) {
        assertEquals(expected.length, actual.length);
        float maxDiff = 0f;
        for (int i = 0; i < expected.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(expected[i] - actual[i]));
        }
        assertTrue(maxDiff <= tolerance, "max difference " + maxDiff + " exceeds tolerance " + tolerance);
    }
}
