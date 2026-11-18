package ai.enr.tests;

import ai.enr.compute.ComputeBackend;
import ai.enr.compute.cpu.CpuBackend;
import ai.enr.compute.cuda.CudaBackend;
import ai.enr.inference.KVCache;
import ai.enr.inference.ModelWeights;
import ai.enr.inference.Transformer;
import ai.enr.inference.TransformerConfig;
import ai.enr.loader.ModelSources;
import ai.enr.loader.api.LoadedModel;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end CUDA inference on the real model: CUDA vs CPU equivalence and
 * llama.cpp oracle agreement. Skipped when no CUDA device is present.
 */
class CudaInferenceIntegrationTest {

    private static final Path TINY = Path.of("../enr-model-loader/src/test/resources/models/tiny-evo.gguf");

    private static final float[] ORACLE_TINY_5 = {
            0.397462696f, 1.15923619f, -1.51992023f, 0.987487793f, -1.3350662f, -1.88342321f,
            -0.402190119f, 0.612882674f, -0.625047028f, 1.78071833f, -1.18758798f, 0.255022556f,
            -0.501894891f, 0.73590678f, -0.805936992f, -1.05058205f
    };

    @Test
    void cudaInferenceMatchesCpuAndOracle() throws Exception {
        CudaBackend cuda = new CudaBackend();
        org.junit.jupiter.api.Assumptions.assumeTrue(cuda.deviceInfo().available(),
                "CUDA device not available");

        LoadedModel model = ModelSources.open(TINY);
        TransformerConfig config = TransformerConfig.from(model.architecture());

        float[] cpuLogits = runForward(model, config, new CpuBackend());
        float[] cudaLogits = runForward(model, config, cuda);

        assertEquals(cpuLogits.length, cudaLogits.length);
        float maxDiff = 0f;
        for (int i = 0; i < cpuLogits.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(cpuLogits[i] - cudaLogits[i]));
        }
        assertTrue(maxDiff <= 2e-3f, "CUDA vs CPU logits differ by " + maxDiff);

        assertClose(ORACLE_TINY_5, cudaLogits, 3e-3f,
                "CUDA logits must match the llama.cpp oracle within F16 KV cache noise");
    }

    @Test
    void cudaGenerationProducesSameArgmaxAsCpu() throws Exception {
        CudaBackend cuda = new CudaBackend();
        org.junit.jupiter.api.Assumptions.assumeTrue(cuda.deviceInfo().available(),
                "CUDA device not available");

        LoadedModel model = ModelSources.open(TINY);
        TransformerConfig config = TransformerConfig.from(model.architecture());
        float[] cpuLogits = runForward(model, config, new CpuBackend());
        float[] cudaLogits = runForward(model, config, cuda);
        assertEquals(argmax(cpuLogits), argmax(cudaLogits));
    }

    private static float[] runForward(LoadedModel model, TransformerConfig config, ComputeBackend backend) {
        Transformer transformer = new Transformer(config, new ModelWeights(model), backend);
        KVCache kv = new KVCache(config.numBlocks(), config.numKvHeads(), config.headDim(), 32);
        return transformer.forward(new int[]{4, 3, 5, 3, 6}, kv);
    }

    private static int argmax(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }
        return best;
    }

    private static void assertClose(float[] expected, float[] actual, float tolerance, String message) {
        assertEquals(expected.length, actual.length);
        float maxDiff = 0f;
        for (int i = 0; i < expected.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(expected[i] - actual[i]));
        }
        assertTrue(maxDiff <= tolerance, message + " (max diff " + maxDiff + ")");
    }
}
