package ai.enr.loader.gguf;

import ai.enr.loader.ModelSources;
import ai.enr.loader.api.LoadedModel;
import ai.enr.loader.api.TensorEntry;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end dequantization validation against llama.cpp: each quantized fixture
 * was produced by {@code llama-quantize} from the F32 {@code evo.gguf} source, so
 * dequantizing a quantized tensor must reproduce the source values within the
 * known per-format quantization error bound.
 */
class GgufQuantizedModelTest {

    private static final Path F32 = Path.of("src/test/resources/models/evo.gguf");

    @Test
    void f16DequantizesCloseToF32() throws Exception {
        assertClose("evo-F16.gguf", 0.005f);
    }

    @Test
    void q8_0DequantizesCloseToF32() throws Exception {
        assertClose("evo-Q8_0.gguf", 0.02f);
    }

    @Test
    void q4_0DequantizesCloseToF32() throws Exception {
        assertClose("evo-Q4_0.gguf", 0.10f);
    }

    @Test
    void q4_1DequantizesCloseToF32() throws Exception {
        assertClose("evo-Q4_1.gguf", 0.12f);
    }

    @Test
    void q5_0DequantizesCloseToF32() throws Exception {
        assertClose("evo-Q5_0.gguf", 0.06f);
    }

    @Test
    void q5_1DequantizesCloseToF32() throws Exception {
        assertClose("evo-Q5_1.gguf", 0.08f);
    }

    private static void assertClose(String quantFile, float relTol) throws Exception {
        LoadedModel f32 = ModelSources.open(F32);
        LoadedModel quant = ModelSources.open(Path.of("src/test/resources/models/" + quantFile));
        assertEquals(f32.tensors().size(), quant.tensors().size(),
                quantFile + " tensor count differs from F32 source");
        for (TensorEntry q : quant.tensors()) {
            TensorEntry f = f32.requireTensor(q.name());
            assertArrayEquals(f.shape(), q.shape(), quantFile + " shape mismatch for " + q.name());
            float[] qd = q.loadFloats();
            float[] fd = f.loadFloats();
            assertEquals(fd.length, qd.length, quantFile + " element count mismatch for " + q.name());

            float maxAbs = 0f;
            for (float v : fd) {
                maxAbs = Math.max(maxAbs, Math.abs(v));
            }
            float maxDiff = 0f;
            for (int i = 0; i < fd.length; i++) {
                assertTrue(Float.isFinite(qd[i]),
                        quantFile + " non-finite dequantized value in " + q.name() + " at index " + i);
                maxDiff = Math.max(maxDiff, Math.abs(qd[i] - fd[i]));
            }
            float bound = relTol * Math.max(maxAbs, 1e-3f);
            assertTrue(maxDiff <= bound,
                    quantFile + " tensor " + q.name() + ": max abs diff " + maxDiff
                            + " exceeds bound " + bound + " (maxAbs " + maxAbs + ")");
        }
    }
}
