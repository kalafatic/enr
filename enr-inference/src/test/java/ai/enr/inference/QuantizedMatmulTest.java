package ai.enr.inference;

import ai.enr.compute.cpu.CpuBackend;
import ai.enr.core.tensor.FloatTensor;
import ai.enr.core.tensor.QuantizedWeight;
import ai.enr.loader.ModelSources;
import ai.enr.loader.api.LoadedModel;
import ai.enr.loader.api.TensorEntry;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Quantized matmul correctness: the block-wise {@link CpuBackend#matmulQuantized}
 * must match the F32 reference (full dequantization + transpose + matmul), which
 * is itself validated against llama.cpp output in the model-loader tests.
 */
class QuantizedMatmulTest {

    private static final Path DIR = Path.of("../enr-model-loader/src/test/resources/models");

    @Test
    void f16MatmulMatchesReference() throws Exception {
        assertMatches("evo-F16.gguf");
    }

    @Test
    void q8_0MatmulMatchesReference() throws Exception {
        assertMatches("evo-Q8_0.gguf");
    }

    @Test
    void q4_0MatmulMatchesReference() throws Exception {
        assertMatches("evo-Q4_0.gguf");
    }

    @Test
    void q4_1MatmulMatchesReference() throws Exception {
        assertMatches("evo-Q4_1.gguf");
    }

    @Test
    void q5_0MatmulMatchesReference() throws Exception {
        assertMatches("evo-Q5_0.gguf");
    }

    @Test
    void q5_1MatmulMatchesReference() throws Exception {
        assertMatches("evo-Q5_1.gguf");
    }

    private static void assertMatches(String fixture) throws Exception {
        LoadedModel model = ModelSources.open(DIR.resolve(fixture));
        CpuBackend backend = new CpuBackend();
        Random rnd = new Random(42);
        int checked = 0;
        for (TensorEntry t : model.tensors()) {
            QuantizedWeight qw = t.quantizedWeight();
            if (qw == null) {
                continue;
            }
            int seqLen = 3;
            int in = qw.cols();
            int out = qw.rows();
            float[] xData = new float[seqLen * in];
            for (int i = 0; i < xData.length; i++) {
                xData[i] = (float) rnd.nextGaussian();
            }
            FloatTensor x = FloatTensor.of(xData, seqLen, in);

            FloatTensor quant = backend.matmulQuantized(x, qw);

            // Reference: dequantize fully (raw [out][in]), transpose to [in][out], matmul.
            float[] raw = qw.dequantize();
            float[] transposed = new float[in * out];
            for (int c = 0; c < out; c++) {
                for (int j = 0; j < in; j++) {
                    transposed[j * out + c] = raw[c * in + j];
                }
            }
            FloatTensor ref = backend.matmul(x, FloatTensor.of(transposed, in, out));

            assertClose(ref.data(), quant.data(), t.name());
            checked++;
        }
        assertTrue(checked > 0, "no quantized tensors found in " + fixture);
    }

    private static void assertClose(float[] expected, float[] actual, String name) {
        assertEquals(expected.length, actual.length, name + " length mismatch");
        float maxDiff = 0f;
        for (int i = 0; i < expected.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(expected[i] - actual[i]));
        }
        assertTrue(maxDiff < 1e-3f,
                name + " quantized matmul max diff " + maxDiff + " exceeds tolerance");
    }
}
