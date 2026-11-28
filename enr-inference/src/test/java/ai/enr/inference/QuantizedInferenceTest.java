package ai.enr.inference;

import ai.enr.compute.cpu.CpuBackend;
import ai.enr.loader.ModelSources;
import ai.enr.loader.api.LoadedModel;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end inference on quantized GGUF models: the dequantized weights must
 * reproduce the F32 reference model's forward pass closely enough that logits
 * and argmax agree. Quantized fixtures were produced from {@code evo.gguf} by
 * {@code llama-quantize}.
 */
class QuantizedInferenceTest {

    private static final Path DIR = Path.of("../enr-model-loader/src/test/resources/models");

    @Test
    void f16InferenceMatchesF32() throws Exception {
        assertMatches("evo-F16.gguf", 0.05f);
    }

    @Test
    void q8_0InferenceMatchesF32() throws Exception {
        assertMatches("evo-Q8_0.gguf", 0.15f);
    }

    @Test
    void q4_0InferenceMatchesF32() throws Exception {
        assertMatches("evo-Q4_0.gguf", 0.60f);
    }

    @Test
    void q4_1InferenceMatchesF32() throws Exception {
        assertMatches("evo-Q4_1.gguf", 0.90f);
    }

    @Test
    void q5_0InferenceMatchesF32() throws Exception {
        assertMatches("evo-Q5_0.gguf", 0.30f);
    }

    @Test
    void q5_1InferenceMatchesF32() throws Exception {
        assertMatches("evo-Q5_1.gguf", 0.30f);
    }

    private static void assertMatches(String quantFile, float tolerance) throws Exception {
        LoadedModel f32 = ModelSources.open(DIR.resolve("evo.gguf"));
        LoadedModel quant = ModelSources.open(DIR.resolve(quantFile));

        float[] f32Logits = forward(f32);
        float[] quantLogits = forward(quant);

        assertEquals(f32Logits.length, quantLogits.length, quantFile + " logits length differs");
        assertEquals(argmax(f32Logits), argmax(quantLogits),
                quantFile + " argmax differs from F32 reference");

        float maxDiff = 0f;
        for (int i = 0; i < f32Logits.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(f32Logits[i] - quantLogits[i]));
        }
        assertTrue(maxDiff <= tolerance,
                quantFile + " max logit diff " + maxDiff + " exceeds tolerance " + tolerance);
    }

    private static float[] forward(LoadedModel model) {
        TransformerConfig config = TransformerConfig.from(model.architecture());
        Transformer transformer = new Transformer(config, new ModelWeights(model), new CpuBackend());
        KVCache kv = new KVCache(config.numBlocks(), config.numKvHeads(), config.headDim(), 8);
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
}
