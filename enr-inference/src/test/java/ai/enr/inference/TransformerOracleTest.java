package ai.enr.inference;

import ai.enr.compute.cpu.CpuBackend;
import ai.enr.inference.session.InferenceSession;
import ai.enr.loader.ModelSources;
import ai.enr.loader.api.LoadedModel;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Numerical equivalence of the ENR inference engine against an independent
 * execution of the real Darwin Forge exported models.
 *
 * Oracle: llama.cpp (CPU) executed on the same real GGUF models with raw token
 * ids [4,3,5,3,6], KV cache F32. llama.cpp stores the KV cache in F16 by
 * default, so agreement is asserted within 2e-3 (the F16 quantization level).
 * The layout conventions were established empirically (see docs and audit).
 */
class TransformerOracleTest {

    private static final Path TINY = Path.of("../enr-model-loader/src/test/resources/models/tiny-evo.gguf");
    private static final Path EVO = Path.of("../enr-model-loader/src/test/resources/models/evo.gguf");

    private static final float[] ORACLE_TINY_5 = {
            0.397462696f, 1.15923619f, -1.51992023f, 0.987487793f, -1.3350662f, -1.88342321f,
            -0.402190119f, 0.612882674f, -0.625047028f, 1.78071833f, -1.18758798f, 0.255022556f,
            -0.501894891f, 0.73590678f, -0.805936992f, -1.05058205f
    };

    private static final float[] ORACLE_TINY_3 = {
            1.17277956f, 1.1369909f, 0.717825055f, -1.07524931f, 0.72039628f, 0.59814024f,
            0.860605955f, -0.0562948138f, 0.488860667f, 0.288450718f, 0.950937092f, -1.28280008f,
            1.36912715f, -1.80195642f, 0.895061374f, 0.29123649f
    };

    private LoadedModel load(Path path) throws ai.enr.loader.api.LoadException {
        return ModelSources.open(path);
    }

    @Test
    void tinyModelMatchesLlamaOracle() throws Exception {
        LoadedModel model = load(TINY);
        Transformer transformer = new Transformer(TransformerConfig.from(model.architecture()),
                new ModelWeights(model), new CpuBackend());
        KVCache kv = new KVCache(1, 2, 4, 32);
        float[] logits = transformer.forward(new int[]{4, 3, 5, 3, 6}, kv);
        assertClose(ORACLE_TINY_5, logits, 2e-3f);
    }

    @Test
    void tinyModelShorterSequenceMatchesOracle() throws Exception {
        LoadedModel model = load(TINY);
        Transformer transformer = new Transformer(TransformerConfig.from(model.architecture()),
                new ModelWeights(model), new CpuBackend());
        KVCache kv = new KVCache(1, 2, 4, 32);
        float[] logits = transformer.forward(new int[]{4, 3, 5}, kv);
        assertClose(ORACLE_TINY_3, logits, 2e-3f);
    }

    @Test
    void kvCacheIncrementalEqualsFullPass() throws Exception {
        LoadedModel model = load(TINY);
        TransformerConfig config = TransformerConfig.from(model.architecture());

        Transformer full = new Transformer(config, new ModelWeights(model), new CpuBackend());
        KVCache fullKv = new KVCache(1, 2, 4, 32);
        float[] fullLogits = full.forward(new int[]{4, 3, 5, 3, 6}, fullKv);

        Transformer inc = new Transformer(config, new ModelWeights(model), new CpuBackend());
        KVCache incKv = new KVCache(1, 2, 4, 32);
        inc.forward(new int[]{4, 3, 5}, incKv);
        inc.forward(new int[]{3}, incKv);
        float[] incLogits = inc.forward(new int[]{6}, incKv);

        assertClose(fullLogits, incLogits, 1e-5f);
    }

    @Test
    void evoModelGeneratesAndMatchesOracleArgmax() throws Exception {
        LoadedModel model = load(EVO);
        Transformer transformer = new Transformer(TransformerConfig.from(model.architecture()),
                new ModelWeights(model), new CpuBackend());
        KVCache kv = new KVCache(1, 2, 32, 8);
        float[] logits = transformer.forward(new int[]{4, 3, 5, 3, 6}, kv);
        assertEquals(21, argmax(logits), "argmax must match llama.cpp oracle (token 21)");
    }

    @Test
    void sessionGenerationProducesTokens() throws Exception {
        LoadedModel model = load(TINY);
        TransformerConfig config = TransformerConfig.from(model.architecture());
        Transformer transformer = new Transformer(config, new ModelWeights(model), new CpuBackend());

        ai.enr.tokenizer.EvoBpeTokenizer tokenizer = new ai.enr.tokenizer.EvoBpeTokenizer(
                model.tokenizer().tokenToId(), model.tokenizer().bosTokenId(),
                model.tokenizer().eosTokenId(), model.tokenizer().unkTokenId(),
                model.tokenizer().padTokenId());

        InferenceSession session = new InferenceSession("tiny", config);
        GenerationEngine engine = new GenerationEngine(transformer);
        var result = engine.generate(session,
                new GenerationEngine.GenerationRequest(new int[]{4, 3, 5}, 8,
                        ai.enr.inference.sample.SamplingConfig.defaults().withTemperature(0.7f),
                        java.util.Set.of()),
                tokenizer, null);

        assertNotNull(result.text());
        assertFalse(result.generatedTokenIds().isEmpty());
        assertNotEquals(GenerationEngine.TerminationReason.ERROR, result.reason(),
                "generation must not fail with ERROR");
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

    private static void assertClose(float[] expected, float[] actual, float tolerance) {
        assertEquals(expected.length, actual.length);
        float maxDiff = 0f;
        for (int i = 0; i < expected.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(expected[i] - actual[i]));
        }
        assertTrue(maxDiff <= tolerance,
                "max logits difference " + maxDiff + " exceeds tolerance " + tolerance);
    }
}
