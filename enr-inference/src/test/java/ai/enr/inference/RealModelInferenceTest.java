package ai.enr.inference;

import ai.enr.compute.cpu.CpuBackend;
import ai.enr.inference.session.InferenceSession;
import ai.enr.inference.sample.SamplingConfig;
import ai.enr.loader.ModelSources;
import ai.enr.loader.api.LoadedModel;
import ai.enr.tokenizer.EvoBpeTokenizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Real-model inference against an actual Ollama GGUF artifact (defaults to
 * Llama 3.1 8B Q4_K_M). Enabled only when {@code enr.model.path} points at an
 * existing GGUF, so it is skipped in environments without the model present.
 *
 * <p>Verifies discovery/parsing (load), GQA configuration, quantized execution,
 * a complete forward pass and token generation.</p>
 */
@EnabledIfSystemProperty(named = "enr.model.path", matches = ".+")
class RealModelInferenceTest {

    private static final String DEFAULT_MODEL =
            System.getProperty("user.home") + "/.ollama/models/blobs/"
                    + "sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29";

    @Test
    void loadsAndGeneratesTokens() throws Exception {
        String path = System.getProperty("enr.model.path", DEFAULT_MODEL);
        Path modelPath = Path.of(path);
        if (!Files.isRegularFile(modelPath)) {
            throw new IllegalStateException("Model not found at " + modelPath
                    + "; set -Denr.model.path to a GGUF model");
        }

        long t0 = System.currentTimeMillis();
        LoadedModel model = ModelSources.open(modelPath);
        long loadMs = System.currentTimeMillis() - t0;

        // Use a small context window so the KV cache does not pre-allocate for
        // the model's full max_seq_len (131072 for Llama 3.1) during the test.
        var arch = model.architecture();
        TransformerConfig config = new TransformerConfig(
                arch.vocabSize(), arch.dModel(), arch.numHeads(), arch.numKvHeads(),
                arch.numBlocks(), arch.dff(), 32, arch.normEps(), arch.positionalEncoding(),
                arch.ropeTheta(), arch.ropeDim(), arch.embeddingTied());

        // GQA configuration must reflect the model's head counts.
        assertTrue(config.numKvHeads() <= config.numHeads(),
                "invalid GQA configuration: kvHeads=" + config.numKvHeads()
                        + " heads=" + config.numHeads());
        assertTrue(config.numKvHeads() > 0);

        Transformer transformer = new Transformer(config, new ModelWeights(model), new CpuBackend());
        KVCache kv = new KVCache(config.numBlocks(), config.numKvHeads(), config.headDim(), 32);

        // Prefill with the BOS token, then a single decode step.
        int bos = model.tokenizer().bosTokenId() >= 0 ? model.tokenizer().bosTokenId() : 1;
        long t1 = System.currentTimeMillis();
        float[] logits = transformer.forward(new int[]{bos}, kv);
        long forwardMs = System.currentTimeMillis() - t1;

        assertEquals(config.vocabSize(), logits.length);
        int argmax = 0;
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > logits[argmax]) {
                argmax = i;
            }
        }
        assertTrue(Float.isFinite(logits[argmax]), "non-finite argmax logit");
        assertTrue(argmax >= 0 && argmax < config.vocabSize());

        // Decode a second token through the generation loop.
        EvoBpeTokenizer tokenizer = new EvoBpeTokenizer(
                model.tokenizer().tokenToId(), model.tokenizer().bosTokenId(),
                model.tokenizer().eosTokenId(), model.tokenizer().unkTokenId(),
                model.tokenizer().padTokenId());
        InferenceSession session = new InferenceSession("real", config);
        GenerationEngine engine = new GenerationEngine(transformer);
        var result = engine.generate(session,
                new GenerationEngine.GenerationRequest(new int[]{bos}, 3,
                        SamplingConfig.defaults().withTemperature(0.0f), Set.of()),
                tokenizer, null);

        assertFalse(result.generatedTokenIds().isEmpty(), "no tokens generated");
        assertNotEquals(GenerationEngine.TerminationReason.ERROR, result.reason(),
                "generation must not fail with ERROR");

        long heap = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        System.out.println("REAL_MODEL name=" + model.name()
                + " params=" + model.totalParameters()
                + " heads=" + config.numHeads() + " kvHeads=" + config.numKvHeads()
                + " loadMs=" + loadMs + " prefillMs=" + forwardMs
                + " heapMiB=" + (heap >> 20));
    }

    /**
     * Verifies the KV cache grows with actual usage, not the theoretical
     * max_seq_len (131072 for Llama 3.1). For each context capacity the cache is
     * filled to exactly that length and its live footprint is measured.
     */
    @Test
    void kvCacheGrowsWithActualContext() throws Exception {
        String path = System.getProperty("enr.model.path", DEFAULT_MODEL);
        Path modelPath = Path.of(path);
        if (!Files.isRegularFile(modelPath)) {
            throw new IllegalStateException("Model not found at " + modelPath);
        }
        LoadedModel model = ModelSources.open(modelPath);
        var arch = model.architecture();
        int layers = arch.numBlocks();
        int kvHeads = arch.numKvHeads();
        int headDim = arch.headDim();
        int maxSeqLen = arch.maxSeqLen();

        for (int capacity : new int[]{32, 512, 2048, 8192}) {
            KVCache kv = new KVCache(layers, kvHeads, headDim, maxSeqLen);
            fillTo(kv, layers, kvHeads, headDim, capacity);
            long allocated = kv.allocatedBytes();
            long theoreticalMax = kv.maxBytes();
            assertEquals(capacity, kv.size());
            // The allocated capacity covers actual usage and stays far below the
            // theoretical max_seq_len window.
            assertTrue(kv.capacity() >= capacity, "capacity must cover actual usage");
            assertTrue(kv.capacity() < maxSeqLen, "must not pre-allocate the full context window");
            assertTrue(allocated < theoreticalMax,
                    "allocated " + allocated + " should be below theoretical max " + theoreticalMax);
            System.out.println("KV_CACHE context=" + capacity
                    + " allocatedCapacity=" + kv.capacity()
                    + " allocatedMiB=" + (allocated >> 20)
                    + " theoreticalMaxMiB=" + (theoreticalMax >> 20));
        }
    }

    private static void fillTo(KVCache kv, int layers, int kvHeads, int headDim, int capacity) {
        int perLayer = capacity * kvHeads * headDim;
        float[] keys = new float[perLayer];
        float[] values = new float[perLayer];
        java.util.Arrays.fill(keys, 0.25f);
        java.util.Arrays.fill(values, 0.5f);
        for (int layer = 0; layer < layers; layer++) {
            kv.append(layer, keys, values, capacity);
        }
    }
}
