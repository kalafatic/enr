package ai.enr.inference;

import ai.enr.compute.cpu.CpuBackend;
import ai.enr.core.tensor.DType;
import ai.enr.loader.api.Architecture;
import ai.enr.loader.api.LoadedModel;
import ai.enr.loader.api.ModelFormat;
import ai.enr.loader.api.TensorEntry;
import ai.enr.loader.api.TensorStores;
import ai.enr.loader.api.TokenizerInfo;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Grouped-query attention correctness. A GQA model (numHeads &gt; numKvHeads)
 * must produce identical logits to its MHA expansion (each KV head duplicated
 * across its query-head group), since GQA is equivalent to MHA with shared K/V.
 */
class GqaTest {

    private static final int D_MODEL = 16;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int HEAD_DIM = D_MODEL / HEADS; // 4
    private static final int DFF = 32;
    private static final int VOCAB = 8;

    @Test
    void gqaMatchesExpandedMha() {
        Random rnd = new Random(1234);
        LoadedModel gqa = buildGqa(rnd);
        LoadedModel mha = buildMha(gqa);

        float[] gqaLogits = forward(gqa);
        float[] mhaLogits = forward(mha);

        assertEquals(gqaLogits.length, mhaLogits.length);
        float maxDiff = 0f;
        for (int i = 0; i < gqaLogits.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(gqaLogits[i] - mhaLogits[i]));
        }
        assertEquals(argmax(gqaLogits), argmax(mhaLogits), "GQA argmax differs from MHA");
        assertTrue(maxDiff < 1e-5f, "GQA vs MHA logits differ by " + maxDiff);
    }

    @Test
    void gqaForwardProducesCorrectDimensions() {
        float[] logits = forward(buildGqa(new Random(7)));
        assertEquals(VOCAB, logits.length);
        for (float v : logits) {
            assertTrue(Float.isFinite(v), "non-finite logit");
        }
    }

    private static float[] forward(LoadedModel model) {
        TransformerConfig config = TransformerConfig.from(model.architecture());
        Transformer transformer = new Transformer(config, new ModelWeights(model), new CpuBackend());
        KVCache kv = new KVCache(config.numBlocks(), config.numKvHeads(), config.headDim(), 8);
        return transformer.forward(new int[]{1, 2, 3, 4, 5}, kv);
    }

    private static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }

    private static LoadedModel buildGqa(Random rnd) {
        List<TensorEntry> tensors = new ArrayList<>();
        int id = 0;
        tensors.add(entry(id++, "token_embd.weight", new long[]{VOCAB, D_MODEL}, rnd));
        tensors.add(entry(id++, "blk.0.attn_norm.weight", new long[]{D_MODEL}, rnd));
        tensors.add(entry(id++, "blk.0.attn_q.weight", new long[]{D_MODEL, D_MODEL}, rnd));
        tensors.add(kvEntry(id++, "blk.0.attn_k.weight", KV_HEADS, null, null, rnd));
        tensors.add(kvEntry(id++, "blk.0.attn_v.weight", KV_HEADS, null, null, rnd));
        tensors.add(entry(id++, "blk.0.attn_output.weight", new long[]{D_MODEL, D_MODEL}, rnd));
        tensors.add(entry(id++, "blk.0.ffn_norm.weight", new long[]{D_MODEL}, rnd));
        tensors.add(entry(id++, "blk.0.ffn_gate.weight", new long[]{D_MODEL, DFF}, rnd));
        tensors.add(entry(id++, "blk.0.ffn_up.weight", new long[]{D_MODEL, DFF}, rnd));
        tensors.add(entry(id++, "blk.0.ffn_down.weight", new long[]{DFF, D_MODEL}, rnd));
        tensors.add(entry(id++, "output_norm.weight", new long[]{D_MODEL}, rnd));
        tensors.add(entry(id, "output.weight", new long[]{D_MODEL, VOCAB}, rnd));
        return assemble(HEADS, KV_HEADS, tensors);
    }

    /** Builds the MHA-expansion reusing the GQA model's weights, expanding only attn_k/v. */
    private static LoadedModel buildMha(LoadedModel gqa) {
        List<TensorEntry> tensors = new ArrayList<>();
        int id = 0;
        for (TensorEntry t : gqa.tensors()) {
            if (t.name().equals("blk.0.attn_k.weight")) {
                tensors.add(kvEntry(id++, t.name(), HEADS, t.loadFloats(), null, null));
            } else if (t.name().equals("blk.0.attn_v.weight")) {
                tensors.add(kvEntry(id++, t.name(), HEADS, null, t.loadFloats(), null));
            } else {
                tensors.add(new TensorEntry(id++, t.name(), t.dtype(), t.shape(), t.layout(), t.offset(),
                        t.compressedSize(), t.uncompressedSize(), t.compression(), t.checksum(),
                        TensorStores.eager(t.loadFloats())));
            }
        }
        return assemble(HEADS, HEADS, tensors);
    }

    private static LoadedModel assemble(int heads, int kvHeads, List<TensorEntry> tensors) {
        Architecture arch = new Architecture()
                .family("evo_llm")
                .vocabSize(VOCAB)
                .dModel(D_MODEL)
                .numHeads(heads)
                .numKvHeads(kvHeads)
                .numBlocks(1)
                .dff(DFF)
                .maxSeqLen(32)
                .normType("rms_norm")
                .normEps(1e-5f)
                .activationType("swiglu")
                .positionalEncoding("sinusoidal")
                .ropeTheta(10000.0f)
                .embeddingTied(false);
        TokenizerInfo tok = new TokenizerInfo().type("BPE").vocabSize(VOCAB)
                .bosTokenId(1).eosTokenId(2).unkTokenId(0).padTokenId(0);
        for (int i = 0; i < VOCAB; i++) {
            tok.addToken("t" + i, i);
        }
        return new LoadedModel(ModelFormat.EVO_V2, "2.0", "gqa", 0, "", arch, tok,
                new LinkedHashMap<>(), tensors);
    }

    /**
     * Builds an attn_k/attn_v weight of shape [dModel][kvHeads*headDim]. When
     * {@code srcK}/{@code srcV} are non-null (MHA expansion), each query head h
     * carries a copy of the GQA group's KV head (h / groups).
     */
    private static TensorEntry kvEntry(int id, String name, int kvHeads, float[] srcK, float[] srcV, Random rnd) {
        int kvDim = kvHeads * HEAD_DIM;
        float[] data = new float[D_MODEL * kvDim];
        if (srcK == null && srcV == null) {
            for (int i = 0; i < data.length; i++) {
                data[i] = (float) (rnd.nextGaussian() * 0.1);
            }
        } else {
            float[] src = srcK != null ? srcK : srcV;
            int groups = HEADS / KV_HEADS;
            for (int h = 0; h < kvHeads; h++) {
                int gqaHead = h / groups;
                for (int i = 0; i < D_MODEL; i++) {
                    System.arraycopy(src, i * (KV_HEADS * HEAD_DIM) + gqaHead * HEAD_DIM,
                            data, i * kvDim + h * HEAD_DIM, HEAD_DIM);
                }
            }
        }
        return new TensorEntry(id, name, DType.F32, new long[]{D_MODEL, kvDim}, 0, 0,
                data.length * 4L, data.length * 4L, 0, "", TensorStores.eager(data));
    }

    private static TensorEntry entry(int id, String name, long[] shape, Random rnd) {
        long count = 1;
        for (long d : shape) {
            count *= d;
        }
        float[] data = new float[(int) count];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) (rnd.nextGaussian() * 0.1);
        }
        return new TensorEntry(id, name, DType.F32, shape, 0, 0, data.length * 4L, data.length * 4L,
                0, "", TensorStores.eager(data));
    }
}
