package ai.enr.loader;

import ai.enr.core.tensor.DType;
import ai.enr.loader.api.Architecture;
import ai.enr.loader.api.LoadedModel;
import ai.enr.loader.api.ModelFormat;
import ai.enr.loader.api.TensorEntry;
import ai.enr.loader.api.TensorStores;
import ai.enr.loader.api.TokenizerInfo;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Deterministic test fixtures for loader tests.
 */
public final class FixtureModels {

    private FixtureModels() {
    }

    public static LoadedModel tiny() {
        int vocab = 16;
        int dModel = 8;
        int dff = 16;
        int heads = 2;
        int blocks = 1;
        int maxSeqLen = 32;

        Architecture arch = new Architecture()
                .family("evo_llm")
                .vocabSize(vocab)
                .dModel(dModel)
                .numHeads(heads)
                .numBlocks(blocks)
                .dff(dff)
                .maxSeqLen(maxSeqLen)
                .normType("rms_norm")
                .normEps(1e-5f)
                .activationType("swiglu")
                .positionalEncoding("sinusoidal")
                .ropeTheta(10000.0f);

        TokenizerInfo tok = new TokenizerInfo()
                .type("BPE")
                .vocabSize(vocab)
                .bosTokenId(1)
                .eosTokenId(2)
                .unkTokenId(0)
                .padTokenId(0);
        tok.addToken("<unk>", 0);
        tok.addToken("<s>", 1);
        tok.addToken("</s>", 2);
        tok.addToken(" ", 3);
        for (int i = 4; i < vocab; i++) {
            tok.addToken("token_" + i, i);
        }

        Map<String, String> metadata = new LinkedHashMap<>();
        metadata.put("fixture", "true");

        Random rnd = new Random(42);
        List<TensorEntry> tensors = new ArrayList<>();
        int id = 0;
        tensors.add(entry(id++, "token_embd.weight", new long[]{vocab, dModel}, rnd));
        for (int b = 0; b < blocks; b++) {
            String p = "blk." + b + ".";
            tensors.add(entry(id++, p + "attn_norm.weight", new long[]{dModel}, rnd));
            tensors.add(entry(id++, p + "attn_q.weight", new long[]{dModel, dModel}, rnd));
            tensors.add(entry(id++, p + "attn_k.weight", new long[]{dModel, dModel}, rnd));
            tensors.add(entry(id++, p + "attn_v.weight", new long[]{dModel, dModel}, rnd));
            tensors.add(entry(id++, p + "attn_output.weight", new long[]{dModel, dModel}, rnd));
            tensors.add(entry(id++, p + "ffn_norm.weight", new long[]{dModel}, rnd));
            tensors.add(entry(id++, p + "ffn_gate.weight", new long[]{dModel, dff}, rnd));
            tensors.add(entry(id++, p + "ffn_up.weight", new long[]{dModel, dff}, rnd));
            tensors.add(entry(id++, p + "ffn_down.weight", new long[]{dff, dModel}, rnd));
        }
        tensors.add(entry(id++, "output_norm.weight", new long[]{dModel}, rnd));
        tensors.add(entry(id, "output.weight", new long[]{dModel, vocab}, rnd));

        return new LoadedModel(ModelFormat.EVO_V2, "2.0", "tiny", 0, "", arch, tok, metadata, tensors);
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
                1, TensorStores.checksumOfFloats(data), TensorStores.eager(data));
    }
}
