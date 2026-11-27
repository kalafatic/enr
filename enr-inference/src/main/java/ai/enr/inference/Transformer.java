package ai.enr.inference;

import ai.enr.compute.ComputeBackend;
import ai.enr.core.tensor.FloatTensor;
import ai.enr.core.tensor.QuantizedWeight;

/**
 * Transformer executor implementing the verified EVO native semantics:
 * RMSNorm pre-norm, sinusoidal absolute PE (evo) or RoPE (gguf/llama),
 * causal multi-head attention with KV cache, SwiGLU FFN, final norm + output
 * projection. Supports grouped-query attention (numKvHeads &lt;= numHeads).
 */
public final class Transformer {

    private final TransformerConfig config;
    private final ModelWeights weights;
    private final ComputeBackend backend;
    private final float attentionScale;

    public Transformer(TransformerConfig config, ModelWeights weights, ComputeBackend backend) {
        this.config = config;
        this.weights = weights;
        this.backend = backend;
        this.attentionScale = 1.0f / (float) Math.sqrt(config.headDim());
    }

    /**
     * Runs the model on the given token ids and returns logits for the last
     * position. Appends new keys/values to the session KV cache.
     */
    public float[] forward(int[] ids, KVCache kvCache) {
        int seqLen = ids.length;
        int dModel = config.dModel();
        int heads = config.numHeads();
        int kvHeads = config.numKvHeads();
        int headDim = config.headDim();
        int kvDim = kvHeads * headDim;
        int startPos = kvCache.size();

        float[] x = new float[seqLen * dModel];
        int vocabSize = embedTokens(ids, x);

        if (!config.usesRope()) {
            float[] pe = new float[seqLen * dModel];
            ai.enr.core.tensor.TensorMath.sinusoidalPositionalEncoding(pe, seqLen, dModel, startPos);
            for (int i = 0; i < x.length; i++) {
                x[i] += pe[i];
            }
        }

        for (int b = 0; b < config.numBlocks(); b++) {
            String p = "blk." + b + ".";
            float[] attnNorm = weights.get(p + "attn_norm.weight").data();
            float[] ffnNorm = weights.get(p + "ffn_norm.weight").data();

            FloatTensor xTensor = FloatTensor.of(x, seqLen, dModel);
            FloatTensor xn = backend.rmsNorm(xTensor, attnNorm, config.normEps());

            FloatTensor q = matmulWeight(xn, p + "attn_q.weight");
            FloatTensor k = matmulWeight(xn, p + "attn_k.weight");
            FloatTensor v = matmulWeight(xn, p + "attn_v.weight");

            float[] qd = q.data().clone();
            float[] kd = k.data().clone();
            float[] vd = v.data().clone();

            if (config.usesRope()) {
                applyRope(qd, seqLen, heads, headDim, startPos);
                applyRope(kd, seqLen, kvHeads, headDim, startPos);
            }

            float[] kFlat = kvCache.keyCache(b);
            float[] vFlat = kvCache.valueCache(b);
            int kvLen = startPos + seqLen;

            float[] keys = new float[heads * kvLen * headDim];
            float[] values = new float[heads * kvLen * headDim];
            int kvHeadDivisor = heads / kvHeads;
            for (int h = 0; h < heads; h++) {
                int kvHead = h / kvHeadDivisor;
                for (int c = 0; c < kvLen; c++) {
                    int targetOff = h * kvLen * headDim + c * headDim;
                    if (c < startPos) {
                        System.arraycopy(kFlat, (c * kvHeads + kvHead) * headDim,
                                keys, targetOff, headDim);
                        System.arraycopy(vFlat, (c * kvHeads + kvHead) * headDim,
                                values, targetOff, headDim);
                    } else {
                        int row = c - startPos;
                        System.arraycopy(kd, row * kvDim + kvHead * headDim,
                                keys, targetOff, headDim);
                        System.arraycopy(vd, row * kvDim + kvHead * headDim,
                                values, targetOff, headDim);
                    }
                }
            }

            float[] qViewData = new float[heads * seqLen * headDim];
            for (int h = 0; h < heads; h++) {
                for (int r = 0; r < seqLen; r++) {
                    System.arraycopy(qd, r * dModel + h * headDim,
                            qViewData, (h * seqLen + r) * headDim, headDim);
                }
            }

            FloatTensor qView = FloatTensor.of(qViewData, heads, seqLen, headDim);
            FloatTensor kView = FloatTensor.of(keys, heads, kvLen, headDim);
            FloatTensor vView = FloatTensor.of(values, heads, kvLen, headDim);
            FloatTensor attnOut = backend.attention(qView, kView, vView, attentionScale, startPos);

            float[] attnFlat = new float[seqLen * dModel];
            float[] attnData = attnOut.data();
            for (int h = 0; h < heads; h++) {
                for (int r = 0; r < seqLen; r++) {
                    System.arraycopy(attnData, (h * seqLen + r) * headDim,
                            attnFlat, r * dModel + h * headDim, headDim);
                }
            }

            FloatTensor o = matmulWeight(FloatTensor.of(attnFlat, seqLen, dModel),
                    p + "attn_output.weight");
            float[] xAfterAttn = new float[x.length];
            for (int i = 0; i < x.length; i++) {
                xAfterAttn[i] = x[i] + o.data()[i];
            }
            x = xAfterAttn;

            FloatTensor xn2 = backend.rmsNorm(FloatTensor.of(x, seqLen, dModel), ffnNorm, config.normEps());
            FloatTensor gate = matmulWeight(xn2, p + "ffn_gate.weight");
            FloatTensor up = matmulWeight(xn2, p + "ffn_up.weight");
            FloatTensor h = backend.siluMul(gate, up);
            FloatTensor ffnOut = matmulWeight(h, p + "ffn_down.weight");
            float[] xAfterFfn = new float[x.length];
            for (int i = 0; i < x.length; i++) {
                xAfterFfn[i] = x[i] + ffnOut.data()[i];
            }
            x = xAfterFfn;

            float[] kvKeys = new float[seqLen * kvHeads * headDim];
            float[] kvValues = new float[seqLen * kvHeads * headDim];
            for (int kvHead = 0; kvHead < kvHeads; kvHead++) {
                for (int r = 0; r < seqLen; r++) {
                    System.arraycopy(kd, r * kvDim + kvHead * headDim,
                            kvKeys, r * kvHeads * headDim + kvHead * headDim, headDim);
                    System.arraycopy(vd, r * kvDim + kvHead * headDim,
                            kvValues, r * kvHeads * headDim + kvHead * headDim, headDim);
                }
            }
            kvCache.append(b, kvKeys, kvValues, seqLen);
        }

        float[] outputNorm = weights.get("output_norm.weight").data();
        FloatTensor xn = backend.rmsNorm(FloatTensor.of(x, seqLen, dModel), outputNorm, config.normEps());

        FloatTensor logits;
        if (config.embeddingTied()) {
            QuantizedWeight embQ = weights.quantizedWeight("token_embd.weight");
            if (embQ != null) {
                logits = backend.matmulQuantized(xn, embQ);
            } else {
                FloatTensor emb = weights.get("token_embd.weight");
                logits = backend.matmul(xn, FloatTensor.of(emb.data(), dModel, vocabSize));
            }
        } else {
            logits = matmulWeight(xn, "output.weight");
        }

        float[] last = new float[vocabSize];
        System.arraycopy(logits.data(), (seqLen - 1) * vocabSize, last, 0, vocabSize);
        return last;
    }

    /** Fills {@code x} with the embedding rows for {@code ids}; returns vocab size. */
    private int embedTokens(int[] ids, float[] x) {
        int dModel = config.dModel();
        QuantizedWeight embQ = weights.quantizedWeight("token_embd.weight");
        if (embQ != null) {
            int vocabSize = embQ.rows();
            float[] row = new float[dModel];
            for (int r = 0; r < ids.length; r++) {
                int id = ids[r];
                if (id < 0 || id >= vocabSize) {
                    id = 1;
                }
                embQ.getRow(id, row);
                System.arraycopy(row, 0, x, r * dModel, dModel);
            }
            return vocabSize;
        }
        FloatTensor embeddings = weights.get("token_embd.weight");
        float[] embData = embeddings.data();
        int vocabSize = (int) embeddings.shape()[0];
        for (int r = 0; r < ids.length; r++) {
            int id = ids[r];
            if (id < 0 || id >= vocabSize) {
                id = 1;
            }
            System.arraycopy(embData, id * dModel, x, r * dModel, dModel);
        }
        return vocabSize;
    }

    /** Matrix product against a weight, dispatching to the quantized path when stored quantized. */
    private FloatTensor matmulWeight(FloatTensor x, String name) {
        QuantizedWeight qw = weights.quantizedWeight(name);
        if (qw != null) {
            return backend.matmulQuantized(x, qw);
        }
        FloatTensor w = weights.get(name);
        return backend.matmul(x, FloatTensor.of(w.data(), w.shape()[0], w.shape()[1]));
    }

    private void applyRope(float[] qkv, int seqLen, int heads, int headDim, int startPos) {
        int nRot = Math.min(config.ropeDim(), headDim);
        if (nRot <= 0) {
            return;
        }
        float theta = config.ropeTheta();
        for (int r = 0; r < seqLen; r++) {
            int absolutePos = startPos + r;
            for (int h = 0; h < heads; h++) {
                int off = r * heads * headDim + h * headDim;
                for (int i = 0; i + 1 < nRot; i += 2) {
                    float invFreq = 1.0f / (float) Math.pow(theta, (double) i / nRot);
                    double angle = (double) absolutePos * invFreq;
                    float cos = (float) Math.cos(angle);
                    float sin = (float) Math.sin(angle);
                    float a = qkv[off + i];
                    float b = qkv[off + i + 1];
                    qkv[off + i] = a * cos - b * sin;
                    qkv[off + i + 1] = a * sin + b * cos;
                }
            }
        }
    }

    public TransformerConfig config() {
        return config;
    }

    public ComputeBackend backend() {
        return backend;
    }

    public ModelWeights weights() {
        return weights;
    }
}
