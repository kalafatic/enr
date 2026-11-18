package ai.enr.inference;

import ai.enr.compute.ComputeBackend;
import ai.enr.core.tensor.FloatTensor;

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
        int startPos = kvCache.size();

        FloatTensor embeddings = weights.get("token_embd.weight");
        float[] embData = embeddings.data();
        int vocabSize = (int) embeddings.shape()[0];

        float[] x = new float[seqLen * dModel];
        for (int r = 0; r < seqLen; r++) {
            int id = ids[r];
            if (id < 0 || id >= vocabSize) {
                id = 1;
            }
            System.arraycopy(embData, id * dModel, x, r * dModel, dModel);
        }

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
            float[] wq = weights.get(p + "attn_q.weight").data();
            float[] wk = weights.get(p + "attn_k.weight").data();
            float[] wv = weights.get(p + "attn_v.weight").data();
            float[] wo = weights.get(p + "attn_output.weight").data();
            float[] wg = weights.get(p + "ffn_gate.weight").data();
            float[] wu = weights.get(p + "ffn_up.weight").data();
            float[] wd = weights.get(p + "ffn_down.weight").data();

            FloatTensor xTensor = FloatTensor.of(x, seqLen, dModel);
            FloatTensor xn = backend.rmsNorm(xTensor, attnNorm, config.normEps());

            FloatTensor q = backend.matmul(xn, FloatTensor.of(wq, dModel, dModel));
            FloatTensor k = backend.matmul(xn, FloatTensor.of(wk, dModel, dModel));
            FloatTensor v = backend.matmul(xn, FloatTensor.of(wv, dModel, dModel));

            float[] qd = q.data().clone();
            float[] kd = k.data().clone();
            float[] vd = v.data().clone();

            if (config.usesRope()) {
                applyRope(qd, seqLen, heads, headDim, startPos);
                applyRope(kd, seqLen, heads, headDim, startPos);
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
                        System.arraycopy(kd, row * dModel + kvHead * headDim,
                                keys, targetOff, headDim);
                        System.arraycopy(vd, row * dModel + kvHead * headDim,
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

            FloatTensor o = backend.matmul(FloatTensor.of(attnFlat, seqLen, dModel),
                    FloatTensor.of(wo, dModel, dModel));
            float[] xAfterAttn = new float[x.length];
            for (int i = 0; i < x.length; i++) {
                xAfterAttn[i] = x[i] + o.data()[i];
            }
            x = xAfterAttn;

            FloatTensor xn2 = backend.rmsNorm(FloatTensor.of(x, seqLen, dModel), ffnNorm, config.normEps());
            FloatTensor gate = backend.matmul(xn2, FloatTensor.of(wg, dModel, config.dff()));
            FloatTensor up = backend.matmul(xn2, FloatTensor.of(wu, dModel, config.dff()));
            FloatTensor h = backend.siluMul(gate, up);
            FloatTensor ffnOut = backend.matmul(h, FloatTensor.of(wd, config.dff(), dModel));
            float[] xAfterFfn = new float[x.length];
            for (int i = 0; i < x.length; i++) {
                xAfterFfn[i] = x[i] + ffnOut.data()[i];
            }
            x = xAfterFfn;

            float[] kvKeys = new float[seqLen * kvHeads * headDim];
            float[] kvValues = new float[seqLen * kvHeads * headDim];
            for (int kvHead = 0; kvHead < kvHeads; kvHead++) {
                for (int r = 0; r < seqLen; r++) {
                    System.arraycopy(kd, r * dModel + kvHead * headDim,
                            kvKeys, r * kvHeads * headDim + kvHead * headDim, headDim);
                    System.arraycopy(vd, r * dModel + kvHead * headDim,
                            kvValues, r * kvHeads * headDim + kvHead * headDim, headDim);
                }
            }
            kvCache.append(b, kvKeys, kvValues, seqLen);
        }

        float[] outputNorm = weights.get("output_norm.weight").data();
        FloatTensor xn = backend.rmsNorm(FloatTensor.of(x, seqLen, dModel), outputNorm, config.normEps());

        FloatTensor wout;
        if (config.embeddingTied()) {
            FloatTensor emb = weights.get("token_embd.weight");
            wout = FloatTensor.of(emb.data(), dModel, vocabSize);
        } else {
            wout = weights.get("output.weight");
        }
        FloatTensor logits = backend.matmul(xn, wout);

        float[] last = new float[vocabSize];
        System.arraycopy(logits.data(), (seqLen - 1) * vocabSize, last, 0, vocabSize);
        return last;
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
