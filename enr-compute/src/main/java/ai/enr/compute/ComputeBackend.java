package ai.enr.compute;

import ai.enr.core.tensor.FloatTensor;
import ai.enr.core.tensor.QuantizedWeight;

/**
 * Compute backend contract. CPU and CUDA backends implement equivalent
 * inference semantics; results must agree within documented tolerance.
 */
public interface ComputeBackend extends AutoCloseable {

    String name();

    DeviceInfo deviceInfo();

    FloatTensor matmul(FloatTensor a, FloatTensor b);

    /**
     * Matrix product against a quantized weight, without expanding the weight to
     * F32. Computes {@code y = x @ W} where {@code x} is {@code [seqLen, W.cols()]}
     * and {@code W} is stored {@code [W.rows()][W.cols()]} (output-major, each
     * output row contiguous). Returns {@code [seqLen, W.rows()]}.
     *
     * <p>The default implementation dequantizes the full weight and delegates to
     * {@link #matmul}; backends that support quantized weights override it with a
     * block-wise implementation.</p>
     */
    default FloatTensor matmulQuantized(FloatTensor x, QuantizedWeight w) {
        int in = w.cols();
        int out = w.rows();
        float[] raw = w.dequantize();
        float[] b = new float[in * out];
        for (int c = 0; c < out; c++) {
            for (int j = 0; j < in; j++) {
                b[j * out + c] = raw[c * in + j];
            }
        }
        return matmul(x, new FloatTensor(new long[]{in, out}, b));
    }

    FloatTensor rmsNorm(FloatTensor x, float[] weight, float eps);

    FloatTensor softmaxRows(FloatTensor x);

    FloatTensor silu(FloatTensor x);

    FloatTensor siluMul(FloatTensor gate, FloatTensor up);

    FloatTensor add(FloatTensor a, FloatTensor b);

    FloatTensor mul(FloatTensor a, FloatTensor b);

    /**
     * Multi-head causal attention: scores = q @ k^T per head with causal masking,
     * softmax, weighted sum over v. RoPE/sinusoidal positioning is applied by the
     * caller before this call.
     *
     * @param q        [heads, seqLen, headDim] query
     * @param k        [heads, kvSeqLen, headDim] key (kvSeqLen >= seqLen)
     * @param v        [heads, kvSeqLen, headDim] value
     * @param scale    attention scale (1/sqrt(headDim))
     * @param startPos absolute token position of the first query row
     * @return [heads, seqLen, headDim] attention output
     */
    FloatTensor attention(FloatTensor q, FloatTensor k, FloatTensor v, float scale, int startPos);

    default void close() {
    }
}
