package ai.enr.compute;

import ai.enr.core.tensor.FloatTensor;

/**
 * Compute backend contract. CPU and CUDA backends implement equivalent
 * inference semantics; results must agree within documented tolerance.
 */
public interface ComputeBackend extends AutoCloseable {

    String name();

    DeviceInfo deviceInfo();

    FloatTensor matmul(FloatTensor a, FloatTensor b);

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
