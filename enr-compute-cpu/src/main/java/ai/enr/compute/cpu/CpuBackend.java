package ai.enr.compute.cpu;

import ai.enr.compute.ComputeBackend;
import ai.enr.compute.DeviceInfo;
import ai.enr.core.tensor.FloatTensor;
import ai.enr.core.tensor.TensorMath;

/**
 * Reference CPU compute backend. Deterministic sequential kernels
 * implemented on top of the shared enr-core tensor math.
 */
public final class CpuBackend implements ComputeBackend {

    public CpuBackend() {
    }

    @Override
    public String name() {
        return "cpu";
    }

    @Override
    public DeviceInfo deviceInfo() {
        return DeviceInfo.cpu();
    }

    @Override
    public FloatTensor matmul(FloatTensor a, FloatTensor b) {
        return TensorMath.matmul(a, b);
    }

    @Override
    public FloatTensor rmsNorm(FloatTensor x, float[] weight, float eps) {
        return TensorMath.rmsNorm(x, weight, eps);
    }

    @Override
    public FloatTensor softmaxRows(FloatTensor x) {
        return TensorMath.softmaxRows(x);
    }

    @Override
    public FloatTensor silu(FloatTensor x) {
        return TensorMath.silu(x);
    }

    @Override
    public FloatTensor siluMul(FloatTensor gate, FloatTensor up) {
        return TensorMath.mul(TensorMath.silu(gate), up);
    }

    @Override
    public FloatTensor add(FloatTensor a, FloatTensor b) {
        return TensorMath.add(a, b);
    }

    @Override
    public FloatTensor mul(FloatTensor a, FloatTensor b) {
        return TensorMath.mul(a, b);
    }

    @Override
    public FloatTensor attention(FloatTensor q, FloatTensor k, FloatTensor v, float scale, int startPos) {
        int heads = (int) q.shape()[0];
        int seqLen = (int) q.shape()[1];
        int kvSeqLen = (int) k.shape()[1];
        int headDim = (int) q.shape()[2];

        float[] qd = q.data();
        float[] kd = k.data();
        float[] vd = v.data();
        float[] out = new float[heads * seqLen * headDim];

        for (int h = 0; h < heads; h++) {
            int qOff = h * seqLen * headDim;
            int kvOff = h * kvSeqLen * headDim;
            for (int r = 0; r < seqLen; r++) {
                int absolutePos = startPos + r;
                int attendUpTo = Math.min(absolutePos, kvSeqLen - 1);
                float maxVal = Float.NEGATIVE_INFINITY;
                float[] scores = new float[kvSeqLen];
                for (int c = 0; c <= attendUpTo; c++) {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) {
                        dot += qd[qOff + r * headDim + d] * kd[kvOff + c * headDim + d];
                    }
                    float score = dot * scale;
                    scores[c] = score;
                    if (score > maxVal) {
                        maxVal = score;
                    }
                }
                float sum = 0f;
                for (int c = 0; c <= attendUpTo; c++) {
                    float e = (float) Math.exp(scores[c] - maxVal);
                    scores[c] = e;
                    sum += e;
                }
                if (sum > 0f) {
                    for (int c = 0; c <= attendUpTo; c++) {
                        scores[c] /= sum;
                    }
                }
                int outOff = qOff + r * headDim;
                for (int c = 0; c <= attendUpTo; c++) {
                    float p = scores[c];
                    if (p == 0f) {
                        continue;
                    }
                    for (int d = 0; d < headDim; d++) {
                        out[outOff + d] += p * vd[kvOff + c * headDim + d];
                    }
                }
            }
        }
        return new FloatTensor(new long[]{heads, seqLen, headDim}, out);
    }
}
