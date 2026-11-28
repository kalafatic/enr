package ai.enr.compute.cpu;

import ai.enr.core.gguf.GgufQuant;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD (Java Vector API / jdk.incubator.vector) kernels for the quantized GEMV.
 * Uses 256-bit vectors (8 floats / 32 bytes). Used only when the incubator
 * module is present at runtime; otherwise the scalar reference path is used.
 */
final class SimdOps {

    private static final VectorSpecies<Float> FSPEC = FloatVector.SPECIES_256;
    private static final int LANES = FSPEC.length(); // 8

    private SimdOps() {
    }

    /** Generic F32 dot product. */
    static float dot(float[] a, int aOff, float[] b, int bOff, int len) {
        FloatVector acc = FloatVector.zero(FSPEC);
        int i = 0;
        for (; i < FSPEC.loopBound(len); i += LANES) {
            FloatVector va = FloatVector.fromArray(FSPEC, a, aOff + i);
            FloatVector vb = FloatVector.fromArray(FSPEC, b, bOff + i);
            acc = va.fma(vb, acc);
        }
        float r = acc.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) {
            r += a[aOff + i] * b[bOff + i];
        }
        return r;
    }

    /** F32 sum reduction. */
    static float sum(float[] x, int xOff, int len) {
        FloatVector acc = FloatVector.zero(FSPEC);
        int i = 0;
        for (; i < FSPEC.loopBound(len); i += LANES) {
            acc = acc.add(FloatVector.fromArray(FSPEC, x, xOff + i));
        }
        float r = acc.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) {
            r += x[xOff + i];
        }
        return r;
    }

    /** dot(x, dequantized Q8_0 block of 32 int8) — fused int8 → float. */
    static float dotQ8_0(float[] x, int xOff, byte[] raw, int rawOff) {
        float scale = GgufQuant.fp16ToFp32((raw[rawOff] & 0xFF) | ((raw[rawOff + 1] & 0xFF) << 8));
        return scale * dotInt8(x, xOff, raw, rawOff + 2);
    }

    /** dot(x, dequantized Q4_K super-block of 256 elements) — fused nibble + scale/min. */
    static float dotQ4_K(float[] x, int xOff, byte[] raw, int rawOff) {
        float d = GgufQuant.fp16ToFp32((raw[rawOff] & 0xFF) | ((raw[rawOff + 1] & 0xFF) << 8));
        float dmin = GgufQuant.fp16ToFp32((raw[rawOff + 2] & 0xFF) | ((raw[rawOff + 3] & 0xFF) << 8));
        int sBase = rawOff + 4;
        int qBase = rawOff + 16;
        float acc = 0f;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            int sm0 = GgufQuant.scaleMinK4(is, raw, sBase);
            float d1 = d * (sm0 & 0xFF);
            float m1 = dmin * ((sm0 >> 8) & 0xFF);
            int sm1 = GgufQuant.scaleMinK4(is + 1, raw, sBase);
            float d2 = d * (sm1 & 0xFF);
            float m2 = dmin * ((sm1 >> 8) & 0xFF);
            acc += d1 * dotNibbles(x, xOff + j, raw, qBase, false)
                    - m1 * sum(x, xOff + j, 32);
            acc += d2 * dotNibbles(x, xOff + j + 32, raw, qBase, true)
                    - m2 * sum(x, xOff + j + 32, 32);
            qBase += 32;
            is += 2;
        }
        return acc;
    }

    /** sum(x[i] * int8[i]) for 32 elements. */
    private static float dotInt8(float[] x, int xOff, byte[] b, int bOff) {
        ByteVector bv = ByteVector.fromArray(ByteVector.SPECIES_256, b, bOff); // 32 bytes
        FloatVector acc = FloatVector.zero(FSPEC);
        for (int part = 0; part < 4; part++) {
            IntVector iv = (IntVector) bv.convert(VectorOperators.B2I, part);
            FloatVector fv = (FloatVector) iv.convert(VectorOperators.I2F, 0);
            FloatVector xv = FloatVector.fromArray(FSPEC, x, xOff + part * LANES);
            acc = xv.fma(fv, acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    /** sum(x[i] * nibble[i]) for 32 elements; nibbles are the low or high 4 bits. */
    private static float dotNibbles(float[] x, int xOff, byte[] b, int bOff, boolean high) {
        ByteVector raw = ByteVector.fromArray(ByteVector.SPECIES_256, b, bOff);
        ByteVector nibbles = high
                ? raw.lanewise(VectorOperators.LSHR, 4)
                : raw.and((byte) 0x0F);
        FloatVector acc = FloatVector.zero(FSPEC);
        for (int part = 0; part < 4; part++) {
            IntVector iv = (IntVector) nibbles.convert(VectorOperators.B2I, part);
            FloatVector fv = (FloatVector) iv.convert(VectorOperators.I2F, 0);
            FloatVector xv = FloatVector.fromArray(FSPEC, x, xOff + part * LANES);
            acc = xv.fma(fv, acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }
}
