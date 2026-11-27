package ai.enr.loader.gguf;

import ai.enr.core.gguf.GgufQuant;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for GGUF quantized dequantization: FP16 half-float conversion,
 * row-size accounting, and single-block decoding of the K-quant formats. The
 * simple block formats (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0) and F16 are additionally
 * validated end-to-end against real llama.cpp output in
 * {@link GgufQuantizedModelTest}; the K-quant formats are validated here against
 * hand-computed expectations derived from the llama.cpp reference layouts.
 */
class GgufQuantTest {

    @Test
    void fp16ToFp32EdgeCases() {
        assertEquals(0.0f, GgufQuant.fp16ToFp32(0x0000));
        assertEquals(1.0f, GgufQuant.fp16ToFp32(0x3C00));
        assertEquals(0.5f, GgufQuant.fp16ToFp32(0x3800));
        assertEquals(-2.0f, GgufQuant.fp16ToFp32(0xC000));
        assertEquals(5.9604645e-8f, GgufQuant.fp16ToFp32(0x0001)); // smallest subnormal
        assertEquals(65504.0f, GgufQuant.fp16ToFp32(0x7BFF));      // largest finite
        assertEquals(Float.POSITIVE_INFINITY, GgufQuant.fp16ToFp32(0x7C00));
        assertEquals(Float.NEGATIVE_INFINITY, GgufQuant.fp16ToFp32(0xFC00));
        assertTrue(Float.isNaN(GgufQuant.fp16ToFp32(0x7E00)));
    }

    @Test
    void rowSizeBytes() {
        assertEquals(2000L, GgufQuant.rowSizeBytes(GgufQuant.GGML_TYPE_F16, 1000));
        assertEquals(18L, GgufQuant.rowSizeBytes(GgufQuant.GGML_TYPE_Q4_0, 32));
        assertEquals(20L, GgufQuant.rowSizeBytes(GgufQuant.GGML_TYPE_Q4_1, 32));
        assertEquals(22L, GgufQuant.rowSizeBytes(GgufQuant.GGML_TYPE_Q5_0, 32));
        assertEquals(24L, GgufQuant.rowSizeBytes(GgufQuant.GGML_TYPE_Q5_1, 32));
        assertEquals(34L, GgufQuant.rowSizeBytes(GgufQuant.GGML_TYPE_Q8_0, 32));
        assertEquals(144L, GgufQuant.rowSizeBytes(GgufQuant.GGML_TYPE_Q4_K, 256));
        assertEquals(176L, GgufQuant.rowSizeBytes(GgufQuant.GGML_TYPE_Q5_K, 256));
        assertEquals(210L, GgufQuant.rowSizeBytes(GgufQuant.GGML_TYPE_Q6_K, 256));
        assertEquals(-1L, GgufQuant.rowSizeBytes(999, 256));
    }

    @Test
    void q4_0SingleBlock() {
        // d = 1.0 (0x3C00 LE), qs[0] = 0x21 -> nibbles 1 (low), 2 (high)
        byte[] raw = new byte[18];
        raw[0] = 0x00;
        raw[1] = 0x3C;
        raw[2] = 0x21;
        float[] out = GgufQuant.dequantize(GgufQuant.GGML_TYPE_Q4_0, raw, 32);
        assertEquals(32, out.length);
        // low nibble -> element 0 = (1 - 8) * 1.0 = -7.0
        assertEquals(-7.0f, out[0]);
        // high nibble -> element 16 = (2 - 8) * 1.0 = -6.0
        assertEquals(-6.0f, out[16]);
        // remaining zero nibbles -> (0 - 8) * 1.0 = -8.0
        assertEquals(-8.0f, out[1]);
        assertEquals(-8.0f, out[17]);
    }

    @Test
    void q8_0SingleBlock() {
        // d = 0.5 (0x3800 LE), qs[0] = 3, qs[1] = 0xFF (-1 signed)
        byte[] raw = new byte[34];
        raw[0] = 0x00;
        raw[1] = 0x38;
        raw[2] = 3;
        raw[3] = (byte) 0xFF;
        float[] out = GgufQuant.dequantize(GgufQuant.GGML_TYPE_Q8_0, raw, 32);
        assertEquals(1.5f, out[0]);   // 3 * 0.5
        assertEquals(-0.5f, out[1]);  // -1 * 0.5
        assertEquals(0.0f, out[2]);   // 0 * 0.5
    }

    @Test
    void q4_KSingleBlock() {
        // d = 1.0, dmin = 0.5, scales = [1,2,3,4, 10,11,12,13, 0,0,0,0], qs[0] = 0x21
        byte[] raw = new byte[144];
        raw[0] = 0x00;
        raw[1] = 0x3C; // d = 1.0
        raw[2] = 0x00;
        raw[3] = 0x38; // dmin = 0.5
        int[] scales = {1, 2, 3, 4, 10, 11, 12, 13, 0, 0, 0, 0};
        for (int i = 0; i < 12; i++) {
            raw[4 + i] = (byte) scales[i];
        }
        raw[16] = 0x21; // qs[0]: low nibble 1, high nibble 2
        float[] out = GgufQuant.dequantize(GgufQuant.GGML_TYPE_Q4_K, raw, 256);
        assertEquals(256, out.length);
        // is=0: sc=1, m=10 -> d1=1.0, m1=5.0; element 0 = 1.0*1 - 5.0 = -4.0
        assertEquals(-4.0f, out[0], 1e-5f);
        // is=1: sc=2, m=11 -> d2=2.0, m2=5.5; element 32 = 2.0*2 - 5.5 = -1.5
        assertEquals(-1.5f, out[32], 1e-5f);
        // remaining zero nibbles in group 0: element 1 = 1.0*0 - 5.0 = -5.0
        assertEquals(-5.0f, out[1], 1e-5f);
        assertEquals(-5.5f, out[33], 1e-5f);
        // is=2 (sc=3, m=12): d1=3.0, m1=6.0 -> element 64 = -6.0
        assertEquals(-6.0f, out[64], 1e-5f);
        // is=3 (sc=4, m=13): d2=4.0, m2=6.5 -> element 96 = -6.5
        assertEquals(-6.5f, out[96], 1e-5f);
        // is=4..7 decode to sc=0,m=0 -> elements 128..255 are zero
        assertEquals(0.0f, out[128], 1e-5f);
        assertEquals(0.0f, out[255], 1e-5f);
    }

    @Test
    void q5_KSingleBlock() {
        // Same layout as Q4_K but with qh (high bits) all zero, so values equal Q4_K.
        byte[] raw = new byte[176];
        raw[0] = 0x00;
        raw[1] = 0x3C;
        raw[2] = 0x00;
        raw[3] = 0x38;
        int[] scales = {1, 2, 3, 4, 10, 11, 12, 13, 0, 0, 0, 0};
        for (int i = 0; i < 12; i++) {
            raw[4 + i] = (byte) scales[i];
        }
        // qh[32] at offset 16..47 all zero; qs[128] at offset 48..175
        raw[48] = 0x21;
        float[] out = GgufQuant.dequantize(GgufQuant.GGML_TYPE_Q5_K, raw, 256);
        assertEquals(-4.0f, out[0], 1e-5f);
        assertEquals(-1.5f, out[32], 1e-5f);
        assertEquals(-5.0f, out[1], 1e-5f);
        assertEquals(0.0f, out[128], 1e-5f);
        assertEquals(0.0f, out[255], 1e-5f);
    }

    @Test
    void q6_KSingleBlock() {
        // ql[0] = 0x21, qh all zero, scales all = 2, d = 1.0
        byte[] raw = new byte[210];
        raw[0] = 0x21;                 // ql[0]: low nibble 1, high nibble 2
        for (int i = 0; i < 16; i++) {
            raw[128 + 64 + i] = 2;     // scales[0..15] = 2 (int8)
        }
        raw[208] = 0x00;
        raw[209] = 0x3C;               // d = 1.0
        float[] out = GgufQuant.dequantize(GgufQuant.GGML_TYPE_Q6_K, raw, 256);
        assertEquals(256, out.length);
        // half=0, l=0 (is=0): q1 = 1 - 32 = -31 -> out[0] = 1.0*2*(-31) = -62
        assertEquals(-62.0f, out[0], 1e-5f);
        // q3 = 2 - 32 = -30 -> out[64] = 1.0*2*(-30) = -60
        assertEquals(-60.0f, out[64], 1e-5f);
        // q2 = 0 - 32 = -32 -> out[32] = -64
        assertEquals(-64.0f, out[32], 1e-5f);
        // q4 = 0 - 32 = -32 -> out[96] = -64
        assertEquals(-64.0f, out[96], 1e-5f);
        // l=1 (is=0): ql[1]=0 -> all -32 -> out[1] = -64
        assertEquals(-64.0f, out[1], 1e-5f);
    }
}
