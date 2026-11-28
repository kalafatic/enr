package ai.enr.core.gguf;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * GGML/GGUF quantized tensor dequantization.
 *
 * <p>Decodes the block-wise quantization formats used by llama.cpp GGUF exports
 * into plain IEEE-754 {@code float[]} values, so the existing F32 inference path
 * can execute quantized Ollama/Forge models unchanged. Layouts and formulas are
 * a faithful port of the reference implementations in llama.cpp
 * {@code ggml/src/ggml-common.h} and {@code ggml/src/ggml-quants.c}.</p>
 *
 * <p>Supported types (GGML type codes): F16 (1), Q4_0 (2), Q4_1 (3), Q5_0 (6),
 * Q5_1 (7), Q8_0 (8), Q4_K (12), Q5_K (13), Q6_K (14). F32 (0) is handled by
 * the memory-mapped F32 path in the GGUF reader, not here.</p>
 */
public final class GgufQuant {

    public static final int GGML_TYPE_F32 = 0;
    public static final int GGML_TYPE_F16 = 1;
    public static final int GGML_TYPE_Q4_0 = 2;
    public static final int GGML_TYPE_Q4_1 = 3;
    public static final int GGML_TYPE_Q5_0 = 6;
    public static final int GGML_TYPE_Q5_1 = 7;
    public static final int GGML_TYPE_Q8_0 = 8;
    public static final int GGML_TYPE_Q4_K = 12;
    public static final int GGML_TYPE_Q5_K = 13;
    public static final int GGML_TYPE_Q6_K = 14;

    private static final int QK = 32;   // simple-quant block size
    private static final int QK_K = 256; // K-quant super-block size

    private GgufQuant() {
    }

    /**
     * Returns whether the given GGML type code is dequantizable by this class.
     */
    public static boolean isSupported(int ggmlType) {
        return ggmlType == GGML_TYPE_F16 || rowSizeBytes(ggmlType, 32) >= 0;
    }

    /**
     * Returns the on-disk byte size of a tensor of {@code elementCount} elements
     * stored with the given GGML type, or -1 if the type is unsupported.
     */
    public static long rowSizeBytes(int ggmlType, long elementCount) {
        return switch (ggmlType) {
            case GGML_TYPE_F16 -> elementCount * 2L;
            case GGML_TYPE_Q4_0 -> elementCount / QK * 18L;
            case GGML_TYPE_Q4_1 -> elementCount / QK * 20L;
            case GGML_TYPE_Q5_0 -> elementCount / QK * 22L;
            case GGML_TYPE_Q5_1 -> elementCount / QK * 24L;
            case GGML_TYPE_Q8_0 -> elementCount / QK * 34L;
            case GGML_TYPE_Q4_K -> elementCount / QK_K * 144L;
            case GGML_TYPE_Q5_K -> elementCount / QK_K * 176L;
            case GGML_TYPE_Q6_K -> elementCount / QK_K * 210L;
            default -> -1L;
        };
    }

    /**
     * Reads {@code elementCount} elements of the given GGML type from
     * {@code file} starting at {@code dataOffset}, dequantizing them to F32.
     */
    public static float[] readAndDequantize(Path file, long dataOffset, int ggmlType, long elementCount) {
        byte[] raw = readRaw(file, dataOffset, ggmlType, elementCount);
        return dequantize(ggmlType, raw, elementCount);
    }

    /**
     * Reads the raw quantized bytes for a tensor of the given type, without
     * dequantizing. Used by the quantized-store path to keep weights in their
     * original packed form.
     */
    public static byte[] readRaw(Path file, long dataOffset, int ggmlType, long elementCount) {
        long rawSize = rowSizeBytes(ggmlType, elementCount);
        if (rawSize < 0) {
            throw new IllegalStateException("Unsupported GGML quantized type " + ggmlType);
        }
        return readBytes(file, dataOffset, rawSize);
    }

    private static byte[] readBytes(Path file, long dataOffset, long length) {
        if (length > Integer.MAX_VALUE - 8) {
            throw new IllegalStateException("GGUF tensor payload too large to read: " + length + " bytes");
        }
        byte[] bytes = new byte[(int) length];
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
            ByteBuffer buffer = ByteBuffer.wrap(bytes);
            long pos = dataOffset;
            while (buffer.hasRemaining()) {
                int read = channel.read(buffer, pos);
                if (read < 0) {
                    break;
                }
                pos += read;
            }
            if (buffer.hasRemaining()) {
                throw new IllegalStateException("GGUF tensor data truncated at offset " + dataOffset
                        + " (needed " + length + " bytes)");
            }
            return bytes;
        } catch (IOException e) {
            throw new IllegalStateException("Failed to read GGUF tensor data from " + file + ": " + e.getMessage(), e);
        }
    }

    /**
     * Dequantizes {@code raw} (already read from disk) into {@code elementCount}
     * F32 values.
     */
    public static float[] dequantize(int ggmlType, byte[] raw, long elementCount) {
        if (elementCount > Integer.MAX_VALUE - 8) {
            throw new IllegalStateException("GGUF tensor too large for float[] storage: " + elementCount);
        }
        int n = (int) elementCount;
        return switch (ggmlType) {
            case GGML_TYPE_F16 -> dequantizeF16(raw, n);
            case GGML_TYPE_Q4_0 -> dequantizeQ4_0(raw, n);
            case GGML_TYPE_Q4_1 -> dequantizeQ4_1(raw, n);
            case GGML_TYPE_Q5_0 -> dequantizeQ5_0(raw, n);
            case GGML_TYPE_Q5_1 -> dequantizeQ5_1(raw, n);
            case GGML_TYPE_Q8_0 -> dequantizeQ8_0(raw, n);
            case GGML_TYPE_Q4_K -> dequantizeQ4_K(raw, n);
            case GGML_TYPE_Q5_K -> dequantizeQ5_K(raw, n);
            case GGML_TYPE_Q6_K -> dequantizeQ6_K(raw, n);
            default -> throw new IllegalStateException("Unsupported GGML quantized type " + ggmlType);
        };
    }

    /**
     * Block size in elements for a quantized type. F16 is treated as 32-element
     * "blocks" (64 raw bytes) for uniform tiling in the quantized matmul.
     */
    public static int blockSize(int ggmlType) {
        return switch (ggmlType) {
            case GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
                    GGML_TYPE_Q8_0 -> QK;
            case GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K -> QK_K;
            default -> throw new IllegalStateException("Unsupported GGML quantized type " + ggmlType);
        };
    }

    /** Raw byte size of one block for a quantized type. */
    public static int blockBytes(int ggmlType) {
        return switch (ggmlType) {
            case GGML_TYPE_F16 -> QK * 2;
            case GGML_TYPE_Q4_0 -> 18;
            case GGML_TYPE_Q4_1 -> 20;
            case GGML_TYPE_Q5_0 -> 22;
            case GGML_TYPE_Q5_1 -> 24;
            case GGML_TYPE_Q8_0 -> 34;
            case GGML_TYPE_Q4_K -> 144;
            case GGML_TYPE_Q5_K -> 176;
            case GGML_TYPE_Q6_K -> 210;
            default -> throw new IllegalStateException("Unsupported GGML quantized type " + ggmlType);
        };
    }

    /**
     * Dequantizes a single block of {@link #blockSize(int)} elements, starting at
     * {@code rawOff} in {@code raw}, writing F32 values into {@code dst} at
     * {@code dstOff}. Used by the quantized matmul to avoid materializing the
     * full weight as F32.
     */
    public static void dequantizeBlock(float[] dst, int dstOff, byte[] raw, int rawOff, int ggmlType) {
        switch (ggmlType) {
            case GGML_TYPE_F16 -> dequantizeBlockF16(dst, dstOff, raw, rawOff);
            case GGML_TYPE_Q4_0 -> dequantizeBlockQ4_0(dst, dstOff, raw, rawOff);
            case GGML_TYPE_Q4_1 -> dequantizeBlockQ4_1(dst, dstOff, raw, rawOff);
            case GGML_TYPE_Q5_0 -> dequantizeBlockQ5_0(dst, dstOff, raw, rawOff);
            case GGML_TYPE_Q5_1 -> dequantizeBlockQ5_1(dst, dstOff, raw, rawOff);
            case GGML_TYPE_Q8_0 -> dequantizeBlockQ8_0(dst, dstOff, raw, rawOff);
            case GGML_TYPE_Q4_K -> dequantizeBlockQ4_K(dst, dstOff, raw, rawOff);
            case GGML_TYPE_Q5_K -> dequantizeBlockQ5_K(dst, dstOff, raw, rawOff);
            case GGML_TYPE_Q6_K -> dequantizeBlockQ6_K(dst, dstOff, raw, rawOff);
            default -> throw new IllegalStateException("Unsupported GGML quantized type " + ggmlType);
        }
    }

    private static void dequantizeBlockF16(float[] dst, int dstOff, byte[] raw, int off) {
        for (int j = 0; j < QK; j++) {
            dst[dstOff + j] = fp16ToFp32(readU16(raw, off + j * 2));
        }
    }

    private static void dequantizeBlockQ4_0(float[] dst, int dstOff, byte[] raw, int off) {
        float d = fp16ToFp32(readU16(raw, off));
        for (int j = 0; j < QK / 2; j++) {
            int b = raw[off + 2 + j] & 0xFF;
            dst[dstOff + j] = ((b & 0x0F) - 8) * d;
            dst[dstOff + j + QK / 2] = ((b >> 4) - 8) * d;
        }
    }

    private static void dequantizeBlockQ4_1(float[] dst, int dstOff, byte[] raw, int off) {
        float d = fp16ToFp32(readU16(raw, off));
        float m = fp16ToFp32(readU16(raw, off + 2));
        for (int j = 0; j < QK / 2; j++) {
            int b = raw[off + 4 + j] & 0xFF;
            dst[dstOff + j] = (b & 0x0F) * d + m;
            dst[dstOff + j + QK / 2] = (b >> 4) * d + m;
        }
    }

    private static void dequantizeBlockQ5_0(float[] dst, int dstOff, byte[] raw, int off) {
        float d = fp16ToFp32(readU16(raw, off));
        int qh = readU32(raw, off + 2);
        for (int j = 0; j < QK / 2; j++) {
            int b = raw[off + 6 + j] & 0xFF;
            int x0 = ((b & 0x0F) | (((qh >> j) & 1) << 4)) - 16;
            int x1 = ((b >> 4) | (((qh >> (j + 16)) & 1) << 4)) - 16;
            dst[dstOff + j] = x0 * d;
            dst[dstOff + j + QK / 2] = x1 * d;
        }
    }

    private static void dequantizeBlockQ5_1(float[] dst, int dstOff, byte[] raw, int off) {
        float d = fp16ToFp32(readU16(raw, off));
        float m = fp16ToFp32(readU16(raw, off + 2));
        int qh = readU32(raw, off + 4);
        for (int j = 0; j < QK / 2; j++) {
            int b = raw[off + 8 + j] & 0xFF;
            int x0 = (b & 0x0F) | (((qh >> j) & 1) << 4);
            int x1 = (b >> 4) | (((qh >> (j + 16)) & 1) << 4);
            dst[dstOff + j] = x0 * d + m;
            dst[dstOff + j + QK / 2] = x1 * d + m;
        }
    }

    private static void dequantizeBlockQ8_0(float[] dst, int dstOff, byte[] raw, int off) {
        float d = fp16ToFp32(readU16(raw, off));
        for (int j = 0; j < QK; j++) {
            dst[dstOff + j] = (byte) raw[off + 2 + j] * d;
        }
    }

    private static void dequantizeBlockQ4_K(float[] dst, int dstOff, byte[] raw, int off) {
        float d = fp16ToFp32(readU16(raw, off));
        float dmin = fp16ToFp32(readU16(raw, off + 2));
        int sBase = off + 4;
        int qBase = off + 16;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            int sm0 = scaleMinK4(is, raw, sBase);
            float d1 = d * (sm0 & 0xFF);
            float m1 = dmin * ((sm0 >> 8) & 0xFF);
            int sm1 = scaleMinK4(is + 1, raw, sBase);
            float d2 = d * (sm1 & 0xFF);
            float m2 = dmin * ((sm1 >> 8) & 0xFF);
            for (int l = 0; l < 32; l++) {
                int b = raw[qBase + l] & 0xFF;
                dst[dstOff + j + l] = d1 * (b & 0xF) - m1;
                dst[dstOff + j + l + 32] = d2 * (b >> 4) - m2;
            }
            qBase += 32;
            is += 2;
        }
    }

    private static void dequantizeBlockQ5_K(float[] dst, int dstOff, byte[] raw, int off) {
        float d = fp16ToFp32(readU16(raw, off));
        float dmin = fp16ToFp32(readU16(raw, off + 2));
        int sBase = off + 4;
        int qhBase = off + 16;
        int qlBase = off + 48;
        int is = 0;
        int u1 = 1;
        int u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            int sm0 = scaleMinK4(is, raw, sBase);
            float d1 = d * (sm0 & 0xFF);
            float m1 = dmin * ((sm0 >> 8) & 0xFF);
            int sm1 = scaleMinK4(is + 1, raw, sBase);
            float d2 = d * (sm1 & 0xFF);
            float m2 = dmin * ((sm1 >> 8) & 0xFF);
            for (int l = 0; l < 32; l++) {
                int ql = raw[qlBase + l] & 0xFF;
                int qh = raw[qhBase + l] & 0xFF;
                int v0 = (ql & 0xF) + ((qh & u1) != 0 ? 16 : 0);
                int v1 = (ql >> 4) + ((qh & u2) != 0 ? 16 : 0);
                dst[dstOff + j + l] = d1 * v0 - m1;
                dst[dstOff + j + l + 32] = d2 * v1 - m2;
            }
            qlBase += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    private static void dequantizeBlockQ6_K(float[] dst, int dstOff, byte[] raw, int off) {
        int qlBase = off;
        int qhBase = off + 128;
        int scBase = off + 192;
        float d = fp16ToFp32(readU16(raw, off + 208));
        for (int half = 0; half < 2; half++) {
            int ql = qlBase + half * 64;
            int qh = qhBase + half * 32;
            int sc = scBase + half * 8;
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int qb = raw[qh + l] & 0xFF;
                int qll = raw[ql + l] & 0xFF;
                int qll32 = raw[ql + l + 32] & 0xFF;
                int q1 = ((qll & 0xF) | (((qb >> 0) & 3) << 4)) - 32;
                int q2 = ((qll32 & 0xF) | (((qb >> 2) & 3) << 4)) - 32;
                int q3 = ((qll >> 4) | (((qb >> 4) & 3) << 4)) - 32;
                int q4 = ((qll32 >> 4) | (((qb >> 6) & 3) << 4)) - 32;
                int base = half * 128 + l;
                dst[dstOff + base] = d * (byte) raw[sc + is] * q1;
                dst[dstOff + base + 32] = d * (byte) raw[sc + is + 2] * q2;
                dst[dstOff + base + 64] = d * (byte) raw[sc + is + 4] * q3;
                dst[dstOff + base + 96] = d * (byte) raw[sc + is + 6] * q4;
            }
        }
    }

    private static float[] dequantizeF16(byte[] raw, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = fp16ToFp32(readU16(raw, i * 2));
        }
        return out;
    }

    private static float[] dequantizeQ4_0(byte[] raw, int n) {
        float[] out = new float[n];
        int off = 0;
        for (int i = 0; i < n; i += QK) {
            float d = fp16ToFp32(readU16(raw, off));
            off += 2;
            for (int j = 0; j < QK / 2; j++) {
                int b = raw[off + j] & 0xFF;
                out[i + j] = (b & 0x0F) - 8;
                out[i + j + QK / 2] = (b >> 4) - 8;
            }
            for (int j = 0; j < QK; j++) {
                out[i + j] *= d;
            }
            off += QK / 2;
        }
        return out;
    }

    private static float[] dequantizeQ4_1(byte[] raw, int n) {
        float[] out = new float[n];
        int off = 0;
        for (int i = 0; i < n; i += QK) {
            float d = fp16ToFp32(readU16(raw, off));
            float m = fp16ToFp32(readU16(raw, off + 2));
            off += 4;
            for (int j = 0; j < QK / 2; j++) {
                int b = raw[off + j] & 0xFF;
                out[i + j] = (b & 0x0F) * d + m;
                out[i + j + QK / 2] = (b >> 4) * d + m;
            }
            off += QK / 2;
        }
        return out;
    }

    private static float[] dequantizeQ5_0(byte[] raw, int n) {
        float[] out = new float[n];
        int off = 0;
        for (int i = 0; i < n; i += QK) {
            float d = fp16ToFp32(readU16(raw, off));
            int qh = readU32(raw, off + 2);
            off += 6;
            for (int j = 0; j < QK / 2; j++) {
                int b = raw[off + j] & 0xFF;
                int x0 = ((b & 0x0F) | (((qh >> j) & 1) << 4)) - 16;
                int x1 = ((b >> 4) | (((qh >> (j + 16)) & 1) << 4)) - 16;
                out[i + j] = x0 * d;
                out[i + j + QK / 2] = x1 * d;
            }
            off += QK / 2;
        }
        return out;
    }

    private static float[] dequantizeQ5_1(byte[] raw, int n) {
        float[] out = new float[n];
        int off = 0;
        for (int i = 0; i < n; i += QK) {
            float d = fp16ToFp32(readU16(raw, off));
            float m = fp16ToFp32(readU16(raw, off + 2));
            int qh = readU32(raw, off + 4);
            off += 8;
            for (int j = 0; j < QK / 2; j++) {
                int b = raw[off + j] & 0xFF;
                int x0 = (b & 0x0F) | (((qh >> j) & 1) << 4);
                int x1 = (b >> 4) | (((qh >> (j + 16)) & 1) << 4);
                out[i + j] = x0 * d + m;
                out[i + j + QK / 2] = x1 * d + m;
            }
            off += QK / 2;
        }
        return out;
    }

    private static float[] dequantizeQ8_0(byte[] raw, int n) {
        float[] out = new float[n];
        int off = 0;
        for (int i = 0; i < n; i += QK) {
            float d = fp16ToFp32(readU16(raw, off));
            off += 2;
            for (int j = 0; j < QK; j++) {
                out[i + j] = (byte) raw[off + j] * d;
            }
            off += QK;
        }
        return out;
    }

    private static float[] dequantizeQ4_K(byte[] raw, int n) {
        float[] out = new float[n];
        int off = 0;
        for (int i = 0; i < n; i += QK_K) {
            float d = fp16ToFp32(readU16(raw, off));
            float dmin = fp16ToFp32(readU16(raw, off + 2));
            int sBase = off + 4;
            int qBase = off + 4 + 12;
            int is = 0;
            for (int j = 0; j < QK_K; j += 64) {
                int sm0 = scaleMinK4(is, raw, sBase);
                float d1 = d * (sm0 & 0xFF);
                float m1 = dmin * ((sm0 >> 8) & 0xFF);
                int sm1 = scaleMinK4(is + 1, raw, sBase);
                float d2 = d * (sm1 & 0xFF);
                float m2 = dmin * ((sm1 >> 8) & 0xFF);
                for (int l = 0; l < 32; l++) {
                    int b = raw[qBase + l] & 0xFF;
                    out[i + j + l] = d1 * (b & 0xF) - m1;
                    out[i + j + l + 32] = d2 * (b >> 4) - m2;
                }
                qBase += 32;
                is += 2;
            }
            off += 4 + 12 + QK_K / 2;
        }
        return out;
    }

    private static float[] dequantizeQ5_K(byte[] raw, int n) {
        float[] out = new float[n];
        int off = 0;
        for (int i = 0; i < n; i += QK_K) {
            float d = fp16ToFp32(readU16(raw, off));
            float dmin = fp16ToFp32(readU16(raw, off + 2));
            int sBase = off + 4;
            int qhBase = off + 4 + 12;
            int qlBase = off + 4 + 12 + QK_K / 8;
            int is = 0;
            int u1 = 1;
            int u2 = 2;
            for (int j = 0; j < QK_K; j += 64) {
                int sm0 = scaleMinK4(is, raw, sBase);
                float d1 = d * (sm0 & 0xFF);
                float m1 = dmin * ((sm0 >> 8) & 0xFF);
                int sm1 = scaleMinK4(is + 1, raw, sBase);
                float d2 = d * (sm1 & 0xFF);
                float m2 = dmin * ((sm1 >> 8) & 0xFF);
                for (int l = 0; l < 32; l++) {
                    int ql = raw[qlBase + l] & 0xFF;
                    int qh = raw[qhBase + l] & 0xFF;
                    int v0 = (ql & 0xF) + ((qh & u1) != 0 ? 16 : 0);
                    int v1 = (ql >> 4) + ((qh & u2) != 0 ? 16 : 0);
                    out[i + j + l] = d1 * v0 - m1;
                    out[i + j + l + 32] = d2 * v1 - m2;
                }
                qlBase += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
            off += 4 + 12 + QK_K / 8 + QK_K / 2;
        }
        return out;
    }

    private static float[] dequantizeQ6_K(byte[] raw, int n) {
        float[] out = new float[n];
        int off = 0;
        for (int i = 0; i < n; i += QK_K) {
            int qlBase = off;
            int qhBase = off + QK_K / 2;
            int scBase = off + QK_K / 2 + QK_K / 4;
            float d = fp16ToFp32(readU16(raw, off + QK_K / 2 + QK_K / 4 + QK_K / 16));
            for (int half = 0; half < 2; half++) {
                int ql = qlBase + half * 64;
                int qh = qhBase + half * 32;
                int sc = scBase + half * 8;
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    int qb = raw[qh + l] & 0xFF;
                    int qll = raw[ql + l] & 0xFF;
                    int qll32 = raw[ql + l + 32] & 0xFF;
                    int q1 = ((qll & 0xF) | (((qb >> 0) & 3) << 4)) - 32;
                    int q2 = ((qll32 & 0xF) | (((qb >> 2) & 3) << 4)) - 32;
                    int q3 = ((qll >> 4) | (((qb >> 4) & 3) << 4)) - 32;
                    int q4 = ((qll32 >> 4) | (((qb >> 6) & 3) << 4)) - 32;
                    out[i + half * 128 + l] = d * (byte) raw[sc + is] * q1;
                    out[i + half * 128 + l + 32] = d * (byte) raw[sc + is + 2] * q2;
                    out[i + half * 128 + l + 64] = d * (byte) raw[sc + is + 4] * q3;
                    out[i + half * 128 + l + 96] = d * (byte) raw[sc + is + 6] * q4;
                }
            }
            off += QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
        }
        return out;
    }

    /**
     * Decodes one of the eight 6-bit scale/min pairs packed into the 12-byte
     * {@code scales} array of a Q4_K/Q5_K super-block. Returns {@code (m << 8) | d}.
     */
    public static int scaleMinK4(int j, byte[] raw, int base) {
        int d;
        int m;
        if (j < 4) {
            d = raw[base + j] & 63;
            m = raw[base + j + 4] & 63;
        } else {
            int qj4 = raw[base + j + 4] & 0xFF;
            d = (qj4 & 0xF) | (((raw[base + j - 4] & 0xFF) >> 6) << 4);
            m = (qj4 >> 4) | (((raw[base + j] & 0xFF) >> 6) << 4);
        }
        return (m << 8) | d;
    }

    /**
     * IEEE 754 binary16 to binary32 conversion, ported exactly from
     * llama.cpp {@code ggml_half_to_bits} (including subnormals, Inf and NaN).
     */
    public static float fp16ToFp32(int h) {
        int sign = (h & 0x8000) << 16;
        int exponent = (h >> 10) & 0x1F;
        int mantissa = h & 0x3FF;
        int bits;
        if (exponent == 0) {
            if (mantissa == 0) {
                bits = sign;
            } else {
                // Subnormal: value = mantissa * 2^-24. Normalise the mantissa so the
                // leading 1 sits at bit 10, tracking the exponent accordingly.
                int e = 1;
                while ((mantissa & 0x400) == 0) {
                    mantissa <<= 1;
                    e--;
                }
                mantissa &= 0x3FF;
                bits = sign | ((e + 127 - 15) << 23) | (mantissa << 13);
            }
        } else if (exponent == 31) {
            bits = sign | 0x7F800000 | (mantissa << 13);
        } else {
            bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
        return Float.intBitsToFloat(bits);
    }

    private static int readU16(byte[] raw, int off) {
        return (raw[off] & 0xFF) | ((raw[off + 1] & 0xFF) << 8);
    }

    private static int readU32(byte[] raw, int off) {
        return (raw[off] & 0xFF)
                | ((raw[off + 1] & 0xFF) << 8)
                | ((raw[off + 2] & 0xFF) << 16)
                | ((raw[off + 3] & 0xFF) << 24);
    }
}
