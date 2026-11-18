package ai.enr.loader.gguf;

import ai.enr.core.io.EndianIO;
import ai.enr.core.tensor.DType;
import ai.enr.loader.api.Architecture;
import ai.enr.loader.api.LoadedModel;
import ai.enr.loader.api.LoadException;
import ai.enr.loader.api.ModelFormat;
import ai.enr.loader.api.TensorEntry;
import ai.enr.loader.api.TensorStores;
import ai.enr.loader.api.TokenizerInfo;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * GGUF v2/v3 reader for Darwin Forge exported models (llama architecture family).
 *
 * Tensor layout conventions (empirically verified against llama.cpp execution
 * of the real evo.gguf / tiny-evo.gguf exports):
 *   - token_embd.weight [d_model, vocab]: canonical = flat reshaped to [vocab, d_model]
 *   - all other 2D weights [d0, d1]: canonical = flat reshaped to [d1, d0], transposed
 *   - 1D norm weights: as-is
 */
public final class GgufReader {

    private static final int GGML_TYPE_F32 = 0;
    private static final int GGML_TYPE_F16 = 1;

    public LoadedModel read(Path path, boolean verifyChecksums) throws LoadException {
        try (InputStream fis = Files.newInputStream(path);
             BufferedInputStream bis = new BufferedInputStream(fis, 1 << 16)) {
            LittleEndian in = new LittleEndian(bis);

            int magic = in.readInt();
            if (magic != 0x46554747) { // "GGUF" little-endian
                throw new LoadException("Not a GGUF file (bad magic): " + path);
            }
            int version = in.readInt();
            if (version < 2 || version > 3) {
                throw new LoadException("Unsupported GGUF version " + version + " (supported: 2, 3)");
            }
            long tensorCount = in.readLong();
            long kvCount = in.readLong();
            if (tensorCount < 0 || tensorCount > 1_000_000 || kvCount < 0 || kvCount > 1_000_000) {
                throw new LoadException("GGUF header counts out of range: tensors=" + tensorCount + ", kv=" + kvCount);
            }

            Map<String, Object> kv = new LinkedHashMap<>();
            for (long i = 0; i < kvCount; i++) {
                String key = in.readString();
                int type = in.readInt();
                Object value = readValue(in, type);
                kv.put(key, value);
            }

            String archFamily = kvString(kv, "general.architecture", "unknown");
            if (!"llama".equals(archFamily)) {
                throw new LoadException("Unsupported GGUF architecture family: '" + archFamily
                        + "' (supported: llama)");
            }

            Architecture architecture = parseArchitecture(kv);
            TokenizerInfo tokenizer = parseTokenizer(kv);

            record TensorInfo(String name, long[] dims, int ggmlType, long offset) {
            }
            List<TensorInfo> infos = new ArrayList<>();
            for (long i = 0; i < tensorCount; i++) {
                String name = in.readString();
                int nDims = in.readInt();
                if (nDims < 1 || nDims > 8) {
                    throw new LoadException("GGUF tensor " + name + " has invalid dim count " + nDims);
                }
                long[] dims = new long[nDims];
                for (int d = 0; d < nDims; d++) {
                    dims[d] = in.readLong();
                }
                int ggmlType = in.readInt();
                long offset = in.readLong();
                infos.add(new TensorInfo(name, dims, ggmlType, offset));
            }

            long alignment = kvLong(kv, "general.alignment", 32);
            long tensorDataSectionStart = alignUp(in.position(), alignment);

            List<TensorEntry> tensors = new ArrayList<>();
            for (int i = 0; i < infos.size(); i++) {
                TensorInfo info = infos.get(i);
                long elementCount = 1;
                for (long d : info.dims) {
                    elementCount *= d;
                }
                DType dtype;
                if (info.ggmlType == GGML_TYPE_F32) {
                    dtype = DType.F32;
                } else if (info.ggmlType == GGML_TYPE_F16) {
                    dtype = DType.F16;
                } else {
                    throw new LoadException("Unsupported GGML tensor type " + info.ggmlType
                            + " for tensor " + info.name + " (supported: F32, F16)");
                }
                long dataOffset = tensorDataSectionStart + info.offset;
                if (dataOffset < 0 || dataOffset >= Files.size(path)) {
                    throw new LoadException("GGUF tensor " + info.name + " has invalid data offset " + dataOffset);
                }

                String canonicalName = info.name;
                long[] canonicalShape = canonicalShape(info.name, info.dims);
                boolean transpose = needsTranspose(info.name, info.dims);

                TensorStores.DTypeLe storeDtype = dtype == DType.F32
                        ? TensorStores.DTypeLe.F32
                        : null;
                if (storeDtype == null) {
                    throw new LoadException("Tensor " + info.name + " uses unsupported dtype " + dtype
                            + " for memory-mapped loading");
                }
                var store = TensorStores.mmapLittleEndian(path, dataOffset, elementCount, storeDtype);
                if (transpose) {
                    store = TensorStores.transpose(store, info.dims[1], info.dims[0]);
                }
                TensorEntry entry = new TensorEntry(i, canonicalName, dtype, canonicalShape, 0,
                        dataOffset, elementCount * dtype.elementBytes(), elementCount * dtype.elementBytes(),
                        0, "", store);
                tensors.add(entry);

                if (verifyChecksums && dtype == DType.F32) {
                    entry.loadFloats();
                }
            }

            architecture.validate();
            String name = kvString(kv, "general.name", fileNameOf(path));
            ModelFormat format = version == 3 ? ModelFormat.GGUF_V3 : ModelFormat.GGUF_V2;
            return new LoadedModel(format, String.valueOf(version), name, 0, "",
                    architecture, tokenizer, Map.of("gguf_version", String.valueOf(version)), tensors);
        } catch (LoadException e) {
            throw e;
        } catch (IOException e) {
            throw new LoadException("Failed to read GGUF file " + path + ": " + e.getMessage(), e);
        }
    }

    private static String fileNameOf(Path path) {
        String file = path.getFileName().toString();
        int dot = file.lastIndexOf('.');
        return dot > 0 ? file.substring(0, dot) : file;
    }

    /**
     * Returns the canonical (EVO-layout) shape for a GGUF tensor.
     * token_embd: [vocab, d_model]; everything else keeps dims order but data is transposed.
     */
    private long[] canonicalShape(String name, long[] dims) {
        if ("token_embd.weight".equals(name) && dims.length == 2) {
            return new long[]{dims[1], dims[0]};
        }
        return dims.clone();
    }

    private boolean needsTranspose(String name, long[] dims) {
        if (dims.length != 2) {
            return false;
        }
        return !"token_embd.weight".equals(name);
    }

    private Architecture parseArchitecture(Map<String, Object> kv) throws LoadException {
        int vocabSize = (int) kvLong(kv, "llama.vocab_size", 0);
        int dModel = (int) kvLong(kv, "llama.embedding_length", 0);
        int dff = (int) kvLong(kv, "llama.feed_forward_length", 0);
        int numBlocks = (int) kvLong(kv, "llama.block_count", 0);
        int numHeads = (int) kvLong(kv, "llama.attention.head_count", 0);
        int numKvHeads = (int) kvLong(kv, "llama.attention.head_count_kv", numHeads);
        int context = (int) kvLong(kv, "llama.context_length", 0);
        float normEps = (float) kvDouble(kv, "llama.attention.layer_norm_rms_epsilon", 1e-5);
        float ropeTheta = (float) kvDouble(kv, "llama.rope.freq_base", 10000.0);
        long ropeDim = kvLong(kv, "llama.rope.dimension_count", 0);

        if (vocabSize <= 0 || dModel <= 0 || numHeads <= 0) {
            throw new LoadException("GGUF metadata missing required llama.* architecture fields");
        }
        Architecture arch = new Architecture()
                .family("llama")
                .vocabSize(vocabSize)
                .dModel(dModel)
                .numHeads(numHeads)
                .numKvHeads(numKvHeads)
                .numBlocks(numBlocks)
                .dff(dff)
                .maxSeqLen(context > 0 ? context : 2048)
                .normType("rms_norm")
                .normEps(normEps)
                .activationType("swiglu")
                .positionalEncoding("rope")
                .ropeTheta(ropeTheta);
        if (ropeDim > 0) {
            arch.ropeDim((int) ropeDim);
        }
        return arch;
    }

    private TokenizerInfo parseTokenizer(Map<String, Object> kv) {
        TokenizerInfo tok = new TokenizerInfo().type("BPE");
        tok.vocabSize((int) kvLong(kv, "llama.vocab_size", 0));
        tok.bosTokenId((int) kvLong(kv, "tokenizer.ggml.bos_token_id", 1));
        tok.eosTokenId((int) kvLong(kv, "tokenizer.ggml.eos_token_id", 2));
        tok.unkTokenId((int) kvLong(kv, "tokenizer.ggml.unknown_token_id", 0));
        tok.padTokenId((int) kvLong(kv, "tokenizer.ggml.padding_token_id", 0));

        Object tokensObj = kv.get("tokenizer.ggml.tokens");
        if (tokensObj instanceof List<?> tokens) {
            for (int i = 0; i < tokens.size(); i++) {
                Object t = tokens.get(i);
                if (t instanceof String s) {
                    tok.addToken(s, i);
                }
            }
        }
        tok.ensureSpecialTokens();
        return tok;
    }

    private Object readValue(LittleEndian in, int type) throws IOException, LoadException {
        return switch (type) {
            case 0, 1 -> (long) in.readUnsignedByteOrByte(type);
            case 2, 3 -> (long) in.readUnsignedShortOrShort(type);
            case 4, 5 -> (long) in.readUnsignedIntOrInt(type);
            case 6 -> in.readFloat();
            case 7 -> in.readBoolean();
            case 8 -> in.readString();
            case 9 -> {
                int elementType = in.readInt();
                long length = in.readLong();
                if (length < 0 || length > 1_000_000) {
                    throw new LoadException("GGUF array length out of range: " + length);
                }
                List<Object> array = new ArrayList<>((int) length);
                for (long i = 0; i < length; i++) {
                    array.add(readValue(in, elementType));
                }
                yield array;
            }
            case 10, 11 -> in.readLong();
            case 12 -> in.readDouble();
            default -> throw new LoadException("Unsupported GGUF metadata value type " + type);
        };
    }

    private String kvString(Map<String, Object> kv, String key, String defaultValue) {
        Object v = kv.get(key);
        return v instanceof String s ? s : defaultValue;
    }

    private long kvLong(Map<String, Object> kv, String key, long defaultValue) {
        Object v = kv.get(key);
        if (v instanceof Number n) {
            return n.longValue();
        }
        if (v instanceof Boolean b) {
            return b ? 1 : 0;
        }
        return defaultValue;
    }

    private double kvDouble(Map<String, Object> kv, String key, double defaultValue) {
        Object v = kv.get(key);
        if (v instanceof Number n) {
            return n.doubleValue();
        }
        return defaultValue;
    }

    private static long alignUp(long value, long alignment) {
        if (alignment <= 0) {
            return value;
        }
        return (value + alignment - 1) / alignment * alignment;
    }

    private static final class LittleEndian {
        private final InputStream in;
        private long position;

        LittleEndian(InputStream in) {
            this.in = in;
        }

        long position() {
            return position;
        }

        int readInt() throws IOException {
            byte[] b = in.readNBytes(4);
            if (b.length != 4) {
                throw new IOException("Unexpected EOF");
            }
            position += 4;
            return EndianIO.readLittleEndianInt(b, 0);
        }

        long readLong() throws IOException {
            byte[] b = in.readNBytes(8);
            if (b.length != 8) {
                throw new IOException("Unexpected EOF");
            }
            position += 8;
            return EndianIO.readLittleEndianLong(b, 0);
        }

        float readFloat() throws IOException {
            return Float.intBitsToFloat(readInt());
        }

        double readDouble() throws IOException {
            return Double.longBitsToDouble(readLong());
        }

        boolean readBoolean() throws IOException {
            int v = in.read();
            if (v < 0) {
                throw new IOException("Unexpected EOF");
            }
            position++;
            return v != 0;
        }

        String readString() throws IOException {
            long len = readLong();
            if (len < 0 || len > (1 << 30)) {
                throw new IOException("Invalid GGUF string length: " + len);
            }
            byte[] bytes = in.readNBytes((int) len);
            if (bytes.length != len) {
                throw new IOException("Unexpected EOF in GGUF string");
            }
            position += len;
            return new String(bytes, StandardCharsets.UTF_8);
        }

        long readUnsignedByteOrByte(int type) throws IOException {
            int v = in.read();
            if (v < 0) {
                throw new IOException("Unexpected EOF");
            }
            position++;
            return type == 0 ? v : (byte) v;
        }

        long readUnsignedShortOrShort(int type) throws IOException {
            byte[] b = in.readNBytes(2);
            if (b.length != 2) {
                throw new IOException("Unexpected EOF");
            }
            position += 2;
            int v = EndianIO.readLittleEndianInt(new byte[]{b[0], b[1], 0, 0}, 0);
            return type == 2 ? v : (short) v;
        }

        long readUnsignedIntOrInt(int type) throws IOException {
            int v = readInt();
            return type == 4 ? Integer.toUnsignedLong(v) : v;
        }
    }
}
