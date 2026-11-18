package ai.enr.loader.evo;

import ai.enr.core.json.Json;
import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonObject;
import ai.enr.core.io.Hashing;
import ai.enr.core.tensor.DType;
import ai.enr.loader.api.Architecture;
import ai.enr.loader.api.LoadedModel;
import ai.enr.loader.api.LoadException;
import ai.enr.loader.api.ModelFormat;
import ai.enr.loader.api.TensorEntry;
import ai.enr.loader.api.TensorStores;
import ai.enr.loader.api.TokenizerInfo;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.EOFException;
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
 * Independent reader for the EVO Native Model Protocol (.evo) v2 container
 * (magic EVO_NAT2). Implemented from the audited format specification;
 * no dependency on EVO or Darwin Forge code.
 */
public final class EvoModelReader {

    public static final String MAGIC_V2 = "EVO_NAT2";

    public LoadedModel read(Path path, LoadOptions options) throws LoadException {
        try (InputStream fis = Files.newInputStream(path);
             BufferedInputStream bis = new BufferedInputStream(fis, 1 << 16);
             DataInputStream dis = new DataInputStream(bis)) {

            byte[] magicBytes = new byte[8];
            dis.mark(16);
            int read = dis.read(magicBytes);
            if (read < 8) {
                throw new LoadException("EVO_PROTOCOL_ERROR: file too small or truncated: " + path);
            }
            String magic = new String(magicBytes, StandardCharsets.US_ASCII);
            if (MAGIC_V2.equals(magic)) {
                return readV2(dis, options, path);
            }
            bis.reset();
            return new EvoV1Reader().readV1(dis, path);
        } catch (LoadException e) {
            throw e;
        } catch (IOException e) {
            throw new LoadException("EVO_PROTOCOL_ERROR: failed to read model file " + path + ": " + e.getMessage(), e);
        }
    }

    private LoadedModel readV2(DataInputStream dis, LoadOptions options, Path path) throws LoadException, IOException {
        short major = dis.readShort();
        short minor = dis.readShort();
        short protocol = dis.readShort();
        short flags = dis.readShort();
        int headerSize = dis.readInt();
        dis.readInt(); // sectionCount (informational; actual sections parsed sequentially)
        long createdAt = dis.readLong();
        byte[] hashBytes = new byte[32];
        dis.readFully(hashBytes);
        String storedHash = new String(hashBytes, StandardCharsets.UTF_8).trim();

        if (major != 2) {
            throw new LoadException("Unsupported .evo format major version " + major + "." + minor
                    + " (supported: 2.x and legacy 1.x)");
        }
        if (headerSize != 64) {
            throw new LoadException("EVO_PROTOCOL_ERROR: unexpected header size " + headerSize + " (expected 64)");
        }

        String modelName = readString32(dis);
        String archJson = readString32(dis);
        String tokJson = readString32(dis);
        String vocabJson = readString32(dis);
        String manifestJson = readString32(dis);
        String metaJson = readString32(dis);

        Architecture architecture = parseArchitecture(archJson);
        TokenizerInfo tokenizer = parseTokenizer(tokJson);
        JsonObject vocabObj = Json.parseObject(vocabJson);
        for (Map.Entry<String, ai.enr.core.json.JsonValue> entry : vocabObj.entries()) {
            ai.enr.core.json.JsonValue value = entry.getValue();
            if (value instanceof ai.enr.core.json.JsonNumber n) {
                tokenizer.addToken(entry.getKey(), (int) n.asLong());
            }
        }
        tokenizer.ensureSpecialTokens();

        Map<String, String> metadata = new LinkedHashMap<>();
        JsonObject metaObj = Json.parseObject(metaJson);
        for (Map.Entry<String, ai.enr.core.json.JsonValue> entry : metaObj.entries()) {
            ai.enr.core.json.JsonValue value = entry.getValue();
            if (value instanceof ai.enr.core.json.JsonString s) {
                metadata.put(entry.getKey(), s.value());
            }
        }

        List<ManifestEntry> manifest = parseManifest(manifestJson);

        int tensorCount = dis.readInt();
        if (tensorCount != manifest.size()) {
            throw new LoadException("EVO_PROTOCOL_ERROR: tensor payload count " + tensorCount
                    + " does not match manifest size " + manifest.size());
        }

        List<TensorEntry> tensors = new ArrayList<>();
        for (ManifestEntry entry : manifest) {
            int compLen = dis.readInt();
            if (compLen < 0) {
                throw new LoadException("EVO_PROTOCOL_ERROR: negative compressed length for tensor " + entry.name);
            }
            byte[] compBytes = new byte[compLen];
            dis.readFully(compBytes);

            if (entry.compression != 1 && entry.compression != 0) {
                throw new LoadException("Unsupported tensor compression scheme " + entry.compression
                        + " for tensor " + entry.name + " (supported: 0=NONE, 1=GZIP)");
            }

            TensorEntry tensor = new TensorEntry(
                    entry.id,
                    entry.name,
                    entry.dtype,
                    entry.shape,
                    entry.layout,
                    entry.offset,
                    entry.compressedSize,
                    entry.uncompressedSize,
                    entry.compression,
                    entry.checksum,
                    entry.compression == 1
                            ? TensorStores.gzip(compBytes, entry.uncompressedSize)
                            : TensorStores.eager(ai.enr.core.io.EndianIO.bigEndianToFloatArray(compBytes)));
            tensors.add(tensor);

            if (options.verifyChecksums()) {
                float[] data = tensor.loadFloats();
                String computed = TensorStores.checksumOfFloats(data);
                if (!entry.checksum.isEmpty() && !entry.checksum.equalsIgnoreCase(computed)) {
                    throw new LoadException("TENSOR_CHECKSUM mismatch for " + entry.name
                            + ": expected " + entry.checksum + ", computed " + computed);
                }
                if (data.length != entry.elementCount()) {
                    throw new LoadException("TENSOR_SHAPE mismatch for " + entry.name + ": shape "
                            + java.util.Arrays.toString(entry.shape) + " element count " + entry.elementCount()
                            + " != stored floats " + data.length);
                }
                if (options.scanForNaN() && containsNonFinite(data)) {
                    throw new LoadException("TENSOR_PAYLOAD contains NaN/Infinity: " + entry.name);
                }
            }
        }

        architecture.validate();
        return new LoadedModel(ModelFormat.EVO_V2, "2." + minor, modelName, createdAt, storedHash,
                architecture, tokenizer, metadata, tensors);
    }

    private boolean containsNonFinite(float[] data) {
        for (float v : data) {
            if (Float.isNaN(v) || Float.isInfinite(v)) {
                return true;
            }
        }
        return false;
    }

    private Architecture parseArchitecture(String json) throws LoadException {
        try {
            JsonObject obj = Json.parseObject(json);
            Architecture arch = new Architecture();
            arch.family(obj.getString("architecture_family", "evo_llm"));
            arch.vocabSize(readInt(obj, "vocab_size", "vocabSize", 0));
            arch.dModel(readInt(obj, "d_model", "dModel", 0));
            arch.numHeads(readInt(obj, "num_heads", "numHeads", 0));
            arch.numKvHeads(readInt(obj, "num_kv_heads", "numKvHeads", 0));
            arch.numBlocks(readInt(obj, "num_blocks", "numBlocks", 0));
            arch.dff(readInt(obj, "dff", "dff", 0));
            arch.maxSeqLen(readInt(obj, "max_seq_len", "maxSeqLen", 0));
            arch.normType(obj.getString("norm_type", "rms_norm"));
            arch.normEps(obj.getFloat("norm_eps", 1e-5f));
            arch.activationType(obj.getString("activation_type", "swiglu"));
            arch.positionalEncoding(obj.getString("positional_encoding", "sinusoidal"));
            arch.ropeTheta(obj.getFloat("rope_theta", 10000.0f));
            arch.embeddingTied(obj.getBoolean("embedding_tied", false));
            return arch;
        } catch (RuntimeException e) {
            throw new LoadException("Invalid architecture descriptor: " + e.getMessage(), e);
        }
    }

    private int readInt(JsonObject obj, String snake, String camel, int defaultValue) {
        if (obj.has(snake) && obj.get(snake) instanceof ai.enr.core.json.JsonNumber n) {
            return (int) n.asLong();
        }
        if (camel != null && obj.has(camel) && obj.get(camel) instanceof ai.enr.core.json.JsonNumber n) {
            return (int) n.asLong();
        }
        return defaultValue;
    }

    private TokenizerInfo parseTokenizer(String json) throws LoadException {
        try {
            JsonObject obj = Json.parseObject(json);
            TokenizerInfo tok = new TokenizerInfo();
            tok.type(obj.getString("tokenizer_type", "BPE"));
            tok.vocabSize(readInt(obj, "vocab_size", "vocabSize", 0));
            tok.bosTokenId(readInt(obj, "bos_token_id", null, 1));
            tok.eosTokenId(readInt(obj, "eos_token_id", null, 2));
            tok.unkTokenId(readInt(obj, "unk_token_id", null, 0));
            tok.padTokenId(readInt(obj, "pad_token_id", null, 0));
            tok.vocabularyChecksum(obj.getString("vocabulary_checksum", ""));
            return tok;
        } catch (RuntimeException e) {
            throw new LoadException("Invalid tokenizer descriptor: " + e.getMessage(), e);
        }
    }

    private List<ManifestEntry> parseManifest(String json) throws LoadException {
        try {
            JsonArray arr = Json.parseArray(json);
            List<ManifestEntry> manifest = new ArrayList<>();
            for (int i = 0; i < arr.size(); i++) {
                JsonObject obj = arr.getObject(i);
                if (obj == null) {
                    throw new LoadException("Invalid manifest entry " + i);
                }
                int id = obj.getInt("id", i);
                String name = obj.getString("name");
                if (name == null || name.isBlank()) {
                    throw new LoadException("Manifest entry " + i + " has no tensor name");
                }
                DType dtype = parseDtype(obj, i);
                JsonArray shapeArr = obj.require("shape") instanceof JsonArray a ? a : null;
                if (shapeArr == null) {
                    throw new LoadException("Manifest entry " + name + " has no shape");
                }
                long[] shape = new long[shapeArr.size()];
                for (int s = 0; s < shapeArr.size(); s++) {
                    shape[s] = shapeArr.getLong(s);
                }
                int layout = obj.getInt("layout", 0);
                long offset = obj.getLong("offset", 0);
                long compressedSize = obj.getLong("compressed_size", 0);
                long uncompressedSize = obj.getLong("uncompressed_size", 0);
                int compression = obj.getInt("compression", 1);
                String checksum = obj.getString("checksum", "");
                manifest.add(new ManifestEntry(id, name, dtype, shape, layout, offset,
                        compressedSize, uncompressedSize, compression, checksum));
            }
            return manifest;
        } catch (RuntimeException e) {
            throw new LoadException("Invalid tensor manifest: " + e.getMessage(), e);
        }
    }

    private DType parseDtype(JsonObject obj, int index) throws LoadException {
        ai.enr.core.json.JsonValue value = obj.get("dtype");
        if (value instanceof ai.enr.core.json.JsonNumber n) {
            try {
                return DType.fromCode((int) n.asLong());
            } catch (IllegalArgumentException e) {
                throw new LoadException("Manifest entry " + index + ": " + e.getMessage());
            }
        }
        if (value instanceof ai.enr.core.json.JsonString s) {
            try {
                return DType.valueOf(s.value().toUpperCase());
            } catch (IllegalArgumentException e) {
                throw new LoadException("Manifest entry " + index + ": unsupported dtype string '" + s.value() + "'");
            }
        }
        throw new LoadException("Manifest entry " + index + ": missing dtype");
    }

    private String readString32(DataInputStream dis) throws IOException {
        int len = dis.readInt();
        if (len < 0 || len > (1 << 30)) {
            throw new IOException("Invalid 32-bit string length: " + len);
        }
        byte[] bytes = new byte[len];
        dis.readFully(bytes);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    record ManifestEntry(int id, String name, DType dtype, long[] shape, int layout, long offset,
                         long compressedSize, long uncompressedSize, int compression, String checksum) {
        long elementCount() {
            long count = 1;
            for (long dim : shape) {
                count *= dim;
            }
            return count;
        }
    }

    public static String computeContentHash(Architecture arch, TokenizerInfo tok, List<TensorEntry> tensors) {
        ai.enr.core.json.JsonObject archObj = new ai.enr.core.json.JsonObject();
        for (Map.Entry<String, Object> e : arch.toMap().entrySet()) {
            putValue(archObj, e.getKey(), e.getValue());
        }
        ai.enr.core.json.JsonObject tokObj = new ai.enr.core.json.JsonObject();
        tokObj.put("tokenizer_type", tok.type());
        tokObj.put("vocab_size", tok.vocabSize());
        tokObj.put("bos_token_id", tok.bosTokenId());
        tokObj.put("eos_token_id", tok.eosTokenId());
        tokObj.put("unk_token_id", tok.unkTokenId());
        tokObj.put("pad_token_id", tok.padTokenId());
        tokObj.put("vocabulary_checksum", tok.vocabularyChecksum());

        var digest = new ai.enr.core.io.Sha256Digest();
        digest.update(Json.write(archObj, false).getBytes(StandardCharsets.UTF_8));
        digest.update(Json.write(tokObj, false).getBytes(StandardCharsets.UTF_8));

        List<TensorEntry> sorted = new ArrayList<>(tensors);
        sorted.sort(java.util.Comparator.comparing(TensorEntry::name));
        for (TensorEntry t : sorted) {
            digest.update(t.name().getBytes(StandardCharsets.UTF_8));
            digest.update(new byte[]{(byte) t.dtype().code()});
            java.nio.ByteBuffer bb = java.nio.ByteBuffer.allocate(Long.BYTES).order(java.nio.ByteOrder.BIG_ENDIAN);
            for (long dim : t.shape()) {
                bb.clear();
                bb.putLong(dim);
                digest.update(bb.array());
            }
        }
        for (TensorEntry t : tensors) {
            byte[] bytes = ai.enr.core.io.EndianIO.floatArrayToBigEndian(t.loadFloats());
            digest.update(bytes);
        }
        return ai.enr.core.io.Hex.encode(digest.digest());
    }

    private static void putValue(JsonObject obj, String key, Object value) {
        if (value instanceof Integer i) {
            obj.put(key, i);
        } else if (value instanceof Long l) {
            obj.put(key, l);
        } else if (value instanceof Float f) {
            obj.put(key, f);
        } else if (value instanceof Double d) {
            obj.put(key, d);
        } else if (value instanceof Boolean b) {
            obj.put(key, b);
        } else {
            obj.put(key, String.valueOf(value));
        }
    }
}
