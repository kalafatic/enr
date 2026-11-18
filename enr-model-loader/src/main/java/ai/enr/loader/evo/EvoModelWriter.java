package ai.enr.loader.evo;

import ai.enr.core.io.EndianIO;
import ai.enr.core.json.Json;
import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonObject;
import ai.enr.loader.api.Architecture;
import ai.enr.loader.api.LoadedModel;
import ai.enr.loader.api.TensorEntry;
import ai.enr.loader.api.TensorStores;
import ai.enr.loader.api.TokenizerInfo;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPOutputStream;

/**
 * Deterministic writer for the .evo v2 container, byte-compatible with the
 * audited EVO Native Model Protocol. Used by ENR to produce test fixtures and
 * to interop with Darwin Forge tooling.
 */
public final class EvoModelWriter {

    public void write(Path path, LoadedModel model) throws IOException {
        List<TensorEntry> tensors = model.tensors();
        byte[][] compressed = new byte[tensors.size()][];

        JsonArray manifestArr = new JsonArray();
        long currentOffset = 0;
        for (int i = 0; i < tensors.size(); i++) {
            TensorEntry t = tensors.get(i);
            byte[] raw = EndianIO.floatArrayToBigEndian(t.loadFloats());
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            try (GZIPOutputStream gzos = new GZIPOutputStream(baos)) {
                gzos.write(raw);
            }
            byte[] comp = baos.toByteArray();
            compressed[i] = comp;

            JsonObject obj = new JsonObject();
            obj.put("id", i);
            obj.put("name", t.name());
            obj.put("dtype", t.dtype().code());
            JsonArray shapeArr = new JsonArray();
            for (long s : t.shape()) {
                shapeArr.add(s);
            }
            obj.put("shape", shapeArr);
            obj.put("layout", 0);
            obj.put("offset", currentOffset);
            obj.put("compressed_size", comp.length);
            obj.put("uncompressed_size", raw.length);
            obj.put("compression", 1);
            obj.put("checksum", TensorStores.checksumOfFloats(t.loadFloats()));
            manifestArr.add(obj);
            currentOffset += comp.length;
        }

        JsonObject archObj = new JsonObject();
        for (Map.Entry<String, Object> e : model.architecture().toMap().entrySet()) {
            putValue(archObj, e.getKey(), e.getValue());
        }
        TokenizerInfo tok = model.tokenizer();
        JsonObject tokObj = new JsonObject();
        tokObj.put("tokenizer_type", tok.type());
        tokObj.put("vocab_size", tok.vocabSize());
        tokObj.put("bos_token_id", tok.bosTokenId());
        tokObj.put("eos_token_id", tok.eosTokenId());
        tokObj.put("unk_token_id", tok.unkTokenId());
        tokObj.put("pad_token_id", tok.padTokenId());
        tokObj.put("vocabulary_checksum", tok.vocabularyChecksum());

        JsonObject vocabObj = new JsonObject();
        for (Map.Entry<String, Integer> e : tok.tokenToId().entrySet()) {
            vocabObj.put(e.getKey(), e.getValue());
        }

        JsonObject metaObj = new JsonObject();
        for (Map.Entry<String, String> e : model.metadata().entrySet()) {
            metaObj.put(e.getKey(), e.getValue());
        }

        String contentHash = EvoModelReader.computeContentHash(model.architecture(), tok, tensors);

        Path parent = path.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path temp = Files.createTempFile(parent, "enr-evo-", ".tmp");

        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(temp.toFile())))) {
            dos.write(EvoModelReader.MAGIC_V2.getBytes(StandardCharsets.US_ASCII));
            dos.writeShort(2);
            dos.writeShort(0);
            dos.writeShort(1);
            dos.writeShort(0);
            dos.writeInt(64);
            dos.writeInt(5);
            dos.writeLong(model.createdAt() > 0 ? model.createdAt() : System.currentTimeMillis());
            byte[] hashBytes = contentHash.getBytes(StandardCharsets.UTF_8);
            byte[] hashPadded = new byte[32];
            System.arraycopy(hashBytes, 0, hashPadded, 0, Math.min(hashBytes.length, 32));
            dos.write(hashPadded);

            writeString32(dos, model.name());
            writeString32(dos, Json.write(archObj, false));
            writeString32(dos, Json.write(tokObj, false));
            writeString32(dos, Json.write(vocabObj, false));
            writeString32(dos, Json.write(manifestArr, false));
            writeString32(dos, Json.write(metaObj, false));

            dos.writeInt(compressed.length);
            for (byte[] comp : compressed) {
                dos.writeInt(comp.length);
                dos.write(comp);
            }
        }
        Files.move(temp, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    private void writeString32(DataOutputStream dos, String str) throws IOException {
        byte[] bytes = str.getBytes(StandardCharsets.UTF_8);
        dos.writeInt(bytes.length);
        dos.write(bytes);
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
