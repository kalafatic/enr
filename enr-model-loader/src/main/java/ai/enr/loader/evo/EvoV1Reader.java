package ai.enr.loader.evo;

import ai.enr.core.io.EndianIO;
import ai.enr.core.tensor.DType;
import ai.enr.loader.api.Architecture;
import ai.enr.loader.api.LoadedModel;
import ai.enr.loader.api.LoadException;
import ai.enr.loader.api.ModelFormat;
import ai.enr.loader.api.TensorEntry;
import ai.enr.loader.api.TensorStores;
import ai.enr.loader.api.TokenizerInfo;

import java.io.DataInputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPInputStream;

/**
 * Read-only support for the legacy .evo v1 format (whole-file GZIP,
 * "EVO_ARTIFACT_V1" magic, writeUTF strings, no checksums).
 */
final class EvoV1Reader {

    LoadedModel readV1(DataInputStream bis, Path path) throws LoadException {
        try (GZIPInputStream gzis = new GZIPInputStream(bis);
             DataInputStream dis = new DataInputStream(gzis)) {

            String magic = dis.readUTF();
            if (!"EVO_ARTIFACT_V1".equals(magic)) {
                throw new LoadException("EVO_PROTOCOL_ERROR: invalid magic '" + magic + "' (not EVO_NAT2, not EVO_ARTIFACT_V1)");
            }
            String modelName = dis.readUTF();
            dis.readUTF(); // modelVersion
            long createdAt = dis.readLong();

            int vocabSize = dis.readInt();
            int dModel = dis.readInt();
            int numHeads = dis.readInt();
            int numBlocks = dis.readInt();
            int dff = dis.readInt();
            int maxSeqLen = dis.readInt();

            dis.readFloat(); // temperature
            dis.readFloat(); // topP
            dis.readInt();   // topK
            dis.readFloat(); // repeatPenalty
            dis.readFloat(); // frequencyPenalty
            dis.readFloat(); // presencePenalty

            int bos = dis.readInt();
            int eos = dis.readInt();
            int unk = dis.readInt();
            int pad = dis.readInt();

            int vocabCount = dis.readInt();
            TokenizerInfo tokenizer = new TokenizerInfo()
                    .vocabSize(vocabSize)
                    .bosTokenId(bos).eosTokenId(eos).unkTokenId(unk).padTokenId(pad);
            for (int i = 0; i < vocabCount; i++) {
                int id = dis.readInt();
                String token = dis.readUTF();
                tokenizer.addToken(token, id);
            }

            int metaSize = dis.readInt();
            Map<String, String> metadata = new LinkedHashMap<>();
            for (int i = 0; i < metaSize; i++) {
                metadata.put(dis.readUTF(), dis.readUTF());
            }

            int weightCount = dis.readInt();
            List<TensorEntry> tensors = new ArrayList<>();
            for (int i = 0; i < weightCount; i++) {
                String name = dis.readUTF();
                int shapeLen = dis.readInt();
                long[] shape = new long[shapeLen];
                for (int d = 0; d < shapeLen; d++) {
                    shape[d] = dis.readLong();
                }
                int dataLen = dis.readInt();
                if (dataLen < 0 || dataLen > (1 << 28)) {
                    throw new LoadException("EVO_PROTOCOL_ERROR: invalid tensor data length " + dataLen + " for " + name);
                }
                float[] data = new float[dataLen];
                for (int j = 0; j < dataLen; j++) {
                    data[j] = dis.readFloat();
                }
                String checksum = TensorStores.checksumOfFloats(data);
                tensors.add(new TensorEntry(i, name, DType.F32, shape, 0, 0,
                        (long) dataLen * 4, (long) dataLen * 4, 1, checksum, TensorStores.eager(data)));
            }

            Architecture architecture = new Architecture()
                    .vocabSize(vocabSize).dModel(dModel).numHeads(numHeads)
                    .numBlocks(numBlocks).dff(dff).maxSeqLen(maxSeqLen);
            architecture.validate();

            String hash = EvoModelReader.computeContentHash(architecture, tokenizer, tensors);
            return new LoadedModel(ModelFormat.EVO_V1, "1.0", modelName, createdAt, hash,
                    architecture, tokenizer, metadata, tensors);
        } catch (LoadException e) {
            throw e;
        } catch (IOException e) {
            throw new LoadException("EVO_PROTOCOL_ERROR: failed to read legacy v1 model " + path + ": " + e.getMessage(), e);
        }
    }
}
