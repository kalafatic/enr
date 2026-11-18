package ai.enr.loader;

import ai.enr.loader.api.LoadedModel;
import ai.enr.loader.api.ModelFormat;
import ai.enr.loader.api.TensorEntry;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Validation against REAL Darwin Forge exported models
 * (copied read-only from the EVO reference repository:
 *  eu.kalafatic.evolution.controller.tests/lib/llama-cpp/{evo.gguf,tiny-evo.gguf}).
 */
class RealModelLoadTest {

    private static final Path TINY = Path.of("src/test/resources/models/tiny-evo.gguf");
    private static final Path EVO = Path.of("src/test/resources/models/evo.gguf");

    @Test
    void loadsRealTinyModel() throws Exception {
        LoadedModel model = ModelSources.open(TINY);
        assertEquals(ModelFormat.GGUF_V3, model.format());
        assertEquals(16, model.architecture().vocabSize());
        assertEquals(8, model.architecture().dModel());
        assertEquals(2, model.architecture().numHeads());
        assertEquals(1, model.architecture().numBlocks());
        assertEquals(16, model.architecture().dff());
        assertEquals(32, model.architecture().maxSeqLen());
        assertEquals("rope", model.architecture().positionalEncoding());
        assertEquals(12, model.tensors().size());
        assertEquals(16, model.tokenizer().vocabSize());
        assertEquals(1, model.tokenizer().bosTokenId());
        assertEquals(2, model.tokenizer().eosTokenId());
    }

    @Test
    void loadsRealEvoModel() throws Exception {
        LoadedModel model = ModelSources.open(EVO);
        assertEquals(ModelFormat.GGUF_V3, model.format());
        assertEquals(100, model.architecture().vocabSize());
        assertEquals(64, model.architecture().dModel());
        assertEquals(2, model.architecture().numHeads());
        assertEquals(1, model.architecture().numBlocks());
        assertEquals(128, model.architecture().dff());
        assertEquals(12, model.tensors().size());
        assertEquals(100, model.tokenizer().vocabSize());
    }

    @Test
    void realModelTensorLayoutAndStats() throws Exception {
        LoadedModel model = ModelSources.open(EVO);
        TensorEntry emb = model.requireTensor("token_embd.weight");
        assertArrayEquals(new long[]{100, 64}, emb.shape());
        float[] embData = emb.loadFloats();
        float std = stddev(embData);
        assertTrue(Math.abs(std - 0.125f) < 0.01f,
                "token embedding std dev " + std + " deviates from expected ~0.125 (1/sqrt(64))");

        TensorEntry gate = model.requireTensor("blk.0.ffn_gate.weight");
        assertArrayEquals(new long[]{64, 128}, gate.shape());
        float[] gateData = gate.loadFloats();
        assertTrue(Math.abs(stddev(gateData) - 0.125f) < 0.01f,
                "ffn_gate std dev " + stddev(gateData) + " deviates from expected ~0.125");

        TensorEntry output = model.requireTensor("output.weight");
        assertArrayEquals(new long[]{64, 100}, output.shape());

        TensorEntry down = model.requireTensor("blk.0.ffn_down.weight");
        assertArrayEquals(new long[]{128, 64}, down.shape());
    }

    @Test
    void validatorPassesRealModels() throws Exception {
        var report = ModelValidator.validate(TINY);
        assertTrue(report.valid(), report.text());
        report = ModelValidator.validate(EVO);
        assertTrue(report.valid(), report.text());
    }

    private static float stddev(float[] data) {
        double mean = 0;
        for (float v : data) {
            mean += v;
        }
        mean /= data.length;
        double var = 0;
        for (float v : data) {
            var += (v - mean) * (v - mean);
        }
        return (float) Math.sqrt(var / data.length);
    }
}
