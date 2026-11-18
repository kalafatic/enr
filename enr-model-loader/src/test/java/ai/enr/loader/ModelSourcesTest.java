package ai.enr.loader;

import ai.enr.loader.api.LoadedModel;
import ai.enr.loader.evo.EvoModelWriter;
import ai.enr.loader.evo.LoadOptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class ModelSourcesTest {

    @TempDir
    Path tempDir;

    @Test
    void rejectsMissingFile() {
        assertThrows(ai.enr.loader.api.LoadException.class,
                () -> ModelSources.open(tempDir.resolve("nope.evo")));
    }

    @Test
    void rejectsUnknownMagic() throws Exception {
        Path bad = tempDir.resolve("bad.evo");
        Files.writeString(bad, "NOTAMODEL");
        assertThrows(ai.enr.loader.api.LoadException.class,
                () -> ModelSources.open(bad));
    }

    @Test
    void evoV2RoundTrip() throws Exception {
        LoadedModel fixture = FixtureModels.tiny();
        Path file = tempDir.resolve("tiny.evo");
        new EvoModelWriter().write(file, fixture);

        LoadedModel loaded = ModelSources.open(file, LoadOptions.DEFAULT);
        assertEquals("tiny", loaded.name());
        assertEquals(16, loaded.architecture().vocabSize());
        assertEquals(8, loaded.architecture().dModel());
        assertEquals(fixture.tensors().size(), loaded.tensors().size());

        var original = fixture.requireTensor("token_embd.weight");
        var reloaded = loaded.requireTensor("token_embd.weight");
        assertArrayEquals(original.loadFloats(), reloaded.loadFloats());

        var gate = loaded.requireTensor("blk.0.ffn_gate.weight");
        assertArrayEquals(fixture.requireTensor("blk.0.ffn_gate.weight").loadFloats(),
                gate.loadFloats());
    }

    @Test
    void evoV2ChecksumViolationDetected() throws Exception {
        LoadedModel fixture = FixtureModels.tiny();
        Path file = tempDir.resolve("tiny.evo");
        new EvoModelWriter().write(file, fixture);

        byte[] bytes = Files.readAllBytes(file);
        for (int i = bytes.length - 1; i > bytes.length - 64 && i >= 0; i--) {
            bytes[i] = (byte) (bytes[i] ^ 0xFF);
        }
        Path corrupt = tempDir.resolve("corrupt.evo");
        Files.write(corrupt, bytes);

        assertThrows(ai.enr.loader.api.LoadException.class,
                () -> ModelSources.open(corrupt, LoadOptions.DEFAULT));
    }
}
