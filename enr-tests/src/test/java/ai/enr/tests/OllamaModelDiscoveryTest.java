package ai.enr.tests;

import ai.enr.core.io.Hashing;
import ai.enr.manager.EnrConfig;
import ai.enr.manager.logs.LogBuffer;
import ai.enr.manager.store.ModelStore;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Ollama model-storage discovery: ENR must discover the same locally available
 * Ollama model artifacts (manifest-registered models plus bare .gguf/.evo files)
 * without copying them, so ENR and Ollama agree on model identifiers.
 */
class OllamaModelDiscoveryTest {

    @TempDir
    Path tempDir;

    private ModelStore storeFor(Path ollamaDir) {
        EnrConfig config = EnrConfig.builder()
                .dataDir(tempDir.resolve("data"))
                .modelsDir(tempDir.resolve("models"))
                .ollamaModelsDir(ollamaDir)
                .build();
        return new ModelStore(config, new LogBuffer(10));
    }

    @Test
    void discoversManifestRegisteredOllamaModels() throws Exception {
        Path ollamaDir = tempDir.resolve("ollama-models");
        Path manifests = ollamaDir.resolve("manifests/registry.ollama.ai/library/testmodel");
        Path blobs = ollamaDir.resolve("blobs");
        Files.createDirectories(manifests);
        Files.createDirectories(blobs);

        byte[] content = "GGUF-test-model-content".getBytes(java.nio.charset.StandardCharsets.UTF_8);
        String digest = Hashing.sha256Hex(content);
        Files.write(blobs.resolve("sha256-" + digest), content);
        Files.writeString(manifests.resolve("1b"),
                "{\"schemaVersion\":2,\"layers\":["
                        + "{\"mediaType\":\"application/vnd.ollama.image.model\","
                        + "\"digest\":\"sha256:" + digest + "\",\"size\":" + content.length + "}]}");

        ModelStore store = storeFor(ollamaDir);
        ModelStore.StoredModel model = store.find("testmodel:1b");

        assertNotNull(model, "manifest-registered Ollama model must be discovered");
        assertEquals("gguf", model.manifest().format());
        assertEquals("ollama-registry", model.manifest().source());
        assertEquals(digest, model.manifest().digest());
        // The artifact is referenced in place (the Ollama blob), never copied.
        assertEquals(blobs.resolve("sha256-" + digest), model.manifest().modelFile());
        assertTrue(model.artifactPresent());
    }

    @Test
    void discoversNonLibraryNamespaceModels() throws Exception {
        Path ollamaDir = tempDir.resolve("ollama-models");
        Path manifests = ollamaDir.resolve("manifests/registry.ollama.ai/acme/custom-model");
        Path blobs = ollamaDir.resolve("blobs");
        Files.createDirectories(manifests);
        Files.createDirectories(blobs);

        byte[] content = "GGUF-custom".getBytes(java.nio.charset.StandardCharsets.UTF_8);
        String digest = Hashing.sha256Hex(content);
        Files.write(blobs.resolve("sha256-" + digest), content);
        Files.writeString(manifests.resolve("latest"),
                "{\"schemaVersion\":2,\"layers\":[{\"mediaType\":\"application/vnd.ollama.image.model\","
                        + "\"digest\":\"sha256:" + digest + "\",\"size\":" + content.length + "}]}");

        ModelStore store = storeFor(ollamaDir);
        // Non-library namespace keeps the namespace in the identifier.
        assertNotNull(store.find("acme-custom-model:latest"));
    }

    @Test
    void discoversBareOllamaArtifacts() throws Exception {
        Path ollamaDir = tempDir.resolve("ollama-models");
        Files.createDirectories(ollamaDir);
        Files.write(ollamaDir.resolve("myforge.gguf"), "GGUF-forged".getBytes());
        Files.write(ollamaDir.resolve("myevo.evo"), "EVO-forged".getBytes());

        ModelStore store = storeFor(ollamaDir);

        assertEquals("ollama-local", store.find("myforge").manifest().source());
        assertEquals("evo", store.find("myevo").manifest().format());
    }

    @Test
    void doesNotDuplicateAcrossSources() throws Exception {
        Path ollamaDir = tempDir.resolve("ollama-models");
        Files.createDirectories(ollamaDir);
        Files.write(ollamaDir.resolve("shared.gguf"), "GGUF".getBytes());

        EnrConfig config = EnrConfig.builder()
                .dataDir(tempDir.resolve("data"))
                .modelsDir(tempDir.resolve("models"))
                .modelDirectories(java.util.List.of(ollamaDir))
                .ollamaModelsDir(ollamaDir)
                .build();
        ModelStore store = new ModelStore(config, new LogBuffer(10));

        // "shared" appears from both model_directories and the Ollama dir; it must be listed once.
        long count = store.list().stream().filter(m -> m.id().equals("shared")).count();
        assertEquals(1, count, "model must not be duplicated across discovery sources");
    }
}
