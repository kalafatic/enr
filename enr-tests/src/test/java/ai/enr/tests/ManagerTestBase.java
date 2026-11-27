package ai.enr.tests;

import ai.enr.client.EnrClient;
import ai.enr.manager.EnrConfig;
import ai.enr.manager.EnrManager;
import ai.enr.server.EnrServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Shared fixture: real EnrManager + real EnrServer on an ephemeral port with
 * isolated data directories and the tiny-evo fixture imported.
 */
abstract class ManagerTestBase {

    protected static final Path FIXTURES =
            Path.of("../enr-model-loader/src/test/resources/models").toAbsolutePath();

    @TempDir
    protected Path tempDir;

    protected EnrManager manager;
    protected EnrServer server;
    protected EnrClient client;
    protected String base;

    @BeforeEach
    void setUpEnv() throws Exception {
        EnrConfig config = EnrConfig.builder()
                .host("127.0.0.1")
                .port(0)
                .dataDir(tempDir.resolve("data"))
                .modelsDir(tempDir.resolve("models"))
                .logDir(tempDir.resolve("logs"))
                .ollamaModelsDir(tempDir.resolve("ollama-models"))
                .defaultMaxTokens(16)
                .maxConcurrentSessions(16)
                .build();
        manager = new EnrManager(config);
        server = new EnrServer(manager, "127.0.0.1", 0);
        server.start();
        base = "http://127.0.0.1:" + server.port();
        client = new EnrClient(base);
    }

    @AfterEach
    void tearDownEnv() {
        server.close();
        manager.close();
    }

    protected Path importTinyEvo(String id) throws Exception {
        manager.store().importModel(FIXTURES.resolve("tiny-evo.gguf"), id);
        return manager.store().modelFile(id);
    }
}
