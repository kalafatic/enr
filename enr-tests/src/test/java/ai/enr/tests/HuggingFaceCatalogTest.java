package ai.enr.tests;

import ai.enr.core.json.Json;
import ai.enr.manager.EnrConfig;
import ai.enr.manager.catalog.Catalogs;
import ai.enr.manager.logs.LogBuffer;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Hugging Face catalog integration: repository resolution, GGUF discovery,
 * quantization selection, ambiguity/absence errors, authentication headers and
 * resumable (HTTP Range) downloads — all against a deterministic mock server.
 */
class HuggingFaceCatalogTest {

    private static final byte[] MODEL_BYTES =
            "GGUF-mock-model-content-for-range-tests".getBytes(java.nio.charset.StandardCharsets.UTF_8);

    @TempDir
    Path tempDir;

    private Catalogs catalogsAt(String baseUrl) {
        EnrConfig config = EnrConfig.builder()
                .huggingFaceApiUrl(baseUrl)
                .huggingFaceToken("")
                .build();
        return new Catalogs(config, new LogBuffer(10));
    }

    // ------------------------------------------------------------ quantization matching

    @Test
    void matchesQuantExactSuffix() {
        assertTrue(Catalogs.matchesQuant("Inkling-2-Mini-UD-Q4_K_M.gguf", "UD-Q4_K_M"));
        assertTrue(Catalogs.matchesQuant("model-Q4_K_M.gguf", "Q4_K_M"));
        assertTrue(Catalogs.matchesQuant("model-Q8_0.gguf", "Q8_0"));
        assertFalse(Catalogs.matchesQuant("model-Q4_K_M.gguf", "UD-Q4_K_M"));
        assertFalse(Catalogs.matchesQuant("mmproj-F16.gguf", "Q4_K_M"));
    }

    @Test
    void matchesQuantIsCaseInsensitive() {
        assertTrue(Catalogs.matchesQuant("model-q4_k_m.gguf", "Q4_K_M"));
        assertTrue(Catalogs.matchesQuant("model-Q4_K_M.gguf", "q4_k_m"));
    }

    // ------------------------------------------------------------ repository resolution

    @Test
    void resolveQuantSelectsMatchingGguf() throws Exception {
        AtomicReference<String> auth = new AtomicReference<>();
        HttpServer server = MockServers.huggingFaceServer(
                List.of("Inkling-2-Mini-UD-Q4_K_M.gguf", "Inkling-2-Mini-UD-Q8_0.gguf", "README.md"),
                Map.of("Inkling-2-Mini-UD-Q4_K_M.gguf", MODEL_BYTES,
                        "Inkling-2-Mini-UD-Q8_0.gguf", MODEL_BYTES),
                auth);
        try {
            Catalogs catalogs = catalogsAt("http://127.0.0.1:" + server.getAddress().getPort());
            Catalogs.ResolvedDownload resolved = catalogs.resolveDownload(
                    "hf:unsloth/Inkling-2-Mini-GGUF:UD-Q4_K_M");
            assertTrue(resolved.url().contains("Inkling-2-Mini-UD-Q4_K_M.gguf"), resolved.url());
            assertFalse(resolved.digest().isBlank(), "lfs oid must be used as digest");
        } finally {
            server.stop(0);
        }
    }

    @Test
    void resolveQuantMissingReportsAvailable() throws Exception {
        AtomicReference<String> auth = new AtomicReference<>();
        HttpServer server = MockServers.huggingFaceServer(
                List.of("Inkling-2-Mini-UD-Q4_K_M.gguf"),
                Map.of("Inkling-2-Mini-UD-Q4_K_M.gguf", MODEL_BYTES),
                auth);
        try {
            Catalogs catalogs = catalogsAt("http://127.0.0.1:" + server.getAddress().getPort());
            IOException ex = assertThrows(IOException.class,
                    () -> catalogs.resolveDownload("hf:unsloth/Inkling-2-Mini-GGUF:Q2_K"));
            assertTrue(ex.getMessage().contains("Q2_K"), ex.getMessage());
            assertTrue(ex.getMessage().contains("Available"), ex.getMessage());
        } finally {
            server.stop(0);
        }
    }

    @Test
    void resolveQuantAmbiguousFails() throws Exception {
        AtomicReference<String> auth = new AtomicReference<>();
        HttpServer server = MockServers.huggingFaceServer(
                List.of("model-Q4_K_M.gguf", "model-Q4_K_M-v2.gguf"),
                Map.of("model-Q4_K_M.gguf", MODEL_BYTES, "model-Q4_K_M-v2.gguf", MODEL_BYTES),
                auth);
        try {
            Catalogs catalogs = catalogsAt("http://127.0.0.1:" + server.getAddress().getPort());
            IOException ex = assertThrows(IOException.class,
                    () -> catalogs.resolveDownload("hf:owner/repo:Q4_K_M"));
            assertTrue(ex.getMessage().contains("ambiguous"), ex.getMessage());
        } finally {
            server.stop(0);
        }
    }

    @Test
    void resolveExplicitFileForm() throws Exception {
        AtomicReference<String> auth = new AtomicReference<>();
        HttpServer server = MockServers.huggingFaceServer(
                List.of("sub/model-q4.gguf"),
                Map.of("sub/model-q4.gguf", MODEL_BYTES),
                auth);
        try {
            Catalogs catalogs = catalogsAt("http://127.0.0.1:" + server.getAddress().getPort());
            Catalogs.ResolvedDownload resolved = catalogs.resolveDownload("hf:owner/repo/sub/model-q4.gguf");
            assertTrue(resolved.url().contains("owner/repo/resolve/main/sub/model-q4.gguf"), resolved.url());
        } finally {
            server.stop(0);
        }
    }

    @Test
    void hfDotCoPrefixIsAccepted() throws Exception {
        AtomicReference<String> auth = new AtomicReference<>();
        HttpServer server = MockServers.huggingFaceServer(
                List.of("Inkling-2-Mini-UD-Q4_K_M.gguf"),
                Map.of("Inkling-2-Mini-UD-Q4_K_M.gguf", MODEL_BYTES),
                auth);
        try {
            Catalogs catalogs = catalogsAt("http://127.0.0.1:" + server.getAddress().getPort());
            Catalogs.ResolvedDownload resolved = catalogs.resolveDownload(
                    "hf.co/unsloth/Inkling-2-Mini-GGUF:UD-Q4_K_M");
            assertTrue(resolved.url().contains("Inkling-2-Mini-UD-Q4_K_M.gguf"), resolved.url());
        } finally {
            server.stop(0);
        }
    }

    @Test
    void authTokenIsSentAsBearer() throws Exception {
        AtomicReference<String> auth = new AtomicReference<>();
        HttpServer server = MockServers.huggingFaceServer(
                List.of("model-Q4_K_M.gguf"),
                Map.of("model-Q4_K_M.gguf", MODEL_BYTES),
                auth);
        try {
            EnrConfig config = EnrConfig.builder()
                    .huggingFaceApiUrl("http://127.0.0.1:" + server.getAddress().getPort())
                    .huggingFaceToken("hf_secret_token")
                    .build();
            Catalogs catalogs = new Catalogs(config, new LogBuffer(10));
            catalogs.resolveDownload("hf:owner/repo:Q4_K_M");
            assertEquals("Bearer hf_secret_token", auth.get());
        } finally {
            server.stop(0);
        }
    }

    // ------------------------------------------------------------ resumable download

    @Test
    void downloadResumesFromPartialFile() throws Exception {
        AtomicReference<String> range = new AtomicReference<>();
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 8);
        server.createContext("/file.bin", exchange -> {
            String r = exchange.getRequestHeaders().getFirst("Range");
            range.set(r);
            exchange.getResponseHeaders().set("Content-Type", "application/octet-stream");
            if (r != null && r.startsWith("bytes=")) {
                long start = Long.parseLong(r.substring(6).split("-")[0]);
                byte[] part = java.util.Arrays.copyOfRange(MODEL_BYTES, (int) start, MODEL_BYTES.length);
                exchange.getResponseHeaders().set("Content-Range",
                        "bytes " + start + "-" + (MODEL_BYTES.length - 1) + "/" + MODEL_BYTES.length);
                exchange.sendResponseHeaders(206, part.length);
                try (OutputStream out = exchange.getResponseBody()) {
                    out.write(part);
                }
            } else {
                exchange.sendResponseHeaders(200, MODEL_BYTES.length);
                try (OutputStream out = exchange.getResponseBody()) {
                    out.write(MODEL_BYTES);
                }
            }
        });
        server.setExecutor(java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor());
        server.start();
        try {
            Path target = tempDir.resolve("model.gguf");
            Path part = Path.of(target + ".part");
            Files.write(part, java.util.Arrays.copyOfRange(MODEL_BYTES, 0, 5));

            ai.enr.manager.download.DownloadManager dm = new ai.enr.manager.download.DownloadManager();
            String digest = ai.enr.core.io.Hashing.sha256Hex(MODEL_BYTES);
            var latch = new java.util.concurrent.CountDownLatch(1);
            dm.start("model", "http://127.0.0.1:" + server.getAddress().getPort() + "/file.bin",
                    target, digest, job -> latch.countDown());
            assertTrue(latch.await(30, java.util.concurrent.TimeUnit.SECONDS));
            assertEquals("bytes=5-", range.get(), "resume must send a Range header");
            assertArrayEquals(MODEL_BYTES, Files.readAllBytes(target));
            assertFalse(Files.exists(part), "part file must be moved to the target");
            dm.close();
        } finally {
            server.stop(0);
        }
    }
}
