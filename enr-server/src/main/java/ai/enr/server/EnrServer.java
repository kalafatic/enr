package ai.enr.server;

import ai.enr.core.json.Json;
import ai.enr.core.json.JsonObject;
import ai.enr.manager.EnrManager;
import ai.enr.runtime.EnrRuntime;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.Logger;

/**
 * Standalone HTTP server for ENR (JDK HttpServer + virtual threads).
 * With an {@link EnrManager}, it serves the full surface: native /api/v1,
 * family /enr/v1, management REST, Ollama-compatible /api and
 * OpenAI-compatible /v1 plus the Web Control Center at /.
 */
public final class EnrServer implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger(EnrServer.class.getName());

    private final EnrRuntime runtime;
    private final EnrManager manager;
    private final HttpServer server;
    private final ExecutorService executor;

    public EnrServer(EnrRuntime runtime, String host, int port) throws IOException {
        this(runtime, null, host, port);
    }

    public EnrServer(EnrManager manager, String host, int port) throws IOException {
        this(manager.runtime(), manager, host, port);
    }

    private EnrServer(EnrRuntime runtime, EnrManager manager, String host, int port) throws IOException {
        this.runtime = runtime;
        this.manager = manager;
        this.server = HttpServer.create(new InetSocketAddress(host, port), 64);
        this.executor = Executors.newVirtualThreadPerTaskExecutor();
        registerRoutes();
    }

    private void registerRoutes() {
        server.createContext("/api/v1/health", ApiHandlers.health(runtime));
        server.createContext("/api/v1/status", ApiHandlers.status(runtime));
        server.createContext("/api/v1/models", ApiHandlers.listModels(runtime));
        server.createContext("/api/v1/models/load", ApiHandlers.loadModel(runtime));
        server.createContext("/api/v1/models/unload", ApiHandlers.unloadModel(runtime));
        server.createContext("/api/v1/models/", ApiHandlers.getModel(runtime));
        server.createContext("/api/v1/inference", ApiHandlers.infer(runtime));
        server.createContext("/api/v1/inference/stream", ApiHandlers.inferStream(runtime));
        server.createContext("/api/v1/inference/cancel", ApiHandlers.cancel(runtime));
        server.createContext("/api/v1/sessions", ApiHandlers.sessions(runtime));
        server.createContext("/api/v1/metrics", ApiHandlers.metrics(runtime));
        server.createContext("/api/v1/devices", ApiHandlers.devices(runtime));
        registerFamilyRoutes();
        if (manager != null) {
            registerManagerRoutes();
            server.createContext("/", new WebUiHandler(manager));
        } else {
            server.createContext("/", new RootHandler());
        }
        server.setExecutor(executor);
    }

    private void registerManagerRoutes() {
        // Ollama-compatible surface
        server.createContext("/api/tags", OllamaApiHandlers.tags(manager));
        server.createContext("/api/show", OllamaApiHandlers.show(manager));
        server.createContext("/api/pull", OllamaApiHandlers.pull(manager));
        server.createContext("/api/delete", OllamaApiHandlers.delete(manager));
        server.createContext("/api/chat", OllamaApiHandlers.chat(manager));
        server.createContext("/api/generate", OllamaApiHandlers.generate(manager));
        server.createContext("/api/ps", OllamaApiHandlers.ps(manager));
        server.createContext("/api/version", OllamaApiHandlers.version(manager));

        // OpenAI-compatible surface
        server.createContext("/v1/models", OpenAiApiHandlers.models(manager));
        server.createContext("/v1/chat/completions", OpenAiApiHandlers.chatCompletions(manager));
        server.createContext("/v1/completions", OpenAiApiHandlers.completions(manager));
        server.createContext("/v1/embeddings", OpenAiApiHandlers.embeddings(manager));

        // Management REST
        server.createContext("/api/v1/catalog/models", ManagerApiHandlers.catalogModels(manager));
        server.createContext("/api/v1/catalog/models/", ManagerApiHandlers.catalogModelRoute(manager));
        server.createContext("/api/v1/catalog/search", ManagerApiHandlers.catalogSearch(manager));
        server.createContext("/api/v1/catalog/pull", ManagerApiHandlers.catalogPull(manager));
        server.createContext("/api/v1/catalog/import", ManagerApiHandlers.catalogImport(manager));
        server.createContext("/api/v1/catalog/verify", ManagerApiHandlers.catalogVerify(manager));
        server.createContext("/api/v1/models/start", ManagerApiHandlers.modelStart(manager));
        server.createContext("/api/v1/models/stop", ManagerApiHandlers.modelStop(manager));
        server.createContext("/api/v1/downloads", ManagerApiHandlers.downloads(manager));
        server.createContext("/api/v1/downloads/cancel", ManagerApiHandlers.downloadCancel(manager));
        server.createContext("/api/v1/providers", ManagerApiHandlers.providers(manager));
        server.createContext("/api/v1/providers/", ManagerApiHandlers.providerRoute(manager));
        server.createContext("/api/v1/chat", ManagerApiHandlers.chat(manager));
        server.createContext("/api/v1/dashboard", ManagerApiHandlers.dashboard(manager));
        server.createContext("/api/v1/diagnostics", ManagerApiHandlers.diagnostics(manager));
        server.createContext("/api/v1/logs", ManagerApiHandlers.logs(manager));
        server.createContext("/api/v1/hardware", ManagerApiHandlers.hardware(manager));
    }

    /**
     * Family ENR 1.0 surface ({@code /enr/v1}) consumed by DCode, TRYST and
     * DStudio. The root context keeps serving the native {@code /api/v1}
     * surface unchanged.
     */
    private void registerFamilyRoutes() {
        server.createContext("/enr/v1/health", FamilyApiHandlers.familyHealth(runtime));
        server.createContext("/enr/v1/models", FamilyApiHandlers.familyListModels(runtime));
        server.createContext("/enr/v1/models/", FamilyApiHandlers.familyModelRoute(runtime));
        server.createContext("/enr/v1/chat", FamilyApiHandlers.familyChat(runtime));
        server.createContext("/enr/v1/chat/stream", FamilyApiHandlers.familyChatStream(runtime));
        server.createContext("/enr/v1/runtime", FamilyApiHandlers.familyRuntime(runtime));
        server.createContext("/enr/v1/providers", FamilyApiHandlers.familyProviders(runtime));
        server.createContext("/enr/v1/capabilities", FamilyApiHandlers.familyCapabilities(runtime));
    }

    public void start() {
        server.start();
        LOG.info(() -> "ENR HTTP server listening on " + server.getAddress());
    }

    public InetSocketAddress address() {
        return server.getAddress();
    }

    public int port() {
        return server.getAddress().getPort();
    }

    @Override
    public void close() {
        server.stop(1);
        executor.shutdown();
    }

    private static final class RootHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            JsonObject body = new JsonObject();
            body.put("name", "EVO Native Runtime (ENR)");
            body.put("version", "2.6.11");
            body.put("api", "/api/v1");
            body.put("familyApi", "/enr/v1");
            byte[] bytes = Json.write(body, false).getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
            exchange.sendResponseHeaders(200, bytes.length);
            try (OutputStream out = exchange.getResponseBody()) {
                out.write(bytes);
            }
        }
    }
}
