package ai.enr.server;

import ai.enr.core.json.Json;
import ai.enr.core.json.JsonObject;
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
 */
public final class EnrServer implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger(EnrServer.class.getName());

    private final EnrRuntime runtime;
    private final HttpServer server;
    private final ExecutorService executor;

    public EnrServer(EnrRuntime runtime, String host, int port) throws IOException {
        this.runtime = runtime;
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
        server.createContext("/", new RootHandler());
        server.setExecutor(executor);
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
            body.put("version", "1.0.0-SNAPSHOT");
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
