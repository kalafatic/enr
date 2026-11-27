package ai.enr.server;

import ai.enr.core.json.JsonObject;
import ai.enr.manager.EnrManager;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

/**
 * Web Control Center: serves the static single-page UI from classpath
 * resources and exposes a small info endpoint. The UI is intentionally
 * dependency-free (vanilla HTML/CSS/JS); it talks to the ENR API over the
 * authoritative API base URL (port 28780), never the Web listener itself.
 */
public final class WebUiHandler implements HttpHandler {

    private final EnrManager manager;
    private final String apiBaseUrl;

    WebUiHandler(EnrManager manager, String apiBaseUrl) {
        this.manager = manager;
        this.apiBaseUrl = apiBaseUrl;
    }

    @Override
    public void handle(HttpExchange exchange) throws IOException {
        String path = exchange.getRequestURI().getPath();
        if ("/api/wcc/info".equals(path)) {
            JsonObject info = new JsonObject();
            info.put("name", "ENR");
            info.put("version", "2.6.11");
            info.put("auth_required", !manager.config().authToken().isBlank());
            info.put("auth_ok", HttpSupport.authorized(exchange, manager.config().authToken()));
            info.put("api_base_url", apiBaseUrl);
            HttpSupport.sendJson(exchange, 200, info);
            return;
        }
        if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
            HttpSupport.sendError(exchange, 405, "METHOD_NOT_ALLOWED", "Only GET allowed");
            return;
        }
        String resource = switch (path) {
            case "/", "/index.html" -> "/web/index.html";
            case "/style.css" -> "/web/style.css";
            case "/app.js" -> "/web/app.js";
            default -> null;
        };
        if (resource == null) {
            HttpSupport.sendError(exchange, 404, "NOT_FOUND", "No such page: " + path);
            return;
        }
        InputStream in = WebUiHandler.class.getResourceAsStream(resource);
        if (in == null) {
            HttpSupport.sendError(exchange, 404, "NOT_FOUND", "Missing web resource: " + resource);
            return;
        }
        byte[] bytes;
        try (in) {
            bytes = in.readAllBytes();
        }
        String contentType = resource.endsWith(".html") ? "text/html"
                : resource.endsWith(".css") ? "text/css"
                : resource.endsWith(".js") ? "application/javascript"
                : "application/octet-stream";
        exchange.getResponseHeaders().set("Content-Type", contentType + "; charset=utf-8");
        exchange.getResponseHeaders().set("Cache-Control", "no-cache");
        exchange.sendResponseHeaders(200, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }
}
