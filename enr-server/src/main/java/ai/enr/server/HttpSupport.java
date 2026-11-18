package ai.enr.server;

import ai.enr.core.json.Json;
import ai.enr.core.json.JsonObject;
import ai.enr.core.json.JsonValue;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

public final class HttpSupport {

    private HttpSupport() {
    }

    public static String readBody(HttpExchange exchange) throws IOException {
        try (InputStream in = exchange.getRequestBody()) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    public static JsonValue parseBody(HttpExchange exchange) throws IOException {
        String body = readBody(exchange);
        if (body.isBlank()) {
            return new JsonObject();
        }
        return Json.parse(body);
    }

    public static void sendJson(HttpExchange exchange, int status, JsonValue body) throws IOException {
        byte[] bytes = Json.write(body, false).getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    public static void sendJson(HttpExchange exchange, int status, JsonObject body) throws IOException {
        sendJson(exchange, status, (JsonValue) body);
    }

    public static void sendError(HttpExchange exchange, int status, String code, String message) throws IOException {
        JsonObject error = new JsonObject();
        JsonObject body = new JsonObject();
        body.put("error", error);
        error.put("code", code);
        error.put("message", message);
        sendJson(exchange, status, body);
    }

    public static JsonObject errorBody(String code, String message) {
        JsonObject error = new JsonObject();
        error.put("code", code);
        error.put("message", message);
        JsonObject body = new JsonObject();
        body.put("error", error);
        return body;
    }

    public static void sendSseEvent(OutputStream out, String event, JsonObject data) throws IOException {
        String payload = Json.write(data, false);
        StringBuilder sb = new StringBuilder();
        if (event != null) {
            sb.append("event: ").append(event).append('\n');
        }
        for (String line : payload.split("\n")) {
            sb.append("data: ").append(line).append('\n');
        }
        sb.append('\n');
        out.write(sb.toString().getBytes(StandardCharsets.UTF_8));
        out.flush();
    }
}
