package ai.enr.tests;

import ai.enr.core.json.Json;
import ai.enr.core.json.JsonObject;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Provider error propagation: upstream HTTP failures must surface to the
 * harness with meaningful status codes instead of a blanket 500. Rate limiting
 * (429) propagates for backoff; upstream timeouts (408) map to 504; upstream
 * auth failures (401/403) and other provider errors map to 502 Bad Gateway so
 * the harness does not mistake them for its own request errors.
 */
class ProviderErrorPropagationTest extends ManagerTestBase {

    private final HttpClient http = HttpClient.newBuilder().build();

    private HttpServer startProvider(int status) throws Exception {
        HttpServer provider = MockServers.statusProvider("sk-test-key", status);
        manager.addProvider("mock", "openai-compatible", "Mock",
                "http://127.0.0.1:" + provider.getAddress().getPort() + "/v1", "sk-test-key", true);
        return provider;
    }

    private HttpResponse<String> chat() throws Exception {
        return post("/api/chat",
                "{\"model\":\"mock:mock-model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
    }

    @Test
    void rateLimitPropagates429() throws Exception {
        HttpServer provider = startProvider(429);
        try {
            HttpResponse<String> response = chat();
            assertEquals(429, response.statusCode(), response.body());
        } finally {
            provider.stop(0);
        }
    }

    @Test
    void upstreamTimeoutMapsTo504() throws Exception {
        HttpServer provider = startProvider(408);
        try {
            HttpResponse<String> response = chat();
            assertEquals(504, response.statusCode(), response.body());
        } finally {
            provider.stop(0);
        }
    }

    @Test
    void upstreamAuthFailureMapsTo502() throws Exception {
        HttpServer provider = startProvider(401);
        try {
            HttpResponse<String> response = chat();
            assertEquals(502, response.statusCode(), response.body());
            // the error must not leak the provider API key
            assertFalse(response.body().contains("sk-test-key"), response.body());
        } finally {
            provider.stop(0);
        }
    }

    @Test
    void providerUnavailablePropagates503() throws Exception {
        HttpServer provider = startProvider(503);
        try {
            HttpResponse<String> response = chat();
            assertEquals(503, response.statusCode(), response.body());
        } finally {
            provider.stop(0);
        }
    }

    @Test
    void openAiSurfacePropagates429() throws Exception {
        HttpServer provider = startProvider(429);
        try {
            HttpResponse<String> response = post("/v1/chat/completions",
                    "{\"model\":\"mock:mock-model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
            assertEquals(429, response.statusCode(), response.body());
        } finally {
            provider.stop(0);
        }
    }

    @Test
    void providerUnreachableReturns502() throws Exception {
        // provider whose port nothing listens on -> connection refused
        manager.addProvider("dead", "openai-compatible", "Dead",
                "http://127.0.0.1:1/v1", "sk-test-key", true);
        HttpResponse<String> response = post("/api/chat",
                "{\"model\":\"dead:dead-model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
        assertEquals(502, response.statusCode(), response.body());
        JsonObject body = Json.parseObject(response.body());
        assertTrue(body.get("error") instanceof ai.enr.core.json.JsonString, "Ollama error must be a plain string");
    }

    // ------------------------------------------------------------ helpers

    private HttpResponse<String> post(String path, String body) throws Exception {
        return http.send(HttpRequest.newBuilder().uri(URI.create(base + path))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body)).build(),
                HttpResponse.BodyHandlers.ofString());
    }
}
