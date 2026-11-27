package ai.enr.server;

import ai.enr.manager.EnrManager;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Independent ENR Web Control Center listener (authoritative port 28880). Serves
 * the static single-page UI and web-specific endpoints only; it never binds or
 * intercepts the ENR API listener (port 28780). The UI communicates with the API
 * over the injected {@code apiBaseUrl}.
 */
public final class EnrWebServer implements AutoCloseable {

    private final HttpServer server;
    private final ExecutorService executor;

    public EnrWebServer(EnrManager manager, String host, int webPort,
                        String apiHost, int apiPort) throws IOException {
        String apiBaseUrl = "http://" + displayHost(apiHost) + ":" + apiPort;
        this.server = HttpServer.create(new InetSocketAddress(host, webPort), 64);
        this.executor = Executors.newVirtualThreadPerTaskExecutor();
        server.createContext("/", new WebUiHandler(manager, apiBaseUrl));
        server.setExecutor(executor);
    }

    public void start() {
        server.start();
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

    private static String displayHost(String host) {
        return "0.0.0.0".equals(host) || host == null ? "127.0.0.1" : host;
    }
}
