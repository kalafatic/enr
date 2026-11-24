package ai.enr.cli;

import ai.enr.manager.service.SystemdSupport;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;

/**
 * Orchestrates ENR as an operating-system service and installs the {@code enr}
 * command. Service management (systemd) is kept here, separate from the
 * inference/runtime implementation. Both service and direct-terminal modes use
 * the same user data directory ({@code ~/.enr}); the user-level unit runs
 * {@code serve} with no explicit config, so it shares the direct-mode defaults.
 */
final class ServiceManager {

    enum Scope {
        SYSTEM, USER
    }

    private static final String UNIT_NAME = SystemdSupport.UNIT_NAME;

    private ServiceManager() {
    }

    static Scope detectScope() {
        boolean root = "root".equals(System.getProperty("user.name"));
        boolean systemWritable = Files.isWritable(Path.of(SystemdSupport.DEFAULT_SERVICE_DIR));
        return (root || systemWritable) ? Scope.SYSTEM : Scope.USER;
    }

    static Scope installedScope() {
        return SystemdSupport.installed() ? Scope.SYSTEM : Scope.USER;
    }

    static boolean installed() {
        return SystemdSupport.installed() || SystemdSupport.installedUser();
    }

    /** Installs the unit (system or user), the CLI wrapper, then starts and verifies health. */
    static void install(Path jarPath, String host, int port) throws IOException {
        String java = SystemdSupport.currentJavaBinary().toString();
        Path logDir = Path.of(System.getProperty("user.home"), ".enr", "logs");
        Files.createDirectories(logDir);
        Scope scope = detectScope();
        if (scope == Scope.SYSTEM) {
            String user = System.getProperty("user.name");
            Path config = Path.of(System.getProperty("user.home"), ".enr", "enr.yaml");
            String unit = SystemdSupport.unitFile(java, jarPath.toAbsolutePath().toString(),
                    config.toAbsolutePath().toString(), user, user, logDir.toAbsolutePath().toString());
            Files.writeString(Path.of(SystemdSupport.DEFAULT_SERVICE_DIR, UNIT_NAME), unit);
            run("systemctl", "daemon-reload");
            run("systemctl", "enable", UNIT_NAME);
        } else {
            SystemdSupport.installUser(Path.of(java), jarPath, logDir);
        }
        installCliWrapper(jarPath, scope);
        start(scope);
        System.out.println("Installed " + UNIT_NAME + " (" + scope.name().toLowerCase() + " scope).");
        System.out.println("  config/data: " + Path.of(System.getProperty("user.home"), ".enr"));
        System.out.println("  logs:        " + logDir);
        verifyHealth(host, port);
    }

    static void uninstall() throws IOException {
        Scope scope = installedScope();
        if (scope == Scope.SYSTEM) {
            SystemdSupport.uninstall();
        } else {
            SystemdSupport.uninstallUser();
        }
        removeCliWrapper(scope);
        System.out.println("Removed " + UNIT_NAME + ".");
    }

    static void start() throws IOException {
        start(installedScope());
    }

    static void start(Scope scope) throws IOException {
        if (scope == Scope.SYSTEM) {
            SystemdSupport.startSystem();
        } else {
            SystemdSupport.startUser();
        }
    }

    static void stop() throws IOException {
        Scope scope = installedScope();
        if (scope == Scope.SYSTEM) {
            SystemdSupport.stopSystem();
        } else {
            SystemdSupport.stopUser();
        }
    }

    static void restart() throws IOException {
        Scope scope = installedScope();
        if (scope == Scope.SYSTEM) {
            SystemdSupport.restartSystem();
        } else {
            SystemdSupport.restartUser();
        }
    }

    static String status() {
        Scope scope = installedScope();
        return scope == Scope.SYSTEM ? SystemdSupport.statusSystem() : SystemdSupport.statusUser();
    }

    static String logs(int lines) {
        Scope scope = installedScope();
        return scope == Scope.SYSTEM ? SystemdSupport.logsSystem(lines) : SystemdSupport.logsUser(lines);
    }

    /** Verifies the service answers the HTTP health endpoint, printing the result. */
    static void verifyHealth(String host, int port) {
        String displayHost = "0.0.0.0".equals(host) ? "127.0.0.1" : host;
        HttpClient http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(2))
                .build();
        for (int attempt = 1; attempt <= 15; attempt++) {
            try {
                HttpResponse<String> response = http.send(HttpRequest.newBuilder()
                        .uri(URI.create("http://" + displayHost + ":" + port + "/api/version"))
                        .timeout(Duration.ofSeconds(2))
                        .GET().build(), HttpResponse.BodyHandlers.ofString());
                if (response.statusCode() == 200) {
                    System.out.println("Service healthy: " + response.body());
                    return;
                }
            } catch (Exception ignored) {
                // service still starting
            }
            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
        System.out.println("Service did not become healthy within the timeout (check 'enr logs').");
    }

    // ------------------------------------------------------------ CLI wrapper in PATH

    private static Path cliWrapperTarget(Scope scope) {
        return scope == Scope.SYSTEM
                ? Path.of("/usr/local/bin/enr")
                : Path.of(System.getProperty("user.home"), ".local", "bin", "enr");
    }

    private static void installCliWrapper(Path jarPath, Scope scope) throws IOException {
        Path target = cliWrapperTarget(scope);
        Files.createDirectories(target.getParent());
        String script = "#!/usr/bin/env bash\n"
                + "# ENR CLI wrapper — installed by 'enr service install'.\n"
                + "if [ -x /usr/lib/jvm/java-25-openjdk-amd64/bin/java ]; then\n"
                + "  JAVA_BIN=/usr/lib/jvm/java-25-openjdk-amd64/bin/java\n"
                + "elif [ -x /usr/lib/jvm/java-21-openjdk-amd64/bin/java ]; then\n"
                + "  JAVA_BIN=/usr/lib/jvm/java-21-openjdk-amd64/bin/java\n"
                + "elif [ -n \"$JAVA_HOME\" ] && [ -x \"$JAVA_HOME/bin/java\" ]; then\n"
                + "  JAVA_BIN=\"$JAVA_HOME/bin/java\"\n"
                + "else\n"
                + "  JAVA_BIN=java\n"
                + "fi\n"
                + "exec \"$JAVA_BIN\" --enable-native-access=ALL-UNNAMED -jar '"
                + jarPath.toAbsolutePath().toString().replace("'", "'\\''") + "' \"$@\"\n";
        Files.writeString(target, script);
        target.toFile().setExecutable(true, false);
        System.out.println("CLI installed: " + target);
    }

    private static void removeCliWrapper(Scope scope) throws IOException {
        Files.deleteIfExists(cliWrapperTarget(scope));
    }

    private static void run(String... command) throws IOException {
        try {
            Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
            String output = new String(process.getInputStream().readAllBytes());
            int exit = process.waitFor();
            if (exit != 0) {
                throw new IOException(String.join(" ", command) + " exited " + exit
                        + (output.isBlank() ? "" : ": " + output.strip()));
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while running " + String.join(" ", command), e);
        }
    }
}
