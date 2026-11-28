package ai.enr.manager.service;

import ai.enr.manager.EnrConfig;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Generates and installs a production systemd unit for the ENR server.
 * The systemd-managed process serves both the ENR Server API and the Web
 * Control Center; closing a browser has no effect on the service.
 */
public final class SystemdSupport {

    public static final String UNIT_NAME = "enr.service";
    public static final String DEFAULT_CONFIG_DIR = "/etc/enr";
    public static final String DEFAULT_SERVICE_DIR = "/etc/systemd/system";

    private SystemdSupport() {
    }

    public static String unitFile(String javaBinary, String jarPath, String configPath,
                                  String user, String group, String logDir) {
        return """
                [Unit]
                Description=ENR Server — Native AI Model Inference Runtime & Web Control Center
                Documentation=https://example.invalid/enr
                After=network-online.target
                Wants=network-online.target

                [Service]
                Type=simple
                User=%s
                Group=%s
                Environment=JAVA_HOME=%s
                ExecStart=%s --enable-native-access=ALL-UNNAMED -jar %s serve -c %s
                Restart=on-failure
                RestartSec=5
                TimeoutStopSec=60
                KillSignal=SIGTERM
                SuccessExitStatus=143
                LimitNOFILE=65536
                StandardOutput=append:%s/enr-server.out.log
                StandardError=append:%s/enr-server.err.log
                # Graceful shutdown: SIGTERM triggers model unloading in the CLI serve command.

                [Install]
                WantedBy=multi-user.target
                """.formatted(user, group, javaHomeOf(javaBinary), javaBinary, jarPath, configPath,
                logDir, logDir);
    }

    private static String javaHomeOf(String javaBinary) {
        Path bin = Path.of(javaBinary).toAbsolutePath();
        Path home = bin.getParent();
        if (home != null && home.getFileName() != null && "bin".equals(home.getFileName().toString())) {
            return home.getParent() == null ? "" : home.getParent().toString();
        }
        return System.getProperty("java.home", "");
    }

    /** Installs the unit into /etc/systemd/system and enables it (requires root). */
    public static void install(Path javaBinary, Path jarPath, Path configPath, String user,
                               String group, Path logDir) throws IOException {
        String unit = unitFile(javaBinary.toAbsolutePath().toString(),
                jarPath.toAbsolutePath().toString(),
                configPath.toAbsolutePath().toString(), user, group,
                logDir.toAbsolutePath().toString());
        Files.writeString(Path.of(DEFAULT_SERVICE_DIR, UNIT_NAME), unit);
        run("systemctl", "daemon-reload");
        run("systemctl", "enable", UNIT_NAME);
    }

    public static void uninstall() throws IOException {
        run("systemctl", "disable", "--now", UNIT_NAME);
        Files.deleteIfExists(Path.of(DEFAULT_SERVICE_DIR, UNIT_NAME));
        run("systemctl", "daemon-reload");
    }

    public static boolean installed() {
        return Files.exists(Path.of(DEFAULT_SERVICE_DIR, UNIT_NAME));
    }

    public static String status() {
        return capture("systemctl", "status", UNIT_NAME, "--no-pager");
    }

    /** Writes a default configuration for a system install when none exists. */
    public static Path ensureSystemConfig(Path configPath, Path modelsDir, Path logDir)
            throws IOException {
        if (Files.exists(configPath)) {
            return configPath;
        }
        Files.createDirectories(configPath.getParent());
        String content = """
                server:
                  host: 0.0.0.0
                  port: 28780
                web_server:
                  enabled: true
                  auto_start: true
                  port: 28880
                runtime:
                  preferred_backend: cpu
                  allow_cuda_fallback: true
                  model_directories: []
                  default_max_tokens: 256
                storage:
                  data_dir: %s
                  models_dir: %s
                  log_dir: %s
                """.formatted(configPath.getParent().toString(), modelsDir.toString(),
                logDir.toString());
        Files.writeString(configPath, content);
        return configPath;
    }

    private static void run(String... command) throws IOException {
        try {
            Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
            process.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while running " + String.join(" ", command), e);
        }
    }

    private static String capture(String... command) {
        try {
            Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
            if (!process.waitFor(10, java.util.concurrent.TimeUnit.SECONDS)) {
                process.destroyForcibly();
            }
            return new String(process.getInputStream().readAllBytes());
        } catch (Exception e) {
            return "systemctl unavailable: " + e.getMessage();
        }
    }

    public static List<String> checkEnvironment() {
        List<String> problems = new ArrayList<>();
        if (!Files.isDirectory(Path.of(DEFAULT_SERVICE_DIR))) {
            problems.add("Directory missing: " + DEFAULT_SERVICE_DIR + " (not a systemd host?)");
        }
        try {
            Process process = new ProcessBuilder("systemctl", "--version").redirectErrorStream(true).start();
            if (!process.waitFor(5, java.util.concurrent.TimeUnit.SECONDS) || process.exitValue() != 0) {
                problems.add("systemctl not available");
            }
        } catch (Exception e) {
            problems.add("systemctl not available: " + e.getMessage());
        }
        return problems;
    }

    public static Path defaultJarPath() {
        return Path.of("enr-cli", "target", "enr-cli-2.6.11-SNAPSHOT.jar").toAbsolutePath();
    }

    public static Path currentJavaBinary() {
        return Path.of(System.getProperty("java.home"), "bin", "java").toAbsolutePath();
    }

    // ------------------------------------------------------------ user-level (systemctl --user)

    public static final String USER_UNIT_DIR =
            Path.of(System.getProperty("user.home"), ".config", "systemd", "user").toString();

    /**
     * User-level unit (no root). Uses the user's own data directory (~/.enr)
     * so a user-level service and direct terminal mode share the same model
     * registry, configuration and storage.
     */
    public static String unitFileUser(String javaBinary, String jarPath, String logDir) {
        return """
                [Unit]
                Description=ENR Server — Native AI Model Inference Runtime & Web Control Center
                After=network-online.target
                Wants=network-online.target

                [Service]
                Type=simple
                ExecStart=%s --enable-native-access=ALL-UNNAMED -jar %s serve
                Restart=on-failure
                RestartSec=5
                TimeoutStopSec=60
                KillSignal=SIGTERM
                SuccessExitStatus=143
                LimitNOFILE=65536
                StandardOutput=append:%s/enr-server.out.log
                StandardError=append:%s/enr-server.err.log

                [Install]
                WantedBy=default.target
                """.formatted(javaBinary, jarPath, logDir, logDir);
    }

    public static void installUser(Path javaBinary, Path jarPath, Path logDir) throws IOException {
        Files.createDirectories(Path.of(USER_UNIT_DIR));
        Files.writeString(Path.of(USER_UNIT_DIR, UNIT_NAME),
                unitFileUser(javaBinary.toAbsolutePath().toString(),
                        jarPath.toAbsolutePath().toString(),
                        logDir.toAbsolutePath().toString()));
        run("systemctl", "--user", "daemon-reload");
        run("systemctl", "--user", "enable", UNIT_NAME);
    }

    public static boolean installedUser() {
        return Files.exists(Path.of(USER_UNIT_DIR, UNIT_NAME));
    }

    public static void uninstallUser() throws IOException {
        run("systemctl", "--user", "disable", "--now", UNIT_NAME);
        Files.deleteIfExists(Path.of(USER_UNIT_DIR, UNIT_NAME));
        run("systemctl", "--user", "daemon-reload");
    }

    // ------------------------------------------------------------ service control (system + user)

    public static void startSystem() throws IOException {
        runChecked("systemctl", "start", UNIT_NAME);
    }

    public static void startUser() throws IOException {
        runChecked("systemctl", "--user", "start", UNIT_NAME);
    }

    public static void stopSystem() throws IOException {
        runChecked("systemctl", "stop", UNIT_NAME);
    }

    public static void stopUser() throws IOException {
        runChecked("systemctl", "--user", "stop", UNIT_NAME);
    }

    public static void restartSystem() throws IOException {
        runChecked("systemctl", "restart", UNIT_NAME);
    }

    public static void restartUser() throws IOException {
        runChecked("systemctl", "--user", "restart", UNIT_NAME);
    }

    public static String statusSystem() {
        return capture("systemctl", "status", UNIT_NAME, "--no-pager");
    }

    public static String statusUser() {
        return capture("systemctl", "--user", "status", UNIT_NAME, "--no-pager");
    }

    public static String logsSystem(int lines) {
        return capture("journalctl", "-u", UNIT_NAME, "-n", String.valueOf(lines), "--no-pager");
    }

    public static String logsUser(int lines) {
        return capture("journalctl", "--user", "-u", UNIT_NAME, "-n", String.valueOf(lines), "--no-pager");
    }

    /** Runs a command and throws on a non-zero exit, returning the combined output. */
    private static void runChecked(String... command) throws IOException {
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
