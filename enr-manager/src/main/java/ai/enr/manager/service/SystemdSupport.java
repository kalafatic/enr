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
                  port: 8111
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
}
