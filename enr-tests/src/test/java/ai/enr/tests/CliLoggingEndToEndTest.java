package ai.enr.tests;

import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * End-to-end CLI logging-control tests: launch the shaded CLI jar with the
 * {@code --logging} / {@code --no-logging} flags and assert on exit code and
 * output. Skipped when the shaded jar has not been built (plain {@code mvn test}).
 */
class CliLoggingEndToEndTest {

    private static final Path JAR =
            Path.of("../enr-cli/target/enr-cli-2.6.11-SNAPSHOT.jar").toAbsolutePath().normalize();

    private record Result(int exit, String output) {
    }

    private Result run(String... args) throws Exception {
        String java = Path.of(System.getProperty("java.home"), "bin", "java").toString();
        List<String> command = new ArrayList<>();
        command.add(java);
        command.add("--enable-native-access=ALL-UNNAMED");
        command.add("-jar");
        command.add(JAR.toString());
        command.addAll(List.of(args));
        Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
        String output = new String(process.getInputStream().readAllBytes());
        int exit = process.waitFor();
        return new Result(exit, output);
    }

    @Test
    void conflictingFlagsAreRejected() throws Exception {
        assumeTrue(Files.exists(JAR), "shaded jar not built");
        Result result = run("--logging", "--no-logging", "version");
        assertNotEquals(0, result.exit, result.output);
        assertTrue(result.output.contains("mutually exclusive"), result.output);
    }

    @Test
    void helpDocumentsBothFlags() throws Exception {
        assumeTrue(Files.exists(JAR), "shaded jar not built");
        Result result = run("--help");
        assertEquals(0, result.exit, result.output);
        assertTrue(result.output.contains("--logging"), result.output);
        assertTrue(result.output.contains("--no-logging"), result.output);
    }

    @Test
    void noLoggingPreservesNormalCommandOutput() throws Exception {
        assumeTrue(Files.exists(JAR), "shaded jar not built");
        Result result = run("--no-logging", "--help");
        assertEquals(0, result.exit, result.output);
        assertTrue(result.output.contains("ENR"), result.output);
        assertTrue(result.output.contains("--logging"), result.output);
    }

    @Test
    void errorsRemainVisibleWhenLoggingDisabled() throws Exception {
        assumeTrue(Files.exists(JAR), "shaded jar not built");
        Result result = run("--no-logging", "definitely-not-a-command");
        assertNotEquals(0, result.exit, result.output);
        assertTrue(result.output.contains("Unknown command"), result.output);
    }

    @Test
    void loggingFlagIsStrippedFromCommandDispatch() throws Exception {
        assumeTrue(Files.exists(JAR), "shaded jar not built");
        // --logging must not be treated as a command name
        Result result = run("--logging", "--help");
        assertEquals(0, result.exit, result.output);
        assertTrue(result.output.contains("Global options"), result.output);
    }
}
