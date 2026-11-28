package ai.enr.tests;

import ai.enr.manager.logs.LoggingConfig;
import org.junit.jupiter.api.Test;

import java.util.logging.Level;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for command-line logging control ({@link LoggingConfig}):
 * flag parsing, level resolution, conflict detection and flag stripping.
 */
class LoggingConfigTest {

    @Test
    void defaultIsInfo() {
        assertEquals(Level.INFO, LoggingConfig.level(new String[]{"version"}));
        assertEquals(Level.INFO, LoggingConfig.level(new String[0]));
    }

    @Test
    void loggingFlagEnablesInfo() {
        assertEquals(Level.INFO, LoggingConfig.level(new String[]{"--logging", "version"}));
        assertEquals(Level.INFO, LoggingConfig.level(new String[]{"serve", "--logging"}));
    }

    @Test
    void noLoggingFlagDisablesRoutineLogging() {
        assertEquals(Level.SEVERE, LoggingConfig.level(new String[]{"--no-logging", "version"}));
        assertEquals(Level.SEVERE, LoggingConfig.level(new String[]{"serve", "--no-logging"}));
        assertEquals(Level.SEVERE, LoggingConfig.level(new String[]{"--no-logging"}));
    }

    @Test
    void conflictingFlagsDetected() {
        assertTrue(LoggingConfig.conflicting(new String[]{"--logging", "--no-logging"}));
        assertTrue(LoggingConfig.conflicting(new String[]{"--no-logging", "version", "--logging"}));
        assertFalse(LoggingConfig.conflicting(new String[]{"--logging"}));
        assertFalse(LoggingConfig.conflicting(new String[]{"--no-logging"}));
        assertFalse(LoggingConfig.conflicting(new String[]{"version"}));
    }

    @Test
    void stripRemovesBothFlagsAndKeepsArgs() {
        assertArrayEquals(new String[]{"version"},
                LoggingConfig.strip(new String[]{"--logging", "version"}));
        assertArrayEquals(new String[]{"version"},
                LoggingConfig.strip(new String[]{"--no-logging", "version"}));
        assertArrayEquals(new String[]{"serve", "-c", "enr.yaml"},
                LoggingConfig.strip(new String[]{"serve", "-c", "enr.yaml", "--no-logging"}));
        assertArrayEquals(new String[0],
                LoggingConfig.strip(new String[]{"--logging", "--no-logging"}));
        // flags appearing as substrings of other args are not stripped
        assertArrayEquals(new String[]{"--logging-verbose"},
                LoggingConfig.strip(new String[]{"--logging-verbose"}));
    }
}
