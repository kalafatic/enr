package ai.enr.tests;

import ai.enr.manager.service.SystemdSupport;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Systemd unit generation for the two operating modes: the system-scope unit
 * (root, multi-user.target) and the user-scope unit (no root, default.target).
 * The user unit runs {@code serve} with no explicit config so a user-level
 * service and direct terminal mode share the same ~/.enr data directory.
 */
class SystemdSupportTest {

    @Test
    void userUnitUsesDefaultConfigAndDefaultTarget() {
        String unit = SystemdSupport.unitFileUser("/opt/jdk/bin/java", "/opt/enr/enr.jar", "/home/u/.enr/logs");
        assertTrue(unit.contains("ExecStart=/opt/jdk/bin/java"), unit);
        assertTrue(unit.contains("-jar /opt/enr/enr.jar serve"), unit);
        assertFalse(unit.contains("-c "), "user unit must not force a system config (shared ~/.enr defaults)");
        assertFalse(unit.contains("User="), "user units must not set User=");
        assertTrue(unit.contains("WantedBy=default.target"), unit);
        assertTrue(unit.contains("KillSignal=SIGTERM"), unit);
    }

    @Test
    void systemUnitSetsUserAndMultiUserTarget() {
        String unit = SystemdSupport.unitFile("/opt/jdk/bin/java", "/opt/enr/enr.jar",
                "/etc/enr/enr.yaml", "enr", "enr", "/var/log/enr");
        assertTrue(unit.contains("User=enr"), unit);
        assertTrue(unit.contains("Group=enr"), unit);
        assertTrue(unit.contains("ExecStart=/opt/jdk/bin/java"), unit);
        assertTrue(unit.contains("serve -c /etc/enr/enr.yaml"), unit);
        assertTrue(unit.contains("WantedBy=multi-user.target"), unit);
    }

    @Test
    void unitNameIsStable() {
        assertEquals("enr.service", SystemdSupport.UNIT_NAME);
    }
}
