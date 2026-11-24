package ai.enr.manager.logs;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;

/**
 * Command-line logging control. Resolves the effective {@code java.util.logging}
 * level from the {@code --logging} / {@code --no-logging} flags. Kept free of
 * any CLI dispatch so it can be unit-tested independently.
 *
 * <p>Semantics: {@code --no-logging} suppresses routine application logging
 * (DEBUG/INFO/WARN) while keeping errors (SEVERE) on stderr; {@code --logging}
 * (and the default) enable INFO-level logging. The flags override the default
 * for the current invocation only and never persist configuration.
 */
public final class LoggingConfig {

    public static final String ENABLE_FLAG = "--logging";
    public static final String DISABLE_FLAG = "--no-logging";

    private LoggingConfig() {
    }

    /** Whether {@code --logging} and {@code --no-logging} are both present (invalid). */
    public static boolean conflicting(String[] args) {
        return contains(args, ENABLE_FLAG) && contains(args, DISABLE_FLAG);
    }

    /** The effective root-logger level: SEVERE when disabled, otherwise INFO. */
    public static Level level(String[] args) {
        return contains(args, DISABLE_FLAG) ? Level.SEVERE : Level.INFO;
    }

    /** Removes the logging-control flags, leaving only the real arguments. */
    public static String[] strip(String[] args) {
        List<String> rest = new ArrayList<>();
        for (String arg : args) {
            if (!ENABLE_FLAG.equals(arg) && !DISABLE_FLAG.equals(arg)) {
                rest.add(arg);
            }
        }
        return rest.toArray(new String[0]);
    }

    private static boolean contains(String[] args, String flag) {
        for (String arg : args) {
            if (flag.equals(arg)) {
                return true;
            }
        }
        return false;
    }
}
