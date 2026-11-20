package ai.enr.manager.logs;

import java.time.Instant;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;

/**
 * In-memory ring buffer of log records, also installable as a JUL handler.
 * Backs the diagnostics log endpoints and keeps the most recent records only.
 */
public final class LogBuffer {

    public record Entry(long timestampMs, String level, String logger, String message) {
    }

    private final int capacity;
    private final Deque<Entry> entries = new ArrayDeque<>();
    private final Handler handler;

    public LogBuffer(int capacity) {
        this.capacity = capacity;
        this.handler = new Handler() {
            private final java.util.logging.SimpleFormatter formatter = new java.util.logging.SimpleFormatter();

            @Override
            public void publish(LogRecord record) {
                append(new Entry(record.getMillis(),
                        record.getLevel() == null ? "INFO" : record.getLevel().getName(),
                        record.getLoggerName() == null ? "" : record.getLoggerName(),
                        formatter.formatMessage(record)));
            }

            @Override
            public void flush() {
            }

            @Override
            public void close() {
            }
        };
        handler.setLevel(Level.ALL);
    }

    public synchronized void append(Entry entry) {
        entries.addLast(entry);
        while (entries.size() > capacity) {
            entries.removeFirst();
        }
    }

    public synchronized void info(String logger, String message) {
        append(new Entry(System.currentTimeMillis(), "INFO", logger, message));
    }

    public synchronized void warn(String logger, String message) {
        append(new Entry(System.currentTimeMillis(), "WARNING", logger, message));
    }

    public synchronized void error(String logger, String message) {
        append(new Entry(System.currentTimeMillis(), "SEVERE", logger, message));
    }

    /** Returns the most recent {@code limit} entries (or all, when limit &lt;= 0). */
    public synchronized List<Entry> recent(int limit) {
        List<Entry> all = new ArrayList<>(entries);
        if (limit > 0 && all.size() > limit) {
            return all.subList(all.size() - limit, all.size());
        }
        return all;
    }

    public synchronized int size() {
        return entries.size();
    }

    public Handler handler() {
        return handler;
    }
}
