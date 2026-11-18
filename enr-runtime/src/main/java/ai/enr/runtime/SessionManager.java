package ai.enr.runtime;

import ai.enr.inference.session.InferenceSession;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Session manager: creates and tracks independent inference sessions,
 * enforces global concurrency limits and idle timeout eviction.
 */
public final class SessionManager {

    private final RuntimeConfig config;
    private final ConcurrentHashMap<String, InferenceSession> sessions = new ConcurrentHashMap<>();
    private final AtomicInteger liveSessions = new AtomicInteger();

    public SessionManager(RuntimeConfig config) {
        this.config = config;
    }

    public InferenceSession create(String modelId, ModelInstance model) {
        if (liveSessions.get() >= config.maxConcurrentSessions()) {
            throw new IllegalStateException("Concurrent session limit reached: "
                    + config.maxConcurrentSessions());
        }
        InferenceSession session = new InferenceSession(modelId, model.transformer().config());
        sessions.put(session.id(), session);
        liveSessions.incrementAndGet();
        model.beginSession();
        return session;
    }

    public Optional<InferenceSession> find(String sessionId) {
        return Optional.ofNullable(sessions.get(sessionId));
    }

    public InferenceSession require(String sessionId) {
        return find(sessionId)
                .orElseThrow(() -> new IllegalArgumentException("Session not found: " + sessionId));
    }

    public void remove(String sessionId, ModelManager models) {
        InferenceSession removed = sessions.remove(sessionId);
        if (removed != null) {
            liveSessions.decrementAndGet();
            ModelInstance model = models.get(removed.modelId());
            if (model != null) {
                model.endSession();
            }
        }
    }

    public List<InferenceSession> list() {
        return sessions.values().stream()
                .sorted(Comparator.comparingLong(InferenceSession::createdAt))
                .toList();
    }

    public List<InferenceSession> listForModel(String modelId) {
        return sessions.values().stream()
                .filter(s -> s.modelId().equals(modelId))
                .toList();
    }

    public int liveCount() {
        return liveSessions.get();
    }

    public int totalCount() {
        return sessions.size();
    }

    /**
     * Evicts sessions idle longer than the configured timeout. Returns evicted ids.
     */
    public List<String> evictIdle(ModelManager models) {
        long now = System.currentTimeMillis();
        long timeoutMs = config.sessionIdleTimeoutSeconds() * 1000L;
        List<String> evicted = sessions.values().stream()
                .filter(s -> now - s.lastActiveAt() > timeoutMs)
                .map(InferenceSession::id)
                .toList();
        for (String id : evicted) {
            remove(id, models);
        }
        return evicted;
    }

    public Map<String, InferenceSession> all() {
        return new ConcurrentHashMap<>(sessions);
    }
}
