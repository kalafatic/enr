package ai.enr.runtime;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Runtime-wide metrics and counters (genuine values only).
 */
public final class Metrics {

    private final AtomicLong totalRequests = new AtomicLong();
    private final AtomicLong totalGeneratedTokens = new AtomicLong();
    private final AtomicLong totalPromptTokens = new AtomicLong();
    private final AtomicLong totalInferenceTimeMs = new AtomicLong();
    private final AtomicLong failedRequests = new AtomicLong();
    private final AtomicLong cancelledRequests = new AtomicLong();
    private final long startedAt = System.currentTimeMillis();

    public void recordRequest(int promptTokens, int generatedTokens, long elapsedMs, boolean failed, boolean cancelled) {
        totalRequests.incrementAndGet();
        totalPromptTokens.addAndGet(promptTokens);
        totalGeneratedTokens.addAndGet(generatedTokens);
        totalInferenceTimeMs.addAndGet(elapsedMs);
        if (failed) {
            failedRequests.incrementAndGet();
        }
        if (cancelled) {
            cancelledRequests.incrementAndGet();
        }
    }

    public long totalRequests() {
        return totalRequests.get();
    }

    public long totalGeneratedTokens() {
        return totalGeneratedTokens.get();
    }

    public long totalPromptTokens() {
        return totalPromptTokens.get();
    }

    public long totalInferenceTimeMs() {
        return totalInferenceTimeMs.get();
    }

    public long failedRequests() {
        return failedRequests.get();
    }

    public long cancelledRequests() {
        return cancelledRequests.get();
    }

    public long uptimeMs() {
        return System.currentTimeMillis() - startedAt;
    }

    public double tokensPerSecond() {
        long ms = totalInferenceTimeMs.get();
        if (ms == 0) {
            return 0;
        }
        return totalGeneratedTokens.get() * 1000.0 / ms;
    }

    public ai.enr.core.json.JsonObject toJson() {
        ai.enr.core.json.JsonObject obj = new ai.enr.core.json.JsonObject();
        obj.put("uptime_ms", uptimeMs());
        obj.put("total_requests", totalRequests());
        obj.put("total_prompt_tokens", totalPromptTokens());
        obj.put("total_generated_tokens", totalGeneratedTokens());
        obj.put("total_inference_time_ms", totalInferenceTimeMs());
        obj.put("failed_requests", failedRequests());
        obj.put("cancelled_requests", cancelledRequests());
        obj.put("tokens_per_second", (double) Math.round(tokensPerSecond() * 100) / 100);
        return obj;
    }
}
