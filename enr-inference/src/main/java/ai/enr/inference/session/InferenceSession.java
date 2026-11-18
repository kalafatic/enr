package ai.enr.inference.session;

import ai.enr.inference.KVCache;
import ai.enr.inference.TransformerConfig;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * An inference session: independent context, KV cache and generation state.
 * Sessions are independent of model instances and never share context.
 */
public final class InferenceSession {

    private final String id;
    private final String modelId;
    private final TransformerConfig config;
    private final KVCache kvCache;
    private final List<Integer> contextTokens;
    private final long createdAt = System.currentTimeMillis();
    private long lastActiveAt = createdAt;
    private volatile boolean cancelled;

    public InferenceSession(String modelId, TransformerConfig config) {
        this.id = UUID.randomUUID().toString();
        this.modelId = modelId;
        this.config = config;
        this.kvCache = new KVCache(config.numBlocks(), config.numKvHeads(),
                config.headDim(), config.maxSeqLen());
        this.contextTokens = new ArrayList<>();
    }

    public InferenceSession(String id, String modelId, TransformerConfig config) {
        this.id = id;
        this.modelId = modelId;
        this.config = config;
        this.kvCache = new KVCache(config.numBlocks(), config.numKvHeads(),
                config.headDim(), config.maxSeqLen());
        this.contextTokens = new ArrayList<>();
    }

    public String id() {
        return id;
    }

    public String modelId() {
        return modelId;
    }

    public KVCache kvCache() {
        return kvCache;
    }

    public List<Integer> contextTokens() {
        return contextTokens;
    }

    public int contextLength() {
        return contextTokens.size();
    }

    public int remainingContext() {
        return config.maxSeqLen() - contextTokens.size();
    }

    public boolean contextExhausted() {
        return contextTokens.size() >= config.maxSeqLen();
    }

    public void touch() {
        lastActiveAt = System.currentTimeMillis();
    }

    public long lastActiveAt() {
        return lastActiveAt;
    }

    public long createdAt() {
        return createdAt;
    }

    public void cancel() {
        cancelled = true;
    }

    public boolean isCancelled() {
        return cancelled;
    }

    public void reset() {
        kvCache.clear();
        contextTokens.clear();
        cancelled = false;
        touch();
    }
}
