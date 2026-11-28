package ai.enr.inference;

/**
 * Per-session KV cache with dynamic, geometric growth. Session-private: one
 * instance per generation session, never shared between concurrent requests.
 *
 * <p>Storage is allocated lazily and grows with the actual sequence length, not
 * the model's theoretical maximum context window. Each layer starts unallocated;
 * on the first append a small initial capacity is reserved, and when the running
 * sequence outgrows it the cache doubles (or jumps directly to the required size
 * for a large prefill), copying preserved K/V values. Layout per layer:
 * {@code [kvSeqLen][kvHead][headDim]}.</p>
 */
public final class KVCache {

    private static final int DEFAULT_INITIAL_CAPACITY = 256;
    private static final float DEFAULT_GROWTH_FACTOR = 2.0f;

    private final int numLayers;
    private final int numKvHeads;
    private final int headDim;
    private final int maxCapacity;
    private final int initialCapacity;
    private final float growthFactor;
    private final float[][] keyCaches;
    private final float[][] valueCaches;
    private int size;
    private int capacity;

    public KVCache(int numLayers, int numKvHeads, int headDim, int maxCapacity) {
        this(numLayers, numKvHeads, headDim, maxCapacity,
                Math.min(maxCapacity, DEFAULT_INITIAL_CAPACITY), DEFAULT_GROWTH_FACTOR);
    }

    public KVCache(int numLayers, int numKvHeads, int headDim, int maxCapacity, int initialCapacity) {
        this(numLayers, numKvHeads, headDim, maxCapacity, initialCapacity, DEFAULT_GROWTH_FACTOR);
    }

    public KVCache(int numLayers, int numKvHeads, int headDim, int maxCapacity,
                   int initialCapacity, float growthFactor) {
        if (numLayers <= 0 || numKvHeads <= 0 || headDim <= 0 || maxCapacity <= 0) {
            throw new IllegalArgumentException("Invalid KV cache dimensions");
        }
        this.numLayers = numLayers;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.maxCapacity = maxCapacity;
        this.initialCapacity = Math.max(1, Math.min(initialCapacity, maxCapacity));
        this.growthFactor = growthFactor <= 1.0f ? DEFAULT_GROWTH_FACTOR : growthFactor;
        this.keyCaches = new float[numLayers][];
        this.valueCaches = new float[numLayers][];
        this.size = 0;
        this.capacity = 0;
    }

    /** Number of tokens currently stored. */
    public int size() {
        return size;
    }

    /** Currently allocated token capacity (0 until the first append). */
    public int capacity() {
        return capacity;
    }

    /** Hard upper bound on token capacity (the configured context window). */
    public int maxCapacity() {
        return maxCapacity;
    }

    public boolean isEmpty() {
        return size == 0;
    }

    /**
     * Appends {@code tokenCount} tokens' keys/values for the given layer,
     * growing storage as needed. Layout per layer: [kvSeqLen][kvHead][headDim].
     */
    public void append(int layer, float[] keys, float[] values, int tokenCount) {
        int required = size + tokenCount;
        if (required > maxCapacity) {
            throw new IllegalStateException("KV cache capacity exceeded: required " + required
                    + " tokens, configured max " + maxCapacity);
        }
        if (required > capacity) {
            grow(required);
        }
        float[] keyCache = keyCaches[layer];
        if (keyCache == null) {
            keyCache = new float[numKvHeads * capacity * headDim];
            keyCaches[layer] = keyCache;
            valueCaches[layer] = new float[numKvHeads * capacity * headDim];
        }
        System.arraycopy(keys, 0, keyCache, size * numKvHeads * headDim, keys.length);
        System.arraycopy(values, 0, valueCaches[layer], size * numKvHeads * headDim, values.length);
        if (layer == numLayers - 1) {
            size += tokenCount;
        }
    }

    /**
     * Grows all allocated layers to the new capacity, preserving existing K/V
     * values. The old buffers are dropped (eligible for GC) so the complete
     * cache is never duplicated beyond the transient copy window.
     */
    private void grow(int required) {
        int newCapacity = computeNewCapacity(required);
        int oldElements = numKvHeads * capacity * headDim;
        int newElements = numKvHeads * newCapacity * headDim;
        try {
            for (int layer = 0; layer < numLayers; layer++) {
                float[] oldK = keyCaches[layer];
                if (oldK == null) {
                    continue;
                }
                float[] oldV = valueCaches[layer];
                float[] newK = new float[newElements];
                float[] newV = new float[newElements];
                System.arraycopy(oldK, 0, newK, 0, oldElements);
                System.arraycopy(oldV, 0, newV, 0, oldElements);
                keyCaches[layer] = newK;
                valueCaches[layer] = newV;
            }
        } catch (OutOfMemoryError e) {
            throw new IllegalStateException("KV cache growth to " + newCapacity
                    + " tokens failed (out of memory); required " + required, e);
        }
        capacity = newCapacity;
    }

    /** Geometric growth: double (or jump to the required size for a large prefill). */
    private int computeNewCapacity(int required) {
        int grown = (int) Math.ceil(capacity * growthFactor);
        if (grown < initialCapacity) {
            grown = initialCapacity;
        }
        if (grown < required) {
            grown = required;
        }
        return Math.min(grown, maxCapacity);
    }

    public float[] keyCache(int layer) {
        return keyCaches[layer];
    }

    public float[] valueCache(int layer) {
        return valueCaches[layer];
    }

    /** Releases all buffers and resets the sequence and capacity. */
    public void clear() {
        for (int i = 0; i < numLayers; i++) {
            keyCaches[i] = null;
            valueCaches[i] = null;
        }
        size = 0;
        capacity = 0;
    }

    /** Bytes currently allocated for key and value storage. */
    public long allocatedBytes() {
        return (long) numLayers * 2L * numKvHeads * capacity * headDim * 4L;
    }

    /** Bytes that would be allocated at the configured maximum context window. */
    public long maxBytes() {
        return (long) numLayers * 2L * numKvHeads * maxCapacity * headDim * 4L;
    }

    /** @deprecated use {@link #allocatedBytes()} for the live footprint. */
    @Deprecated
    public long estimatedBytes() {
        return allocatedBytes();
    }
}
