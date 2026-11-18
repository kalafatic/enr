package ai.enr.inference;

/**
 * Per-session KV cache. Session-private: one instance per generation session,
 * never shared between concurrent requests.
 */
public final class KVCache {

    private final int numLayers;
    private final int numKvHeads;
    private final int headDim;
    private final int maxCapacity;
    private final float[][] keyCaches;
    private final float[][] valueCaches;
    private int size;

    public KVCache(int numLayers, int numKvHeads, int headDim, int maxCapacity) {
        this.numLayers = numLayers;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.maxCapacity = maxCapacity;
        this.keyCaches = new float[numLayers][];
        this.valueCaches = new float[numLayers][];
    }

    public int size() {
        return size;
    }

    public int capacity() {
        return maxCapacity;
    }

    public boolean isEmpty() {
        return size == 0;
    }

    /**
     * Appends {@code tokenCount} tokens' keys/values for the given layer.
     * Layout per layer: [kvHead][kvSeqLen][headDim].
     */
    public void append(int layer, float[] keys, float[] values, int tokenCount) {
        if (size + tokenCount > maxCapacity) {
            throw new IllegalStateException("KV cache capacity exceeded: " + maxCapacity);
        }
        float[] keyCache = keyCaches[layer];
        if (keyCache == null) {
            keyCache = new float[numKvHeads * maxCapacity * headDim];
            keyCaches[layer] = keyCache;
            valueCaches[layer] = new float[numKvHeads * maxCapacity * headDim];
        }
        System.arraycopy(keys, 0, keyCache, size * numKvHeads * headDim, keys.length);
        System.arraycopy(values, 0, valueCaches[layer], size * numKvHeads * headDim, values.length);
        if (layer == numLayers - 1) {
            size += tokenCount;
        }
    }

    public float[] keyCache(int layer) {
        return keyCaches[layer];
    }

    public float[] valueCache(int layer) {
        return valueCaches[layer];
    }

    public void clear() {
        for (int i = 0; i < numLayers; i++) {
            keyCaches[i] = null;
            valueCaches[i] = null;
        }
        size = 0;
    }

    public long estimatedBytes() {
        long perLayer = (long) numKvHeads * maxCapacity * headDim * 4L;
        return perLayer * 2L * numLayers;
    }
}
