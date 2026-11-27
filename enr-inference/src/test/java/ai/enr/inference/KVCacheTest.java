package ai.enr.inference;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Dynamic KV-cache growth and data-preservation tests.
 */
class KVCacheTest {

    private static final int HEADS = 2;
    private static final int HEAD_DIM = 4;

    @Test
    void startsSmallAndGrowsGeometrically() {
        KVCache kv = new KVCache(1, HEADS, HEAD_DIM, 1000, 16);
        assertEquals(0, kv.capacity(), "no allocation before first append");
        assertEquals(0, kv.allocatedBytes());

        append(kv, 10, 0); // prefill -> initial capacity 16
        assertEquals(10, kv.size());
        assertEquals(16, kv.capacity(), "initial capacity reserved on first append");

        append(kv, 10, 10); // size 20 > 16 -> grow to 32
        assertEquals(20, kv.size());
        assertEquals(32, kv.capacity());

        append(kv, 13, 20); // size 33 > 32 -> grow to 64
        assertEquals(33, kv.size());
        assertEquals(64, kv.capacity());
    }

    @Test
    void preservesValuesAcrossGrowth() {
        KVCache kv = new KVCache(1, HEADS, HEAD_DIM, 1000, 4);
        float[] first = append(kv, 4, 0);
        append(kv, 4, 4); // size 8 > 4 -> grow to 8
        assertPosition(kv, 0, first);
        assertEquals(8, kv.size());
    }

    @Test
    void preservesAllValuesAcrossMultipleBoundaries() {
        KVCache kv = new KVCache(1, HEADS, HEAD_DIM, 1000, 8);
        Random rnd = new Random(99);
        List<float[]> batches = new ArrayList<>();
        int total = 0;
        while (total < 130) {
            int n = Math.min(1 + rnd.nextInt(7), 130 - total);
            batches.add(append(kv, n, total));
            total += n;
        }
        assertEquals(130, kv.size());
        assertTrue(kv.capacity() < 1000, "should not allocate the full context window");

        float[] cache = kv.keyCache(0);
        int pos = 0;
        for (float[] batch : batches) {
            int tokens = batch.length / (HEADS * HEAD_DIM);
            for (int i = 0; i < batch.length; i++) {
                assertEquals(batch[i], cache[(pos + i / (HEADS * HEAD_DIM)) * HEADS * HEAD_DIM
                        + (i / HEAD_DIM) % HEADS * HEAD_DIM + i % HEAD_DIM], 1e-6f);
            }
            pos += tokens;
        }
    }

    @Test
    void multiLayerGrowthPreservesEachLayer() {
        KVCache kv = new KVCache(3, HEADS, HEAD_DIM, 1000, 4);
        float[][] first = appendAllLayers(kv, 4, 0);
        appendAllLayers(kv, 4, 4); // grow 4 -> 8 across all 3 layers
        assertEquals(8, kv.size());
        assertEquals(8, kv.capacity());
        for (int layer = 0; layer < 3; layer++) {
            float[] cache = kv.keyCache(layer);
            for (int i = 0; i < first[layer].length; i++) {
                assertEquals(first[layer][i], cache[(i / (HEADS * HEAD_DIM)) * HEADS * HEAD_DIM
                        + (i / HEAD_DIM) % HEADS * HEAD_DIM + i % HEAD_DIM], 1e-6f);
            }
        }
    }

    @Test
    void clearReleasesBuffers() {
        KVCache kv = new KVCache(1, HEADS, HEAD_DIM, 1000, 16);
        append(kv, 20, 0);
        assertTrue(kv.allocatedBytes() > 0);
        kv.clear();
        assertEquals(0, kv.size());
        assertEquals(0, kv.capacity());
        assertEquals(0, kv.allocatedBytes());
        assertNull(kv.keyCache(0));
        assertNull(kv.valueCache(0));
    }

    @Test
    void exceedsCapacityThrows() {
        KVCache kv = new KVCache(1, HEADS, HEAD_DIM, 16);
        append(kv, 16, 0);
        assertThrows(IllegalStateException.class, () -> append(kv, 1, 16));
    }

    @Test
    void largePrefillJumpsDirectlyToRequiredCapacity() {
        KVCache kv = new KVCache(1, HEADS, HEAD_DIM, 10000, 32);
        append(kv, 4096, 0);
        assertEquals(4096, kv.capacity(), "large prefill jumps directly to the required size");
        assertEquals(4096, kv.size());
    }

    @Test
    void allocatedBytesReflectsLiveFootprint() {
        KVCache kv = new KVCache(1, HEADS, HEAD_DIM, 1000, 8);
        append(kv, 8, 0);
        assertEquals(1L * 2L * HEADS * 8 * HEAD_DIM * 4L, kv.allocatedBytes());
        assertEquals(1L * 2L * HEADS * 1000 * HEAD_DIM * 4L, kv.maxBytes());
        assertTrue(kv.allocatedBytes() < kv.maxBytes());
    }

    private static float[] append(KVCache kv, int n, int startPos) {
        int perLayer = n * HEADS * HEAD_DIM;
        float[] keys = new float[perLayer];
        float[] values = new float[perLayer];
        for (int i = 0; i < perLayer; i++) {
            keys[i] = startPos * 1000.0f + i + 0.5f;
            values[i] = startPos * 1000.0f + i + 100.5f;
        }
        kv.append(0, keys, values, n);
        return keys;
    }

    /** Appends to every layer, returning the per-layer key arrays for the first layer of each. */
    private static float[][] appendAllLayers(KVCache kv, int n, int startPos) {
        float[][] result = new float[3][];
        int perLayer = n * HEADS * HEAD_DIM;
        for (int layer = 0; layer < 3; layer++) {
            float[] keys = new float[perLayer];
            float[] values = new float[perLayer];
            for (int i = 0; i < perLayer; i++) {
                keys[i] = startPos * 1000.0f + layer * 100000.0f + i + 0.5f;
                values[i] = startPos * 1000.0f + layer * 100000.0f + i + 100.5f;
            }
            kv.append(layer, keys, values, n);
            result[layer] = keys;
        }
        return result;
    }

    private static void assertPosition(KVCache kv, int pos, float[] batch) {
        float[] cache = kv.keyCache(0);
        for (int i = 0; i < batch.length; i++) {
            int head = (i / HEAD_DIM) % HEADS;
            int d = i % HEAD_DIM;
            int tok = i / (HEADS * HEAD_DIM);
            assertEquals(batch[i], cache[(pos + tok) * HEADS * HEAD_DIM + head * HEAD_DIM + d],
                    1e-6f, "position " + (pos + tok));
        }
    }
}
