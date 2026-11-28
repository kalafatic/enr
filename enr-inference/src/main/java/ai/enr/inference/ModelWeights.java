package ai.enr.inference;

import ai.enr.core.tensor.FloatTensor;
import ai.enr.loader.api.LoadedModel;
import ai.enr.loader.api.TensorEntry;

import java.util.HashMap;
import java.util.Map;

/**
 * Lazily materialized canonical weight registry bound to a LoadedModel.
 * Weights are converted to canonical EVO layout by the loader; this class
 * caches FloatTensor views for the inference engine.
 */
public final class ModelWeights {

    private final LoadedModel model;
    private final Map<String, FloatTensor> cache = new HashMap<>();

    public ModelWeights(LoadedModel model) {
        this.model = model;
    }

    public FloatTensor get(String name) {
        FloatTensor cached = cache.get(name);
        if (cached != null) {
            return cached;
        }
        synchronized (cache) {
            cached = cache.get(name);
            if (cached != null) {
                return cached;
            }
            TensorEntry entry = model.tensor(name).orElseThrow(
                    () -> new IllegalStateException("Missing tensor: " + name + " in model " + model.name()));
            float[] data = entry.loadFloats();
            FloatTensor tensor = new FloatTensor(entry.shape(), data);
            cache.put(name, tensor);
            return tensor;
        }
    }

    /**
     * Returns the quantized weight for a tensor if it is stored in its original
     * GGUF quantized form, otherwise {@code null} (F32 path). Quantized weights
     * are not F32-expanded and are not cached here.
     */
    public ai.enr.core.tensor.QuantizedWeight quantizedWeight(String name) {
        TensorEntry entry = model.tensor(name).orElseThrow(
                () -> new IllegalStateException("Missing tensor: " + name + " in model " + model.name()));
        return entry.quantizedWeight();
    }

    public LoadedModel model() {
        return model;
    }

    public void release(String name) {
        synchronized (cache) {
            cache.remove(name);
        }
    }

    public int loadedTensorCount() {
        synchronized (cache) {
            return cache.size();
        }
    }

    public long loadedBytes() {
        synchronized (cache) {
            long total = 0;
            for (FloatTensor t : cache.values()) {
                total += t.size() * 4L;
            }
            return total;
        }
    }
}
