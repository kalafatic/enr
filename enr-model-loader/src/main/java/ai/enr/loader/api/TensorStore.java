package ai.enr.loader.api;

import ai.enr.core.tensor.QuantizedWeight;

public interface TensorStore {

    float[] loadFloats();

    boolean isLoaded();

    /**
     * Returns the underlying quantized weight if this store keeps the tensor in
     * its original GGUF quantized form, otherwise {@code null}. Quantized stores
     * still support {@link #loadFloats()} (full dequantization) as a reference.
     */
    default QuantizedWeight quantizedWeight() {
        return null;
    }
}
