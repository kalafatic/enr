package ai.enr.loader.api;

import ai.enr.core.gguf.GgufQuant;
import ai.enr.core.io.EndianIO;
import ai.enr.core.io.Hex;
import ai.enr.core.tensor.QuantizedWeight;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.zip.GZIPInputStream;
import java.io.ByteArrayInputStream;

public final class TensorStores {

    private TensorStores() {
    }

    public static TensorStore eager(float[] data) {
        return new EagerStore(data);
    }

    /**
     * Lazily materialized store backed by a supplier. The supplier is invoked at
     * most once, on the first {@link TensorStore#loadFloats()} call, and the result
     * is cached. Used for dequantized GGUF tensors, mirroring the lazy behaviour of
     * the memory-mapped F32 store.
     */
    public static TensorStore lazy(java.util.function.Supplier<float[]> loader) {
        return new LazyStore(loader);
    }

    public static TensorStore gzip(byte[] compressed, long uncompressedSize) {
        return new GzipStore(compressed, (int) uncompressedSize);
    }

    public static TensorStore mmapLittleEndian(Path file, long dataOffset, long elementCount, DTypeLe dtype) {
        return new MmapLeStore(file, dataOffset, elementCount, dtype);
    }

    /**
     * A store that keeps a quantized GGUF tensor in its original packed form and
     * exposes it as a {@link QuantizedWeight} for block-wise computation. The
     * weight is laid out {@code [rows][cols]} (output-major). {@link TensorStore#loadFloats()}
     * performs a full dequantization in the canonical (transposed, if requested)
     * layout, matching the F32 path.
     */
    public static TensorStore quantized(Path file, long dataOffset, int ggmlType, long elementCount,
                                        int rows, int cols, boolean transpose) {
        return new QuantStore(file, dataOffset, ggmlType, elementCount, rows, cols, transpose);
    }

    public static TensorStore transpose(TensorStore source, long rows, long cols) {
        return new TransposeStore(source, rows, cols);
    }

    public enum DTypeLe {
        F32
    }

    private record EagerStore(float[] data) implements TensorStore {
        @Override
        public float[] loadFloats() {
            return data.clone();
        }

        @Override
        public boolean isLoaded() {
            return true;
        }
    }

    private static final class LazyStore implements TensorStore {
        private final java.util.function.Supplier<float[]> loader;
        private volatile float[] cache;

        LazyStore(java.util.function.Supplier<float[]> loader) {
            this.loader = loader;
        }

        @Override
        public float[] loadFloats() {
            float[] local = cache;
            if (local != null) {
                return local.clone();
            }
            synchronized (this) {
                if (cache == null) {
                    cache = loader.get();
                }
                return cache.clone();
            }
        }

        @Override
        public boolean isLoaded() {
            return cache != null;
        }
    }

    private static final class GzipStore implements TensorStore {
        private final byte[] compressed;
        private final int uncompressedSize;
        private volatile float[] cache;

        GzipStore(byte[] compressed, int uncompressedSize) {
            this.compressed = compressed;
            this.uncompressedSize = uncompressedSize;
        }

        @Override
        public float[] loadFloats() {
            float[] local = cache;
            if (local != null) {
                return local.clone();
            }
            synchronized (this) {
                if (cache == null) {
                    try (GZIPInputStream gzis = new GZIPInputStream(new ByteArrayInputStream(compressed))) {
                        byte[] raw = gzis.readNBytes(uncompressedSize);
                        if (raw.length != uncompressedSize) {
                            throw new IllegalStateException("GZIP tensor payload truncated: expected "
                                    + uncompressedSize + " bytes, got " + raw.length);
                        }
                        cache = EndianIO.bigEndianToFloatArray(raw);
                    } catch (IOException e) {
                        throw new IllegalStateException("Failed to decompress tensor payload", e);
                    }
                }
                return cache.clone();
            }
        }

        @Override
        public boolean isLoaded() {
            return cache != null;
        }
    }

    private static final class TransposeStore implements TensorStore {
        private final TensorStore source;
        private final long rows;
        private final long cols;
        private volatile float[] cache;

        TransposeStore(TensorStore source, long rows, long cols) {
            this.source = source;
            this.rows = rows;
            this.cols = cols;
        }

        @Override
        public float[] loadFloats() {
            float[] local = cache;
            if (local != null) {
                return local.clone();
            }
            synchronized (this) {
                if (cache == null) {
                    float[] src = source.loadFloats();
                    float[] out = new float[src.length];
                    int r = (int) rows;
                    int c = (int) cols;
                    for (int i = 0; i < r; i++) {
                        for (int j = 0; j < c; j++) {
                            out[j * r + i] = src[i * c + j];
                        }
                    }
                    cache = out;
                }
                return cache.clone();
            }
        }

        @Override
        public boolean isLoaded() {
            return cache != null;
        }
    }

    private static final class QuantStore implements TensorStore {
        private final Path file;
        private final long dataOffset;
        private final int ggmlType;
        private final long elementCount;
        private final int rows;
        private final int cols;
        private final boolean transpose;
        private volatile QuantizedWeight cached;

        QuantStore(Path file, long dataOffset, int ggmlType, long elementCount, int rows, int cols,
                   boolean transpose) {
            this.file = file;
            this.dataOffset = dataOffset;
            this.ggmlType = ggmlType;
            this.elementCount = elementCount;
            this.rows = rows;
            this.cols = cols;
            this.transpose = transpose;
        }

        @Override
        public QuantizedWeight quantizedWeight() {
            QuantizedWeight local = cached;
            if (local != null) {
                return local;
            }
            synchronized (this) {
                if (cached == null) {
                    byte[] raw = GgufQuant.readRaw(file, dataOffset, ggmlType, elementCount);
                    cached = new QuantizedWeight(ggmlType, rows, cols, raw);
                }
                return cached;
            }
        }

        @Override
        public float[] loadFloats() {
            float[] raw = quantizedWeight().dequantize();
            if (!transpose) {
                return raw;
            }
            float[] canonical = new float[raw.length];
            for (int c = 0; c < rows; c++) {
                for (int j = 0; j < cols; j++) {
                    canonical[j * rows + c] = raw[c * cols + j];
                }
            }
            return canonical;
        }

        @Override
        public boolean isLoaded() {
            return cached != null;
        }
    }

    private static final class MmapLeStore implements TensorStore {
        private final Path file;
        private final long dataOffset;
        private final long elementCount;
        private final DTypeLe dtype;
        private volatile float[] cache;

        MmapLeStore(Path file, long dataOffset, long elementCount, DTypeLe dtype) {
            this.file = file;
            this.dataOffset = dataOffset;
            this.elementCount = elementCount;
            this.dtype = dtype;
        }

        @Override
        public float[] loadFloats() {
            float[] local = cache;
            if (local != null) {
                return local.clone();
            }
            synchronized (this) {
                if (cache == null) {
                    cache = readFloats();
                }
                return cache.clone();
            }
        }

        private float[] readFloats() {
            if (dtype != DTypeLe.F32) {
                throw new IllegalStateException("Unsupported mmap dtype: " + dtype);
            }
            long byteLength = elementCount * 4L;
            if (byteLength > Integer.MAX_VALUE - 8) {
                throw new IllegalStateException("Tensor too large for float[] storage: " + elementCount);
            }
            try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
                MappedByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, dataOffset, byteLength);
                ByteBuffer le = buffer.order(java.nio.ByteOrder.LITTLE_ENDIAN);
                float[] out = new float[(int) elementCount];
                le.asFloatBuffer().get(out);
                return out;
            } catch (IOException e) {
                throw new IllegalStateException("Failed to memory-map tensor data from " + file, e);
            }
        }

        @Override
        public boolean isLoaded() {
            return cache != null;
        }
    }

    public static String checksumOfFloats(float[] data) {
        return Hex.encode(ai.enr.core.io.Hashing.sha256(EndianIO.floatArrayToBigEndian(data)));
    }

    public static boolean fileExists(Path path) {
        return Files.exists(path);
    }
}
