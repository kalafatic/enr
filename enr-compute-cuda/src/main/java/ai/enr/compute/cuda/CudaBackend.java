package ai.enr.compute.cuda;

import ai.enr.compute.ComputeBackend;
import ai.enr.compute.DeviceInfo;
import ai.enr.compute.DeviceType;
import ai.enr.compute.cpu.CpuBackend;
import ai.enr.core.tensor.FloatTensor;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;

/**
 * CUDA compute backend: FFM bindings to the CUDA driver API with PTX kernels
 * embedded as resources. Falls back to CPU when the driver or a GPU is not
 * available; exposes genuine device information and VRAM accounting.
 */
public final class CudaBackend implements ComputeBackend {

    private static final Logger LOG = Logger.getLogger(CudaBackend.class.getName());
    private static final int BLOCK = 256;
    private static final int TILE = 16;

    private final CpuBackend cpuFallback = new CpuBackend();
    private final boolean available;
    private final int deviceOrdinal;
    private final String deviceName;
    private final long totalMemory;
    private final String driverVersion;
    private final String unavailableReason;

    private Arena arena;
    private long module;
    private final AtomicLong allocatedBytes = new AtomicLong();
    private final java.util.HashMap<float[], DeviceTensor> deviceCache = new java.util.HashMap<>();
    private final java.util.HashMap<float[], Integer> arraySeen = new java.util.HashMap<>();
    private static final int DEVICE_CACHE_MAX_TENSORS = 1024;
    private static final long DEVICE_CACHE_MAX_BYTES = 1L << 30; // 1 GiB resident cap

    private long fnMatmul;
    private long fnMatmulT;
    private long fnRmsnorm;
    private long fnSiluMul;
    private long fnAdd;
    private long fnMul;
    private long fnSoftmaxMasked;
    private long fnScaleInplace;

    public CudaBackend() {
        boolean ok = false;
        int ordinal = -1;
        String name = "unknown";
        long total = 0;
        String driver = "unknown";
        String reason = null;
        try {
            if (!CudaDriver.available()) {
                reason = "libcuda.so.1 not found";
            } else {
                try (Arena probe = Arena.ofConfined()) {
                    int rc = CudaDriver.cuInit(0);
                    if (rc != CudaDriver.CUDA_SUCCESS) {
                        reason = "cuInit failed: " + rc;
                    } else {
                        driver = String.valueOf(CudaDriver.cuDriverGetVersion(probe));
                        int count = CudaDriver.cuDeviceGetCount(probe);
                        if (count <= 0) {
                            reason = "no CUDA devices";
                        } else {
                            ordinal = CudaDriver.cuDeviceGet(probe, 0);
                            name = CudaDriver.cuDeviceGetName(probe, ordinal);
                            total = CudaDriver.cuDeviceTotalMem(probe, ordinal);
                            ok = true;
                        }
                    }
                }
            }
        } catch (Throwable t) {
            reason = t.getMessage();
        }
        this.available = ok;
        this.deviceOrdinal = ordinal;
        this.deviceName = name;
        this.totalMemory = total;
        this.driverVersion = driver;
        this.unavailableReason = reason;
    }

    @Override
    public String name() {
        return "cuda";
    }

    @Override
    public DeviceInfo deviceInfo() {
        if (!available) {
            return DeviceInfo.unavailable(DeviceType.CUDA, 0, unavailableReason);
        }
        try (Arena probe = Arena.ofConfined()) {
            long[] info = CudaDriver.cuMemGetInfo(probe);
            return new DeviceInfo(deviceOrdinal, DeviceType.CUDA, deviceName,
                    totalMemory, info[0], driverVersion, true);
        } catch (Throwable t) {
            return new DeviceInfo(deviceOrdinal, DeviceType.CUDA, deviceName,
                    totalMemory, 0, driverVersion, true);
        }
    }

    private void ensureReady() {
        if (!available) {
            throw new IllegalStateException("CUDA backend unavailable: " + unavailableReason);
        }
        if (module != 0) {
            return;
        }
        try {
            arena = Arena.ofShared();
            int rc = CudaDriver.cuCtxCreate(arena, deviceOrdinal);
            if (rc != CudaDriver.CUDA_SUCCESS) {
                throw new IllegalStateException("cuCtxCreate failed: " + rc);
            }
            String ptx = loadPtx();
            try (Arena local = Arena.ofConfined()) {
                MemorySegment ptxSeg = local.allocateFrom(ptx, StandardCharsets.UTF_8);
                module = CudaDriver.cuModuleLoadData(arena, ptxSeg);
            }
            fnMatmul = CudaDriver.cuModuleGetFunction(arena, module, "matmul");
            fnMatmulT = CudaDriver.cuModuleGetFunction(arena, module, "matmulT");
            fnRmsnorm = CudaDriver.cuModuleGetFunction(arena, module, "rmsnorm");
            fnSiluMul = CudaDriver.cuModuleGetFunction(arena, module, "silu_mul");
            fnAdd = CudaDriver.cuModuleGetFunction(arena, module, "add");
            fnMul = CudaDriver.cuModuleGetFunction(arena, module, "mul");
            fnSoftmaxMasked = CudaDriver.cuModuleGetFunction(arena, module, "softmax_rows_masked");
            fnScaleInplace = CudaDriver.cuModuleGetFunction(arena, module, "scale_inplace");
        } catch (Throwable t) {
            throw new IllegalStateException("CUDA backend initialization failed: " + t.getMessage(), t);
        }
    }

    private static String loadPtx() {
        try (InputStream in = CudaBackend.class.getResourceAsStream("/ai/enr/compute/cuda/enr_kernels.ptx")) {
            if (in == null) {
                throw new IllegalStateException("PTX kernel resource not found");
            }
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to load PTX kernels", e);
        }
    }

    private long allocDevice(long bytes) {
        ensureReady();
        try {
            long ptr = CudaDriver.cuMemAlloc(arena, bytes);
            allocatedBytes.addAndGet(bytes);
            return ptr;
        } catch (Throwable t) {
            throw new IllegalStateException("Device allocation failed: " + t.getMessage(), t);
        }
    }

    private void freeDevice(long ptr, long bytes) {
        try {
            CudaDriver.cuMemFree(ptr);
        } catch (Throwable ignored) {
            // best effort on shutdown
        }
        allocatedBytes.addAndGet(-bytes);
    }

    /**
     * Releases a device tensor unless it is resident in the weight cache
     * (cached tensors are owned by the cache and freed on close/eviction).
     */
    private void release(DeviceTensor tensor) {
        synchronized (deviceCache) {
            for (DeviceTensor cached : deviceCache.values()) {
                if (cached.ptr == tensor.ptr) {
                    return;
                }
            }
        }
        freeDevice(tensor.ptr, tensor.bytes);
    }

    public long allocatedVramBytes() {
        return allocatedBytes.get();
    }

    private DeviceTensor upload(float[] data) {
        DeviceTensor cached = cachedDeviceTensor(data);
        if (cached != null) {
            return cached;
        }
        long bytes = (long) data.length * 4;
        long ptr = allocDevice(bytes);
        try (Arena local = Arena.ofConfined()) {
            MemorySegment host = local.allocate(bytes);
            MemorySegment.copy(MemorySegment.ofArray(data), 0, host, 0, bytes);
            int rc = CudaDriver.cuMemcpyHtoD(ptr, host, bytes);
            if (rc != CudaDriver.CUDA_SUCCESS) {
                throw new IllegalStateException("H2D copy failed: " + rc);
            }
        } catch (Throwable t) {
            freeDevice(ptr, bytes);
            if (t instanceof IllegalStateException e) {
                throw e;
            }
            throw new IllegalStateException("H2D copy failed: " + t.getMessage(), t);
        }
        DeviceTensor tensor = new DeviceTensor(ptr, bytes);
        rememberForCaching(data, tensor);
        return tensor;
    }

    /**
     * Returns a cached device copy if the same float[] instance has been
     * uploaded before (stable arrays: model weights). One-shot activation
     * arrays are never cached. The KV cache buffers are excluded by the
     * caller (they are mutated in place between calls).
     */
    private DeviceTensor cachedDeviceTensor(float[] data) {
        synchronized (deviceCache) {
            return deviceCache.get(data);
        }
    }

    private void rememberForCaching(float[] data, DeviceTensor tensor) {
        synchronized (deviceCache) {
            if (arraySeen.size() > 8192) {
                arraySeen.clear();
            }
            int seen = arraySeen.getOrDefault(data, 0) + 1;
            arraySeen.put(data, seen);
            if (seen >= 2 && !deviceCache.containsKey(data)) {
                if (deviceCache.size() < DEVICE_CACHE_MAX_TENSORS
                        && allocatedBytes.get() < DEVICE_CACHE_MAX_BYTES) {
                    deviceCache.put(data, tensor);
                } else {
                    clearDeviceCache();
                    deviceCache.put(data, tensor);
                }
            }
        }
    }

    private void clearDeviceCache() {
        synchronized (deviceCache) {
            for (DeviceTensor tensor : deviceCache.values()) {
                try {
                    freeDevice(tensor.ptr, tensor.bytes);
                } catch (RuntimeException ignored) {
                    // best effort
                }
            }
            deviceCache.clear();
        }
    }

    private float[] download(DeviceTensor tensor) {
        float[] out = new float[(int) (tensor.bytes / 4)];
        try (Arena local = Arena.ofConfined()) {
            MemorySegment host = local.allocate(tensor.bytes);
            int rc = CudaDriver.cuMemcpyDtoH(host, tensor.ptr, tensor.bytes);
            if (rc != CudaDriver.CUDA_SUCCESS) {
                throw new IllegalStateException("D2H copy failed: " + rc);
            }
            MemorySegment.copy(host, 0, MemorySegment.ofArray(out), 0, tensor.bytes);
        } catch (Throwable t) {
            throw new IllegalStateException("D2H copy failed: " + t.getMessage(), t);
        }
        return out;
    }

    private void launch(long function, long[] ptrArgs, int[] intArgs, int gridX, int gridY,
                        int blockX, int blockY) {
        launch(function, ptrArgs, intArgs, gridX, gridY, blockX, blockY, 0);
    }

    private void launch(long function, long[] ptrArgs, int[] intArgs, int gridX, int gridY,
                        int blockX, int blockY, int sharedMemBytes) {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment paramArray = local.allocate(ValueLayout.ADDRESS, ptrArgs.length + intArgs.length);
            int idx = 0;
            for (long ptr : ptrArgs) {
                MemorySegment slot = local.allocate(ValueLayout.ADDRESS);
                slot.set(ValueLayout.ADDRESS, 0, MemorySegment.ofAddress(ptr));
                paramArray.setAtIndex(ValueLayout.ADDRESS, idx++, slot);
            }
            for (int value : intArgs) {
                MemorySegment slot = local.allocate(ValueLayout.JAVA_INT);
                slot.set(ValueLayout.JAVA_INT, 0, value);
                paramArray.setAtIndex(ValueLayout.ADDRESS, idx++, slot);
            }
            CudaDriver.cuLaunchKernel(function, gridX, gridY, 1, blockX, blockY, 1, sharedMemBytes, paramArray);
            int rc = CudaDriver.cuCtxSynchronize();
            if (rc != CudaDriver.CUDA_SUCCESS) {
                throw new IllegalStateException("CUDA kernel launch/sync failed: " + rc);
            }
        } catch (IllegalStateException e) {
            throw e;
        } catch (Throwable t) {
            throw new IllegalStateException("CUDA kernel launch failed: " + t.getMessage(), t);
        }
    }

    private record DeviceTensor(long ptr, long bytes) {
    }

    @Override
    public FloatTensor matmul(FloatTensor a, FloatTensor b) {
        if (!available) {
            return cpuFallback.matmul(a, b);
        }
        int m = (int) a.shape()[0];
        int k = (int) a.shape()[1];
        int n = (int) b.shape()[1];
        DeviceTensor da = upload(a.data());
        DeviceTensor db = upload(b.data());
        DeviceTensor dc = new DeviceTensor(allocDevice((long) m * n * 4), (long) m * n * 4);
        try {
            launch(fnMatmul, new long[]{da.ptr, db.ptr, dc.ptr}, new int[]{m, n, k},
                    (n + TILE - 1) / TILE, (m + TILE - 1) / TILE, TILE, TILE);
            return new FloatTensor(new long[]{m, n}, download(dc));
        } finally {
            release(da);
            release(db);
            release(dc);
        }
    }

    @Override
    public FloatTensor rmsNorm(FloatTensor x, float[] weight, float eps) {
        if (!available) {
            return cpuFallback.rmsNorm(x, weight, eps);
        }
        int rows = (int) x.shape()[0];
        int dim = (int) x.shape()[1];
        DeviceTensor dx = upload(x.data());
        DeviceTensor dw = upload(weight);
        DeviceTensor dy = new DeviceTensor(allocDevice((long) rows * dim * 4), (long) rows * dim * 4);
        try {
            launch(fnRmsnorm, new long[]{dx.ptr, dw.ptr, dy.ptr}, new int[]{dim, Float.floatToIntBits(eps)},
                    rows, 1, BLOCK, 1, BLOCK * 4);
            return new FloatTensor(x.shape(), download(dy));
        } finally {
            release(dx);
            release(dw);
            release(dy);
        }
    }

    @Override
    public FloatTensor softmaxRows(FloatTensor x) {
        if (!available) {
            return cpuFallback.softmaxRows(x);
        }
        int rows = (int) x.shape()[0];
        int cols = (int) x.shape()[1];
        DeviceTensor dx = upload(x.data());
        DeviceTensor dy = new DeviceTensor(allocDevice((long) rows * cols * 4), (long) rows * cols * 4);
        try {
            launch(fnSoftmaxMasked, new long[]{dx.ptr, dy.ptr}, new int[]{rows, cols, cols - 1, rows, cols},
                    rows, 1, BLOCK, 1, BLOCK * 4);
            return new FloatTensor(x.shape(), download(dy));
        } finally {
            release(dx);
            release(dy);
        }
    }

    @Override
    public FloatTensor silu(FloatTensor x) {
        if (!available) {
            return cpuFallback.silu(x);
        }
        int n = (int) x.size();
        DeviceTensor dx = upload(x.data());
        DeviceTensor dy = new DeviceTensor(allocDevice((long) n * 4), (long) n * 4);
        try {
            launch(fnSiluMul, new long[]{dx.ptr, dx.ptr, dy.ptr}, new int[]{n},
                    (n + BLOCK - 1) / BLOCK, 1, BLOCK, 1);
            return new FloatTensor(x.shape(), download(dy));
        } finally {
            release(dx);
            release(dy);
        }
    }

    @Override
    public FloatTensor siluMul(FloatTensor gate, FloatTensor up) {
        if (!available) {
            return cpuFallback.siluMul(gate, up);
        }
        int n = (int) gate.size();
        DeviceTensor dg = upload(gate.data());
        DeviceTensor du = upload(up.data());
        DeviceTensor dy = new DeviceTensor(allocDevice((long) n * 4), (long) n * 4);
        try {
            launch(fnSiluMul, new long[]{dg.ptr, du.ptr, dy.ptr}, new int[]{n},
                    (n + BLOCK - 1) / BLOCK, 1, BLOCK, 1);
            return new FloatTensor(gate.shape(), download(dy));
        } finally {
            release(dg);
            release(du);
            release(dy);
        }
    }

    @Override
    public FloatTensor add(FloatTensor a, FloatTensor b) {
        if (!available) {
            return cpuFallback.add(a, b);
        }
        int n = (int) a.size();
        DeviceTensor da = upload(a.data());
        DeviceTensor db = upload(b.data());
        DeviceTensor dc = new DeviceTensor(allocDevice((long) n * 4), (long) n * 4);
        try {
            launch(fnAdd, new long[]{da.ptr, db.ptr, dc.ptr}, new int[]{n},
                    (n + BLOCK - 1) / BLOCK, 1, BLOCK, 1);
            return new FloatTensor(a.shape(), download(dc));
        } finally {
            release(da);
            release(db);
            release(dc);
        }
    }

    @Override
    public FloatTensor mul(FloatTensor a, FloatTensor b) {
        if (!available) {
            return cpuFallback.mul(a, b);
        }
        int n = (int) a.size();
        DeviceTensor da = upload(a.data());
        DeviceTensor db = upload(b.data());
        DeviceTensor dc = new DeviceTensor(allocDevice((long) n * 4), (long) n * 4);
        try {
            launch(fnMul, new long[]{da.ptr, db.ptr, dc.ptr}, new int[]{n},
                    (n + BLOCK - 1) / BLOCK, 1, BLOCK, 1);
            return new FloatTensor(a.shape(), download(dc));
        } finally {
            release(da);
            release(db);
            release(dc);
        }
    }

    @Override
    public FloatTensor attention(FloatTensor q, FloatTensor k, FloatTensor v, float scale, int startPos) {
        if (!available) {
            return cpuFallback.attention(q, k, v, scale, startPos);
        }
        int heads = (int) q.shape()[0];
        int seqLen = (int) q.shape()[1];
        int kvSeqLen = (int) k.shape()[1];
        int headDim = (int) q.shape()[2];

        FloatTensor scores = matmulT(q, k); // [heads*seqLen, heads*kvSeqLen]
        scaleInPlace(scores, scale);
        FloatTensor probs = maskedSoftmax(scores, startPos, seqLen, kvSeqLen);
        float[] vFlat = v.data();
        FloatTensor vMatrix = new FloatTensor(new long[]{(long) heads * kvSeqLen, headDim}, vFlat);
        FloatTensor out = matmul(probs, vMatrix); // [heads*seqLen, headDim]
        return new FloatTensor(new long[]{heads, seqLen, headDim}, out.data());
    }

    private void scaleInPlace(FloatTensor tensor, float scalar) {
        int n = (int) tensor.size();
        DeviceTensor d = upload(tensor.data());
        try {
            launch(fnScaleInplace, new long[]{d.ptr}, new int[]{Float.floatToIntBits(scalar), n},
                    (n + BLOCK - 1) / BLOCK, 1, BLOCK, 1);
            float[] scaled = download(d);
            System.arraycopy(scaled, 0, tensor.data(), 0, n);
        } finally {
            release(d);
        }
    }

    private FloatTensor matmulT(FloatTensor a, FloatTensor b) {
        int m = (int) (a.shape()[0] * a.shape()[1]);
        int k = (int) a.shape()[2];
        int n = (int) (b.shape()[0] * b.shape()[1]);
        DeviceTensor da = upload(a.data());
        DeviceTensor db = upload(b.data());
        DeviceTensor dc = new DeviceTensor(allocDevice((long) m * n * 4), (long) m * n * 4);
        try {
            launch(fnMatmulT, new long[]{da.ptr, db.ptr, dc.ptr}, new int[]{m, n, k},
                    (n + TILE - 1) / TILE, (m + TILE - 1) / TILE, TILE, TILE);
            return new FloatTensor(new long[]{m, n}, download(dc));
        } finally {
            release(da);
            release(db);
            release(dc);
        }
    }

    private FloatTensor maskedSoftmax(FloatTensor scores, int startPos, int seqLen, int kvSeqLen) {
        int rows = (int) scores.shape()[0];
        int cols = (int) scores.shape()[1];
        DeviceTensor dx = upload(scores.data());
        DeviceTensor dy = new DeviceTensor(allocDevice((long) rows * cols * 4), (long) rows * cols * 4);
        try {
            launch(fnSoftmaxMasked, new long[]{dx.ptr, dy.ptr},
                    new int[]{rows, cols, startPos, seqLen, kvSeqLen},
                    rows, 1, BLOCK, 1, BLOCK * 4);
            return new FloatTensor(scores.shape(), download(dy));
        } finally {
            release(dx);
            release(dy);
        }
    }

    @Override
    public void close() {
        if (module != 0) {
            try {
                CudaDriver.cuModuleUnload(module);
            } catch (Throwable ignored) {
                // best effort
            }
            module = 0;
        }
        if (arena != null) {
            arena.close();
            arena = null;
        }
    }
}
