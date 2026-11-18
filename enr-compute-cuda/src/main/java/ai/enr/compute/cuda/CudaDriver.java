package ai.enr.compute.cuda;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.util.Optional;

/**
 * Minimal CUDA driver API bindings via the FFM API (JEP 454).
 * No toolkit required at runtime; the driver JIT-compiles the bundled PTX.
 */
public final class CudaDriver {

    public static final int CUDA_SUCCESS = 0;
    public static final int CU_CTX_SCHED_AUTO = 0;
    public static final int CU_MEMCPY_HOST_TO_DEVICE = 1;
    public static final int CU_MEMCPY_DEVICE_TO_HOST = 2;

    private static final Linker LINKER = Linker.nativeLinker();
    private static final Arena LIB_ARENA = Arena.ofShared();

    private static final SymbolLookup CUDA;
    private static final SymbolLookup FALLBACK;

    static {
        SymbolLookup cuda = null;
        SymbolLookup fallback = null;
        try {
            cuda = SymbolLookup.libraryLookup("libcuda.so.1", LIB_ARENA);
        } catch (IllegalArgumentException e) {
            try {
                cuda = SymbolLookup.libraryLookup("libcuda.so", LIB_ARENA);
            } catch (IllegalArgumentException e2) {
                fallback = SymbolLookup.libraryLookup(System.mapLibraryName("cuda"), LIB_ARENA);
            }
        }
        CUDA = cuda;
        FALLBACK = fallback;
    }

    public static boolean available() {
        return lookup("cuInit") != null;
    }

    private static Optional<MemorySegment> lookup(String name) {
        try {
            SymbolLookup lookup = CUDA != null ? CUDA : FALLBACK;
            return lookup.find(name);
        } catch (RuntimeException e) {
            return Optional.empty();
        }
    }

    private static MethodHandle handle(String name, FunctionDescriptor descriptor) {
        Optional<MemorySegment> found = lookup(name);
        if (found.isEmpty()) {
            throw new IllegalStateException("CUDA driver function not available: " + name);
        }
        return LINKER.downcallHandle(found.get(), descriptor);
    }

    private static final MethodHandle CU_INIT = handle("cuInit",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
    private static final MethodHandle CU_DRIVER_GET_VERSION = handle("cuDriverGetVersion",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
    private static final MethodHandle CU_DEVICE_GET_COUNT = handle("cuDeviceGetCount",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
    private static final MethodHandle CU_DEVICE_GET = handle("cuDeviceGet",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
    private static final MethodHandle CU_DEVICE_GET_NAME = handle("cuDeviceGetName",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
    private static final MethodHandle CU_DEVICE_TOTAL_MEM = handle("cuDeviceTotalMem_v2",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
    private static final MethodHandle CU_CTX_CREATE = handle("cuCtxCreate_v2",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
    private static final MethodHandle CU_CTX_DESTROY = handle("cuCtxDestroy_v2",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
    private static final MethodHandle CU_CTX_SYNCHRONIZE = handle("cuCtxSynchronize",
            FunctionDescriptor.of(ValueLayout.JAVA_INT));
    private static final MethodHandle CU_MEM_GET_INFO = handle("cuMemGetInfo_v2",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    private static final MethodHandle CU_MEM_ALLOC = handle("cuMemAlloc_v2",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
    private static final MethodHandle CU_MEM_FREE = handle("cuMemFree_v2",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
    private static final MethodHandle CU_MEMCPY_HTO_D = handle("cuMemcpyHtoD_v2",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
    private static final MethodHandle CU_MEMCPY_DTO_H = handle("cuMemcpyDtoH_v2",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
    private static final MethodHandle CU_MODULE_LOAD_DATA = handle("cuModuleLoadData",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    private static final MethodHandle CU_MODULE_GET_FUNCTION = handle("cuModuleGetFunction",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    private static final MethodHandle CU_LAUNCH_KERNEL = handle("cuLaunchKernel",
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                    ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    private static final MethodHandle CU_MODULE_UNLOAD = handle("cuModuleUnload",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

    private CudaDriver() {
    }

    public static int cuInit(int flags) throws Throwable {
        return (int) CU_INIT.invoke(flags);
    }

    public static int cuDriverGetVersion(Arena arena) throws Throwable {
        MemorySegment version = arena.allocate(ValueLayout.JAVA_INT);
        int rc = (int) CU_DRIVER_GET_VERSION.invoke(version);
        return version.get(ValueLayout.JAVA_INT, 0);
    }

    public static int cuDeviceGetCount(Arena arena) throws Throwable {
        MemorySegment count = arena.allocate(ValueLayout.JAVA_INT);
        int rc = (int) CU_DEVICE_GET_COUNT.invoke(count);
        if (rc != CUDA_SUCCESS) {
            return 0;
        }
        return count.get(ValueLayout.JAVA_INT, 0);
    }

    public static int cuDeviceGet(Arena arena, int ordinal) throws Throwable {
        MemorySegment device = arena.allocate(ValueLayout.JAVA_INT);
        int rc = (int) CU_DEVICE_GET.invoke(device, ordinal);
        if (rc != CUDA_SUCCESS) {
            return -1;
        }
        return device.get(ValueLayout.JAVA_INT, 0);
    }

    public static String cuDeviceGetName(Arena arena, int device) throws Throwable {
        MemorySegment name = arena.allocate(256);
        int rc = (int) CU_DEVICE_GET_NAME.invoke(name, 256, device);
        if (rc != CUDA_SUCCESS) {
            return "unknown";
        }
        byte[] bytes = new byte[256];
        for (int i = 0; i < 256; i++) {
            byte b = name.get(ValueLayout.JAVA_BYTE, i);
            if (b == 0) {
                return new String(bytes, 0, i, java.nio.charset.StandardCharsets.UTF_8);
            }
            bytes[i] = b;
        }
        return new String(bytes, java.nio.charset.StandardCharsets.UTF_8);
    }

    public static long cuDeviceTotalMem(Arena arena, int device) throws Throwable {
        MemorySegment total = arena.allocate(ValueLayout.JAVA_LONG);
        int rc = (int) CU_DEVICE_TOTAL_MEM.invoke(total, device);
        if (rc != CUDA_SUCCESS) {
            return 0;
        }
        return total.get(ValueLayout.JAVA_LONG, 0);
    }

    public static int cuCtxCreate(Arena arena, int device) throws Throwable {
        MemorySegment ctx = arena.allocate(ValueLayout.ADDRESS);
        return (int) CU_CTX_CREATE.invoke(ctx, CU_CTX_SCHED_AUTO, device);
    }

    public static int cuCtxDestroy(int ctx) throws Throwable {
        return (int) CU_CTX_DESTROY.invoke(ctx);
    }

    public static int cuCtxSynchronize() throws Throwable {
        return (int) CU_CTX_SYNCHRONIZE.invoke();
    }

    public static long[] cuMemGetInfo(Arena arena) throws Throwable {
        MemorySegment free = arena.allocate(ValueLayout.JAVA_LONG);
        MemorySegment total = arena.allocate(ValueLayout.JAVA_LONG);
        int rc = (int) CU_MEM_GET_INFO.invoke(free, total);
        if (rc != CUDA_SUCCESS) {
            return new long[]{0, 0};
        }
        return new long[]{free.get(ValueLayout.JAVA_LONG, 0), total.get(ValueLayout.JAVA_LONG, 0)};
    }

    public static long cuMemAlloc(Arena arena, long bytes) throws Throwable {
        MemorySegment ptr = arena.allocate(ValueLayout.ADDRESS);
        int rc = (int) CU_MEM_ALLOC.invoke(ptr, bytes);
        if (rc != CUDA_SUCCESS) {
            throw new IllegalStateException("cuMemAlloc failed: " + rc);
        }
        return ptr.get(ValueLayout.ADDRESS, 0).address();
    }

    public static int cuMemFree(long devicePtr) throws Throwable {
        MemorySegment ptr = MemorySegment.ofAddress(devicePtr);
        return (int) CU_MEM_FREE.invoke(ptr);
    }

    public static int cuMemcpyHtoD(long devicePtr, MemorySegment host, long bytes) throws Throwable {
        MemorySegment d = MemorySegment.ofAddress(devicePtr);
        return (int) CU_MEMCPY_HTO_D.invoke(d, host, bytes);
    }

    public static int cuMemcpyDtoH(MemorySegment host, long devicePtr, long bytes) throws Throwable {
        MemorySegment d = MemorySegment.ofAddress(devicePtr);
        return (int) CU_MEMCPY_DTO_H.invoke(host, d, bytes);
    }

    public static long cuModuleLoadData(Arena arena, MemorySegment ptx) throws Throwable {
        MemorySegment module = arena.allocate(ValueLayout.ADDRESS);
        int rc = (int) CU_MODULE_LOAD_DATA.invoke(module, ptx);
        if (rc != CUDA_SUCCESS) {
            throw new IllegalStateException("cuModuleLoadData failed: " + rc);
        }
        return module.get(ValueLayout.ADDRESS, 0).address();
    }

    public static long cuModuleGetFunction(Arena arena, long module, String name) throws Throwable {
        MemorySegment function = arena.allocate(ValueLayout.ADDRESS);
        try (Arena local = Arena.ofConfined()) {
            MemorySegment nameSeg = local.allocateFrom(name);
            int rc = (int) CU_MODULE_GET_FUNCTION.invoke(function, MemorySegment.ofAddress(module), nameSeg);
            if (rc != CUDA_SUCCESS) {
                throw new IllegalStateException("cuModuleGetFunction(" + name + ") failed: " + rc);
            }
        }
        return function.get(ValueLayout.ADDRESS, 0).address();
    }

    public static void cuLaunchKernel(long function, int gridX, int gridY, int gridZ,
                                      int blockX, int blockY, int blockZ, int sharedMem,
                                      MemorySegment kernelParams) throws Throwable {
        MemorySegment f = MemorySegment.ofAddress(function);
        CU_LAUNCH_KERNEL.invoke(f, gridX, gridY, gridZ, blockX, blockY, blockZ,
                sharedMem, MemorySegment.NULL, kernelParams, MemorySegment.NULL);
    }

    public static int cuModuleUnload(long module) throws Throwable {
        return (int) CU_MODULE_UNLOAD.invoke(MemorySegment.ofAddress(module));
    }
}
