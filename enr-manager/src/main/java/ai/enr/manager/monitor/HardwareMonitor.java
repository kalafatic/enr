package ai.enr.manager.monitor;

import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonObject;
import ai.enr.runtime.EnrRuntime;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.OperatingSystemMXBean;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Hardware monitoring: CPU load, RAM, GPU utilization and VRAM. GPU figures
 * come from {@code nvidia-smi} when present and from the CUDA backend device
 * info; absence is reported as empty, never fabricated.
 */
public final class HardwareMonitor {

    private final OperatingSystemMXBean osBean = ManagementFactory.getOperatingSystemMXBean();
    private final MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();

    public JsonObject toJson(EnrRuntime runtime) {
        JsonObject root = new JsonObject();
        root.put("cpu", cpuJson());
        root.put("ram", ramJson());
        root.put("gpus", gpuJson(runtime));
        return root;
    }

    private JsonObject cpuJson() {
        JsonObject cpu = new JsonObject();
        cpu.put("processors", osBean.getAvailableProcessors());
        cpu.put("load_average", Math.round(osBean.getSystemLoadAverage() * 100) / 100.0);
        if (osBean instanceof com.sun.management.OperatingSystemMXBean extended) {
            double load = extended.getCpuLoad();
            cpu.put("utilization_percent", load < 0 ? null : Math.round(load * 10000) / 100.0);
        }
        return cpu;
    }

    private JsonObject ramJson() {
        JsonObject ram = new JsonObject();
        if (osBean instanceof com.sun.management.OperatingSystemMXBean extended) {
            ram.put("total_bytes", extended.getTotalMemorySize());
            ram.put("free_bytes", extended.getFreeMemorySize());
        }
        long heapUsed = memoryBean.getHeapMemoryUsage().getUsed();
        long heapMax = memoryBean.getHeapMemoryUsage().getMax();
        ram.put("jvm_heap_used_bytes", heapUsed);
        ram.put("jvm_heap_max_bytes", heapMax);
        return ram;
    }

    private JsonArray gpuJson(EnrRuntime runtime) {
        JsonArray gpus = new JsonArray();
        for (JsonObject device : runtime.devices().stream()
                .filter(d -> !"cpu".equals(d.type().id()))
                .map(d -> {
                    JsonObject g = new JsonObject();
                    g.put("index", d.index());
                    g.put("name", d.name());
                    g.put("driver_version", d.driverVersion());
                    g.put("total_memory_bytes", d.totalMemoryBytes());
                    g.put("free_memory_bytes", d.freeMemoryBytes());
                    return g;
                })
                .toList()) {
            gpus.add(device);
        }
        mergeNvidiaSmi(gpus);
        return gpus;
    }

    /**
     * Enriches GPU entries with live utilization from nvidia-smi when the
     * binary is present. Failures are ignored: driver data above stays valid.
     */
    private void mergeNvidiaSmi(JsonArray gpus) {
        List<JsonObject> smi = readNvidiaSmi();
        if (smi.isEmpty()) {
            return;
        }
        for (int i = 0; i < Math.min(gpus.size(), smi.size()); i++) {
            JsonObject existing = gpus.getObject(i);
            JsonObject fresh = smi.get(i);
            if (fresh.has("utilization_percent")) {
                existing.put("utilization_percent", fresh.getInt("utilization_percent", -1));
            }
            if (fresh.has("temperature_c")) {
                existing.put("temperature_c", fresh.getInt("temperature_c", -1));
            }
            existing.put("name", fresh.getString("name", existing.getString("name", "")));
            existing.put("total_memory_bytes", fresh.getLong("total_memory_bytes", existing.getLong("total_memory_bytes", 0)));
            existing.put("used_memory_bytes", fresh.getLong("used_memory_bytes", 0));
        }
    }

    private List<JsonObject> readNvidiaSmi() {
        List<JsonObject> result = new ArrayList<>();
        try {
            Process process = new ProcessBuilder(
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits")
                    .redirectErrorStream(true)
                    .start();
            if (!process.waitFor(5, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                return result;
            }
            if (process.exitValue() != 0) {
                return result;
            }
            String output = new String(process.getInputStream().readAllBytes());
            for (String line : output.split("\n")) {
                String[] parts = line.strip().split(",\\s*");
                if (parts.length < 5) {
                    continue;
                }
                JsonObject gpu = new JsonObject();
                gpu.put("name", parts[0]);
                gpu.put("utilization_percent", parseLong(parts[1]));
                gpu.put("used_memory_bytes", parseLong(parts[2]) * 1024L * 1024L);
                gpu.put("total_memory_bytes", parseLong(parts[3]) * 1024L * 1024L);
                gpu.put("temperature_c", parseLong(parts[4]));
                result.add(gpu);
            }
        } catch (Exception e) {
            // nvidia-smi absent or failed: driver data remains authoritative
        }
        return result;
    }

    private static long parseLong(String value) {
        try {
            return Long.parseLong(value.trim());
        } catch (NumberFormatException e) {
            return -1;
        }
    }
}
