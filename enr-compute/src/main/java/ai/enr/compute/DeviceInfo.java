package ai.enr.compute;

public record DeviceInfo(
        int index,
        DeviceType type,
        String name,
        long totalMemoryBytes,
        long freeMemoryBytes,
        String driverVersion,
        boolean available) {

    public static DeviceInfo cpu() {
        Runtime runtime = Runtime.getRuntime();
        return new DeviceInfo(0, DeviceType.CPU, "CPU (JVM heap)",
                runtime.maxMemory(), runtime.maxMemory() - (runtime.totalMemory() - runtime.freeMemory()),
                System.getProperty("java.version", "unknown"), true);
    }

    public static DeviceInfo unavailable(DeviceType type, int index, String reason) {
        return new DeviceInfo(index, type, "unavailable: " + reason, 0, 0, "unknown", false);
    }

    @Override
    public String toString() {
        return "DeviceInfo[" + type + ":" + index + " '" + name + "' available=" + available
                + " memory=" + (totalMemoryBytes >> 20) + "MB free=" + (freeMemoryBytes >> 20) + "MB]";
    }
}
