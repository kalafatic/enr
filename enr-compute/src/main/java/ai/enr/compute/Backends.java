package ai.enr.compute;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.ServiceLoader;

public final class Backends {

    private Backends() {
    }

    public static List<ComputeBackend> discover() {
        List<ComputeBackend> backends = new ArrayList<>();
        try {
            for (ComputeBackend backend : ServiceLoader.load(ComputeBackend.class)) {
                backends.add(backend);
            }
        } catch (java.util.ServiceConfigurationError error) {
            // A backend compiled for a newer JDK (e.g. the CUDA module) cannot
            // be loaded by this runtime; ignore it and continue with the rest.
        }
        return backends;
    }

    public static Optional<ComputeBackend> cpu() {
        return discover().stream()
                .filter(b -> b.deviceInfo().type() == DeviceType.CPU)
                .findFirst();
    }

    public static Optional<ComputeBackend> cuda() {
        return discover().stream()
                .filter(b -> b.deviceInfo().type() == DeviceType.CUDA && b.deviceInfo().available())
                .findFirst();
    }

    public static Optional<ComputeBackend> bestAvailable() {
        Optional<ComputeBackend> cuda = cuda();
        return cuda.isPresent() ? cuda : cpu();
    }
}
