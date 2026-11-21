package ai.enr.tests.modelmgmt;

import ai.enr.manager.models.DownloadLayer;
import ai.enr.manager.models.DownloadProgress;
import ai.enr.manager.models.DownloadState;
import ai.enr.manager.models.ModelDescriptor;
import ai.enr.manager.models.ModelProvider;
import ai.enr.manager.models.ProviderCapability;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

/**
 * Test-only model provider: scripted progress, controllable failures and
 * cancellation, in-memory installed-model store. One instance per provider id
 * (shared through the factory so tests can configure behavior before pulls).
 */
public final class StubModelProvider implements ModelProvider {

    private static final Set<ProviderCapability> CAPABILITIES = Set.of(
            ProviderCapability.MODEL_PULL,
            ProviderCapability.MODEL_LIST,
            ProviderCapability.MODEL_REMOVE,
            ProviderCapability.DOWNLOAD_PROGRESS,
            ProviderCapability.MODEL_METADATA);

    static final Map<String, StubModelProvider> INSTANCES = new ConcurrentHashMap<>();

    private final String id;
    final List<ModelDescriptor> installed = new CopyOnWriteArrayList<>();
    volatile List<DownloadProgress> script = List.of();
    volatile long stepDelayMs = 10;
    volatile long loopDelayMs = 0;
    volatile RuntimeException pullError = null;
    volatile CountDownLatch pullStarted = new CountDownLatch(1);
    volatile CountDownLatch pullReturned = new CountDownLatch(1);
    public volatile long pullCount = 0;
    volatile String lastPulledModel;

    StubModelProvider(String id) {
        this.id = id;
    }

    public static StubModelProvider instance(String id) {
        return INSTANCES.computeIfAbsent(id, StubModelProvider::new);
    }

    public void reset() {
        script = List.of();
        stepDelayMs = 10;
        loopDelayMs = 0;
        pullError = null;
        installed.clear();
        pullStarted = new CountDownLatch(1);
        pullReturned = new CountDownLatch(1);
        pullCount = 0;
        lastPulledModel = null;
    }

    public void addInstalled(String modelId) {
        installed.add(new ModelDescriptor(id, modelId, modelId, "sha256:stub", 1024,
                System.currentTimeMillis(), true, Map.of()));
    }

    /** Progress events replayed (in order) by the next pull. */
    public void script(List<DownloadProgress> events) {
        this.script = List.copyOf(events);
    }

    /** Endless progress loop used for cancellation/duplicate tests. */
    public void endlessLoop(long delayMs) {
        this.script = List.of();
        this.loopDelayMs = delayMs;
    }

    /** Makes the next pull throw after the first progress event. */
    public void failWith(String message) {
        this.pullError = new RuntimeException(message);
    }

    public void awaitPullStarted() throws InterruptedException {
        pullStarted.await();
    }

    public void awaitPullReturned() throws InterruptedException {
        pullReturned.await();
    }

    @Override
    public String id() {
        return id;
    }

    @Override
    public String displayName() {
        return "Stub Provider " + id;
    }

    @Override
    public Set<ProviderCapability> capabilities() {
        return CAPABILITIES;
    }

    @Override
    public List<ModelDescriptor> listInstalled() {
        return List.copyOf(installed);
    }

    @Override
    public List<ModelDescriptor> listAvailable() {
        return List.of();
    }

    @Override
    public ModelDescriptor describe(String modelId) {
        for (ModelDescriptor descriptor : installed) {
            if (descriptor.id().equals(modelId)) {
                return descriptor;
            }
        }
        throw new UnsupportedOperationException("stub cannot describe " + modelId);
    }

    @Override
    public void pull(String modelId, BooleanSupplier cancellationRequested,
                     Consumer<DownloadProgress> progress) throws Exception {
        pullCount++;
        lastPulledModel = modelId;
        CountDownLatch started = pullStarted;
        CountDownLatch returned = pullReturned;
        started.countDown();
        try {
            progress.accept(new DownloadProgress(DownloadState.CONNECTING, -1, 0, "", List.of()));
            if (pullError != null) {
                progress.accept(DownloadProgress.failed(pullError.getMessage()));
                throw pullError;
            }
            for (DownloadProgress event : script) {
                if (cancellationRequested.getAsBoolean()) {
                    progress.accept(new DownloadProgress(DownloadState.CANCEL_REQUESTED,
                            -1, 0, "cancelled by user", List.of()));
                    return;
                }
                Thread.sleep(stepDelayMs);
                progress.accept(event);
            }
            long total = 0;
            while (loopDelayMs > 0) {
                if (cancellationRequested.getAsBoolean()) {
                    progress.accept(new DownloadProgress(DownloadState.CANCEL_REQUESTED,
                            total, total, "cancelled by user", List.of()));
                    return;
                }
                total += 1000;
                progress.accept(new DownloadProgress(DownloadState.DOWNLOADING, -1, total,
                        "", List.of(new DownloadLayer("sha256:loop", -1, total, "pulling"))));
                Thread.sleep(loopDelayMs);
            }
        } finally {
            returned.countDown();
        }
    }

    @Override
    public void remove(String modelId) throws Exception {
        boolean removed = installed.removeIf(d -> d.id().equals(modelId));
        if (!removed) {
            throw new IllegalArgumentException("Model not installed: " + modelId);
        }
    }
}
