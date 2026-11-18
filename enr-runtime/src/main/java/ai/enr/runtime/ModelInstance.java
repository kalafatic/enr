package ai.enr.runtime;

import ai.enr.compute.ComputeBackend;
import ai.enr.compute.DeviceInfo;
import ai.enr.inference.GenerationEngine;
import ai.enr.inference.ModelWeights;
import ai.enr.inference.Transformer;
import ai.enr.inference.TransformerConfig;
import ai.enr.loader.api.LoadedModel;
import ai.enr.tokenizer.EvoBpeTokenizer;

import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A loaded model instance bound to a compute backend. Weights are lazily
 * materialized; the instance owns its residency and execution permits.
 */
public final class ModelInstance {

    public enum State {
        LOADING, READY, UNLOADING, ERROR
    }

    private final String id;
    private final Path path;
    private final LoadedModel loadedModel;
    private final Transformer transformer;
    private final EvoBpeTokenizer tokenizer;
    private final ComputeBackend backend;
    private volatile State state = State.READY;
    private volatile String error;
    private final AtomicLong activeSessions = new AtomicLong();
    private final AtomicLong totalRequests = new AtomicLong();
    private final long loadedAt = System.currentTimeMillis();

    public ModelInstance(String id, Path path, LoadedModel loadedModel, ComputeBackend backend) {
        this.id = id;
        this.path = path;
        this.loadedModel = loadedModel;
        this.backend = backend;
        TransformerConfig config = TransformerConfig.from(loadedModel.architecture());
        this.transformer = new Transformer(config, new ModelWeights(loadedModel), backend);
        this.tokenizer = new EvoBpeTokenizer(
                loadedModel.tokenizer().tokenToId(),
                loadedModel.tokenizer().bosTokenId(),
                loadedModel.tokenizer().eosTokenId(),
                loadedModel.tokenizer().unkTokenId(),
                loadedModel.tokenizer().padTokenId());
    }

    public String id() {
        return id;
    }

    public Path path() {
        return path;
    }

    public LoadedModel loadedModel() {
        return loadedModel;
    }

    public Transformer transformer() {
        return transformer;
    }

    public EvoBpeTokenizer tokenizer() {
        return tokenizer;
    }

    public ComputeBackend backend() {
        return backend;
    }

    public State state() {
        return state;
    }

    public void state(State state) {
        this.state = state;
    }

    public String error() {
        return error;
    }

    public void error(String error) {
        this.error = error;
    }

    public long activeSessions() {
        return activeSessions.get();
    }

    public void beginSession() {
        activeSessions.incrementAndGet();
        totalRequests.incrementAndGet();
    }

    public void endSession() {
        activeSessions.decrementAndGet();
    }

    public long totalRequests() {
        return totalRequests.get();
    }

    public long loadedAt() {
        return loadedAt;
    }

    public long weightBytesLoaded() {
        return transformer.weights().loadedBytes();
    }

    public long totalWeightBytes() {
        return loadedModel.weightBytes();
    }

    public DeviceInfo deviceInfo() {
        return backend.deviceInfo();
    }

    public GenerationEngine newGenerationEngine() {
        return new GenerationEngine(transformer);
    }

    @Override
    public String toString() {
        return "ModelInstance[" + id + " state=" + state + " active=" + activeSessions.get()
                + " " + loadedModel.architecture() + "]";
    }
}
