# ENR Architecture

Date: 2026-11-16
Status: PHASE 1 deliverable

## 1. Product Identity

**EVO Native Runtime (ENR)** — Native AI Model Inference & Orchestration Engine.
A standalone, self-contained Java 21 runtime that loads, executes and manages native EVO models
(`.evo` v2/v1 and Darwin Forge GGUF v3 exports) without Ollama, llama.cpp or any external
inference server.

## 2. Architecture Principles

1. **Standalone**: single JVM, JDK-only runtime dependencies, Maven build, no OSGi/RCP.
2. **Contract-driven loading**: the `.evo`/GGUF formats are interoperability contracts; format
   handling is isolated in `enr-model-loader` behind `ModelSource`.
3. **Compute backend abstraction**: `enr-compute` defines the execution contract; CPU and CUDA
   implement equivalent semantics; backend selection and fallback are explicit and observable.
4. **Numerical fidelity**: the CPU backend is the reference; math replicates the verified EVO
   semantics; CUDA results must match within tolerance.
5. **Multi-model by design**: registry + lifecycle + resource accounting + scheduling; models are
   loaded/unloaded explicitly; never unloaded mid-execution.
6. **Session isolation**: sessions own KV caches and context; no shared mutable generation state.
7. **Observability**: metrics, diagnostics, device and memory reporting are first-class.
8. **Layered consumption**: everything the HTTP API exposes is available through the Java API
   which wraps the runtime service API — no side channels.

## 3. Logical Component Diagram

```
                    ┌───────────────────────────────────────────┐
                    │                 enr-cli                   │
                    └───────────────┬───────────────────────────┘
                                    │ (embedded or HTTP client)
                    ┌───────────────▼───────────────┐   ┌──────────────────┐
                    │          enr-server           │   │   enr-java-api   │
                    │  REST + SSE (JDK HttpServer)  │◄──┤  (HTTP client)   │
                    └───────────────┬───────────────┘   └──────────────────┘
                                    │
                    ┌───────────────▼───────────────────────────────┐
                    │                 enr-runtime                   │
                    │  ModelRegistry · Scheduler · ResourceLimits   │
                    │  SessionManager · Metrics · Health            │
                    └───────┬───────────────────────┬──────────────┘
                            │                       │
                 ┌──────────▼──────────┐   ┌────────▼──────────────────────┐
                 │   enr-inference    │   │       enr-model-loader        │
                 │ Graph · KV cache   │   │  .evo v2/v1 · GGUF v3         │
                 │ Sampling · Gen     │   │  Validate · Map/Lazy access   │
                 └──────────┬─────────┘   └───────────────────────────────┘
                            │                         ▲
                 ┌──────────▼──────────┐   ┌──────────┴──────────┐
                 │    enr-compute      │   │    enr-tokenizer    │
                 │  Backend SPI ·      │   │ BPE byte-fallback   │
                 │  Devices · Memory   │   │ incremental decode  │
                 └─────┬────────┬──────┘   └─────────────────────┘
                       │        │
            ┌──────────▼───┐ ┌──▼──────────────┐
            │ enr-compute- │ │ enr-compute-    │
            │ cpu          │ │ cuda (FFM+PTX)  │
            └──────────────┘ └─────────────────┘
                       │        │
                 ┌─────▼────────▼─────┐
                 │      enr-core      │
                 │ Tensors · dtypes   │
                 │ JSON · config · io │
                 └────────────────────┘
```

## 4. Data Flow (inference request)

1. HTTP/Java API request validated by `enr-server` → `InferenceRequest`.
2. `enr-runtime` resolves the model (load if policy allows), acquires an execution permit
   (resource-aware scheduler), creates a `GenerationSession` (own KV cache + context).
3. `enr-inference` tokenizes the prompt, runs prefill, then autoregressive decode steps on the
   selected `ComputeBackend`, streaming tokens upward.
4. Cancellation/timeouts propagate from the session handle; KV cache is discarded; the model
   instance stays resident for subsequent requests.

## 5. Key Interfaces (stable API surface)

- `ModelSource.open(Path, LoadOptions) -> LoadedModel` — format detection + validation
- `LoadedModel` — architecture, weights (mapped/lazy), tokenizer data, metadata
- `ComputeBackend` — `allocate/transfer/execute(kernels)` + `DeviceInfo`
- `ModelInstance` — a loaded model bound to a backend with resident weights
- `GenerationSession` — context, KV cache, sampling config, streaming listener, cancellation
- `EnrRuntime` — registry, load/unload, inference, sessions, status, metrics
- `EnrClient` (Java API) and HTTP endpoints (see ENR-API-SPECIFICATION.md)

## 6. Concurrency Model

- Server: JDK virtual threads per request.
- Runtime: per-model read-write execution permits; concurrent sessions share model weights
  read-only; KV caches are session-private.
- CPU kernels: deterministic sequential kernels by default; parallelization only for large
  independent loops (row-wise), documented.
- CUDA: single stream per model instance + session-serialized kernels, pinned host staging.
- All shared state guarded; no static mutable global state in the runtime.

## 7. Resource Management

- CPU: configurable worker budget; heap accounting via explicit arena estimates of resident
  weights.
- GPU: VRAM budget per device; per-model footprint estimates; eviction of idle models under
  pressure; never evict a model with active sessions.
- Model residency states: `NOT_LOADED → LOADING → READY → UNLOADING`, with health per state.

## 8. Configuration & Operations

Single `enr.yaml` (or defaults): HTTP bind, model directories, backend preference, memory limits,
concurrency, logging, session persistence. CLI `enr start|stop|status|models|load|unload|infer|
devices|diagnostics`. See ENR-DEPLOYMENT-GUIDE.md.

## 9. Extension Points (versioned, minimal)

- New model format versions: `enr-model-loader` `FormatReader` SPI keyed by magic/version.
- New quantization: dtype decoders in `enr-core` + loader mapping.
- New compute backends: `ComputeBackend` implementation + registration.
No speculative extension frameworks beyond these.
