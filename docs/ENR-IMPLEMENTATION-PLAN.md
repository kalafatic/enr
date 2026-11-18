# ENR Implementation Plan

Date: 2026-11-16
Status: PHASE 0 deliverable — plan approved by audit; implementation proceeds inside ENR repository only.

## 1. Build & Runtime Constraints

- Java 21 (Maven `release=21`), zero runtime dependencies (JDK only).
- Maven multi-module build. JUnit 5 for tests (test-scope only).
- Explicitly NOT used: Eclipse RCP, OSGi, SWT, Ollama, llama.cpp, EVO/Forge libraries, any
  third-party runtime library. No HTTP framework (JDK `com.sun.net.httpserver` + virtual threads).
- JSON: small internal parser/writer in `enr-core` (needed for `.evo` sections and the HTTP API).
- CUDA: Java FFM (JEP 454) bindings to the CUDA driver API + precompiled PTX kernels embedded as
  resources; explicit CPU fallback.

## 2. Module Plan

| Module | Purpose | Key content |
|---|---|---|
| `enr-core` | primitives | `Tensor`/`FloatTensor`, dtypes, shapes, JSON, hex/SHA-256 helpers, config, logging, clock |
| `enr-model-loader` | format loading | `.evo` v2/v1 readers, GGUF v3 reader, canonical `LoadedModel`/`TensorEntry`, validators, diagnostics, memory-mapped/lazy access, `.evo` writer + fixture tooling |
| `enr-tokenizer` | tokenization | byte-fallback longest-match encoder, incremental UTF-8 decoder, special tokens |
| `enr-compute` | backend SPI | `ComputeBackend`, `DeviceInfo`, memory accounting, execution API |
| `enr-compute-cpu` | CPU kernels | deterministic matmul, elementwise ops, softmax, RMSNorm, attention primitives, multi-threading |
| `enr-compute-cuda` | CUDA kernels | FFM driver API, device discovery, allocation, H2D/D2H, PTX matmul/elementwise/softmax, VRAM accounting, fallback |
| `enr-inference` | model execution | model graph builder, transformer blocks, KV cache, sampling, generation engine, sessions |
| `enr-runtime` | multi-model orchestration | model registry/lifecycle, scheduler, resource limits, request queue, eviction, health/metrics |
| `enr-server` | HTTP | REST endpoints + SSE streaming, validation, error codes, cancellation |
| `enr-java-api` | client | synchronous/asynchronous client, streaming, models, sessions, devices |
| `enr-cli` | operations | `start/stop/status/models/load/unload/infer/devices/diagnostics` |
| `enr-tests` | integration | end-to-end tests: loaders, inference equivalence, multi-model, API, runtime lifecycle, performance |

## 3. Phase Sequencing (with exit criteria)

1. **Core + loader** (`enr-core`, `enr-model-loader`): parse real `evo.gguf` + `tiny-evo.gguf`;
   validate tensors/checksums; `.evo` v2 writer/reader round-trip; corrupted-file rejection tests.
   Exit: loader integration tests green on real files.
2. **Tokenizer** (`enr-tokenizer`): encode/decode parity with verified semantics; incremental
   decode streaming tests.
   Exit: tokenizer tests green; round-trip on real vocab from `evo.gguf`.
3. **CPU compute + inference** (`enr-compute`, `enr-compute-cpu`, `enr-inference`): op unit tests
   (matmul, attention, softmax, RMSNorm, PE, KV cache), then full forward + generation.
   Exit: ENR CPU forward matches independent reference forward within tolerance; generation
   produces text; KV-cache path equals full-path logits.
4. **CUDA backend** (`enr-compute-cuda`): device discovery (RTX 5070 Ti), allocation, transfers,
   PTX kernels, equivalence vs CPU backend, fallback logic.
   Exit: CUDA vs CPU logits within tolerance; genuine device execution reported; CPU-only mode
   fully green when CUDA absent.
5. **Runtime + generation/session management** (`enr-runtime`): multi-model registry, concurrent
   sessions, isolation, scheduling, eviction, metrics.
   Exit: two models, concurrent requests, no cross-contamination, unload while idle only.
6. **HTTP server + Java API** (`enr-server`, `enr-java-api`): all documented endpoints incl. SSE,
   cancellation; client sync/async + streaming.
   Exit: API tests green (health, models, inference, stream, cancel, invalid requests, concurrency).
7. **CLI + ops** (`enr-cli`): all commands; config (bind, dirs, CPU/CUDA, limits, logging).
   Exit: independent start/stop/status/infer round-trip via CLI against the server.
8. **Acceptance + performance** (`enr-tests`): full suite, load/TTFT/tps/memory/VRAM/concurrency
   measurements from real executions; final documents updated with measured values.

## 4. Test Fixture Strategy

- Real fixtures: `evo.gguf`, `tiny-evo.gguf` copied from the reference repo into
  `enr-tests/src/test/resources/models/` (read-only copies, documented provenance).
- Generated fixtures: `.evo` v2 files produced by the ENR fixture tool from the GGUF weights
  (deterministic content hash, checksums verified).
- Corruption matrix: truncated header, bad magic, wrong checksum, bad dtype, shape mismatch,
  NaN weights, wrong version, truncated payloads.

## 5. Acceptance Criteria Mapping

| Criterion | Verified by |
|---|---|
| Builds independently | Maven build on JDK 21 in clean checkout |
| Starts independently | CLI `enr start` with no external services |
| Loads a supported native `.evo` model | loader + fixture tests; `enr models load` |
| Actual CPU inference | forward/generation tests + CLI `enr infer` |
| Actual CUDA inference (where supported) | CUDA backend tests on RTX 5070 Ti; genuine device execution |
| Two models managed | multi-model integration test |
| Concurrent independent requests | concurrency/session isolation tests |
| Working HTTP endpoints | API integration tests |
| Usable Java API | client tests against live server |
| Genuine runtime/resource info | metrics/device endpoints + memory accounting tests |
| Passes documented tests | `mvn test` full suite green |
| No external repo modifications | git status of external repos untouched (read-only enforcement) |

## 6. Risk-First Ordering

R1 (no real `.evo`) is addressed in Phase 1 via GGUF→`.evo` fixture generation. R2 (PE semantics)
is resolved by tests in Phase 3. R11 (no toolkit) is addressed in Phase 4 with driver-API + PTX.
No phase blocks on Darwin Forge changes.
