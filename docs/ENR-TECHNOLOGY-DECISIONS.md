# ENR Technology Decisions

Date: 2026-11-16
Status: PHASE 1 deliverable

## 1. Decisions

| # | Topic | Decision | Rationale |
|---|---|---|---|
| D1 | Language/version | Java 21 | Required by spec; virtual threads, FFM API, pattern matching |
| D2 | Build | Maven multi-module, `release=21` | Standard, ubiquitous, pinning toolchain independent of machine default JDK |
| D3 | Runtime dependencies | Zero (JDK only) | Independence requirement; reproducible deployment; no supply-chain surface |
| D4 | HTTP server | `com.sun.net.httpserver` + virtual threads | Built-in, lightweight, sufficient for versioned REST + SSE; no framework needed |
| D5 | Streaming | Server-Sent Events (`text/event-stream`) | Spec requirement; trivially consumable by Java client and browsers |
| D6 | JSON | internal minimal parser/writer | Needed for `.evo` sections and API; avoids dependency on third-party JSON lib |
| D7 | Client HTTP | `java.net.http.HttpClient` | JDK standard for the Java API and CLI remote mode |
| D8 | Tensors | `float[]`-backed row-major `FloatTensor` + dtype-aware storage contract | Matches audited EVO layout; simple; memory-mappable via ByteBuffer views |
| D9 | CPU kernels | hand-written deterministic loops; selective row-parallel via common pool | Numerical determinism first; parallelism where profiling justifies |
| D10 | CUDA backend | Java FFM (JEP 454) bindings to `libcuda.so.1` driver API; PTX kernels embedded as resources | No toolkit/NVRTC on target machine; driver JIT compiles PTX; zero native build step |
| D11 | CUDA compute | custom PTX kernels for matmul/elementwise/softmax; no cuBLAS | No cuBLAS on machine; kernels are simple and testable; can swap in cuBLAS later |
| D12 | Fallback | explicit backend fallback chain `cuda → cpu` with recorded reason | Spec requirement; observable, testable |
| D13 | Sampling | replicate audited algorithm (greedy/temp/topK/topP/penalties) | Format/interop contract; deterministic seeds supported for tests |
| D14 | KV cache | session-private preallocated growable buffers per layer | Isolation + zero-copy growth; bounded by context limit |
| D15 | Concurrency | virtual threads at server; per-model RW execution permits; session-private caches | Simplicity + isolation without global locks |
| D16 | Scheduling | priority FIFO queue per model + global CPU/VRAM budget admission | Deterministic, observable, satisfies multi-model requirements without a framework |
| D17 | Config | single YAML-subset file (simple key/value + sections) parsed internally | Avoids third-party YAML dependency; small documented schema |
| D18 | CLI arg parsing | hand-written minimal parser | Avoids dependency; command surface is small and stable |
| D19 | Logging | `java.util.logging` + single-line formatter | JDK-only; adequate for operations |
| D20 | Tests | JUnit 5 (Jupiter) via Surefire | Standard; test-scope only, not a runtime dependency |
| D21 | Model loading | memory-mapped read-only + lazy decompression on first use; optional eager mode | Large-file support; bounded startup cost |
| D22 | Format compatibility | `FormatReader` SPI keyed by magic + version | Isolates format evolution without touching inference |
| D23 | OSGi/RCP | excluded; optional adapter noted for later work packages | ENR must not depend on Eclipse stack |

## 2. Explicitly Rejected

- Spring Boot / Quarkus / Micronaut — heavyweight frameworks unjustified for this scope.
- JCUDA/JCublas — third-party native binding library; FFM is JDK-native.
- Forking llama.cpp / Ollama — contradicts the core product requirement.
- Kotlin/Scala — Java 21 is the mandated language.
- gRPC — HTTP/JSON+SSE is the documented contract.
- Third-party JSON (Jackson/Gson) — internal parser is small and audited.
- In-house YAML — full YAML parsing rejected; documented flat subset only.

## 3. Consequences

- The distribution is a single Maven-built set of jars + a launcher script; no native binaries
  except the optional CUDA PTX resources (embedded).
- Everything runs as one JVM process; embedded mode (CLI + runtime in-process) and client mode
  (CLI + remote server) share the same client code.
- Numerical behavior is deterministic for a given backend configuration; CUDA tolerance is
  documented (1e-4 relative for matmul-heavy paths).
