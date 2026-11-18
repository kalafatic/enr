# ENR Module Structure

Date: 2026-11-16
Status: PHASE 1 deliverable

## 1. Maven Layout

```
evo-native-runtime/
  pom.xml                        parent: enr.version, JDK 21 release, JUnit 5, surefire
  enr-core/                      primitives, tensors, dtypes, JSON, io, config, metrics types
  enr-model-loader/              .evo v2/v1 + GGUF v3 readers, writer, validators, fixtures
  enr-tokenizer/                 encode/decode, incremental decoder
  enr-compute/                   ComputeBackend SPI, DeviceInfo, MemoryAllocator contract
  enr-compute-cpu/               reference CPU kernels
  enr-compute-cuda/              FFM CUDA driver bindings + PTX kernels + fallback
  enr-inference/                 graph construction, KV cache, sampling, GenerationEngine
  enr-runtime/                   ModelRegistry, scheduler, sessions, metrics, health
  enr-server/                    HTTP REST + SSE
  enr-java-api/                  client API (sync/async/streaming)
  enr-cli/                       command line tool + config loading
  enr-tests/                     integration + acceptance + performance tests
```

## 2. Module Dependencies (one direction only)

```
enr-core
   ▲
   ├── enr-model-loader ──► enr-core
   ├── enr-tokenizer   ──► enr-core
   ├── enr-compute     ──► enr-core
   │      ├── enr-compute-cpu ──► enr-compute, enr-core
   │      └── enr-compute-cuda ──► enr-compute, enr-core
   ├── enr-inference   ──► enr-core, enr-compute, enr-tokenizer, enr-model-loader
   ├── enr-runtime     ──► enr-inference, enr-compute(-cpu/-cuda), enr-model-loader
   ├── enr-server      ──► enr-runtime
   ├── enr-java-api    ──► enr-core only (HTTP client, no internal runtime dependency)
   ├── enr-cli         ──► enr-runtime (embedded), enr-java-api (remote), enr-server
   └── enr-tests       ──► everything (test scope)
```

Rationale: `enr-java-api` deliberately depends only on `enr-core` so that consumers import a
thin client without pulling in the runtime implementation. The CLI supports both embedded mode
(in-process runtime) and client mode (talking to a running server).

## 3. Package Conventions

- `ai.enr.core.tensor`, `ai.enr.core.json`, `ai.enr.core.io`, `ai.enr.core.config`
- `ai.enr.loader.evo`, `ai.enr.loader.gguf`, `ai.enr.loader.api`
- `ai.enr.tokenizer`
- `ai.enr.compute.api`, `ai.enr.compute.cpu`, `ai.enr.compute.cuda`
- `ai.enr.inference`, `ai.enr.inference.sample`, `ai.enr.inference.session`
- `ai.enr.runtime`, `ai.enr.runtime.sched`, `ai.enr.runtime.metrics`
- `ai.enr.server`, `ai.enr.server.api`
- `ai.enr.client`
- `ai.enr.cli`

Group id `ai.enr`, artifacts `enr-*`, single version property `enr.version`.

## 4. Dependency Policy

- Runtime: **none** beyond the JDK (21).
- Test: JUnit Jupiter + Maven Surefire only.
- No logging framework: `java.util.logging` configured by `enr-cli`/`enr-server` with a formatter.
- JSON: internal `ai.enr.core.json` (parser + writer), used by loader and server alike.
- Build pinned to `maven.compiler.release=21`; no annotation processors.

## 5. Deliverable Sizes / Boundaries

- Each module has its own unit tests; `enr-tests` only contains cross-module integration,
  acceptance and performance tests.
- No module depends on another module's test classes.
- `enr-compute-cuda` compiles and tests green even without GPU: device discovery reports
  unavailable and falls back to CPU (documented in test output).
