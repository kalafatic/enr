# ENR — Architecture Audit & Gap Report (Ollama Compatibility & Web Control Center)

Date: 2026-11-19
Status: PHASE 1 deliverable — audit of `/home/petr/git/evo-native-runtime-161126` (this repository; the
referenced `/home/petr/git/evo-native-runtime-121126` path does not exist — this copy *is* the ENR repository).

## 1. Existing Architecture (verified, not assumed)

Java 21 multi-module Maven reactor (`maven.compiler.release=21`; only `enr-compute-cuda` uses release 25).
Zero runtime dependencies beyond the JDK (JSON parser, HTTP server and CUDA driver bindings are all internal).

| Module | Responsibility (verified) |
|---|---|
| `enr-core` | Primitives: `FloatTensor`/`DType`/`TensorMath`, internal JSON (`Json`, `JsonObject`, `JsonArray`), SHA-256 (`Hashing`), hex/endian IO |
| `enr-model-loader` | Model formats: `.evo` v2/v1 readers, GGUF v3 reader (`GgufReader`), canonical `LoadedModel`, validators, checksums |
| `enr-tokenizer` | `EvoBpeTokenizer`: longest-match BPE with `<0xXX>` byte fallback, incremental UTF-8 decoder |
| `enr-compute` | Backend SPI (`ComputeBackend`, `DeviceInfo`), `Backends` discovery via `ServiceLoader` |
| `enr-compute-cpu` | Deterministic CPU kernels (matmul, attention, softmax, RMSNorm, RoPE) |
| `enr-compute-cuda` | CUDA driver API via FFM + embedded PTX kernels (JDK 25 module) |
| `enr-inference` | `Transformer`, KV cache, sampler, `GenerationEngine` (autoregressive loop, streaming, cancellation), `InferenceSession` |
| `enr-runtime` | `EnrRuntime` facade + `ModelManager` (in-memory model registry, load/unload, eviction), `SessionManager`, `Metrics` |
| `enr-server` | `EnrServer`: JDK `HttpServer` + virtual threads. `/api/v1/*` (health, status, models, load/unload, inference, SSE stream, cancel, sessions, metrics, devices) and `/enr/v1/*` family surface |
| `enr-java-api` | `EnrClient` (sync/async/streaming) over the HTTP API |
| `enr-cli` | `EnrCli` (start/stop/status/models/load/unload/infer/devices/diagnostics), `CliConfig` (minimal YAML subset), `enr.sh` launcher |
| `enr-tests` | Integration tests (API, family API, CUDA, performance benchmark) |

### 1.1 Startup mechanism
`enr start [-c enr.yaml]` → `EnrCli.start()` parses YAML subset (`CliConfig`) → builds `RuntimeConfig` →
`new EnrRuntime(config)` → auto-loads all `*.evo`/`*.gguf` from `runtime.model_directories` → `new EnrServer(...)` →
pid file `/tmp/enr.pid` → blocks on shutdown-hook latch. `enr stop` kills the pid. **No systemd unit, no crash recovery,
no log files** (JUL console only).

### 1.2 Runtime implementations
Two compute backends: CPU (`enr-compute-cpu`, deterministic reference) and CUDA (`enr-compute-cuda`, FFM driver API +
PTX, verified numerically equivalent on RTX 5070 Ti). Backend selection: `ModelManager.selectBackend` with explicit
CPU fallback policy (`allowCudaFallback`). No external runtime integrations (no Ollama/llama.cpp processes).

### 1.3 Model discovery & installation
Discovery = directory scan (`ModelManager.discoverModelFiles()`, `*.evo`/`*.gguf`). Loading = parse headers, lazy
weight materialization. **No download, no install, no catalogs, no import, no progress, no verification workflow,
no deletion** — the only state is the in-memory `ModelManager.instances` map. Model registry is not persisted.

### 1.4 Inference APIs
`POST /api/v1/inference` (blocking) and `/api/v1/inference/stream` (SSE `start`/`token`/`done`) with options
(max_tokens, temperature, seed, stop_token_ids, session_id, timeout). Sessions hold KV cache + context across turns.
Cancellation via request/session id. **No chat-message based API, no Ollama or OpenAI wire compatibility.**

### 1.5 DeepSeek implementation
**There is none.** A `grep` for deepseek/openai/anthropic/gemini/ollama/huggingface across sources found zero
implementations. `/enr/v1/providers` returns a hard-coded `enr-native` provider stub only. The "DeepSeek provider"
from the request is a gap to be filled, not an existing component to reuse.

### 1.6 Service lifecycle & configuration
CLI start/stop + pid file; `enr.yaml` (read-only, YAML subset: server.host/port, runtime.preferred_backend,
model_directories, max_concurrent_sessions, default_max_tokens). `RuntimeConfig` record with builder. No persistent
state, no secrets handling, no authentication, no remote-admin controls, no health-based restart.

### 1.7 GPU/CUDA integration
`Backends.discover()` (ServiceLoader) → `DeviceInfo` (index, type, name, total/free memory, driver version).
`CudaBackend` queries the CUDA driver directly. `nvidia-smi` is referenced in docs as optional diagnostics only —
no utilization metrics today.

### 1.8 Web components
**None.** `GET /` returns a JSON name/version payload. No UI assets exist.

### 1.9 Integration contracts (EVO, DStudio, DCode, TRYST)
`/enr/v1/*` family surface (health, models + load/unload routes, chat, chat/stream, runtime, providers, capabilities)
with the family TEIP error envelope — mapped for DCode/TRYST/DStudio consumption. The `ENR-FUTURE-*` docs are
specifications only ("no changes made"). These contracts must be preserved byte-compatible.

### 1.10 Baseline state
`JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64 mvn clean test` → **54 tests, 0 failures** across all 13 modules.
Build works on JDK 25 (CUDA module requires release 25).

## 2. Gap Analysis (requirement → status)

| Requirement | Status |
|---|---|
| Local model management (list/inspect/start/stop/remove) | Partial: in-memory load/unload only; no persistence, no remove, no metadata store |
| Model discovery, download, install, import, progress, cancel, verify | **Missing entirely** |
| Catalogs (Ollama Library, Hugging Face) | **Missing** |
| Provider management (config, test, discovery, health, diagnostics, secrets) | **Missing** (hard-coded stub) |
| Cloud providers (DeepSeek, OpenAI, Anthropic, Gemini, OpenAI-compatible) | **Missing** |
| Local providers (Ollama, llama.cpp, other OpenAI-compatible servers) | **Missing** |
| Model Router (no silent switching, explicit local/cloud separation) | **Missing** |
| Runtime simplification (model-first UX, auto backend compat) | Partial: backend selection exists; UX is path-based not model-based |
| systemd service (boot start, restart, crash recovery, graceful shutdown, logs, health) | **Missing** |
| Web Control Center (6 pages) | **Missing** |
| Ollama API compatibility (`/api/tags` … `/api/version`) | **Missing** |
| OpenAI API compatibility (`/v1/models`, chat/completions, completions, embeddings) | **Missing** |
| Hardware monitoring (CPU/RAM/GPU utilization, VRAM) | Partial: static device memory info only |
| Active inference request observability | Partial: internal map, not exposed |
| Diagnostics (system/runtime/provider logs, GPU/API diagnostics) | Partial: CLI-only summary; no logs endpoint |
| Authentication for remote administration | **Missing** |
| Secure credential storage, no keys in logs | **Missing** (no credentials exist yet) |
| Single source of truth | Native runtime is centralized today; new features must delegate to one facade |
| Preserve existing APIs | Constraint — `/api/v1/*`, `/enr/v1/*`, Java client, CLI surface stay compatible |

## 3. Implementation Strategy (this work package)

1. **New module `enr-manager`** — the single authoritative facade (`EnrManager`): persistent `ModelStore`
   (installed-model registry + manifests), `DownloadManager` (progress/cancel/SHA-256 verify), `Catalogs`
   (Ollama registry + Hugging Face search), `ProviderRegistry` (all provider types, secret storage 0600, redaction),
   `ProviderClient` (OpenAI-compat/Anthropic/Gemini wire protocols), `ModelRouter` (explicit routing, never silent),
   `HardwareMonitor` (CPU/RAM/GPU utilization), `LogBuffer` (JUL ring buffer), `EnrConfig` (persistent config).
2. **`enr-server`** — Ollama-compatible endpoints, OpenAI-compatible endpoints, catalog/providers/diagnostics/logs/
   hardware REST APIs, static Web Control Center (vanilla JS, no frameworks), optional bearer-token auth.
3. **`enr-cli`** — `list/pull/remove/import/chat/providers/service/serve` commands; `enr service install|uninstall`
   generating a production systemd unit (auto-start, crash recovery `Restart=always`, centralized logs, health
   endpoint) with ENR Server and Web Control Center served by one process but independent of any browser.
4. **`enr-tests`** — hermetic integration tests: mock Ollama registry + mock cloud providers (local `HttpServer`),
   Ollama/OpenAI endpoint conformance, store/router/provider lifecycle, systemd unit content.
5. Keep `/api/v1/*`, `/enr/v1/*` and the Java client byte-compatible; everything new delegates to `EnrManager`.

## 4. Explicitly Documented Limitations (by design)

- Native embeddings: ENR's transformer exposes logits only → `/v1/embeddings` returns 501 for native models
  (pass-through for providers that support it). Documented in the API compatibility report.
- Native chat templates: local models are prompted with a simple role-marked template; no Jinja template engine.
- Anthropic/Gemini expose chat only; `/v1/completions` for those providers returns 501.
