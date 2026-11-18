# EVO Native Runtime (ENR)

**Native AI Model Inference & Orchestration Engine**

ENR is a standalone, independent inference runtime that loads and executes native
EVO models (`.evo` v2/v1 and Darwin Forge GGUF v3 exports) with its own compute
engines — no Ollama, no llama.cpp, no external inference server, no Eclipse
RCP/OSGi.

## Verified Capabilities

- **Model loading**: `.evo` v2 (read+write), `.evo` v1 (read), GGUF v2/v3
  (llama family) — validated against real Darwin Forge exports (`evo.gguf`,
  `tiny-evo.gguf`)
- **CPU inference**: deterministic reference backend — transformer execution,
  RMSNorm, causal multi-head attention (GQA), RoPE/sinusoidal positional
  encoding, SwiGLU FFN, KV cache, sampling (greedy/temperature/top-K/top-P/
  penalties), streaming generation
- **CUDA inference**: FFM bindings to the CUDA driver API + embedded PTX kernels,
  verified on NVIDIA RTX 5070 Ti — numerically equivalent to CPU and to a
  llama.cpp oracle (≤2e-3, F16-KV-cache noise)
- **Multi-model runtime**: registry, lifecycle, concurrency, session isolation,
  resource accounting, controlled eviction
- **HTTP API** (`/api/v1`): health, status, models, load/unload, inference,
  SSE streaming, cancellation, sessions, metrics, devices
- **Java 21 client** (`enr-java-api`), sync/async/streaming
- **CLI**: start/stop/status/models/load/unload/infer/devices/diagnostics

## Quick Start

```bash
export JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64   # JDK 25 (CUDA) or 21 (CPU-only)
mvn clean install
./enr.sh start -c enr.yaml                            # starts the server
./enr.sh load models/tiny-evo.gguf --backend cuda     # or omit --backend for CPU
./enr.sh infer tiny-evo "hello" --stream
./enr.sh diagnostics
```

## Modules

`enr-core` · `enr-model-loader` · `enr-tokenizer` · `enr-compute` (+`-cpu`, `-cuda`)
· `enr-inference` · `enr-runtime` · `enr-server` · `enr-java-api` · `enr-cli` ·
`enr-tests`

## Documentation

See `docs/` for the architecture baseline, model format audit and compatibility,
compute backends, API specification, test report, performance report, deployment
guide and future integration specifications.

## Design Principles

- Zero runtime dependencies beyond the JDK
- The `.evo`/GGUF formats are interoperability contracts, not shared runtime code
- CPU is the numerical reference; CUDA must agree within tolerance
- No invented numbers: every metric in the reports comes from real executions
