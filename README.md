# EVO Native Runtime (ENR)

**Simple, production-ready LLM runtime and provider gateway — with an Ollama-like experience.**

ENR loads and executes native EVO models (`.evo` v2/v1 and GGUF v2/v3 exports) with its own compute
engines — no Ollama, no llama.cpp, no external inference server required. On top of the native
runtime it provides a unified gateway: local model management, cloud providers (DeepSeek, OpenAI,
Anthropic, Gemini, OpenAI-compatible), Ollama-compatible and OpenAI-compatible APIs, a Web Control
Center, a CLI and systemd integration.

## Verified Capabilities

- **Native inference**: CPU (deterministic reference) and CUDA (FFM driver API + PTX, RTX 5070 Ti
  verified) — F32/F16 tensors
- **Model management**: discover, search catalogs (Ollama Library + Hugging Face), pull with live
  progress + cancellation + SHA-256 verification, import, inspect, start/stop, remove
- **Provider gateway**: local (Ollama, llama.cpp, OpenAI-compatible servers) and cloud (DeepSeek,
  OpenAI, Anthropic, Gemini) providers with connection testing, model discovery, health and
  secure credential storage (0600, redacted everywhere)
- **Model router**: deterministic, never silent — local ids win, explicit `provider:model` prefixes,
  ambiguous names fail loudly
- **APIs**: Ollama-compatible (`/api/tags|show|pull|delete|chat|generate|ps|version`) and
  OpenAI-compatible (`/v1/models|chat/completions|completions|embeddings`) plus the native
  `/api/v1/*` and family `/enr/v1/*` surfaces — see docs/ENR-API-COMPATIBILITY.md
- **Web Control Center**: dependency-free single-page UI (Dashboard, Models, Providers, Runtime,
  Playground, Diagnostics) served by the same server
- **Systemd service**: `enr service install` — boot start, crash recovery, graceful shutdown with
  model unloading, centralized logs, HTTP health endpoint
- **Java 21 client** (`enr-java-api`), sync/async/streaming — unchanged

## Quick Start

```bash
export JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64   # JDK 25 (CUDA) or 21 (CPU-only)
mvn clean install
./enr.sh serve -c enr.yaml                            # server + Web UI on http://127.0.0.1:8111/

./enr.sh pull tinyllama                               # download from the Ollama Library
./enr.sh pull hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf
./enr.sh list
./enr.sh chat tinyllama "hello" --multi
./enr.sh infer tiny-evo "hello" --stream              # native /api/v1 inference
./enr.sh provider add --type deepseek --api-key sk-...
./enr.sh chat deepseek:deepseek-chat "hello"
./enr.sh service install                              # production systemd unit (root)
./enr.sh diagnostics
```

## Modules

`enr-core` · `enr-model-loader` · `enr-tokenizer` · `enr-compute` (+`-cpu`, `-cuda`) ·
`enr-inference` · `enr-runtime` · `enr-manager` (NEW: single source of truth for models, providers,
downloads, routing, diagnostics) · `enr-server` · `enr-java-api` · `enr-cli` · `enr-tests`

## Documentation

- `docs/ENR-SIMPLIFICATION-AUDIT.md` — architecture audit & gap report
- `docs/ENR-IMPLEMENTED-FUNCTIONALITY.md` — what was built, with verification evidence
- `docs/ENR-API-COMPATIBILITY.md` — endpoint-by-endpoint compatibility matrix
- `docs/ENR-INSTALLATION-OPS.md` — installation & operational guide
- `docs/` — architecture baseline, model format audit, compute backends, API specification,
  test/performance reports, deployment guide, future integration specs

## Design Principles

- One backend: the Web UI, CLI, REST API and system service all delegate to `EnrManager` — no
  duplicate registries, routers or lifecycles
- Zero runtime dependencies beyond the JDK
- The `.evo`/GGUF formats are interoperability contracts, not shared runtime code
- CPU is the numerical reference; CUDA must agree within tolerance
- No invented numbers: every metric comes from real executions; a model is reported running only
  when its real runtime state is verified
- Never silent: ambiguous or unknown model names produce explicit errors with alternatives
- No credentials in logs or API responses
