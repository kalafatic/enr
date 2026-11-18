# ENR Deployment Guide

Date: 2026-11-16

## 1. Requirements

- JDK 21+ for CPU-only operation; **JDK 25+ for the CUDA backend** (the CUDA module
  uses the final FFM API)
- Maven 3.8+ to build
- Optional: NVIDIA GPU + driver (`libcuda.so.1`); no CUDA toolkit required at runtime
  (PTX kernels are embedded and driver-JIT compiled)

## 2. Build

```bash
export JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64   # or java-21 for CPU-only
mvn clean install
```

Produces 12 artifacts under `ai.enr:*`. Runtime dependencies are JDK-only.

## 3. Configuration (`enr.yaml`)

```yaml
runtime:
  preferred_backend: cpu          # cpu | cuda
  model_directories:
    - /path/to/models             # *.evo / *.gguf auto-loaded at startup
  max_concurrent_sessions: 8
  default_max_tokens: 256
  session_idle_timeout_seconds: 1800
server:
  host: 127.0.0.1
  port: 8111
```

All keys optional; defaults shown above.

## 4. Running

```bash
enr start -c enr.yaml            # starts server, auto-loads models
enr stop                         # graceful shutdown via pid file
enr status
enr models
enr load /path/model.gguf --backend cuda
enr unload model --force
enr infer model "Hello" --max-tokens 64 --temperature 0.7 --stream
enr devices
enr diagnostics
```

The CLI talks to a running server over HTTP (client mode); `enr start` hosts the
runtime in-process. ENR operates standalone — no Ollama, llama.cpp, EVO, or any
external service is required.

## 5. Operations Notes

- Models are identified by file name (without extension); loading an already
  loaded id returns the existing instance (backend is chosen at first load)
- A model with active sessions cannot be unloaded without `--force`
- Sessions: per-request implicit (default) or persistent via `session_id` in
  inference requests; persistent sessions expire after the idle timeout
- Cancellation: `POST /api/v1/inference/cancel` with the stream's `request_id`
- Backends: `cpu` and `cuda`; when CUDA is unavailable ENR falls back to CPU and
  logs the reason; `GET /api/v1/devices` reports genuine device state

## 6. Model Compatibility

Supported model files: `.evo` v2 (EVO Native Model Protocol), `.evo` v1 (legacy,
read-only), GGUF v2/v3 with `general.architecture=llama` (the Darwin Forge export
format). See ENR-MODEL-FORMAT-COMPATIBILITY.md for layout conventions and the
`.evo` ↔ GGUF verification methodology.
