# ENR — Installation & Operations Guide

Date: 2026-11-19

## 1. Requirements

- Linux (systemd for the service mode), JDK 21 (CPU-only) or JDK 25 (CUDA backend). Build: Maven 3.8+.
- Optional: NVIDIA GPU + driver (CUDA backend), `nvidia-smi` (utilization metrics only).

## 2. Build

```bash
export JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64   # or java-21 for CPU-only
mvn clean package            # runs tests; append -DskipTests to skip
./enr.sh help                # the shaded CLI jar is enr-cli/target/enr-cli-2.6.11-SNAPSHOT.jar
```

## 3. Quick start (foreground)

```bash
./enr.sh serve -c enr.yaml                 # server + Web Control Center
# http://127.0.0.1:8111/                   Web UI
# http://127.0.0.1:8111/api                Ollama-compatible API
# http://127.0.0.1:8111/v1                 OpenAI-compatible API
```

`enr start` behaves identically but writes `/tmp/enr.pid`; `enr stop` stops it.

## 4. Models

```bash
enr list                                   # installed models
enr pull tinyllama                          # Ollama Library (NAME[:TAG])
enr pull hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf
enr pull llama3.2:1b --no-stream            # blocking pull
enr import /path/to/model.gguf --id my-llm
enr remove tinyllama
enr load  my-llm --backend cuda            # load a raw path (legacy)
enr infer my-llm "hello" --stream
enr chat  my-llm "hello" --multi           # interactive chat (cloud: enr chat deepseek:deepseek-chat)
```

Models live in `storage.models_dir` (default `~/.enr/models/<id>/<id>.gguf` + `manifest.json`).
Downloads are SHA-256 verified against the catalog digest before registration.

## 5. Providers

```bash
enr provider add --type deepseek --api-key sk-...              # id defaults to the type
enr provider add --type openai --api-key sk-... --id openai
enr provider add --type anthropic --api-key sk-ant-... --id claude
enr provider add --type gemini --api-key AIza... --id gemini
enr provider add --type openai-compatible --base-url http://127.0.0.1:11434/v1 --id ollama-local
enr provider list
enr provider test deepseek
enr provider models deepseek
enr provider remove deepseek
```

Keys are stored in `<data_dir>/state.json` (permissions 0600) and are never returned by the API or
written to logs. Cloud models are addressed as `provider:model` (e.g. `deepseek:deepseek-chat`).

## 6. System service (systemd)

```bash
sudo ./enr.sh service install \
  --jar $PWD/enr-cli/target/enr-cli-2.6.11-SNAPSHOT.jar \
  --user enr --log-dir /var/log/enr
sudo systemctl start enr && systemctl status enr
sudo systemctl enable enr        # already enabled by install; starts at boot
```

- Unit: `/etc/systemd/system/enr.service` (`Restart=on-failure`, `RestartSec=5`, SIGTERM graceful
  shutdown with model unloading, log append to `/var/log/enr`, `multi-user.target`)
- Config: `/etc/enr/enr.yaml` (created from a system template: binds `0.0.0.0:8111`,
  models at `/var/lib/enr/models`, logs at `/var/log/enr`)
- Remove: `sudo ./enr.sh service uninstall`; inspect: `enr service unit | systemd-analyze verify`
- Health endpoint for monitoring: `GET /api/v1/health` → 200 healthy / 503 with checks

The service is a single process serving both the API and the Web Control Center. Closing the
browser never terminates inference; the UI talks to the same REST API as every other client.

## 7. Configuration reference (`enr.yaml`)

```yaml
server:
  host: 127.0.0.1
  port: 8111
  auth_token: ""              # optional Bearer token for the admin API + Web UI

runtime:
  preferred_backend: cpu      # cpu | cuda (falls back to cpu unless allow_cuda_fallback: false)
  allow_cuda_fallback: true
  model_directories: []       # bare .gguf/.evo files discovered here are treated as installed
  max_concurrent_sessions: 8
  default_max_tokens: 256
  session_idle_timeout_seconds: 1800
  request_timeout_seconds: 300
  enable_metrics: true

storage:
  data_dir: ~/.enr
  models_dir: ~/.enr/models
  log_dir: ~/.enr/logs

catalogs:
  ollama_registry: https://registry.ollama.ai
  ollama_search: https://ollama.com
  huggingface_api: https://huggingface.co
```

## 8. Web Control Center

Open `http://<host>:<port>/`. Pages: Dashboard, Models (discover/pull/import/start/stop/verify/
remove with progress), Providers, Runtime, Playground (streaming chat + model comparison),
Diagnostics (logs/system/API pings). When `server.auth_token` is set, press the Key button and
enter the token.

## 9. Operations notes

- Logs: JUL → console (systemd captures to journal) + `<log_dir>/enr.log` (rotating, 3×5 MB) +
  in-memory buffer exposed at `/api/v1/logs?source=&tail=`
- Never-silent routing: local ids win; `provider:model` is explicit; ambiguous/unknown names error
  with alternatives
- Quantized GGUF (Q4_K_M, ...): download/verify/install works; native execution requires F16/F32
  exports — the start call explains this and suggests a cloud provider
- Backups: models dir + data dir (`state.json` contains provider configuration)
