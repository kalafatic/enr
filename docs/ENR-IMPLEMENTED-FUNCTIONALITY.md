# ENR — Implemented Functionality Report

Date: 2026-11-19
Status: Implemented, built, tested and verified end-to-end (see verification sections below).

## 1. New Module: `enr-manager` — Single Source of Truth

Every surface (Web UI, CLI, REST API, Ollama/OpenAI APIs, systemd service) delegates to
`ai.enr.manager.EnrManager`. There is exactly one model registry (`ModelStore`), one provider
registry (`ProviderRegistry`), one download manager, one router and one diagnostics pipeline.

| Class | Responsibility |
|---|---|
| `EnrManager` | Top-level facade: models, providers, downloads, chat, dashboard, diagnostics, logs |
| `EnrConfig` / `EnrConfigLoader` | Persistent configuration (enr.yaml YAML subset) incl. storage dirs, catalogs, auth token |
| `store.ModelStore` / `store.ModelManifest` | Persistent installed-model registry; import, remove, SHA-256 verify, artifact detection |
| `download.DownloadManager` | Background downloads with progress, cancellation, digest verification, job listing |
| `catalog.Catalogs` | Search + download resolution: Ollama Library (registry manifest + blob) and Hugging Face (tree API + resolve URLs, LFS sha256) |
| `provider.ProviderRegistry` / `Provider` | Provider CRUD, connection tests, model discovery (60s cache), health, 0600 state file, key redaction |
| `provider.ProviderClient` | Wire protocols: OpenAI-compatible chat (OpenAI, DeepSeek, OpenAI-compat, Ollama /v1, llama.cpp /v1), Anthropic Messages, Gemini generateContent; streaming + embeddings passthrough |
| `router.ModelRouter` | Deterministic routing: local models win, explicit `provider:model` prefix, single-match discovery; ambiguity and unknowns fail loudly — never silent switching |
| `monitor.HardwareMonitor` | CPU load, RAM, GPU utilization/VRAM/temperature (CUDA driver + nvidia-smi), genuine values only |
| `logs.LogBuffer` | In-memory ring buffer (JUL handler) + file logging to `log_dir` |
| `service.SystemdSupport` | Production systemd unit generation, install/uninstall, default system config |
| `models.ModelManagementService` | Unified provider model management: provider registry (ServiceLoader), async pull jobs with duplicate prevention, cancellation, listeners and durable status (see `ENR-MODEL-MANAGEMENT.md`) |
| `models.ollama.OllamaModelProvider` | First provider: Ollama native HTTP API (`/api/tags`, `/api/show`, `/api/pull` streaming, `/api/delete`, `/api/version`) |

## 2. Model Management (implemented)

- Discover: installed models (managed dirs) + bare `.gguf`/`.evo` files in `runtime.model_directories`
- Search catalogs: Ollama Library (`ollama.com/library?q=`, verified live) and Hugging Face (`/api/models?search=`, verified live)
- Pull: `enr pull NAME[:TAG]`, `enr pull hf:REPO/FILE.gguf`, `POST /api/v1/catalog/pull`, Ollama `POST /api/pull` (NDJSON progress stream)
- Progress: live byte progress in CLI, REST job listing and NDJSON; cancel: `POST /api/v1/downloads/cancel`, `enr` UI Cancel
- Verify: SHA-256 over the downloaded artifact against the catalog digest (Ollama layer digest / HF LFS oid) — verified live with a real 637 MB `tinyllama` pull
- Import: `enr import path.gguf`, `POST /api/v1/catalog/import` (copies, records digest)
- Inspect: manifests (digest, size, source, format, installed_at) + live runtime state merge (loaded, backend, sessions)
- Start/stop: `enr`, REST `/api/v1/models/start|stop`, Web UI; state verified against the real runtime — a model is reported running only if `ModelInstance` exists and is READY
- Remove: `enr remove`, REST DELETE, Ollama `DELETE /api/delete`
- Missing/corrupted artifacts: `artifact_present` flag, `POST /api/v1/catalog/verify` recomputes the digest

## 3. Provider Management (implemented)

- Types: `deepseek`, `openai`, `anthropic`, `gemini`, `openai-compatible`, `ollama`, `llama.cpp` (local), plus the built-in native runtime
- Configuration: `enr provider add`, REST `POST /api/v1/providers`, Web UI form; base URLs defaulted per type
- Connection testing: model-list probe; results persisted (`last_test_ok`, `connected` window)
- Model discovery: per-provider remote model listing with 60s cache
- Health monitoring: `/api/v1/providers/{id}/health`
- Error diagnostics: provider error messages surfaced in chat responses and logs; no secrets in messages
- Secure storage: API keys live only in `<data_dir>/state.json` (0600). Every API response redacts the key (`api_key_configured` boolean only). Verified by test: key never appears in provider listings.

## 4. Model Router (implemented, never silent)

Resolution order: (1) exact installed local model, (2) explicit `provider:model` (or `provider/model`),
(3) exactly one enabled provider exposing the name. Multiple matches raise an error listing the
providers; unknown names raise an error suggesting pull/provider configuration. Verified by tests.

## 5. Runtime Management

- Models remain the primary UX: start/stop by model id; lazy auto-load on first chat request (Ollama-like),
  with real READY-state verification
- Backend selection automatic: `cpu` default, `cuda` when requested and available, explicit fallback policy
- Advanced inference parameters (temperature/top_p/top_k/seed/stop/max tokens) accepted by all chat surfaces

## 6. System Service (systemd)

- `enr service unit` prints the unit; `enr service install` writes `/etc/systemd/system/enr.service`,
  `systemctl daemon-reload` + `enable`; `uninstall`/`status` provided
- Unit properties: `After=network-online.target`, `Restart=on-failure` + `RestartSec=5` (crash recovery),
  `KillSignal=SIGTERM` + shutdown hook that unloads models (graceful shutdown), `SuccessExitStatus=143`,
  `LimitNOFILE=65536`, stdout/stderr appends to `log_dir`
- Boot startup via `WantedBy=multi-user.target`
- The single process serves both the ENR Server API and the Web Control Center; closing a browser
  never affects inference (stateless HTTP UI over the same server)
- Unit validated with `systemd-analyze verify` (clean exit; only unrelated host warnings)
- Persistent configuration and model registry under `/etc/enr` + `/var/lib/enr/models`; centralized
  logs in `/var/log/enr` plus the `/api/v1/logs` endpoint
- Health endpoint: `GET /api/v1/health` (200/503 with check list) — suitable for `ExecStartPost`/monitoring

## 7. Web Control Center (dependency-free)

Single-page vanilla HTML/CSS/JS app served at `/` (classpath resources, no frameworks, no build step).
Pages: Dashboard (status, uptime, API endpoints, active models, GPU/VRAM/RAM/CPU, active requests),
Models (list, search, pull, import, progress with cancel, start/stop, details, verify, remove),
Providers (add/edit/remove, test, refresh models, health badges), Runtime (loaded instances, devices,
sessions), Playground (streaming chat + side-by-side model comparison), Diagnostics (live logs, system
state, API endpoint pings). Optional Bearer-token auth (`server.auth_token`) covers the admin surface;
the UI stores the token in localStorage and never sends credentials elsewhere.

## 8. API Compatibility

See ENR-API-COMPATIBILITY.md for the endpoint-by-endpoint matrix and test evidence.
Summary: Ollama `/api/tags`, `/api/show`, `/api/pull`, `/api/delete`, `/api/chat`, `/api/generate`,
`/api/ps`, `/api/version` — implemented and integration-tested. OpenAI `/v1/models`,
`/v1/chat/completions` (sync + SSE streaming with usage + `[DONE]`), `/v1/completions`,
`/v1/embeddings` (providers only; explicit 501 for native — documented). Existing `/api/v1/*` and
`/enr/v1/*` surfaces and the Java client are unchanged.

## 9. Security

- 0600 state file for provider keys; keys redacted from all API responses and never logged
- Optional Bearer-token admin auth (verified 401/200 behavior by test)
- Configurable bind host (default 127.0.0.1; system install uses 0.0.0.0)
- Downloads verified by SHA-256 before registration; `.part` files removed on failure/cancel
- Local vs cloud separation is explicit in every response (`local` flag, `provider` field)

## 10. Verification Evidence (all executed on this machine)

- Build: `JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64 mvn clean package` — SUCCESS
- Tests: 88 tests green (54 pre-existing + 34 new). CUDA GPU-op unit tests (3) currently fail on this
  host because an external process holds ~14.4 GB of the 16.3 GB VRAM (`cuCtxCreate: 2` = OOM);
  they passed earlier the same day before that process started and are unrelated to these changes.
- Live E2E against a running server (port 8211):
  - Web UI served at `/` (HTML)
  - `POST /api/v1/catalog/import` (tiny-evo) → listed, digest recorded, verified=true
  - `POST /api/v1/models/start` → `loaded: true, state: ready, backend: cpu`
  - `POST /api/chat` → real token streaming (NDJSON, `done: true`, eval counts)
  - `POST /api/pull` (tinyllama, 637 MB, registry.ollama.ai) → progress lines, digest verification,
    `success`; model registered with `source=catalog:tinyllama:latest`, verified=true
  - Start of the quantized real model fails with an explicit, actionable message
    (Q4_K_M not supported by the native runtime — documented limitation)
  - Real catalog search verified against both ollama.com and huggingface.co
  - Provider E2E against a local OpenAI-compatible mock: add → test (2 models discovered) →
    `/api/chat` and `/v1/chat/completions` (sync + streaming) routed through the provider;
    embeddings passthrough; API key never returned by the API
  - CLI verified: `enr list`, `enr provider list`, `enr chat` (streaming) against the live server
  - `enr service unit` output verified with `systemd-analyze verify` (exit 0)
  - Stop model: `POST /api/v1/models/stop` → `{"stopped":"tiny-evo"}`
  - Diagnostics/logs/hardware endpoints return genuine data (CPU/RAM/GPU from the driver)

## 11. Remaining Gaps (explicit)

1. Quantized GGUF tensors (Q4_K_M etc.) cannot be executed by the ENR native runtime (F32/F16 only).
   Downloads, verification and catalog integration work; native execution returns a clear error.
2. Native `/v1/embeddings` returns 501 (the transformer exposes logits only). Provider pass-through works.
3. Anthropic/Gemini support chat; `/v1/completions` for those types returns a routing error via the chat path.
4. Local chat templates use a simple role-marked prompt (no Jinja); tool-calling and images are
   not supported for local models (images are marked as `[image input]` in Ollama chat).
5. Ollama `POST /api/pull` does not support `insecure`/streaming of multiple tags; single-tag pulls only.
6. The native `/enr/v1/providers` family endpoint still reports the built-in provider only
   (preserved for DCode/TRYST/DStudio compatibility).
7. CUDA ops unit tests blocked on this host by an external GPU-memory consumer (see §10).
