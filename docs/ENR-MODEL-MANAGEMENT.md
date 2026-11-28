# ENR — Unified LLM Model Management & Provider Pull Infrastructure

Date: 2026-11-21
Status: Implemented, tested and live-verified (ENR 2.6.11, DCode 2.6.11, DStudio 2.6.11).

## 1. Purpose

One shared model management implementation for the family: ENR owns the
**model management service**, the **provider abstraction** and the **download
job infrastructure**. DCode and DStudio consume it over HTTP — they hold no
model management state, no download jobs and no provider registries of their
own. A model pulled through any surface (ENR web UI, ENR CLI, DCode RCP/web,
DStudio RCP/web) lands in the provider's store exactly once and is visible
everywhere.

```
DCode RCP (ModelManagerView) ─┐
DCode Web (Model Manager)  ───┼── DCode server /api/v1/models/* ──┐
DStudio RCP (ModelManagerView)┐│                                   │
DStudio Web (panel)       ───┼┼── DStudio server /api/models/*  ──┤
ENR Web (Model Manager)   ───┤│                                   ▼
ENR CLI (enr models …)    ───┼┼─────────────────► ENR /api/v1/models/* 
ENR Java API (EnrClient)  ───┘└───────────────────  │
                                                     ▼
                                    ModelManagementService (jobs, listeners,
                                    duplicate prevention, persistence)
                                                     │
                                     ModelProvider SPI (ServiceLoader)
                                                     │
                                        OllamaModelProvider (native HTTP API)
```

## 2. Architecture (ENR)

| Component | Module | Role |
|---|---|---|
| `ai.enr.manager.models.ModelManagementService` | enr-manager | Single owner of download jobs: provider registry, async execution, duplicate prevention, cancellation, progress listeners, durable job status |
| `ai.enr.manager.models.ModelProvider` | enr-manager | Provider SPI: capabilities, installed/available listing, blocking pull with progress + cancellation supplier, remove, ping |
| `ai.enr.manager.models.ModelProviderFactory` | enr-manager | ServiceLoader extension point (`META-INF/services/ai.enr.manager.models.ModelProviderFactory`) |
| `ai.enr.manager.models.ollama.OllamaModelProvider` | enr-manager | First provider: Ollama native HTTP API (`/api/tags`, `/api/show`, `/api/pull` streaming, `/api/delete`, `/api/version`) |
| `DownloadJob/Status/State/Progress/Layer` | enr-manager | Immutable job model; aggregate progress = sum of layer totals/completions, never per-layer averages |
| `ManagerApiHandlers` + `EnrServer` | enr-server | HTTP surface (see §4) |
| `EnrClient` | enr-java-api | Java client methods (see §5) |
| `EnrCli` | enr-cli | `enr models …` commands (see §6) |
| Web UI | enr-server/resources/web | Model Manager page (app.js) |

Model management is deliberately separate from inference selection: pulling
a model never changes the currently selected inference model; a completed
pull simply makes the model appear in the provider's installed list, which
the existing inference-model selection surfaces already consume.

## 3. Configuration

enr.yaml:

```yaml
model_providers:
  ollama.endpoint: http://localhost:11434
  ollama.connect_timeout_ms: 10000
  ollama.read_timeout_ms: 60000
  ollama.download_timeout_ms: 0      # 0 = no overall limit
  ollama.enabled: true
```

Keys are flat `<providerId>.<option>` pairs (the YAML subset supports two
levels only). Options: `endpoint`, `connect_timeout_ms`, `read_timeout_ms`,
`download_timeout_ms`, `enabled`. Each provider keeps independent
configuration; the Ollama provider is enabled by default with
`http://localhost:11434`.

Job state is persisted to `dataDir/model-download-jobs.json` (0600) after
every transition; on restart unfinished jobs are reconciled to `FAILED`
("interrupted by restart") — downloads cannot be resumed.

## 4. HTTP API

All endpoints share the ENR server (no second port). Auth: the admin bearer
token (when configured) protects mutations; GETs are public like the rest of
the management surface.

| Endpoint | Method | Semantics |
|---|---|---|
| `/api/v1/models/providers` | GET | Providers with capabilities + reachability |
| `/api/v1/models/providers/{provider}/available` | GET | Models available for download (empty without `MODEL_DISCOVERY`) |
| `/api/v1/models/providers/{provider}/installed` | GET | Installed models |
| `/api/v1/models/providers/{provider}/pull` | POST `{"model":"qwen3-vl:30b"}` | 202 + `{jobId, provider, model, status:"QUEUED"}`; duplicate pull of an active job returns the same job |
| `/api/v1/models/downloads` | GET | All jobs, newest first |
| `/api/v1/models/downloads/{jobId}` | GET | Job snapshot (status, operation, total/completed bytes, percent, layers, error, timestamps) |
| `/api/v1/models/downloads/{jobId}` | DELETE | Idempotent cancel (`{job_id, cancelled}`) |
| `/api/v1/models/providers/{provider}/{modelId}` | DELETE | Remove installed model |

Errors use the standard envelope `{"error":{"code","message"}}` with codes
`PROVIDER_NOT_FOUND`, `DOWNLOAD_NOT_FOUND`, `INVALID_REQUEST`,
`PULL_UNSUPPORTED`, `REMOVE_FAILED`. Model identifiers containing `/` must be
URL-encoded (`%2F`) in path segments.

Download states: `QUEUED → CONNECTING → DOWNLOADING → VERIFYING → COMPLETED`,
plus `FAILED`, `CANCEL_REQUESTED`, `CANCELLED`.

## 5. Java API (EnrClient)

```java
EnrClient client = new EnrClient("http://127.0.0.1:28780");

EnrClient.ModelJob job = client.pullModel("ollama", "qwen3-vl:30b");
// -> job.jobId(), job.provider(), job.model(), job.status()

JsonObject status = client.downloadStatus(job.jobId());
while (!"COMPLETED".equals(status.getString("status", ""))) {
    Thread.sleep(500);
    status = client.downloadStatus(job.jobId());
}

client.modelProviders();          // providers + capabilities
client.modelInstalled("ollama");  // installed models
client.cancelDownload(job.jobId());
client.removeModel("ollama", "llama3:8b");
```

## 6. CLI

```sh
enr models providers                  # list providers + capabilities
enr models installed ollama           # installed models of a provider
enr models available ollama           # available models (empty for Ollama)
enr models pull ollama qwen3-vl:30b   # pull with live progress polling
enr models status <jobId>
enr models cancel <jobId>
enr models remove ollama llama3:8b
```

## 7. Adding a new provider

1. Implement `ModelProvider` (capabilities must be declared explicitly) and a
   `ModelProviderFactory` with a public no-arg constructor.
2. Register the factory in
   `META-INF/services/ai.enr.manager.models.ModelProviderFactory`.
3. Configure it in enr.yaml `model_providers` (`<id>.endpoint`, timeouts,
   `enabled`). Nothing else is required — the service discovers factories via
   `ServiceLoader`, so the core `ModelManagementService` is never modified.

Provider contract: `pull(modelId, cancellationRequested, progress)` blocks
until complete/cancelled/failed; the provider must poll `cancellationRequested`
regularly and abort its native transfer. Progress is reported as
`DownloadProgress(state, totalBytes, completedBytes, error, layers)`; totals
are the provider-reported byte sums and may change during a pull.

## 8. DCode integration

- `eu.kalafatic.dcode.config.provider.ModelManagementClient` — wire client to
  ENR (`dcode.enr.endpoint` system property, default family endpoint
  `http://127.0.0.1:28780`). Returns raw JSON strings so no Jackson types
  cross the OSGi bundle boundary. `pullAndWait` backs the legacy synchronous
  `RuntimeAdapter.pullModel` contract (OllamaProvider no longer implements a
  private Ollama pull).
- `DCodeApi` `/api/v1/models/*` — passthrough surface (providers, installed,
  pull, jobs list/status/cancel, remove). Upstream failures surface as
  structured `UPSTREAM_UNAVAILABLE` (503).
- Web: Model Manager view (`web/js/models.js`, nav button "Model Manager"),
  1 s progress polling, cancel/remove.
- RCP: `eu.kalafatic.dcode.ui.views.ModelManagerView` (provider combo, model
  input, pull/refresh, live progress bars via `Display.timerExec` polling,
  cancel/remove), registered as `eu.kalafatic.dcode.ui.views.modelmanager`.

## 9. DStudio integration

- `eu.kalafatic.dstudio.teip.server.ModelManagementHttpHandler`
  (`/api/models`) — passthrough to ENR; ENR endpoint resolved from the
  configured ENR inference engine (`endpoint` config), the
  `dstudio.enr.endpoint` system property, or `http://localhost:28780`.
- `eu.kalafatic.dstudio.application.models.ModelManagementClient` — RCP-side
  client over the local DStudio backend with plain DTO records (no Jackson
  types cross bundles).
- Web: Model Manager panel in the Web IDE sidebar (app.js), same endpoints.
- RCP: `eu.kalafatic.dstudio.ui.inference.ModelManagerView`
  (`eu.kalafatic.dstudio.ui.view.modelmanager`, in the Harness perspective).

## 10. Testing

ENR (`enr-tests`, hermetic — no real Ollama needed):

- `ModelManagementServiceTest` — discovery, job lifecycle, duplicate
  prevention, cancellation, failure, concurrency, already-installed skip,
  listener notification, persistence/restart recovery.
- `OllamaModelProviderTest` — against a mock Ollama API (`MockServers.OllamaMock`):
  tags/show/version, streaming pull with multiple layers and changing totals,
  error lines, HTTP failures, truncated streams, cancellation, read timeout,
  download timeout, removal.
- `ModelManagementHttpTest` — the REST surface end-to-end incl. the
  `EnrClient` methods.

DCode: `eu.kalafatic.dcode.tests.server.ModelManagementApiTest` (mock ENR,
9 cases). DStudio: `eu.kalafatic.dstudio.tests.ModelManagementHttpApiTest`
(mock ENR, 8 cases).

Live verification performed on this host (real Ollama 0.40.16 at
localhost:11434, real ENR server, real DStudio headless product):

- providers → `ollama` reachable; installed → 23 real models;
- `smollm:135m` pulled through ENR end-to-end (12% → 49% → 87% → COMPLETED,
  91,727,784 bytes, aggregate layer totals);
- already-installed pull → COMPLETED "already installed" without a download;
- live cancel: DOWNLOADING 6% → CANCELLED (ENR), DOWNLOADING 19% → CANCELLED
  (through the DStudio surface);
- jobs persisted across an ENR restart;
- `qwen3-vl:30b` was NOT downloaded (20 GB) — ordinary builds/tests never
  download large models.

## 11. Known limitations

- Download jobs survive restart as `FAILED` only; no resume of interrupted
  transfers (same policy as the existing catalog download manager).
- Ollama has no native server-side pull cancellation; ENR cancels by aborting
  the streaming HTTP connection — verified live.
- Providers are static after startup (config changes require a restart);
  the SPI supports adding providers without touching the core service.
- DCode's legacy `POST /api/v1/runtimes/{id}/models/{modelId}/pull` remains
  for API compatibility but now executes through the shared ENR service
  (`ModelManagementClient.pullAndWait`), not a private Ollama implementation.
