# ENR Initial Test Report

Initial functional testing, integration verification, defect correction and stabilization iteration for ENR (EVO Native Runtime).

- Iteration date: 2026-11-23.
- Primary agent: OpenCode with cloud DeepSeek (`deepseek/deepseek-v4-pro`), confirmed by the harness model identity and `~/.config/opencode/opencode.jsonc` (`deepseek` provider → `deepseek-v4-pro`).
- No local Ollama model was used as the primary agent. Local ENR-native inference was tested through ENR itself.

> This report records a **new iteration** against the relocated repository
> `/home/petr/git/enr-221126` (remote `github.com/kalafatic/enr`). A prior iteration
> (2026-11-21) ran against the former path `/home/petr/git/evo-native-runtime-161126`;
> its fixes are already present in this repository's source and are re-verified here.
> The original 2026-11-21 baseline is retained in the git history of this repository.

## 1. Environment

| Item | Value |
| --- | --- |
| ENR repository | `/home/petr/git/enr-221126` (remote: github.com/kalafatic/enr), branch `master`, clean tree at start |
| ENR version | 2.6.11-SNAPSHOT |
| ENR endpoint | `http://127.0.0.1:8111` (enr.yaml) |
| ENR start/stop | `./enr.sh serve -c enr.yaml` (shaded CLI jar); SIGTERM for graceful shutdown |
| Build | `JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64 mvn package` — 13 modules, `release=21` |
| Java | JDK 21 (`21.0.13.1`) and JDK 25 (`25.0.5.1`) installed; default `java` is 1.8 (unused) |
| Maven | 3.8.7 |
| Hardware | 64 vCPU, 121 GiB RAM, NVIDIA RTX 5070 Ti (16 GB, driver 595.99.02) |
| OpenCode | v2.4.0 (`/home/petr/.opencode/bin/opencode`) |
| OpenCode agent model | `deepseek/deepseek-v4-pro` (cloud) — this agent |
| Real Ollama (reference) | 0.40.16 at `http://127.0.0.1:11434` |
| ENR local models | `evo` (53952 params, ctx 4), `tiny-evo` (920 params, ctx 32) — GGUF, CPU backend |
| ENR data dir | `~/.enr` (models, logs, provider registry) |
| Audit storage | `/home/petr/.evo/audit/` (evo-audit engine, `/home/petr/git/evo-audit-221126`) |

## 2. Architecture observations

- `enr-server` (JDK `com.sun.net.httpserver`, virtual threads, zero external deps) exposes:
  - Native `/api/v1/*` (`health/status/models/models/load|unload/inference/inference/stream/inference/cancel/sessions/metrics/devices`).
  - Family `/enr/v1/*` (`health/models/chat/chat/stream/runtime/providers/capabilities`).
  - Ollama-compatible `/api/*` (`version/tags/show/pull/delete/chat/generate/ps`).
  - OpenAI-compatible `/v1/*` (`models/chat/completions/completions/embeddings`).
  - Management REST `/api/v1/{catalog,models,providers,downloads,chat,dashboard,diagnostics,logs,hardware}`.
  - Web Control Center at `/`.
- `EnrManager` is the single facade: routing (`ModelRouter`) → local (`EnrRuntime`) or provider (`ProviderClient`).
- `ProviderClient` supports OpenAI-compatible (DeepSeek/OpenAI/Ollama `/v1`/llama.cpp), Anthropic, Gemini; tool calling is relayed only for OpenAI-compatible routes and rejected explicitly (400) elsewhere.
- Local native inference is a real transformer forward pass (CPU/CUDA) over `.evo`/GGUF models; the bundled `evo`/`tiny-evo` are synthetic fixtures with vocabularies of literal `token_N` strings, so their decoded output is deterministic but not meaningful text.
- `ChatMessage` carries text plus optional tool-call/reasoning payload; tool calling is fully wired for OpenAI-compatible provider routes.

## 3. Initial Baseline (before this iteration's modifications)

Legend: PASS / PARTIAL / FAIL / BLOCKED / NOT TESTED.

Baseline build: `mvn clean package` — **BUILD SUCCESS**. Automated tests: **156 tests, 0 failures** (all modules; `enr-tests` = 106).

| # | Area | Result | Evidence |
| --- | --- | --- | --- |
| B1 | Runtime lifecycle (start/load/unload/discovery) | PASS | `evo` + `tiny-evo` auto-loaded on CPU; `/api/ps`, `/api/tags` correct |
| B2 | Native inference (CPU) | PASS | `tiny-evo "hello"` → deterministic `token_4`; tokens `[4,2]` (EOS excluded from text) |
| B3 | Streaming (native SSE, Ollama NDJSON, OpenAI SSE) | PASS | correct framing + final done chunk |
| B4 | Context / multi-turn | PASS | prompt flattening; no cross-request contamination (deterministic repeats) |
| B5 | Stability / concurrency | PARTIAL | `max_concurrent_sessions=8` enforced; excess → HTTP 500 (see F4) |
| C | Providers | PASS | none configured at baseline; add/test/remove + key redaction verified |
| D1 | Native API | PASS | all endpoints respond |
| D2 | Ollama protocol | PASS | version/tags/show/ps/chat/generate; capabilities + model_info present |
| D2 | `/api/show` `{"model":...}` fallback | PASS | (fix from prior iteration re-verified) |
| D3 | Tool calling (provider route) | PASS | live DeepSeek structured `tool_calls` + streaming + round-trip |
| D3 | Tool calling (native text-only model) | PASS | explicit 400 "does not support tools" (matches real Ollama) |
| D4 | Malformed JSON status | **FAIL** | returned **404**; should be **400** (real Ollama returns 400) |
| D4 | Unknown model | PASS | 404 plain-string error |
| D4 | Missing model | PASS | 400 |
| E1 | OpenCode + cloud DeepSeek | PASS | this agent (deepseek/deepseek-v4-pro) |
| E2 | OpenCode + ENR | PARTIAL | discovery + routing verified; agentic inference BLOCKED (text-only native model) |
| F | Terminal execution | PASS | real shell, real outputs |
| G1 | Java end-to-end (DeepSeek agent) | PASS | exact `ENR_AUTONOMOUS_TEST_OK`, exit 0 |

## 4. Root-cause analysis (this iteration)

1. **Malformed JSON → 404 (should be 400)** — `Json.parse` throws `IllegalArgumentException`;
   the handlers' `catch (IllegalArgumentException e) → 404` (intended for "model not found") also
   catches JSON parse errors. Affects `/api/chat`, `/api/generate`, `/api/show`, `/api/delete`,
   `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/api/v1/chat`.
   Real Ollama returns `400 {"error":"invalid character ..."}`.
2. **`EnrClient.unloadModel()`/`cancel()` silently ignore HTTP errors** — these methods call
   `request(...)` but never `asObject(...)`, so a 404 ("Model not loaded") or 409 ("model in use")
   is swallowed; `enr unload nope` printed "Unloaded model: nope".
3. **`enr.yaml` stale `model_directories`** — pointed at the former repo path
   `/home/petr/git/evo-native-runtime-161126/...`; models auto-loaded from the old location.
4. **Concurrent-session-limit rejection → 500** (finding, not fixed) — `SessionManager` throws
   `IllegalStateException`, which maps to the generic `catch (Exception) → 500`; should be 429/503.

## 5. Fixes implemented (this iteration)

### F1. Malformed JSON → 400 on protocol endpoints
- **Files**: `enr-core/.../json/JsonParseException.java` (NEW), `enr-core/.../json/Json.java`,
  `enr-server/.../OllamaApiHandlers.java`, `OpenAiApiHandlers.java`, `ManagerApiHandlers.java`.
- **Root cause**: see §4.1.
- **Correction**: new `JsonParseException extends IllegalArgumentException` thrown by `Json.parse`;
  handlers catch it specifically → 400 before the `IllegalArgumentException` → 404 catch.
  (`/v1/embeddings` body parse moved inside its try.)
- **Verification**: malformed JSON → 400 on `/api/chat`, `/api/generate`, `/api/show`,
  `/v1/chat/completions`; unknown model still 404. Regression tests: `chatMalformedJsonReturns400`,
  `generateMalformedJsonReturns400`, `openAiChatMalformedJsonReturns400`.

### F2. `EnrClient.unloadModel()`/`cancel()` propagate errors
- **File**: `enr-java-api/.../EnrClient.java`.
- **Root cause**: see §4.2.
- **Correction**: wrap both in `asObject(request(...))`.
- **Verification**: `enr unload nope` → `ERROR: Model not loaded: nope` (was success).
  Regression test: `unloadModelNotLoadedFails` (expects `EnrException` status 404).

### F3. `enr.yaml` model directory path
- **File**: `enr.yaml`.
- **Correction**: `model_directories` → `/home/petr/git/enr-221126/enr-model-loader/src/test/resources/models`.
- **Verification**: server log shows models loaded from the current repo path.

## 6. Post-fix results

- Build: `mvn clean package` — **BUILD SUCCESS** (13 modules).
- Tests: **160 tests, 0 failures** (156 baseline + 4 new regression tests; `enr-tests` = 110).

| Scenario | Before | After |
| --- | --- | --- |
| Malformed JSON `/api/chat` | FAIL (404) | **PASS (400)** |
| Malformed JSON `/api/generate` | FAIL (404) | **PASS (400)** |
| Malformed JSON `/api/show` | FAIL (404) | **PASS (400)** |
| Malformed JSON `/v1/chat/completions` | FAIL (404) | **PASS (400)** |
| Unknown model | PASS (404) | PASS (404) |
| `enr unload` of non-loaded model | FAIL (silent success) | **PASS (error)** |
| Model dir resolves to current repo | FAIL (stale path) | **PASS** |
| Native inference determinism | PASS | PASS |
| Live DeepSeek tool calling (Ollama + OpenAI + streaming) | PASS | PASS |
| OpenCode + DeepSeek (this agent) | PASS | PASS |
| Java end-to-end | PASS | PASS |

## 7. Remaining limitations

- **L1 — ENR-native models are text-only toys**: `tiny-evo` (920 params, ctx 32) and `evo`
  (53952 params, ctx 4) are synthetic fixtures with a literal `token_N` vocabulary. They cannot
  produce meaningful text, and structured tool calling on native models is rejected with 400
  (`capabilities: ["completion"]` only) — this mirrors real Ollama for models without the `tools`
  capability.
- **L2 — OpenCode agentic inference through ENR-native inference is BLOCKED**: OpenCode's agentic
  `build` agent always sends tool definitions; ENR's text-only native models reject them (400).
  Discovery and request routing through ENR are verified, and the rejection is honest, but a
  successful agentic multi-turn through a native model is not possible with the bundled models.
- **L3 — concurrent-session-limit rejection uses HTTP 500** instead of 429/503 (finding F4, deferred).
- **L4 — native `/api/v1/inference` requires an explicit model load** (no on-demand auto-load),
  unlike the Ollama `/api/chat`/`/api/generate` path which auto-loads via `ensureLoaded`. This is a
  deliberate low-level vs high-level API distinction.
- **L5 — Anthropic/Gemini tool translation** not implemented; tool requests on those provider types
  fail explicitly with 400 (no silent dropping) — unchanged from the prior iteration.
- **L6 — `/v1/embeddings`** returns 501 for local models (native runtime exposes logits only);
  supported only via OpenAI-compatible providers.

## 8. Audit evidence

- Audit run persisted (append-only) under `/home/petr/.evo/audit/`:
  - runId: `enr-func-test-20261123-iter1` (13 modules, 4 failures recorded, `DEGRADED`).
  - A pre-existing run `evo-native-v2-integrity-20261123` (separate EVO-platform campaign) was preserved, not overwritten.
- Server log: `/tmp/opencode/enr-campaign/enr-server.log`.
- Java e2e artifact: `/tmp/opencode/enr-campaign/java-e2e/` (`ENR_AUTONOMOUS_TEST_OK`, exit 0).

## 9. Files modified (this iteration)

| File | Change |
| --- | --- |
| `enr-core/src/main/java/ai/enr/core/json/JsonParseException.java` | NEW — `JsonParseException extends IllegalArgumentException` |
| `enr-core/src/main/java/ai/enr/core/json/Json.java` | `error()` returns `JsonParseException` |
| `enr-java-api/src/main/java/ai/enr/client/EnrClient.java` | `unloadModel`/`cancel` check HTTP status |
| `enr-server/src/main/java/ai/enr/server/OllamaApiHandlers.java` | malformed JSON → 400 (show/delete/chat/generate) |
| `enr-server/src/main/java/ai/enr/server/OpenAiApiHandlers.java` | malformed JSON → 400 (chat/completions/embeddings) |
| `enr-server/src/main/java/ai/enr/server/ManagerApiHandlers.java` | malformed JSON → 400 (chat) |
| `enr-tests/src/test/java/ai/enr/tests/OllamaToolCallingTest.java` | +3 regression tests (malformed JSON → 400) |
| `enr-tests/src/test/java/ai/enr/tests/ApiIntegrationTest.java` | +1 regression test (unload not-loaded fails) |
| `enr.yaml` | `model_directories` → current repo path |

> Note: `target/` build artifacts (`.class`/`.jar`) are tracked in this repository's git history
> (pre-existing); rebuilding regenerated them. They were not committed, and no Git commit/push was
> performed, per the campaign restrictions.

---

# Iteration 2 — Harness Gateway (2026-11-23)

Objective: make ENR a reliable LLM gateway connecting providers/models to agentic harnesses
(OpenCode), with the full agentic lifecycle — not just the first inference request.

## Result summary

**P0 ACHIEVED — ENR is READY FOR HARNESS INTEGRATION** for the gateway role. A real OpenCode
session, configured to use ENR as its only LLM backend (model `deepseek:deepseek-flash` routed
through ENR to the DeepSeek provider), completed a full agentic coding task: read a source file,
edit it, run a shell command, inspect the output, continue reasoning, and produce a final answer —
with **all LLM traffic passing through ENR** (7 chat requests logged, conversation growing 1→10
messages, 14 tool definitions, streaming throughout).

## Changes (this iteration)

| Area | Change |
| --- | --- |
| Provider error propagation | `ChatResult` gains a `status` field + `gatewayStatus` mapping: 429→429, 408→504, 401/403→502, 502/503/504→same, other→502 (was blanket 500). Propagated through `EnrManager` (`error_status`) to Ollama/OpenAI/Manager handlers. |
| Request logging | `EnrManager` logs each chat/chatStream request (model, target, provider, message count, tools, stream) — enables lifecycle verification via `/api/v1/logs`. |
| CLI commands | Added `enr ps`, `enr show <model>`, `enr run <model>`, `enr rm <model>`, `enr version`. |
| Ollama endpoints | Added `/api/copy` (duplicate installed model) and `/api/embed` (local→501, OpenAI-compatible provider forwarded). |
| Regression tests | `ProviderErrorPropagationTest` (6 tests), `/api/copy` + `/api/embed` tests (4 tests). |

## Compatibility matrix (verified this iteration)

### Ollama-compatible API (`/api/*`)

| Endpoint | Status | Notes |
| --- | --- | --- |
| `/api/version` | Fully compatible | `{"version":"2.6.11","name":"ENR"}` |
| `/api/tags` | Fully compatible | local models + `capabilities` |
| `/api/ps` | Fully compatible | loaded models + `parameter_size` |
| `/api/show` | Fully compatible | `model`/`name` fallback, `capabilities`, `model_info` |
| `/api/chat` | Fully compatible | non-stream + NDJSON stream, multi-turn, tool calling |
| `/api/generate` | Fully compatible | non-stream + stream |
| `/api/embed` | Partially compatible | local→501; forwarded to OpenAI-compatible providers |
| `/api/pull` | Partially compatible | streaming progress; catalogs (Ollama Library + Hugging Face GGUF) |
| `/api/delete` | Fully compatible | removes installed model |
| `/api/copy` | Fully compatible | duplicates an installed model |
| `/api/create` | **Unsupported** | Modelfile→model creation deferred (out of harness-integration scope) |

### OpenAI-compatible API (`/v1/*`)

| Endpoint | Status | Notes |
| --- | --- | --- |
| `/v1/models` | Fully compatible | local + provider-qualified model ids |
| `/v1/chat/completions` | Fully compatible | non-stream + SSE stream, tool calling, `reasoning_content` |
| `/v1/completions` | Fully compatible | text completion |
| `/v1/embeddings` | Partially compatible | local→501; forwarded to OpenAI-compatible providers |

## OpenCode integration (real, not mocked)

- Config: custom `enr` provider (openai-compatible) → `baseURL http://127.0.0.1:8111/v1`,
  model `ds-flash` → `modelID "deepseek:deepseek-flash"`, `tools: true`.
- Task: edit `calc.py` (add `multiply`), run `python3 -c "from calc import multiply; print(multiply(6,7))"`,
  report actual output.
- Result: `multiply` added, command output `42` reported accurately.
- Evidence: ENR `/api/v1/logs` shows 7 `chat` requests (`target=provider provider=deepseek`,
  `messages=1..10`, `tools=14`, `stream=true`); the file was edited on disk. No direct
  OpenCode→DeepSeek connection existed (the `enr` provider only knows `127.0.0.1:8111`).

## Build and test results

- `mvn clean package` (JDK 25, release 21) — **BUILD SUCCESS**, 13 modules.
- Tests: **170 tests, 0 failures** (`enr-tests` = 120; was 110 at end of iter1).

## Remaining limitations

- **Native EVO models are text-only toys** (no native structured tool calling, tiny context) —
  secondary per campaign; agentic inference is served through providers.
- **`/api/create` not implemented** (Modelfile→model creation deferred).
- **Concurrent-session-limit rejection still returns HTTP 500** (should be 429) — pre-existing finding.
- **Anthropic/Gemini tool translation** not implemented (explicit 400).

## Audit evidence (iter2)

- Audit run: `enr-harness-gateway-20261123-iter2` (8 modules, 1 resolved finding, `DEGRADED`),
  chained to `previousRunId: enr-func-test-20261123-iter1`. All three runs preserved append-only.
- OpenCode test project: `/tmp/opencode/enr-agent-test/`.

---

# Iteration 3 — Complete Ollama-Compatible Model Lifecycle + Hugging Face (2026-11-23)

Objective: make ENR a practical drop-in Ollama alternative for local model acquisition and execution,
with Hugging Face GGUF repositories as the primary source.

## Implemented (this iteration)

- **Hugging Face GGUF discovery**: `hf.co/OWNER/REPO:QUANT` (and `hf:`, `huggingface.co:` prefixes)
  resolves the repository via the HF tree API and selects the single GGUF matching the quantization
  tag; missing/ambiguous quantizations produce clear errors listing alternatives. `lfs.oid` is used
  as the SHA-256 verification digest.
- **HF token auth**: `catalogs.huggingface_token` (or `HF_TOKEN` env) sent as a Bearer header for
  gated repositories — never logged or returned by the API.
- **Resumable downloads**: `DownloadManager` resumes via `Range` from a `.part` file; SHA-256
  verification; explicit cancel cleans up, network interruption keeps the partial for retry.
- **CLI**: `enr copy <src> <dst>`, `enr stop <model>`, and `enr run <model>` with automatic pull
  (skip if installed), explicit load, and honest load-failure reporting.
- **HTTP**: `/api/v1/catalog/ensure` and `/api/v1/catalog/copy` (manager endpoints).
- **Fixed pre-existing bug**: HF repo was parsed as only the first path segment
  (`hf:OWNER/REPO/FILE.gguf` produced a broken `OWNER/resolve/main/REPO/FILE.gguf` URL).

## Real-world acceptance results

| Step | Result |
| --- | --- |
| Model initially absent | PASS |
| `enr pull hf.co/bartowski/SmolLM2-135M-Instruct-GGUF:Q4_K_M` | PASS — 100.6 MB downloaded, SHA-256 verified, registered |
| Persistent registration | PASS — survives restart |
| `enr run` without re-download | PASS — skips download (`installed:true`) |
| Real inference | **BLOCKED** — native runtime supports F32/F16 tensors only; `Q4_K_M` (GGML type 8) rejected with a clear error |
| `enr stop` / `enr ps` / `/api/tags` / `/api/show` | PASS |
| `enr copy` / `enr rm` + storage cleanup | PASS |

The exact acceptance target `hf.co/unsloth/Inkling-2-Mini-GGUF:UD-Q4_K_M` resolves correctly to the
16.6 GB `Inkling-2-Mini-UD-Q4_K_M.gguf` (verified against the live HF API) but was not fully
downloaded (≈1.7 h at the measured 2.7 MB/s); a smaller real GGUF was used to verify the complete
download → register → restart → remove lifecycle.

## Test results

- `enr-tests`: **129 tests, 0 failures** (was 120; +9 `HuggingFaceCatalogTest`).
- Full build compiles cleanly. **CUDA module (4 tests) BLOCKED by environmental GPU OOM** — the
  real Ollama `llama-server` occupies 14.4 GB of the 16 GB GPU (`cuCtxCreate failed: 2`), unrelated
  to these changes.

## Remaining limitations

- **Quantized GGUF (Q4_K_M etc.) cannot be executed natively** — the native runtime supports F32/F16
  tensors only (P3-secondary; download ≠ inference). Use F16/F32 exports or a cloud provider.
- **`/api/create` (Modelfile→model)** remains unimplemented.
- CUDA tests currently BLOCKED by GPU contention from the real Ollama process.

## Audit evidence (iter3)

- Audit run: `enr-hf-lifecycle-20261123-iter3`, chained to `previousRunId: enr-harness-gateway-20261123-iter2`.
- Real download evidence: `SmolLM2-135M-Instruct-GGUF:Q4_K_M` (100.6 MB, verified, then removed).


