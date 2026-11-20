# ENR — API Compatibility Report

Date: 2026-11-19
Status: Implemented and integration-tested (`enr-tests` module: `OllamaCompatibilityTest`,
`OpenAiCompatibilityTest`, `ManagerTest`, plus pre-existing `ApiIntegrationTest`/`FamilyApiIntegrationTest`).

## 1. Ollama-compatible endpoints

| Endpoint | Method | Status | Notes |
|---|---|---|---|
| `/api/tags` | GET | Implemented, tested | Installed models with `name`, `model`, `modified_at`, `size`, `digest` (`sha256:`), `details` |
| `/api/show` | POST | Implemented, tested | `modelfile`, `parameters`, `template`, `details` (format/family/parameter_size); 404 for unknown |
| `/api/pull` | POST | Implemented, tested | Body `{name, stream}`. NDJSON progress: `pulling manifest`, `downloading <digest>` (total/completed), `verifying sha256 digest`, `success`; errors as `{"error": ...}`. Blocking mode returns single `{"status":"success"}` |
| `/api/delete` | DELETE | Implemented, tested | 200 (empty) or 404 `{"error":"Model not installed: ..."}`; unloads first when loaded |
| `/api/chat` | POST | Implemented, tested | `{model, messages, stream, options}`. Sync: `{model, created_at, message, done, done_reason, total_duration, prompt_eval_count, eval_count}`. Stream: NDJSON lines with final `done:true` + counts |
| `/api/generate` | POST | Implemented, tested | `{model, prompt, stream, options}` → `response` field; same NDJSON streaming |
| `/api/ps` | GET | Implemented, tested | Running (loaded) models with `name`, `size`, `details.parameter_size`, `expires_at`, `size_vram` |
| `/api/version` | GET | Implemented, tested | `{"version":"2.6.11","name":"ENR"}` |

Routing notes: `model` may be an installed local id or `provider:model` (e.g. `deepseek:deepseek-chat`).
Local models are lazy-loaded on first request and the runtime state is verified. Options supported:
`temperature`, `top_p`, `top_k`, `num_predict`, `seed`, `stop`.

Unsupported (explicitly): image inputs are accepted but ignored (content replaced with `[image input]`);
`keep_alive`/session management of Ollama are not implemented; `template`/`system` Ollama fields are
folded into the role-marked prompt.

## 2. OpenAI-compatible endpoints

| Endpoint | Method | Status | Notes |
|---|---|---|---|
| `/v1/models` | GET | Implemented, tested | `{object:"list", data:[{id, object:"model", created, owned_by:"enr"}]}`; ids are local models plus `provider:model` entries |
| `/v1/chat/completions` | POST | Implemented, tested | Sync: `chat.completion` with `message`, `finish_reason`, `usage` (token counts). Streaming (`stream:true`): SSE `data:` chunks with `delta.content`, final chunk with `finish_reason` + `usage`, terminated by `data: [DONE]` |
| `/v1/completions` | POST | Implemented, tested | `text_completion` with `choices[].text`, `finish_reason`, `usage` |
| `/v1/embeddings` | POST | Partially | Pass-through for OpenAI-compatible/DeepSeek/OpenAI providers (tested); **501 for native local models** — the ENR transformer exposes logits only; Anthropic/Gemini/Ollama/llama.cpp types return 501 with the reason |

Supported parameters: `model`, `messages` (string content; array content parts collapsed to text),
`stream`, `stream_options.include_usage`, `max_tokens`/`max_completion_tokens`, `temperature`,
`top_p`, `top_k`, `seed`, `stop`.

## 3. Native + family surfaces (unchanged, still tested)

- `/api/v1/health|status|models|models/load|models/unload|models/{id}|inference|inference/stream|inference/cancel|sessions|metrics|devices`
- `/enr/v1/*` family surface (DCode/TRYST/DStudio contract, TEIP error envelope)
- `enr-java-api` client — byte-compatible

## 4. New management REST surface (single backend for UI/CLI/API)

| Endpoint | Purpose |
|---|---|
| `GET/POST /api/v1/providers`, `PUT/DELETE /api/v1/providers/{id}`, `POST /{id}/test`, `GET /{id}/models`, `GET /{id}/health` | Provider registry (admin-auth protected) |
| `GET /api/v1/catalog/models`, `GET/DELETE /api/v1/catalog/models/{id}` | Installed models + runtime state |
| `GET /api/v1/catalog/search?q=&limit=` | Ollama + Hugging Face discovery |
| `POST /api/v1/catalog/pull`, `GET /api/v1/downloads`, `POST /api/v1/downloads/cancel` | Downloads with progress/cancel |
| `POST /api/v1/catalog/import`, `POST /api/v1/catalog/verify` | GGUF import and digest verification |
| `POST /api/v1/models/start`, `POST /api/v1/models/stop` | Model lifecycle by id |
| `POST /api/v1/chat` | Unified streaming chat (SSE `chunk` events; used by the Web UI Playground) |
| `GET /api/v1/dashboard`, `GET /api/v1/diagnostics`, `GET /api/v1/logs?source=&tail=`, `GET /api/v1/hardware` | Observability |
| `GET /api/wcc/info` | Web UI auth info |

Admin endpoints (providers, catalog mutations, downloads cancel, models start/stop, logs,
diagnostics) require `Authorization: Bearer <server.auth_token>` when a token is configured;
inference endpoints (Ollama/OpenAI/native chat) remain open. Health stays public for monitoring.

## 5. Error envelope

- Native/family APIs keep their existing envelopes (unchanged).
- Ollama surface: `{"error": ...}` objects; `/api/delete` uses `{"error":"model 'x' not found"}`; status codes 400/404/405/500 as appropriate.
- OpenAI surface: JSON error body with `error.message` for non-200 responses; 501 for unsupported embeddings with an explicit message.

## 6. Test evidence

- `OllamaCompatibilityTest` (10 tests): version, tags, ps, show, chat sync+stream, generate, pull via mock registry (progress phases asserted), delete, delete-404.
- `OpenAiCompatibilityTest` (8 tests): models, chat sync, chat SSE (`[DONE]` asserted), completions, embeddings 501 (native), cloud chat routing, wrong-key rejection, embeddings passthrough.
- `ManagerTest` (13 tests): store lifecycle, start/stop state verification, key redaction, connection test, router determinism (local wins, explicit prefix, loud ambiguity/unknown), download cancel, admin auth 401/200, systemd unit content, system config generation, unified chat stream, hardware genuineness, catalog search against mock registry.
- Pre-existing suites remain green (API, family API, loader, tokenizer, CPU compute, inference oracle, runtime).

## 7. Documented limitations (recap)

- Native inference: F32/F16 tensors only; quantized GGUF (Q4_K_M etc.) downloads fine but cannot run natively (explicit error). Real-world metadata gap fixed: `llama.vocab_size` now falls back to the token-array length (Ollama library exports).
- Native embeddings: 501 by design.
- Anthropic/Gemini: chat + streaming only.
