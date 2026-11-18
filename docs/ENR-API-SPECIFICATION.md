# ENR API Specification

Date: 2026-11-16
Status: implemented and tested end-to-end

Base path: `/api/v1` — JSON request/response bodies, UTF-8.

## Endpoints

### GET /api/v1/health
`200 {"status":"healthy","checks":[]}` — `503 {"status":"unhealthy","checks":[...]}` when
the runtime is stopped, a model is not READY, or no CPU backend exists.

### GET /api/v1/status
`200` runtime status: `running`, `version`, `uptime_ms`, `models_loaded`,
`sessions_active`, `total_requests`, `total_generated_tokens`, `tokens_per_second`.

### GET /api/v1/models
`200 {"models":[ModelInfo...]}`. ModelInfo: `id`, `path`, `state` (loading|ready|
unloading|error), `architecture`, `vocab_size`, `d_model`, `num_heads`, `num_kv_heads`,
`num_blocks`, `dff`, `max_seq_len`, `positional_encoding`, `parameters`,
`weight_bytes`, `backend`, `active_sessions`, `total_requests`, `loaded_at`.

### POST /api/v1/models/load
Body: `{"path": "...", "backend": "cpu"|"cuda"}` (backend optional).
`200 {"model": ModelInfo}`. Errors: `400 INVALID_REQUEST`, `422 LOAD_FAILED`.

### POST /api/v1/models/unload
Body: `{"id": "...", "force": false}`.
`200 {"unloaded": "id"}`. Errors: `404 MODEL_NOT_FOUND`, `409 MODEL_IN_USE`.

### GET /api/v1/models/{id}
`200 {"model": ModelInfo}` / `404 MODEL_NOT_FOUND`.

### POST /api/v1/inference
Body: `{"model": "...", "prompt": "...", "max_tokens": N, "temperature": F,
"top_p": F, "top_k": N, "repeat_penalty": F, "seed": L, "session_id": "...",
"stop_token_ids": [...]}` (all optional except model+prompt).
`200` result: `request_id`, `model`, `text`, `tokens`, `prompt_tokens`,
`generated_tokens`, `reason` (EOS_REACHED|MAX_TOKENS_REACHED|MAX_SEQ_LEN_REACHED|
CANCELLED|ERROR), `elapsed_ms`.
Errors: `400 INVALID_REQUEST`, `500 INFERENCE_FAILED`.

### POST /api/v1/inference/stream
Same body as /inference. Response `text/event-stream`:
`event: start` (request_id, model), repeated `event: token` (`token_id`, `text`),
`event: done` (`reason`, `prompt_tokens`, `generated_tokens`, `elapsed_ms`,
`full_text`).

### POST /api/v1/inference/cancel
Body: `{"request_id": "...", "session_id": "..."}` (either). `200 {"cancelled": true}`.

### GET /api/v1/sessions
`200 {"sessions":[{"id","model","context_length","created_at","last_active_at"}...]}`.

### GET /api/v1/metrics
`200` counters: `uptime_ms`, `total_requests`, `total_prompt_tokens`,
`total_generated_tokens`, `total_inference_time_ms`, `failed_requests`,
`cancelled_requests`, `tokens_per_second`.

### GET /api/v1/devices
`200 {"devices":[{"index","type","name","total_memory_bytes","free_memory_bytes",
"driver_version","available"}...]}` — genuine device reporting.

## Conventions

- Unknown routes / methods: `404` / `405 METHOD_NOT_ALLOWED`
- Errors: `{"error":{"code": "...", "message": "..."}}`
- Sessions: created implicitly per request when `session_id` is absent (scoped to
  that request); explicit sessions persist context until deleted or idle-timeout
- Versioning: `/api/v1` prefix; breaking changes will introduce `/api/v2`
