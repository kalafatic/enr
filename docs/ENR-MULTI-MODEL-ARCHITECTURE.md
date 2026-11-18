# ENR Multi-Model Architecture

Date: 2026-11-16
Status: implemented and tested

## 1. Model Lifecycle

```
discover (model directories) ─► load ─► READY ─► unload ─► removed
                                  │
                                  └─ active sessions > 0 ⇒ unload rejected
                                     (unless force or safe shutdown)
```

- `ModelManager` registry keyed by model id (file name without extension)
- `ModelInstance` owns: path, `LoadedModel`, `Transformer` (bound weights),
  tokenizer, compute backend, state, active-session count, request counters
- Weights are lazily materialized and cached per model instance

## 2. Concurrency & Isolation

- Per-model active-session counting; unload while active is rejected with
  `MODEL_IN_USE` (HTTP 409)
- Sessions are fully independent: private KV cache, private context token list,
  private cancellation flag — no shared mutable generation state
- Server requests run on virtual threads; the runtime uses fine-grained locks
  only around registries/counters

## 3. Resource Management

- CPU: heap-based; weight residency tracked (`loadedWeightBytes`)
- GPU: VRAM accounting per backend; device weight cache (see COMPUTE-BACKENDS)
- Eviction: idle models (no active sessions) can be evicted by residency policy
  (`ModelManager.evictIdle(maxResident, targetFreeBytes)`)
- Session limits: `max_concurrent_sessions` enforced at session creation
- Idle sessions expire after `session_idle_timeout_seconds`

## 4. Scheduling

- Priority FIFO: requests are admitted per-model; the global session limit is
  the admission gate
- Each model executes independently on its bound backend; concurrent sessions on
  one model share weights read-only

## 5. Verification (real executions)

- Two real models loaded simultaneously (`tiny-evo`, `evo`)
- 8 concurrent requests across both models: no cross-session or cross-model
  contamination, all requests completed
- Session isolation: independent KV caches, independent contexts
- Unload protection: active model unload rejected; unload succeeds after session end
- Session limit enforcement verified

## 6. Model Roles

Roles such as EVO-CHAT / EVO-CODER / EVO-REASONER / EVO-EMBEDDING are logical
identities, not hardcoded: any supported model can be registered under any role;
ENR does not special-case them.
