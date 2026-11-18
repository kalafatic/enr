# ENR Future DCode Integration (Specification Only)

Date: 2026-11-16
Status: NOT implemented — documented for a later work package. No DCode changes made.

## 1. Objective

Prepare a coding-oriented inference contract for DCode without modifying DCode.

## 2. Coding Inference Contract (ENR side, ready now)

- **Coding model selection**: models are logical ids; load a coding model and
  target it by id (`/api/v1/inference` body `model`)
- **Streaming**: `POST /api/v1/inference/stream` (SSE `start`/`token`/`done`)
- **Cancellation**: `POST /api/v1/inference/cancel` with the stream `request_id`
- **Context handling**: persistent ENR sessions (`session_id`) hold the KV cache
  and token history between tool calls — multi-turn coding sessions map 1:1
- **Tool-oriented inference**: ENR supports stop-token sets
  (`stop_token_ids`) suitable for tool-call delimiters; structured output
  parsing remains the caller's concern

## 3. Recommended Future DCode Flow

1. DCode selects/loads a coding model through ENR (HTTP or Java client)
2. Each coding session opens an ENR session; prompts accumulate context in the
   ENR KV cache (bounded by the model's max sequence length)
3. Tool-call boundaries signaled via `stop_token_ids` + streaming
4. Cancellation/timeout propagates to ENR (session-scoped cancel)

## 4. Explicitly Out of Scope Here

- DCode modifications
- Tool/agent loop implementation inside ENR (ENR stays an inference runtime)
