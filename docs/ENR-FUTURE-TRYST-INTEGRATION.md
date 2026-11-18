# ENR Future TRYST Integration (Specification Only)

Date: 2026-11-16
Status: NOT implemented — documented for a later work package. No TRYST changes made.

## 1. Objective

TRYST may consume ENR as its native model inference backend without any
modification of TRYST in this development cycle.

## 2. Recommended Consumption Modes

1. **HTTP mode (preferred for isolation)**: TRYST spawns or connects to an ENR
   server and calls the versioned API (`/api/v1/inference`, `/inference/stream`,
   `/inference/cancel`). Streaming via SSE maps directly onto TRYST's streaming
   UI/agent contracts.
2. **Java API mode**: TRYST (Java 21, RCP/OSGi) embeds `enr-java-api` as a bundle;
   the client is plain Java with no OSGi dependencies, so it can be wrapped in a
   thin OSGi service if required (adapter only — ENR core stays OSGi-free).

## 3. Contract Notes

- Sessions: TRYST conversations map to ENR sessions (`session_id`) for
  multi-turn context, or use stateless per-request inference
- Cancellation: propagate user cancellation to
  `POST /api/v1/inference/cancel`
- Coding tasks: use the DCode-oriented contract described in
  ENR-FUTURE-DCODE-INTEGRATION.md (model selection, streaming, cancellation,
  context handling)

## 4. Explicitly Out of Scope Here

- Modifying TRYST to call ENR
- Any ENR dependency on TRYST classes or services
