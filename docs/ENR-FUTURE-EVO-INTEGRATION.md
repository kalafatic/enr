# ENR Future EVO Integration (Specification Only)

Date: 2026-11-16
Status: NOT implemented — documented for a later work package. No EVO changes made.

## 1. Objective

Make ENR an inference provider inside the EVO platform while retaining the
existing architecture layers:

- Darwin Engine + existing EVO session management remain the application-level
  orchestration layer
- ENR provides native `.evo`/GGUF inference (the current EVO serving path spawns
  llama.cpp/Ollama subprocesses via `LlamaService`/`OllamaService`)
- Ollama remains an optional external provider alongside ENR

## 2. Recommended Integration Points (verified during audit)

1. **Provider abstraction**: EVO's controller already distinguishes service
   backends (`OllamaService`, `LlamaService`); ENR should be introduced as an
   additional service implementation speaking the same message contract
   (`Message`-based generate), backed by the ENR Java client (`enr-java-api`)
   or the ENR HTTP API — no changes to the Darwin orchestration layer.
2. **Model distribution**: Darwin Forge exports GGUF today; ENR loads those
   files directly. Consider also producing `.evo` v2 from the export pipeline
   (see ENR-MODEL-FORMAT-COMPATIBILITY.md recommendations).
3. **Darwin candidate evaluation**: `ReferenceEvoInferenceEngine` evaluates
   candidate `EvoLlmModel` instances in-process. ENR could serve as a batch
   evaluation backend for exported snapshots; the in-process engine remains the
   canonical evaluator.
4. **Resource placement**: EVO should treat ENR like the existing external
   providers (lifecycle start/stop via the supervisor) rather than embedding it.

## 3. What Must NOT Happen

- No second orchestration architecture inside EVO
- No removal of the Darwin Engine session model
- No EVO code changes during the ENR development cycle

## 4. Interface Contract for the Future Work Package

- Health: `GET /api/v1/health`
- Model lifecycle: `POST /api/v1/models/load|unload`, `GET /api/v1/models`
- Inference: `POST /api/v1/inference`, `POST /api/v1/inference/stream`,
  `POST /api/v1/inference/cancel`
- Client: `ai.enr.client.EnrClient` (Java 21, no RCP/OSGi dependency)
