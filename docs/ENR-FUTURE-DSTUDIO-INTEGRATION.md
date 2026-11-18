# ENR Future DStudio Integration (Specification Only)

Date: 2026-11-16
Status: NOT implemented — documented for a later work package. No DStudio changes made.

## 1. Objective

DStudio may manage ENR as an external runtime service (start/stop, model
placement, health and resource monitoring) without changes to DStudio in this
cycle.

## 2. Management Contract

- Process lifecycle: `enr start|stop` (pid file at `/tmp/enr.pid`), plus a
  configurable `enr.yaml` for binding, model directories, backend and limits
- Health: `GET /api/v1/health` (200/503 with check list)
- Status: `GET /api/v1/status`; devices: `GET /api/v1/devices`
- Model placement: `POST /api/v1/models/load` (`backend` cpu|cuda),
  `POST /api/v1/models/unload`
- Metrics: `GET /api/v1/metrics` (genuine counters only)

## 3. Resource Policies DStudio May Set

- `preferred_backend`, `max_concurrent_sessions`, `default_max_tokens`
- `session_idle_timeout_seconds`
- Model directories for auto-load

## 4. Explicitly Out of Scope Here

- Implementing DStudio management of ENR
- ENR embedding into DStudio
