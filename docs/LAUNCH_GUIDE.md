# ENR — Launch Guide

ENR is a standalone native runtime — no Eclipse RCP product. The authoritative launch target is the shaded CLI JAR.

| Launch target | Command | Notes |
|---|---|---|
| **CLI** | `java -jar enr-cli/target/enr-cli-2.6.11-SNAPSHOT.jar <command>` | `--help` shows usage |
| **Server (Development)** | `java -jar enr-cli/target/enr-cli-2.6.11-SNAPSHOT.jar start -p 18081` | HTTP server on 127.0.0.1:18081 (`{"name":"EVO Native Runtime (ENR)",...}` at `/`) |
| **Debug** | run the same JAR under a debugger (JDK 25) | |
| Stop | `java -jar …/enr-cli….jar stop` or kill the process | |
| Status | `java -jar …/enr-cli….jar status --url http://127.0.0.1:18081` | |
| Legacy | `./enr.sh <command>` | dev wrapper, prefers JDK 25; depends on a pre-generated classpath file |

Server API: `GET /api/v1/…`, family API `GET /enr/v1/…`.

## Verified (2026-11-19)
- Direct launch: `enr start -p 18081` → server responded with JSON at `/`.
- UPM launch integration: `jar` launch config `enr-server` with `{java25}` env and HTTP readiness → state reached **READY** (pid, port 18081), UPM `stop` → **STOPPED**, port released.
- CUDA backend requires an NVIDIA GPU (not available on this host).
