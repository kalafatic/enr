# ENR — Build Guide

Authoritative parent: `/home/petr/git/evo-native-runtime-161126` (root `pom.xml`).

Prerequisites: **JDK 25** for the full reactor (enr-compute-cuda uses finalized FFM APIs, release 25); Maven 3.9.

| Operation | Command (from repo root) | Output | Exit |
|---|---|---|---|
| Full Build | `JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64 mvn -B -DskipTests clean verify` | 12 modules; `enr-cli/target/enr-cli-2.6.11-SNAPSHOT.jar` (shaded, self-contained) | 0 |
| Clean Build | same | — | 0 |
| Build Selected Module | `mvn -B -DskipTests clean verify -pl <module> -am` | module + deps | 0 |
| Test | `mvn -B test` | 18 integration tests (CPU) + CUDA tests (GPU-dependent) | 0 |
| Validate | `mvn -B validate` | fast | 0 |
| Package | `mvn -B -DskipTests package` | all module jars | 0 |

JDK rules: parent sets Java 21 for most modules; `enr-compute-cuda` overrides release 25 and cannot compile on JDK 21. Use JDK 25 for reactor builds; CPU-only module builds work with JDK 21.

## Verified results (2026-11-19)
- Full reactor build with JDK 25: **SUCCESS**.
- `enr-tests`: 18/18 green (FamilyApi, CudaInference, Api integration).
- `CudaBackendTest`: 4/9 green; 5 errors `cuCtxCreate failed: 2` — no NVIDIA GPU/driver on this host (environmental).
- UPM: `validate` SUCCESS, full `build` SUCCESS (JDK 25 auto-selected via `requiresJava=25` detection).

## ENR CLI packaging fix
`enr-cli/pom.xml` now applies `maven-shade-plugin` (main class `ai.enr.cli.EnrCli`) — previously the CLI JAR was thin (3 classes) and could not run standalone (`NoClassDefFoundError`). The shaded JAR is the authoritative launch artifact.
