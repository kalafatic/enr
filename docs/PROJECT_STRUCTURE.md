# ENR — Project Structure

Authoritative parent/aggregator: **`pom.xml` (repository root)** — `ai.enr:enr-parent` 2.6.11-SNAPSHOT (plain Maven, no OSGi/Tycho)

| Layer | Location | Notes |
|---|---|---|
| Repository root | `/home/petr/git/evo-native-runtime-161126` | Git root; parent POM at root |
| Build system | Plain Maven (compiler release 21) | `enr-compute-cuda` requires release **25** → build the reactor with JDK 25 |
| Feature/product | none (standalone native runtime) | — |
| Runtime/launch | `enr-cli` self-contained executable JAR (`ai.enr.cli.EnrCli`) | `enr start` runs the HTTP server |
| Config | `enr.yaml` | runtime configuration |
| Legacy launcher | `enr.sh` (dev wrapper; depends on a cached classpath file) | superseded by the shaded CLI jar |

## Reactor modules (12)

| Module | Role |
|---|---|
| enr-core | Core tensor/JSON primitives |
| enr-model-loader | Model acquisition/loading |
| enr-tokenizer | BPE tokenizer |
| enr-compute | Compute backend SPI |
| enr-compute-cpu | CPU backend |
| enr-compute-cuda | CUDA backend (FFM API; Java 25) |
| enr-inference | Inference engine |
| enr-runtime | Runtime orchestration |
| enr-server | HTTP server (embedded in enr-cli start) |
| enr-java-api | Public API facade |
| enr-cli | CLI entry point (`ai.enr.cli.EnrCli`), shaded executable JAR |
| enr-tests | Integration tests |

CLI commands: `enr start [-c config] [-p port]`, `stop`, `status`, `models`, `load`, `unload`, `infer`, `devices`.
