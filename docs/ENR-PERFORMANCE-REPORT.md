# ENR Performance Report

Date: 2026-11-16
Status: all numbers measured from actual executions (`PerformanceBenchmark`,
`enr-tests` module, JVM JDK 25, NVIDIA RTX 5070 Ti 16 GB, driver 595.99.02)

## 1. Model Loading

| Model | Parameters | Weight bytes | Load time |
|---|---|---|---|
| tiny-evo.gguf | 920 | 3 680 B | 19–22 ms |
| evo.gguf | 53 952 | 215 808 B | 3 ms |

(GGUF: header+metadata parse; tensor payloads memory-mapped and converted lazily
on first use. `.evo` v2: GZIP tensors decompressed on first use.)

## 2. Time To First Token (prefill)

| Model | Prefill tokens | CPU | CUDA |
|---|---|---|---|
| tiny-evo.gguf | 5 | 1–2 ms | 148–156 ms |
| evo.gguf | 3 | ≤1 ms | 113–119 ms |

## 3. Decode Speed (tokens per second)

| Model | CPU | CUDA |
|---|---|---|
| tiny-evo.gguf (20-token decode) | 10 000–20 000 tok/s | 476–500 tok/s |

## 4. Concurrent Request Throughput

64 requests, 8 concurrent workers, tiny-evo model, CPU backend:

- wall time 76 ms
- **842 requests/s**
- average request latency 9.2 ms

## 5. Memory

| Metric | Value |
|---|---|
| CPU heap (JVM max) | 32 GB reported |
| GPU total VRAM (RTX 5070 Ti) | 16 303 MiB |
| GPU free at measurement | ~15.3 GB |
| VRAM after ops (no residency) | 0 bytes (verified release) |
| VRAM with weight cache (tiny-evo) | 10 128 bytes resident |

## 6. CPU vs CUDA Analysis (honest)

For the currently available models (≤54k parameters), the CPU backend is faster:
the CUDA path is dominated by per-operation host-device transfers and kernel
launch overhead (~5 µs per launch × ~30 kernels/step). The CUDA backend is
numerically verified (equivalent results within tolerance) and its weight cache
already improves decode 220 → 500 tok/s. Expected crossover for realistically
sized models (≥ ~10M parameters) comes from matmul-bound work; the defined
follow-up optimization is full weight residency + persistent KV buffers on
device, eliminating per-op transfers.

## 7. Measurement Conditions

- JVM: OpenJDK 25.0 (launcher prefers JDK 25; CPU-only path also runs on 21)
- No other GPU load during measurements (nvidia-smi: 271 MiB used by desktop)
- Benchmark source: `enr-tests/src/test/java/ai/enr/tests/PerformanceBenchmark.java`
- All figures re-measurable with:
  `mvn -pl enr-tests test -Dtest=PerformanceBenchmark -Dsurefire.failIfNoSpecifiedTests=false`
