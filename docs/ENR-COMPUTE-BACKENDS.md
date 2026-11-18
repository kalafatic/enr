# ENR Compute Backends

Date: 2026-11-16
Status: CPU verified; CUDA verified on NVIDIA RTX 5070 Ti

## 1. Backend SPI

`ai.enr.compute.ComputeBackend` (enr-compute) defines the execution contract:

- `matmul(a, b)` — `[m,k] × [k,n]`
- `rmsNorm(x, weight, eps)`
- `softmaxRows(x)`
- `silu(x)`, `siluMul(gate, up)`, `add(a, b)`, `mul(a, b)`
- `attention(q, k, v, scale, startPos)` — causal multi-head attention
  (q `[heads, seqLen, headDim]`, k/v `[heads, kvSeqLen, headDim]`), RoPE/sinusoidal
  positioning applied by the caller
- `deviceInfo()` — genuine device/backend reporting
- backends register via `META-INF/services`

Backend selection: `runtime.preferred_backend` (cpu|cu da) with explicit fallback to
CPU and a recorded reason. A backend is never reported as executing when the
computation ran on CPU.

## 2. CPU Backend (`enr-compute-cpu`)

- Reference implementation on shared `enr-core` tensor math
- Deterministic sequential accumulation (ikj-order matmul), stable numerics
- No dependencies; runs on any Java 21+ JVM

## 3. CUDA Backend (`enr-compute-cuda`)

Verified against real hardware:

```
DeviceInfo[CUDA:0 'NVIDIA GeForce RTX 5070 Ti' available=true memory=15841MB free=15343MB]
```

### Design

- Java FFM API (final in JDK 25) bindings to the CUDA **driver API** (`libcuda.so.1`)
  — no CUDA toolkit, no NVRTC, no native build at runtime
- PTX kernels (compute_90, driver-JIT compiled for sm_120) embedded as a resource
- Kernels: tiled matmul (16×16 shared-memory tiles), matmulT (for Q·Kᵀ),
  block-reduced RMSNorm, head-aware masked softmax (shared-memory reduction),
  SiLU-mul, add, mul, in-place scale
- Attention = matmulT → in-place scale → head-aware masked softmax → matmul
  (all on device, single sync per kernel launch)

### Memory management

- VRAM accounting: every allocation/free tracked; `cuMemGetInfo` for device totals
- Device-side weight cache: stable weight arrays are detected by reuse and kept
  resident (capped at 1024 tensors / 1 GiB), avoiding per-op H2D transfers;
  activations are transferred per operation
- All device memory released on close; tests assert zero residual allocation

### Verification (all real executions)

- All kernel ops vs CPU reference within tolerance (matmul 1e-3, softmax 1e-4)
- Full model forward (real `tiny-evo.gguf`): CUDA vs CPU max logits diff ≤ 2e-3
- CUDA logits vs llama.cpp oracle within F16-KV-cache noise (3e-3)
- CPU-only mode fully green when CUDA absent (backend reports unavailable)

### Constraints

- The CUDA module targets JDK 25 bytecode (FFM final API); the rest of ENR is
  Java 21 bytecode. CUDA therefore requires running ENR on JDK 25; CPU-only
  operation works on JDK 21.
- The current per-op transfer model favors correctness; for large models the
  planned improvement is full weight residency + persistent KV cache buffers
  (defined, not yet implemented).

## 4. Fallback Behaviour

`ModelManager.selectBackend("cuda")`:

1. tries the CUDA backend (device discovery via `cuInit`/`cuDeviceGet`)
2. if unavailable and `allow_cuda_fallback=true` → CPU with a logged warning
3. if unavailable and fallback disabled → explicit error

The selected backend is exposed per model (`GET /api/v1/models` → `backend`) and
globally (`GET /api/v1/devices`).
