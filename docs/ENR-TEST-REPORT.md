# ENR Test Report

Date: 2026-11-16
Status: `mvn clean install` — 61 tests, 0 failures, 0 errors, 0 skipped
(Build JDK 25 for the CUDA module; all other modules release-21.)

## 1. Test Inventory

| Module | Tests | Coverage |
|---|---|---|
| enr-model-loader | 8 | `.evo` v2 round-trip, checksum violation, bad magic, missing file; REAL GGUF models (`evo.gguf`, `tiny-evo.gguf`): metadata, tensor layout, stats, validator |
| enr-tokenizer | 8 | longest-match encoding, byte fallback, UTF-8 round-trip, special tokens, incremental decoder streaming |
| enr-compute-cpu | 12 | matmul values/dims/rejection, RMSNorm reference + weight scaling, softmax invariance/stability, causal attention + KV-offset masking, SiLU/SwiGLU, sinusoidal PE reference, transpose, add/mul |
| enr-compute-cuda | 9 | REAL GPU (RTX 5070 Ti): device discovery, matmul (small+512×512), RMSNorm, softmax, SiLU-mul, add/mul, attention vs CPU, VRAM release accounting |
| enr-inference | 5 | **ENR forward vs llama.cpp oracle on real models** (2e-3 = F16 KV cache noise; argmax/top-5 identical on evo.gguf), KV-cache incremental == full pass, generation lifecycle |
| enr-runtime | 6 | two models + 8 concurrent requests (no contamination), session isolation, unload-while-active rejection, unknown model rejection, session limit, status/health/devices |
| enr-tests | 13 | full HTTP API end-to-end (health, status, models, load/unload, inference, streaming SSE, cancellation, invalid requests, concurrency, sessions, metrics, devices); CUDA full-model inference vs CPU vs llama.cpp oracle |

Total: 61.

## 2. Verification Against Real Models

- Real Darwin Forge exports loaded and validated: `evo.gguf` (54k params, d_model=64,
  vocab=100) and `tiny-evo.gguf` (920 params, d_model=8, vocab=16)
- ENR forward logits match the llama.cpp oracle within 2e-3 (llama's F16 KV-cache
  quantization) on both models; argmax/top-5 ordering identical
- ENR CUDA forward matches ENR CPU forward within 2e-3
- `.evo` v2 writer/reader verified by round-trip; corrupted files rejected with
  specific diagnostics (magic, checksum, shape, NaN)

## 3. Numerical Correctness Summary

| Aspect | Method | Tolerance |
|---|---|---|
| ops vs hand-computed references | unit tests | 1e-5..1e-7 |
| CUDA kernels vs CPU | unit tests | 1e-3 (matmul), 1e-4 (softmax/rms) |
| full model vs llama.cpp | oracle tests | 2e-3 |
| KV-cache incremental vs full | equivalence test | 1e-5 |

## 4. Known Test Limitations (disclosed)

1. No real `.evo` v2 file exists in the EVO repository (all placeholders);
   `.evo` compatibility is verified via spec-level round-trip and cross-checked
   tensor math, not against a real Darwin Forge-produced `.evo` file.
2. `evo.gguf` declares context_length=4; generation tests on it are bounded by
   that real model property.
3. CUDA tests are skipped automatically when no GPU is present; they were
   executed and passed on the RTX 5070 Ti in this environment.
4. llama.cpp comparison uses a CPU build as an external oracle (never a runtime
   dependency).
