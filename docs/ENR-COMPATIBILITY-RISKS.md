# ENR Compatibility Risks

Date: 2026-11-16
Status: PHASE 0 deliverable

## 1. Risk Register

| # | Risk | Severity | Evidence | Mitigation |
|---|---|---|---|---|
| R1 | No real `.evo` file exists to validate the `.evo` loader | HIGH | All `.evo` files are 17-byte placeholders | Generate `.evo` v2 fixtures from real GGUF weights; round-trip tests; byte-level golden tests against the audited writer layout; mark real-file compatibility as "format-spec verified, real-file pending" |
| R2 | Descriptor claims RoPE but implementation uses sinusoidal absolute PE | HIGH | `EvoArchitectureDescriptor` default `"rope"` vs `PositionalEncoding.forward` (additive sine/cosine) | ENR replicates implementation semantics; explicit config flag; numerical tests; documented recommendation to Darwin Forge |
| R3 | GGUF exports may contain placeholder weights (ffn_gate/norms = 1.0) | MEDIUM | `OllamaExporter2` fills gate/norm tensors with 1.0 | ENR test detects and reports; inference still runs; document finding |
| R4 | llama.cpp (RoPE) vs native EVO math (sinusoidal PE) divergence on identical GGUF | MEDIUM | llama architecture metadata includes rope keys; native math has no RoPE | Numerical comparison test; ENR always uses native semantics; document that llama.cpp oracle comparison is expected to differ unless exporter/training changes |
| R5 | `EvoModelWriter` writes `sectionCount=5` but serializes 6 sections | LOW | writer source line `dos.writeInt(5)` + 6 `writeString32` calls | ENR reader ignores the count field |
| R6 | Stored content hash truncated to 32 bytes (hex is 64 chars) | LOW | writer zero-pads 32-byte buffer | ENR recomputes hashes; stored value treated as informational |
| R7 | Non-F32 dtypes defined but never produced; INT4/INT8 quantizer produces float arrays of rounded values, not packed integers | MEDIUM | `EvoModelQuantizer` stores quantized values as floats | ENR loader supports F32 payloads; explicit rejection + diagnostics for other dtypes until a producer is verified |
| R8 | Endianness mix: `.evo` big-endian, GGUF little-endian | MEDIUM | writer/reader sources | Per-format explicit byte-order handling + cross-endianness unit tests |
| R9 | Float accumulation order differences (ikj matmul, parallel paths) → last-ulp divergence from EVO reference | LOW | `SimpleTensor.matmul` parallel branches | ENR CPU backend uses deterministic sequential accumulation by default; parallel modes opt-in; tolerance-based comparisons |
| R10 | JDK default on this machine is Java 8; EVO builds use old toolchains | MEDIUM | environment check | ENR pins JDK 21 via Maven toolchain/`maven.compiler.release=21` and launcher scripts |
| R11 | No CUDA toolkit (no nvcc/NVRTC) on target machine | MEDIUM | `libcuda.so.1` present; no toolkit | CUDA backend via driver API + PTX embedded as resources (PTX compiled offline/test-time); explicit CPU fallback; device capability reported |
| R12 | `SimpleBPETokenizer` greedy longest-match differs from real BPE merge ordering | LOW | tokenizer source uses prefix-first longest match | ENR replicates the verified algorithm exactly (contract: byte-identical tokenization of vocab strings) |
| R13 | Vocabulary stored as JSON object loses ordering guarantees | LOW | v2 writer iterates `Map.Entry` | ENR preserves order from file parsing (LinkedHashMap equivalent); tokenization is order-insensitive by algorithm design (longest match) |
| R14 | Darwin Forge may evolve the format mid-development | MEDIUM | parallel development of EVO | Version-gated loading, compatibility layer, migration notes; ENR never blocks on EVO changes |
| R15 | Concurrent parallel development conflicts with external repositories | — | task rules | ENR only modifies its own repository; external repos strictly read-only |

## 2. Numerical Fidelity Strategy

- Reference implementation (ENR CPU backend) replicates the audited EVO math order of operations:
  RMSNorm formula, causal attention loop (scale `1/sqrt(headDim)`), SwiGLU `silu(gate)*up`,
  sinusoidal PE table, KV-cache concatenation, sampling pipeline.
- Equivalence tests:
  1. Unit-level: op-by-op against hand-computed values.
  2. Model-level: ENR CPU forward vs an independent straightforward reference forward on the same
     weights (different code path, same math), tolerance 1e-5 relative.
  3. Cross-runtime: llama.cpp CPU output for `tiny-evo.gguf`/`evo.gguf` vs ENR (documented
     divergence expectations from R2/R4; assert where semantics should coincide).
- No invented numbers: all performance/timing values come from executed benchmarks.

## 3. Interoperability Contract Freeze

ENR treats the following as frozen contracts (from the audit date):
- `.evo` v2 byte layout and JSON schemas (§2 of FORMAT AUDIT)
- v1 legacy layout (§3)
- GGUF v3 conventions as produced by Darwin Forge (§4)
- canonical tensor names and shapes
- tokenizer semantics (`<0xXX>` byte tokens, special tokens, longest-match encode, incremental
  UTF-8 decode)

Any future deviation must be version-gated. Proposed format changes are documented for later
coordination with Darwin Forge maintainers (see ENR-MODEL-FORMAT-COMPATIBILITY.md).
