# ENR Model Format Compatibility

Date: 2026-11-16
Status: verified against real model files and independent execution

## 1. Supported Formats

| Format | Magic | ENR support | Status |
|---|---|---|---|
| `.evo` v2 | `EVO_NAT2` | read + write | verified by round-trip + cross-implementation checks; no real Darwin Forge `.evo` file exists yet |
| `.evo` v1 (legacy) | gzip(`EVO_ARTIFACT_V1`) | read only | implemented per audited spec |
| GGUF v3 | `GGUF` v3 | read | **verified against real Darwin Forge exports** (`evo.gguf`, `tiny-evo.gguf`) |
| GGUF v2 | `GGUF` v2 | read | same parser as v3 |

## 2. Verification Methodology

1. Audited the EVO reference implementation line-by-line (read-only).
2. Located the real exported models: `evo.gguf` (d_model=64, vocab=100) and
   `tiny-evo.gguf` (d_model=8, vocab=16).
3. Built an independent C oracle against the llama.cpp CPU library that feeds raw
   token ids directly (bypassing the tokenizer) and dumps logits.
4. Empirically determined the exact tensor layout conventions by matching ENR
   forward logits to the llama.cpp oracle (tolerance 2e-3 = llama's F16 KV cache
   quantization; identical argmax/top-5 ordering).
5. ENR `.evo` v2 writer/reader round-trip verified byte-for-byte; corrupted-file
   matrix (bad magic, truncated payloads, checksum violations) rejected with
   diagnostics.

## 3. Canonical Tensor Layout (ENR internal)

All tensors are stored row-major in the EVO reference convention:

- `token_embd.weight` `[vocabSize, dModel]`
- `blk.{i}.attn_q/k/v/output.weight` `[dModel, dModel]`
- `blk.{i}.ffn_gate/up.weight` `[dModel, dff]`, `blk.{i}.ffn_down.weight` `[dff, dModel]`
- `blk.{i}.attn_norm/ffn_norm.weight`, `output_norm.weight` `[dModel]`
- `output.weight` `[dModel, vocabSize]` (untied)

GGUF conversion rules (empirically verified):

- `token_embd.weight`: flat data reinterpreted as `[vocab, dModel]` (no reorder)
- all other 2D weights: flat data reshaped `[d1, d0]` then transposed
- 1D norms: as-is
- GGUF tensor offsets are relative to the aligned tensor-data section start

## 4. Math Semantics Contract

- RMSNorm, eps 1e-5, weight scale
- Causal multi-head attention, scale `1/sqrt(headDim)`, grouped-query attention
  (`numKvHeads <= numHeads`)
- SwiGLU FFN: `silu(x·W_gate) * (x·W_up)`, then `·W_down`
- Positional encoding:
  - `.evo` models: sinusoidal absolute PE (the audited implementation behavior,
    despite the descriptor's `"rope"` default)
  - GGUF models: RoPE (`llama.rope.dimension_count`, `freq_base`) — chosen so that
    ENR matches the GGUF export semantics and the llama.cpp oracle
- Sampling: greedy (temperature<=0), temperature softmax, top-K, top-P,
  repeat/frequency/presence penalties

## 5. Integrity

- `.evo` v2: per-tensor SHA-256 checksums (big-endian floats) verified on load;
  model content hash recomputed (stored header hash is truncated to 32 bytes by
  the reference writer and treated as informational)
- GGUF: no per-tensor checksums in format; structural validation + NaN scans
- Dtype policy: F32 payloads fully supported; F16/BF16/INT8/INT4 declared by the
  format but rejected with explicit diagnostics until a verified producer exists

## 6. Format Evolution Strategy

- Version-gated readers; unknown major versions rejected with diagnostics
- All format-specific code isolated in `enr-model-loader` behind `ModelSources`
- Proposed future changes (documented for Darwin Forge coordination):
  1. Produce a real `.evo` v2 file from the export pipeline for validation
  2. Reconcile `positional_encoding` metadata with actual math
  3. Add a full 64-byte content hash field (current writer truncates to 32)
  4. Serialize real F16/quantized payloads if quantization is used in export
  5. Consider a container-level checksum section
