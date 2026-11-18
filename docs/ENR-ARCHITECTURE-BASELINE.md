# ENR Architecture Baseline

Date: 2026-11-16
Status: PHASE 0 deliverable (verified read-only audit of external repositories)

## 1. Scope

This document establishes the verified baseline of the existing EVO platform and Darwin Forge
architecture that is relevant to EVO Native Runtime (ENR). All findings are based on direct
inspection of source code and actual model files in the read-only reference repository
`evolution-121126` (EVO platform). No external repository was modified.

## 2. Repositories Inspected

| Repository | Path | Mode | Relevant Content |
|---|---|---|---|
| EVO platform | `/home/petr/git/evolution-121126` | read-only | Darwin Forge modules, `.evo` protocol, tokenizers, inference engine, real GGUF exports |
| ENR (this) | `/home/petr/git/evo-native-runtime-161126` | read-write | target implementation repository (empty at audit start) |
| llama.cpp | `/home/petr/git/llama.cpp` | read-only (reference only) | CPU build available for numerical cross-validation |

TRYST, DStudio, DCode were not required for the format audit; they are consumers, not producers
of the model format. They are documented in later integration deliverables.

## 3. EVO Platform Structure (verified)

Key Darwin Forge modules in the EVO platform (all Eclipse RCP / OSGi based):

- `eu.kalafatic.evolution.forge.math` — Tensor API (`Tensor`) + row-major float implementation (`SimpleTensor`)
- `eu.kalafatic.evolution.forge.model` — model core (`EvoLlmModel`, `EvoLlmArchitecture`), transformer
  components, inference (`ReferenceEvoInferenceEngine`, `KVCache`), model protocol
  (`EvoModelReader`/`EvoModelWriter` — the `.evo` container format)
- `eu.kalafatic.evolution.forge.tokenizer` — tokenizers (`SimpleBPETokenizer`, `EvoTokenizer`,
  `WhitespaceTokenizer`, `WordTokenizer`)
- `eu.kalafatic.evolution.forge.trainer` — training (checkpoints, optimizers)
- `eu.kalafatic.evolution.forge.runtime` — training execution engines (graph/linear)
- `eu.kalafatic.evolution.gpu` — GPU diagnostics/training via `nvidia-smi` process execution
- `eu.kalafatic.evolution.forge.agent.api` — GGUF parser, Ollama/llama.cpp exporters (`OllamaExporter2`)
- `eu.kalafatic.evolution.controller` — `LlamaService` (runs llama.cpp binaries as subprocess),
  `OllamaService`-compatible message handling

## 4. How EVO Executes Models Today (verified)

1. Darwin Forge trains `EvoLlmModel` instances in-process (Java tensors).
2. Models are exported primarily as **GGUF v3** files via `OllamaExporter2` (llama architecture family).
3. Production inference in EVO is performed by **external processes**: llama.cpp binaries
   (`LlamaCppRunner` / `LlamaService`) or Ollama. The Java-native `ReferenceEvoInferenceEngine`
   exists and is used for Darwin candidate evaluation, but the primary serving path is llama.cpp/Ollama.
4. The `.evo` binary container format is fully defined (`EvoModelReader`/`EvoModelWriter`, v2 + legacy v1)
   but **no real `.evo` file exists in the repository** — only 17-byte placeholders.

Implication for ENR: ENR must implement real native inference (no llama.cpp/Ollama), support the
defined `.evo` format, and additionally support GGUF v3 as the de-facto export format in which real
trained EVO models currently exist.

## 5. Existing Native Math Semantics (verified, must be replicated)

The canonical EVO model math (`EvoLlmModel`, `TransformerBlock`, `MultiHeadAttention`, `FeedForward`,
`RMSNorm`, `PositionalEncoding`, `Embedding`):

1. Embedding lookup: `[vocabSize, dModel]` row-major, token id → row.
2. **Sinusoidal absolute positional encoding** added to embeddings
   (`sin(pos / 10000^(i/dModel))`, `cos(...)` for odd indices). No learned PE, no RoPE in the
   forward pass despite the architecture descriptor claiming `"rope"`.
3. Per block (pre-norm):
   - `RMSNorm(x) = x / sqrt(mean(x^2) + 1e-5) * weight`, weight init 1.0
   - Q = x·WQ, K = x·WK, V = x·WV with `[dModel, dModel]` matrices
   - MHA: `numHeads` heads of `headDim = dModel / numHeads`, scale `1/sqrt(headDim)`,
     causal masking, softmax, concat, `·WO`
   - residual add
   - `RMSNorm` → FFN SwiGLU: `h = silu(x·W1) * (x·W3)`, `out = h·W2`
     with W1,W3 `[dModel, dff]`, W2 `[dff, dModel]`
   - residual add
4. Final `RMSNorm`, then `lmHead` `[dModel, vocabSize]` (untied by default).
5. KV cache: concatenated K/V per layer, re-used across autoregressive steps.
6. Sampling: greedy if `temperature <= 0`; else temperature softmax, top-K, top-P (nucleus),
   repeat/frequency/presence penalties.

Tensor storage: Java `float[]`, row-major. `matmul`: `[m,k] × [k,n] → [m,n]`, ikj loop order,
with parallel paths for large matrices (note: accumulation order affects last-ulp results).

## 6. Weight Naming Convention (verified, canonical)

```
token_embd.weight            [vocabSize, dModel]
blk.{i}.attn_norm.weight     [dModel]
blk.{i}.attn_q.weight        [dModel, dModel]
blk.{i}.attn_k.weight        [dModel, dModel]
blk.{i}.attn_v.weight        [dModel, dModel]
blk.{i}.attn_output.weight   [dModel, dModel]
blk.{i}.ffn_norm.weight      [dModel]
blk.{i}.ffn_gate.weight      [dModel, dff]
blk.{i}.ffn_up.weight        [dModel, dff]
blk.{i}.ffn_down.weight      [dff, dModel]
output_norm.weight           [dModel]
output.weight                [dModel, vocabSize]   (untied lm head)
```

Parameter order in `EvoLlmModel.parameters()`: embedding, then per block
(attn_norm, WQ, WK, WV, WO, ffn_norm, W1, W3, W2), then output_norm, then lm head.
This order is the positional fallback when loading directory exports (`weights.bin`).

## 7. Environment Verified on This Machine

- Java 21 available: `/usr/lib/jvm/java-21-openjdk-amd64` (default `java` is JDK 8 — build scripts must pin JDK 21)
- Maven 3.8.7
- NVIDIA RTX 5070 Ti, 16 GB VRAM, driver 595.99.02 (CUDA 13.2 driver capability)
- `libcuda.so.1` present (CUDA driver API available); **no nvcc / CUDA toolkit / NVRTC installed**
- llama.cpp CPU build exists (reference oracle only, not a runtime dependency)

## 8. Architectural Consequences for ENR

1. ENR is a standalone JDK 21 application with zero runtime dependencies beyond the JDK.
2. The `.evo` v2 container and legacy v1 formats are the interoperability contract.
3. GGUF v3 loading is required to run the real trained EVO models that exist today.
4. The native math (sinusoidal PE, RMSNorm, SwiGLU, causal MHA) is the reference semantics
   ENR must reproduce bit-compatibly within float tolerance.
5. CUDA backend must work with driver API only (FFM + precompiled PTX), since no toolkit is installed.
6. ENR must be independently buildable/testable via Maven with JDK 21.
