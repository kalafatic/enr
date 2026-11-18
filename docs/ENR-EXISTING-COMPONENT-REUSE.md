# ENR Existing Component Reuse

Date: 2026-11-16
Status: PHASE 0 deliverable

## 1. Policy

ENR is an independent implementation. It must not depend on EVO, Darwin Forge, TRYST, DStudio,
DCode, Eclipse RCP/OSGi/SWT, Ollama or llama.cpp at runtime. Reuse is limited to:

- **Interoperability contracts** (file formats, math semantics, naming conventions) verified from
  read-only inspection.
- **Reference behavior** for numerical validation (llama.cpp CPU build as an external test oracle).
- **Ideas/patterns** re-expressed as independent code.

## 2. Component-by-Component Assessment

| Existing Component (EVO) | Location | Reuse Decision for ENR |
|---|---|---|
| `EvoModelWriter`/`EvoModelReader` (`.evo` v2/v1) | `forge.model/protocol` | Reimplement independently. Format contract verified line-by-line (see FORMAT AUDIT). |
| `EvoArchitectureDescriptor` / `EvoTokenizerDescriptor` / `EvoTensorDescriptor` / `EvoDtype` | `forge.model/protocol` | Reimplement as ENR records/enums with identical JSON field names and enum codes. |
| `EvoModelContentHash` (SHA-256 scheme) | `forge.model/protocol` | Reimplement; hash algorithm details copied as a contract (must match byte-for-byte to validate existing files). |
| `EvoModelIntegrity` / `EvoModelInspector` | `forge.model/protocol` | Reimplement with equivalent checks + stricter ENR diagnostics. |
| `SimpleTensor` (float, row-major) | `forge.math` | Reimplement in `enr-core` with explicit matmul conventions; add dtype-aware storage. |
| `EvoLlmModel` / `TransformerBlock` / `MultiHeadAttention` / `FeedForward` / `RMSNorm` / `PositionalEncoding` / `Embedding` | `forge.model/llm` | Reimplement in `enr-inference` replicating exact math semantics. |
| `ReferenceEvoInferenceEngine` + `KVCache` + sampling | `forge.model/inference` | Reimplement; sampling algorithm contract preserved (greedy/temp/topK/topP/penalties). |
| `SimpleBPETokenizer` (longest-match + `<0xXX>` byte fallback + incremental UTF-8 decoder) | `forge.tokenizer` | Reimplement in `enr-tokenizer`; token semantics (`<0xXX>`, special tokens) are a format contract. |
| `EvoTokenizer`, `WhitespaceTokenizer`, `WordTokenizer` | `forge.tokenizer` | Not required initially; `SimpleBPETokenizer` semantics cover `.evo`/GGUF vocabularies. |
| GGUF parser (`GGUFReader`) | `forge.agent.api/gguf` | Reimplement in `enr-model-loader` (GGUF v3 spec is public; verified against real files). |
| `OllamaExporter2` (GGUF writer) | `forge.agent.api/export` | Not needed for inference; used read-only to derive GGUF layout conventions. |
| `LlamaService` / `LlamaCppRunner` (process-based serving) | `forge.controller` | Explicitly NOT reused — ENR's purpose is to replace this pattern with native inference. |
| GPU detection via `nvidia-smi` subprocess | `forge.gpu` | Pattern noted; ENR CUDA backend queries the CUDA driver API directly (FFM), with `nvidia-smi` only as optional diagnostics. |
| Eclipse RCP / OSGi / SWT runtime | whole platform | Explicitly excluded. ENR is a plain Java application. |
| llama.cpp binaries / Ollama | external | Used ONLY as a test oracle for numerical cross-validation of real GGUF models. Never a runtime dependency. |

## 3. Contracts Adopted Verbatim (interoperability, not code)

1. `.evo` magic bytes: `EVO_NAT2` (v2), `EVO_ARTIFACT_V1` (legacy v1).
2. `.evo` v2 header field layout and 32-bit length-prefixed UTF-8 string encoding.
3. `EvoDtype` numeric codes: F32=0, F16=1, BF16=2, INT8=3, INT4=4.
4. Tensor layout code 0 = ROW_MAJOR; compression codes 0=NONE, 1=GZIP.
5. Tensor checksum: SHA-256 hex over big-endian IEEE-754 floats.
6. Model content hash composition (architecture JSON + tokenizer JSON + sorted manifest + float payloads).
7. Canonical weight names (Section 6 of ARCHITECTURE BASELINE).
8. Tokenizer special-token semantics: `<unk>`, `<s>`, `</s>`, `<pad>`, byte tokens `<0xXX>`.
9. GGUF v3 tensor naming and transpose conventions as produced by the EVO exporter.
10. Math semantics: sinusoidal absolute PE, RMSNorm(1e-5), causal MHA (scale 1/sqrt(headDim)),
    SwiGLU FFN, pre-norm residuals, KV-cache concatenation.

## 4. What ENR Adds (independent value)

- Dtype-aware tensor storage and a compute backend SPI (CPU / CUDA).
- Memory-mapped, lazy, selective tensor loading for large models.
- Multi-model lifecycle, resource accounting, scheduling, sessions, cancellation.
- HTTP API, Java 21 client, CLI, metrics — none of which exist in the forge modules.
- Strict validation with actionable diagnostics for the model format.
- CUDA backend operating via the driver API (no toolkit dependency at runtime).

## 5. Integration Recommendations for EVO (documented, NOT implemented here)

1. Consider writing a real `.evo` file from `GpuTrainerMain`/export pipelines to validate the
   format end-to-end (currently only GGUF files are produced).
2. The architecture descriptor advertises `positional_encoding: "rope"` while the implementation
   uses sinusoidal absolute PE — reconcile metadata with actual math.
3. GGUF exporter writes `ffn_gate` and norm weights as all-ones placeholders in one code path
   (`OllamaExporter2`) — verify and fix in Darwin Forge.
4. Consider adopting ENR as the native inference provider to retire the llama.cpp subprocess path.
