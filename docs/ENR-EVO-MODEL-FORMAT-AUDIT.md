# ENR EVO Model Format Audit

Date: 2026-11-16
Status: PHASE 0 deliverable (verified against source code and actual files)

## 1. Summary

Darwin Forge defines a binary model container `.evo` ("EVO Native Model Protocol") with two
generations: **v2** (current, magic `EVO_NAT2`) and **legacy v1** (magic `EVO_ARTIFACT_V1`,
whole-file GZIP). Both are fully specified in source code. **No real `.evo` file exists in the
repository** — every `.evo` file found is a 17-byte placeholder containing the text
`EVO-PACKAGE-BYTES`. Real trained models exist only as **GGUF v3** exports (`evo.gguf`,
`tiny-evo.gguf`, and test copies).

ENR therefore implements: (a) the `.evo` v2 + v1 loaders per the verified spec, (b) a GGUF v3
loader for the real exported models, and (c) a documented test fixture generator that produces
real `.evo` v2 files (validated by round-trip and by numerical equivalence with the same weights
in GGUF and in llama.cpp).

## 2. `.evo` v2 Binary Layout (verified from EvoModelWriter/EvoModelReader)

All integers big-endian, all strings 32-bit-length-prefixed UTF-8 (not `writeUTF`).

```
Offset  Size  Field
0       8     magic "EVO_NAT2" (ASCII)
8       2     formatVersionMajor = 2
10      2     formatVersionMinor = 0
12      2     protocolVersion = 1
14      2     flags = 0
16      4     headerSize = 64
20      4     sectionCount = 5        (note: 6 string sections are actually written)
24      8     createdAt (epoch millis)
32      32    modelContentHash (UTF-8, zero-padded to 32 bytes)
64      str32 modelName
       str32 architecture JSON
       str32 tokenizer JSON
       str32 vocabulary JSON (token -> id)
       str32 manifest JSON (array of tensor descriptors)
       str32 metadata JSON (string -> string)
       4     tensorCount
       per tensor:
         4     compressedLength
         n     GZIP-compressed float payload
```

### Tensor payload encoding (v2)

- Raw payload: IEEE-754 **big-endian** floats, one per element, row-major.
- Each tensor GZIP-compressed individually (compression code 1; 0 = NONE is defined but unused).
- `EvoDtype` codes: F32=0, F16=1, BF16=2, INT8=3, INT4=4. **In practice only F32 is serialized**
  (writer always emits float arrays; F16/quantized dtypes are defined but the writer does not
  emit them).
- Layout code 0 = ROW_MAJOR.

### Manifest entry (JSON)

```
{ "id": int, "name": string, "dtype": int, "shape": [long...], "layout": 0,
  "offset": long, "compressed_size": long, "uncompressed_size": long,
  "compression": 1, "checksum": "hex" }
```

- `checksum` = SHA-256 hex digest of the **uncompressed big-endian float bytes** of that tensor.
- `offset` = cumulative offset across compressed payloads (informational in v2; payloads are
  length-prefixed sequentially).

### Model content hash

SHA-256 over, in order:
1. architecture JSON (UTF-8 bytes, exactly as serialized)
2. tokenizer JSON (UTF-8 bytes)
3. manifest entries **sorted by canonical name**: name UTF-8, 1 byte dtype code, each shape dim as big-endian long
4. all float payloads concatenated as big-endian floats (tensor list order)

This must be replicated byte-for-byte to validate files produced by Darwin Forge.

### Architecture descriptor JSON (v2)

```json
{
  "architecture_family": "evo_llm",
  "vocab_size": int, "d_model": int, "num_heads": int, "num_kv_heads": int,
  "num_blocks": int, "dff": int, "max_seq_len": int,
  "norm_type": "rms_norm", "norm_eps": 1e-5,
  "activation_type": "swiglu",
  "positional_encoding": "rope",      // metadata vs implementation discrepancy, see §6
  "rope_theta": 10000.0,
  "embedding_tied": false
}
```

Legacy camelCase keys (`vocabSize`, `dModel`, ...) are accepted by `fromJsonObject` as fallback.

### Tokenizer descriptor JSON (v2)

```json
{
  "tokenizer_type": "BPE",
  "vocab_size": int,
  "bos_token_id": int, "eos_token_id": int, "unk_token_id": int, "pad_token_id": int,
  "vocabulary_checksum": "hex-or-empty"
}
```

Vocabulary is a separate JSON object mapping token string → id. Special tokens `<s>`, `</s>`,
`<unk>`, `<pad>` update the descriptor IDs when present.

## 3. `.evo` Legacy v1 Layout (verified)

- Whole file is GZIP; inside: `writeUTF` magic `EVO_ARTIFACT_V1`, then `writeUTF` strings
  (16-bit length prefixed) and big-endian scalars via DataOutputStream:
  modelName, modelVersion, createdAt(long), vocabSize, dModel, numHeads, numBlocks, dff,
  maxSeqLen, sampling params (temperature, topP, topK, repeat/frequency/presence penalties),
  special token IDs, vocab map (id, token), metadata map, then weights: (name, shapeLen, shape
  longs, dataLen, dataLen×float).
- No per-tensor checksums in v1; content hash recomputed on read.

## 4. GGUF v3 (real export format — verified against actual files)

Inspected files (real content, F32, `general.file_type = 0`):

| File | d_model | dff | heads | blocks | vocab | ctx | tensors | size |
|---|---|---|---|---|---|---|---|---|
| `evo.gguf` | 64 | 128 | 2 | 1 | 100 | 4 | 12 | 219 232 B |
| `tiny-evo.gguf` | 8 | 16 | 2 | 1 | 16 | 32 | 12 | 5 728 B |

Verified metadata keys: `general.architecture = llama`, `llama.context_length`,
`llama.embedding_length`, `llama.feed_forward_length`, `llama.block_count`,
`llama.attention.head_count`, `llama.attention.head_count_kv`,
`llama.vocab_size`, `llama.attention.layer_norm_rms_epsilon` (1e-5),
`llama.attention.key_length`, `llama.attention.value_length`,
`llama.rope.dimension_count`, `llama.rope.freq_base` (10000),
`tokenizer.ggml.tokens/scores/token_type`, special token IDs.

Tensor names and shapes (verified):

```
token_embd.weight         [d_model, vocab]      (transpose of EVO [vocab, d_model])
blk.{i}.attn_norm.weight  [d_model]
blk.{i}.attn_q/k/v/output.weight  [d_model, d_model]
blk.{i}.ffn_norm.weight   [d_model]
blk.{i}.ffn_gate.weight   [d_model, dff]
blk.{i}.ffn_up.weight     [d_model, dff]
blk.{i}.ffn_down.weight   [dff, d_model]
output_norm.weight        [d_model]
output.weight             [d_model, vocab]      (transpose of EVO [d_model, vocab]... identity)
```

Exporter convention (`OllamaExporter2`, verified): GGUF dims are the reverse of the Java dims
and matrix data is transposed before writing. For square matrices the net effect is the stored
GGUF matrix equals the transpose of the EVO-internal matrix when both are read row-major with
their declared dims. For `token_embd.weight`, GGUF stores `[d_model, vocab]` (EVO holds
`[vocab, d_model]`). ENR canonicalizes all loaded tensors to the EVO layout at load time
(transpose where required), keeping one internal convention.

Token vocab in GGUF: standard `tokenizer.ggml.tokens` string array; EVO models use a
simple vocabulary (chars, words, `<0xXX>` byte tokens) that maps 1:1 onto the
`SimpleBPETokenizer` semantics.

## 5. Existing Model Files Inventory

| Path | Kind | Status |
|---|---|---|
| `.../lib/models/evo.evo` (+8 copies) | `.evo` | **placeholder** (17 bytes, `EVO-PACKAGE-BYTES`) |
| `.../lib/models/evo.gguf`, `test-model.gguf` | GGUF v3 | placeholder copies (identical 219 232 B file as below) |
| `.../lib/llama-cpp/evo.gguf` | GGUF v3 | **real** trained/initialized model, 219 KB |
| `.../lib/llama-cpp/tiny-evo.gguf` | GGUF v3 | **real**, 5.7 KB |
| `.../lib/models/{config.json, tokenizer.json, weights.bin, Modelfile}` | directory export | placeholder-sized stubs |
| `.../llama-cpp/test-evo-*.gguf` | GGUF v3 | copies of `evo.gguf` (same size, per-file verification needed for checksums) |

Conclusion: **there is no usable real `.evo` file to validate against.** Validation strategy:

1. ENR `.evo` v2 writer/reader round-trip on generated fixtures (including corrupted-file tests).
2. Convert real GGUF weights into a `.evo` v2 fixture with the ENR fixture tool.
3. Prove numerical equivalence: ENR(evo fixture) == ENR(gguf source) == llama.cpp(gguf source)
   within float tolerance, for logits of the real trained model.

## 6. Format Issues and Compatibility Risks

1. **PE metadata mismatch**: `positional_encoding: "rope"` + `rope_theta` in the descriptor, but
   the actual forward pass applies **sinusoidal absolute positional encoding** (added, not rotated)
   and no RoPE anywhere. GGUF export includes `llama.rope.*` keys, so llama.cpp will apply RoPE —
   producing different outputs than native EVO math. ENR follows the **actual implementation
   semantics** (sinusoidal PE) for `.evo`, and for GGUF loads uses the same native semantics,
   documenting the divergence from llama.cpp behavior (verified numerically in tests).
2. **`sectionCount=5` written but 6 string sections serialized** — readers must not rely on the count.
3. **v2 header `headerSize=64`** matches the fixed header; hash is 32-byte padded UTF-8 (hex string
   is 64 chars, truncated to 32 bytes — a latent corruption of the stored hash; readers cannot
   verify the full 64-char hex from the stored field, so integrity checks must recompute, not compare).
4. **F16/BF16/INT8/INT4 dtypes defined but never produced**; ENR loader rejects non-F32 payloads
   with a clear diagnostic until a producing implementation is verified (no silent assumptions).
5. **No file-level checksum of the container**; only per-tensor checksums (uncompressed floats).
6. **GGUF exporter placeholder bug**: one export path writes `ffn_gate` and all norm weights as
   all-ones; whether the shipped `evo.gguf` contains trained gate weights must be verified
   numerically (ENR test verifies and documents).
7. **Legacy v1 GZIP + writeUTF** has no checksums and 64 KB `writeUTF` limits; kept for read-only
   compatibility.
8. **`embedding_tied` support**: writer has no code path producing tied models; ENR handles both.
9. **JSON number precision**: manifest shapes as longs — fine. `norm_eps`/`rope_theta` as doubles
   in JSON; ENR parses as float/double to match.
10. **Endianness**: all `.evo` numeric data is big-endian; GGUF is little-endian. Loaders must be
    explicit and tested (cross-endian fixtures).

## 7. Format Version Handling Policy (ENR)

- Accept `.evo` v2 (major=2), accept legacy v1 for read-only compatibility.
- Reject unknown major versions with explicit diagnostics naming the version.
- Accept GGUF v2/v3 for llama-family models with the verified conventions.
- All format-specific logic isolated in `enr-model-loader` behind a common `ModelSource` API.
