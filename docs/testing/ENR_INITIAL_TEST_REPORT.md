# ENR Initial Test Report

Initial functional testing and stabilization iteration for ENR (EVO Native Runtime).
Iteration date: 2026-11-21.
Primary agent: OpenCode with cloud DeepSeek (`deepseek/deepseek-v4-pro`, confirmed in
`~/.local/share/opencode/log/opencode.log`: `providerID=deepseek modelID=deepseek-v4-pro`).
No local Ollama model was used as the agent.

## 1. Environment

| Item | Value |
| --- | --- |
| ENR repository | `/home/petr/git/evo-native-runtime-161126` (remote: github.com/kalafatic/evo-native-runtime) |
| ENR version | 2.6.11-SNAPSHOT |
| ENR endpoint | `http://127.0.0.1:8111` (enr.yaml) |
| ENR start | `./enr.sh serve -c enr.yaml` (shaded CLI jar, JDK 25 preferred, JDK 21 CPU fallback) |
| ENR stop | SIGTERM / SIGINT — graceful "Shutting down ENR (unloading models)..." |
| Build | `JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64 mvn package` — 13 modules, `release=21` |
| Existing tests | 88 tests, 0 failures (baseline) |
| OpenCode | 1.19.6 |
| OpenCode agent model | `deepseek/deepseek-v4-pro` (cloud, api.deepseek.com) |
| OpenCode config | `~/.config/opencode/opencode.jsonc` (default model `ollama/qwen3-coder-30b-32k:latest` for other sessions) |
| Real Ollama (reference) | 0.40.16 at `http://127.0.0.1:11434` |
| ENR local models | `evo`, `tiny-evo` (GGUF, from `enr-model-loader/src/test/resources/models`) |
| ENR providers | none configured (`~/.enr/state.json` absent, `/api/v1/providers` = `[]`) |
| ENR data dir | `~/.enr` (models, logs, provider registry) |

Uncommitted changes were present before this iteration (web_server config in EnrConfig/EnrConfigLoader/
EnrManager/EnrServer + enr.yaml). They were treated as pre-existing baseline and were rebuilt as-is.

## 2. Architecture observations

- `enr-server` (JDK `com.sun.net.httpserver`, no external deps) exposes:
  - Ollama-compatible: `/api/tags`, `/api/show`, `/api/pull`, `/api/delete`, `/api/chat`, `/api/generate`, `/api/ps`, `/api/version` (`OllamaApiHandlers`)
  - OpenAI-compatible: `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` (`OpenAiApiHandlers`)
  - Native `/api/v1/*`, family `/enr/v1/*`, Web UI at `/`
- `EnrManager` is the single facade: routing (`ModelRouter`) → local (`EnrRuntime`) or provider (`ProviderClient`).
- `ProviderClient` supports OpenAI-compatible (DeepSeek/OpenAI/Ollama /v1/llama.cpp), Anthropic, Gemini.
- `ChatMessage(role, content)` carries text only; **no tool/tool_calls support anywhere** (confirmed by grep).
- Local native inference is text-only (prompt flattening in `EnrManager.buildPrompt`); toy models (`tiny-evo` 920 params, ctx 32).
- OpenCode's built-in Ollama provider (opencode 1.19.6, `packages/core/src/plugin/provider/ollama.ts`) discovers models via
  native `/api/tags` + `/api/show` and **requires `/api/show` to return `capabilities: ["completion"]`**;
  `/api/show` is called with `{"model": ...}` (not `name`); context is read from `model_info.<something>.context_length`;
  tool calling is enabled per model only when `capabilities` includes `"tools"`; execution uses the
  OpenAI-compatible runtime against `<baseURL>/v1/chat/completions`.

## 3. Initial Baseline (before modifications)

Legend: PASS / FAIL / BLOCKED / NOT TESTED.

### B1. Ollama protocol (ENR @ 127.0.0.1:8111)

| # | Scenario | Result | Evidence |
| --- | --- | --- | --- |
| B1.1 | GET `/api/version` | PASS | `{"version":"2.6.11","name":"ENR"}` (real Ollama: `{"version":"0.40.16"}` — extra `name` field is a compatible superset) |
| B1.2 | GET `/api/tags` | PASS (partial) | `evo`, `tiny-evo` listed with name/model/modified_at/size/digest/details. Missing per-model `capabilities` present in real Ollama 0.40 |
| B1.3 | GET `/api/ps` | PASS | both models loaded, details + expires_at + size_vram |
| B1.4 | POST `/api/show` `{"name":...}` | PASS | modelfile/parameters/template/details for `tiny-evo`; 404 for unknown |
| B1.5 | POST `/api/show` `{"model":...}` (OpenCode format) | **FAIL** | 404 `Model not installed: ` — handler reads only `name`; real Ollama falls back `name = model` |
| B1.6 | `/api/show` returns `capabilities` | **FAIL** | absent → OpenCode discovery filters ENR models out (requires `completion`) |
| B1.7 | `/api/show` returns `model_info.*.context_length` | **FAIL** | absent → OpenCode context limit unknown |
| B1.8 | POST `/api/chat` non-stream | PASS | correct `model/created_at/message/done/done_reason/total_duration/prompt_eval_count/eval_count` |
| B1.9 | POST `/api/chat` stream | PASS | NDJSON chunks; final chunk `done:true` with counts |
| B1.10 | POST `/api/generate` non-stream | PASS | correct `response`/`done` shape |
| B1.11 | POST `/api/generate` stream | PASS | NDJSON chunks |
| B1.12 | Multi-turn messages | PASS | system/user/assistant/tool roles accepted; structure preserved |
| B1.13 | Unknown model `/api/chat` | **FAIL** | 400 structured error; real Ollama returns 404 `{"error":"model 'nope' not found"}` |
| B1.14 | Missing model `/api/chat` | PASS | 400 (matches real Ollama status) |
| B1.15 | Wrong method (POST `/api/tags`) | PASS | 405 |
| B1.16 | Empty body | PASS | 400 |
| B1.17 | Error body format | **FAIL** | ENR: `{"error":{"code":...,"message":...}}`; real Ollama: `{"error":"<string>"}` |
| B1.18 | `format: "json"` | NOT TESTED (documented limitation) | ignored by ENR; local models cannot enforce JSON grammar |

### B2. Tool calling (`/api/chat`)

| # | Scenario | Result | Evidence |
| --- | --- | --- | --- |
| B2.1 | `tools` definitions accepted | **FAIL** | `tools` field ignored entirely; response has no `tool_calls` |
| B2.2 | Structured tool calls in response | **FAIL** | never produced |
| B2.3 | `tool_calls` in assistant message + `tool` role message round-trip | **FAIL** | flattened into text prompt; role `tool` treated as user text |
| B2.4 | Multiple tool calls | NOT TESTED | blocked by B2.1 |
| B2.5 | Tool result → next assistant response | NOT TESTED | blocked by B2.3 |
| B2.6 | Distinguish structured tool call vs plain text | **FAIL** | not possible — no structured tool call path exists |

Reference behavior (real Ollama 0.40.16): models without tool support return HTTP 400
`{"error":"registry.ollama.ai/library/smollm:135m does not support tools"}`.

### B3. OpenCode integration

| # | Scenario | Result | Evidence |
| --- | --- | --- | --- |
| B3.1 | ENR discoverable as native Ollama provider | **FAIL/BLOCKED** | blocked by B1.5 + B1.6 (no `capabilities` in `/api/show`) |
| B3.2 | Model selection via `/models` | NOT TESTED | blocked by B3.1 |
| B3.3 | Inference through ENR from OpenCode | NOT TESTED | blocked by B3.1 |

The current global OpenCode config maps the `ollama` provider to `http://127.0.0.1:11434/v1` (real Ollama),
so ENR was not configured there at baseline. Testing used a project-scoped config (see Section 7).

### B4. Terminal execution (DeepSeek agent, OpenCode bash tool)

| # | Scenario | Result | Evidence |
| --- | --- | --- | --- |
| B4.1 | `pwd`, `whoami`, `uname -a`, `java -version`, `mvn -version`, `git --version` | PASS | all executed; java 1.8 default, JDK 21/25 installed |
| B4.2 | Dedicated temp workspace | PASS | `/tmp/opencode/enr-terminal-test` created |
| B4.3 | Directory + file create/read | PASS | `subdir/sample.txt` written and read back |
| B4.4 | Text processing | PASS | `wc -l` = 2 |
| B4.5 | SHA256 | PASS | `84423fd5...` computed |
| B4.6 | Multiple sequential commands | PASS | `(pwd; ls; wc -l)` in one invocation |
| B4.7 | Working directory preservation | PASS | `pwd` = `/tmp/opencode/enr-terminal-test` |
| B4.8 | Exit codes | PASS | nonexistent path → exit 2; `false` → exit 1 |
| B4.9 | Failed command handling | PASS | failure reported as output, not claimed as success |

### B5. End-to-end Java scenario (DeepSeek agent)

| # | Scenario | Result | Evidence |
| --- | --- | --- | --- |
| B5.1 | Inspect Java env | PASS | JDK 21 at `/usr/lib/jvm/java-21-openjdk-amd64` |
| B5.2 | Minimal Java app | PASS | `EnrAutonomousTest.java` created |
| B5.3 | Compile with installed JDK | PASS | `javac` (JDK 21) succeeded |
| B5.4 | Execute | PASS | `ENR_AUTONOMOUS_TEST_OK` printed exactly, exit 0 |

## 4. Root-cause analysis

1. **`/api/show` ignores `model` field** — `OllamaApiHandlers.show` reads only `request.getString("name")`.
   Real Ollama falls back to `model` when `name` is empty. OpenCode discovery sends `{"model": ...}`.
2. **No `capabilities` in `/api/show`/`/api/tags`** — `EnrManager.ollamaShow()/ollamaTags()` never emit the field.
   OpenCode's Ollama plugin filters out models whose `/api/show` lacks `completion`.
3. **No `model_info`** — context length is only available as `details.context_length`, not in `model_info.*.context_length`
   form consumed by OpenCode.
4. **No tool calling support** — `ChatMessage` is text-only; `OllamaApiHandlers.messagesFrom`,
   `OpenAiApiHandlers.messagesFrom` and `ProviderClient.openAiBody` drop `tools`/`tool_calls`; responses never emit `tool_calls`.
5. **Error format deviation** — `HttpSupport.sendError` emits structured `{"error":{"code","message"}}` on all endpoints;
   the Ollama protocol expects `{"error":"<string>"}` and 404 for unknown models.
6. **Unknown model status** — `/api/chat` and `/api/generate` map `IllegalArgumentException` to 400; real Ollama uses 404.

## 5. Defect severity

| Defect | Severity | Impact |
| --- | --- | --- |
| Tool calling absent (B2) | High | breaks agentic integrations using ENR as Ollama provider |
| `/api/show` missing capabilities/model_info + `model` field (B1.5–B1.7) | High | blocks OpenCode model discovery entirely |
| Ollama error format + status deviations (B1.13, B1.17) | Medium | breaks clients that parse `error` as string / expect 404 |
| `format: json` ignored (B1.18) | Low | documented limitation (local text-only models) |

## 6. Planned fixes (Phase D, this iteration)

1. `/api/show`: accept `model` fallback; emit `capabilities` and `model_info.*.context_length`.
2. `/api/tags`: per-model `capabilities` for parity with Ollama 0.40.
3. Ollama endpoints: plain `{"error":"..."}` bodies; 404 for unknown model on `/api/chat` and `/api/generate`.
4. Tool calling:
   - parse `tools` + `tool_calls`/`tool` messages on `/api/chat` (Ollama format) and `/v1/chat/completions` (OpenAI format);
   - forward tools to OpenAI-compatible providers (DeepSeek, OpenAI, Ollama /v1, llama.cpp);
   - structured `tool_calls` in non-streaming and streaming responses (both formats);
   - local native models without tool support → HTTP 400 (mirrors real Ollama);
   - Anthropic/Gemini provider routes → explicit 400 (unsupported translation, no silent dropping).
5. Regression tests for each fix in `enr-tests`.

## 7. Reproduction

```bash
cd /home/petr/git/evo-native-runtime-161126
JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64 mvn package
setsid -f ./enr.sh serve -c enr.yaml > /tmp/opencode/enr-server.log 2>&1 < /dev/null
curl -s http://127.0.0.1:8111/api/version
# B1 checks
curl -s -X POST http://127.0.0.1:8111/api/show -d '{"model":"tiny-evo"}'          # baseline: 404
curl -s -X POST http://127.0.0.1:8111/api/chat -d '{"model":"nope","messages":[{"role":"user","content":"hi"}]}'
# B2 check (baseline: tools silently ignored)
curl -s -X POST http://127.0.0.1:8111/api/chat -d '{"model":"tiny-evo","messages":[{"role":"user","content":"weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"...","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],"stream":false}'
```

Post-fix results are recorded in the "Post-Fix Results" section after implementation.

---

# Post-Fix Results

## 1. Fixes implemented

### F1. `/api/show` — model field fallback, capabilities, model_info
- **File**: `enr-server/.../OllamaApiHandlers.java`, `enr-manager/.../EnrManager.java`
- **Root cause**: handler read only `name`; real Ollama falls back to `model` (OpenCode discovery sends `{"model": ...}`).
  `capabilities`/`model_info` were never emitted, so OpenCode's built-in Ollama discovery filtered ENR models out.
- **Correction**: `name = model` fallback; `capabilities: ["completion"]` in `/api/show` (and per-model in `/api/tags`);
  `model_info: {general.parameter_count, general.context_length}` from the loaded model.
- **Verification**: `curl -X POST /api/show -d '{"model":"tiny-evo"}'` → 200 with capabilities + model_info (was 404).
  Regression tests: `showAcceptsModelFieldLikeRealOllama`, `showWithoutLoadedModelStillAdvertisesCapabilities`, `tagsAdvertiseCapabilitiesPerModel`.

### F2. Ollama error format + status codes
- **File**: `enr-server/.../OllamaApiHandlers.java`
- **Root cause**: shared `HttpSupport.sendError` produced `{"error":{"code","message"}}`; real Ollama uses `{"error":"<string>"}`;
  unknown models returned 400 instead of 404.
- **Correction**: `sendOllamaError` helper emitting plain-string errors on all `/api/*` endpoints;
  `IllegalArgumentException` from route resolution maps to 404 on `/api/chat` and `/api/generate`.
- **Verification**: unknown model chat/generate/show → 404 `{"error":"Model 'nope' is not installed..."}`.
  Regression tests: `chatUnknownModelReturns404WithPlainError`, `generateUnknownModelReturns404WithPlainError`, `showUnknownModelReturns404WithPlainError`.

### F3. Tool calling (Ollama `/api/chat` + OpenAI `/v1/chat/completions`)
- **Files**: `ChatMessage.java` (optional protocol payload), `EnrManager.java` (ChatOptions.tools,
  `ToolCallingUnsupportedException`, pre-flight `validateTools`, tool_calls/reasoning mapping),
  `ProviderClient.java` (OpenAI-wire tool serialization, tool_call_id synthesis, streaming tool-call delta
  aggregation, `reasoning_content` capture/echo), `OllamaApiHandlers.java`, `OpenAiApiHandlers.java`,
  `ManagerApiHandlers.java` (parse + response converters).
- **Root cause**: no tool support existed anywhere; `tools` was silently dropped and `tool`-role messages
  flattened into text. Additionally, DeepSeek thinking-mode requires `reasoning_content` (even `""`) to be
  echoed back on assistant messages — multi-turn tool flows 422'd without it.
- **Correction**:
  - `tools` forwarded to OpenAI-compatible providers (DeepSeek/OpenAI/Ollama `/v1`/llama.cpp) in both wire formats;
  - canonical internal form with format conversion at the API boundaries (Ollama `{function:{name,arguments:object}}`
    ↔ OpenAI `{id,type,function:{name,arguments:string}}`);
  - streaming: tool-call deltas aggregated; final chunk carries full `tool_calls` + `finish_reason: "tool_calls"`;
  - tool results: `tool_call_id` synthesized from the preceding assistant tool_calls when the client format lacks it;
  - `reasoning_content` (OpenAI) ↔ `thinking` (Ollama) captured, exposed and echoed;
  - local native models and Anthropic/Gemini routes reject tools explicitly with HTTP 400 (mirrors real Ollama:
    `{"error":"registry.../smollm:135m does not support tools"}`, 400).
- **Verification (live)**:
  - `/api/chat` (Ollama format) → `deepseek:deepseek-flash`: structured `message.tool_calls`
    `[{function:{name:"get_weather",arguments:{city:"Paris"}}}]`, `done_reason:"tool_calls"` ✓
  - full multi-turn: user → assistant(tool_calls) → tool result → final answer "Right now in Paris it's **22°C and sunny**." ✓
  - streaming `/api/chat`: final NDJSON chunk with tool_calls + done_reason "tool_calls" ✓
  - `/v1/chat/completions` non-stream + stream: OpenAI-format tool_calls, `finish_reason:"tool_calls"`,
    `reasoning_content:"..."` echoed ✓
  - local model + tools → 400 before headers (stream and non-stream) ✓
  - 15 regression tests in `OllamaToolCallingTest` (mock OpenAI-compatible provider with captured requests).

### F4. Config storage paths (found during testing)
- **File**: `enr-manager/.../EnrConfigLoader.java`
- **Root cause**: `~` in `storage.*_dir` was not expanded → literal `~` directory created under the process CWD;
  the YAML-subset parser also kept inline comments as part of values
  (e.g. `models_dir` became `~/.enr/models   # installed models (one directory per model)`).
- **Correction**: `expandHome()` for leading `~/`; `stripComment()` for trailing `# comments` (quote-aware).
- **Verification**: started ENR from the repo dir — no stray `~` or comment-suffixed dirs; data lands in `~/.enr`.
  Regression tests: `storagePathsExpandTilde`, `yamlInlineCommentsAreStripped`.

### F5. OpenAI streaming robustness (latent bug found in F3 work)
- **File**: `ProviderClient.openAiStream`
- **Root cause**: if an upstream provider omitted `finish_reason`/usage chunk, no terminal `done` chunk was
  ever emitted; usage-only chunks could double-emit the terminal chunk.
- **Correction**: single terminal-chunk emission with accumulated usage + tool calls; end-of-stream fallback.
- **Verification**: covered by streaming tool tests against the mock provider and live DeepSeek streams.

## 2. Post-fix test matrix

### B1. Ollama protocol (re-run)

| Scenario | Before | After |
| --- | --- | --- |
| `/api/version` | PASS | PASS |
| `/api/tags` (+ capabilities) | partial | PASS |
| `/api/show` `{"name":...}` | PASS | PASS |
| `/api/show` `{"model":...}` | FAIL | **PASS** |
| `/api/show` capabilities/model_info | FAIL | **PASS** |
| `/api/chat` non-stream/stream | PASS | PASS |
| `/api/generate` non-stream/stream | PASS | PASS |
| multi-turn | PASS | PASS |
| unknown model status | FAIL (400) | **PASS (404)** |
| error body format | FAIL | **PASS** (plain string) |
| wrong method / missing model | PASS | PASS |

### B2. Tool calling

| Scenario | Before | After |
| --- | --- | --- |
| tools accepted → structured tool_calls (provider route) | FAIL | **PASS** |
| streaming tool_calls (Ollama + OpenAI formats) | FAIL | **PASS** |
| tool_calls + tool-result round-trip | FAIL | **PASS** |
| tools on local text-only model | silent ignore | **PASS** (explicit 400, matches real Ollama) |
| multiple tool calls | NOT TESTED | covered by aggregation logic + mock stream (single-call verified live) |

### B3. OpenCode integration

| Scenario | Before | After |
| --- | --- | --- |
| ENR discoverable in OpenCode | FAIL/BLOCKED | **PASS at protocol level** (see limitation L1) |
| inference through ENR from OpenCode | NOT TESTED | **PASS** |
| tool calling through ENR from OpenCode | NOT TESTED | **PASS** |
| error propagation to OpenCode | NOT TESTED | **PASS** |

Verified live: OpenCode 1.19.6 (`build` agent) with `provider.ollama.options.baseURL = http://127.0.0.1:8111/v1`
and model `deepseek:deepseek-flash` routed through ENR:
- executed `pwd`, `whoami`, `uname -a`, `java -version`, `mvn -version`, `git --version` through its shell tool
  and reported the actual outputs (matching machine reality, incl. JAVA_TOOL_OPTIONS side effect);
- created/compiled/ran a Java program — exact output `ENR_AUTONOMOUS_TEST_OK`, exit 0 — reported real results.
- text-only `tiny-evo` + agent tools → ENR 400 "does not support tools" surfaced cleanly in OpenCode.

### B4. Terminal execution — PASS (unchanged; documented in baseline)

### B5. End-to-end Java scenario

| Scenario | Before | After |
| --- | --- | --- |
| direct (DeepSeek agent) | PASS | PASS |
| through OpenCode → ENR → DeepSeek | NOT TESTED | **PASS** (exact `ENR_AUTONOMOUS_TEST_OK`, exit 0) |

## 3. Build and test results (post-fix)

- `JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64 mvn package` — BUILD SUCCESS (13 modules)
- `mvn test` — all modules green; `enr-tests`: **105 tests, 0 failures** (88 baseline + 17 new regression tests)

## 4. Remaining limitations

- **L1 — OpenCode V1 discovery**: the installed OpenCode (1.19.6) is the V1 config generation; its Ollama
  integration uses the OpenAI-compatible `/v1` surface (`npm: @ai-sdk/openai-compatible`) and config-declared
  models — the native `/api/tags`+`/api/show` auto-discovery belongs to OpenCode V2 (verified in V2 source;
  ENR now satisfies its requirements: `capabilities:["completion"]`, `model_info.*.context_length`, `{"model":...}`).
  ENR's `/v1/chat/completions` (with the new tool/reasoning support) is exactly the surface V1 uses.
- **L2 — local native models are text-only**: no structured tool calls, `format:"json"` not enforced;
  tools requests are rejected with 400 (advertised as `["completion"]` only). This mirrors real Ollama behavior
  for models without the `tools` capability.
- **L3 — Anthropic/Gemini tool translation**: not implemented; tool requests on those provider types fail
  explicitly with 400 (no silent dropping).
- **L4 — streamed token counts for DeepSeek**: DeepSeek streaming does not always include usage
  (`stream_options.include_usage` not honored), so `prompt_eval_count`/`eval_count` can be 0 in streamed
  final chunks. Non-stream responses carry correct counts.
- **L5 — multi-tool-call aggregation** verified against the mock stream; live DeepSeek produced single tool
  calls in the tested scenarios.

## 5. Out-of-scope observations

- `enr-server/src/main/resources/web/*` (theme toggle) changed during the session by an external process
  (UPM design-standard sync); not part of this iteration and left untouched.
- Pre-existing uncommitted changes (web_server config support in EnrConfig/EnrConfigLoader/EnrManager/EnrServer
  + enr.yaml) were treated as baseline and were not authored by this iteration.

## 6. Files modified (this iteration)

| File | Change |
| --- | --- |
| `enr-manager/.../provider/ChatMessage.java` | optional payload for tool_calls/tool_call_id/reasoning |
| `enr-manager/.../provider/ProviderClient.java` | tools in ChatOptions; tool-aware OpenAI serialization; streaming tool-call aggregation; reasoning_content capture/echo; robust stream termination |
| `enr-manager/.../EnrManager.java` | ChatOptions.tools; ToolCallingUnsupportedException; validateTools; tool_calls/reasoning mapping; show/tags capabilities + model_info |
| `enr-manager/.../EnrConfigLoader.java` | tilde expansion + inline-comment stripping |
| `enr-server/.../OllamaApiHandlers.java` | model-field fallback; plain-string errors; 404 mapping; tool parsing/response conversion; thinking field |
| `enr-server/.../OpenAiApiHandlers.java` | tools/reasoning parse + response; validateTools pre-flight |
| `enr-server/.../ManagerApiHandlers.java` | tools in ChatOptions; validateTools pre-flight |
| `enr-tests/.../MockServers.java` | tool-calling mock OpenAI provider |
| `enr-tests/.../OllamaToolCallingTest.java` | NEW — 17 regression tests |
| `docs/testing/ENR_INITIAL_TEST_REPORT.md` | this report |
| `docs/testing/ENR_TESTING_GUIDE.md` | NEW — reproducible testing guide |
