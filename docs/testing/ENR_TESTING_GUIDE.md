# ENR Testing Guide

How to start ENR, integrate it with OpenCode, and reproduce the functional test campaign documented in `ENR_INITIAL_TEST_REPORT.md`. All instructions are based on the verified environment:

- ENR 2.6.11-SNAPSHOT at `/home/petr/git/enr-221126`, endpoint `http://127.0.0.1:8111`.
- JDK 25 (`/usr/lib/jvm/java-25-openjdk-amd64`) preferred for the CUDA backend; JDK 21 also works (CPU).
- OpenCode **v2.4.0** (`/home/petr/.opencode/bin/opencode`).
- Real Ollama 0.40.16 at `http://127.0.0.1:11434` (reference only).

## 1. Build, start and stop ENR

```bash
cd /home/petr/git/enr-221126
export JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64
mvn clean package                                     # builds 13 modules, runs all tests

# Detached start (survives the launching shell):
setsid -f ./enr.sh serve -c enr.yaml > /tmp/opencode/enr-campaign/enr-server.log 2>&1 < /dev/null

# Stop (graceful: SIGTERM unloads models):
# match only the java process, never the launcher shell:
kill -TERM "$(ps -eo pid,args | grep '[j]ava' | grep 'enr-cli.*serve' | awk '{print $1}')"
```

Storage resolves under `~/.enr`. Local models come from `runtime.model_directories`
(`enr-model-loader/src/test/resources/models`: `tiny-evo`, `evo`).

## 2. Verify runtime readiness

```bash
curl -s http://127.0.0.1:8111/api/version   # {"version":"2.6.11","name":"ENR"}
curl -s http://127.0.0.1:8111/api/v1/status # {"running":true,...,"models_loaded":2}
curl -s http://127.0.0.1:8111/api/v1/health # {"status":"healthy","checks":[]}
```

A startup message alone is not sufficient — confirm `/api/v1/status` reports `running: true`
and the expected `models_loaded` count.

## 3. Discover available models

```bash
./enr.sh list                       # installed models with state/backend/size
./enr.sh ps                         # loaded models (name/parameters/size)
./enr.sh show tiny-evo              # model info: format/parameters/context/caps/model_info
./enr.sh version                    # runtime version
curl -s http://127.0.0.1:8111/api/tags | python3 -m json.tool   # name/model/digest/details/capabilities
curl -s http://127.0.0.1:8111/api/ps    | python3 -m json.tool   # loaded models + parameter_size
curl -s -X POST http://127.0.0.1:8111/api/show -d '{"model":"tiny-evo"}'  # note: "model" (OpenCode format)
```

`/api/show` must return `capabilities: ["completion"]` and `model_info.general.context_length`
(this is what OpenCode V2's native Ollama discovery requires).

CLI quick reference (all delegate to the same EnrManager services as the HTTP API):
`enr serve|start|stop|status`, `enr pull`, `enr list`, `enr ps`, `enr show <model>`,
`enr run <model> [prompt]` (alias of interactive `chat`), `enr load`, `enr unload`,
`enr remove` / `enr rm`, `enr infer`, `enr chat`, `enr provider ...`, `enr service ...`,
`enr version`.

## 4. Pull Hugging Face GGUF models

```bash
# Quantization auto-discovery: ENR selects the single GGUF matching the tag
./enr.sh pull hf.co/unsloth/Inkling-2-Mini-GGUF:UD-Q4_K_M
./enr.sh pull hf.co/bartowski/SmolLM2-135M-Instruct-GGUF:Q4_K_M

# Explicit file (subdirectories allowed)
./enr.sh pull hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf

# Ollama Library
./enr.sh pull llama3.2:1b
```

Quantization resolution lists the repository's GGUF files via the Hugging Face tree API and matches
the tag against the filename; an absent or ambiguous tag produces a clear error. The `lfs.oid` is
used as the SHA-256 verification digest. Gated/private repos use `catalogs.huggingface_token` in
`enr.yaml` or the `HF_TOKEN` environment variable (never logged or returned by the API). Downloads
resume via HTTP Range from a `.part` file after interruption.

`enr run <model>` auto-pulls when the model is not yet installed, then loads it and starts
interactive inference; it reports a clear error (not success) if loading fails.

## 5. Configure an ENR-native model

ENR-native models are GGUF/`.evo` files. Add their directory to `enr.yaml`:

Restart ENR; models are auto-loaded. To load/unload manually:

```bash
./enr.sh load  /home/petr/git/enr-221126/enr-model-loader/src/test/resources/models/tiny-evo.gguf --backend cpu
./enr.sh unload tiny-evo        # errors (non-zero) if not loaded
```

## 6. Execute native inference tests

```bash
# Native /api/v1 inference (raw prompt, requires the model to be loaded):
./enr.sh infer tiny-evo "hello"                 # -> token_4 (deterministic)
./enr.sh infer tiny-evo "hello" --max-tokens 1  # -> MAX_TOKENS_REACHED
curl -s -X POST http://127.0.0.1:8111/api/v1/inference -d '{"model":"tiny-evo","prompt":"hello"}'
# -> {"text":"token_4","tokens":[4,2],"prompt_tokens":5,"generated_tokens":2,"reason":"EOS_REACHED",...}

# Streaming (SSE):
curl -s -N -X POST http://127.0.0.1:8111/api/v1/inference/stream -d '{"model":"tiny-evo","prompt":"hello"}'
```

The bundled models are synthetic fixtures with a literal `token_N` vocabulary; output is
deterministic but not meaningful text. Verify: EOS token (id 2) is excluded from decoded text,
generation terminates, and repeats are byte-identical.

## 7. Configure external providers

```bash
./enr.sh provider add --type deepseek --api-key "$(cat /home/petr/.config/deepseek-api-key)" \
        --id deepseek --base-url https://api.deepseek.com/v1
./enr.sh provider test deepseek        # OK — 2 model(s) discovered (deepseek-flash, deepseek-v4-pro)
./enr.sh provider list                 # api_key_configured:true (key never exposed)
./enr.sh provider remove deepseek      # cleanup after testing
```

Route to a provider with a `provider:model` prefix, e.g. `deepseek:deepseek-flash`. Local model
IDs win; ambiguous names fail loudly.

## 8. Configure OpenCode with ENR (Ollama-compatible provider)

OpenCode V2's built-in `ollama` provider auto-discovers via native `/api/tags` + `/api/show`.
Point it at ENR with a project-scoped config (the `baseURL` includes `/v1`; OpenCode derives the
native `/api/*` discovery paths from it):

```jsonc
// /tmp/opencode/enr-opencode-test/opencode.jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "providers": {
    "ollama": {
      "settings": { "baseURL": "http://127.0.0.1:8111/v1" },
      "models": {
        "tiny-evo": {
          "name": "ENR tiny-evo (native)",
          "capabilities": { "tools": false, "input": ["text"], "output": ["text"] },
          "limit": { "context": 32, "output": 16 }
        }
      }
    }
  }
}
```

```bash
cd /tmp/opencode/enr-opencode-test
opencode models                      # should list ollama/evo and ollama/tiny-evo (from ENR)
opencode run -m ollama/tiny-evo "say hello"
```

Verified behaviour: ENR models are discovered, and a request routed through ENR reaches ENR's
native pipeline. Because the native models are text-only, OpenCode's agentic `build` agent sends
tools and ENR rejects them honestly (`400 "does not support tools"`) — exactly like real Ollama
for a model without the `tools` capability. A successful agentic multi-turn through the bundled
native models is therefore not possible (documented limitation, not an ENR defect).

### 7.1 Full agentic integration through a provider (verified end-to-end)

For real agentic tool use, route a provider-backed model through ENR. This was verified live:
OpenCode → ENR → DeepSeek, full lifecycle (discover → converse → tool call → execute → result →
continue → final answer), all traffic through ENR.

```jsonc
// /tmp/opencode/enr-agent-test/opencode.jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "providers": {
    "enr": {
      "name": "ENR Gateway",
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": { "baseURL": "http://127.0.0.1:8111/v1" },
      "models": {
        "ds-flash": {
          "name": "DeepSeek Flash via ENR",
          "modelID": "deepseek:deepseek-flash",   // routed by ENR to the DeepSeek provider
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"] },
          "limit": { "context": 65536, "output": 8192 }
        }
      }
    }
  }
}
```

```bash
cd /tmp/opencode/enr-agent-test
opencode models                              # -> enr/ds-flash
opencode run --auto -m enr/ds-flash "..."    # agentic coding task
```

Verify the full lifecycle passed through ENR by checking the request log:

```bash
curl -s http://127.0.0.1:8111/api/v1/logs | python3 -c \
  'import sys,json; [print(e["message"]) for e in json.load(sys.stdin)["logs"] if e["logger"]=="chat"]'
# expect: request model=deepseek:deepseek-flash target=provider provider=deepseek messages=N tools=M stream=true
# with messages growing across turns (1,2,4,6,8,10...)
```

The harness must not connect directly to the provider: here the `enr` provider only knows
`127.0.0.1:8111`, and the model id `deepseek:deepseek-flash` is resolved by ENR's router.

## 9. Configure OpenCode with cloud DeepSeek

Already configured globally in `~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "providers": {
    "deepseek": {
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": {
        "baseURL": "https://api.deepseek.com",
        "apiKey": "{file:/home/petr/.config/deepseek-api-key}"
      },
      "models": { "deepseek-v4-pro": { "name": "DeepSeek V4 Pro (1M)", /* capabilities + limit */ } }
    }
  }
}
```

Use with `opencode run -m deepseek/deepseek-v4-pro "..."`. This is the primary engineering agent
for this campaign (confirmed cloud, not local).

## 10. Execute native API tests

```bash
B=http://127.0.0.1:8111
curl -s $B/api/v1/health
curl -s $B/api/v1/models            # loaded models: id/architecture/vocab_size/max_seq_len/backend
curl -s $B/api/v1/devices          # CPU + CUDA devices
curl -s $B/api/v1/metrics          # total_requests, generated_tokens, failed_requests
curl -s -X POST $B/api/v1/models/unload -d '{"id":"tiny-evo","force":false}'   # 404 if not loaded
```

## 11. Execute Ollama protocol tests

Reference contract: https://docs.ollama.com/api

```bash
B=http://127.0.0.1:8111
curl -s $B/api/version
curl -s $B/api/tags
curl -s $B/api/ps
curl -s -X POST $B/api/show -d '{"model":"tiny-evo"}'      # "model" field (OpenCode format)
curl -s -X POST $B/api/show -d '{"name":"tiny-evo"}'       # "name" also works
curl -s -X POST $B/api/chat -d '{"model":"tiny-evo","messages":[{"role":"user","content":"hello"}],"stream":false}'
curl -s -N -X POST $B/api/chat -d '{"model":"tiny-evo","stream":true,"messages":[{"role":"user","content":"hello"}]}'
curl -s -X POST $B/api/generate -d '{"model":"tiny-evo","prompt":"hello"}'

# Error handling (Ollama contract: plain-string error; 400 malformed JSON / missing field; 404 unknown model)
curl -s -w '\n%{http_code}\n' -X POST $B/api/chat -d 'not json'                    # 400
curl -s -w '\n%{http_code}\n' -X POST $B/api/chat -d '{"model":"nope","messages":[]}' # 404
```

## 12. Test structured tool calling

Requires a provider (Section 6). Example (Ollama wire format, non-streaming):

```bash
B=http://127.0.0.1:8111
TOOLS='[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'

curl -s -X POST $B/api/chat -d "{\"model\":\"deepseek:deepseek-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"Weather in Paris? Use the tool.\"}],\"tools\":$TOOLS,\"stream\":false}"
# expect: message.tool_calls [{function:{name:"get_weather",arguments:{city:"Paris"}}}], done_reason:"tool_calls"
```

Multi-turn round-trip: feed the tool result back as a `role:"tool"` message (with the assistant
message's `thinking` field echoed) to get the final natural-language answer. OpenAI format
(`/v1/chat/completions`) returns `tool_calls` with `id/type/function.arguments` (JSON string) and
`finish_reason:"tool_calls"`. Native text-only models reject tools with 400 — expected, not a bug.

## 13. Verify actual terminal execution

Use OpenCode's own shell/terminal tool (no separate framework). Verify the agent reports the
actual outputs — never invented results:

```bash
pwd; whoami; uname -a; java -version; mvn -version; git --version
mkdir -p /tmp/opencode/enr-campaign/java-e2e && cd /tmp/opencode/enr-campaign/java-e2e
```

Check that a failed command is reported as a failure (exit code + stderr), and that the agent uses
the returned output in subsequent reasoning.

## 14. Reproduce the Java end-to-end scenario

```bash
mkdir -p /tmp/opencode/enr-campaign/java-e2e && cd /tmp/opencode/enr-campaign/java-e2e
cat > EnrAutonomousTest.java <<'EOF'
public class EnrAutonomousTest {
    public static void main(String[] args) { System.out.println("ENR_AUTONOMOUS_TEST_OK"); }
}
EOF
/usr/lib/jvm/java-21-openjdk-amd64/bin/javac EnrAutonomousTest.java
/usr/lib/jvm/java-21-openjdk-amd64/bin/java EnrAutonomousTest
# expected: ENR_AUTONOMOUS_TEST_OK, exit 0
```

## 15. Interpret results

- A test is PASS only with execution evidence (response bodies, exit codes, actual file contents).
- Ollama-protocol errors are plain strings: `{"error":"..."}` with 400 (bad request / unsupported
  tools), 404 (unknown model), 405 (wrong method). OpenAI-surface errors keep the structured
  `{"error":{"code","message"}}` shape.
- Native models are text-only (`capabilities:["completion"]`); tool requests fail with 400 —
  expected. `finish_reason` values: `stop`, `tool_calls`, `error`, `cancelled`; Ollama streaming
  surfaces them as `done_reason`.
- `max_concurrent_sessions` (8 by default) rejects excess concurrent requests; currently reported
  with HTTP 500 (see report limitation L3).

## 16. Diagnose common failures

| Symptom | Likely cause / check |
| --- | --- |
| `Model not loaded: <id>` from `enr infer` | native `/api/v1/inference` requires an explicit load; use `enr load` first (or use `/api/chat` which auto-loads) |
| 404 `Model '<id>' is not installed...` | model not in `model_directories` and not exposed by any provider; check `enr.yaml` path |
| `does not support tools` (400) | text-only native model or Anthropic/Gemini route; expected — no structured tool calls |
| `Concurrent session limit reached: 8` | exceeded `max_concurrent_sessions`; retry or raise the limit |
| OpenCode "Ollama has no models" | confirm `/api/tags` and `/api/show` are reachable at the derived path and `/api/show` reports the `completion` capability |
| Malformed JSON → 404 | should not happen after this iteration's fix; if it does, the fix is regressed (expect 400) |

## Cleanup after testing

- Remove any provider added only for testing (`./enr.sh provider remove <id>`).
- Stop ENR (Section 1). Test workspaces live under `/tmp/opencode/` and are disposable.
- Audit runs are persisted append-only under `~/.evo/audit/`; never overwrite them.
