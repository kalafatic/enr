# ENR Testing Guide

How to start ENR, integrate it with OpenCode, and reproduce the functional test
campaign documented in `ENR_INITIAL_TEST_REPORT.md`. All instructions are based
on the verified environment: ENR 2.6.11-SNAPSHOT at
`/home/petr/git/evo-native-runtime-161126`, endpoint `http://127.0.0.1:8111`,
OpenCode 1.19.6 (V1 config generation), real Ollama 0.40.16 at
`http://127.0.0.1:11434` (reference only).

## 1. Build and start ENR

```bash
cd /home/petr/git/evo-native-runtime-161126
export JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64   # JDK 21 also works (CPU)
mvn package                                          # runs all tests

# Detached start (survives the launching shell):
setsid -f ./enr.sh serve -c enr.yaml > /tmp/opencode/enr-server.log 2>&1 < /dev/null

# Health check
curl -s http://127.0.0.1:8111/api/version    # {"version":"2.6.11","name":"ENR"}

# Stop
/usr/bin/kill -TERM "$(pgrep -f 'enr-cli.*jar.*serve' | head -1)"
```

Storage resolves under `~/.enr` (`storage.*` paths in enr.yaml support `~/` and
inline `# comments`). Local models come from
`runtime.model_directories` (`enr-model-loader/src/test/resources/models`:
`tiny-evo`, `evo`).

## 2. Run the automated tests

```bash
cd /home/petr/git/evo-native-runtime-161126
JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64 mvn test
# enr-tests currently: 105 tests, 0 failures
JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64 mvn test -pl enr-tests -am
```

Key suites: `OllamaCompatibilityTest` (protocol baseline),
`OllamaToolCallingTest` (regression tests for show/capabilities/errors/tools),
`OpenAiCompatibilityTest`, `ApiIntegrationTest`, `ManagerTest`,
`FamilyApiIntegrationTest`, `ModelManagementHttpTest`.

## 3. Ollama protocol tests

Reference contract: https://docs.ollama.com/api

```bash
ENR=http://127.0.0.1:8111

# Discovery + info
curl -s $ENR/api/version
curl -s $ENR/api/tags          # models with capabilities
curl -s $ENR/api/ps
curl -s -X POST $ENR/api/show -d '{"model":"tiny-evo"}'   # note: "model" field (OpenCode format)
curl -s -X POST $ENR/api/show -d '{"name":"tiny-evo"}'    # "name" also works

# Chat, non-streaming and streaming (NDJSON)
curl -s -X POST $ENR/api/chat -d '{"model":"tiny-evo","messages":[{"role":"user","content":"hello"}],"stream":false}'
curl -s -N -X POST $ENR/api/chat -d '{"model":"tiny-evo","stream":true,"messages":[{"role":"user","content":"hello"}]}'

# Generate
curl -s -X POST $ENR/api/generate -d '{"model":"tiny-evo","prompt":"hello"}'

# Error handling (Ollama contract: plain-string error, 404 unknown model, 400 missing field)
curl -s -w '\n%{http_code}\n' -X POST $ENR/api/chat -d '{"model":"nope","messages":[{"role":"user","content":"hi"}]}'
```

Expected: `/api/show` returns `capabilities:["completion"]` and
`model_info.general.context_length`; unknown model → 404
`{"error":"Model 'nope' is not installed ..."}`.

## 4. Tool calling tests

### 4.1 Structured tool calls via a cloud provider (Ollama wire format)

Requires a DeepSeek provider in ENR (`~/.enr/state.json`):

```bash
# One-time provider setup (uses the key file from the OpenCode config)
curl -s -X POST $ENR/api/v1/providers -H 'Content-Type: application/json' \
  -d '{"id":"deepseek","type":"deepseek","name":"DeepSeek",
       "base_url":"https://api.deepseek.com/v1",
       "api_key":"'$(cat /home/petr/.config/deepseek-api-key)'","enabled":true}'
curl -s -X POST $ENR/api/v1/providers/deepseek/test
```

```bash
TOOLS='[{"type":"function","function":{"name":"get_weather",
  "description":"Get the current weather for a city",
  "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'

# 1st turn — model requests the tool (non-streaming)
curl -s -X POST $ENR/api/chat -d '{"model":"deepseek:deepseek-flash",
  "messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],
  "tools":'$TOOLS',"stream":false}'
# expect: message.tool_calls [{function:{name:"get_weather",arguments:{city:"Paris"}}}], done_reason:"tool_calls"

# 2nd turn — feed the tool result back (note the echoed "thinking" field)
curl -s -X POST $ENR/api/chat -d '{"model":"deepseek:deepseek-flash",
  "messages":[
    {"role":"user","content":"What is the weather in Paris?"},
    {"role":"assistant","content":"I will check.","thinking":"",
     "tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"Paris"}}}]},
    {"role":"tool","content":"The current temperature in Paris is 22 degrees, sunny."}],
  "tools":'$TOOLS',"stream":false}'
# expect: final natural-language answer, done_reason:"stop"

# Streaming variant — final NDJSON chunk carries tool_calls
curl -s -N -X POST $ENR/api/chat -d '{"model":"deepseek:deepseek-flash","stream":true,
  "messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],
  "tools":'$TOOLS'}'
```

### 4.2 OpenAI wire format (`/v1/chat/completions`)

```bash
curl -s -X POST $ENR/v1/chat/completions -d '{"model":"deepseek:deepseek-flash",
  "messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],
  "tools":[{"type":"function","function":{"name":"get_weather",
  "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],"stream":false}'
# expect: message.tool_calls with id/type/function.arguments (JSON string),
# reasoning_content echoed, finish_reason "tool_calls"
```

### 4.3 Unsupported routes fail loudly (never silent)

```bash
# local text-only model + tools → 400, mirroring real Ollama
curl -s -w '\n%{http_code}\n' -X POST $ENR/api/chat -d '{"model":"tiny-evo",
  "messages":[{"role":"user","content":"weather?"}],"tools":'$TOOLS'}'
# Anthropic/Gemini provider routes with tools → 400
```

## 5. OpenCode integration (OpenCode 1.19.6, V1 config)

OpenCode V1 integrates with Ollama through the OpenAI-compatible `/v1` surface
(`npm: @ai-sdk/openai-compatible`), per the official V1 docs
(https://opencode.ai/v1/docs/providers). Point the `ollama` provider at ENR via
a project-scoped config:

```jsonc
// /tmp/opencode/enr-opencode-test/opencode.jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "ENR (Ollama-compatible)",
      "options": { "baseURL": "http://127.0.0.1:8111/v1" },
      "models": {
        "tiny-evo": { "name": "ENR tiny-evo (test)", "tool_call": false },
        "ds-flash": {
          "id": "deepseek:deepseek-flash",
          "name": "DeepSeek via ENR (tool calling)",
          "tool_call": true,
          "limit": { "context": 65536, "output": 8192 }
        }
      }
    }
  }
}
```

Notes:
- The model key (`ds-flash`) is the OpenCode model ID; `id` is the model name
  sent to ENR, which routes `deepseek:deepseek-flash` to the configured provider.
- Text-only models must set `"tool_call": false`; the `build` agent always sends
  tools otherwise, and ENR rejects them explicitly (400), exactly like real
  Ollama does for models without the `tools` capability.
- For OpenCode **V2** (not installed here), the built-in `ollama` provider
  auto-discovers models via native `/api/tags` + `/api/show` — ENR satisfies its
  requirements (`capabilities:["completion"]`, `model_info.*.context_length`,
  `{"model":...}` in `/api/show`).

### 5.1 Model selection

```bash
cd /tmp/opencode/enr-opencode-test
opencode models                 # lists config-declared models (provider/model IDs)
opencode run -m ollama/tiny-evo "reply with exactly: ENR_OPENTEST_OK"
```

### 5.2 Terminal execution verification (tool calling end-to-end)

```bash
cd /tmp/opencode/enr-opencode-test
opencode run -m ollama/ds-flash "Use your shell tool to run these commands one by
one and report the ACTUAL output of each (do not guess):
pwd; whoami; uname -a; java -version; mvn -version; git --version"
```

The chain exercised: OpenCode build agent → ENR `/v1/chat/completions` →
DeepSeek → structured tool_calls → ENR → OpenCode shell tool → real execution →
results → DeepSeek → final answer. Verify the reported outputs match the real
machine values.

## 6. Java end-to-end scenario

```bash
mkdir -p /tmp/opencode/enr-opencode-test/java-e2e
cd /tmp/opencode/enr-opencode-test
opencode run -m ollama/ds-flash "In /tmp/opencode/enr-opencode-test/java-e2e:
1) inspect the java environment (ls /usr/lib/jvm), 2) create a minimal Java program
EnrAutonomousTest.java whose main prints EXACTLY 'ENR_AUTONOMOUS_TEST_OK',
3) compile it with /usr/lib/jvm/java-21-openjdk-amd64/bin/javac,
4) run it with the matching java binary, 5) report the exact program output and exit code"
```

Expected: output `ENR_AUTONOMOUS_TEST_OK`, compile/run exit 0, and the agent
must report the actual values — never invented results.

## 7. Interpreting results

- A test is PASS only with execution evidence (response bodies, exit codes,
  actual file contents).
- Ollama-protocol errors are plain strings: `{"error":"..."}` with 400 (bad
  request / unsupported tools), 404 (unknown model), 405 (wrong method).
- OpenAI-surface errors keep the structured
  `{"error":{"code","message"}}` shape.
- Local native models are text-only: capabilities `["completion"]` only; tool
  requests fail with 400 — expected behavior, not a bug.
- `finish_reason` values: `stop`, `tool_calls`, `error`, `cancelled`; Ollama
  streaming surfaces them as `done_reason` in the final NDJSON chunk;
  OpenAI streaming as `choices[0].finish_reason` followed by `data: [DONE]`.

## 8. Cleanup after testing

- Stop ENR (`kill -TERM`, see Section 1) and any test OpenCode servers.
- Remove the DeepSeek provider if it was added only for testing
  (`DELETE /api/v1/providers/deepseek` or delete `~/.enr/state.json` if it did
  not exist before).
- Test workspaces live under `/tmp/opencode/` and are disposable.
