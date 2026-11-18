# ENR Java API

Date: 2026-11-16
Status: implemented and tested against the live server

`ai.enr.client.EnrClient` (artifact `enr-java-api`) — a standalone Java 21 client
depending only on `enr-core`. No Eclipse RCP/OSGi; an OSGi adapter may be added
later without touching the core runtime.

## Construction

```java
EnrClient client = new EnrClient("http://127.0.0.1:8111");
```

## Synchronous

```java
boolean healthy = client.health();

var status = client.status();                 // JsonObject

List<ModelInfo> models = client.models();     // id, path, state, arch dims, backend...
ModelInfo model = client.loadModel("/path/model.gguf", "cuda");
ModelInfo info = client.model("model");
client.unloadModel("model", false);

InferenceResult r = client.infer("model", "hello world");         // defaults
InferenceResult r2 = client.infer("model", "hi",
        new EnrClient.InferenceRequest()
                .maxTokens(128)
                .temperature(0.7f)
                .topP(0.9f)
                .seed(42)
                .session("session-uuid"));

List<JsonObject> sessions = client.sessions();
JsonObject metrics = client.metrics();
List<Device> devices = client.devices();
```

`InferenceResult`: `text`, `tokens`, `promptTokens`, `generatedTokens`, `reason`,
`elapsedMs`.

## Asynchronous

```java
CompletableFuture<InferenceResult> future = client.inferAsync("model", "prompt", req);
```

## Streaming (SSE)

```java
client.inferStream("model", "prompt", new EnrClient.InferenceRequest().maxTokens(64),
        token -> System.out.print(token.text()),       // per token
        result -> System.out.println(result.reason())); // completion
```

## Cancellation

```java
client.cancel(requestId, sessionId); // either id
```

## Errors

`EnrClient.EnrException(status, code, message)` for HTTP and transport errors.

## Integration notes

- The client speaks only HTTP; it never imports runtime internals
- `HttpClient` instances can be injected (`new EnrClient(url, http)`) for
  connection pooling/custom TLS in consuming applications
