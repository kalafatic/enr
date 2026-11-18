package ai.enr.tests;

import ai.enr.compute.ComputeBackend;
import ai.enr.compute.cpu.CpuBackend;
import ai.enr.compute.cuda.CudaBackend;
import ai.enr.inference.KVCache;
import ai.enr.inference.ModelWeights;
import ai.enr.inference.Transformer;
import ai.enr.inference.TransformerConfig;
import ai.enr.loader.ModelSources;
import ai.enr.loader.api.LoadedModel;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

/**
 * Performance measurements from actual executions (no invented numbers).
 * Results are printed and recorded in ENR-PERFORMANCE-REPORT.md.
 */
class PerformanceBenchmark {

    private static final Path TINY = Path.of("../enr-model-loader/src/test/resources/models/tiny-evo.gguf");
    private static final Path EVO = Path.of("../enr-model-loader/src/test/resources/models/evo.gguf");

    @Test
    void measureCpuAndCuda() throws Exception {
        for (Path modelPath : new Path[]{TINY, EVO}) {
            System.out.println("=== Model: " + modelPath.getFileName() + " ===");
            long loadStart = System.nanoTime();
            LoadedModel model = ModelSources.open(modelPath);
            long loadMs = (System.nanoTime() - loadStart) / 1_000_000;
            System.out.println("load_ms=" + loadMs + " params=" + model.totalParameters()
                    + " weight_bytes=" + model.weightBytes());

            TransformerConfig config = TransformerConfig.from(model.architecture());
            measure(model, config, new CpuBackend(), "cpu");
            CudaBackend cuda = new CudaBackend();
            if (cuda.deviceInfo().available()) {
                measure(model, config, cuda, "cuda");
            } else {
                System.out.println("cuda: unavailable");
            }
        }
    }

    @Test
    void measureConcurrentThroughput() throws Exception {
        ai.enr.runtime.EnrRuntime runtime = new ai.enr.runtime.EnrRuntime(
                ai.enr.runtime.RuntimeConfig.builder()
                        .addModelDirectory(TINY.toAbsolutePath().getParent())
                        .maxConcurrentSessions(32)
                        .defaultMaxTokens(16)
                        .build());
        runtime.loadModel(TINY.toAbsolutePath());

        int total = 64;
        int concurrency = 8;
        var pool = java.util.concurrent.Executors.newFixedThreadPool(concurrency);
        var options = ai.enr.runtime.EnrRuntime.InferenceOptions.defaults();
        long start = System.nanoTime();
        var futures = new java.util.ArrayList<java.util.concurrent.Future<Long>>();
        for (int i = 0; i < total; i++) {
            futures.add(pool.submit(() -> {
                long t0 = System.nanoTime();
                runtime.infer("tiny-evo", "token_4 token_5 token_6", options, null);
                return System.nanoTime() - t0;
            }));
        }
        long totalLatency = 0;
        for (var f : futures) {
            totalLatency += f.get();
        }
        pool.shutdown();
        long elapsedMs = (System.nanoTime() - start) / 1_000_000;
        System.out.println("concurrent_requests=" + total + " concurrency=" + concurrency
                + " wall_ms=" + elapsedMs
                + " throughput_requests_per_sec=" + String.format("%.1f", total * 1000.0 / elapsedMs)
                + " avg_request_latency_ms=" + String.format("%.1f", totalLatency / 1e6 / total));
        runtime.shutdown();
    }

    private void measure(LoadedModel model, TransformerConfig config, ComputeBackend backend, String label) {
        Transformer transformer = new Transformer(config, new ModelWeights(model), backend);
        int maxSeqLen = config.maxSeqLen();
        int promptLen = Math.min(5, maxSeqLen - 1);
        int[] prompt = new int[promptLen];
        for (int i = 0; i < promptLen; i++) {
            prompt[i] = 4 + i % 3;
        }

        long t0 = System.nanoTime();
        KVCache kv = new KVCache(config.numBlocks(), config.numKvHeads(), config.headDim(), maxSeqLen);
        float[] logits = transformer.forward(prompt, kv);
        long ttftMs = (System.nanoTime() - t0) / 1_000_000;
        System.out.println(label + "_ttft_ms=" + ttftMs + " (prefill " + prompt.length + " tokens)");

        int steps = Math.min(20, maxSeqLen - prompt.length - 1);
        long decodeStart = System.nanoTime();
        float[] last = logits;
        for (int i = 0; i < steps; i++) {
            int token = argmax(last);
            last = transformer.forward(new int[]{token}, kv);
        }
        long decodeMs = (System.nanoTime() - decodeStart) / 1_000_000;
        double tokPerSec = decodeMs > 0 ? steps * 1000.0 / decodeMs : Double.POSITIVE_INFINITY;
        System.out.println(label + "_decode_tokens=" + steps + " decode_ms=" + decodeMs
                + " tokens_per_sec=" + String.format("%.1f", tokPerSec));
        System.out.println(label + "_vram_used=" + (backend instanceof CudaBackend c ? c.allocatedVramBytes() : 0));
    }

    private static int argmax(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }
        return best;
    }
}
