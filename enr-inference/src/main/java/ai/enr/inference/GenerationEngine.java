package ai.enr.inference;

import ai.enr.inference.sample.Sampler;
import ai.enr.inference.sample.SamplingConfig;
import ai.enr.inference.session.InferenceSession;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Autoregressive generation engine: prefill + decode loop with KV cache,
 * streaming token listener, cancellation and stop conditions.
 */
public final class GenerationEngine {

    public enum TerminationReason {
        EOS_REACHED,
        MAX_TOKENS_REACHED,
        MAX_SEQ_LEN_REACHED,
        STOP_TOKEN,
        CANCELLED,
        ERROR
    }

    public record GenerationResult(
            List<Integer> generatedTokenIds,
            String text,
            int promptTokens,
            TerminationReason reason,
            long elapsedMs,
            float[] lastLogits) {
    }

    public record GenerationRequest(
            int[] promptTokenIds,
            int maxTokens,
            SamplingConfig sampling,
            Set<Integer> stopTokenIds) {
    }

    public interface TokenListener {
        void onToken(int tokenId, String tokenText);

        default void onComplete() {
        }
    }

    private final Transformer transformer;

    public GenerationEngine(Transformer transformer) {
        this.transformer = transformer;
    }

    public GenerationResult generate(InferenceSession session, GenerationRequest request,
                                     ai.enr.tokenizer.EvoBpeTokenizer tokenizer,
                                     TokenListener listener) {
        long startTime = System.currentTimeMillis();
        Sampler sampler = new Sampler(request.sampling());
        List<Integer> history = new ArrayList<>(session.contextTokens());
        List<Integer> generated = new ArrayList<>();
        var incrementalDecoder = tokenizer.newDecoder();
        float[] lastLogits = null;
        int promptTokens = 0;
        int eosTokenId = tokenizer.eosTokenId();

        try {
            int[] promptIds = request.promptTokenIds();
            if (promptIds.length == 0) {
                int bos = tokenizer.bosTokenId() >= 0 ? tokenizer.bosTokenId() : 1;
                promptIds = new int[]{bos};
            }
            if (session.contextLength() + promptIds.length > transformer.config().maxSeqLen()) {
                return finish(generated, incrementalDecoder, startTime, promptTokens,
                        TerminationReason.MAX_SEQ_LEN_REACHED, null);
            }
            promptTokens = promptIds.length;
            lastLogits = transformer.forward(promptIds, session.kvCache());
            for (int id : promptIds) {
                session.contextTokens().add(id);
            }
            history.addAll(session.contextTokens());

            for (int step = 0; step < request.maxTokens(); step++) {
                if (session.isCancelled()) {
                    return finish(generated, incrementalDecoder, startTime, promptTokens,
                            TerminationReason.CANCELLED, lastLogits);
                }
                if (session.contextExhausted()) {
                    return finish(generated, incrementalDecoder, startTime, promptTokens,
                            TerminationReason.MAX_SEQ_LEN_REACHED, lastLogits);
                }

                int nextToken = sampler.sample(lastLogits, history);

                lastLogits = transformer.forward(new int[]{nextToken}, session.kvCache());
                session.contextTokens().add(nextToken);
                history.add(nextToken);
                generated.add(nextToken);

                String piece = incrementalDecoder.accept(nextToken);
                if (listener != null) {
                    listener.onToken(nextToken, piece);
                }

                boolean isEos = (eosTokenId >= 0 && nextToken == eosTokenId);
                boolean isStop = request.stopTokenIds() != null && request.stopTokenIds().contains(nextToken);
                if (isEos || isStop) {
                    return finish(generated, incrementalDecoder, startTime, promptTokens,
                            TerminationReason.EOS_REACHED, lastLogits);
                }
            }
            return finish(generated, incrementalDecoder, startTime, promptTokens,
                    TerminationReason.MAX_TOKENS_REACHED, lastLogits);
        } catch (RuntimeException e) {
            return finish(generated, incrementalDecoder, startTime, promptTokens,
                    TerminationReason.ERROR, lastLogits);
        } finally {
            if (listener != null) {
                listener.onComplete();
            }
        }
    }

    private GenerationResult finish(List<Integer> generated,
                                    ai.enr.tokenizer.EvoBpeTokenizer.IncrementalDecoder decoder,
                                    long startTime, int promptTokens,
                                    TerminationReason reason, float[] lastLogits) {
        decoder.flush();
        return new GenerationResult(generated, decoder.text(), promptTokens, reason,
                System.currentTimeMillis() - startTime, lastLogits);
    }
}
