package ai.enr.inference.sample;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Sampling pipeline replicating the verified EVO reference semantics:
 * repeat/frequency/presence penalties, temperature scaling, softmax,
 * top-K, top-P (nucleus), categorical draw. Deterministic with a seed.
 */
public final class Sampler {

    private final SamplingConfig config;
    private final Random rng;

    public Sampler(SamplingConfig config) {
        this.config = config;
        this.rng = config.seed() != null ? new Random(config.seed()) : new Random();
    }

    public int sample(float[] rawLogits, List<Integer> tokenHistory) {
        int vocabSize = rawLogits.length;
        float[] logits = rawLogits.clone();

        float repeatPenalty = config.repeatPenalty();
        float freqPenalty = config.frequencyPenalty();
        float presPenalty = config.presencePenalty();

        if (!tokenHistory.isEmpty()) {
            int[] counts = new int[vocabSize];
            for (int token : tokenHistory) {
                if (token >= 0 && token < vocabSize) {
                    counts[token]++;
                }
            }
            for (int i = 0; i < vocabSize; i++) {
                if (counts[i] > 0) {
                    if (repeatPenalty > 1.0f) {
                        if (logits[i] < 0) {
                            logits[i] *= repeatPenalty;
                        } else {
                            logits[i] /= repeatPenalty;
                        }
                    }
                    logits[i] -= counts[i] * freqPenalty + presPenalty;
                }
            }
        }

        if (config.greedy()) {
            int argmax = 0;
            float maxVal = logits[0];
            for (int i = 1; i < vocabSize; i++) {
                if (logits[i] > maxVal) {
                    maxVal = logits[i];
                    argmax = i;
                }
            }
            return argmax;
        }

        float temperature = config.temperature();
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < vocabSize; i++) {
            logits[i] /= temperature;
            if (logits[i] > maxLogit) {
                maxLogit = logits[i];
            }
        }

        float[] probs = new float[vocabSize];
        float sumExp = 0f;
        for (int i = 0; i < vocabSize; i++) {
            probs[i] = (float) Math.exp(logits[i] - maxLogit);
            sumExp += probs[i];
        }
        for (int i = 0; i < vocabSize; i++) {
            probs[i] /= sumExp;
        }

        int topK = config.topK();
        if (topK > 0 && topK < vocabSize) {
            float[] sorted = probs.clone();
            Arrays.sort(sorted);
            float cutoff = sorted[vocabSize - topK];
            float newSum = 0f;
            for (int i = 0; i < vocabSize; i++) {
                if (probs[i] < cutoff) {
                    probs[i] = 0f;
                } else {
                    newSum += probs[i];
                }
            }
            if (newSum > 0f) {
                for (int i = 0; i < vocabSize; i++) {
                    probs[i] /= newSum;
                }
            }
        }

        float topP = config.topP();
        if (topP < 1.0f && topP > 0.0f) {
            Integer[] indices = new Integer[vocabSize];
            for (int i = 0; i < vocabSize; i++) {
                indices[i] = i;
            }
            Arrays.sort(indices, (a, b) -> Float.compare(probs[b], probs[a]));
            float cumulative = 0f;
            boolean cutoffReached = false;
            float newSum = 0f;
            for (int idx : indices) {
                if (cutoffReached) {
                    probs[idx] = 0f;
                } else {
                    cumulative += probs[idx];
                    newSum += probs[idx];
                    if (cumulative >= topP) {
                        cutoffReached = true;
                    }
                }
            }
            if (newSum > 0f) {
                for (int i = 0; i < vocabSize; i++) {
                    probs[i] /= newSum;
                }
            }
        }

        float r = rng.nextFloat();
        float accum = 0f;
        for (int i = 0; i < vocabSize; i++) {
            accum += probs[i];
            if (r <= accum) {
                return i;
            }
        }
        return vocabSize - 1;
    }

    public static List<Integer> copyHistory(List<Integer> history) {
        return new ArrayList<>(history);
    }
}
