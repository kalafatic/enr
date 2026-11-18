package ai.enr.inference.sample;

public record SamplingConfig(
        float temperature,
        float topP,
        int topK,
        float repeatPenalty,
        float frequencyPenalty,
        float presencePenalty,
        Long seed) {

    public static SamplingConfig defaults() {
        return new SamplingConfig(0.0f, 0.9f, 40, 1.1f, 0.0f, 0.0f, null);
    }

    public SamplingConfig withTemperature(float temperature) {
        return new SamplingConfig(temperature, topP, topK, repeatPenalty, frequencyPenalty, presencePenalty, seed);
    }

    public SamplingConfig withSeed(long seed) {
        return new SamplingConfig(temperature, topP, topK, repeatPenalty, frequencyPenalty, presencePenalty, seed);
    }

    public SamplingConfig withMaxTokens(int ignored) {
        return this;
    }

    public boolean greedy() {
        return temperature <= 0.0f;
    }
}
