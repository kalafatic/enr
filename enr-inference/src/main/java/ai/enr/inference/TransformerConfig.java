package ai.enr.inference;

import ai.enr.loader.api.Architecture;

public record TransformerConfig(
        int vocabSize,
        int dModel,
        int numHeads,
        int numKvHeads,
        int numBlocks,
        int dff,
        int maxSeqLen,
        float normEps,
        String positionalEncoding,
        float ropeTheta,
        int ropeDim,
        boolean embeddingTied) {

    public static TransformerConfig from(Architecture arch) {
        return new TransformerConfig(
                arch.vocabSize(),
                arch.dModel(),
                arch.numHeads(),
                arch.numKvHeads(),
                arch.numBlocks(),
                arch.dff(),
                arch.maxSeqLen(),
                arch.normEps(),
                arch.positionalEncoding(),
                arch.ropeTheta(),
                arch.ropeDim(),
                arch.embeddingTied());
    }

    public int headDim() {
        return dModel / numHeads;
    }

    public boolean usesRope() {
        return "rope".equalsIgnoreCase(positionalEncoding);
    }
}
