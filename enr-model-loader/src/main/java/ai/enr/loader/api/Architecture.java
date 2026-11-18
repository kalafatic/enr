package ai.enr.loader.api;

import java.util.LinkedHashMap;
import java.util.Map;

public final class Architecture {

    private String family = "evo_llm";
    private int vocabSize;
    private int dModel;
    private int numHeads;
    private int numKvHeads;
    private int numBlocks;
    private int dff;
    private int maxSeqLen;
    private String normType = "rms_norm";
    private float normEps = 1e-5f;
    private String activationType = "swiglu";
    private String positionalEncoding = "sinusoidal";
    private float ropeTheta = 10000.0f;
    private int ropeDim;
    private boolean embeddingTied;

    public String family() {
        return family;
    }

    public Architecture family(String family) {
        this.family = family;
        return this;
    }

    public int vocabSize() {
        return vocabSize;
    }

    public Architecture vocabSize(int vocabSize) {
        this.vocabSize = vocabSize;
        return this;
    }

    public int dModel() {
        return dModel;
    }

    public Architecture dModel(int dModel) {
        this.dModel = dModel;
        return this;
    }

    public int numHeads() {
        return numHeads;
    }

    public Architecture numHeads(int numHeads) {
        this.numHeads = numHeads;
        return this;
    }

    public int numKvHeads() {
        return numKvHeads > 0 ? numKvHeads : numHeads;
    }

    public Architecture numKvHeads(int numKvHeads) {
        this.numKvHeads = numKvHeads;
        return this;
    }

    public int numBlocks() {
        return numBlocks;
    }

    public Architecture numBlocks(int numBlocks) {
        this.numBlocks = numBlocks;
        return this;
    }

    public int dff() {
        return dff;
    }

    public Architecture dff(int dff) {
        this.dff = dff;
        return this;
    }

    public int maxSeqLen() {
        return maxSeqLen;
    }

    public Architecture maxSeqLen(int maxSeqLen) {
        this.maxSeqLen = maxSeqLen;
        return this;
    }

    public String normType() {
        return normType;
    }

    public Architecture normType(String normType) {
        this.normType = normType;
        return this;
    }

    public float normEps() {
        return normEps;
    }

    public Architecture normEps(float normEps) {
        this.normEps = normEps;
        return this;
    }

    public String activationType() {
        return activationType;
    }

    public Architecture activationType(String activationType) {
        this.activationType = activationType;
        return this;
    }

    public String positionalEncoding() {
        return positionalEncoding;
    }

    public Architecture positionalEncoding(String positionalEncoding) {
        this.positionalEncoding = positionalEncoding;
        return this;
    }

    public float ropeTheta() {
        return ropeTheta;
    }

    public Architecture ropeTheta(float ropeTheta) {
        this.ropeTheta = ropeTheta;
        return this;
    }

    public int ropeDim() {
        return ropeDim > 0 ? ropeDim : headDim();
    }

    public Architecture ropeDim(int ropeDim) {
        this.ropeDim = ropeDim;
        return this;
    }

    public boolean embeddingTied() {
        return embeddingTied;
    }

    public Architecture embeddingTied(boolean embeddingTied) {
        this.embeddingTied = embeddingTied;
        return this;
    }

    public int headDim() {
        return dModel / numHeads;
    }

    public void validate() throws LoadException {
        if (vocabSize <= 0) throw new LoadException("Invalid vocab_size: " + vocabSize);
        if (dModel <= 0) throw new LoadException("Invalid d_model: " + dModel);
        if (numHeads <= 0) throw new LoadException("Invalid num_heads: " + numHeads);
        if (numBlocks <= 0) throw new LoadException("Invalid num_blocks: " + numBlocks);
        if (dff <= 0) throw new LoadException("Invalid dff: " + dff);
        if (maxSeqLen <= 0) throw new LoadException("Invalid max_seq_len: " + maxSeqLen);
        if (dModel % numHeads != 0) {
            throw new LoadException("d_model (" + dModel + ") not divisible by num_heads (" + numHeads + ")");
        }
    }

    public long parameterCount() {
        long params = (long) vocabSize * dModel;
        params += (long) numBlocks * (4L * dModel * dModel + 3L * dModel * dff + 2L * dModel);
        params += dModel;
        if (!embeddingTied) {
            params += (long) dModel * vocabSize;
        }
        return params;
    }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("architecture_family", family);
        m.put("vocab_size", vocabSize);
        m.put("d_model", dModel);
        m.put("num_heads", numHeads);
        m.put("num_kv_heads", numKvHeads());
        m.put("num_blocks", numBlocks);
        m.put("dff", dff);
        m.put("max_seq_len", maxSeqLen);
        m.put("norm_type", normType);
        m.put("norm_eps", normEps);
        m.put("activation_type", activationType);
        m.put("positional_encoding", positionalEncoding);
        m.put("rope_theta", ropeTheta);
        if (ropeDim > 0) {
            m.put("rope_dim", ropeDim);
        }
        m.put("embedding_tied", embeddingTied);
        return m;
    }

    @Override
    public String toString() {
        return "Architecture[vocab=" + vocabSize + ", d_model=" + dModel + ", heads=" + numHeads
                + ", kv_heads=" + numKvHeads() + ", blocks=" + numBlocks + ", dff=" + dff
                + ", max_seq_len=" + maxSeqLen + ", norm=" + normType + ", act=" + activationType
                + ", pe=" + positionalEncoding + "]";
    }
}
