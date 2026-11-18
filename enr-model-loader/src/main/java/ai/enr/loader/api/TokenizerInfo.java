package ai.enr.loader.api;

import java.util.LinkedHashMap;
import java.util.Map;

public final class TokenizerInfo {

    private String type = "BPE";
    private int vocabSize;
    private int bosTokenId = 1;
    private int eosTokenId = 2;
    private int unkTokenId;
    private int padTokenId;
    private String vocabularyChecksum = "";
    private final LinkedHashMap<String, Integer> tokenToId = new LinkedHashMap<>();
    private final LinkedHashMap<Integer, String> idToToken = new LinkedHashMap<>();

    public String type() {
        return type;
    }

    public TokenizerInfo type(String type) {
        this.type = type;
        return this;
    }

    public int vocabSize() {
        return vocabSize > 0 ? vocabSize : tokenToId.size();
    }

    public TokenizerInfo vocabSize(int vocabSize) {
        this.vocabSize = vocabSize;
        return this;
    }

    public int bosTokenId() {
        return bosTokenId;
    }

    public TokenizerInfo bosTokenId(int bosTokenId) {
        this.bosTokenId = bosTokenId;
        return this;
    }

    public int eosTokenId() {
        return eosTokenId;
    }

    public TokenizerInfo eosTokenId(int eosTokenId) {
        this.eosTokenId = eosTokenId;
        return this;
    }

    public int unkTokenId() {
        return unkTokenId;
    }

    public TokenizerInfo unkTokenId(int unkTokenId) {
        this.unkTokenId = unkTokenId;
        return this;
    }

    public int padTokenId() {
        return padTokenId;
    }

    public TokenizerInfo padTokenId(int padTokenId) {
        this.padTokenId = padTokenId;
        return this;
    }

    public String vocabularyChecksum() {
        return vocabularyChecksum;
    }

    public TokenizerInfo vocabularyChecksum(String vocabularyChecksum) {
        this.vocabularyChecksum = vocabularyChecksum;
        return this;
    }

    public void addToken(String token, int id) {
        tokenToId.put(token, id);
        idToToken.putIfAbsent(id, token);
    }

    public Map<String, Integer> tokenToId() {
        return tokenToId;
    }

    public Map<Integer, String> idToToken() {
        return idToToken;
    }

    public void ensureSpecialTokens() {
        if (tokenToId.containsKey("<s>")) {
            bosTokenId = tokenToId.get("<s>");
        }
        if (tokenToId.containsKey("</s>")) {
            eosTokenId = tokenToId.get("</s>");
        }
        if (tokenToId.containsKey("<unk>")) {
            unkTokenId = tokenToId.get("<unk>");
        }
        if (tokenToId.containsKey("<pad>")) {
            padTokenId = tokenToId.get("<pad>");
        }
    }
}
