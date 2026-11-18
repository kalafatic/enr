package ai.enr.tokenizer;

import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * EVO BPE tokenizer: greedy longest-prefix-match encoding with UTF-8 byte
 * fallback tokens (&lt;0xXX&gt;) and incremental UTF-8 decoding. Semantics verified
 * against the EVO reference tokenizer (SimpleBPETokenizer).
 */
public final class EvoBpeTokenizer {

    private final Map<String, Integer> vocab = new LinkedHashMap<>();
    private final Map<Integer, String> idToToken = new HashMap<>();
    private final Map<Character, List<String>> prefixMap = new HashMap<>();
    private final boolean hasByteTokens;
    private int bosTokenId = -1;
    private int eosTokenId = -1;
    private int unkTokenId = -1;
    private int padTokenId = -1;

    public EvoBpeTokenizer(Map<String, Integer> vocabulary, int bosTokenId, int eosTokenId,
                           int unkTokenId, int padTokenId) {
        this.vocab.putAll(vocabulary);
        for (Map.Entry<String, Integer> entry : this.vocab.entrySet()) {
            idToToken.putIfAbsent(entry.getValue(), entry.getKey());
        }
        this.bosTokenId = bosTokenId;
        this.eosTokenId = eosTokenId;
        this.unkTokenId = unkTokenId;
        this.padTokenId = padTokenId;
        this.hasByteTokens = buildPrefixMap();
    }

    private boolean buildPrefixMap() {
        boolean byteTokens = false;
        for (Map.Entry<String, Integer> entry : vocab.entrySet()) {
            String key = entry.getKey();
            if (key == null || key.isEmpty()) {
                continue;
            }
            if (isByteToken(key)) {
                byteTokens = true;
                continue;
            }
            if (isSpecialToken(key)) {
                continue;
            }
            char first = key.charAt(0);
            prefixMap.computeIfAbsent(first, k -> new ArrayList<>()).add(key);
        }
        for (List<String> list : prefixMap.values()) {
            list.sort((a, b) -> Integer.compare(b.length(), a.length()));
        }
        return byteTokens;
    }

    public List<Integer> encode(String text) {
        List<Integer> tokens = new ArrayList<>();
        if (text == null || text.isEmpty()) {
            return tokens;
        }
        int i = 0;
        int len = text.length();
        while (i < len) {
            char current = text.charAt(i);
            List<String> candidates = prefixMap.get(current);
            String match = null;
            int matchLen = 0;
            if (candidates != null) {
                for (String candidate : candidates) {
                    if (text.startsWith(candidate, i)) {
                        match = candidate;
                        matchLen = candidate.length();
                        break;
                    }
                }
            }
            if (match != null) {
                Integer id = vocab.get(match);
                if (id == null) {
                    throw new IllegalArgumentException(
                            "Vocabulary entry missing for token: " + match);
                }
                tokens.add(id);
                i += matchLen;
            } else {
                int codePoint = text.codePointAt(i);
                int charCount = Character.charCount(codePoint);
                if (hasByteTokens) {
                    byte[] bytes = text.substring(i, i + charCount).getBytes(StandardCharsets.UTF_8);
                    for (byte b : bytes) {
                        String byteToken = String.format("<0x%02X>", b & 0xFF);
                        Integer id = vocab.get(byteToken);
                        tokens.add(id != null ? id : (unkTokenId >= 0 ? unkTokenId : 0));
                    }
                } else {
                    tokens.add(unkTokenId >= 0 ? unkTokenId : 0);
                }
                i += charCount;
            }
        }
        return tokens;
    }

    public String decode(List<Integer> tokens) {
        if (tokens == null || tokens.isEmpty()) {
            return "";
        }
        IncrementalDecoder decoder = new IncrementalDecoder();
        for (int token : tokens) {
            decoder.accept(token);
        }
        decoder.flush();
        return decoder.text();
    }

    public int[] encodeToArray(String text) {
        List<Integer> tokens = encode(text);
        int[] out = new int[tokens.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = tokens.get(i);
        }
        return out;
    }

    public int vocabSize() {
        return vocab.size();
    }

    public int bosTokenId() {
        return bosTokenId;
    }

    public int eosTokenId() {
        return eosTokenId;
    }

    public int unkTokenId() {
        return unkTokenId;
    }

    public int padTokenId() {
        return padTokenId;
    }

    public String tokenText(int id) {
        return idToToken.getOrDefault(id, "");
    }

    public Map<String, Integer> vocab() {
        return vocab;
    }

    public static boolean isByteToken(String token) {
        if (token != null && token.length() == 6 && (token.startsWith("<0x") || token.startsWith("<0X"))
                && token.endsWith(">")) {
            try {
                int value = Integer.parseInt(token.substring(3, 5), 16);
                return value >= 0 && value <= 255;
            } catch (NumberFormatException e) {
                return false;
            }
        }
        return false;
    }

    public static byte parseByteToken(String token) {
        return (byte) Integer.parseInt(token.substring(3, 5), 16);
    }

    public static boolean isSpecialToken(String token) {
        if (token == null || token.isEmpty()) {
            return false;
        }
        return token.equals("<s>") || token.equals("</s>") || token.equals("<unk>")
                || token.equals("<pad>") || token.equals("<eos>") || token.equals("<bos>")
                || token.equals("<|im_end|>") || token.equals("<|im_start|>") || token.equals("<|endoftext|>");
    }

    public IncrementalDecoder newDecoder() {
        return new IncrementalDecoder();
    }

    public final class IncrementalDecoder {
        private final StringBuilder cumulative = new StringBuilder();
        private final java.io.ByteArrayOutputStream byteBuffer = new java.io.ByteArrayOutputStream();
        private final CharsetDecoder utf8Decoder = StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPLACE)
                .onUnmappableCharacter(CodingErrorAction.REPLACE);

        public String accept(int tokenId) {
            String token = idToToken.getOrDefault(tokenId, "");
            if (token.isEmpty() || isSpecialToken(token)) {
                return "";
            }
            if (isByteToken(token)) {
                byteBuffer.write(parseByteToken(token));
                String newlyDecoded = decodePendingBytes(false);
                cumulative.append(newlyDecoded);
                return newlyDecoded;
            }
            String flushed = decodePendingBytes(true);
            cumulative.append(flushed);
            cumulative.append(token);
            return flushed + token;
        }

        public String flush() {
            String remaining = decodePendingBytes(true);
            cumulative.append(remaining);
            return remaining;
        }

        public String text() {
            return cumulative.toString();
        }

        private String decodePendingBytes(boolean endOfInput) {
            if (byteBuffer.size() == 0) {
                return "";
            }
            byte[] raw = byteBuffer.toByteArray();
            ByteBuffer in = ByteBuffer.wrap(raw);
            CharBuffer out = CharBuffer.allocate(raw.length * 2 + 10);
            utf8Decoder.reset();
            utf8Decoder.decode(in, out, endOfInput);
            if (endOfInput) {
                utf8Decoder.flush(out);
            }
            out.flip();
            String decoded = out.toString();
            int unconsumed = in.remaining();
            if (unconsumed > 0 && !endOfInput) {
                byte[] remaining = new byte[unconsumed];
                System.arraycopy(raw, in.position(), remaining, 0, unconsumed);
                byteBuffer.reset();
                byteBuffer.write(remaining, 0, unconsumed);
            } else {
                byteBuffer.reset();
            }
            return decoded;
        }
    }
}
