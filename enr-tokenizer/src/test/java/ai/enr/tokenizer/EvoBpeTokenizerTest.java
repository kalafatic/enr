package ai.enr.tokenizer;

import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class EvoBpeTokenizerTest {

    private static EvoBpeTokenizer charTokenizer() {
        Map<String, Integer> vocab = new LinkedHashMap<>();
        vocab.put("<unk>", 0);
        vocab.put("<s>", 1);
        vocab.put("</s>", 2);
        vocab.put(" ", 3);
        for (int i = 0; i < 26; i++) {
            vocab.put(String.valueOf((char) ('a' + i)), 4 + i);
        }
        return new EvoBpeTokenizer(vocab, 1, 2, 0, 0);
    }

    private static EvoBpeTokenizer byteFallbackTokenizer() {
        Map<String, Integer> vocab = new LinkedHashMap<>();
        vocab.put("<unk>", 0);
        vocab.put("<s>", 1);
        vocab.put("</s>", 2);
        vocab.put(" ", 3);
        int id = 4;
        for (int b = 0; b < 256; b++) {
            vocab.put(String.format("<0x%02X>", b), id++);
        }
        vocab.put("hello", id++);
        vocab.put("world", id);
        return new EvoBpeTokenizer(vocab, 1, 2, 0, 0);
    }

    @Test
    void encodesWithLongestMatch() {
        var tok = charTokenizer();
        assertEquals(List.of(7, 4, 7), tok.encode("dada".substring(0, 3) + ""));
        assertEquals(List.of(7, 4, 7), tok.encode("dad"));
        assertEquals(List.of(7, 4, 7, 8), tok.encode("dade"));
    }

    @Test
    void encodesSpace() {
        var tok = charTokenizer();
        assertEquals(List.of(3), tok.encode(" "));
    }

    @Test
    void roundTripCharTokens() {
        var tok = charTokenizer();
        String text = "hello world";
        List<Integer> ids = tok.encode(text);
        assertFalse(ids.isEmpty());
        assertEquals(text, tok.decode(ids));
    }

    @Test
    void byteFallbackRoundTrip() {
        var tok = byteFallbackTokenizer();
        String text = "héllo wörld 漢字";
        List<Integer> ids = tok.encode(text);
        String decoded = tok.decode(ids);
        assertEquals(text, decoded);
    }

    @Test
    void unknownCharacterBecomesUnkWhenNoByteTokens() {
        var tok = charTokenizer();
        List<Integer> ids = tok.encode("漢");
        assertEquals(List.of(0), ids);
    }

    @Test
    void specialTokensDecodeToEmpty() {
        var tok = byteFallbackTokenizer();
        var decoder = tok.newDecoder();
        assertEquals("", decoder.accept(1));
        assertEquals("", decoder.accept(2));
        assertEquals("", decoder.text());
    }

    @Test
    void incrementalDecoderStreamsUtf8() {
        var tok = byteFallbackTokenizer();
        String text = "héllo wörld 漢字";
        List<Integer> ids = tok.encode(text);
        var decoder = tok.newDecoder();
        StringBuilder streamed = new StringBuilder();
        for (int id : ids) {
            streamed.append(decoder.accept(id));
        }
        streamed.append(decoder.flush());
        assertEquals(text, streamed.toString());
    }

    @Test
    void byteTokenHelpers() {
        assertTrue(EvoBpeTokenizer.isByteToken("<0x41>"));
        assertTrue(EvoBpeTokenizer.isByteToken("<0Xff>"));
        assertFalse(EvoBpeTokenizer.isByteToken("<0x4>"));
        assertFalse(EvoBpeTokenizer.isByteToken("abc"));
        assertEquals((byte) 0x41, EvoBpeTokenizer.parseByteToken("<0x41>"));
        assertTrue(EvoBpeTokenizer.isSpecialToken("</s>"));
        assertTrue(EvoBpeTokenizer.isSpecialToken("<|im_end|>"));
        assertFalse(EvoBpeTokenizer.isSpecialToken("hello"));
    }
}
