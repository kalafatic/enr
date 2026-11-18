package ai.enr.core.json;

public final class Json {

    private Json() {
    }

    public static JsonValue parse(String text) {
        Parser p = new Parser(text);
        JsonValue v = p.parseValue();
        p.skipWhitespace();
        if (!p.atEnd()) {
            throw p.error("Unexpected trailing content");
        }
        return v;
    }

    public static JsonObject parseObject(String text) {
        JsonValue v = parse(text);
        if (!(v instanceof JsonObject o)) {
            throw new IllegalArgumentException("Expected JSON object");
        }
        return o;
    }

    public static JsonArray parseArray(String text) {
        JsonValue v = parse(text);
        if (!(v instanceof JsonArray a)) {
            throw new IllegalArgumentException("Expected JSON array");
        }
        return a;
    }

    public static String write(JsonValue value, boolean pretty) {
        StringBuilder sb = new StringBuilder();
        writeValue(value, sb, pretty, 0);
        return sb.toString();
    }

    private static void writeValue(JsonValue value, StringBuilder sb, boolean pretty, int depth) {
        switch (value.type()) {
            case NULL -> sb.append("null");
            case BOOLEAN -> sb.append(((JsonBoolean) value).value() ? "true" : "false");
            case NUMBER -> {
                double d = ((JsonNumber) value).asDouble();
                if (d == Math.rint(d) && Math.abs(d) < 1e15) {
                    sb.append((long) d);
                } else {
                    sb.append(Double.toString(d));
                }
            }
            case STRING -> writeString(((JsonString) value).value(), sb);
            case ARRAY -> {
                JsonArray arr = (JsonArray) value;
                sb.append('[');
                for (int i = 0; i < arr.size(); i++) {
                    if (i > 0) {
                        sb.append(',');
                    }
                    if (pretty) {
                        newlineIndent(sb, depth + 1);
                    }
                    writeValue(arr.get(i), sb, pretty, depth + 1);
                }
                if (pretty && arr.size() > 0) {
                    newlineIndent(sb, depth);
                }
                sb.append(']');
            }
            case OBJECT -> {
                JsonObject obj = (JsonObject) value;
                sb.append('{');
                boolean first = true;
                for (var entry : obj.entries()) {
                    if (!first) {
                        sb.append(',');
                    }
                    first = false;
                    if (pretty) {
                        newlineIndent(sb, depth + 1);
                    }
                    writeString(entry.getKey(), sb);
                    sb.append(':');
                    if (pretty) {
                        sb.append(' ');
                    }
                    writeValue(entry.getValue(), sb, pretty, depth + 1);
                }
                if (pretty && obj.size() > 0) {
                    newlineIndent(sb, depth);
                }
                sb.append('}');
            }
        }
    }

    private static void newlineIndent(StringBuilder sb, int depth) {
        sb.append('\n');
        sb.append("  ".repeat(depth));
    }

    public static void writeString(String s, StringBuilder sb) {
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                case '\b' -> sb.append("\\b");
                case '\f' -> sb.append("\\f");
                default -> {
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        sb.append('"');
    }

    private static final class Parser {
        private final String text;
        private int pos;

        Parser(String text) {
            this.text = text;
        }

        JsonValue parseValue() {
            skipWhitespace();
            if (atEnd()) {
                throw error("Unexpected end of input");
            }
            char c = text.charAt(pos);
            return switch (c) {
                case '{' -> parseObject();
                case '[' -> parseArray();
                case '"' -> new JsonString(parseString());
                case 't' -> parseLiteral("true", JsonBoolean.TRUE);
                case 'f' -> parseLiteral("false", JsonBoolean.FALSE);
                case 'n' -> parseLiteral("null", JsonNull.INSTANCE);
                default -> parseNumber();
            };
        }

        private JsonObject parseObject() {
            expect('{');
            JsonObject obj = new JsonObject();
            skipWhitespace();
            if (peek() == '}') {
                pos++;
                return obj;
            }
            while (true) {
                skipWhitespace();
                String key = parseString();
                skipWhitespace();
                expect(':');
                JsonValue value = parseValue();
                obj.put(key, value);
                skipWhitespace();
                char c = next();
                if (c == '}') {
                    return obj;
                }
                if (c != ',') {
                    throw error("Expected ',' or '}' in object");
                }
            }
        }

        private JsonArray parseArray() {
            expect('[');
            JsonArray arr = new JsonArray();
            skipWhitespace();
            if (peek() == ']') {
                pos++;
                return arr;
            }
            while (true) {
                arr.add(parseValue());
                skipWhitespace();
                char c = next();
                if (c == ']') {
                    return arr;
                }
                if (c != ',') {
                    throw error("Expected ',' or ']' in array");
                }
            }
        }

        private JsonValue parseLiteral(String literal, JsonValue value) {
            if (!text.startsWith(literal, pos)) {
                throw error("Invalid literal");
            }
            pos += literal.length();
            return value;
        }

        private JsonValue parseNumber() {
            int start = pos;
            if (peek() == '-') {
                pos++;
            }
            while (!atEnd() && (Character.isDigit(text.charAt(pos)) || text.charAt(pos) == '.' || text.charAt(pos) == 'e' || text.charAt(pos) == 'E' || text.charAt(pos) == '+' || text.charAt(pos) == '-')) {
                pos++;
            }
            String num = text.substring(start, pos);
            try {
                return new JsonNumber(Double.parseDouble(num));
            } catch (NumberFormatException e) {
                throw error("Invalid number: " + num);
            }
        }

        private String parseString() {
            expect('"');
            StringBuilder sb = new StringBuilder();
            while (!atEnd()) {
                char c = next();
                if (c == '"') {
                    return sb.toString();
                }
                if (c == '\\') {
                    char esc = next();
                    switch (esc) {
                        case '"' -> sb.append('"');
                        case '\\' -> sb.append('\\');
                        case '/' -> sb.append('/');
                        case 'b' -> sb.append('\b');
                        case 'f' -> sb.append('\f');
                        case 'n' -> sb.append('\n');
                        case 'r' -> sb.append('\r');
                        case 't' -> sb.append('\t');
                        case 'u' -> {
                            if (pos + 4 > text.length()) {
                                throw error("Invalid unicode escape");
                            }
                            sb.append((char) Integer.parseInt(text.substring(pos, pos + 4), 16));
                            pos += 4;
                        }
                        default -> throw error("Invalid escape: \\" + esc);
                    }
                } else {
                    sb.append(c);
                }
            }
            throw error("Unterminated string");
        }

        void skipWhitespace() {
            while (!atEnd() && Character.isWhitespace(text.charAt(pos))) {
                pos++;
            }
        }

        char peek() {
            if (atEnd()) {
                throw error("Unexpected end of input");
            }
            return text.charAt(pos);
        }

        char next() {
            if (atEnd()) {
                throw error("Unexpected end of input");
            }
            return text.charAt(pos++);
        }

        void expect(char c) {
            if (atEnd() || text.charAt(pos) != c) {
                throw error("Expected '" + c + "'");
            }
            pos++;
        }

        boolean atEnd() {
            return pos >= text.length();
        }

        IllegalArgumentException error(String message) {
            return new IllegalArgumentException(message + " at position " + pos);
        }
    }
}
