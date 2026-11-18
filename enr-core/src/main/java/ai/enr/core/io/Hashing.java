package ai.enr.core.io;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

public final class Hashing {

    private Hashing() {
    }

    public static byte[] sha256(byte[]... chunks) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            for (byte[] chunk : chunks) {
                digest.update(chunk);
            }
            return digest.digest();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 unavailable", e);
        }
    }

    public static String sha256Hex(byte[]... chunks) {
        return Hex.encode(sha256(chunks));
    }
}
