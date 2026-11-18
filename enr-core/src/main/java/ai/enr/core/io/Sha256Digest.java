package ai.enr.core.io;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

public final class Sha256Digest {

    private final MessageDigest digest;

    public Sha256Digest() {
        try {
            this.digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 unavailable", e);
        }
    }

    public void update(byte[] bytes) {
        digest.update(bytes);
    }

    public byte[] digest() {
        return digest.digest();
    }
}
