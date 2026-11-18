package ai.enr.loader;

import ai.enr.loader.api.LoadedModel;
import ai.enr.loader.api.LoadException;
import ai.enr.loader.evo.EvoModelReader;
import ai.enr.loader.evo.LoadOptions;
import ai.enr.loader.gguf.GgufReader;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

public final class ModelSources {

    private ModelSources() {
    }

    public static LoadedModel open(Path path) throws LoadException {
        return open(path, LoadOptions.DEFAULT);
    }

    public static LoadedModel open(Path path, LoadOptions options) throws LoadException {
        if (path == null || !Files.exists(path)) {
            throw new LoadException("Model file does not exist: " + path);
        }
        if (Files.isDirectory(path)) {
            throw new LoadException("Model path is a directory, expected a model file: " + path);
        }
        byte[] head = readHead(path, 8);
        String magic = new String(head, java.nio.charset.StandardCharsets.US_ASCII);

        if ("GGUF".equals(magic.substring(0, 4))) {
            return new GgufReader().read(path, options.verifyChecksums());
        }
        if (EvoModelReader.MAGIC_V2.equals(magic)) {
            return new EvoModelReader().read(path, options);
        }
        if (magic.startsWith("\u001f\u008b")) { // gzip: legacy v1 candidate
            return new EvoModelReader().read(path, options);
        }
        throw new LoadException("Unrecognized model format in " + path
                + ": magic bytes are not EVO_NAT2, EVO_ARTIFACT_V1 (gzip) or GGUF");
    }

    private static byte[] readHead(Path path, int length) throws LoadException {
        try (InputStream in = Files.newInputStream(path)) {
            return in.readNBytes(length);
        } catch (IOException e) {
            throw new LoadException("Failed to read model header from " + path + ": " + e.getMessage(), e);
        }
    }
}
