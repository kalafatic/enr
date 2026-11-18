package ai.enr.loader.api;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public final class LoadedModel {

    private final ModelFormat format;
    private final String formatVersion;
    private final String name;
    private final long createdAt;
    private final String contentHash;
    private final Architecture architecture;
    private final TokenizerInfo tokenizer;
    private final Map<String, String> metadata;
    private final List<TensorEntry> tensors;
    private final Map<String, TensorEntry> tensorsByName;

    public LoadedModel(ModelFormat format, String formatVersion, String name, long createdAt,
                       String contentHash, Architecture architecture, TokenizerInfo tokenizer,
                       Map<String, String> metadata, List<TensorEntry> tensors) {
        this.format = format;
        this.formatVersion = formatVersion;
        this.name = name;
        this.createdAt = createdAt;
        this.contentHash = contentHash;
        this.architecture = architecture;
        this.tokenizer = tokenizer;
        this.metadata = new LinkedHashMap<>(metadata);
        this.tensors = new ArrayList<>(tensors);
        this.tensorsByName = new LinkedHashMap<>();
        for (TensorEntry entry : tensors) {
            this.tensorsByName.put(entry.name(), entry);
        }
    }

    public ModelFormat format() {
        return format;
    }

    public String formatVersion() {
        return formatVersion;
    }

    public String name() {
        return name;
    }

    public long createdAt() {
        return createdAt;
    }

    public String contentHash() {
        return contentHash;
    }

    public Architecture architecture() {
        return architecture;
    }

    public TokenizerInfo tokenizer() {
        return tokenizer;
    }

    public Map<String, String> metadata() {
        return metadata;
    }

    public List<TensorEntry> tensors() {
        return tensors;
    }

    public Optional<TensorEntry> tensor(String name) {
        return Optional.ofNullable(tensorsByName.get(name));
    }

    public TensorEntry requireTensor(String name) throws LoadException {
        TensorEntry entry = tensorsByName.get(name);
        if (entry == null) {
            throw new LoadException("Required tensor missing: " + name);
        }
        return entry;
    }

    public long totalParameters() {
        long total = 0;
        for (TensorEntry t : tensors) {
            total += t.elementCount();
        }
        return total;
    }

    public long weightBytes() {
        long total = 0;
        for (TensorEntry t : tensors) {
            total += t.elementCount() * t.dtype().elementBytes();
        }
        return total;
    }

    @Override
    public String toString() {
        return "LoadedModel[" + name + ", format=" + format.label() + ", " + architecture
                + ", tensors=" + tensors.size() + ", params=" + totalParameters() + "]";
    }
}
