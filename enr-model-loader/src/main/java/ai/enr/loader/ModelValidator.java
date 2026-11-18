package ai.enr.loader;

import ai.enr.loader.api.LoadedModel;
import ai.enr.loader.api.TensorEntry;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class ModelValidator {

    public static final class Report {
        private final boolean valid;
        private final List<String> passes = new ArrayList<>();
        private final List<String> errors = new ArrayList<>();

        Report(boolean valid, List<String> passes, List<String> errors) {
            this.valid = valid;
            this.passes.addAll(passes);
            this.errors.addAll(errors);
        }

        public boolean valid() {
            return valid;
        }

        public List<String> passes() {
            return passes;
        }

        public List<String> errors() {
            return errors;
        }

        public String text() {
            StringBuilder sb = new StringBuilder();
            sb.append("ENR MODEL VALIDATION REPORT\n");
            sb.append("===========================\n");
            sb.append("Overall: ").append(valid ? "PASS" : "FAIL").append('\n');
            sb.append("Passed checks: ").append(passes.size()).append('\n');
            for (String pass : passes) {
                sb.append("  [PASS] ").append(pass).append('\n');
            }
            if (!errors.isEmpty()) {
                sb.append("Failed checks: ").append(errors.size()).append('\n');
                for (String error : errors) {
                    sb.append("  [FAIL] ").append(error).append('\n');
                }
            }
            return sb.toString();
        }
    }

    private ModelValidator() {
    }

    public static Report validate(Path path) {
        List<String> passes = new ArrayList<>();
        List<String> errors = new ArrayList<>();
        try {
            LoadedModel model = ModelSources.open(path);
            passes.add("Format detection and structural decoding: " + model.format().label());

            try {
                model.architecture().validate();
                passes.add("Architecture descriptor validation: " + model.architecture());
            } catch (Exception e) {
                errors.add("Architecture: " + e.getMessage());
            }

            int checked = 0;
            for (TensorEntry tensor : model.tensors()) {
                try {
                    float[] data = tensor.loadFloats();
                    if (data.length != tensor.elementCount()) {
                        errors.add("Tensor " + tensor.name() + ": shape " + java.util.Arrays.toString(tensor.shape())
                                + " element count " + tensor.elementCount() + " != stored floats " + data.length);
                        continue;
                    }
                    for (float v : data) {
                        if (Float.isNaN(v) || Float.isInfinite(v)) {
                            errors.add("Tensor " + tensor.name() + ": contains NaN/Infinity");
                            break;
                        }
                    }
                    if (tensor.dtype() == ai.enr.core.tensor.DType.F32) {
                        checked++;
                    }
                } catch (RuntimeException e) {
                    errors.add("Tensor " + tensor.name() + ": " + e.getMessage());
                }
            }
            passes.add("Tensor payload validation (" + checked + " F32 tensors, "
                    + model.totalParameters() + " parameters)");

            if (!model.tokenizer().tokenToId().isEmpty()) {
                passes.add("Tokenizer vocabulary: " + model.tokenizer().vocabSize() + " tokens, BOS="
                        + model.tokenizer().bosTokenId() + ", EOS=" + model.tokenizer().eosTokenId());
            } else {
                errors.add("Tokenizer: empty vocabulary");
            }
        } catch (Exception e) {
            errors.add("Load/validation failure: " + e.getMessage());
        }
        return new Report(errors.isEmpty(), passes, errors);
    }
}
