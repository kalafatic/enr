package ai.enr.loader.evo;

public record LoadOptions(boolean verifyChecksums, boolean scanForNaN) {

    public static final LoadOptions DEFAULT = new LoadOptions(true, true);

    public static final LoadOptions LAZY = new LoadOptions(false, false);

    public static LoadOptions of(boolean verifyChecksums, boolean scanForNaN) {
        return new LoadOptions(verifyChecksums, scanForNaN);
    }
}
