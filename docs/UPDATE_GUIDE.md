# ENR — Update Guide

Git is controlled by the user. Safe update workflow (also exposed by UPM):

1. Inspect status; **stop on local changes** (never discard).
2. `git fetch --all --prune`.
3. Divergence analysis (outgoing/incoming).
4. Stop on unpushed local commits.
5. Require explicit confirmation.
6. `git pull --ff-only` only.
7. Update dependencies: `mvn -B -U -DskipTests clean verify` (JDK 25) when requested.
8. Validate: `mvn -B validate`.

No resets, no rebases, no forced pushes, no silent branch switches. Never silently upgrade dependency versions — `-U` refreshes declared versions only.
