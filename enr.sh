#!/usr/bin/env bash
# ENR launcher: runs the CLI from the reactor build output.
# Prefers JDK 25 (required for the CUDA backend); falls back to JDK 21 (CPU-only).
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [ -x /usr/lib/jvm/java-25-openjdk-amd64/bin/java ]; then
  JAVA_BIN=/usr/lib/jvm/java-25-openjdk-amd64/bin/java
elif [ -x /usr/lib/jvm/java-21-openjdk-amd64/bin/java ]; then
  JAVA_BIN=/usr/lib/jvm/java-21-openjdk-amd64/bin/java
else
  JAVA_BIN=java
fi
CP="$ROOT/enr-cli/target/classes:$(cat /tmp/opencode/enr-cp.txt)"
exec "$JAVA_BIN" --enable-native-access=ALL-UNNAMED -cp "$CP" ai.enr.cli.EnrCli "$@"
