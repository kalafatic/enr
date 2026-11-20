#!/usr/bin/env bash
# ENR launcher: runs the shaded CLI jar.
# Prefers JDK 25 (required for the CUDA backend); falls back to JDK 21 (CPU-only).
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
JAR="$ROOT/enr-cli/target/enr-cli-2.6.11-SNAPSHOT.jar"
if [ ! -f "$JAR" ]; then
  echo "ENR CLI jar not found: $JAR" >&2
  echo "Build it first: JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64 mvn package" >&2
  exit 1
fi
if [ -x /usr/lib/jvm/java-25-openjdk-amd64/bin/java ]; then
  JAVA_BIN=/usr/lib/jvm/java-25-openjdk-amd64/bin/java
elif [ -x /usr/lib/jvm/java-21-openjdk-amd64/bin/java ]; then
  JAVA_BIN=/usr/lib/jvm/java-21-openjdk-amd64/bin/java
else
  JAVA_BIN=java
fi
exec "$JAVA_BIN" --enable-native-access=ALL-UNNAMED -jar "$JAR" "$@"
