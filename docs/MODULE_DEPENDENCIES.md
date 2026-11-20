# ENR — Module Dependencies

```
enr-core
enr-model-loader  → enr-core
enr-tokenizer     → enr-core
enr-compute       → enr-core
enr-compute-cpu   → enr-core, enr-compute
enr-compute-cuda  → enr-core, enr-compute, enr-compute-cpu   (Java 25)
enr-inference     → enr-core, enr-compute, enr-model-loader, enr-tokenizer
enr-runtime       → enr-core, enr-inference, enr-compute
enr-server        → enr-runtime, enr-java-api
enr-java-api      → enr-core, enr-runtime
enr-cli           → enr-core, enr-runtime, enr-server, enr-java-api, enr-compute-cuda (runtime)
enr-tests         → all of the above
```

Dependency management: `enr-parent/pom.xml` (dependencyManagement for `ai.enr:*`, JUnit 5.10.2). All coordinates `ai.enr:enr-*` 2.6.11-SNAPSHOT.
