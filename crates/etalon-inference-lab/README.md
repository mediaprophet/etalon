# Etalon inference lab

Cold, format-neutral inference configuration experiments, independent of QualiaDB.
Native model execution (including P64) is supplied through `ExperimentExecutor`;
GGUF import is a separate model-format boundary. There is no database, network,
provider discovery or activation dependency in this crate.

`config_space`, `search`, `pareto` and `hypothesis` preserve the configuration,
Sobol/surrogate/bandit, six-objective/profile and research belief APIs.
`campaign::try_run_optimization_campaign` reports persistence and resume errors.
The compatibility campaign API returns an explicitly failed report on error.
Executors must implement `resume_compatible` using model, corpus, runtime, device,
schema and policy identities before a persisted history can be resumed.

`auto_improve` preserves generation/elite/plateau search and CSV/JSON/text/apply
packages through `AutoImproveHost`. `LabMode` keeps product governance modes in
the host. Candidate ranking aggregates all samples of equal execution options.
The old lock-in package name and apply script are retained for compatibility;
the package is a research recommendation and never executes its apply script.

`ablate` injects backend execution; `timeline`, `micro` and `device_roof` own
portable result/report models. Backend counters, actual device tests, semantic
verification/repair, optional external comparisons and product activation remain
host adapters. `runner` defines executor, telemetry, quality, cancellation and
artifact interfaces. `worker` provides hidden Windows child execution, bounded
stdout/stderr, timeout, cancellation and invalid-result rejection. Workers must
not delegate execution to detached descendants.

`TrialEvidence` separates raw/repaired text and quality, consumed seed, requested
and actual work, identities, direct measurements, estimates and unavailable
metrics. Legacy JSONL/CSV fields remain readable; missing evidence is explicitly
incomplete. Token-count quality proxies and unknown backend receipts cannot enter
the live six-objective frontier. Cost is a composite, not a monetary price.
A ledger sample is not a peak-memory measurement; total turn duration is not TTFT.

`qualification::qualify_candidate` uses fresh paired measurements and a held-out
prompt at a frozen improvement threshold. It returns technical evidence only.
The host still owns corpus/policy authorization and production activation.

Artifact reads and serialized writes are bounded; JSONL append retains serialized single-writer ownership with exclusive sidecar markers
(interrupted ownership fails closed and requires explicit recovery) and exposes explicit byte-budget APIs. Staged report publication
uses same-directory RAII temporary files. Corrupt or truncated evidence is an
error. No cleanup traverses broad temp or workspace directories.

This source migration does not qualify any executor/device/model combination.
Runtime and test execution are intentionally deferred by the migration owner.
