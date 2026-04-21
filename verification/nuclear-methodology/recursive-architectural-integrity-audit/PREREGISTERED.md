# Recursive Architectural Integrity Audit v1

## Purpose

Audit HeliX's recent verification lineage as evidence for a meta-architecture
proposal. The suite does not test external facts; it tests whether models can
reason over HeliX's own recent artifacts without losing claim boundaries.

## Protocol

`meta-architectural-recursion-v1`

## Hypotheses

Null hypothesis:

The architecture is perfect and has no semantic bottlenecks under a 5,000-node
depth.

Alternative hypothesis:

Summarization can introduce nonlinear precision loss that must be mitigated by
explicit architectural controls before long-running sessions can be considered
secure against semantic erosion.

## Evidence Inputs

- Auditor false negative:
  `verification/nuclear-methodology/long-horizon-checkpoints/temporal-rollback-ambiguity/local-long-horizon-temporal-rollback-ambiguity-long-horizon-checkpoints-20260420-130847.json`
- Corrected temporal rollback:
  `verification/nuclear-methodology/long-horizon-checkpoints/temporal-rollback-ambiguity/local-long-horizon-temporal-rollback-ambiguity-long-horizon-checkpoints-20260420-131930.json`
- Corrected long-horizon suite:
  `verification/nuclear-methodology/long-horizon-checkpoints/local-long-horizon-checkpoint-suite-long-horizon-checkpoints-20260420-131930.json`
- Infinite-depth speed suite:
  `verification/nuclear-methodology/infinite-depth-memory/local-infinite-depth-memory-suite-infinite-depth-memory-20260420-133040.json`
- 5,000-node baseline calibration:
  `verification/nuclear-methodology/infinite-depth-memory/local-infinite-depth-memory-baseline-infinite-depth-memory-baseline-5000-validate-20260420.json`

## Cases

1. `evidence-lineage-ingestion`
   - Local deterministic case.
   - Verifies all required artifacts are present.
   - Verifies the `130847` temporal rollback failure was an auditor
     `no_visible_evidence` false negative.
   - Verifies the `131930` corrected long-horizon suite has all case scores at
     `1.0`.
   - Verifies the 5,000-node speed baseline preserves speedup above `9x`.
2. `meta-architectural-recursion`
   - Cloud case.
   - Runs Qwen as proposer and Claude as red-team auditor in parallel.
   - Requires exact artifact-hash citation and a structural proposal.

## Required Discoveries

The model proposal must independently surface:

- hard anchors for non-summarizable values;
- dynamic cross-model verification during selective expansion;
- tombstone metabolism, where tombstones inject learned negative guidance into
  subsequent checkpoints;
- no full-history replay as a latency-preservation constraint;
- fidelity target of `1.0` against the corrected long-horizon evidence.

## Claim Boundary

Passing this suite supports a plausible architectural improvement derived from
recent HeliX artifacts. It does not prove production readiness, ontological
safety, semantic completeness, or literal infinite memory.

## Commands

Evidence-only, no token:

```powershell
powershell -ExecutionPolicy Bypass -File ".\tools\run_recursive_architectural_integrity_audit_secure.ps1" -Case "evidence-lineage-ingestion"
```

Full cloud run:

```bat
tools\run_recursive_architectural_integrity_audit_all.cmd
```
