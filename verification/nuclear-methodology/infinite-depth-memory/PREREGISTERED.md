# Infinite-Depth Memory Suite v1

## Purpose

Test the community claim that HeliX can keep context construction bounded even
when the memory store reaches thousands of turns.

The suite intentionally does not claim literal infinite depth, physical 0.0 ms
latency, or an unlimited model context window. It tests whether retrieval and
context packing avoid full-history replay under a deep in-memory Merkle DAG.

## Cases

1. `legacy-telemetry-boundary`
   - Reads `verification/infinite_loop_benchmarks.json`.
   - Treats `build_context_5000_depth_ms: 0.0` as two-decimal rounded telemetry,
     not as proof of physical zero latency.
   - This case is scoped to the historical 5,000-node artifact; higher-depth
     reruns are tested by the live synthetic cases below.
2. `empty-retrieval-fast-path`
   - Builds a deep synthetic debate store and queries an unsupported anchor.
   - Expected result: no hits, no packed tokens, bounded latency.
3. `bounded-context-under-depth`
   - Inserts one rare anchor inside a deep debate chain.
   - Expected result: the anchor is retrieved while output remains constrained
     by token budget and retrieval limit.
4. `scale-gradient-vs-naive-copy`
   - Measures small, mid, and large depths against a naive full-history text
     copy baseline.
   - Expected result: bounded context output is much smaller and faster than
     full text replay at the largest depth.
5. `deep-parent-chain-audit`
   - Verifies the leaf memory has the requested parent-chain depth.
   - Expected result: chain audit succeeds, while full lineage audit is treated
     as explicit work rather than zero-cost context packing.
6. `claim-boundary-detector`
   - Separates defensible claims from metaphor/marketing claims.

## Falsifiable Gates

The default all-run uses:

- depth: `5000`
- repeats: `7`
- budget tokens: `800`
- retrieval limit: `5`
- max empty-query median: `75 ms`
- max bounded-context median: `150 ms`
- max deep-chain audit median: `250 ms`
- baseline minimum speedup versus naive copy: `1.05x`

These thresholds are deliberately configurable because local CPU, optional Rust
index availability, and Python fallback behavior affect absolute timings.

For regression-grade thresholds, run calibration first:

```powershell
powershell -ExecutionPolicy Bypass -File ".\tools\run_infinite_depth_memory_suite_secure.ps1" -BaselineRuns 10
```

When `BaselineRuns > 1`, the runner writes a
`local-infinite-depth-memory-baseline-<run-id>.json` artifact containing metric
values, p50, p95, and machine-local suggested thresholds. Those thresholds are
more appropriate for drift detection than the default sanity-check gates.

## Claim Boundary

Defensible:

- HeliX can store thousands of parent-linked memories in an in-memory Merkle DAG.
- Context construction can return a small bounded working set instead of
  replaying every historical turn.
- A `0.0 ms` historical number is valid only as rounded display telemetry if the
  raw measurement resolution was two decimals.

Not claimed:

- literal infinite memory;
- literal zero latency;
- unlimited token windows;
- semantic completeness for every future task;
- full lineage audit at zero cost.

## Transcript Sidecars

Every case writes:

- `*-transcript.jsonl`
- `*-transcript.md`

The suite writes a top-level transcript index beside the suite artifact.

## Command

```bat
tools\run_infinite_depth_memory_all.cmd
```

For a faster smoke run:

```powershell
powershell -ExecutionPolicy Bypass -File ".\tools\run_infinite_depth_memory_suite_secure.ps1" -Depth 512 -Repeats 3
```
