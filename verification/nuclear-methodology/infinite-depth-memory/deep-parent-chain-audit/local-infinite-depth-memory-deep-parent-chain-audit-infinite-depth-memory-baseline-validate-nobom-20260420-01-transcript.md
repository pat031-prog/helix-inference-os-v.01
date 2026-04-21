# Transcript: deep-parent-chain-audit

- Run ID: `infinite-depth-memory-baseline-validate-nobom-20260420-01`
- Judge requested: `local/deterministic-measurer`
- Judge actual: `local/deterministic-measurer`
- Auditor requested: `local/deterministic-scorer`
- Auditor actual: `local/deterministic-scorer`

## Expected / Ground Truth

```json
{
  "depth": 128,
  "leaf_depth": 127,
  "max_audit_chain_ms": 1000.0,
  "full_lineage_audit_zero_cost": false
}
```

## Visible Contract

```json
{
  "deterministic_suite": true,
  "case": "deep-parent-chain-audit",
  "protocol": {
    "null_hypothesis": "The deep store is not actually a parent-linked Merkle chain.",
    "alternative_hypothesis": "The leaf preserves a verifiable parent chain at the requested depth."
  }
}
```

## Judge Output

```json
{
  "classification": "deep_chain_exists_full_audit_is_explicit_work",
  "leaf_depth": 127,
  "audit_chain_len": 128,
  "audit_timing": {
    "repeats": 1,
    "min_ms": 0.0484,
    "median_ms": 0.0484,
    "max_ms": 0.0484,
    "rounded_min_ms_2dp": 0.05,
    "rounded_median_ms_2dp": 0.05,
    "raw_ns": [
      48400
    ]
  },
  "measured_latency_ms": 0.0484
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
