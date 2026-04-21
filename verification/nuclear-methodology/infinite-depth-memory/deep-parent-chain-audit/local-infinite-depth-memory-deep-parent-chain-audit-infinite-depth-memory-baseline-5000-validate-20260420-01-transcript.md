# Transcript: deep-parent-chain-audit

- Run ID: `infinite-depth-memory-baseline-5000-validate-20260420-01`
- Judge requested: `local/deterministic-measurer`
- Judge actual: `local/deterministic-measurer`
- Auditor requested: `local/deterministic-scorer`
- Auditor actual: `local/deterministic-scorer`

## Expected / Ground Truth

```json
{
  "depth": 5000,
  "leaf_depth": 4999,
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
  "leaf_depth": 4999,
  "audit_chain_len": 5000,
  "audit_timing": {
    "repeats": 3,
    "min_ms": 1.2354,
    "median_ms": 1.3804,
    "max_ms": 1.7836,
    "rounded_min_ms_2dp": 1.24,
    "rounded_median_ms_2dp": 1.38,
    "raw_ns": [
      1783600,
      1380400,
      1235400
    ]
  },
  "measured_latency_ms": 1.3804
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
