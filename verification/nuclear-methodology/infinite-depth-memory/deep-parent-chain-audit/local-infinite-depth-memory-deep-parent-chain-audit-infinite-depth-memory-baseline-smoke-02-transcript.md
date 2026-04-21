# Transcript: deep-parent-chain-audit

- Run ID: `infinite-depth-memory-baseline-smoke-02`
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
    "min_ms": 0.0613,
    "median_ms": 0.0613,
    "max_ms": 0.0613,
    "rounded_min_ms_2dp": 0.06,
    "rounded_median_ms_2dp": 0.06,
    "raw_ns": [
      61300
    ]
  },
  "measured_latency_ms": 0.0613
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
