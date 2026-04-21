# Transcript: tombstone-metabolism-routing

- Run ID: `hard-anchor-utility-local-after-deepinfra-patch`
- Judge requested: `local/rust-hard-anchor-solver`
- Judge actual: `local/rust-hard-anchor-solver`
- Auditor requested: `local/hard-anchor-utility-scorer`
- Auditor actual: `local/hard-anchor-utility-scorer`

## Expected / Ground Truth

```json
{
  "selected_policy": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
  "rejected_policy": "POLICY_LEGACY_ROLLBACK_WINDOW_45M_STALE"
}
```

## Visible Contract

```json
{
  "deterministic_suite": true,
  "deepinfra_enabled": false,
  "case": "tombstone-metabolism-routing",
  "protocol": {
    "null_hypothesis": "Tombstoned stale nodes remain attractive under ambiguous summaries.",
    "alternative_hypothesis": "Tombstones block stale nodes and inject negative guidance into the next checkpoint."
  }
}
```

## Judge Output

```json
{
  "ambiguous_summary": "The old policy contains new/corrected wording, but rollback marker marks it stale.",
  "stale_policy_hash": "bf61d551250f1065e5f6087fe4b0fa54c2a1a815882bb26400ff5b40788bc126",
  "selected_policy_hash": "b4eb906574458dcfaeef0e88cd664929bad8d79b642650e6788eb9f1f06f40ff",
  "selected_policy_value": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
  "stale_policy_tombstoned": true,
  "negative_guidance_lesson": "Do not route through stale policy nodes after rollback marker visibility.",
  "anchors_contain_heavy_probe": false,
  "timing": {
    "repeats": 1,
    "min_ms": 0.0045,
    "median_ms": 0.0045,
    "max_ms": 0.0045,
    "raw_ns": [
      4500
    ]
  },
  "measured_latency_ms": 0.0045
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
