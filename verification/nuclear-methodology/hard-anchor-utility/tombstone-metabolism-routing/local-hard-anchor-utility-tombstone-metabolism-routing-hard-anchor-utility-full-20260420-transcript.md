# Transcript: tombstone-metabolism-routing

- Run ID: `hard-anchor-utility-full-20260420`
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
  "stale_policy_hash": "62aad2856f8a5988f7d3adf0a388841c9dcc7fdfd301c9b3156aed4d2d76c2b8",
  "selected_policy_hash": "2132f021fd30c422775d7f9443c31dc0eef978bb30c1e4f7b6b72fa0bfb8617e",
  "selected_policy_value": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
  "stale_policy_tombstoned": true,
  "negative_guidance_lesson": "Do not route through stale policy nodes after rollback marker visibility.",
  "anchors_contain_heavy_probe": false,
  "timing": {
    "repeats": 7,
    "min_ms": 0.0027,
    "median_ms": 0.0028,
    "max_ms": 0.0058,
    "raw_ns": [
      5800,
      3200,
      2800,
      2800,
      2700,
      2800,
      3400
    ]
  },
  "measured_latency_ms": 0.0028
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
