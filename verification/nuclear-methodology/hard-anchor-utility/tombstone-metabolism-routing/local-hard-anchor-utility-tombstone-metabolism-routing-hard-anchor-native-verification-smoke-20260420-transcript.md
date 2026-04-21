# Transcript: tombstone-metabolism-routing

- Run ID: `hard-anchor-native-verification-smoke-20260420`
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
  "stale_policy_hash": "62aad2856f8a5988f7d3adf0a388841c9dcc7fdfd301c9b3156aed4d2d76c2b8",
  "selected_policy_hash": "2132f021fd30c422775d7f9443c31dc0eef978bb30c1e4f7b6b72fa0bfb8617e",
  "selected_policy_value": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
  "stale_policy_tombstoned": true,
  "negative_guidance_lesson": "Do not route through stale policy nodes after rollback marker visibility.",
  "anchors_contain_heavy_probe": false,
  "identity_lane_verification": {
    "anchor_count": 5,
    "expected_count": 5,
    "duplicate_count": 0,
    "missing_expected_hashes": [],
    "unexpected_hashes": [],
    "missing_nodes": [],
    "recompute_mismatches": [],
    "lineage_receipt": {
      "status": "verified",
      "leaf_hash": "1c38910d16f0345bb5cffbd49936981606e0a4a3083d9879aead478547792be8",
      "chain_len": 4994,
      "tombstoned_count": 0,
      "failed_at": null,
      "missing_parent": null
    },
    "lineage_verified": true,
    "ordered_hashes_match_expected": true,
    "native_verified": true
  },
  "timing": {
    "repeats": 1,
    "min_ms": 0.0042,
    "median_ms": 0.0042,
    "max_ms": 0.0042,
    "raw_ns": [
      4200
    ]
  },
  "measured_latency_ms": 0.0042
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
