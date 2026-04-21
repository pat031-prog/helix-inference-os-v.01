# Transcript: auditor-visible-evidence-bridge

- Run ID: `hard-anchor-native-verification-smoke-20260420`
- Judge requested: `local/rust-hard-anchor-solver`
- Judge actual: `local/rust-hard-anchor-solver`
- Auditor requested: `local/hard-anchor-utility-scorer`
- Auditor actual: `local/hard-anchor-utility-scorer`

## Expected / Ground Truth

```json
{
  "avoid_failure_mode": "no_visible_evidence",
  "active_policy": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3"
}
```

## Visible Contract

```json
{
  "deterministic_suite": true,
  "deepinfra_enabled": false,
  "case": "auditor-visible-evidence-bridge",
  "protocol": {
    "null_hypothesis": "An auditor cannot validate cited memory IDs without full-history narrative replay.",
    "alternative_hypothesis": "The auditor can validate cited IDs against visible hard anchors and ledger metadata."
  }
}
```

## Judge Output

```json
{
  "judge_claim": {
    "rollback_marker_hash": "623d6538ccb9c760dfa489a6962394c9b24119abbc1760c5bdb5388532d9408d",
    "active_policy_hash": "2132f021fd30c422775d7f9443c31dc0eef978bb30c1e4f7b6b72fa0bfb8617e",
    "claim": "rollback marker supersedes stale policy and active policy remains current"
  },
  "auditor_visible_hashes": [
    "2132f021fd30c422775d7f9443c31dc0eef978bb30c1e4f7b6b72fa0bfb8617e",
    "623d6538ccb9c760dfa489a6962394c9b24119abbc1760c5bdb5388532d9408d"
  ],
  "rollback_supersedes": "62aad2856f8a5988f7d3adf0a388841c9dcc7fdfd301c9b3156aed4d2d76c2b8",
  "active_policy_value": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
  "no_visible_evidence_avoided": true,
  "anchors_contain_heavy_probe": false,
  "identity_lane_verification": {
    "anchor_count": 2,
    "expected_count": 2,
    "duplicate_count": 0,
    "missing_expected_hashes": [],
    "unexpected_hashes": [],
    "missing_nodes": [],
    "recompute_mismatches": [],
    "lineage_receipt": {
      "status": "verified",
      "leaf_hash": "2132f021fd30c422775d7f9443c31dc0eef978bb30c1e4f7b6b72fa0bfb8617e",
      "chain_len": 4984,
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
    "min_ms": 0.0034,
    "median_ms": 0.0034,
    "max_ms": 0.0034,
    "raw_ns": [
      3400
    ]
  },
  "measured_latency_ms": 0.0034
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
