# Transcript: auditor-visible-evidence-bridge

- Run ID: `hard-anchor-utility-smoke-20260420`
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
    "rollback_marker_hash": "82033a015b118bec72ba5e3493e07deaebd215c196e02382c7d6b76030f9bd22",
    "active_policy_hash": "b444250b196619f70535c2ae1b9a1ae80809a585eab544509c1b9b36b574f92e",
    "claim": "rollback marker supersedes stale policy and active policy remains current"
  },
  "auditor_visible_hashes": [
    "82033a015b118bec72ba5e3493e07deaebd215c196e02382c7d6b76030f9bd22",
    "b444250b196619f70535c2ae1b9a1ae80809a585eab544509c1b9b36b574f92e"
  ],
  "rollback_supersedes": "69de033288a8b7ffb6f54285f081086bfddcb5dd2671407fd612bb41a7933354",
  "active_policy_value": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
  "no_visible_evidence_avoided": true,
  "anchors_contain_heavy_probe": false,
  "timing": {
    "repeats": 2,
    "min_ms": 0.0018,
    "median_ms": 0.0025,
    "max_ms": 0.0032,
    "raw_ns": [
      3200,
      1800
    ]
  },
  "measured_latency_ms": 0.0025
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
