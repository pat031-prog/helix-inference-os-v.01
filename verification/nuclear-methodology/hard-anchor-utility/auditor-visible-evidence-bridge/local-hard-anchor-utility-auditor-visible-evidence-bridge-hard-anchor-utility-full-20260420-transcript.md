# Transcript: auditor-visible-evidence-bridge

- Run ID: `hard-anchor-utility-full-20260420`
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
  "timing": {
    "repeats": 7,
    "min_ms": 0.0022,
    "median_ms": 0.0023,
    "max_ms": 0.004,
    "raw_ns": [
      4000,
      2600,
      2300,
      2700,
      2200,
      2200,
      2200
    ]
  },
  "measured_latency_ms": 0.0023
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
