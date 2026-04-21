# Transcript: auditor-visible-evidence-bridge

- Run ID: `hard-anchor-utility-local-after-deepinfra-patch`
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
    "rollback_marker_hash": "da1872b88da7178207440fdd60070aa20e16b1f0e2190f8dc3ff72bfb3297f6e",
    "active_policy_hash": "b4eb906574458dcfaeef0e88cd664929bad8d79b642650e6788eb9f1f06f40ff",
    "claim": "rollback marker supersedes stale policy and active policy remains current"
  },
  "auditor_visible_hashes": [
    "b4eb906574458dcfaeef0e88cd664929bad8d79b642650e6788eb9f1f06f40ff",
    "da1872b88da7178207440fdd60070aa20e16b1f0e2190f8dc3ff72bfb3297f6e"
  ],
  "rollback_supersedes": "bf61d551250f1065e5f6087fe4b0fa54c2a1a815882bb26400ff5b40788bc126",
  "active_policy_value": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
  "no_visible_evidence_avoided": true,
  "anchors_contain_heavy_probe": false,
  "timing": {
    "repeats": 1,
    "min_ms": 0.0039,
    "median_ms": 0.0039,
    "max_ms": 0.0039,
    "raw_ns": [
      3900
    ]
  },
  "measured_latency_ms": 0.0039
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
