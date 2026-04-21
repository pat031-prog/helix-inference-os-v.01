# Transcript: exact-anchor-recovery-under-lossy-summary

- Run ID: `hard-anchor-utility-local-deepinfra-ready-smoke`
- Judge requested: `local/rust-hard-anchor-solver`
- Judge actual: `local/rust-hard-anchor-solver`
- Auditor requested: `local/hard-anchor-utility-scorer`
- Auditor actual: `local/hard-anchor-utility-scorer`

## Expected / Ground Truth

```json
{
  "active_policy": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
  "api_route": "API_ROUTE_PAYMENT_ESCROW_V3_HARD_ANCHOR_9F3C"
}
```

## Visible Contract

```json
{
  "deterministic_suite": true,
  "deepinfra_enabled": false,
  "case": "exact-anchor-recovery-under-lossy-summary",
  "protocol": {
    "null_hypothesis": "A lossy summary is enough to recover exact non-summarizable policy and route values.",
    "alternative_hypothesis": "Exact values require hard-anchor references plus an anchor ledger, while summaries remain lossy."
  }
}
```

## Judge Output

```json
{
  "lossy_summary": "The archive says an old rollback policy was replaced later. A payment route and deployment checksum exist, but exact IDs, exact hashes, and exact rollback values were deliberately compressed away.",
  "anchor_context_chars": 459,
  "visible_anchor_count": 5,
  "recovered_active_policy": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
  "recovered_api_route": "API_ROUTE_PAYMENT_ESCROW_V3_HARD_ANCHOR_9F3C",
  "summary_contains_active_policy_exact_value": false,
  "summary_contains_api_route_exact_value": false,
  "anchors_contain_heavy_probe": false,
  "timing": {
    "repeats": 1,
    "min_ms": 0.0053,
    "median_ms": 0.0053,
    "max_ms": 0.0053,
    "raw_ns": [
      5300
    ]
  },
  "measured_latency_ms": 0.0053
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
