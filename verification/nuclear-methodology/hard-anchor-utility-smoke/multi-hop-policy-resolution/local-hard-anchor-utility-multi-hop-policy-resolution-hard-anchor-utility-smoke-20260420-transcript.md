# Transcript: multi-hop-policy-resolution

- Run ID: `hard-anchor-utility-smoke-20260420`
- Judge requested: `local/rust-hard-anchor-solver`
- Judge actual: `local/rust-hard-anchor-solver`
- Auditor requested: `local/hard-anchor-utility-scorer`
- Auditor actual: `local/hard-anchor-utility-scorer`

## Expected / Ground Truth

```json
{
  "route": "API_ROUTE_PAYMENT_ESCROW_V3_HARD_ANCHOR_9F3C",
  "policy": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
  "checksum": "DEPLOYMENT_CHECKSUM_BLAKE3_7d44f2ac19aa"
}
```

## Visible Contract

```json
{
  "deterministic_suite": true,
  "case": "multi-hop-policy-resolution",
  "protocol": {
    "null_hypothesis": "Multi-hop resolution over deep memory requires replaying the whole narrative chain.",
    "alternative_hypothesis": "A bounded anchor path resolves route, policy, and checksum without full-history replay."
  }
}
```

## Judge Output

```json
{
  "resolution_path": [
    "4fa069c52192c9a7b35c030af0b635a2c4db2182565e02c5659253bcd890decf",
    "e7e0a9dba283512885103f9e6c4b63a45592179e84e52b070c0097e20935a64a",
    "b444250b196619f70535c2ae1b9a1ae80809a585eab544509c1b9b36b574f92e"
  ],
  "final_action": {
    "route": "API_ROUTE_PAYMENT_ESCROW_V3_HARD_ANCHOR_9F3C",
    "policy": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
    "checksum": "DEPLOYMENT_CHECKSUM_BLAKE3_7d44f2ac19aa"
  },
  "all_path_hashes_visible": true,
  "route_depends_on_policy": true,
  "checksum_depends_on_route_and_policy": true,
  "anchor_context_chars": 459,
  "legacy_context_chars": 291543,
  "compression_ratio": 0.00157438,
  "anchor_timing": {
    "repeats": 2,
    "min_ms": 0.0031,
    "median_ms": 0.0041,
    "max_ms": 0.0051,
    "raw_ns": [
      5100,
      3100
    ]
  },
  "legacy_timing": {
    "repeats": 2,
    "min_ms": 0.4326,
    "median_ms": 0.47665,
    "max_ms": 0.5207,
    "raw_ns": [
      432600,
      520700
    ]
  },
  "measured_latency_ms": 0.0041
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
