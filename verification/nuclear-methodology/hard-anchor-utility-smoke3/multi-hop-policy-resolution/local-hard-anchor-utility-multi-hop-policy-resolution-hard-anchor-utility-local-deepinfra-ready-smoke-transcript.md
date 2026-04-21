# Transcript: multi-hop-policy-resolution

- Run ID: `hard-anchor-utility-local-deepinfra-ready-smoke`
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
  "deepinfra_enabled": false,
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
    "08a1796a12757480a54cc0bf1fdc1d13d3b575cf6bcf9a9db6db0e1e3325fb72",
    "ddc4ee4be78b115fb94b3d73a2eb1ad39814e52a810d9fa4a4626eaf444478b2",
    "b4eb906574458dcfaeef0e88cd664929bad8d79b642650e6788eb9f1f06f40ff"
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
  "legacy_context_chars": 60315,
  "compression_ratio": 0.00761005,
  "anchor_timing": {
    "repeats": 1,
    "min_ms": 0.006,
    "median_ms": 0.006,
    "max_ms": 0.006,
    "raw_ns": [
      6000
    ]
  },
  "legacy_timing": {
    "repeats": 1,
    "min_ms": 0.1669,
    "median_ms": 0.1669,
    "max_ms": 0.1669,
    "raw_ns": [
      166900
    ]
  },
  "measured_latency_ms": 0.006
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
