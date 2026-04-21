# Transcript: multi-hop-policy-resolution

- Run ID: `hard-anchor-tombstone-fusion-smoke-20260420`
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
    "1c38910d16f0345bb5cffbd49936981606e0a4a3083d9879aead478547792be8",
    "ec6857b8d709d56b9c6c3f6e9f5cdf5adba695dc6da7b22f84189fd2ddb6c1a1",
    "2132f021fd30c422775d7f9443c31dc0eef978bb30c1e4f7b6b72fa0bfb8617e"
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
  "legacy_context_chars": 41534215,
  "compression_ratio": 1.105e-05,
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
  "anchor_timing": {
    "repeats": 1,
    "min_ms": 0.004,
    "median_ms": 0.004,
    "max_ms": 0.004,
    "raw_ns": [
      4000
    ]
  },
  "legacy_timing": {
    "repeats": 1,
    "min_ms": 81.5338,
    "median_ms": 81.5338,
    "max_ms": 81.5338,
    "raw_ns": [
      81533800
    ]
  },
  "measured_latency_ms": 0.004
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
