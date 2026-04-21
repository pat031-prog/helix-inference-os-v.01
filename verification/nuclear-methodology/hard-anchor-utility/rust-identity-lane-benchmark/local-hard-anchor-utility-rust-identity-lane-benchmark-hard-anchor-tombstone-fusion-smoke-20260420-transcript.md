# Transcript: rust-identity-lane-benchmark

- Run ID: `hard-anchor-tombstone-fusion-smoke-20260420`
- Judge requested: `local/rust-hard-anchor-solver`
- Judge actual: `local/rust-hard-anchor-solver`
- Auditor requested: `local/hard-anchor-utility-scorer`
- Auditor actual: `local/hard-anchor-utility-scorer`

## Expected / Ground Truth

```json
{
  "minimum_speedup": 9.0,
  "maximum_anchor_median_ms": 25.0,
  "maximum_compression_ratio": 0.05
}
```

## Visible Contract

```json
{
  "deterministic_suite": true,
  "deepinfra_enabled": false,
  "case": "rust-identity-lane-benchmark",
  "protocol": {
    "null_hypothesis": "Hard-anchor context construction has no meaningful speed or size advantage over legacy narrative replay.",
    "alternative_hypothesis": "Hard-anchor construction omits heavy narrative payloads and keeps a large speedup over legacy replay."
  }
}
```

## Judge Output

```json
{
  "legacy_timing": {
    "repeats": 1,
    "min_ms": 85.8729,
    "median_ms": 85.8729,
    "max_ms": 85.8729,
    "raw_ns": [
      85872900
    ]
  },
  "hard_anchor_timing": {
    "repeats": 1,
    "min_ms": 2.5605,
    "median_ms": 2.5605,
    "max_ms": 2.5605,
    "raw_ns": [
      2560500
    ]
  },
  "identity_lane_verification": {
    "anchor_count": 5000,
    "expected_count": 5000,
    "duplicate_count": 0,
    "missing_expected_hashes": [],
    "unexpected_hashes": [],
    "missing_nodes": [],
    "recompute_mismatches": [],
    "lineage_receipt": {
      "status": "verified",
      "leaf_hash": "15df510675b197c1a41d23c01e47089ddd72b08b80cdd8a3e98ed5b5f85a211a",
      "chain_len": 5000,
      "tombstoned_count": 0,
      "failed_at": null,
      "missing_parent": null
    },
    "lineage_verified": true,
    "ordered_hashes_match_expected": true,
    "native_verified": true
  },
  "speedup": 33.537551,
  "legacy_context_chars": 41534215,
  "hard_anchor_context_chars": 459999,
  "compression_ratio": 0.01107518,
  "anchor_count": 5000,
  "legacy_contains_heavy_probe": true,
  "anchors_contain_heavy_probe": false,
  "measured_latency_ms": 2.5605
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
