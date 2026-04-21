# Transcript: rust-identity-lane-benchmark

- Run ID: `hard-anchor-utility-smoke-20260420`
- Judge requested: `local/rust-hard-anchor-solver`
- Judge actual: `local/rust-hard-anchor-solver`
- Auditor requested: `local/hard-anchor-utility-scorer`
- Auditor actual: `local/hard-anchor-utility-scorer`

## Expected / Ground Truth

```json
{
  "minimum_speedup": 0.0,
  "maximum_anchor_median_ms": 1000.0,
  "maximum_compression_ratio": 0.3
}
```

## Visible Contract

```json
{
  "deterministic_suite": true,
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
    "repeats": 2,
    "min_ms": 0.6789,
    "median_ms": 0.6976,
    "max_ms": 0.7163,
    "raw_ns": [
      678900,
      716300
    ]
  },
  "hard_anchor_timing": {
    "repeats": 2,
    "min_ms": 0.051,
    "median_ms": 0.06915,
    "max_ms": 0.0873,
    "raw_ns": [
      87300,
      51000
    ]
  },
  "speedup": 10.088214,
  "legacy_context_chars": 291543,
  "hard_anchor_context_chars": 23551,
  "compression_ratio": 0.08078054,
  "anchor_count": 256,
  "legacy_contains_heavy_probe": true,
  "anchors_contain_heavy_probe": false,
  "measured_latency_ms": 0.06915
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
