# Transcript: rust-identity-lane-benchmark

- Run ID: `hard-anchor-utility-full-20260420`
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
    "repeats": 7,
    "min_ms": 76.0868,
    "median_ms": 86.336,
    "max_ms": 91.7367,
    "raw_ns": [
      91736700,
      91359000,
      86336000,
      76086800,
      82431500,
      85275700,
      89454900
    ]
  },
  "hard_anchor_timing": {
    "repeats": 7,
    "min_ms": 2.3854,
    "median_ms": 2.5293,
    "max_ms": 3.4387,
    "raw_ns": [
      3438700,
      3171200,
      3110400,
      2529300,
      2423000,
      2385400,
      2490700
    ]
  },
  "speedup": 34.134345,
  "legacy_context_chars": 41534215,
  "hard_anchor_context_chars": 459999,
  "compression_ratio": 0.01107518,
  "anchor_count": 5000,
  "legacy_contains_heavy_probe": true,
  "anchors_contain_heavy_probe": false,
  "measured_latency_ms": 2.5293
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
