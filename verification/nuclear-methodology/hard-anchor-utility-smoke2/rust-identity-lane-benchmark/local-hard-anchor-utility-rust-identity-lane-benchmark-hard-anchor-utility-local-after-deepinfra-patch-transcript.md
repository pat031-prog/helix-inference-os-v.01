# Transcript: rust-identity-lane-benchmark

- Run ID: `hard-anchor-utility-local-after-deepinfra-patch`
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
    "min_ms": 0.104,
    "median_ms": 0.104,
    "max_ms": 0.104,
    "raw_ns": [
      104000
    ]
  },
  "hard_anchor_timing": {
    "repeats": 1,
    "min_ms": 0.0282,
    "median_ms": 0.0282,
    "max_ms": 0.0282,
    "raw_ns": [
      28200
    ]
  },
  "speedup": 3.687943,
  "legacy_context_chars": 60315,
  "hard_anchor_context_chars": 8831,
  "compression_ratio": 0.14641466,
  "anchor_count": 96,
  "legacy_contains_heavy_probe": true,
  "anchors_contain_heavy_probe": false,
  "measured_latency_ms": 0.0282
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
