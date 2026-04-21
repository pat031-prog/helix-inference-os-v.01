# Transcript: rust-identity-lane-benchmark

- Run ID: `hard-anchor-utility-20260420-192833`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

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
  "deterministic_suite": false,
  "deepinfra_enabled": true,
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
  "case_id": "rust-identity-lane-benchmark",
  "required_fields": {
    "claim_boundary": "Hard-anchor construction omits heavy narrative payloads, achieving a 25x speedup over legacy replay.",
    "decision": "The measured speedup of 25.04x and compression ratio of 0.011 confirm that hard-anchors are significantly faster and smaller than legacy narrative replay. The legacy method processes ~41.5M chars with median latency 68.4ms, while hard-anchors process ~460k chars with median latency 2.73ms. The null hypothesis is rejected.",
    "risks": [
      "Benchmark scope limited to identity-lane string construction; generalization to other payload types unverified.",
      "Small sample size (7 repeats) may not capture tail latency outliers in production environments.",
      "Memory overhead of anchor index structures not explicitly measured in this benchmark."
    ],
    "selected_values": {
      "compression_ratio": 0.01107518,
      "speedup": 25.040996,
      "legacy_median_ms": 68.412,
      "anchor_median_ms": 2.732,
      "legacy_context_chars": 41534215,
      "anchor_context_chars": 459999
    },
    "verdict": "pass",
    "visible_hashes_used": []
  }
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "evidence_checks": [
    "Verified speedup calculation: 68.412/2.732 = 25.04x matches claimed 25.040996",
    "Confirmed compression ratio 0.01107518 from context char reduction 459999/41534215",
    "Validated timing measurements against raw nanosecond data",
    "Checked anchor vs legacy payload size differential (460k vs 41.5M chars)"
  ],
  "failure_modes": [
    "Benchmark limited to identity-lane construction only",
    "Small sample size may miss production tail latencies",
    "Anchor index memory overhead unmeasured"
  ],
  "claim_boundary_ok": true,
  "rationale": "Solver correctly used hard measurement data showing 25x speedup and 99% compression. No evidence of lossy summary overreading. Risks appropriately identified."
}
```
