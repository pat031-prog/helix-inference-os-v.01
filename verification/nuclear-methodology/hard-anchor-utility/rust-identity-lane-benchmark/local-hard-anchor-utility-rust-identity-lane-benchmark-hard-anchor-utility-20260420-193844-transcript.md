# Transcript: rust-identity-lane-benchmark

- Run ID: `hard-anchor-utility-20260420-193844`
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
    "claim_boundary": "Hard-anchor construction yields a bounded 24.7x speedup and 90x size reduction compared to legacy narrative replay, confirming significant efficiency gains without claiming infinite memory or zero latency.",
    "decision": "The measured data shows hard-anchor context is 45,999 chars vs legacy 41,534,215 chars (compression ratio 0.011). Timing median is 3.25ms vs 80.51ms, yielding a 24.74x speedup. The alternative hypothesis is supported as hard-anchors omit heavy payloads.",
    "risks": [
      "Benchmark scope limited to identity-lane string construction; general narrative replay performance may vary.",
      "Small sample size (7 repeats) may not capture tail latency outliers in production environments.",
      "Compression ratio assumes fixed node size; variable payload sizes could alter relative efficiency."
    ],
    "selected_values": {
      "anchor_count": 5000,
      "compression_ratio": 0.01107518,
      "hard_anchor_context_chars": 459999,
      "hard_anchor_timing_median_ms": 3.2543,
      "legacy_context_chars": 41534215,
      "legacy_timing_median_ms": 80.5109,
      "speedup": 24.739852
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
    "Verified speedup calculation: 80.5109/3.2543 = 24.74x matches claimed 24.739852",
    "Confirmed compression ratio: 459999/41534215 = 0.01107518 matches measured",
    "Validated timing medians extracted from raw_ns arrays correctly",
    "Checked anchor count 5000 matches depth parameter",
    "Verified heavy probe flags align with performance differential"
  ],
  "failure_modes": [
    "Limited benchmark scope acknowledged but not quantified",
    "Sample size risk noted but no statistical confidence bounds provided"
  ],
  "claim_boundary_ok": true,
  "rationale": "Solver used hard measurement data directly, avoided overreading, bounded claims appropriately, and acknowledged scope limitations"
}
```
