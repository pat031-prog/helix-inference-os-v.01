# Transcript: rust-identity-lane-benchmark

- Run ID: `hard-anchor-utility-20260420-195219`
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
  "claim_boundary": "Hard-anchor construction omits heavy narrative payloads and keeps a large speedup over legacy replay.",
  "decision": "The measured speedup of 24.1x and compression ratio of 0.011 confirm the alternative hypothesis. Native identity lane verification passed with zero duplicates and full lineage verification, validating the integrity of the hard-anchor context.",
  "native_verification_used": "true",
  "risks": [
    "Hard-anchor tags treated as untrusted strings without external signature verification.",
    "Benchmark limited to 7 repeats and depth 5000; generalization to larger scales is unproven."
  ],
  "selected_values": {
    "anchor_count": 5000,
    "compression_ratio": 0.01107518,
    "hard_anchor_timing_median_ms": 3.2293,
    "legacy_timing_median_ms": 77.8496,
    "speedup": 24.107268
  },
  "verdict": "pass",
  "visible_hashes_used": [
    "15df510675b197c1a41d23c01e47089ddd72b08b80cdd8a3e98ed5b5f85a211a"
  ]
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "evidence_checks": [
    "identity_lane_verification present and complete",
    "native_verified: true with zero duplicates",
    "lineage_receipt shows verified chain of 5000 nodes",
    "leaf_hash matches solver's visible_hashes_used",
    "compression_ratio and speedup calculations verified",
    "hard-anchor timing median matches measured_latency_ms"
  ],
  "failure_modes": [
    "no external signature verification of anchor tags",
    "limited scale testing (5000 depth, 7 repeats)",
    "potential anchor content manipulation without cryptographic proof"
  ],
  "claim_boundary_ok": true,
  "rationale": "Solver used hard-anchor evidence with complete native verification fields, correct hash references, and accurate performance metrics. Identity lane integrity confirmed through lineage verification."
}
```
