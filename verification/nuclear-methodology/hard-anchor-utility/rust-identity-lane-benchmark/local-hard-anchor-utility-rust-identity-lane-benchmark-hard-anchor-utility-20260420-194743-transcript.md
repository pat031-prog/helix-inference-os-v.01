# Transcript: rust-identity-lane-benchmark

- Run ID: `hard-anchor-utility-20260420-194743`
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
  "claim_boundary": "Hard-anchor construction yields a 31.66x speedup and 98.9% size reduction compared to legacy narrative replay, with verified identity lane integrity.",
  "decision": "The measured local result confirms a significant performance advantage for hard-anchors. The median timing for hard-anchors is 3.2313 ms versus 102.3152 ms for legacy, yielding a 31.66x speedup. Size is reduced from 41,534,215 chars to 459,999 chars (compression ratio 0.011). Crucially, identity_lane_verification shows native_verified: true and lineage_verified: true, validating the integrity of the anchor construction. The alternative hypothesis is supported.",
  "native_verification_used": "true",
  "risks": [
    "Hard-anchor tags are treated as untrusted strings unless verified by identity_lane_verification; while verified here, external consumers must replicate this verification.",
    "Benchmark depth is limited to 5000 nodes; performance characteristics at scale may vary.",
    "Compression ratio is high, but decompression overhead for legacy systems is not measured."
  ],
  "selected_values": {
    "hard_anchor_median_ms": 3.2313,
    "legacy_median_ms": 102.3152,
    "speedup": 31.663789,
    "compression_ratio": 0.01107518,
    "hard_anchor_context_chars": 459999,
    "legacy_context_chars": 41534215
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
    "native_verified: true confirmed",
    "lineage_verified: true confirmed",
    "leaf_hash 15df510675b197c1a41d23c01e47089ddd72b08b80cdd8a3e98ed5b5f85a211a matches solver claim",
    "anchor_count 5000 matches expected_count",
    "no missing_nodes or recompute_mismatches",
    "performance claims match measured values exactly"
  ],
  "failure_modes": [
    "Scale dependency: 5000 node limit may not represent production workloads",
    "External verification burden: consumers must independently verify identity lanes"
  ],
  "claim_boundary_ok": true,
  "rationale": "Solver correctly used hard-anchor verification with complete identity_lane_verification proof fields, avoiding lossy summary reliance. Performance and compression claims are precisely supported by measured data."
}
```
