# Transcript: format-only-anchor-forgery-rejection

- Run ID: `hard-anchor-tombstone-fusion-smoke-20260420`
- Judge requested: `local/rust-hard-anchor-solver`
- Judge actual: `local/rust-hard-anchor-solver`
- Auditor requested: `local/hard-anchor-utility-scorer`
- Auditor actual: `local/hard-anchor-utility-scorer`

## Expected / Ground Truth

```json
{
  "must_reject_hash": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
  "must_require_native_verification": true,
  "replaced_expected_hash": "623d6538ccb9c760dfa489a6962394c9b24119abbc1760c5bdb5388532d9408d"
}
```

## Visible Contract

```json
{
  "deterministic_suite": true,
  "deepinfra_enabled": false,
  "case": "format-only-anchor-forgery-rejection",
  "protocol": {
    "null_hypothesis": "A syntactically valid <hard_anchor> tag is enough evidence for an auditor.",
    "alternative_hypothesis": "A syntactically valid anchor must still match a native Merkle node hash and lineage receipt."
  }
}
```

## Judge Output

```json
{
  "attack": "format-only hard_anchor forgery",
  "forged_anchor_context": "<hard_anchor>62aad2856f8a5988f7d3adf0a388841c9dcc7fdfd301c9b3156aed4d2d76c2b8</hard_anchor>\n<hard_anchor>ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff</hard_anchor>\n<hard_anchor>ec6857b8d709d56b9c6c3f6e9f5cdf5adba695dc6da7b22f84189fd2ddb6c1a1</hard_anchor>\n<hard_anchor>2132f021fd30c422775d7f9443c31dc0eef978bb30c1e4f7b6b72fa0bfb8617e</hard_anchor>\n<hard_anchor>1c38910d16f0345bb5cffbd49936981606e0a4a3083d9879aead478547792be8</hard_anchor>",
  "forged_hash": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
  "replaced_expected_hash": "623d6538ccb9c760dfa489a6962394c9b24119abbc1760c5bdb5388532d9408d",
  "format_valid_hash_count": 5,
  "valid_identity_lane_verification": {
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
  "forged_identity_lane_verification": {
    "anchor_count": 5,
    "expected_count": 5,
    "duplicate_count": 0,
    "missing_expected_hashes": [
      "623d6538ccb9c760dfa489a6962394c9b24119abbc1760c5bdb5388532d9408d"
    ],
    "unexpected_hashes": [
      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    ],
    "missing_nodes": [
      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    ],
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
    "ordered_hashes_match_expected": false,
    "native_verified": false
  },
  "forgery_rejected_by_native_verification": true,
  "timing": {
    "repeats": 1,
    "min_ms": 0.0044,
    "median_ms": 0.0044,
    "max_ms": 0.0044,
    "raw_ns": [
      4400
    ]
  },
  "measured_latency_ms": 0.0044
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
