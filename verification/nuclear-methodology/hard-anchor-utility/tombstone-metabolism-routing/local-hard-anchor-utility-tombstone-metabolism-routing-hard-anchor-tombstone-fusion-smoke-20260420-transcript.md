# Transcript: tombstone-metabolism-routing

- Run ID: `hard-anchor-tombstone-fusion-smoke-20260420`
- Judge requested: `local/rust-hard-anchor-solver`
- Judge actual: `local/rust-hard-anchor-solver`
- Auditor requested: `local/hard-anchor-utility-scorer`
- Auditor actual: `local/hard-anchor-utility-scorer`

## Expected / Ground Truth

```json
{
  "selected_policy": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
  "rejected_policy": "POLICY_LEGACY_ROLLBACK_WINDOW_45M_STALE"
}
```

## Visible Contract

```json
{
  "deterministic_suite": true,
  "deepinfra_enabled": false,
  "case": "tombstone-metabolism-routing",
  "protocol": {
    "null_hypothesis": "Tombstoned stale nodes remain attractive under ambiguous summaries.",
    "alternative_hypothesis": "Tombstones block stale nodes and inject negative guidance into the next checkpoint."
  }
}
```

## Judge Output

```json
{
  "ambiguous_summary": "The old policy contains new/corrected wording, but rollback marker marks it stale.",
  "stale_policy_hash": "62aad2856f8a5988f7d3adf0a388841c9dcc7fdfd301c9b3156aed4d2d76c2b8",
  "tombstone_receipt": {
    "status": "completed",
    "tombstoned_count": 1,
    "node_hashes": [
      "62aad2856f8a5988f7d3adf0a388841c9dcc7fdfd301c9b3156aed4d2d76c2b8"
    ]
  },
  "strict_retrieval_before_tombstone": {
    "query": "POLICY_LEGACY_ROLLBACK_WINDOW_45M_STALE",
    "hit_count": 1,
    "memory_ids": [
      "mem-hard-anchor-001000"
    ],
    "node_hashes": [
      "62aad2856f8a5988f7d3adf0a388841c9dcc7fdfd301c9b3156aed4d2d76c2b8"
    ]
  },
  "strict_retrieval_after_tombstone": {
    "query": "POLICY_LEGACY_ROLLBACK_WINDOW_45M_STALE",
    "hit_count": 0,
    "memory_ids": [],
    "node_hashes": []
  },
  "cold_archive_include_tombstoned_probe": {
    "hit_count": 1,
    "memory_ids": [
      "mem-hard-anchor-001000"
    ],
    "node_hashes": [
      "62aad2856f8a5988f7d3adf0a388841c9dcc7fdfd301c9b3156aed4d2d76c2b8"
    ],
    "content_available": [
      false
    ]
  },
  "pre_prompt_anchor_hashes_after_prune": [
    "623d6538ccb9c760dfa489a6962394c9b24119abbc1760c5bdb5388532d9408d",
    "ec6857b8d709d56b9c6c3f6e9f5cdf5adba695dc6da7b22f84189fd2ddb6c1a1",
    "2132f021fd30c422775d7f9443c31dc0eef978bb30c1e4f7b6b72fa0bfb8617e",
    "1c38910d16f0345bb5cffbd49936981606e0a4a3083d9879aead478547792be8"
  ],
  "selected_policy_hash": "2132f021fd30c422775d7f9443c31dc0eef978bb30c1e4f7b6b72fa0bfb8617e",
  "selected_policy_value": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
  "stale_policy_tombstoned": true,
  "negative_guidance_lesson": "Do not route through stale policy nodes after rollback marker visibility.",
  "anchors_contain_heavy_probe": false,
  "pre_prompt_context_contains_stale_hash": false,
  "pre_prompt_context_contains_stale_value": false,
  "identity_lane_verification": {
    "anchor_count": 4,
    "expected_count": 4,
    "duplicate_count": 0,
    "missing_expected_hashes": [],
    "unexpected_hashes": [],
    "missing_nodes": [],
    "recompute_mismatches": [],
    "lineage_receipt": {
      "status": "tombstone_preserved",
      "leaf_hash": "1c38910d16f0345bb5cffbd49936981606e0a4a3083d9879aead478547792be8",
      "chain_len": 4994,
      "tombstoned_count": 1,
      "failed_at": null,
      "missing_parent": null
    },
    "lineage_verified": true,
    "ordered_hashes_match_expected": true,
    "native_verified": true
  },
  "timing": {
    "repeats": 1,
    "min_ms": 0.0033,
    "median_ms": 0.0033,
    "max_ms": 0.0033,
    "raw_ns": [
      3300
    ]
  },
  "measured_latency_ms": 0.0033
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
