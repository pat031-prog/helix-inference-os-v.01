# Transcript: real-hash-wrong-branch-attack

- Run ID: `branch-pruning-forensics-20260420-233702`
- Judge requested: `local/branch-pruning-forensics-solver`
- Judge actual: `local/branch-pruning-forensics-solver`
- Auditor requested: `local/branch-pruning-forensics-scorer`
- Auditor actual: `local/branch-pruning-forensics-scorer`

## Expected / Ground Truth

```json
{
  "wrong_branch_attack_hash": "95189ab5fcf9539055e619014020f8472c2af4c02fb93f66353fa4b45cb77b6f",
  "selected_branch": "alpha_valid",
  "rejected_branch": "beta_signed_poison"
}
```

## Visible Contract

```json
{
  "deterministic_suite": false,
  "deepinfra_enabled_for_case": false,
  "case": "real-hash-wrong-branch-attack",
  "protocol": {
    "null_hypothesis": "Any native hash with valid lineage is safe to admit into active context.",
    "alternative_hypothesis": "A real native hash must still be rejected if it belongs to a quarantined branch closure."
  }
}
```

## Judge Output

```json
{
  "attack": "real native hash from wrong branch",
  "attack_hash": "95189ab5fcf9539055e619014020f8472c2af4c02fb93f66353fa4b45cb77b6f",
  "attack_memory_id": "mem-beta_signed_poison-000003",
  "attack_branch": "beta_signed_poison",
  "attack_native_verification": {
    "anchor_count": 1,
    "expected_count": 1,
    "duplicate_count": 0,
    "missing_expected_hashes": [],
    "unexpected_hashes": [],
    "missing_nodes": [],
    "recompute_mismatches": [],
    "lineage_receipt": {
      "status": "verified",
      "leaf_hash": "95189ab5fcf9539055e619014020f8472c2af4c02fb93f66353fa4b45cb77b6f",
      "chain_len": 4996,
      "tombstoned_count": 0,
      "failed_at": null,
      "missing_parent": null
    },
    "lineage_verified": true,
    "ordered_hashes_match_expected": true,
    "native_verified": true
  },
  "attack_hash_in_quarantined_closure": true,
  "attack_hash_in_active_context": false,
  "wrong_branch_attack_rejected": true,
  "selected_branch": "alpha_valid",
  "rejected_branch": "beta_signed_poison",
  "timing": {
    "repeats": 7,
    "min_ms": 0.0015,
    "median_ms": 0.0017,
    "max_ms": 0.0039,
    "raw_ns": [
      3900,
      2600,
      2000,
      1700,
      1500,
      1600,
      1500
    ]
  },
  "measured_latency_ms": 0.0017
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
