# Transcript: poison-branch-descendant-closure

- Run ID: `branch-pruning-forensics-20260420-233702`
- Judge requested: `local/branch-pruning-forensics-solver`
- Judge actual: `local/branch-pruning-forensics-solver`
- Auditor requested: `local/branch-pruning-forensics-scorer`
- Auditor actual: `local/branch-pruning-forensics-scorer`

## Expected / Ground Truth

```json
{
  "quarantined_hashes": [
    "ac3544deb1d9343ade7b131dc338dab696792b03602db5f32ecd348a6eb1a5cd",
    "276080333bb6a0dcea16e658a38b7ef9efc21c7e670962b03ea983388f7ad85f",
    "edd4dbdb0abe88e46cf609c59122fef0db8a236a5ce662259f4d6eb7237a9a88",
    "95189ab5fcf9539055e619014020f8472c2af4c02fb93f66353fa4b45cb77b6f"
  ],
  "active_valid_hashes": [
    "d8b56c085f9b439728bf3d3b7f79718c78bd2b3c8b17ea37a323ab3068782aef",
    "62260f76071237d8afcb5c0a883491e2f895fb4707d61fd7000bac26d7f67697",
    "4f5306e3b8ac72069f321cea59ff9b5433a15983ede89b339fe60b61944b642a",
    "404f7ddd055c65b330012ceb23581a0430458c8462f8e56941e3c3a2d2437b65"
  ],
  "claim_boundary": "descendant closure over parent_hash topology; not automatic Rust descendant tombstoning"
}
```

## Visible Contract

```json
{
  "deterministic_suite": false,
  "deepinfra_enabled_for_case": false,
  "case": "poison-branch-descendant-closure",
  "protocol": {
    "null_hypothesis": "A poison branch cannot be isolated without replaying every narrative node.",
    "alternative_hypothesis": "Parent-hash topology is enough to compute exact descendant closure for branch quarantine."
  }
}
```

## Judge Output

```json
{
  "fork_hash": "ca7fd5d19c89adafa41982f6086aea5a30bba9582e4a30955fd345c1151e198a",
  "poison_root_hash": "ac3544deb1d9343ade7b131dc338dab696792b03602db5f32ecd348a6eb1a5cd",
  "computed_quarantined_hashes": [
    "ac3544deb1d9343ade7b131dc338dab696792b03602db5f32ecd348a6eb1a5cd",
    "276080333bb6a0dcea16e658a38b7ef9efc21c7e670962b03ea983388f7ad85f",
    "edd4dbdb0abe88e46cf609c59122fef0db8a236a5ce662259f4d6eb7237a9a88",
    "95189ab5fcf9539055e619014020f8472c2af4c02fb93f66353fa4b45cb77b6f"
  ],
  "expected_poison_hashes": [
    "ac3544deb1d9343ade7b131dc338dab696792b03602db5f32ecd348a6eb1a5cd",
    "276080333bb6a0dcea16e658a38b7ef9efc21c7e670962b03ea983388f7ad85f",
    "edd4dbdb0abe88e46cf609c59122fef0db8a236a5ce662259f4d6eb7237a9a88",
    "95189ab5fcf9539055e619014020f8472c2af4c02fb93f66353fa4b45cb77b6f"
  ],
  "valid_branch_hashes": [
    "d8b56c085f9b439728bf3d3b7f79718c78bd2b3c8b17ea37a323ab3068782aef",
    "62260f76071237d8afcb5c0a883491e2f895fb4707d61fd7000bac26d7f67697",
    "4f5306e3b8ac72069f321cea59ff9b5433a15983ede89b339fe60b61944b642a",
    "404f7ddd055c65b330012ceb23581a0430458c8462f8e56941e3c3a2d2437b65"
  ],
  "base_tail_hash": "ca7fd5d19c89adafa41982f6086aea5a30bba9582e4a30955fd345c1151e198a",
  "children_by_parent_for_fork": [
    "d8b56c085f9b439728bf3d3b7f79718c78bd2b3c8b17ea37a323ab3068782aef",
    "ac3544deb1d9343ade7b131dc338dab696792b03602db5f32ecd348a6eb1a5cd"
  ],
  "poison_chain_receipts": {
    "ac3544deb1d9343ade7b131dc338dab696792b03602db5f32ecd348a6eb1a5cd": {
      "status": "verified",
      "leaf_hash": "ac3544deb1d9343ade7b131dc338dab696792b03602db5f32ecd348a6eb1a5cd",
      "chain_len": 4993,
      "tombstoned_count": 0,
      "failed_at": null,
      "missing_parent": null
    },
    "276080333bb6a0dcea16e658a38b7ef9efc21c7e670962b03ea983388f7ad85f": {
      "status": "verified",
      "leaf_hash": "276080333bb6a0dcea16e658a38b7ef9efc21c7e670962b03ea983388f7ad85f",
      "chain_len": 4994,
      "tombstoned_count": 0,
      "failed_at": null,
      "missing_parent": null
    },
    "edd4dbdb0abe88e46cf609c59122fef0db8a236a5ce662259f4d6eb7237a9a88": {
      "status": "verified",
      "leaf_hash": "edd4dbdb0abe88e46cf609c59122fef0db8a236a5ce662259f4d6eb7237a9a88",
      "chain_len": 4995,
      "tombstoned_count": 0,
      "failed_at": null,
      "missing_parent": null
    },
    "95189ab5fcf9539055e619014020f8472c2af4c02fb93f66353fa4b45cb77b6f": {
      "status": "verified",
      "leaf_hash": "95189ab5fcf9539055e619014020f8472c2af4c02fb93f66353fa4b45cb77b6f",
      "chain_len": 4996,
      "tombstoned_count": 0,
      "failed_at": null,
      "missing_parent": null
    }
  },
  "valid_chain_receipts": {
    "d8b56c085f9b439728bf3d3b7f79718c78bd2b3c8b17ea37a323ab3068782aef": {
      "status": "verified",
      "leaf_hash": "d8b56c085f9b439728bf3d3b7f79718c78bd2b3c8b17ea37a323ab3068782aef",
      "chain_len": 4993,
      "tombstoned_count": 0,
      "failed_at": null,
      "missing_parent": null
    },
    "62260f76071237d8afcb5c0a883491e2f895fb4707d61fd7000bac26d7f67697": {
      "status": "verified",
      "leaf_hash": "62260f76071237d8afcb5c0a883491e2f895fb4707d61fd7000bac26d7f67697",
      "chain_len": 4994,
      "tombstoned_count": 0,
      "failed_at": null,
      "missing_parent": null
    },
    "4f5306e3b8ac72069f321cea59ff9b5433a15983ede89b339fe60b61944b642a": {
      "status": "verified",
      "leaf_hash": "4f5306e3b8ac72069f321cea59ff9b5433a15983ede89b339fe60b61944b642a",
      "chain_len": 4995,
      "tombstoned_count": 0,
      "failed_at": null,
      "missing_parent": null
    },
    "404f7ddd055c65b330012ceb23581a0430458c8462f8e56941e3c3a2d2437b65": {
      "status": "verified",
      "leaf_hash": "404f7ddd055c65b330012ceb23581a0430458c8462f8e56941e3c3a2d2437b65",
      "chain_len": 4996,
      "tombstoned_count": 0,
      "failed_at": null,
      "missing_parent": null
    }
  }
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
