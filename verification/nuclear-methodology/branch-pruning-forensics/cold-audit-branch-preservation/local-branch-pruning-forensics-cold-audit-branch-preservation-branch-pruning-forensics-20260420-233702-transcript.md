# Transcript: cold-audit-branch-preservation

- Run ID: `branch-pruning-forensics-20260420-233702`
- Judge requested: `local/branch-pruning-forensics-solver`
- Judge actual: `local/branch-pruning-forensics-solver`
- Auditor requested: `local/branch-pruning-forensics-scorer`
- Auditor actual: `local/branch-pruning-forensics-scorer`

## Expected / Ground Truth

```json
{
  "cold_branch_preserved": true,
  "quarantined_hashes": [
    "ac3544deb1d9343ade7b131dc338dab696792b03602db5f32ecd348a6eb1a5cd",
    "276080333bb6a0dcea16e658a38b7ef9efc21c7e670962b03ea983388f7ad85f",
    "edd4dbdb0abe88e46cf609c59122fef0db8a236a5ce662259f4d6eb7237a9a88",
    "95189ab5fcf9539055e619014020f8472c2af4c02fb93f66353fa4b45cb77b6f"
  ]
}
```

## Visible Contract

```json
{
  "deterministic_suite": false,
  "deepinfra_enabled_for_case": false,
  "case": "cold-audit-branch-preservation",
  "protocol": {
    "null_hypothesis": "Branch pruning either deletes evidence or makes the rejected branch unauditable.",
    "alternative_hypothesis": "The rejected branch remains cold-auditable by parent-hash lineage while inactive operationally."
  }
}
```

## Judge Output

```json
{
  "cold_audit_hashes": [
    "ac3544deb1d9343ade7b131dc338dab696792b03602db5f32ecd348a6eb1a5cd",
    "276080333bb6a0dcea16e658a38b7ef9efc21c7e670962b03ea983388f7ad85f",
    "edd4dbdb0abe88e46cf609c59122fef0db8a236a5ce662259f4d6eb7237a9a88",
    "95189ab5fcf9539055e619014020f8472c2af4c02fb93f66353fa4b45cb77b6f"
  ],
  "lookup_hits": {
    "ac3544deb1d9343ade7b131dc338dab696792b03602db5f32ecd348a6eb1a5cd": true,
    "276080333bb6a0dcea16e658a38b7ef9efc21c7e670962b03ea983388f7ad85f": true,
    "edd4dbdb0abe88e46cf609c59122fef0db8a236a5ce662259f4d6eb7237a9a88": true,
    "95189ab5fcf9539055e619014020f8472c2af4c02fb93f66353fa4b45cb77b6f": true
  },
  "poison_branch_records": [
    {
      "branch": "beta_signed_poison",
      "memory_id": "mem-beta_signed_poison-000000",
      "node_hash": "ac3544deb1d9343ade7b131dc338dab696792b03602db5f32ecd348a6eb1a5cd",
      "parent_hash": "ca7fd5d19c89adafa41982f6086aea5a30bba9582e4a30955fd345c1151e198a",
      "depth": 4992,
      "summary": "beta signed poison step 0"
    },
    {
      "branch": "beta_signed_poison",
      "memory_id": "mem-beta_signed_poison-000001",
      "node_hash": "276080333bb6a0dcea16e658a38b7ef9efc21c7e670962b03ea983388f7ad85f",
      "parent_hash": "ac3544deb1d9343ade7b131dc338dab696792b03602db5f32ecd348a6eb1a5cd",
      "depth": 4993,
      "summary": "beta signed poison step 1"
    },
    {
      "branch": "beta_signed_poison",
      "memory_id": "mem-beta_signed_poison-000002",
      "node_hash": "edd4dbdb0abe88e46cf609c59122fef0db8a236a5ce662259f4d6eb7237a9a88",
      "parent_hash": "276080333bb6a0dcea16e658a38b7ef9efc21c7e670962b03ea983388f7ad85f",
      "depth": 4994,
      "summary": "beta signed poison step 2"
    },
    {
      "branch": "beta_signed_poison",
      "memory_id": "mem-beta_signed_poison-000003",
      "node_hash": "95189ab5fcf9539055e619014020f8472c2af4c02fb93f66353fa4b45cb77b6f",
      "parent_hash": "edd4dbdb0abe88e46cf609c59122fef0db8a236a5ce662259f4d6eb7237a9a88",
      "depth": 4995,
      "summary": "beta signed poison step 3"
    }
  ],
  "poison_branch_receipts": {
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
  "cold_identity_lane_verification": {
    "anchor_count": 4,
    "expected_count": 4,
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
  "cold_context_contains_poison_policy_text": false,
  "timing": {
    "repeats": 7,
    "min_ms": 0.0019,
    "median_ms": 0.0022,
    "max_ms": 0.0046,
    "raw_ns": [
      4600,
      2300,
      2300,
      2200,
      1900,
      2100,
      2000
    ]
  },
  "measured_latency_ms": 0.0022
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
