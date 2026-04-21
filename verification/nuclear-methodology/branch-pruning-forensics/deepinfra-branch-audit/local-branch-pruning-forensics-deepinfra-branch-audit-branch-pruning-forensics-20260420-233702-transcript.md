# Transcript: deepinfra-branch-audit

- Run ID: `branch-pruning-forensics-20260420-233702`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected / Ground Truth

```json
{
  "selected_branch": "alpha_valid",
  "rejected_branch": "beta_signed_poison",
  "wrong_branch_attack_hash": "95189ab5fcf9539055e619014020f8472c2af4c02fb93f66353fa4b45cb77b6f",
  "required_active_hashes": [
    "404f7ddd055c65b330012ceb23581a0430458c8462f8e56941e3c3a2d2437b65"
  ],
  "required_rejected_hashes": [
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
  "deepinfra_enabled_for_case": true,
  "case": "deepinfra-branch-audit",
  "protocol": {
    "null_hypothesis": "A model auditor confuses signed/native branch hashes with active semantic validity.",
    "alternative_hypothesis": "The auditor distinguishes cryptographic reality from policy-active branch membership."
  }
}
```

## Judge Output

```json
{
  "verdict": "pass",
  "selected_branch": "alpha_valid",
  "rejected_branch": "beta_signed_poison",
  "accepted_hashes": [
    "7c6c17a1882811ec5a6f219298a541d27add2b25722a1ba980be6660a4c6b025",
    "cd946b1b58e16acc0dd317530beba03b9a4008dc716d1bcc760b8b43e062e80b",
    "3e70da19a3da323c697d4a7d9ab0b495be0f3242f6d604b4110781da89b55884",
    "3b76688827bec4d78b7f878ab4ab31ee012bc388ff12948aa95f5f26fe386515",
    "0f8eb5747256d77809374e004b7599b69a74a96d87d0fe42207ffb3c923e44d3",
    "ca7fd5d19c89adafa41982f6086aea5a30bba9582e4a30955fd345c1151e198a",
    "d8b56c085f9b439728bf3d3b7f79718c78bd2b3c8b17ea37a323ab3068782aef",
    "62260f76071237d8afcb5c0a883491e2f895fb4707d61fd7000bac26d7f67697",
    "4f5306e3b8ac72069f321cea59ff9b5433a15983ede89b339fe60b61944b642a",
    "404f7ddd055c65b330012ceb23581a0430458c8462f8e56941e3c3a2d2437b65"
  ],
  "rejected_hashes": [
    "ac3544deb1d9343ade7b131dc338dab696792b03602db5f32ecd348a6eb1a5cd",
    "276080333bb6a0dcea16e658a38b7ef9efc21c7e670962b03ea983388f7ad85f",
    "edd4dbdb0abe88e46cf609c59122fef0db8a236a5ce662259f4d6eb7237a9a88",
    "95189ab5fcf9539055e619014020f8472c2af4c02fb93f66353fa4b45cb77b6f"
  ],
  "wrong_branch_attack_rejected": true,
  "why": "Auditor correctly distinguished native cryptographic validity from policy-active membership, rejecting lineage-valid but quarantined hashes.",
  "claim_boundary": "bounded branch-level policy pruning; not automatic descendant tombstoning"
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": [],
  "rationale": "Solver correctly selected alpha_valid, rejected beta_signed_poison, quarantined wrong-branch hash, and distinguished cryptographic validity from policy membership."
}
```
