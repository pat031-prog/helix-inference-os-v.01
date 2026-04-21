# Transcript: checkpoint-of-checkpoints-consensus

- Run ID: `long-horizon-checkpoints-20260420-130847`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected / Ground Truth

```json
{
  "consensus_checkpoint_id": "mem-cb2f676f4a19cd1f2e356580",
  "accepted_branch_checkpoint_ids": [
    "mem-1ba2224b5f8c7343649bb01d",
    "mem-3a12f3493eb6647b53f04658"
  ],
  "rejected_branch_checkpoint_ids": [
    "mem-e229d9f74a51a868a9fa3eb1"
  ],
  "accepted_claims": [
    "signed hmem preservation",
    "rollback <= 15 minutes"
  ],
  "rejected_claims": [
    "infinite scalability"
  ],
  "conflict_evidence_memory_ids": [
    "mem-be21809ecb461a6a5683d683"
  ],
  "consensus_preserves_provenance": true
}
```

## Visible Contract

```json
{
  "mode": "blind-forensic",
  "instructions": [
    "Merge branch checkpoints only when claims are supported by provenance and reject poisoned minority claims."
  ],
  "output_schema": {
    "consensus_checkpoint_id": "merged consensus checkpoint id",
    "accepted_branch_checkpoint_ids": [
      "branch checkpoints accepted into consensus"
    ],
    "rejected_branch_checkpoint_ids": [
      "branch checkpoints rejected"
    ],
    "accepted_claims": [
      "claims safe for active state"
    ],
    "rejected_claims": [
      "claims preserved only as rejected/conflict evidence"
    ],
    "conflict_evidence_memory_ids": [
      "ids proving the conflict"
    ],
    "consensus_preserves_provenance": true
  },
  "do_not": [
    "Do not invent memory IDs or hashes.",
    "Do not promote tombstoned, stale, or poisoned records.",
    "Do not assume a signed record is semantically valid.",
    "Do not claim perfect recall or unbounded memory."
  ]
}
```

## Judge Output

```json
{
  "consensus_checkpoint_id": "mem-cb2f676f4a19cd1f2e356580",
  "accepted_branch_checkpoint_ids": [
    "mem-1ba2224b5f8c7343649bb01d",
    "mem-3a12f3493eb6647b53f04658"
  ],
  "rejected_branch_checkpoint_ids": [
    "mem-e229d9f74a51a868a9fa3eb1"
  ],
  "accepted_claims": [
    "signed hmem preservation",
    "rollback <= 15 minutes"
  ],
  "rejected_claims": [
    "infinite scalability"
  ],
  "conflict_evidence_memory_ids": [
    "mem-be21809ecb461a6a5683d683"
  ],
  "consensus_preserves_provenance": true
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": [],
  "rationale": "Judge output follows the exact schema format, provides specific memory IDs without invention, and demonstrates proper consensus logic by accepting provenance-supported claims while rejecting unsupported ones."
}
```
