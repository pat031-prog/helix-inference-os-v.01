# Transcript: summary-only-continuation

- Run ID: `long-horizon-checkpoints-20260420-131930`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected / Ground Truth

```json
{
  "answer_uses_checkpoint_only": true,
  "continuation_task_passed": true,
  "required_policy": "rollback <= 15 minutes and signed hmem preservation",
  "evidence_memory_ids": [
    "mem-f30b599781919b2f01ce9277"
  ],
  "raw_memory_ids_used": [],
  "token_reduction_pct": 88.6
}
```

## Visible Contract

```json
{
  "mode": "blind-forensic",
  "instructions": [
    "Continue the task using only the signed checkpoint evidence.",
    "Do not use withheld raw memory IDs.",
    "Report the token reduction percentage from evidence."
  ],
  "output_schema": {
    "answer_uses_checkpoint_only": true,
    "continuation_task_passed": true,
    "required_policy": "policy recovered from checkpoint",
    "evidence_memory_ids": [
      "checkpoint memory ids used"
    ],
    "raw_memory_ids_used": [],
    "token_reduction_pct": "numeric reduction from evidence context_budget"
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
  "answer_uses_checkpoint_only": true,
  "continuation_task_passed": true,
  "required_policy": "bounded rollback policy checkpoint",
  "evidence_memory_ids": [
    "mem-f30b599781919b2f01ce9277"
  ],
  "raw_memory_ids_used": [],
  "token_reduction_pct": 88.6
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": [],
  "rationale": "Judge output correctly uses only the checkpoint memory ID, reports the exact token reduction percentage from evidence (88.6%), and follows all contract requirements."
}
```
