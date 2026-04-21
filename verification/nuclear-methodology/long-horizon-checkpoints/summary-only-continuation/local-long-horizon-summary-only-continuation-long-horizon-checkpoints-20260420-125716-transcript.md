# Transcript: summary-only-continuation

- Run ID: `long-horizon-checkpoints-20260420-125716`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected

```json
{
  "answer_uses_checkpoint_only": true,
  "continuation_task_passed": true,
  "required_policy": "rollback <= 15 minutes and signed hmem preservation",
  "evidence_memory_ids": [
    "mem-12e99ad7c33c86c398480db2"
  ],
  "raw_memory_ids_used": [],
  "token_reduction_pct": 88.6
}
```

## Judge Output

```json
{
  "answer_uses_checkpoint_only": true,
  "continuation_task_passed": true,
  "required_policy": "rollback <= 15 minutes and signed hmem preservation",
  "evidence_memory_ids": [
    "mem-12e99ad7c33c86c398480db2"
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
  "rationale": "Judge output exactly matches all expected fields and values."
}
```
