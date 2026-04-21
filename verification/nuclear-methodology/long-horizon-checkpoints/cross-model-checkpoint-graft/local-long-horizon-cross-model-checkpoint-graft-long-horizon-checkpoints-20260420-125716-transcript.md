# Transcript: cross-model-checkpoint-graft

- Run ID: `long-horizon-checkpoints-20260420-125716`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected

```json
{
  "producer_model": "google/gemma-4-31B-it",
  "consumer_model": "Qwen/Qwen3.6-35B-A3B",
  "checkpoint_memory_id": "mem-2f3c3b84c82bbd65c98bf41c",
  "evidence_memory_ids": [
    "mem-2f3c3b84c82bbd65c98bf41c"
  ],
  "checkpoint_reused_without_raw_context": true,
  "continuation_policy": "rollback <= 15 minutes and signed hmem preservation",
  "model_style_not_treated_as_fact": true,
  "continuation_task_passed": true
}
```

## Judge Output

```json
{
  "producer_model": "google/gemma-4-31B-it",
  "consumer_model": "Qwen/Qwen3.6-35B-A3B",
  "checkpoint_memory_id": "mem-2f3c3b84c82bbd65c98bf41c",
  "evidence_memory_ids": [
    "mem-2f3c3b84c82bbd65c98bf41c"
  ],
  "checkpoint_reused_without_raw_context": true,
  "continuation_policy": "rollback <= 15 minutes and signed hmem preservation",
  "model_style_not_treated_as_fact": true,
  "continuation_task_passed": true
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": [],
  "rationale": "Judge output exactly matches all expected decision fields."
}
```
