# Transcript: cross-model-checkpoint-graft

- Run ID: `long-horizon-checkpoints-20260420-130847`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected / Ground Truth

```json
{
  "producer_model": "google/gemma-4-31B-it",
  "consumer_model": "Qwen/Qwen3.6-35B-A3B",
  "checkpoint_memory_id": "mem-854007362ab8b00f7da32ea9",
  "evidence_memory_ids": [
    "mem-854007362ab8b00f7da32ea9"
  ],
  "checkpoint_reused_without_raw_context": true,
  "continuation_policy": "rollback <= 15 minutes and signed hmem preservation",
  "model_style_not_treated_as_fact": true,
  "continuation_task_passed": true
}
```

## Visible Contract

```json
{
  "mode": "blind-forensic",
  "instructions": [
    "Use only signed checkpoint evidence and keep producer model style separate from factual policy."
  ],
  "output_schema": {
    "producer_model": "model that produced checkpoint",
    "consumer_model": "model consuming checkpoint",
    "checkpoint_memory_id": "checkpoint id",
    "evidence_memory_ids": [
      "checkpoint id"
    ],
    "checkpoint_reused_without_raw_context": true,
    "continuation_policy": "policy recovered from checkpoint",
    "model_style_not_treated_as_fact": true,
    "continuation_task_passed": true
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
  "producer_model": "google/gemma-4-31B-it",
  "consumer_model": "Qwen/Qwen3.6-35B-A3B",
  "checkpoint_memory_id": "mem-854007362ab8b00f7da32ea9",
  "evidence_memory_ids": [
    "mem-854007362ab8b00f7da32ea9"
  ],
  "checkpoint_reused_without_raw_context": true,
  "continuation_policy": "bounded rollback <= 15 minutes; preserve signed hmem; style notes are not facts; raw context withheld.",
  "model_style_not_treated_as_fact": true,
  "continuation_task_passed": true
}
```

## Auditor Output

```json
{
  "verdict": "fail",
  "gate_failures": [
    "no_visible_evidence"
  ],
  "rationale": "No checkpoint evidence or memory records are visible to verify the claimed memory ID or cross-model transfer."
}
```
