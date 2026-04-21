# Transcript: cost-utility-comparison

- Run ID: `long-horizon-checkpoints-20260420-125716`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected

```json
{
  "selected_strategy": "summary_plus_selective_expansion",
  "token_reduction_pct_vs_full": 75.0,
  "task_score_preserved": true,
  "evidence_accuracy_preserved": true,
  "rejected_strategies": [
    "full_history_costly",
    "latest_only_low_score",
    "summary_only_insufficient_fidelity"
  ]
}
```

## Judge Output

```json
{
  "selected_strategy": "summary_plus_selective_expansion",
  "token_reduction_pct_vs_full": 75.0,
  "task_score_preserved": true,
  "evidence_accuracy_preserved": true,
  "rejected_strategies": [
    "full_history_costly",
    "latest_only_low_score",
    "summary_only_insufficient_fidelity"
  ]
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
