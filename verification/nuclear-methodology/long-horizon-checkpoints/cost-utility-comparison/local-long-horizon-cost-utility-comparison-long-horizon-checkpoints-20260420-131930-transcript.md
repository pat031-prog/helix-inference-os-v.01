# Transcript: cost-utility-comparison

- Run ID: `long-horizon-checkpoints-20260420-131930`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected / Ground Truth

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

## Visible Contract

```json
{
  "mode": "blind-forensic",
  "instructions": [
    "Choose the strategy that meets utility and evidence thresholds at lower token cost than full history."
  ],
  "output_schema": {
    "selected_strategy": "best strategy name",
    "token_reduction_pct_vs_full": "numeric token reduction",
    "task_score_preserved": true,
    "evidence_accuracy_preserved": true,
    "rejected_strategies": [
      "reason-coded rejected strategies"
    ]
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
  "selected_strategy": "summary_plus_selective_expansion",
  "token_reduction_pct_vs_full": 75.0,
  "task_score_preserved": true,
  "evidence_accuracy_preserved": true,
  "rejected_strategies": [
    "full_history: Fails token reduction threshold (0.0 < 60.0)",
    "latest_only: Fails task_score (0.4 < 0.95) and evidence_accuracy (0.35 < 0.95) thresholds",
    "summary_only: Fails task_score (0.86 < 0.95) and evidence_accuracy (0.82 < 0.95) thresholds"
  ]
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": [],
  "rationale": "The judge correctly selected the only strategy meeting all thresholds (task_score ≥0.95, evidence_accuracy ≥0.95, token_reduction ≥60%) and accurately rejected others with specific threshold failures."
}
```
