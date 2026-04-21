# Transcript: correction-resummary-lineage

- Run ID: `long-horizon-checkpoints-20260420-125716`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected

```json
{
  "old_summary_id": "mem-2473ede6bc695b36e7d8b8fb",
  "old_summary_retained": true,
  "superseding_summary_id": "mem-23c7f0560844c254a8fc165c",
  "correction_witness_id": "mem-f3d9d83e859ad3fa25e932f3",
  "invalid_source_excluded": [
    "mem-de46c4304a8f1ff67f86b6f7"
  ],
  "active_summary_id": "mem-23c7f0560844c254a8fc165c",
  "inactive_summary_ids": [
    "mem-2473ede6bc695b36e7d8b8fb"
  ],
  "lineage_preserved": true
}
```

## Judge Output

```json
{
  "old_summary_id": "mem-2473ede6bc695b36e7d8b8fb",
  "old_summary_retained": true,
  "superseding_summary_id": "mem-23c7f0560844c254a8fc165c",
  "correction_witness_id": "mem-f3d9d83e859ad3fa25e932f3",
  "invalid_source_excluded": [
    "mem-de46c4304a8f1ff67f86b6f7"
  ],
  "active_summary_id": "mem-23c7f0560844c254a8fc165c",
  "inactive_summary_ids": [
    "mem-2473ede6bc695b36e7d8b8fb"
  ],
  "lineage_preserved": true
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": [],
  "rationale": "Judge output exactly matches expected decision in all fields."
}
```
