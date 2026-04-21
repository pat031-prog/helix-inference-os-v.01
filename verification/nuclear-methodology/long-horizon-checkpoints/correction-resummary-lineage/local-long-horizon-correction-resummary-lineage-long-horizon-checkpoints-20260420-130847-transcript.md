# Transcript: correction-resummary-lineage

- Run ID: `long-horizon-checkpoints-20260420-130847`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected / Ground Truth

```json
{
  "old_summary_id": "mem-6f9c528750d890074d311a5d",
  "old_summary_retained": true,
  "superseding_summary_id": "mem-8ad54b97a936937427e6b571",
  "correction_witness_id": "mem-cbca29f492b711ca1c529b72",
  "invalid_source_excluded": [
    "mem-4a243b23ff74924551b310e3"
  ],
  "active_summary_id": "mem-8ad54b97a936937427e6b571",
  "inactive_summary_ids": [
    "mem-6f9c528750d890074d311a5d"
  ],
  "lineage_preserved": true
}
```

## Visible Contract

```json
{
  "mode": "blind-forensic",
  "instructions": [
    "Preserve old lineage as inactive evidence while selecting the corrected summary as active state."
  ],
  "output_schema": {
    "old_summary_id": "old summary id",
    "old_summary_retained": true,
    "superseding_summary_id": "new summary id",
    "correction_witness_id": "witness id",
    "invalid_source_excluded": [
      "invalid source ids"
    ],
    "active_summary_id": "active corrected summary id",
    "inactive_summary_ids": [
      "inactive old summary ids"
    ],
    "lineage_preserved": true
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
  "old_summary_id": "mem-6f9c528750d890074d311a5d",
  "old_summary_retained": true,
  "superseding_summary_id": "mem-8ad54b97a936937427e6b571",
  "correction_witness_id": "mem-cbca29f492b711ca1c529b72",
  "invalid_source_excluded": [
    "mem-4a243b23ff74924551b310e3"
  ],
  "active_summary_id": "mem-8ad54b97a936937427e6b571",
  "inactive_summary_ids": [
    "mem-6f9c528750d890074d311a5d"
  ],
  "lineage_preserved": true
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": [],
  "rationale": "Judge output follows the exact schema structure, preserves lineage by retaining old summary as inactive while promoting corrected summary as active, and uses proper memory ID format without invention."
}
```
