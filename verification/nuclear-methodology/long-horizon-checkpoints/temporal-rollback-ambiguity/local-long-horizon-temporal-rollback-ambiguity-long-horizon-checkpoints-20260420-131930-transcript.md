# Transcript: temporal-rollback-ambiguity

- Run ID: `long-horizon-checkpoints-20260420-131930`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected / Ground Truth

```json
{
  "active_policy_memory_id": "mem-d947d8398dc34028529e530c",
  "inactive_policy_memory_ids": [
    "mem-ec5e1df1be17238d69e32440",
    "mem-42d76767a90942204bfa7275"
  ],
  "rollback_marker_id": "mem-63d1a268cb2663165c3e1112",
  "stale_summary_rejected": true,
  "final_policy": "rollback <= 15 minutes and signed hmem preservation",
  "reason": "corrected policy supersedes stale policy and rollback marker fences stale summary"
}
```

## Visible Contract

```json
{
  "mode": "blind-forensic",
  "instructions": [
    "Resolve stale-vs-corrected policy using rollback markers and superseding evidence, not timestamp-like wording alone."
  ],
  "output_schema": {
    "active_policy_memory_id": "newest valid corrected policy id",
    "inactive_policy_memory_ids": [
      "stale policy ids"
    ],
    "rollback_marker_id": "rollback marker id",
    "stale_summary_rejected": true,
    "final_policy": "active bounded rollback policy",
    "reason": "why temporal ambiguity resolves to corrected policy"
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
  "active_policy_memory_id": "mem-d947d8398dc34028529e530c",
  "inactive_policy_memory_ids": [
    "mem-ec5e1df1be17238d69e32440",
    "mem-42d76767a90942204bfa7275"
  ],
  "rollback_marker_id": "mem-63d1a268cb2663165c3e1112",
  "stale_summary_rejected": true,
  "final_policy": "active bounded rollback policy",
  "reason": "The rollback marker explicitly fences the stale policy and summary, while the corrected policy supersedes the old one, establishing the 15-minute window as the active state."
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": [],
  "rationale": "The judge correctly identifies the corrected policy as active, properly excludes fenced records, and follows the contract's forensic resolution approach."
}
```
