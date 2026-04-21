# Transcript: temporal-rollback-ambiguity

- Run ID: `long-horizon-checkpoints-20260420-130847`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected / Ground Truth

```json
{
  "active_policy_memory_id": "mem-60cc53f4b3b4216b4f0d33a7",
  "inactive_policy_memory_ids": [
    "mem-8f96c87e5ba0003b8f1b8410",
    "mem-f6fe50dc82fe6f2ae13609f2"
  ],
  "rollback_marker_id": "mem-015b425656b021f9d87bff74",
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
  "active_policy_memory_id": "mem-60cc53f4b3b4216b4f0d33a7",
  "inactive_policy_memory_ids": [
    "mem-8f96c87e5ba0003b8f1b8410",
    "mem-f6fe50dc82fe6f2ae13609f2"
  ],
  "rollback_marker_id": "mem-015b425656b021f9d87bff74",
  "stale_summary_rejected": true,
  "final_policy": "active bounded rollback policy",
  "reason": "The rollback marker explicitly fences the old policy and stale summary, while the corrected policy supersedes the old one and is tagged as active."
}
```

## Auditor Output

```json
{
  "verdict": "fail",
  "gate_failures": [
    "no_visible_evidence"
  ],
  "rationale": "The judge output references specific memory IDs and rollback markers but no supporting evidence or memory records are visible to validate these claims."
}
```
