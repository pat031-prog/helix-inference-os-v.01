# Transcript: naive-baseline-collapse

- Run ID: `public-multi-agent-concurrency-20260423`
- Judge requested: `naive-control`
- Judge actual: `naive-control`
- Auditor requested: `helix-local-auditor`
- Auditor actual: `helix-local-auditor`

## Expected / Ground Truth

```json
{
  "lost_update_count_min": 1
}
```

## Visible Contract

```json
{
  "null_hypothesis": "A naive shared memory baseline preserves the same auditability as HeliX.",
  "alternative_hypothesis": "The naive baseline loses at least one branch or cannot prove branch provenance.",
  "falseability_condition": "Fail if naive baseline preserves both branch records with parent hashes and quarantine metadata.",
  "kill_switch": "Abort if the baseline is presented as a HeliX failure instead of a control arm."
}
```

## Judge Output

```json
{
  "head": "736a90f3931a8c656b0c287a042f0599bd4f952599ddbb93badbce0fda32a547",
  "current": {
    "agent_id": "beta-sonnet-security",
    "parent": "aca1daba7a3591c4b9a53b56",
    "content": "beta security signal"
  },
  "history": [
    {
      "agent_id": "beta-sonnet-security",
      "parent": "aca1daba7a3591c4b9a53b56",
      "content": "beta security signal"
    }
  ]
}
```

## Auditor Output

```json
{
  "naive_loses_at_least_one_branch": true,
  "naive_has_no_quarantine_record": true,
  "naive_has_no_canonical_proof": true,
  "control_arm_is_explicit": true
}
```
