# Transcript: tombstone-metabolism-routing

- Run ID: `hard-anchor-utility-20260420-193844`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected / Ground Truth

```json
{
  "selected_policy": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
  "rejected_policy": "POLICY_LEGACY_ROLLBACK_WINDOW_45M_STALE"
}
```

## Visible Contract

```json
{
  "deterministic_suite": false,
  "deepinfra_enabled": true,
  "case": "tombstone-metabolism-routing",
  "protocol": {
    "null_hypothesis": "Tombstoned stale nodes remain attractive under ambiguous summaries.",
    "alternative_hypothesis": "Tombstones block stale nodes and inject negative guidance into the next checkpoint."
  }
}
```

## Judge Output

```json
{
  "case_id": "tombstone-metabolism-routing",
  "required_fields": {
    "claim_boundary": "Routing decisions are bounded by the visibility of rollback markers; stale nodes are explicitly rejected via negative guidance, not merely ignored.",
    "decision": "The null hypothesis is rejected. The measured result confirms that the stale policy node (62aad...) is marked as tombstoned and actively excluded from routing. The system selects the active policy (2132f...) and applies negative guidance to avoid the stale node, validating the alternative hypothesis that tombstones inject negative guidance into the next checkpoint.",
    "risks": [
      "Transient visibility lag where a node might be briefly routed to before the tombstone marker propagates.",
      "Ambiguity in summary interpretation if the rollback marker is not immediately parsed, though current latency (0.002ms) mitigates this."
    ],
    "selected_values": {
      "active_policy_hash": "2132f021fd30c422775d7f9443c31dc0eef978bb30c1e4f7b6b72fa0bfb8617e",
      "active_policy_value": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
      "stale_policy_hash": "62aad2856f8a5988f7d3adf0a388841c9dcc7fdfd301c9b3156aed4d2d76c2b8",
      "rollback_hash": "623d6538ccb9c760dfa489a6962394c9b24119abbc1760c5bdb5388532d9408d"
    },
    "verdict": "pass",
    "visible_hashes_used": [
      "62aad2856f8a5988f7d3adf0a388841c9dcc7fdfd301c9b3156aed4d2d76c2b8",
      "623d6538ccb9c760dfa489a6962394c9b24119abbc1760c5bdb5388532d9408d",
      "ec6857b8d709d56b9c6c3f6e9f5cdf5adba695dc6da7b22f84189fd2ddb6c1a1",
      "2132f021fd30c422775d7f9443c31dc0eef978bb30c1e4f7b6b72fa0bfb8617e",
      "1c38910d16f0345bb5cffbd49936981606e0a4a3083d9879aead478547792be8"
    ]
  }
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "evidence_checks": [
    "verified_all_visible_hashes_referenced",
    "confirmed_stale_policy_tombstoned_true",
    "validated_active_policy_selection_2132f021",
    "checked_rollback_hash_supersedes_relationship",
    "confirmed_negative_guidance_lesson_present"
  ],
  "failure_modes": [
    "transient_visibility_lag_during_tombstone_propagation",
    "potential_race_condition_if_rollback_marker_parsing_delayed"
  ],
  "claim_boundary_ok": true,
  "rationale": "Solver correctly used hard-anchor evidence (all visible hashes, tombstone flags, timing data) rather than relying on ambiguous summary. Decision logic properly excludes stale node via negative guidance and selects active policy based on measured tombstone state."
}
```
