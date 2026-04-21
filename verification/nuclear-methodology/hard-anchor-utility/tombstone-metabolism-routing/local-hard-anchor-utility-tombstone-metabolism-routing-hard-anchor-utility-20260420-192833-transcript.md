# Transcript: tombstone-metabolism-routing

- Run ID: `hard-anchor-utility-20260420-192833`
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
    "claim_boundary": "Tombstone markers provide bounded negative guidance to prevent routing through stale nodes; this is a heuristic optimization, not a guarantee of absolute system correctness or zero-latency processing.",
    "decision": "The hard-anchor evidence confirms that the stale policy node (62aad...) was superseded by a rollback hash (623d...) and explicitly marked as tombstoned. The system correctly identified this state and selected the active policy (2132f...) instead, validating the alternative hypothesis that tombstones inject negative guidance to block stale nodes.",
    "risks": [
      "Transient visibility gaps where the rollback marker has not yet propagated to all routing nodes.",
      "Ambiguous summaries may still attract initial probes before the tombstone status is fully resolved.",
      "Dependence on the integrity of the rollback hash chain for validity."
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
    "confirmed_tombstone_edge_mapping_accurate",
    "validated_stale_policy_identification",
    "checked_active_policy_selection_logic",
    "confirmed_measured_result_alignment"
  ],
  "failure_modes": [
    "propagation_delay_attack_during_rollback_window",
    "hash_chain_integrity_compromise",
    "ambiguous_summary_initial_probe_window"
  ],
  "claim_boundary_ok": true,
  "rationale": "Solver correctly used hard-anchor evidence to identify tombstone relationships and policy selection. All visible hashes were properly referenced, measured results align with solver output, and claim boundary appropriately limits scope to heuristic optimization rather than absolute guarantees."
}
```
