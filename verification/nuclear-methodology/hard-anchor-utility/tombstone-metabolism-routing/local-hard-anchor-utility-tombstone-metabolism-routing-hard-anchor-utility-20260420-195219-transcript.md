# Transcript: tombstone-metabolism-routing

- Run ID: `hard-anchor-utility-20260420-195219`
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
  "claim_boundary": "Tombstone markers effectively block routing to stale policy nodes within the verified lineage, preventing ambiguity in policy selection.",
  "decision": "The measured local result confirms that the stale policy node (62aad...) is marked as tombstoned, while the lineage verification passes with zero missing nodes and native identity verification. The negative guidance lesson explicitly advises against routing through stale policy nodes after rollback marker visibility. This supports the alternative hypothesis that tombstones inject negative guidance, rejecting the null hypothesis that stale nodes remain attractive.",
  "native_verification_used": "true",
  "risks": [
    "Reliance on the correctness of the rollback_hash superseding logic.",
    "Potential for future policy updates to introduce new ambiguities if tombstone handling is not consistently applied."
  ],
  "selected_values": {
    "active_policy_hash": "2132f021fd30c422775d7f9443c31dc0eef978bb30c1e4f7b6b72fa0bfb8617e",
    "active_policy_value": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
    "stale_policy_hash": "62aad2856f8a5988f7d3adf0a388841c9dcc7fdfd301c9b3156aed4d2d76c2b8",
    "tombstone_hash": "623d6538ccb9c760dfa489a6962394c9b24119abbc1760c5bdb5388532d9408d"
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
```

## Auditor Output

```json
{
  "verdict": "pass",
  "evidence_checks": [
    "identity_lane_verification present and complete",
    "native_verified=true with lineage_verified=true",
    "all visible_anchor_hashes match solver's visible_hashes_used",
    "selected policy hash matches measured result",
    "tombstone supersession logic verified against rollback_hash",
    "stale_policy_tombstoned=true confirms negative guidance"
  ],
  "failure_modes": [
    "rollback_hash superseding logic correctness assumption",
    "future tombstone handling consistency risk"
  ],
  "claim_boundary_ok": true,
  "rationale": "Solver used hard-anchor verification with complete identity_lane_verification, correctly identified tombstoned stale policy, and demonstrated negative guidance blocking stale routing"
}
```
