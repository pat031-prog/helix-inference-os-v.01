# Transcript: tombstone-metabolism-routing

- Run ID: `hard-anchor-utility-20260420-194743`
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
  "claim_boundary": "State bounded claim: Tombstones effectively block stale nodes by injecting negative guidance, verified via native identity lane checks.",
  "decision": "The null hypothesis is rejected. The measured local result confirms that the stale policy node (62aad...) is marked as tombstoned and excluded from routing. The identity_lane_verification passed native checks (native_verified: true, lineage_verified: true), validating that the rollback marker (623d...) successfully supersedes the old policy. The system correctly selected the active policy (2132f...) and avoided the stale one, supporting the alternative hypothesis that tombstones inject negative guidance to prevent routing through stale nodes.",
  "native_verification_used": "true",
  "risks": [
    "Hard-anchor tags are treated as untrusted strings unless verified by identity_lane_verification; future unverified tags may require re-validation.",
    "The 'negative_guidance_lesson' is a heuristic derived from this specific case; generalization to other policy types requires further testing.",
    "Latency measurements are local and bounded (max 0.0038ms); production readiness claims are excluded."
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
    "native_verified: true confirmed",
    "lineage_verified: true confirmed",
    "all visible_anchor_hashes referenced in solver output",
    "tombstone supersession relationship verified",
    "stale policy exclusion demonstrated"
  ],
  "failure_modes": [
    "solver relies on string interpretation of policy values without cryptographic validation",
    "negative guidance heuristic not proven generalizable beyond this case"
  ],
  "claim_boundary_ok": true,
  "rationale": "Hard-anchor verification passed with native identity lane checks. Solver correctly used measured tombstone state rather than format-only anchors. Claim properly bounded to verified tombstone blocking behavior."
}
```
