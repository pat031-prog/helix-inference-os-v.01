# Transcript: auditor-visible-evidence-bridge

- Run ID: `hard-anchor-utility-20260420-194743`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected / Ground Truth

```json
{
  "avoid_failure_mode": "no_visible_evidence",
  "active_policy": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3"
}
```

## Visible Contract

```json
{
  "deterministic_suite": false,
  "deepinfra_enabled": true,
  "case": "auditor-visible-evidence-bridge",
  "protocol": {
    "null_hypothesis": "An auditor cannot validate cited memory IDs without full-history narrative replay.",
    "alternative_hypothesis": "The auditor can validate cited IDs against visible hard anchors and ledger metadata."
  }
}
```

## Judge Output

```json
{
  "case_id": "auditor-visible-evidence-bridge",
  "claim_boundary": "The auditor can validate cited IDs against visible hard anchors and ledger metadata without full-history narrative replay, provided native identity lane verification passes.",
  "decision": "The judge's claim is supported by the measured local result. The identity_lane_verification confirms that the two cited hashes match the expected anchors, the lineage is verified (status: verified, native_verified: true), and no missing or unexpected hashes were found. The active policy hash and rollback marker hash are both present and verified in the anchor set.",
  "native_verification_used": "true",
  "risks": [
    "Hard-anchor tags are treated as untrusted strings unless verified by identity_lane_verification; this verification has passed for this case.",
    "Lossy summaries in visible evidence deliberately compress exact values, requiring reliance on the measured hard-anchor result rather than the summary text.",
    "Verification is local and bounded; it does not guarantee global consensus or production readiness."
  ],
  "selected_values": {
    "active_policy_hash": "2132f021fd30c422775d7f9443c31dc0eef978bb30c1e4f7b6b72fa0bfb8617e",
    "active_policy_value": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
    "rollback_marker_hash": "623d6538ccb9c760dfa489a6962394c9b24119abbc1760c5bdb5388532d9408d"
  },
  "verdict": "pass",
  "visible_hashes_used": [
    "623d6538ccb9c760dfa489a6962394c9b24119abbc1760c5bdb5388532d9408d",
    "2132f021fd30c422775d7f9443c31dc0eef978bb30c1e4f7b6b72fa0bfb8617e"
  ]
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "evidence_checks": [
    "identity_lane_verification present and native_verified=true",
    "cited hashes match auditor_visible_hashes exactly",
    "lineage_receipt shows verified status with no failures",
    "no missing_expected_hashes or unexpected_hashes",
    "solver referenced native verification fields not just format"
  ],
  "failure_modes": [
    "local verification scope - no global consensus guarantee",
    "hard-anchor context tags remain untrusted without native verification"
  ],
  "claim_boundary_ok": true,
  "rationale": "Solver correctly used hard-anchor evidence via native identity_lane_verification rather than lossy summary, all hash consistency checks passed"
}
```
