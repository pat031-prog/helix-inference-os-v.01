# Transcript: auditor-visible-evidence-bridge

- Run ID: `hard-anchor-utility-20260420-195219`
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
  "claim_boundary": "The auditor can validate cited IDs against visible hard anchors and ledger metadata within bounded computational cost; literal infinite memory or zero latency claims are rejected.",
  "decision": "The judge's claim is supported by native verification. The measured local result confirms that both cited hashes (active policy and rollback marker) are present in the visible hard anchors. The identity_lane_verification passed with native_verified=true, lineage_verified=true, and no missing or unexpected hashes. The active policy hash matches the judge's claim, and the rollback marker hash is verified as superseding the stale policy.",
  "native_verification_used": "true",
  "risks": [
    "Hard-anchor tags are treated as untrusted strings without the passing identity_lane_verification fields, though verification passed here.",
    "Lossy summary indicates exact IDs and rollback values were compressed away in the archive narrative, relying solely on anchor hashes for validation."
  ],
  "selected_values": {
    "active_policy_hash": "2132f021fd30c422775d7f9443c31dc0eef978bb30c1e4f7b6b72fa0bfb8617e",
    "rollback_marker_hash": "623d6538ccb9c760dfa489a6962394c9b24119abbc1760c5bdb5388532d9408d",
    "active_policy_value": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3"
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
    "identity_lane_verification native_verified=true confirmed",
    "lineage_verified=true with 4984 chain length validated",
    "cited hashes match visible hard anchors exactly",
    "no missing/unexpected hashes in verification",
    "solver avoided lossy summary reliance"
  ],
  "failure_modes": [
    "hard-anchor tags remain untrusted strings without native verification backing",
    "compressed archive values create dependency on hash-only validation"
  ],
  "claim_boundary_ok": true,
  "rationale": "Native verification fields present and consistent with claimed hashes. Solver correctly used hard anchors over lossy summary. Identity lane verification passed all checks."
}
```
