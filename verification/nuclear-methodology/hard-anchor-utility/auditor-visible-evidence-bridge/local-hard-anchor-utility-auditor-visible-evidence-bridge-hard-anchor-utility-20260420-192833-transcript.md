# Transcript: auditor-visible-evidence-bridge

- Run ID: `hard-anchor-utility-20260420-192833`
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
  "claim_boundary": "Auditor can validate cited IDs against visible hard anchors and ledger metadata within bounded latency; does not imply infinite memory or zero latency.",
  "decision": "The judge's cited hashes match the visible hard anchors exactly. The measured local result confirms these hashes are present in the auditor-visible set. The claim that the rollback marker supersedes the stale policy is supported by the presence of both hashes in the anchor context, allowing verification without full-history narrative replay.",
  "risks": [
    "Lossy summary explicitly states exact IDs and hashes were compressed away in the archive, relying solely on the hard anchor context for verification.",
    "Verification is limited to the specific hashes provided; broader policy history is not auditable via this bridge.",
    "Latency is low but non-zero; 'zero latency' claims are invalid."
  ],
  "selected_values": {
    "active_policy_value": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
    "rollback_marker_hash": "623d6538ccb9c760dfa489a6962394c9b24119abbc1760c5bdb5388532d9408d",
    "active_policy_hash": "2132f021fd30c422775d7f9443c31dc0eef978bb30c1e4f7b6b72fa0bfb8617e"
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
    "judge_cited_hashes match visible_hard_anchor_context exactly",
    "solver_visible_hashes_used match auditor_visible_hashes",
    "selected_values align with measured_local_result",
    "no_visible_evidence_avoided confirmed true",
    "latency bounds acknowledged (non-zero)"
  ],
  "failure_modes": [
    "archive compression attack via lossy_summary dependency",
    "limited scope verification (only provided hashes auditable)",
    "timing side-channel potential despite low latency"
  ],
  "claim_boundary_ok": true,
  "rationale": "Solver correctly used hard anchors instead of lossy summary, hash verification passes, claim boundary appropriately scoped to visible evidence without infinite memory claims"
}
```
