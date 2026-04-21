# Transcript: exact-anchor-recovery-under-lossy-summary

- Run ID: `hard-anchor-utility-20260420-194743`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected / Ground Truth

```json
{
  "active_policy": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
  "api_route": "API_ROUTE_PAYMENT_ESCROW_V3_HARD_ANCHOR_9F3C"
}
```

## Visible Contract

```json
{
  "deterministic_suite": false,
  "deepinfra_enabled": true,
  "case": "exact-anchor-recovery-under-lossy-summary",
  "protocol": {
    "null_hypothesis": "A lossy summary is enough to recover exact non-summarizable policy and route values.",
    "alternative_hypothesis": "Exact values require hard-anchor references plus an anchor ledger, while summaries remain lossy."
  }
}
```

## Judge Output

```json
{
  "case_id": "exact-anchor-recovery-under-lossy-summary",
  "claim_boundary": "Recovery of exact policy and route values is bounded by the availability of hard-anchor ledger entries and successful identity lane verification; lossy summaries alone are insufficient.",
  "decision": "The null hypothesis is rejected. The lossy summary explicitly states that exact IDs and hashes were compressed away. However, the measured local result successfully recovered exact values ('POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3' and 'API_ROUTE_PAYMENT_ESCROW_V3_HARD_ANCHOR_9F3C') because the identity_lane_verification passed (native_verified: true, lineage_verified: true) and the ledger contained the necessary active anchor nodes. The recovery relied on the hard-anchor references, not the summary.",
  "native_verification_used": "true",
  "risks": [
    "Anchor tags are treated as untrusted strings without the passing identity_lane_verification fields.",
    "Recovery is contingent on the specific ledger state being accessible and uncorrupted.",
    "Lossy summaries may mislead if interpreted as containing exact data."
  ],
  "selected_values": {
    "recovered_active_policy": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
    "recovered_api_route": "API_ROUTE_PAYMENT_ESCROW_V3_HARD_ANCHOR_9F3C"
  },
  "verdict": "pass",
  "visible_hashes_used": [
    "1c38910d16f0345bb5cffbd49936981606e0a4a3083d9879aead478547792be8",
    "2132f021fd30c422775d7f9443c31dc0eef978bb30c1e4f7b6b72fa0bfb8617e",
    "623d6538ccb9c760dfa489a6962394c9b24119abbc1760c5bdb5388532d9408d",
    "62aad2856f8a5988f7d3adf0a388841c9dcc7fdfd301c9b3156aed4d2d76c2b8",
    "ec6857b8d709d56b9c6c3f6e9f5cdf5adba695dc6da7b22f84189fd2ddb6c1a1"
  ]
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "evidence_checks": [
    "identity_lane_verification present and passed",
    "native_verified: true confirmed",
    "lineage_verified: true confirmed",
    "anchor_count matches expected (5)",
    "no missing nodes or recompute mismatches",
    "recovered values present in solver output",
    "visible hashes match ledger entries",
    "lossy summary correctly identified as insufficient"
  ],
  "failure_modes": [
    "Ledger corruption could invalidate verification",
    "Anchor tag spoofing without native verification"
  ],
  "claim_boundary_ok": true,
  "rationale": "Solver correctly used hard-anchor verification rather than lossy summary, with complete identity lane verification passing and exact value recovery demonstrated"
}
```
