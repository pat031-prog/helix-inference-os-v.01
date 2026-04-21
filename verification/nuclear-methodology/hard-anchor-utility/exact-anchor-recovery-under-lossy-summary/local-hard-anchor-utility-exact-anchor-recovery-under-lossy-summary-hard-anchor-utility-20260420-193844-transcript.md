# Transcript: exact-anchor-recovery-under-lossy-summary

- Run ID: `hard-anchor-utility-20260420-193844`
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
  "claim_boundary": "Recovery of exact policy and route values is bounded by the availability of hard-anchor references in the ledger; lossy summaries alone are insufficient for exact reconstruction.",
  "decision": "The null hypothesis is rejected. The lossy summary explicitly states that exact IDs and hashes were compressed away, yet the system successfully recovered 'POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3' and 'API_ROUTE_PAYMENT_ESCROW_V3_HARD_ANCHOR_9F3C'. This recovery was only possible by referencing the specific hard-anchors in the ledger (specifically the active_policy and api_route nodes), proving that exact values require hard-anchor references rather than lossy summaries.",
  "risks": [
    "Dependency on ledger integrity: If the specific anchor nodes (e.g., 2132f021...) are tombstoned or lost, exact recovery fails.",
    "Summary ambiguity: The lossy summary is misleading if interpreted as the sole source of truth, potentially leading to incorrect routing decisions if anchors are unavailable.",
    "Stale policy leakage: The presence of the stale_policy anchor (62aad2...) requires strict adherence to the rollback lesson to prevent routing through obsolete configurations."
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
    "Verified lossy summary explicitly states exact values were compressed away",
    "Confirmed recovered values not present in summary text",
    "Validated solver referenced specific ledger anchor hashes",
    "Checked solver avoided stale policy node per rollback lesson",
    "Confirmed all 5 visible anchor hashes were properly utilized"
  ],
  "failure_modes": [
    "No verification that recovered values actually match anchor content",
    "Solver could fabricate plausible-looking values without true anchor access",
    "Missing validation of dependency chain integrity"
  ],
  "claim_boundary_ok": true,
  "rationale": "Solver correctly demonstrated hard-anchor dependency by recovering exact values absent from lossy summary, properly identified all anchor nodes, and respected rollback constraints. However, audit cannot verify if recovered values truly match anchor contents."
}
```
