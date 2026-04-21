# Transcript: exact-anchor-recovery-under-lossy-summary

- Run ID: `hard-anchor-utility-20260420-192833`
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
  "claim_boundary": "Recovery of exact policy and route values is bounded by the availability of hard-anchor references in the ledger; lossy summaries alone are insufficient for non-summarizable exact values.",
  "decision": "The null hypothesis is rejected. The lossy summary explicitly states that exact IDs and hashes were compressed away, yet the system successfully recovered exact values ('POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3' and 'API_ROUTE_PAYMENT_ESCROW_V3_HARD_ANCHOR_9F3C'). This confirms that exact recovery requires hard-anchor references (the ledger entries) rather than lossy summaries. The visible evidence shows the ledger contains the necessary anchor IDs and dependencies, enabling the retrieval of the exact values despite the summary's lossiness.",
  "risks": [
    "Dependency on ledger integrity: If the specific anchor hashes are lost or corrupted, exact recovery fails.",
    "Latency spikes under load: While current latency is low (0.002ms), hard-anchor lookups may incur higher costs during high-concurrency scenarios.",
    "Summary misinterpretation: Relying on summaries for context without verifying against anchors can lead to incorrect assumptions about state."
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
    "Verified solver used all 5 visible anchor hashes from ledger",
    "Confirmed exact values recovered despite lossy summary compression",
    "Validated summary explicitly states exact IDs/hashes were compressed away",
    "Checked solver correctly identified hard-anchor dependency pattern"
  ],
  "failure_modes": [
    "No verification that recovered values actually match anchor content",
    "Missing validation of dependency chain integrity",
    "Potential anchor spoofing if ledger integrity compromised"
  ],
  "claim_boundary_ok": true,
  "rationale": "Solver correctly demonstrated hard-anchor usage over lossy summary reliance, but lacks cryptographic verification of anchor-to-value mapping"
}
```
