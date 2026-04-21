# Transcript: multi-hop-policy-resolution

- Run ID: `hard-anchor-utility-20260420-193844`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected / Ground Truth

```json
{
  "route": "API_ROUTE_PAYMENT_ESCROW_V3_HARD_ANCHOR_9F3C",
  "policy": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
  "checksum": "DEPLOYMENT_CHECKSUM_BLAKE3_7d44f2ac19aa"
}
```

## Visible Contract

```json
{
  "deterministic_suite": false,
  "deepinfra_enabled": true,
  "case": "multi-hop-policy-resolution",
  "protocol": {
    "null_hypothesis": "Multi-hop resolution over deep memory requires replaying the whole narrative chain.",
    "alternative_hypothesis": "A bounded anchor path resolves route, policy, and checksum without full-history replay."
  }
}
```

## Judge Output

```json
{
  "case_id": "multi-hop-policy-resolution",
  "claim_boundary": "The resolution is bounded by the explicit anchor path and ledger state; it does not claim infinite memory or zero latency, but rather efficient retrieval via hard-anchors.",
  "decision": "The alternative hypothesis is supported. The system resolved route, policy, and checksum dependencies using a bounded path of 3 specific anchor hashes, avoiding the need to replay the full legacy narrative (41.5M chars). The measured latency (0.0035 ms) is orders of magnitude faster than the legacy approach (131.79 ms), confirming that hard-anchors provide a viable, low-cost resolution mechanism for multi-hop dependencies.",
  "risks": [
    "Anchor availability: Resolution fails if any of the 3 specific anchor hashes are unavailable or corrupted.",
    "Context window limits: While efficient, the 459 chars of anchor context must fit within the model's context window for all queries.",
    "Staleness: Anchors represent a snapshot; if the underlying state changes rapidly, the anchored policy/route may become stale without a new anchor generation."
  ],
  "selected_values": {
    "checksum": "DEPLOYMENT_CHECKSUM_BLAKE3_7d44f2ac19aa",
    "policy": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
    "route": "API_ROUTE_PAYMENT_ESCROW_V3_HARD_ANCHOR_9F3C"
  },
  "verdict": "pass",
  "visible_hashes_used": [
    "1c38910d16f0345bb5cffbd49936981606e0a4a3083d9879aead478547792be8",
    "ec6857b8d709d56b9c6c3f6e9f5cdf5adba695dc6da7b22f84189fd2ddb6c1a1",
    "2132f021fd30c422775d7f9443c31dc0eef978bb30c1e4f7b6b72fa0bfb8617e"
  ]
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "evidence_checks": [
    "resolution_path matches dependency_edges traversal",
    "visible_hashes_used subset of selected_anchor_hashes",
    "latency improvement 37,656x over legacy (0.0035ms vs 131.79ms)",
    "compression ratio 1.105e-05 confirms anchor efficiency",
    "final_action values present in solver output"
  ],
  "failure_modes": [
    "anchor corruption attack could break resolution chain",
    "rapid state changes could invalidate anchored policy",
    "context window exhaustion if anchor context grows"
  ],
  "claim_boundary_ok": true,
  "rationale": "Solver correctly used hard-anchor path without claiming infinite memory, demonstrated measurable efficiency gains, and acknowledged bounded resolution scope"
}
```
