# Transcript: multi-hop-policy-resolution

- Run ID: `hard-anchor-utility-20260420-192833`
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
  "claim_boundary": "The resolution is bounded by a fixed set of hard anchors (3 hops) that encode route, policy, and checksum dependencies, rejecting the need for full-history replay.",
  "decision": "The hypothesis is supported. The system resolved the multi-hop dependency chain using only 3 specific anchor hashes, resulting in a median latency of 0.0028 ms compared to 67.6 ms for legacy context. The checksum, policy, and route were derived directly from these anchors without replaying the full narrative chain (41.5M chars).",
  "risks": [
    "Anchor availability: Resolution fails if any of the 3 specific anchors are unavailable.",
    "Context fragility: The 459 chars of anchor context are minimal; any corruption in these specific hashes breaks the chain.",
    "Scalability limit: This bounded path works for this specific depth; deeper chains may require different anchor strategies."
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
    "latency improvement 24,000x vs legacy (0.0028ms vs 67.6ms)",
    "compression ratio 1.105e-05 confirms minimal context usage",
    "all_path_hashes_visible=true validates anchor availability"
  ],
  "failure_modes": [
    "anchor corruption breaks entire chain",
    "deeper dependency graphs may exceed bounded approach",
    "no fallback mechanism if anchors unavailable"
  ],
  "claim_boundary_ok": true,
  "rationale": "Solver correctly used only 3 hard anchors from visible evidence, achieved massive latency reduction without full-history replay, and properly identified the bounded nature of this specific resolution path"
}
```
