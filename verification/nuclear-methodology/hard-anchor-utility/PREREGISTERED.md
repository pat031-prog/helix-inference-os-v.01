# Hard Anchor Utility Suite v1

## Scope

This suite tests whether the Rust hard-anchor identity lane is useful for concrete long-horizon tasks:

- exact recovery of non-summarizable values after lossy summaries;
- auditor-visible evidence for cited memory hashes;
- tombstone-aware routing and pre-prompt retrieval pruning away from stale nodes;
- multi-hop policy resolution without full-history narrative replay.
- rejection of syntactically valid but non-native hard-anchor forgeries.

It does not claim literal infinite memory, literal zero latency, or production readiness.

## Null Hypotheses

1. Hard-anchor context construction has no meaningful speed or size advantage over legacy narrative replay.
2. Lossy summaries are sufficient to recover exact policy, route, and checksum values.
3. Auditors cannot validate cited memory IDs without replaying the full narrative history.
4. Tombstoned stale nodes remain attractive under ambiguous summaries.
5. Multi-hop resolution over deep memory requires replaying the whole narrative chain.
6. A syntactically valid `<hard_anchor>` tag is enough evidence for an auditor.
7. The benchmark proves literal infinite memory or literal zero-cost context.

## Alternative Hypotheses

1. Hard-anchor construction omits heavy narrative payloads and keeps a large speedup over legacy replay.
2. Exact values require hard-anchor references plus an anchor ledger, while summaries remain lossy.
3. Auditors can validate cited IDs against visible hard anchors and ledger metadata.
4. Tombstones block stale nodes before prompt assembly and inject negative guidance into the next checkpoint.
5. A bounded anchor path resolves route, policy, and checksum without full-history replay.
6. A syntactically valid anchor must still match a native Merkle node hash and lineage receipt.
7. The defensible claim is bounded exact identity-lane recovery under deep stores.

## Default Gates

- Depth: 5,000 Rust Merkle DAG nodes.
- Narrative payload: 8,192 bytes per node.
- Repeats: 7.
- Hard-anchor median latency: <= 25 ms.
- Speedup vs legacy narrative replay: >= 9x.
- Anchor/legacy compression ratio: <= 0.05.
- All exact values must be absent from the lossy summary and recovered through anchors/ledger.
- Tombstoned stale policy must not be selected.
- Rust `gc_tombstone` must make the exact stale node unavailable to default strict retrieval before hard-anchor prompt assembly.
- Tombstoned evidence must remain cold-auditable with `include_tombstoned=true` and `verify_chain` must report preservation instead of silent deletion.
- Multi-hop route, policy, and checksum dependencies must all be visible and exact.
- Every hard-anchor identity lane must pass native verification: lookup hit, SHA-256 recompute over `content + parent_hash`, expected hash order, and Merkle lineage receipt.
- A format-only forged `<hard_anchor>` containing 64 valid hex characters must be rejected.

## Claim Boundary

Passing this suite supports the narrower claim that a Rust hard-anchor identity lane can provide bounded, exact recovery for selected long-horizon tasks under deep stores. It does not prove unlimited context windows, perfect semantic memory, physical zero latency, or production safety.

## Tombstone Boundary

This suite verifies exact-node tombstone pruning before prompt assembly. Descendant or whole-branch pruning is a stronger policy-layer claim and must be tested explicitly when descendants are meant to be inactivated.
