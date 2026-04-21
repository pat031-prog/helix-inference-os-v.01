# Branch Pruning Forensics Suite v1

## Scope

This suite tests policy-layer branch quarantine over a Rust Merkle DAG:

- computing descendant closure from `parent_hash` topology;
- excluding an entire quarantined branch before prompt assembly;
- preserving the rejected branch as cold forensic evidence;
- rejecting real, native, lineage-valid hashes when they belong to the wrong branch.

It does not claim that Rust `gc_tombstone` automatically tombstones descendants, does not delete evidence, and does not prove literal infinite memory or production readiness.

## Null Hypotheses

1. A poison branch cannot be isolated without replaying every narrative node.
2. Quarantined branch hashes still leak into operational hard-anchor context.
3. Branch pruning either deletes evidence or makes the rejected branch unauditable.
4. Any native hash with valid lineage is safe to admit into active context.
5. A model auditor confuses signed/native branch hashes with active semantic validity.

## Alternative Hypotheses

1. Parent-hash topology is enough to compute exact descendant closure for branch quarantine.
2. Policy-level closure pruning removes the quarantined branch before prompt assembly.
3. The rejected branch remains cold-auditable by parent-hash lineage while inactive operationally.
4. A real native hash must still be rejected if it belongs to a quarantined branch closure.
5. The auditor distinguishes cryptographic reality from policy-active branch membership.

## Default Gates

- Depth: 5,000 Rust Merkle DAG nodes.
- Branch depth: 4 nodes per branch.
- Narrative payload: 2,048 bytes per node.
- Repeats: 7.
- Active hard-anchor context must pass native identity-lane verification.
- Quarantined branch hashes and exact poison policy values must be absent from operational context.
- The safe branch terminal hash must remain active.
- The poison branch must remain lookupable and parent-chain auditable as cold evidence.
- The wrong-branch attack hash must be native and lineage-valid, but rejected by branch quarantine.

## Claim Boundary

Passing this suite supports the narrower claim that HeliX can apply policy-layer branch pruning over a native Merkle DAG before prompt assembly while retaining cold evidence. It does not prove automatic descendant tombstoning, physical deletion, global consensus, perfect semantic memory, or production safety.
