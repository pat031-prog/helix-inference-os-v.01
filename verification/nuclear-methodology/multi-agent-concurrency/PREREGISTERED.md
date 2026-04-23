# multi-agent-concurrency-v1

Question: Can HeliX preserve concurrent multi-agent work without losing or silently overwriting a stale-parent branch?

Null hypothesis: Under concurrent Alpha/Beta work from the same observed parent, HeliX either loses one branch or admits both branches as canonical context.

Alternative hypothesis: HeliX preserves both writes as evidence, keeps exactly one canonical head, quarantines stale-parent equivocation, and lets Gamma merge with explicit hashes.

Metrics:
- branch_preservation_rate
- canonical_head_stability
- quarantined_branch_count
- default_context_exclusion
- forensic_context_inclusion
- gamma_hash_citation_rate
- naive_baseline_lost_update_count

Falseability condition: If the stale-parent branch is not preserved, if it appears in default context, if canonical head is replaced by the stale branch, or if Gamma merges without citing both branch hashes, publish failure.

Kill-switch: If any branch content, prompt, API token, or private header leaks into public artifacts outside sanitized previews, abort public swarm claims.

Control arms:
- naive last-write-wins shared state
- default HeliX context without quarantined branches
- forensic HeliX context with quarantined branches

Threat model: local single-workspace MemoryCatalog plus DeepInfra/OpenAI-compatible model calls. Out of scope: distributed consensus, compromised local signing key, provider intent, hidden model identity, adaptive agents that can mutate HeliX internals.
