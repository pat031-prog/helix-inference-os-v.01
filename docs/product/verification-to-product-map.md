# Verification To Product Map

This map turns HeliX verification artifacts into commercial runtime features.

## Principle

The verification folder should not be a museum of JSON artifacts. It should feed task behavior.

Each strong idea becomes one of three things:

- a flow profile
- a task capsule check
- a lab profile

## Protocol Map

| Verification Idea | Product Feature | User-Facing Mode |
| --- | --- | --- |
| hard-anchor-utility | exact references without context bloat | `doc-grounded` |
| branch-pruning-forensics | preserve bad branches without polluting context | `multi-review`, `deep-lab` |
| cognitive-drift-rollback | signed mistakes become navigable fences | `web-recursive`, `resilient-task` |
| rollback-fence-replay | failed attempts are excluded from retry context | `patch-safe`, `resilient-task` |
| signed-poison-invariant | signed data is not automatically trusted | `doc-grounded`, `privacy-swarm` |
| multi-agent-concurrency | reconcile competing agents and stale parents | `multi-review` |
| long-horizon-checkpoints | resumable tasks with bounded continuation | `deep-lab` |
| provider-substitution | track provider/model identity drift | `multi-review`, `deep-lab` |
| meta-microsite blueprint | build a product artifact about HeliX itself | `web-recursive` |
| hybrid-research blueprint | local privacy shield plus cloud reasoning | `privacy-swarm` |
| resilient-pipeline blueprint | fail closed, swap agent, retry safely | `resilient-task` |

## Flow Profiles

### web

Purpose: build static pages, explainers and editorial demos quickly.

Guarantees: sandbox provenance, captured patch, human trust card.

### web-recursive

Purpose: build a page, review it, refine it and preserve the loop.

Guarantees: bounded recursion, drift/claim warnings, balanced verification.

### patch-safe

Purpose: code edits that should not touch the real repo until accepted.

Guarantees: sandbox, patch hash, apply gate.

### doc-grounded

Purpose: local document analysis with exact file/anchor discipline.

Guarantees: compact grounding, no full-history dump, source poison guard.

### privacy-swarm

Purpose: sensitive local material, anonymized handoff and redaction-aware output.

Guarantees: minimization, local claim boundary, no secret echo by design.

### resilient-task

Purpose: longer or brittle tasks where failures must not contaminate retry context.

Guarantees: rollback fence, failed attempt preserved, strict local checks.

### multi-review

Purpose: architecture, product and release decisions with visible disagreement.

Guarantees: model roles, disagreement record, provider audit hook.

### deep-lab

Purpose: explicit evidence runs and public claim support.

Guarantees: no accidental foreground deep scan, nuclear suites are opt-in.

## Claim Ladder

HeliX should keep public wording bounded.

Safe claims:

- HeliX captured a patch from a sandbox run.
- HeliX recorded the engine, changed files and patch hash.
- HeliX preserved claim boundaries and warnings.
- HeliX can run deeper verification when explicitly requested.

Avoid claims:

- HeliX proves a model answer is true.
- HeliX guarantees bug-free patches.
- HeliX provides distributed consensus.
- HeliX has literal infinite memory.

## Next Runtime Integration

1. Keep `/flow list`, `/flow show` and `/flow run` aligned with this map.
2. Attach source protocol IDs to every task capsule.
3. Add `flow_profile` and `protocols_used` to trust cards.
4. Let `/trust last` render protocols as human cards.
5. Let `/verify last --level balanced` select lab checks by flow profile.
