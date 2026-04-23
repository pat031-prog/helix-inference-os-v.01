# HeliX Detailed Claims Appendix

This file is a **detailed appendix**, not the canonical public starting point.

Start public review here instead:

- [../CLAIMS.md](../CLAIMS.md)
- [../THREAT_MODEL.md](../THREAT_MODEL.md)
- [../NULL_RESULTS.md](../NULL_RESULTS.md)
- [../REPRODUCING.md](../REPRODUCING.md)
- [../evidence/index.json](../evidence/index.json)

This appendix keeps broader or secondary repo lanes visible without making them the top-line identity of HeliX.

## Primary Anchor Claims

| Claim | Public wording | Canonical public evidence | Caveat |
| --- | --- | --- | --- |
| `local-core-trust-checkpoints` | HeliX records local signed receipts, local signed head checkpoints, canonical lineage state and quarantine so a thread can be audited locally. | [local-core-trust-checkpoints](../evidence/rigorous/local-core-trust-checkpoints.json) | Local workspace trust only. |
| `provider-returned-model-audit` | HeliX preserves requested model, provider-returned model, digests and lineage so metadata mismatches are auditable. | [provider-returned-model-audit](../evidence/empirical-observations/provider-returned-model-audit.json) | Provider-returned metadata only; no hidden identity or intent claim. |
| `lineage-vs-integrity` | HeliX distinguishes structurally valid chains from canonical lineage. | [lineage-vs-integrity](../evidence/rigorous/lineage-vs-integrity.json) | Strong local mechanics claim, not a universal adversarial proof. |
| `infinite-depth-memory` | HeliX demonstrates bounded context construction under deep stores. | [infinite-depth-memory](../evidence/rigorous/infinite-depth-memory.json) | Not literal infinite memory. |

## Secondary and Historical Lanes

These remain in the repo and may be useful for deeper readers, but they are not the top-level public story in this pass.

| Lane | Representative evidence | Current boundary |
| --- | --- | --- |
| Hybrid runtime cache / Zamba local | [`verification/hybrid-memory-frontier-summary.json`](../verification/hybrid-memory-frontier-summary.json) | Secondary research track; not primary public identity. |
| Transformer GPU compression | [`verification/remote-transformers-gpu-summary.json`](../verification/remote-transformers-gpu-summary.json) | Measured model set only. |
| Session OS / local control plane | [`verification/local-session-catalog-smoke.json`](../verification/local-session-catalog-smoke.json) | Control-plane evidence, not semantic quality. |
| Rust playback / watch UI | [`verification/local-zamba2-stress-dashboard.json`](../verification/local-zamba2-stress-dashboard.json) | Playback surface only. |

## Experimental Methodology Lanes

| Lane | Why kept | Public boundary |
| --- | --- | --- |
| Ghost continuity and contamination | Useful to study memory contamination and adjudication | Experimental/supportive only; not a general memory benchmark. |
| Identity and self-description gauntlets | Useful for prompt-boundary and claim-hygiene work | Do not cite as consciousness or semantic truth evidence. |
| Ouroboros / protocol sketches | Useful as design or methodology notes | Not the current storage core unless current code/tests back it. |

## Raw Artifact Canonicality

Historical and raw artifacts still live in `verification/`, but the canonical public navigation surface is now:

- `evidence/index.json`
- the bucket manifests under `evidence/`
- the root claim/boundary docs
