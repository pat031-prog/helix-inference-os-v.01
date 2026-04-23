# HeliX Public Claims

This is the canonical public claim registry for the current HeliX repo state.

Public evidence starts from:

- [evidence/index.json](evidence/index.json)
- [THREAT_MODEL.md](THREAT_MODEL.md)
- [NULL_RESULTS.md](NULL_RESULTS.md)
- [REPRODUCING.md](REPRODUCING.md)

## Anchor Claims

| Claim | Tier | Status | Public evidence | Falsifier | Threat model scope |
| --- | --- | --- | --- | --- | --- |
| Provider-returned model mismatch is auditable | B | `empirically_observed` | [provider-returned-model-audit](evidence/empirical-observations/provider-returned-model-audit.json) | `requested_model == actual_model` for the cited probes, missing returned metadata, or a broken replay trail | Provider-returned metadata only. No claim about provider intent, hidden identity, SLA breach, or behavior outside recorded runs. |
| Valid chain does not imply authentic lineage | A | `mechanics_verified` | [lineage-vs-integrity](evidence/rigorous/lineage-vs-integrity.json) | Recall drops below the documented bound, false-positive rate exceeds the documented bound, or canonical/quarantine semantics fail to separate forged vs legitimate lineage | Local fixture/mechanics claim. Excludes compromised production keys and does not claim adaptive attacker coverage unless explicitly added. |
| Local core trust is verifiable | A | `mechanics_verified` | [local-core-trust-checkpoints](evidence/rigorous/local-core-trust-checkpoints.json) | Receipt tamper is accepted, checkpoint verification fails silently, quarantine no longer isolates non-canonical branches, or exported local proofs stop verifying | Local workspace trust only. No Rekor/CT anchoring, no global non-equivocation, no semantic truth guarantee, no hidden provider identity proof. |
| Bounded context under deep stores is demonstrated | A | `mechanics_verified` | [infinite-depth-memory](evidence/rigorous/infinite-depth-memory.json) | Retrieval no longer stays bounded under the documented depth/budget settings, or the claim boundary expands to "literal infinite memory" | Bounded-context mechanics claim only. Not literal infinite memory and not a universal latency claim. |
| Memory context can improve task recall with low search overhead | C | `experimental_supportive` | [external-memory-overhead](evidence/experimental/external-memory-overhead.json) | Memory-on no longer beats baseline in the cited live run, or the overhead bound is not reproducible within the recorded setup | Single live run with network-latency denominator. Supportive evidence only until wider replication and greener current suites. |

## Secondary Bounded Lanes

These lanes are public and reproducible, but they are not the top-line identity of HeliX in this pass.

| Lane | Tier | Status | Public evidence | Public wording |
| --- | --- | --- | --- | --- |
| Concurrent agent stale-parent race preservation | B | `bounded_methodology_verified` | [multi-agent-concurrency](evidence/experimental/multi-agent-concurrency.json) | Cite as a local methodology lane showing canonical-head survival, quarantine, and explicit merge-by-hash. Do not cite it as distributed consensus, production multi-writer correctness, or universal agent-swarm proof. |

## Wording Rules

- Say **"auditable provider-returned mismatch"**, not "proved provider deception".
- Say **"valid chain does not imply authentic lineage"**, not "HeliX solves lineage forever".
- Say **"local signed receipts and local signed head checkpoints"**, not "global transparency log".
- Say **"bounded context under deep stores"**, not "infinite memory".
- Say **"supportive live overhead evidence"**, not "universal low-overhead memory benchmark".

## Public Reading Order

1. Read this file.
2. Read [THREAT_MODEL.md](THREAT_MODEL.md).
3. Inspect [evidence/index.json](evidence/index.json).
4. Check [NULL_RESULTS.md](NULL_RESULTS.md) before making broad claims.
5. Use [REPRODUCING.md](REPRODUCING.md) for commands and current status snapshots.
