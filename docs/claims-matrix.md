# HeliX Claims Matrix

The claims matrix is the public wording guardrail for HeliX Memory. It keeps the story strong without overclaiming: every claim has a status, public wording, evidence artifacts, and a caveat.

Canonical artifact:

- [`verification/helix-claims-matrix.json`](../verification/helix-claims-matrix.json)

## Statuses

| Status | Meaning |
| --- | --- |
| `verified` | Backed by local or remote artifacts and safe to say publicly with the listed caveat. |
| `promising` | Interesting signal, but too small, local, or incomplete for a strong public claim. |
| `blocked` | Known limitation, reproduction issue, hardware boundary, or lane that should not be used as evidence yet. |

## Verified Claims

| Claim | Public wording | Evidence | Caveat |
| --- | --- | --- | --- |
| `local-core-trust-checkpoints` | HeliX records local signed receipts, local signed head checkpoints, canonical lineage state, equivocation quarantine and exportable session proofs. | [`tests/test_memory_catalog.py`](../tests/test_memory_catalog.py), [`tests/test_v4_signed_receipts.py`](../tests/test_v4_signed_receipts.py) | Local workspace trust only. This is not Rekor/CT anchoring, global non-equivocation, semantic truth, or hidden provider identity proof. |
| `provider-returned-model-audit` | HeliX records requested model, provider-returned model, prompt/output digests, latency and lineage so model mismatches are auditable. | [`local-provider-substitution-ledger-20260418-154218.json`](../verification/local-provider-substitution-ledger-20260418-154218.json), [`local-hydrogen-table-drop-live-20260418-153210.json`](../verification/local-hydrogen-table-drop-live-20260418-153210.json) | The detector is `requested_model != actual_model`; this does not prove provider intent, hidden identity, or behavior outside the recorded run. |
| `lineage-vs-integrity` | HeliX distinguishes structurally valid chains from canonical lineage, so valid later inserts do not automatically replace the canonical branch. | [`local-v4-lineage-forgery-gauntlet.json`](../verification/local-v4-lineage-forgery-gauntlet.json), [`tests/test_memory_catalog.py`](../tests/test_memory_catalog.py) | Local fixture plus unit-level canonical/quarantine coverage; external adversarial review is still future work. |
| `transformer-kv-gpu` | HeliX compresses Transformer KV cache on the verified GPU suite while preserving generated-token match against the baseline. | [`remote-transformers-gpu-summary.json`](../verification/remote-transformers-gpu-summary.json) | Limited to the measured Qwen/SmolLM Transformer suite. |
| `hybrid-runtime-cache` | On local Zamba2, the strongest hybrid runtime result is architecture-aware: KV compression where KV exists plus recurrent-state compression where Mamba dominates. | [`hybrid-memory-frontier-summary.json`](../verification/hybrid-memory-frontier-summary.json), [`local-zamba2-hxq-vs-vanilla-summary.json`](../verification/local-zamba2-hxq-vs-vanilla-summary.json) | CPU-local Zamba2-1.2B result; GPU hybrid validation is still next. |
| `hybrid-session-integrity` | HeliX can serialize and restore a complete hybrid Zamba2 session with bit-perfect SHA-256 snapshot integrity. | [`local-zamba2-stress-state-juggler.json`](../verification/local-zamba2-stress-state-juggler.json), [`local-zamba2-stress-state-juggler-session.json`](../verification/local-zamba2-stress-state-juggler-session.json) | Hash matching proves snapshot integrity; it does not prove semantic understanding by itself. |
| `restore-equivalence` | Under the local restore-equivalence probe, deterministic continuation from the restored hybrid state matched the pre-restore continuation. | [`local-zamba2-stress-restore-equivalence.json`](../verification/local-zamba2-stress-restore-equivalence.json), [`local-zamba2-stress-restore-equivalence-session.json`](../verification/local-zamba2-stress-restore-equivalence-session.json) | Short deterministic probe, not a broad semantic or long-context benchmark. |
| `adaptive-promotion-safety` | On an adversarial context-switch prompt, HeliX promoted blocks to higher precision and kept logits finite instead of forcing INT4 at all costs. | [`local-zamba2-stress-context-switcher.json`](../verification/local-zamba2-stress-context-switcher.json), [`local-zamba2-stress-context-switcher.jsonl.gz`](../verification/local-zamba2-stress-context-switcher.jsonl.gz) | Ratio drops under stress; that is expected graceful degradation. |
| `rust-tui-playback` | `helix-watch` provides a Rust terminal playback surface for mission telemetry and compact receipts. | [`local-zamba2-stress-dashboard.json`](../verification/local-zamba2-stress-dashboard.json) | Playback-only in v1. |
| `session-os-catalog` | HeliX has a local Session OS catalog that indexes model/agent sessions with audit status, token hashes and parent-chain metadata. | [`local-session-catalog-smoke.json`](../verification/local-session-catalog-smoke.json) | Control-plane evidence; it does not imply semantic quality by itself. |
| `hybrid-prefix-checkpoint-v0` | HeliX can reuse an exact saved Zamba2 prefix checkpoint by restoring Transformer KV plus Mamba recurrent state, then computing only the suffix. | [`local-hybrid-prefix-checkpoint-summary.json`](../verification/local-hybrid-prefix-checkpoint-summary.json) | Exact checkpoint only; no arbitrary Mamba slicing, and the local claim is top-1/finitude rather than bit-exact logits. |
| `session-os-memory-concurrency` | HeliX MemoryCatalog can preserve local concurrent agent observations and memories under the SQLite WAL pattern. | [`local-memory-catalog-concurrency.json`](../verification/local-memory-catalog-concurrency.json) | Local single-process/threaded stress test with one SQLite connection per worker; not a distributed database claim. |
| `session-os-memory-decay-selection` | Under a tight recall budget, HeliX keeps high-importance memories ahead of low-priority scratch memories. | [`local-memory-decay-selection.json`](../verification/local-memory-decay-selection.json) | Deterministic lexical and priority-based selection; not embedding-quality semantic retrieval. |
| `hlx-layer-chaos-integrity` | HeliX full verification blocks a corrupted `.hlx` layer slice before the injector returns it for layer execution. | [`local-hlx-layer-chaos.json`](../verification/local-hlx-layer-chaos.json) | Receipt-only remains a hot-path mode, not an integrity claim. |
| `hmem-agent-wiring` | HeliX agents can automatically observe tool calls into hmem, inject bounded memory context at session start, and query memory plus knowledge through one hybrid search surface. | [`local-hmem-wiring-smoke.json`](../verification/local-hmem-wiring-smoke.json) | Local wiring evidence with heuristic/model-optional observation compression; not embedding-grade retrieval. |

## Promising Claims

| Claim | Current signal | Why it is not verified yet |
| --- | --- | --- |
| `long-context-coder-quality` | Best staged runtime-cache reduction was `1.42x`, with a marginal speedup. | Identifier recall was `0/2`, so it should not be used as a quality claim. |
| `agentic-code-prompts` | The prompt suite suggests code/agentic prompts are a useful evaluation lane. | The evidence is small and local; it is not a general agent benchmark. |
| `session-os-prefix-reuse-transformer` | Exact-prefix Transformer reuse restored a cached prefix and computed only new follow-up tokens in local smoke; the compressed claim variant kept top-1 stable with finite logits. | Transformer-only and exact-prefix; the native-dense diagnostic is not bit-exact in the current local artifact. |
| `session-os-openai-api` | HeliX exposes a local OpenAI-shaped chat completions response with HeliX session lifecycle metadata. | First endpoint is non-streaming and local; broader framework compatibility still needs external client tests. |
| `session-branching-git-for-llms` | HeliX can represent shared base sessions plus per-agent deltas as a verified parent-chain. | Branching v1 is metadata plus avoided-rewrite estimate; physical token-slice storage is future work. |
| `agent-framework-openai-showcase` | A standard OpenAI-compatible client can call HeliX's local chat completions surface and receive session lifecycle metadata. | The showcase validates client plumbing, not model quality or streaming/tool-calling support. |
| `rust-python-layer-slice-soak` | A short local soak repeatedly saves and loads `.hlx` layer slices while tracking RSS and p95 load time. | A short soak is a regression guard, not a formal allocator proof. |
| `blueprint-meta-microsite-demo` | HeliX can run a Blueprint workload that coordinates agents, private `.hlx` state, shared `hmem` and a scheduler to produce a quality-first Meta Microsite artifact. | Fallback or mixed mode proves orchestration and renderer quality; real-model generation quality needs a no-fallback run. |

## Experimental Methodology Artifacts

| Artifact lane | Why kept | Public boundary |
| --- | --- | --- |
| Ghost / contamination runs | They probe retrieval contamination, doppelganger records and receipt-adjudication failure modes. | Experimental evidence only; do not cite as a general memory-quality benchmark. |
| Identity-trust gauntlets | They stress prompt boundaries, self-description and claim hygiene. | Do not cite as consciousness, identity, or semantic-truth evidence. |
| Ouroboros artifacts | They preserve useful protocol sketches and lineage-forensics ideas. | Methodology/design notes only; not the current storage core unless backed by current code/tests. |

## Blocked Claims

| Claim | Why blocked |
| --- | --- |
| `hxq-zamba2-local` | Local HXQ Zamba2 runs have shown loader/runtime or non-finite-logit issues. |
| `zamba2-27b-local` | Out of scope for the current 12GB RAM laptop profile. |
| `airllm-real-smoke` | Optional real AirLLM smoke is skipped unless AirLLM and a local compatible model are already present. |

## Required Wording

Use this exact distinction when presenting State Juggler:

> The hash match proves bit-perfect integrity of the serialized/restored snapshot. It does not prove semantic understanding by itself.

Use this exact distinction when presenting Restore Equivalence:

> The restore-equivalence probe shows that a short deterministic continuation from the restored hybrid state matched the pre-restore continuation. It is not a broad semantic evaluation.

Use this exact distinction when presenting Prefix Reuse:

> Transformer prefix reuse v0 is exact-prefix and Transformer-only. The compressed local claim is top-1 stable with finite logits, not bit-exact.

Use this exact distinction when presenting Hybrid Prefix Checkpoints:

> Hybrid prefix v0 restores exact saved checkpoint boundaries by loading Transformer KV plus Mamba recurrent state. It does not slice Mamba recurrent state at arbitrary token positions.

Use this exact distinction when presenting Core Trust:

> Local signed checkpoints prove the workspace key selected a canonical head for a thread. They do not prove semantic truth, global non-equivocation, provider intent, or hidden model identity.

That is the clean public framing: strong evidence, precise boundaries.
