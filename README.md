# HeliX Inference OS

**Verifiable memory and lineage infrastructure for multi-model agent pipelines.**

HeliX wraps inference calls with a tamper-evident MerkleDAG, active external memory
(BM25/WAND over SHA-256-chained nodes), and a per-call audit ledger. Prompts,
outputs, provider-returned model metadata, retrieval events, and state transitions
are recorded as bounded evidence artifacts.

## 5-line technical summary

1. Detector: provider model mismatch is `requested_model != actual_model` from the API response.
2. HeliX does not infer hidden model identity; it preserves requested model, provider-returned model, digests, latency and lineage so mismatches are auditable.
3. Reproduce: `python -m pytest -q tests/test_provider_integrity_observatory.py`.
4. Limitation: mismatch evidence is not proof of bad-faith provider behavior or behavior outside the recorded run.
5. Evidence: start with [`verification/public-evidence-index.json`](verification/public-evidence-index.json) and [`verification/README-reviewer.md`](verification/README-reviewer.md).

## Three bounded claims

Every public claim below points to a test and a JSON artifact. The artifact's
`claim_boundary` or `public_claim_boundary` is part of the claim.

### Claim 1 - Provider-returned model metadata is auditable

HeliX records the model requested by the client and the model returned by an
OpenAI-compatible API response. If they differ, the mismatch is preserved with
call id, latency, prompt digest, output digest and run metadata.

| Requested model | Provider-returned model | Real run |
| --- | --- | --- |
| `meta-llama/Llama-3.2-3B-Instruct` | `meta-llama/Llama-3.2-11B-Vision-Instruct` | `provider-integrity-observatory-20260418-154218` |
| `mistralai/Mistral-7B-Instruct-v0.3` | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | `provider-integrity-observatory-20260418-154218` |
| `Qwen/Qwen2.5-7B-Instruct` | `Qwen/Qwen3-14B` | `provider-integrity-observatory-20260418-154218` |

```powershell
python -m pytest -q tests/test_provider_integrity_observatory.py
python -m pytest -q tests/test_v4_provider_substitution_longitudinal.py
```

Evidence:
[`verification/local-provider-integrity-observatory-20260418-154218.json`](verification/local-provider-integrity-observatory-20260418-154218.json),
[`verification/local-provider-substitution-ledger-20260418-154218.json`](verification/local-provider-substitution-ledger-20260418-154218.json)

Caveat: this is a provider-returned metadata audit. It does not prove a hidden
model identity, provider intent, or a contractual violation by itself.

### Claim 2 - Valid chain integrity is not the same as authentic lineage

`verify_chain` confirms parent-hash structure. It does not prove that a later
node is the authentic continuation of a branch. HeliX records both structural
integrity and lineage/authenticity checks.

Forgery gauntlet fixture: 240 forged nodes plus 1,200 legitimate nodes across
naive, schema-aware, hash-aware and signature-aware attack arms.

| Attack arm | Precision | Recall | F1 | False positive rate | p95 detection |
| --- | ---: | ---: | ---: | ---: | ---: |
| naive | 1.0 | 1.0 | 1.0 | 0.0 | 0.040 ms |
| schema-aware | 1.0 | 1.0 | 1.0 | 0.0 | 0.040 ms |
| hash-aware | 1.0 | 1.0 | 1.0 | 0.0 | 0.040 ms |
| signature-aware | 1.0 | 1.0 | 1.0 | 0.0 | 0.040 ms |

```powershell
python -m pytest -q tests/test_v4_lineage_forgery_gauntlet.py
```

Evidence:
[`verification/local-v4-lineage-forgery-gauntlet.json`](verification/local-v4-lineage-forgery-gauntlet.json)

Caveat: this artifact is `mechanics_verified`; it is a local/fixture lane, not an
external adversarial audit.

### Claim 3 - External memory can improve task performance with low retrieval overhead

In a real API run, HeliX external memory improved a 4-task battery while
recording citations and rejecting fake memory. Retrieval was small relative to
LLM latency in that run.

| Metric | Value |
| --- | ---: |
| `ghost_signature_score` | 0.9603 |
| `memory_on_win_rate` | 1.0 |
| `false_memory_rejected` | true |
| `shell_identity_anchor_rate` | 1.0 |
| Context search average | 3.7 ms |
| LLM latency average | 1,863 ms |
| Retrieval overhead vs LLM | 0.1986% |

```powershell
python -m pytest -q tests/test_ghost_in_the_shell_live.py
python -m pytest -q tests/test_v4_memory_contamination_triad.py
```

Evidence:
[`verification/local-ghost-in-the-shell-live-20260418-154420.json`](verification/local-ghost-in-the-shell-live-20260418-154420.json),
[`verification/local-v4-memory-contamination-triad.json`](verification/local-v4-memory-contamination-triad.json)

Caveat: memory can also hurt when raw retrieval is contaminated. HeliX keeps
negative findings visible. See the failed Ghost v2 run:
[`verification/local-ghost-in-the-shell-live-v2-20260418-160448.json`](verification/local-ghost-in-the-shell-live-v2-20260418-160448.json).

## Architecture

```text
Inference call
  -> HeliX audit ledger
       records prompt_digest, output_digest, requested_model, actual_model
  -> HeliX state server
       stores MerkleDAG nodes and BM25/WAND retrieval index
  -> Verification
       verify_chain for structure
       audit_chain / receipts / branch checks for authenticity claims
```

Rust crates:

- `helix-merkle-dag` - lock-free SHA-256 MerkleDAG and PyO3 bindings.
- `helix-state-core` - BM25/WAND retrieval over DAG nodes.
- `helix-state-server` - IPC state server for concurrent agent writes.
- `helix-watch` - terminal playback for telemetry and receipts.

Python layer:

- `helix_proto` - ledger, call recording and state event pipeline.
- `helix_substrate` - agent wiring, memory injection and session utilities.

## Reviewer entry points

- [`docs/provider-model-audit.md`](docs/provider-model-audit.md) - precise provider metadata audit wording.
- [`docs/reddit-response-notes.md`](docs/reddit-response-notes.md) - short public replies to common critiques.
- [`docs/claims-matrix.md`](docs/claims-matrix.md) - wording guardrails.
- [`verification/README-reviewer.md`](verification/README-reviewer.md) - artifact interpretation rules.
- [`verification/public-evidence-index.json`](verification/public-evidence-index.json) - canonical public evidence and SHA-256 hashes.

## Running tests

```powershell
python -m pytest -q tests/test_verification_hardening.py
python -m pytest -q tests/test_v4_provider_substitution_longitudinal.py
python tools/helix_claim_lint.py verification --scope batch-20260418
```

Real-cloud tests require a freshly rotated provider token and should be run
through the secure PowerShell wrappers in `tools/`, which prompt with hidden
input and do not persist tokens.

## License

GNU Affero General Public License v3.0 - see [LICENSE](LICENSE).

Copyright (C) 2026 Patricio Valbusa.
