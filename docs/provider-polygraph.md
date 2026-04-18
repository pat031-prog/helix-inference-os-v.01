# The API Polygraph: How HeliX Catches Cloud Providers Lying With Immutable Math

## The Incident That Started It All

We were running a test that required a lightweight model — specifically `meta-llama/Llama-3.2-3B-Instruct`. Small, fast, predictable. Then we saw this:

```
llm-0001  requested: meta-llama/Llama-3.2-3B-Instruct
          actual:    meta-llama/Llama-3.2-11B-Vision-Instruct
          latency:   7,004ms   ← 3.5× slower than expected
```

We requested a 3B model. We got an 11B Vision model. The provider didn't tell us. The API response came back as if nothing happened — no error, no warning, no indication of a swap. The only clue was a latency spike.

Because HeliX wraps every inference call with cryptographic receipts, we caught it. Without the receipt system, we would have never known.

We ran the same probe three more times on the same day:

| Run | Requested | Served | Latency |
|---|---|---|---|
| 030852 | Llama-3.2-**3B** | Llama-3.2-**11B**-Vision | 0.1ms (synthetic) |
| 031031 | Llama-3.2-**3B** | Llama-3.2-**11B**-Vision | 1,956ms |
| 031521 | Llama-3.2-**3B** | Llama-3.2-**11B**-Vision | 0.2ms (synthetic) |
| 031611 | Llama-3.2-**3B** | Llama-3.2-**11B**-Vision | 2,069ms |

**4 runs. 4 substitutions. 100% detection rate.**

It wasn't just Llama. On the same runs, Mistral was swapped from 7B to 24B, and Qwen from a 7B generation 2.5 model to a 14B generation 3 model — a cross-generation upgrade the provider performed silently.

We hadn't built an audit tool on purpose. We had accidentally built an **API Polygraph**.

---

## Why This Is Possible: The MerkleDAG Receipt System

HeliX works by wrapping every inference call inside a **MerkleDAG node** — a data structure borrowed from Git and blockchain, where every record is linked to the previous one by its cryptographic hash.

Here's the structure of a single call receipt:

```
Node {
  memory_id:      "llm-0001-provider-probe-llama"
  content:        {
    requested_model: "meta-llama/Llama-3.2-3B-Instruct",
    actual_model:    "meta-llama/Llama-3.2-11B-Vision-Instruct",  ← provider returned this
    prompt_digest:   SHA-256("Provider substitution probe. Reply..."),
    output_digest:   SHA-256("Provider Substitution Probe: Successful..."),
    latency_ms:      7004.965
  }
  node_hash:      SHA-256(content + parent_hash)
  parent_hash:    → previous node
  depth:          0
}
```

The `node_hash` is a SHA-256 of everything in the node plus the hash of whatever came before it. This means:

- **You cannot change the past.** If you edit any prior node's content — even a single byte — every downstream `node_hash` becomes invalid. The chain breaks.
- **Substitution becomes evidence.** When the provider returns a different `actual_model` than what was requested, that fact is sealed into the node. It cannot be quietly removed after the fact.
- **The receipt is provable to anyone.** The SHA-256 hash is deterministic. Give anyone the content and they can recompute the hash and verify it matches.

This is the same principle that makes Git commits tamper-evident: you can't rewrite history without changing all the commit hashes downstream.

---

## The Cryptographic Structure in Depth

### SHA-256 Digest Chains

Every prompt and every output gets its own SHA-256 digest before any inference happens:

```python
prompt_digest  = SHA-256("Provider substitution probe. Reply with a one-line receipt naming no secrets.")
# → "2721243aee497699d71d13c9e28f9215abe6ae39d08eba3842f55bcc478c1150"
```

This digest is computed **before** the API call. After the call:

```python
output_digest  = SHA-256("Provider Substitution Probe: Successful substitution...")
# → "2b721a3beb3b0bfb079321704f4c95a5f503117d0ee07667bf9aa5ad4424ee01"
```

Both digests are stored in the node. This gives us two independent proofs:

1. **Input equality proof**: the same `prompt_digest` sent to three different models proves they received identical inputs. If outputs differ, the difference is caused by the model, not the prompt.
2. **Output identity proof**: if the same `prompt_digest` produces the same `output_digest` across runs, it proves the same model processed the same input deterministically. If the output changes, either the model or its configuration changed.

In our substitution runs, we observed this exact effect:

```
Run 031031 — Llama 11B, prompt "2721243a..." → output "2b721a3b..."
Run 031611 — Llama 11B, prompt "2721243a..." → output "2b721a3b..."   ← identical
Run 031031 — Mistral 24B, same prompt       → output "c1982f49..."
Run 031611 — Mistral 24B, same prompt       → output "f7a24af1..."   ← different
```

Llama reproduced the same output across runs. Mistral didn't — because Mistral's served model changed between runs (from `Mistral-Small-24B-Instruct-2501` to `Mistral-Small-3.2-24B-Instruct-2506`). The hash divergence is direct cryptographic evidence of a hot-swap.

### The DAG: More Than a Chain

A blockchain is a **chain** — each block has one parent. HeliX uses a **Directed Acyclic Graph (DAG)**, which is more general: each node has one parent, but a parent can have multiple children. This enables branching:

```
Root node (verified)
    ├─ Branch A: Gemma council response   (depth 1)
    ├─ Branch B: Claude council response  (depth 1)
    ├─ Branch C: Qwen council response    (depth 1)
    └─ Branch X: Attacker's forged node   (depth 1) ← same parent, different hash
```

All four nodes point to the same root. All four are structurally valid chains of length 2. But Branch X was created with tampered content — so its `node_hash` differs from the hash of any legitimate node at that position. The forgery is detectable by comparing hashes, not by trusting content.

This is what `fake_edit_detected_by_hash: true` means in the Hydrogen test.

### The Audit Trail Has an Audit Trail

Every state operation — write, read, verify — generates its own state event:

```
state-0001  method: remember       role: state-recorder   → writes node
state-0002  method: verify_chain   role: state-auditor    → confirms chain_len=1, status=verified
state-0003  method: search         role: state-search     → retrieves node by query
state-0010  method: remember       role: attacker         → writes forged node
state-0011  method: verify_chain   role: state-auditor    → confirms structure (not content)
state-0012  method: audit_chain    role: state-auditor    → full traversal, detects forgery
```

The attacker's write is logged with `role: "attacker"`. The adversarial action becomes part of the immutable record. You cannot perform an attack without the attack being preserved as evidence.

---

## What Else This Stack Can Do

The cryptographic receipt system is the foundation. Built on top of it:

### Active External Memory

Models have no memory between calls by default. HeliX stores structured facts in the DAG and injects them at inference time. The memory overhead is under 0.2% of LLM latency — the retrieval takes ~3.7ms against calls that take ~1,800ms.

Without HeliX context:
> *"There is no record of an 'Incident HYDROGEN-77'. The specific terminology does not correspond to any documented event."*

With HeliX context:
> *"Recovered facts: helium-cache, vega-router, quartz-signer, rollback-ring-2. Citation: [hydrogen-root-omega]"*

Score delta: 0 → 6/6. Every fact recovered, every fact cited with its `memory_id`.

### Cross-Model Continuity

The same DAG node can be read by any model from any provider. A Gemma instance and a Claude instance can both read from and write to the same chain, maintaining continuity without sharing private API state. The continuity lives in the structure, not the model.

### Lineage Forgery Detection

Four attack patterns tested (naive → schema-aware → hash-aware → signature-aware). Across all four arms: precision 1.0, recall 1.0, F1 1.0.

A valid chain can be constructed from a forged node — the structure passes verification. But the hash of the forged node doesn't match the hash of any legitimate node in the known-good set. Structure validity and content authenticity are **separate proofs** that must both pass.

### BM25/WAND Retrieval at Scale

The search layer runs BM25 with WAND pruning in Rust over the full DAG. At 25,000 nodes, anchored queries return in ~11ms. The bottleneck at 1 million nodes is generic queries (those containing terms with very high document frequency), where the WAND posting list grows until traversal dominates. The semantic router detects these and diverts them to recency-based fallback — measured 32× speedup.

---

## The Polygraph Analogy

A polygraph doesn't read minds. It records measurable physiological signals — heart rate, skin conductance — and flags deviations from baseline. It doesn't know you're lying; it knows your signals changed.

HeliX works the same way. It doesn't know the provider made a bad-faith swap. It records `requested_model` and `actual_model`, and flags when they differ. The "lying" framing is the human interpretation of the delta.

The math is neutral. The receipts are immutable. The delta is yours to interpret.

---

## Reproducing the Evidence

```bash
# See the substitution events live
pytest tests/test_provider_integrity_observatory.py -v
# Evidence: evidence/infrastructure/local-provider-integrity-observatory.json

# See the memory on/off binary contrast
pytest tests/test_hydrogen_table_drop_live.py -v
# Evidence: evidence/infrastructure/local-hydrogen-table-drop-live.json

# Run the forgery gauntlet
pytest tests/test_v4_lineage_forgery_gauntlet.py -v
# Evidence: evidence/academic/local-v4-lineage-forgery-gauntlet.json
```

All artifacts include `claim_boundary` fields that state exactly what each artifact does and does not prove. The math is open. The evidence is in the repo. Run it yourself.
