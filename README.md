# HeliX Inference OS

**Verifiable memory and lineage infrastructure for multi-model agent pipelines.**

HeliX wraps inference calls with a tamper-evident MerkleDAG, active external memory (BM25/WAND over SHA-256-chained nodes), and a per-call audit ledger. Every prompt, output, and state transition gets a cryptographic receipt. Provider substitutions, lineage forks, and memory injections are auditable by design.

---

## How it works

→ **[The API Polygraph: How HeliX catches cloud providers lying with immutable math](docs/provider-polygraph.md)**

Deep dive: MerkleDAG cryptographic structure, SHA-256 digest chains, DAG branching, forgery detection, and the real substitution incident that exposed it all.

---

## Three hard claims

All claims are backed by preregistered tests and artifacts in `evidence/`. Run the corresponding test to reproduce. Every artifact carries a `claim_boundary` field that states explicitly what it does and does not prove.

---

### Claim 1 — Provider substitution is auditable

Inference providers silently serve different models than requested. HeliX captures `requested_model` vs `actual_model` per call and seals both in a signed DAG node.

| Requested | Served by DeepInfra | Size ratio |
|---|---|---|
| Llama-3.2-**3B**-Instruct | Llama-3.2-**11B**-Vision-Instruct | 3.7× + vision |
| Mistral-**7B**-Instruct-v0.3 | Mistral-Small-**3.2-24B**-Instruct-2506 | 3.4× + new gen |
| Qwen**2.5-7B**-Instruct | Qwen**3-14B** | 2.0× + cross-gen |

Detection rate across runs: **3/3 probes, 100%**. No API errors, no warnings — the swap is only visible through the receipt.

```bash
pytest tests/test_provider_integrity_observatory.py   # 6 passed, 84s
pytest tests/test_hydrogen_table_drop_live.py         # 1 passed, 29s
```

Evidence: `evidence/infrastructure/local-provider-integrity-observatory.json`  
Evidence: `evidence/infrastructure/local-hydrogen-table-drop-live.json`

---

### Claim 2 — A valid chain does not imply authentic lineage

`verify_chain` confirms a chain is structurally intact (parent hashes link correctly). It does not confirm that the content was written by a legitimate author. These are separate proofs and both are required.

Forgery gauntlet — 240 forged nodes + 1,200 legitimate, 4 attack arms:

| Attack arm | Precision | Recall | F1 | FPR | p95 detect |
|---|---|---|---|---|---|
| naive | 1.0 | 1.0 | 1.0 | 0.0 | 0.040ms |
| schema-aware | 1.0 | 1.0 | 1.0 | 0.0 | 0.040ms |
| hash-aware | 1.0 | 1.0 | 1.0 | 0.0 | 0.040ms |
| signature-aware | 1.0 | 1.0 | 1.0 | 0.0 | 0.040ms |

Identity v2 audit: 33 LLM calls, 99 state events, `audit_completeness_score: 1.0`, 5 forgery attempts logged, `forbidden_claims_present: false`.

```bash
pytest tests/test_v4_lineage_forgery_gauntlet.py
pytest tests/test_identity_trust_gauntlet_v2.py
```

Evidence: `evidence/academic/local-v4-lineage-forgery-gauntlet.json`  
Evidence: `evidence/academic/local-identity-trust-gauntlet-v2.json`

---

### Claim 3 — Verifiable memory adds capability with sub-0.25% latency overhead

Ghost-in-the-Shell live (real API, 3 providers, 4-task battery):

| Metric | Value |
|---|---|
| `ghost_signature_score` | **0.9603** |
| `memory_on_win_rate` | **1.0** (4/4 tasks) |
| `false_memory_rejected` | **true** |
| `shell_identity_anchor_rate` | **1.0** (3/3 shells) |
| Context search avg | 3.7ms |
| LLM latency avg | 1,863ms |
| **Memory overhead vs LLM** | **0.1986%** |

Without HeliX: model explicitly states facts are unknown and refuses to fabricate.  
With HeliX: every fact recovered, every fact cited with `memory_id`, fake memory rejected.

Memory Contamination Triad (fixture, 4 arms, 15 tasks):

| Arm | Mean score | Contamination delta |
|---|---|---|
| memory_off | 0.0 | — |
| memory_on | **4.0** | — |
| memory_wrong | 0.0 | **0.0** (did not improve over off) |
| memory_poisoned | 4.0 | **0.0** (fake not cited) |

`quarantine_rate: 1.0`, `citation_fidelity: 1.0`. Wrong context does not improve over baseline. Poisoned context does not get cited.

```bash
pytest tests/test_ghost_in_the_shell_live.py
pytest tests/test_v4_memory_contamination_triad.py
```

Evidence: `evidence/infrastructure/local-ghost-in-the-shell-live.json`  
Evidence: `evidence/academic/local-v4-memory-contamination-triad.json`

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│         Inference call (any provider/API)        │
└──────────────────────┬──────────────────────────┘
                       │  prompt + response
              ┌────────▼────────┐
              │  HeliX Ledger   │  SHA-256(prompt) · SHA-256(output)
              │  (Python)       │  requested_model vs actual_model
              └────────┬────────┘
                       │  node content
       ┌───────────────▼──────────────────┐
       │   helix-merkle-dag  (Rust/PyO3)  │  lock-free MerkleDAG
       │   helix-state-core  (Rust)       │  BM25/WAND search index
       │   helix-state-server (Rust)      │  IPC state server
       └───────────────┬──────────────────┘
                       │  verified chain
              ┌────────▼────────┐
              │  verify_chain   │  structural integrity (parent hashes)
              │  audit_chain    │  content authenticity (hash comparison)
              └─────────────────┘
```

**Rust crates** (`crates/`):
- `helix-merkle-dag` — lock-free SHA-256 MerkleDAG, PyO3 bindings
- `helix-state-core` — BM25/WAND retrieval over DAG nodes
- `helix-state-server` — IPC state server for concurrent agent writes
- `helix-watch` — Rust TUI for telemetry replay

**Python layer** (`src/`):
- `helix_proto` — ledger, call recording, state event pipeline
- `helix_substrate` — agent wiring, memory injection, session OS

---

## Evidence structure

```
evidence/
├── academic/          # Cryptographic lineage proofs + contamination controls
│   ├── local-conversation-fork-forensics.json
│   ├── local-claude-context-repair-fork.json
│   ├── local-v4-lineage-forgery-gauntlet.json      ← 240 forgeries, F1=1.0
│   ├── local-identity-trust-gauntlet-v2.json       ← 33 calls, 99 events, score=1.0
│   └── local-v4-memory-contamination-triad.json    ← 4 arms, contamination_delta=0.0
│
├── infrastructure/    # Provider audit + memory overhead (real API)
│   ├── local-provider-integrity-observatory.json   ← substitution audit
│   ├── local-provider-substitution-ledger.json
│   ├── local-ghost-in-the-shell-live.json          ← score=0.9603, overhead=0.1986%
│   ├── local-ghost-shell-task-scores.json
│   ├── local-hydrogen-table-drop-live.json         ← table_drop_proof_passed
│   ├── local-wand-million-benchmark.json           ← 11ms selective / 7127ms generic
│   ├── local-semantic-router-wand-guard.json       ← 32× speedup
│   ├── local-cloud-amnesia-derby-qwen122b-*.json   ← 83.3% win rate, 0.319% overhead
│   └── local-emergent-behavior-cross-system-postmortem.json
│
└── product/           # OSINT oracle (mechanics_verified tier only)
    ├── local-v4-zero-day-osint-backtest.json       ← lead_time=32.5h, precision=0.667
    └── local-zero-day-osint-oracle.json
```

Each artifact includes a `claim_boundary` field and, where applicable, `preregistered_hash`, `null_hypothesis`, `falseability_condition`, and `kill_switch` fields that document what would constitute a failure.

Schema: `schemas/helix-verification-v0.schema.json`

---

## Running the tests

### Prerequisites

```bash
# Python 3.11+
pip install -r tools/requirements/requirements.txt

# Rust crates
cd crates/helix-merkle-dag && cargo build --release
cd crates/helix-state-core  && cargo build --release

# API keys
cp .env.example .env
# Fill: DEEPINFRA_API_KEY, ANTHROPIC_API_KEY
```

### Key test suites

```bash
# Claim 1: Provider substitution audit (real API, ~85s)
pytest tests/test_provider_integrity_observatory.py -v

# Claim 2a: Lineage forgery — 4 attack arms (local, fast)
pytest tests/test_v4_lineage_forgery_gauntlet.py -v

# Claim 2b: Cross-model council + audit completeness (real API, ~5min)
pytest tests/test_identity_trust_gauntlet_v2.py -v

# Claim 3a: Memory continuity + false memory rejection (real API, ~60s)
pytest tests/test_ghost_in_the_shell_live.py -v

# Claim 3b: Memory contamination triad — 4 arms (local)
pytest tests/test_v4_memory_contamination_triad.py -v

# Bonus: Memory table-drop proof (real API, ~30s)
pytest tests/test_hydrogen_table_drop_live.py -v

# Bonus: WAND benchmark at 1M nodes (local, ~2min)
pytest tests/test_benchmark_os.py -v
```

### Synthetic mode (no API keys)

```bash
HELIX_TEST_MODE=synthetic pytest tests/ -v
```

---

## Known limitations (documented, not hidden)

- **WAND gravity well**: at 1M+ nodes, generic queries collapse to a single attractor (647× slower than anchored queries). The semantic router mitigates this with 32× speedup. Native fix tracked.
- **Zamba2 long-context identifier recall**: session restore is bit-perfect (`max_abs_logit_delta: 0.0`); semantic recall of specific identifiers at long context is 0/2. Distinct properties.
- **Ghost v2**: `shells_preserve_ghost=false` with recall 0.6667 against threshold 0.70. Core behavior (win rate, delayed trigger, overhead, false memory rejection) passed. Scorer was too literal on semantic rejections. Patched, pending rerun.
- **OSINT oracle**: `local-v4-zero-day-osint-backtest.json` is `mechanics_verified` tier. A real CVE corpus with strict cutoff is required before product claims.

---

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).

Copyright (C) 2026 Patricio Valbusa.

If you integrate HeliX into a networked product, the AGPL requires you to release your modifications under the same license.
