# HeliX Inference OS

**Verifiable memory and lineage infrastructure for multi-model agent pipelines.**

HeliX wraps inference calls with a tamper-evident MerkleDAG, active external memory (BM25/WAND over SHA-256-chained nodes), and a per-call audit ledger. Every prompt, output, and state transition gets a cryptographic receipt. Provider substitutions, lineage forks, and memory injections are auditable by design.

---

## Three hard claims

All claims are backed by preregistered tests and artifacts in `evidence/`. Run the corresponding test to reproduce.

### Claim 1 — Provider substitution is auditable

Inference providers silently serve different models than requested. HeliX captures `requested_model` vs `actual_model` per call and signs both in a ledger node.

| Requested | Served by DeepInfra | Ratio |
|---|---|---|
| Llama-3.2-**3B**-Instruct | Llama-3.2-**11B**-Vision-Instruct | 3.7× |
| Mistral-**7B**-Instruct-v0.3 | Mistral-Small-**3.2-24B**-Instruct-2506 | 3.4× |
| Qwen**2.5-7B**-Instruct | Qwen**3-14B** | 2.0× |

Detection rate across 4 consecutive runs: **3/3 probes, 100%**.  
Artifact: `evidence/infrastructure/local-provider-integrity-observatory.json`

```bash
pytest tests/test_provider_integrity_observatory.py   # 6 passed, 84s
```

### Claim 2 — Lineage forgery is detected

A node with tampered content produces a different SHA-256 than the legitimate node at the same chain position. Multi-arm gauntlet (naive → schema-aware → hash-aware → signature-aware):

| Attack arm | Precision | Recall | F1 |
|---|---|---|---|
| naive | 1.0 | 1.0 | 1.0 |
| schema-aware | 1.0 | 1.0 | 1.0 |
| hash-aware | 1.0 | 1.0 | 1.0 |
| signature-aware | 1.0 | 1.0 | 1.0 |

Artifact: `evidence/academic/local-v4-lineage-forgery-gauntlet.json`

```bash
pytest tests/test_v4_lineage_forgery_gauntlet.py
```

### Claim 3 — Active memory overhead is under 0.2% of LLM latency

Measured against Qwen3.5-9B in Ghost-in-the-Shell live (real API, 4-task battery):

| Metric | Value |
|---|---|
| Memory-on win rate | **1.0** (4/4 tasks) |
| Context search avg | 3.7ms |
| LLM latency avg | 1863ms |
| Memory overhead vs LLM | **0.20%** |

Without HeliX context: score 0, model explicitly states facts are unknown.  
With HeliX context: score 8/8, every fact cited with `memory_id`.

Artifact: `evidence/infrastructure/local-ghost-in-the-shell-live.json`

```bash
pytest tests/test_ghost_in_the_shell_live.py
```

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
              │  verify_chain   │  parent_hash linkage
              │  audit_chain    │  completeness score = 1.0
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
├── academic/          # Cryptographic lineage proofs
│   ├── local-conversation-fork-forensics.json
│   ├── local-claude-context-repair-fork.json
│   └── local-v4-lineage-forgery-gauntlet.json
│
├── infrastructure/    # Provider audit + memory overhead
│   ├── local-provider-integrity-observatory.json
│   ├── local-provider-substitution-ledger.json
│   ├── local-ghost-in-the-shell-live.json
│   ├── local-hydrogen-table-drop-live.json
│   ├── local-wand-million-benchmark.json
│   ├── local-semantic-router-wand-guard.json
│   ├── local-cloud-amnesia-derby-qwen122b-quality-gated-*.json
│   └── local-emergent-behavior-cross-system-postmortem.json
│
└── product/           # OSINT oracle backtest (mechanics_verified tier)
    ├── local-v4-zero-day-osint-backtest.json
    └── local-zero-day-osint-oracle.json
```

Each artifact contains:
- `run_id` — timestamp-stamped unique identifier
- `prompt_digest` / `output_digest` — SHA-256 of inputs and outputs
- `requested_model` / `actual_model` — substitution evidence
- `claim_boundary` — explicit statement of what the artifact does and does not prove

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
# Provider substitution audit (real API, ~85s)
pytest tests/test_provider_integrity_observatory.py -v

# Memory continuity across shell swaps (real API, ~60s)
pytest tests/test_ghost_in_the_shell_live.py -v

# Lineage forgery — 4 attack arms (local)
pytest tests/test_v4_lineage_forgery_gauntlet.py -v

# Memory table-drop proof + forgery detection (real API, ~30s)
pytest tests/test_hydrogen_table_drop_live.py -v

# WAND benchmark at 1M nodes (local, ~2min)
pytest tests/test_benchmark_os.py -v

# Cloud amnesia derby — memory win rate (real API, ~3min)
pytest tests/test_cloud_amnesia_derby.py -v
```

### Synthetic mode (no API keys)

```bash
HELIX_TEST_MODE=synthetic pytest tests/ -v
```

---

## Known limitations (documented, not hidden)

- **WAND gravity well**: at 1M+ nodes, generic queries collapse to a single top-hit attractor (647× slower than anchored queries). The semantic router mitigates this with a 32× speedup. A native WAND upper-bound diagnostic is tracked for the next release.
- **Zamba2 long-context identifier recall**: session restore is bit-perfect (`max_abs_logit_delta: 0.0`); semantic recall of specific identifiers at long context is 0/2. These are separate properties.
- **OSINT oracle tier**: `local-v4-zero-day-osint-backtest.json` is at `mechanics_verified` tier — fixture validates metric computation and downgrade behavior. A real CVE corpus with strict cutoff is required before product claims.

---

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).

Copyright (C) 2026 Patricio Valbusa.

If you integrate HeliX into a networked product, the AGPL requires you to release your modifications under the same license.
