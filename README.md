# HeliX Inference OS

**HeliX wraps stochastic model calls in a deterministic evidence cage.**

It records and preserves:

- requested model
- provider-returned model
- prompt/output digests
- signed receipts
- Merkle lineage
- canonical head
- replayable artifacts
- falsifiable claim boundaries

HeliX is not a single model and not just a chat wrapper. It is a local shell, memory substrate, audit ledger, replay system, and verification lab for multi-model agent pipelines.

## Start Here

- Researchers: [CLAIMS.md](CLAIMS.md), [THREAT_MODEL.md](THREAT_MODEL.md), and the curated public layer under [evidence/](evidence/README.md)
- Practitioners: [docs/provider-model-audit.md](docs/provider-model-audit.md), [REPRODUCING.md](REPRODUCING.md), and `tools/helix_replay.py`
- Auditors: [CLAIMS.md](CLAIMS.md), [THREAT_MODEL.md](THREAT_MODEL.md), [NULL_RESULTS.md](NULL_RESULTS.md), and the raw reviewer notes in [verification/README-reviewer.md](verification/README-reviewer.md)

## What HeliX Is

HeliX is a **deterministic evidence cage around stochastic model calls**:

- **Core**: signed receipts, hashes, replay, Merkle lineage, canonical head, quarantine
- **Infra**: memory catalog, state server, concurrent agent writes, bounded retrieval
- **Audit**: requested vs actual model metadata, artifact replay, claim linting, transcript preservation
- **Research**: lineage forgery, contamination, multi-agent concurrency, long-horizon checkpoint and infinite-depth memory gauntlets
- **Product direction**: black-box recorder for AI agents, Git-like memory branching, future visual audit UI

What HeliX does **not** claim by default:

- semantic truth from hashes
- hidden provider identity beyond returned metadata
- provider intent
- global transparency-log guarantees
- universal memory-quality wins across every benchmark lane

## Canonical Public Evidence

The public entrypoint is now the curated layer under [evidence/](evidence/README.md):

- [evidence/index.json](evidence/index.json)
- [CLAIMS.md](CLAIMS.md)
- [THREAT_MODEL.md](THREAT_MODEL.md)
- [NULL_RESULTS.md](NULL_RESULTS.md)
- [REPRODUCING.md](REPRODUCING.md)

`verification/` still exists, but it is the **raw lab/archive tree**: historical artifacts, manifests, preregistrations, scratch runs, viewer assets, and suite outputs. Public claims should start from `evidence/`, not from ad hoc files in `verification/`.

## Current Anchor Claims

These are the main public claims HeliX stands behind in this repo state:

1. **Provider-returned model mismatch is auditable**
   HeliX preserves `requested_model`, `actual_model`, digests, latency, and lineage so metadata mismatches can be replayed and inspected later.

2. **A valid chain does not imply authentic lineage**
   HeliX distinguishes integrity of an append-only chain from selection of the canonical branch.

3. **Local core trust is verifiable**
   Signed receipts, local signed head checkpoints, quarantine, and exportable proofs provide local auditability for a workspace thread.

4. **Deep-memory overhead evidence is supportive, not universal**
   The low-overhead live memory result is kept as experimental/supportive evidence until broader replication and stronger baselines are restored.

Secondary bounded lanes, such as the multi-agent concurrency methodology suite, stay public under `evidence/experimental/` with narrower wording than the anchor claims above.

See [CLAIMS.md](CLAIMS.md) for wording, tiers, falsifiers, and scope boundaries.

## Repo Layers

Read the repo in this order:

1. **Product shell**
   `helix.cmd`, `src/helix_proto/helix_cli.py`, `src/helix_proto/api.py`
2. **Deterministic evidence cage**
   `signed_receipts.py`, `artifact_replay.py`, `evidence_ingest.py`, `verification_hardening.py`, `schemas/`
3. **Verifiable memory + lineage**
   `helix_kv/memory_catalog.py`, `helix_kv/merkle_dag.py`, `crates/helix-merkle-dag/`, `crates/helix-state-server/`
4. **Stochastic model boundary**
   `provider_audit.py`, `agent.py`, `tools.py`, model/router glue
5. **Evidence / compliance record**
   `evidence/`, `verification/`, claim docs, replay tooling
6. **Research gauntlets**
   `tests/test_v4_*.py`, suite tests, `tools/run_*_suite_v1.py`, `verification/nuclear-methodology/`
7. **Future product surface**
   `crates/helix-watch/`, `verification/viewer/`, future `frontend/` / `web/`

## CLI Quickstart

Run the shell from the repo root:

```cmd
helix
```

or:

```cmd
python -m helix_proto.helix_cli interactive
```

Useful commands:

```text
/help
/status
/model list
/router why TEXT
/mode list
/tech TEXT
/explore TEXT
/memory QUERY
/verify latest
/trust current
```

Install the launcher into the current Python environment:

```cmd
tools\install_helix_cli.cmd
```

## Replay and Reproduction

Verify an artifact offline:

```powershell
python tools\helix_replay.py --mode verify-only --artifact verification\local-v4-lineage-forgery-gauntlet.json
```

Run the public claim lint:

```powershell
python tools\helix_claim_lint.py --public-evidence evidence\index.json --public-docs .
```

See [REPRODUCING.md](REPRODUCING.md) for the current public claim batch and token-required live suites.

## Secondary Research Tracks

HeliX still contains broader and older research lanes, including hybrid memory/runtime-cache work, Zamba local studies, and performance-oriented experiments. They remain in the repo, but they are **not** the primary public identity of HeliX in this pass.

Start those lanes here if you need them:

- [docs/claims-matrix.md](docs/claims-matrix.md)
- [docs/provider-model-audit.md](docs/provider-model-audit.md)
- [docs/inference-os-architecture.md](docs/inference-os-architecture.md)
- `verification/` for raw historical artifacts and viewer assets
