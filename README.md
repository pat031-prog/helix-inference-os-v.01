# HeliX Inference OS

**A cryptographic agent shell for multi-model inference, local memory, evidence
replay, and certification suites.**

HeliX is not a single language model. It is the deterministic layer around
models: a CLI, router, memory subsystem, audit ledger, Merkle DAG, verification
tooling, and methodology suite runner. The goal is to make agent work inspectable
instead of ephemeral: prompts, outputs, selected models, provider metadata,
memory hits, artifacts, transcripts, and verification results become bounded
evidence.

```text
User / CLI
  -> HeliX router
       chooses local/cloud model profile by task type
  -> model call or local agent loop
       chat, research, code, audit, legal debate, RAG, tooling
  -> memory and evidence layer
       Merkle DAG nodes, signed/hashed receipts, JSONL/MD transcripts
  -> verifier
       artifact replay, claim boundaries, manifests, certification suites
```

## What Makes HeliX Different

- **Model-agnostic shell**: routes between DeepInfra, OpenAI-compatible
  endpoints, Anthropic-compatible profiles, Ollama, llama.cpp, and local runtime
  paths where configured.
- **Certified memory, not just chat history**: conversation turns and verified
  artifacts can be ingested into local HeliX memory, searched, and tied back to
  receipts instead of being treated as loose context.
- **Merkle DAG lineage**: memory records and evidence artifacts are chained with
  SHA-256-backed structure so integrity and provenance can be checked after the
  run.
- **Hard-anchor lane**: identity-critical references can travel as lightweight
  anchors instead of lossy summaries, preserving exact IDs/hashes under long
  horizon compression.
- **Tombstone and branch-pruning methodology**: invalidated branches are pruned
  before prompt injection. The model should not need to spend tokens reading old
  "do not use this" records.
- **Evidence-first test suites**: suite runs emit JSON artifacts, run manifests,
  logs, and JSONL/Markdown transcripts. The claim boundary lives with the
  artifact.
- **Agent shell UX**: interactive terminal with themes, model routing, key
  storage, evidence search, `/task` mode, and certification commands.

## Current CLI

Start the interactive shell from the repo root:

```cmd
helix
```

or directly:

```cmd
python -m helix_proto.helix_cli interactive
```

Install a launcher into the active Python environment:

```cmd
tools\install_helix_cli.cmd
```

The generated launcher sets `PYTHONPATH` to this repo and runs:

```cmd
python -m helix_proto.helix_cli %*
```

### First Run

```text
Provider [deepinfra] (Enter = default):
Model [auto] (Enter = default):
[helix] DEEPINFRA_API_TOKEN loaded from HeliX config.
* session: helix-interactive-...
* transcript: C:\Users\...\AppData\Local\HeliX\sessions\...
HeliX >
```

Tokens are read from environment variables, a hidden prompt, or optional HeliX
user config. Transcripts redact token values.

Save or remove a provider token:

```cmd
helix auth save deepinfra
helix auth forget deepinfra
```

Inside the shell:

```text
/key save
/key forget
/config
```

### Useful Interactive Commands

```text
/help                         Show directives
/status                       Show provider, model, workspace, transcript paths
/provider NAME                Switch provider
/model NAME                   Switch model or alias
/model list                   List model aliases and router blueprints
/route TEXT                   Explain model auto-routing for a prompt
/router NAME                  Change routing policy
/theme NAME                   industrial-brutalist, industrial-neon, xerox, brown-console
/raw on|off                   Toggle raw model output after cleaned answer
/evidence refresh [QUERY]     Verify and ingest artifacts from verification/
/evidence latest [N]          Show latest certified evidence memories
/evidence search QUERY        Search certified evidence memories
/evidence show MEMORY_ID      Show receipt and chain status
/verify PATH|latest|search Q  Verify or discover artifact JSONs
/memory QUERY                 Search unified HeliX memory
/task GOAL                    Run stronger agentic mode
/tools                        List tools exposed to the runner
/apply last                   Apply last proposed patch after confirmation
/cert SUITE [-- args]         Run a registered certification suite
/cert-dry SUITE [-- args]     Print suite command without running it
/exit                         Leave the session
```

Natural language defaults to chat. Repo/debug/patch prompts route toward
`/task`; suite/certification prompts route toward `/cert` when recognized.

## Model Router

The default router policy is `balanced`.

| Intent | Default profile | Model ID |
| --- | --- | --- |
| chat | `chat` | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` |
| reasoning | `reasoning` | `google/gemma-4-31B` |
| research | `research` | `Qwen/Qwen3.5-122B-A10B` |
| code | `code` | `Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo` |
| agentic | `agentic` | `Qwen/Qwen3.5-122B-A10B` |
| audit/legal/claims | `sonnet` | `anthropic/claude-4-sonnet` |
| vision | `llama-vision` | `meta-llama/Llama-3.2-11B-Vision-Instruct` |

Other built-in router policies:

- `current`: legacy HeliX behavior before the Qwen/Gemma/Mistral rebalance.
- `qwen-gemma-mistral`: explicit hybrid stack.
- `cheap`: lower-cost default path with code/audit escapes.
- `premium`: stronger engineering and reasoning path.

Inspect routing without making a model call:

```cmd
helix route "arregla este bug de pytest en el repo" --provider deepinfra
helix models list
helix providers list
```

## Evidence And Memory

HeliX stores two different kinds of history:

1. **Session transcripts**: interactive CLI turns are written as JSONL under the
   configured HeliX data directory, normally `AppData\Local\HeliX\sessions` on
   Windows.
2. **Verified evidence memories**: artifact JSONs under `verification/` can be
   verified, ingested, searched, and tied back to chain status.

Examples:

```text
/evidence refresh
/evidence latest 10
/evidence search hard-anchor
/verify latest
/verify search branch-pruning
/memory policy rag debate
```

Offline verification:

```cmd
python tools\helix_replay.py --mode verify-only --artifact verification\local-v4-lineage-forgery-gauntlet.json
helix cert verify verification\local-v4-lineage-forgery-gauntlet.json
```

## Certification Suites

Registered suites can be run through the CLI:

```cmd
helix cert list
helix cert run infinite-depth-memory
helix cert run branch-pruning-forensics --provider deepinfra
helix cert run policy-rag-legal-debate --provider deepinfra -- --tokens 1200
```

Direct Windows wrappers are also available:

```cmd
tools\run_nuclear_methodology_all.cmd
tools\run_post_nuclear_methodology_all.cmd
tools\run_long_horizon_checkpoint_all.cmd
tools\run_recursive_architectural_integrity_audit_all.cmd
tools\run_hard_anchor_utility_all.cmd
tools\run_branch_pruning_forensics_all.cmd
tools\run_infinite_depth_memory_all.cmd
tools\run_policy_rag_legal_debate_all.cmd
```

Suite outputs normally include:

- timestamped suite artifact JSON;
- per-case artifact JSON;
- run manifest JSON;
- evidence log;
- transcript exports as `.jsonl` and `.md`;
- `artifact_payload_sha256` for canonical payload hashing;
- external manifest hash policy when self-hashing would be circular.

## Current Bounded Results

The repo intentionally distinguishes product behavior from claim boundaries.
Start with:

- [`verification/public-evidence-index.json`](verification/public-evidence-index.json)
- [`verification/README-reviewer.md`](verification/README-reviewer.md)
- [`docs/claims-matrix.md`](docs/claims-matrix.md)

Representative evidence currently committed:

| Claim lane | Status | Evidence |
| --- | --- | --- |
| Provider-returned model metadata audit | `empirically_observed` | `verification/local-provider-integrity-observatory-20260418-154218.json` |
| Lineage forgery mechanics | `mechanics_verified` | `verification/local-v4-lineage-forgery-gauntlet.json` |
| External memory low overhead | `empirically_observed` | `verification/local-ghost-in-the-shell-live-20260418-154420.json` |
| Raw contaminated retrieval negative finding | `falsification_preserved` | `verification/local-ghost-in-the-shell-live-v2-20260418-160448.json` |
| Doppelganger war with receipt adjudication | `empirically_observed` | `verification/local-ghost-in-the-shell-live-v2-20260419-014000.json` |
| Bounded context under deep store | `mechanics_verified` | `verification/nuclear-methodology/infinite-depth-memory/local-infinite-depth-memory-suite-infinite-depth-memory-20260420-133040.json` |

Important boundary: HeliX does **not** claim literal infinite memory or physical
zero latency. The deep-memory artifact supports bounded context construction
under a 5,000-node local store. The historical `0.0 ms` wording is treated as
rounded telemetry, not as a physics claim.

## Latest Local Validation

The CLI and router test suite currently passes:

```cmd
python -B -m pytest tests\test_helix_cli.py -q
```

Result from the current branch:

```text
51 passed in 36.61s
```

Focused methodology tests include:

```cmd
python -m pytest tests\test_infinite_depth_memory_suite.py -q
python -m pytest tests\test_long_horizon_checkpoint_suite.py -q
python -m pytest tests\test_hard_anchor_utility_suite.py -q
python -m pytest tests\test_branch_pruning_forensics_suite.py -q
python -m pytest tests\test_policy_rag_legal_debate_suite.py -q
python -m pytest tests\test_architectural_recursion_audit.py -q
```

Native hard-anchor smoke:

```cmd
python tests\test_hard_anchors_rust.py
```

Observed local run:

```text
Legacy median latency:       73.4322 ms
Hard anchors median latency: 2.5258 ms
Speedup:                     29.0728x
```

Treat this as a local benchmark on one machine, not a universal performance
claim.

## Architecture

Rust/native layer:

- `crates/helix-merkle-dag`: SHA-256 Merkle DAG and PyO3 bindings.
- `crates/helix-state-core`: retrieval/indexing primitives.
- `crates/helix-state-server`: IPC state server for concurrent agent writes.
- `helix_kv`: Python/Rust bridge and memory catalog integration.

Python layer:

- `src/helix_proto/helix_cli.py`: CLI, router, auth, suites, memory commands.
- `src/helix_proto/helix_cli_chrome.py`: terminal UI themes and rendering.
- `src/helix_proto/helix_cli_agent_shell.py`: conservative agent shell loop.
- `src/helix_proto/evidence_ingest.py`: verified evidence memory ingestion.
- `src/helix_proto/artifact_replay.py`: offline artifact verification/replay.
- `src/helix_proto/provider_audit.py`: provider metadata audit utilities.
- `tools/transcript_exports.py`: JSONL/Markdown transcript sidecars.
- `tools/artifact_integrity.py`: payload hashing without self-hash circularity.

## Product Shape

HeliX is evolving into three connected surfaces:

1. **Interactive shell**: a Codex/Claude-Code-style terminal for chat, code,
   research, evidence lookup, and task execution.
2. **Certification lab**: repeatable methodology suites that produce artifacts,
   manifests, logs, and transcripts.
3. **Evidence browser/API substrate**: local artifact verification, memory
   search, web viewer assets, and OpenAI-compatible integration points.

The design principle is conservative: claims are only public when a runnable test
or committed artifact supports them, and negative findings stay in the record.

## License

GNU Affero General Public License v3.0 - see [LICENSE](LICENSE).

Copyright (C) 2026 Patricio Valbusa.
