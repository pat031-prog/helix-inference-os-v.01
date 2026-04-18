# HeliX hmem Wiring v0

This pass connects the dependency-light `MemoryCatalog` to the agent/runtime path so HeliX memory behaves like an operating layer, not a manual side table.

Canonical artifact:

- [`verification/local-hmem-wiring-smoke.json`](../verification/local-hmem-wiring-smoke.json)

## What Is Wired

- Agent startup now calls `hmem.build_context()` and injects bounded memory context into planner state.
- Every agent tool call now calls `hmem.observe_tool_call()` automatically.
- Tool observations are stored as raw observations and promoted to episodic memories.
- `helix.search` is now available as the preferred hybrid retrieval tool.
- `memory.search` reads from hmem first and falls back to legacy JSONL memory.
- Legacy `rag.search` remains available for compatibility.
- `/v1/chat/completions` keeps memory injection through the same hmem facade.

## New Local Memory Endpoints

- `POST /memory/observe`
- `POST /memory/search`
- `POST /memory/context`
- `POST /memory/graph`
- `GET /memory/graph`
- `GET /memory/stats`

`/agent/memory/search` remains as a compatibility alias.

## Optional Observation Compression

Observation compression is opt-in through `memory_compressor_alias`.

The default path is still heuristic and local. If a local model alias is provided, HeliX calls the model to compress long observations into a short operational memory. If the model is missing or fails, agent execution continues with the heuristic summary.

This is intentionally conservative: a compressor model should improve recall quality, not become another hard dependency in the hot path.

## Claim Boundary

Verified:

- Auto-observation of tool calls.
- Startup memory-context injection.
- Hybrid memory plus knowledge search surface.
- Local memory graph over sessions, memories and observations.
- Optional model-backed compression hook.

Not claimed yet:

- Embedding-grade retrieval quality.
- CrewAI/LangChain production compatibility.
- Automatic use of Qwen3.5-4B as a default compressor.
- Semantic correctness improvement from memory injection.

