# HeliX Session OS v2: AgentMemory Recall and AirLLM Bridge

This pass keeps HeliX in control of the inference-session hot path.

## What We Took From AgentMemory

AgentMemory is useful as a product pattern: capture observations, distill memories, search them with a small token budget, and inject only the relevant context into the next agent turn.

HeliX absorbs that idea without adding AgentMemory as a dependency. The local implementation is `MemoryCatalog`, backed by stdlib SQLite. It stores observations, distilled memories, and links between memories and HeliX sessions.

Default search is intentionally austere:

- SQLite FTS5 when available.
- Lexical fallback when FTS5 is unavailable.
- No ChromaDB.
- No vector database.
- No Node server.
- No `iii-engine`.

This keeps semantic recall above the prompt while `.hlx` session state remains below the prompt.

## What We Did Not Copy

HeliX does not import AgentMemory's server, MCP tools, viewer, graph engine, provider stack, or optional embeddings in this phase.

Those are good product references, but they are not allowed to sit in the inference hot path. HeliX's core claim remains compressed, verifiable, restorable model state.

## AirLLM Direction

AirLLM is a different layer. It decomposes models into layer shards and loads weights layer by layer during forward. That makes it interesting as a future execution backend for machines with small VRAM, but dangerous as a direct dependency before HeliX has a stable layer-slice interface.

The v2 bridge adds a dependency-free adapter seam:

- `LayerLifecycleAdapter`: activate, run, and unload one layer.
- `LayerCacheInjector`: read only the HeliX cache slice for the active layer.
- `run_mock_airllm_loop`: prove the choreography without installing AirLLM.

The future fork target is an “AirLLM HeliX Edition” where HeliX calls are injected around AirLLM's layer load and forward loop:

- `load_layer_to_cpu`
- `move_layer_to_device`
- per-layer forward execution
- layer unload/cleanup

## Layer-Slice `.hlx`

Layer-slice v1 adds metadata that maps cache arrays to a model layer:

- `layer_index`
- `layer_name`
- `block_type`
- `cache_kind`
- `token_start`
- `token_count`
- `architecture`

The first read path returns CPU arrays/tensors. It does not use raw pointer injection and does not promise zero-copy into PyTorch storage.

Transformer KV slices can be addressed per layer. Hybrid Zamba/Mamba remains exact checkpoint restore only; arbitrary recurrent-state slicing is intentionally unsupported until it is proven safe.

## Commands

```powershell
python tools\run_local_agent_memory_catalog_smoke.py --output-dir verification
python tools\run_local_hlx_layer_slice_smoke.py --output-dir verification
python tools\run_local_airllm_bridge_smoke.py --output-dir verification
python tools\run_local_memory_augmented_openai_smoke.py --output-dir verification
```

Expected artifacts:

- `verification/local-agent-memory-catalog-smoke.json`
- `verification/local-hlx-layer-slice-smoke.json`
- `verification/local-airllm-bridge-smoke.json`
- `verification/local-memory-augmented-openai-smoke.json`

## Claim Boundaries

Verified after this pass:

- HeliX can store and search local agent memories without heavyweight dependencies.
- HeliX can inject bounded memory context through the OpenAI-compatible endpoint when explicitly requested.
- HeliX can address `.hlx` cache slices by layer metadata.
- HeliX can simulate an AirLLM-style layer lifecycle without importing AirLLM.

Not claimed yet:

- Real AirLLM giant-model execution.
- Raw pointer injection into PyTorch tensors.
- Arbitrary Mamba recurrent-state prefix slicing.
- Vector retrieval quality comparable to AgentMemory's embedding path.
