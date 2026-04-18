# HeliX Inference OS Architecture

HeliX is moving from a compression runtime toward an inference-session operating system. The core distinction is simple:

> HeliX does not think. HeliX governs.

Models generate tokens. HeliX owns the deterministic parts around them: lifecycle, routing, memory, session state, integrity, budgets and handoff.

Canonical artifacts:

- [`verification/local-inference-os-architecture-summary.json`](../verification/local-inference-os-architecture-summary.json)
- [`verification/local-blueprint-stack-catalog.json`](../verification/local-blueprint-stack-catalog.json)
- [`verification/local-blueprint-meta-microsite-demo.json`](../verification/local-blueprint-meta-microsite-demo.json)

## The Four Layers

| Layer | Name | Role |
| --- | --- | --- |
| 1 | Active Model | The model currently loaded for token execution. It is ephemeral and interchangeable. |
| 2 | Private `.hlx` State | Architecture-specific computed work: KV cache, Mamba recurrent state, hashes and deferred audit. |
| 3 | Shared `hmem` | Portable semantic memory: observations, tool outputs, handoffs and graph context. |
| 4 | Multimodel Scheduler | Chooses which model wakes, which session restores, which memory enters the prompt and when audit closes. |

## Private State vs Shared Memory

`.hlx` is not semantic memory. It is private computed state and is only safe for the same `(model_id, agent_id)` and compatible architecture.

`hmem` is not KV cache. It is portable context that can move between models through prompts, summaries, memories and graph links.

The useful system is the combination: private state preserves compute, shared memory preserves meaning.

## Blueprints

Blueprints are HeliX workloads. They describe a team of agents, model preferences, task order, memory policy, session policy and outputs.

The first runnable Blueprint is `Meta Microsite`: HeliX orchestrates agents to build a page explaining HeliX itself.

The v0 catalog also defines:

- `frontend-factory`
- `dba-data-engineering`
- `legal-research`

These are spec smokes first, not model-quality claims.

## Claim Boundary

The Meta Microsite proves Blueprint orchestration, hmem wiring, private `.hlx` state artifacts, deferred audit and deterministic quality-first rendering.

If it runs in fallback or mixed mode, it does not prove model generation quality. Real-model generation quality requires a run where the artifact records prepared local aliases and no fallback content.

