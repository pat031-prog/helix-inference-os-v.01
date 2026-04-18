# HeliX Multimodel Hypervisor v0

This experiment validates the next scheduling layer above the session core:

> Each model keeps its own session state. HeliX swaps model weights, restores the matching `(model_id, agent_id)` session, generates a short continuation, saves a new checkpoint, and verifies the checkpoints after the fast path.

It does **not** share KV cache between different models. Coordination happens through external text handoffs and artifact hashes.

## Command

```powershell
python tools\run_local_multimodel_hypervisor.py --scenario coder-writer --code-model qwen-1.5b --writer-model gpt2-fast --max-new-tokens 8 --prompt-tokens 192 --codec rust-hlx-buffered-flat --audit-policy deferred --output-dir verification --local-files-only --device cpu
```

Artifact:

```text
verification/local-multimodel-hypervisor-demo.json
```

## Local result

The laptop run completed with:

- Models: `qwen-1.5b` and `gpt2-fast`
- Agents: `code_reviewer` and `writer`
- Model activations/swaps: `3`
- Restored sessions: `1`
- `all_restore_hash_matches=true`
- `all_pending_receipts_loaded=true`
- `all_final_audits_verified=true`
- Total wall time: `256.07s`

Measured swap times:

- Qwen initial activation: `3309.58ms`
- Qwen to GPT-2: `2592.22ms`
- GPT-2 back to Qwen: `3226.17ms`

The restored Qwen `code_reviewer` session matched before continuation:

```text
session_hash_before = a130e59258a95d2b
session_hash_loaded = a130e59258a95d2b
restore_hash_match  = true
```

Final deferred audits closed as `verified` for all saved checkpoints.

## Caveat

This is a systems demo, not a quality demo. The generated text is short and not especially useful because the local run used tiny token budgets on CPU. The verified claim is:

> HeliX can swap between two different local models, restore a model-specific agent session after another model ran, and later close cryptographic audit on all checkpoints.

The stronger future demo should use two instruction-tuned cached models, or a GPU box, so the agent text quality can match the scheduler story.
