# Unified Memory Story

## Thesis

`helix-kv` started as a practical KV-cache compression package for Hugging Face Transformers. The strongest current result is now broader than that:

- on pure Transformers, the verified GPU suite shows competitive real-world KV reduction on Qwen2.5 and SmolLM2 while preserving `generated_match_vs_baseline=true`: `verification/remote-transformers-gpu-summary.json`
- on hybrid Zamba2, the local runtime shows that the bottleneck shifts from KV cache to recurrent state, and the combined runtime reaches `3.57x` total runtime-cache reduction plus `1.50x` speedup: `verification/local-zamba2-hxq-vs-vanilla-summary.json`

That makes the public narrative less about "another KV quantizer" and more about architecture-aware inference memory compression across Transformer, Mamba, and hybrid Mamba-Transformer models.

## Why this is interesting

- Hugging Face documents Zamba2 as a hybrid architecture with explicit `layers_block_type` and a dedicated `Zamba2HybridDynamicCache` path: <https://huggingface.co/docs/transformers/en/model_doc/zamba2>
- Q-Mamba frames state-cache compression as a first-class problem for state-space models and reports DSQ-based memory reduction with limited quality loss: <https://aclanthology.org/2025.findings-acl.551/>
- KIVI and KVQuant represent the Transformer-side KV compression lane: <https://arxiv.org/abs/2402.02750>, <https://arxiv.org/abs/2401.18079>
- Slender-Mamba and MambaQuant show active work on the Mamba-side quantization lane: <https://aclanthology.org/2025.coling-main.316/>, <https://arxiv.org/abs/2501.13484>

Inference from those sources plus the local artifacts:

- current literature is split across KV-cache compression and Mamba-state quantization
- the current Helix hybrid runtime is aimed at the gap between those two camps
- the local Zamba2 results support that framing because `KV-only gain` is tiny on total runtime memory, while `Mamba-state-only gain` is large

## Verified evidence in this repo

### Pure Transformer GPU evidence

Source: `verification/remote-transformers-gpu-summary.json`

- `turbo-int8-hadamard` is the current fidelity default across the three primary GPU models
- `turbo-int8k-4bitv` is the strongest verified KV reducer, reaching `2.60x` on the primary model set and `2.63x` at 4096 prompt tokens on Qwen2.5-3B

### Hybrid Zamba2 local evidence

Source: `verification/local-zamba2-hxq-vs-vanilla-summary.json`

- `KV-only gain`: `1.01x` total runtime-cache ratio vs native
- `Mamba-state-only gain`: `3.40x` total runtime-cache ratio vs native
- `Combined hybrid gain`: `3.57x` total runtime-cache ratio vs native and `1.50x` speedup vs native

### Prompt-category evidence

Source: `verification/local-zamba2-prompt-suite-code-daily.json`

- `code`: `1.04x` average speedup vs native
- `daily`: `0.90x` average speedup vs native

The same combined mode kept the same memory ratio across categories, which suggests the memory win is architecture-driven while latency remains workload-sensitive.

## Four tracks to prioritize

### 1. GPU Hybrid Validation

- Reproduce Zamba2 hybrid results on Linux/CUDA with the fast Mamba path enabled
- Start with `Zyphra/Zamba2-1.2B-Instruct-v2`, then extend to `Zyphra/Zamba2-2.7B-Instruct-v2`
- Sweep at least three context lengths with four variants only:
  - `native-dense`
  - `turbo-int8-hadamard`
  - `q-mamba-dsq-int4`
  - `turbo-int8-hadamard+q-mamba-dsq-int4`

Recommended runner: `tools/run_hybrid_memory_sweep.py`

### 2. HXQ Reproduction, Not Reinvention

- Treat `EchoLabs33/zamba2-1.2b-hxq` as a separate weight-compression axis, not as part of the hybrid-state runtime itself
- Use the model-card path first: `pip install "helix-substrate[hf]"` and `import helix_substrate`
- Only compare `vanilla`, `vanilla+runtime`, `HXQ`, and `HXQ+runtime` after `logits_finite=true` on repeated prompts

Current blocking artifact: `verification/local-zamba2-hxq-direct-diagnostics.json`

### 3. Publication Positioning

The clean research framing is:

- memory bottlenecks are architecture-dependent
- Transformer inference needs KV compression
- hybrid Mamba-Transformer inference needs both KV compression and recurrent-state compression
- persistent compressed sessions make the system operationally useful, not only numerically interesting

### 4. Product Path

- Keep the prototype inside Transformers first
- Move to a serving integration only after GPU hybrid proof exists
- Avoid centering the first public story on TGI, because the current hybrid-memory novelty is closer to runtime research than to serving packaging

## Immediate next experiments

- Store raw generated text plus `generated_ids` in prompt-suite artifacts so qualitative examples are publishable
- Run the same four-prompt suite on GPU hybrid models
- Add the larger `Zamba2-2.7B` sweep
- Add a context-length plot where the key signal is `KV bytes vs recurrent-state bytes vs total runtime-cache bytes`
- Keep the summary figure current through `tools/build_hybrid_memory_figure.py`

## Figure and summary artifact

- Figure: `docs/figures/hybrid-memory-frontier.svg`
- Backing JSON: `verification/hybrid-memory-frontier-summary.json`

Both are derived only from existing verification JSONs.
