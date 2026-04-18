# Zamba2 Local Research Notes

## Primary sources

- Hugging Face Transformers Zamba2 docs: <https://huggingface.co/docs/transformers/en/model_doc/zamba2>
- Vanilla model reference: <https://huggingface.co/Zyphra/Zamba2-1.2B-Instruct-v2>
- HXQ model reference: <https://huggingface.co/EchoLabs33/zamba2-1.2b-hxq>
- Q-Mamba paper (Findings of ACL 2025): <https://aclanthology.org/2025.findings-acl.551/>
- BitMamba-2 model card used only as context on low-bit SSM work: <https://huggingface.co/Zhayr1/BitMamba-2-1B>

## Why this matters for Helix-KV

The Transformers Zamba2 docs describe the model as hybrid and expose `layers_block_type` plus a dedicated hybrid cache path. That matches the local Helix implementation: KV compression only applies to the transformer-backed layers, while the Mamba recurrent state is tracked separately.

Q-Mamba is the closest reference for the new experimental runtime added here. The paper motivates decoupled-scale quantization for state-space models and frames state compression as the main memory lever once KV is no longer dominant.

## Local findings backed by verification JSONs

### 1. Vanilla Zamba2 confirms the hybrid memory split

Source: `verification/local-zamba2-vanilla-hybrid-cpu-smoke.json`

- `native-dense`:
  - `total_time_s = 183.85698829998728`
  - `kv_cache_bytes = 1671168`
  - `mamba_state_bytes = 41168896`
  - `hybrid_total_cache_bytes = 42840064`
- `turbo-int8-hadamard`:
  - `total_time_s = 174.9765938000055`
  - `kv_cache_ratio_vs_native = 1.6035372144436257`
  - `hybrid_total_runtime_cache_ratio_vs_native ~= 1.0149`
  - `prompt_perplexity_delta_pct_vs_native = 6.449445891785935`

Takeaway: compressing only transformer KV helps, but the total hybrid cache barely moves because the recurrent Mamba state dominates memory.

### 2. Q-Mamba-style runtime moves the real bottleneck

Source: `verification/local-zamba2-qmamba-runtime-cpu-smoke.json`

- `q-mamba-dsq-int4`:
  - `mamba_state_runtime_bytes = 10944304`
  - `mamba_state_runtime_ratio_vs_native = 3.761673286853143`
  - `hybrid_total_runtime_cache_ratio_vs_native = 3.3958352093365987`
  - `speedup_vs_native = 1.215467997189594`
  - `prompt_perplexity_delta_pct_vs_native = 0.0`
- `turbo-int8-hadamard+q-mamba-dsq-int4`:
  - `hybrid_total_runtime_cache_bytes = 11986480`
  - `hybrid_total_runtime_cache_ratio_vs_native = 3.5740320761391167`
  - `speedup_vs_native = 1.4952353797959292`
  - `prompt_perplexity_delta_pct_vs_native = 6.449445891785935`

Takeaway: on this local Zamba2 setup, state compression matters much more than KV compression. The combined mode gives the strongest footprint win because it attacks both parts of the hybrid cache.

### 3. Prompt-category behavior is different for code vs daily prompts

Source: `verification/local-zamba2-prompt-suite-code-daily.json`

- Code prompts:
  - `avg_speedup_vs_native = 1.0449358950932264`
  - `avg_prompt_perplexity_delta_pct_vs_native = -1.538338276182795`
  - `avg_hybrid_total_runtime_cache_ratio_vs_native = 3.5740320761391167`
- Daily prompts:
  - `avg_speedup_vs_native = 0.8969903375540891`
  - `avg_prompt_perplexity_delta_pct_vs_native = 0.7873854293342814`
  - `avg_hybrid_total_runtime_cache_ratio_vs_native = 3.5740320761391167`

Takeaway: the footprint win is stable across categories, but runtime and perplexity move differently by prompt family. Code prompts were slightly more favorable than everyday prompts in this CPU-local run.

### 4. HXQ is still not stable enough for benchmark comparison

Source: `verification/local-zamba2-hxq-direct-diagnostics.json`

- `effective_model_ref = "EchoLabs33/zamba2-1.2b-hxq"`
- `weight_runtime_source = "pypi"`
- `logits_finite = false`
- `nan_count = 512000`
- `load_error = null`
- `forward_error = null`

Takeaway: the canonical HXQ ref loads, but the final benchmark smoke prompt still produces non-finite logits in the final reproducible diagnostic JSON. That means the local fix is good enough to load the checkpoint, but not good enough to trust comparative generation numbers yet.

## Practical comparison framing

For hybrid Zamba2, local evidence now supports this framing:

- `KV-only gain`: modest total-cache improvement because KV is a small slice of the full hybrid state.
- `Mamba-state-only gain`: the main memory lever.
- `Combined hybrid gain`: best overall local footprint result.
- `Vanilla vs HXQ`: still blocked until HXQ becomes prompt-stable under the same diagnostic prompt.

The summary figure derived from these artifacts is `docs/figures/hybrid-memory-frontier.svg`, backed by `verification/hybrid-memory-frontier-summary.json`.

The reusable runner for the next Linux/CUDA pass is `tools/run_hybrid_memory_sweep.py`.

## Local environment notes

- CPU-only local run.
- The fast Mamba path was unavailable because `mamba-ssm` / `causal-conv1d` binary wheels were not available in the tested Windows environment.
- All numeric claims above come from the JSON artifacts listed in `verification/`.
