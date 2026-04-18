# Helix Memory: Unified Memory Compression for Transformer and Hybrid Mamba-Transformer Inference

## Abstract

Inference memory bottlenecks are architecture-dependent. Pure Transformer decoders accumulate large KV caches that scale with context length, while hybrid Mamba-Transformer models shift a larger share of runtime memory into recurrent state. Helix Memory is the research direction built around the packaged `helix-kv` runtime: a torch-native framework for multi-mode KV compression, persistent compressed sessions, and experimental recurrent-state compression for hybrid models. The current verified GPU suite on Qwen2.5-1.5B, SmolLM2-1.7B, and Qwen2.5-3B shows up to `2.60x` KV reduction against native bf16 while preserving `generated_match_vs_baseline=true` on the primary model set. The current local hybrid Zamba2 runtime shows that KV-only compression barely changes total hybrid memory, while Q-Mamba-style state compression reaches `3.57x` total runtime-cache reduction and `1.50x` speedup in the combined mode. This report reframes the project from a KV-only package into a broader architecture-aware memory-compression story, grounded only in the benchmark JSON artifacts committed under `verification/`.

Verified benchmark sources used by this outline:

- `verification/remote-qwen25-1.5b-transformers-gpu.json`
- `verification/remote-smollm2-1.7b-transformers-gpu.json`
- `verification/remote-qwen25-3b-transformers-gpu.json`
- `verification/remote-transformers-gpu-summary.json`
- `verification/local-zamba2-vanilla-hybrid-cpu-smoke.json`
- `verification/local-zamba2-qmamba-runtime-cpu-smoke.json`
- `verification/local-zamba2-prompt-suite-code-daily.json`
- `verification/local-zamba2-hxq-vs-vanilla-summary.json`
- `verification/hybrid-memory-frontier-summary.json`

## 1. Introduction

- Transformer inference is memory-bound because KV cache grows with prompt length, layer count, and number of attention heads.
- Hybrid Mamba-Transformer models change that profile: transformer KV is still present, but recurrent state becomes a separate and often larger memory term.
- Existing methods typically optimize one side of that boundary.
- Helix Memory targets the practical gap: one benchmarkable runtime path that handles both KV compression and hybrid-state compression while preserving session serialization.

## 2. Method

### 2.1 Transformer KV Compression

- `helix-kv` provides dense, int8, asymmetric int8/4-bit, and adaptive per-layer KV modes.
- Hadamard rotation is the current promoted path for fidelity.
- Sessions serialize to `session.json` plus compressed tensor storage and can be resumed on the same model config.

### 2.2 Hybrid Cache Split

- Zamba2-style models are detected through `layers_block_type`.
- Transformer-backed layers continue through the KV compression path.
- Mamba-backed layers keep separate recurrent `conv_states` and `ssm_states`.

### 2.3 Experimental Recurrent-State Compression

- The current hybrid runtime adds `mamba_state_precision="q-mamba-dsq-int4"`.
- The runtime compresses recurrent state after each forward step, materializes it before the next decode step, and tracks bytes plus compression/materialization time.
- This is deliberately framed as experimental and local-first until a Linux/CUDA validation pass exists.

### 2.4 Persistent Sessions

- Dense and compressed cache objects support save/load.
- Hybrid cache sessions now retain metadata for transformer-layer selection and Mamba-state precision.

## 3. Experiments

### 3.1 Pure Transformer GPU Suite

Models:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`

Headline results from `verification/remote-transformers-gpu-summary.json`:

| Model | Best-fidelity mode | Best-compression mode |
| --- | --- | --- |
| Qwen2.5-1.5B | `turbo-int8-hadamard`: `1.96x` KV, `0.00%` PPL delta, `match=true` | `turbo-int8k-4bitv`: `2.60x` KV, `0.78%` PPL delta, `match=true` |
| SmolLM2-1.7B | `turbo-int8-hadamard`: `1.93x` KV, `0.00%` PPL delta, `match=true` | `turbo-int8k-4bitv`: `2.58x` KV, `0.00%` PPL delta, `match=true` |
| Qwen2.5-3B | `turbo-int8-hadamard`: `1.96x` KV, `0.00%` PPL delta, `match=true` | `turbo-int8k-4bitv`: `2.60x` KV, `3.17%` PPL delta, `match=true` |

### 3.2 Hybrid Local Zamba2 Suite

Model:

- `Zyphra/Zamba2-1.2B-Instruct-v2`

Variants:

- `native-dense`
- `turbo-int8-hadamard`
- `q-mamba-dsq-int4`
- `turbo-int8-hadamard+q-mamba-dsq-int4`

Headline results from `verification/local-zamba2-hxq-vs-vanilla-summary.json`:

| Hybrid result | Runtime-cache ratio vs native | Speedup vs native | Interpretation |
| --- | --- | --- | --- |
| `KV-only gain` | `1.01x` | `1.05x` | KV compression alone barely moves total hybrid memory. |
| `Mamba-state-only gain` | `3.40x` | `1.22x` | Recurrent-state compression is the main lever. |
| `Combined hybrid gain` | `3.57x` | `1.50x` | Best current local hybrid result. |

### 3.3 Prompt Categories

Source: `verification/local-zamba2-prompt-suite-code-daily.json`

- `code`: `1.04x` average speedup vs native with the combined hybrid mode
- `daily`: `0.90x` average speedup vs native with the same memory ratio

Inference from the artifact: memory reduction appears architecture-driven and stable across prompt families, while latency remains workload-sensitive.

### 3.4 HXQ Status

- HXQ-compatible metadata and installed-first loading are implemented.
- `EchoLabs33/zamba2-1.2b-hxq` currently loads, but the final local diagnostic still reports non-finite logits under the benchmark smoke prompt.
- No public generation-quality claim should be made for HXQ Zamba2 until `verification/local-zamba2-hxq-direct-diagnostics.json` becomes finite and reproducible.

## 4. Analysis

- Pure Transformer runs still justify the package-facing `helix-kv` story because those models are strongly KV-bound.
- Hybrid Zamba2 changes the optimization target: recurrent state dominates total runtime memory, so KV-only compression is no longer the most important lever.
- The combined runtime is the current frontier result in this repo because it compresses both the transformer and state-space sides of a hybrid cache.
- Persistent sessions matter for both regimes because they make the compressed state operationally reusable, not just benchmark-friendly.

## 5. Related Work

- KIVI and KVQuant represent the Transformer KV-cache quantization lane.
- Q-Mamba, Slender-Mamba, and MambaQuant represent active Mamba-side compression and quantization work.
- Zamba2 provides the architectural bridge where both memory terms matter in one inference path.
- Inference from these sources plus the repo artifacts: the literature is still fragmented across KV-only and state-only compression stories, while the current Helix prototype is aimed at the hybrid boundary.

## 6. Limitations

- The current hybrid evidence is CPU-local and experimental.
- No custom CUDA or Triton kernels are included yet.
- The fast Mamba path was unavailable in the tested Windows environment.
- HXQ Zamba2 is not numerically stable enough yet for publication-quality comparison.
- Qualitative prompt examples need stronger serialization of raw decoded text, not only metrics.

## 7. Conclusion

- `helix-kv` remains a strong package surface for practical Hugging Face KV compression.
- The broader research contribution emerging from the repo is now unified memory compression for Transformer and hybrid Mamba-Transformer inference.
- The next gate is GPU hybrid validation on Linux/CUDA plus stable HXQ reproduction in the canonical environment.
