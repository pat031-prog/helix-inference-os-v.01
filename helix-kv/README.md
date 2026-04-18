# helix-kv

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-green)

Drop-in KV cache compression for HuggingFace Transformers.

`helix-kv` is the standalone package surface inside the public Helix monorepo. It focuses on KV-cache footprint reduction, persistent compressed sessions, adaptive per-layer policies, and benchmarking against real Hugging Face generation paths.

The broader research direction around this package is now `Helix Memory`: architecture-aware inference memory compression across pure Transformers and hybrid Mamba-Transformer models. The current positioning note lives in [`../docs/unified-memory-story.md`](../docs/unified-memory-story.md).

## Install

```bash
pip install -e .
pip install -e ".[torch]"
pip install -e ".[hxq]"
```

## Quickstart

Programmatic benchmark:

```python
from helix_kv.benchmark import build_transformers_variant_set, run_transformers_kv_benchmark

report = run_transformers_kv_benchmark(
    "Qwen/Qwen2.5-1.5B-Instruct",
    prompt_text="Summarize why KV cache compression matters.",
    max_new_tokens=32,
    kv_variants=build_transformers_variant_set("stable"),
    device="cuda",
)

print(report["rows"][1]["name"])
print(report["rows"][1]["kv_cache_ratio_vs_native"])
print(report["rows"][1]["generated_match_vs_baseline"])
```

Prompt-prefill cache with session save/load:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from helix_kv import TransformersCompressedKVCache

model_ref = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_ref, torch_dtype="auto").to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_ref)
inputs = tokenizer("Hello from helix-kv.", return_tensors="pt").to("cuda")

cache = TransformersCompressedKVCache(
    model.config,
    kv_cache_precision="turbo-int8",
    kv_rotation_mode="hadamard",
    kv_hot_window=4,
)

with torch.inference_mode():
    _ = model(**inputs, past_key_values=cache, use_cache=True, return_dict=True)

cache.save("session_hf")
restored = TransformersCompressedKVCache.load("session_hf", model_config=model.config, device="cuda")
```

Standalone CLI:

```bash
helix-kv benchmark-transformers-kv Qwen/Qwen2.5-1.5B-Instruct \
  --variant-set stable \
  --device cuda \
  --output verification/transformers-kv-benchmark.json
```

## Verified GPU Snapshot

Primary source: [`../verification/remote-transformers-gpu-summary.json`](../verification/remote-transformers-gpu-summary.json)

KV ratios below are measured against the native bf16 KV cache from the corresponding benchmark JSONs.

| Model | `turbo-int8-hadamard` | `turbo-int8k-4bitv` | Source |
| --- | --- | --- | --- |
| `Qwen/Qwen2.5-1.5B-Instruct` | `1.96x` KV, `0.00%` PPL delta, `match=true` | `2.60x` KV, `0.78%` PPL delta, `match=true` | [`../verification/remote-qwen25-1.5b-transformers-gpu.json`](../verification/remote-qwen25-1.5b-transformers-gpu.json) |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `1.93x` KV, `0.00%` PPL delta, `match=true` | `2.58x` KV, `0.00%` PPL delta, `match=true` | [`../verification/remote-smollm2-1.7b-transformers-gpu.json`](../verification/remote-smollm2-1.7b-transformers-gpu.json) |
| `Qwen/Qwen2.5-3B-Instruct` | `1.96x` KV, `0.00%` PPL delta, `match=true` | `2.60x` KV, `3.17%` PPL delta, `match=true` | [`../verification/remote-qwen25-3b-transformers-gpu.json`](../verification/remote-qwen25-3b-transformers-gpu.json) |

Long-context reference:

- `Qwen/Qwen2.5-3B-Instruct` at 4096 prompt tokens kept `match=true` with `1.97x` KV reduction for `turbo-int8-hadamard` and `2.63x` for `turbo-int8k-4bitv`: [`../verification/remote-transformers-gpu-summary.json`](../verification/remote-transformers-gpu-summary.json)

## Hybrid-Memory Research Snapshot

The package surface is still named `helix-kv`, but the repo now also contains an experimental hybrid runtime for Zamba2-style models where KV is only part of the memory story.

Source: [`../verification/local-zamba2-hxq-vs-vanilla-summary.json`](../verification/local-zamba2-hxq-vs-vanilla-summary.json)

| Hybrid local result | Runtime-cache ratio vs native | Speedup vs native | Note |
| --- | --- | --- | --- |
| `KV-only gain` | `1.01x` | `1.05x` | Transformer KV shrinks, but total hybrid memory barely moves. |
| `Mamba-state-only gain` | `3.40x` | `1.22x` | Recurrent-state compression moves the real bottleneck. |
| `Combined hybrid gain` | `3.57x` | `1.50x` | Best local result on `Zyphra/Zamba2-1.2B-Instruct-v2`. |

Prompt-category behavior for that same local run is separated in [`../verification/local-zamba2-prompt-suite-code-daily.json`](../verification/local-zamba2-prompt-suite-code-daily.json):

- `code`: `1.04x` average speedup with the same `3.57x` runtime-cache ratio
- `daily`: `0.90x` average speedup with the same `3.57x` runtime-cache ratio

The generated figure used for the broader narrative is [`../docs/figures/hybrid-memory-frontier.svg`](../docs/figures/hybrid-memory-frontier.svg), backed by [`../verification/hybrid-memory-frontier-summary.json`](../verification/hybrid-memory-frontier-summary.json).

## Available Modes

| Mode | Status | Notes |
| --- | --- | --- |
| `native-dense` | stable | Baseline Hugging Face `DynamicCache`-style dense storage. |
| `turbo-int8-hadamard` | stable | Best default for fidelity on current verified GPU runs. |
| `turbo-int8k-4bitv` | stable | Best verified footprint reduction on the primary model set. |
| `turbo-int8k-4bitv-online` | experimental | Refined 4-bit V codebooks from prompt calibration. |
| `adaptive-m9-h20` | research | Per-layer promotion based on kurtosis and K/V norm ratio. |
| `adaptive-asymmetric-m9-h20` | research | Independent K/V selection with asymmetric scaling. |
| `helix-optimal` | experimental | Community-inspired composite profile with protected layers and Sparse V probe support. |
| `turbo-qjl` | deprecated | Kept for backward compatibility only. Not recommended for new work. |

## What This Is Not

- It is not a serving stack. Use vLLM, TGI, SGLang, or your serving runtime of choice around it.
- It does not replace GPTQ, AWQ, or HXQ. Those compress model weights; `helix-kv` compresses KV cache and serialized sessions.
- It does not ship custom CUDA or Triton kernels yet. Current implementation is torch-native.
- It does not require training or finetuning.

## Related Work

| Project | Main focus | Where `helix-kv` differs |
| --- | --- | --- |
| `KIVI` | Asymmetric low-bit KV quantization | `helix-kv` exposes multiple modes plus persistent sessions. |
| `KVQuant` | Quantized KV cache with calibration-oriented analysis | `helix-kv` emphasizes drop-in HF benchmarking and serialization. |
| `TurboQuant` | Rotation-heavy practical KV quantization | `helix-kv` adds save/load sessions, adaptive profiles, and mode sweeps. |
| `kvpress` | Serving-oriented KV compression tooling | `helix-kv` is package-first and benchmark-first rather than a serving runtime. |
| `helix-online-kv` | Earlier online calibration experiments | `helix-kv` is the packaged, multi-mode public surface. |

For the hybrid-memory direction, the repo also tracks Zamba2 docs plus state-compression references such as Q-Mamba in [`../docs/zamba2-local-research.md`](../docs/zamba2-local-research.md).

## Architecture

```text
prompt tokens
   |
   v
prefill on HF model
   |
   +--> hot window tail kept exact
   |
   +--> cold prefix compressed (int8 / 4bit / adaptive)
   |
   +--> session.json + kv_cache.npz
   |
   +--> load() / resume() on the same model config
```

## API Reference

- `KVConfig`: config object for the standalone exported-model runtime.
- `CompressedKVCache`: cache/session manager for the exported Helix runtime.
- `TransformersCompressedKVCache`: Hugging Face cache object with save/load support.
- `AdaptiveKVPolicy`: runtime policy helper for agent-aware and phase-aware switching.
- `build_transformers_variant_set()`: stable, asymmetry-sweep, and community benchmark presets.
- `run_transformers_kv_benchmark()`: one-call benchmark runner for Hugging Face models.

## HXQ And Gemma Compatibility

HXQ:

- EchoLabs33 refs are detected as weight-compressed models.
- Benchmark JSONs now record `is_hxq_compressed`, `hxq_model_ref`, `weight_compression_method`, `weight_runtime_source`, `model_size_bytes`, `model_vram_bytes`, and `total_inference_footprint_bytes`.
- The remote runner installs `helix-substrate` and `mamba-scan-lite` first, then falls back to the local repo copy only if the published runtime path fails.

Gemma 3/4:

- `google/gemma-3-*` and `google/gemma-4-*` refs use a processor-aware textual adapter path.
- The benchmark JSON records `input_adapter`, `processor_used`, `chat_template_used`, `gated_model`, and `hf_auth_required`.
- Gated Gemma refs require an accepted Hugging Face license and token on the benchmark machine.

## Citation

```bibtex
@misc{helixkv2026,
  title        = {Helix Memory: Unified Memory Compression for Transformer and Hybrid Mamba-Transformer Inference},
  author       = {Helix},
  year         = {2026},
  note         = {Technical report outline in docs/technical-report.md},
  howpublished = {\url{https://github.com/<your-org>/<your-repo>}}
}
```
