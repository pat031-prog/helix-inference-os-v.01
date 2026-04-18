# Mail Draft To Josh

Subject: HXQ + Helix Memory integration path is ready

Hey Josh,

I just finished wiring the `helix-kv` benchmark/runtime path for HXQ-aware models plus the metadata we need for a clean integration story.

The broader frame is getting stronger too: pure Transformers are still a KV-compression story, but local Zamba2 hybrid runs now show that the real memory bottleneck shifts into recurrent state. That gives us a much bigger shared pitch than "just another KV quantizer".

What is already in place:

- installed-first HXQ loading with local repo fallback
- benchmark JSON metadata for `is_hxq_compressed`, `hxq_model_ref`, `weight_runtime_source`, `model_size_bytes`, `model_vram_bytes`, and `total_inference_footprint_bytes`
- processor-aware Gemma 3/4 textual benchmark path
- updated remote runner that installs `helix-substrate` and `mamba-scan-lite`
- standalone `helix-kv` CLI and package docs

Current verified public GPU baseline is here:

- `verification/remote-transformers-gpu-summary.json`
- `verification/remote-qwen25-1.5b-transformers-gpu.json`
- `verification/remote-smollm2-1.7b-transformers-gpu.json`
- `verification/remote-qwen25-3b-transformers-gpu.json`

The next GPU run on my side is set up to produce:

- `verification/hxq-qwen3b-compatibility.json`
- `verification/hxq-vs-vanilla-qwen3b.json`
- `verification/hxq-zamba2-compatibility.json` if the Transformers path loads cleanly

Once those JSONs are in place, the collaboration pitch is straightforward:

your weights + my KV/runtime-state compression + my runtime/serialization path = a strong joint paper/demo story

If you want, I can send you the exact benchmark command set next so we can reproduce the same matrix on your side too.

Abrazo,

<your-name>
