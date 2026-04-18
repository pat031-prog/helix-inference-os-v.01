# Launch Post Draft

## Short Version

Shipping `helix-kv`, and with it a broader direction we are calling Helix Memory.

On pure Transformers, the verified GPU suite already shows up to `2.60x` KV reduction against native bf16 with `generated_match_vs_baseline=true` on Qwen2.5 and SmolLM2. On hybrid Zamba2, the local runtime shows the bottleneck shifting from KV to recurrent state, with `3.57x` total runtime-cache reduction and `1.50x` speedup in the combined mode.

## Numbers To Highlight

Sources:

- `verification/remote-transformers-gpu-summary.json`
- `verification/local-zamba2-hxq-vs-vanilla-summary.json`

| Regime | Headline result |
| --- | --- |
| Pure Transformer GPU | up to `2.60x` KV reduction with `match=true` |
| Hybrid Zamba2 local | `3.57x` total runtime-cache reduction in the combined mode |
| Hybrid Zamba2 local | `1.50x` speedup vs native in the combined mode |

## Suggested Post

Shipping `helix-kv`: a drop-in Hugging Face KV compression package that is turning into something broader.

What is already verified:

- pure Transformer GPU runs with up to `2.60x` KV reduction and `generated_match_vs_baseline=true`
- persistent compressed sessions
- multi-mode benchmarking through normal Hugging Face generation paths
- experimental hybrid-memory runtime on Zamba2 showing that recurrent state, not KV, becomes the main memory bottleneck

What changed for us conceptually:

- this is no longer only a KV story
- pure Transformers are KV-bound
- hybrid Mamba-Transformer models need both KV compression and recurrent-state compression

Repo:

- `<repo-url>`

Technical report:

- `<repo-url>/blob/main/docs/technical-report.md`

Unified-memory positioning note:

- `<repo-url>/blob/main/docs/unified-memory-story.md`

Built solo from Buenos Aires with Codex + Claude.
