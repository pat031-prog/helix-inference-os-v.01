# HeliX vs Agent Memory Below the Prompt

This note compares HeliX against **Agent Memory Below the Prompt: Persistent Q4 KV Cache for Multi-Agent LLM Inference on Edge Devices** (`arXiv:2603.04428`).

Source: https://huggingface.co/papers/2603.04428

## What the paper establishes

Shkolnikov frames the same core pressure point HeliX is targeting: multi-agent systems on edge devices cannot keep every agent KV cache resident forever. The paper reports persistent Q4 KV cache, cache restoration, a block pool, `BatchQuantizedKVCache`, and cross-phase context injection. Its headline claim is TTFT reduction up to `136x` on larger Apple Silicon runs.

That makes it the closest public neighbor to HeliX's session-memory direction.

## What this evidence sprint adds

Artifacts:

- `verification/local-ttft-cold-warm-summary.json`
- `verification/local-agent-capacity-budget.json`
- `verification/agent-memory-comparison-summary.json`

The sprint measures:

- cold full-prefill TTFT
- warm restored-session TTFT including restore
- warm compute-only TTFT after restore
- session load time
- deferred pending time and final verified time
- how many measured/projected agent sessions fit in a fixed memory budget

## Local results from this sprint

On this laptop, with `rust-hlx-buffered-flat`, `audit_policy=deferred`, `128` prefix tokens, `16` followup tokens, and `5` repeats:

| Model | Cold TTFT p50 | Warm TTFT p50 including restore | Speedup incl. restore | Warm compute-only p50 | Top-1 match | Max logit delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `gpt2` | `1004.20ms` | `563.56ms` | `1.78x` | `498.63ms` | `true` | `0.0191` |
| `Qwen/Qwen2.5-1.5B-Instruct` | `73228.95ms` | `8871.84ms` | `8.25x` | `8746.83ms` | `true` | `0.125` |

The Qwen number is the strongest local Agent-Memory-style signal: the restored-session path moves p50 TTFT from roughly `73.2s` to `8.87s` while preserving the selected token.

The `10 GiB` capacity table projects from measured session sizes:

| Model | Measured context | Measured per-agent session | Agents fit measured | Projected 4K per-agent session | Agents fit at 4K |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gpt2` | `128` tokens | `2.70 MB` | `3977` | `86.38 MB` | `124` |
| `Qwen/Qwen2.5-1.5B-Instruct` | `128` tokens | `2.02 MB` | `5307` | `64.74 MB` | `165` |

Projection rows are linear estimates, not direct 4K measurements.

## Claim boundary

HeliX should not claim Shkolnikov-scale TTFT numbers from this local evidence. The local run is intentionally small, CPU-only, and cache-only. The correct claim is:

> HeliX now reports Agent-Memory-style TTFT and memory-budget evidence while preserving its own differentiators: `.hlx` Rust persistence, pending vs verified audit states, Merkle/SHA-256 receipts, hybrid Zamba evidence, and multimodel session scheduling.

## What HeliX still needs from this lane

- True prefix matching over stored `session_token_ids`.
- Cross-phase context injection without recomputing previous turns.
- An OpenAI-compatible `/v1/chat/completions` entrypoint.
- Larger TTFT runs on GPU or Apple Silicon.
