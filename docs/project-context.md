# Helix Project Context

Last updated: 2026-03-30 (HF-native K/V scaling split + adaptive-asymmetric + HF session timing)
Workspace root: `C:\Users\Big Duck\proyectos\helix-backend-repo`

This file is the canonical cross-session handoff for the repo. The goal is that another tool, another agent, or a future session can open only this file and recover the current architecture, status, benchmarks, and next steps without rebuilding context from scratch.

## 1. What Helix Is

Helix is a local-first inference and agent stack with:

- a custom tensor runtime for exported Hugging Face models,
- a GGUF runtime through `llama-cpp-python`,
- a workspace model registry with aliases,
- three assistant roles: `general`, `code`, `legal`,
- local quality benchmarks and tool-calling benchmarks,
- an API with SSE streaming,
- a simple local frontend demo,
- ongoing R&D on KV-cache compression inspired by recent papers.

## 2. High-Level Status

Current overall state:

- The local base model is integrated and usable.
- Tool calling was fixed and promoted from `0%` to `75%` case success.
- The browser demo works locally.
- The best verified KV compression path today is our own TurboQuant-style line, not Josh's current packages.
- `turbo-int8` with Hadamard rotation is the current stable default.
- `turbo-4bit` is implemented and promising.
- `turbo-qjl` improved a lot, but is still experimental.
- A standalone `helix_kv` package now exists inside the repo with its own `pyproject`, README, and tests.
- `CompressedKVCache.load(path)` now restores config from `session.json`, so compressed sessions can be resumed without re-specifying `export_dir`.
- A remote benchmark runner now exists at `tools/remote_benchmark.py` with SSH/SCP dry-run validation.
- A real remote TinyLlama benchmark now confirms that `turbo-int8-hadamard` can be made much faster without losing generation match, and that adaptive mode becomes stable when the terminal layers are protected from overly aggressive 4-bit compression.
- The HF-native `benchmark-transformers-kv` path now uses a torch/CUDA compression backend instead of GPU->CPU->NumPy roundtrips for the main int8 path.
- The HF-native benchmark now reports native baseline (`bf16` on Qwen), fp32-equivalent reference ratios, and explicit GQA metadata (`num_attention_heads`, `num_key_value_heads`, `gqa_group_size`).
- The HF-native benchmark now supports `variant_set=stable|asymmetry-sweep`, including additive rows for `turbo-int8k-4bitv-perchannel`, `turbo-4bit-perchannel`, and `adaptive-asymmetric-m9-h20`.
- HF-native compressed caches now support real `save(path)` / `load(path, ...)` roundtrips, and the benchmark reports `session_save_time_ms` and `session_load_time_ms` in addition to size.
- Adaptive-asymmetric is now implemented for the HF route: each layer can choose `K_mode` and `V_mode` independently, and reports `layer_kv_mode_counts`.
- Real GPU benchmarks on `Qwen2.5-1.5B-Instruct`, `SmolLM2-1.7B-Instruct`, and `Qwen2.5-3B-Instruct` now exist under `verification/remote-*-transformers-gpu.json`.
- A long-context real GPU run for `Qwen2.5-3B-Instruct @ 4096/32` now exists at `verification/remote-qwen25-3b-transformers-gpu-4096.json`.
- The consolidated readout for the current HF-native GPU line is `verification/remote-transformers-gpu-summary.json`.
- A local asymmetry-sweep smoke report now exists at `verification/transformers-asymmetry-sweep-tiny/benchmark_report.json`.
- A local `Qwen2.5-1.5B-Instruct` 4-bit smoke now exists at `verification/transformers-4bit-perchannel-local-smoke.json`; the key result is honest and important: `turbo-4bit-perchannel` still fails the local promotion gate and remains worse than the current `turbo-int8k-4bitv` path.
- A local adaptive-policy benchmark now exists (`benchmark-gpt-kv-policy`) to compare static modes vs policy-driven switching.
- Layer-adaptive compression is now implemented and session-persistent.
- Asymmetric `K/V` compression is now benchmarked (`K=4bit`, `V=int8`) and keeps generation stable.
- Hot-window exact cache is now implemented as an architectural base for future speedups.
- Selective attention (approximate scoring + top-K sparse decode) is now implemented.
- Inter-step selective index caching is now implemented, with periodic full refresh.
- Block scoring for the cold prefix is now implemented as an optional experimental path.
- Incremental coarse-summary caching is now implemented for `turbo-int8`, `turbo-4bit`, and `turbo-qjl`, so block scoring no longer recomputes summaries from scratch after every spill.
- A unified 10-row KV landscape benchmark now exists in one JSON, combining runtime fidelity/speed with session-size data.
- A post-optimization remote comparison summary now exists at `verification/remote-optimization-summary.json`.
- The raw before/after remote reports are:
  - `verification/remote-tinyllama-1.1b.json`
  - `verification/remote-tinyllama-1.1b-opt2.json`

## 3. Core Repo Areas

Main repo:

- `C:\Users\Big Duck\proyectos\helix-backend-repo`

Important directories:

- Backend / CLI / API: `src/helix_proto`
- Standalone KV package source: `helix_kv`
- Standalone KV package metadata/tests: `helix-kv`
- Weight-compression bridge/runtime: `src/helix_substrate`
- Frontend demo: `frontend`
- Local run scripts: `scripts`
- Remote benchmark runner: `tools/remote_benchmark.py`
- Workspace models and assistants: `workspace`
- GGUF files: `workspace-gguf`
- Benchmarks and research outputs: `benchmark-output`, `verification`
- Tests: `tests`
- Docs / handoffs: `docs`

## 4. Local Base Model

Current promoted local model:

- Alias: `qwen35-4b-q4`
- File: `workspace-gguf\qwen35-4b-q4_k_m.gguf`
- Size: `2,708,804,000` bytes (`~2.71 GB`)

Status:

- Registered in workspace.
- Assigned as base for assistants.
- Original HF cache copy was removed after promotion.

Main benchmark report:

- `benchmark-output\qwen35-4b-q4-bench\benchmark_report.json`

Final local-assistant metrics:

- `general = 0.6667`
- `code = 0.6667`
- `legal = 0.2`
- `tokens_per_second = 2.7004`

Interpretation:

- `general` and `code` are good for a local CPU 4B Q4 model.
- `legal` is weak and should not be treated as premium legal quality.

## 5. Tool Calling History

Final report:

- `benchmark-output\qwen35-4b-q4-toolcall\tool_call_report.json`

Timeline:

1. Initial state:
   - `step_success_rate = 0.0`
   - `json_valid_rate = 0.0`
   - `case_success_rate = 0.0`

2. Intermediate fixes:
   - stripped `<think>...</think>`
   - more robust parser
   - prompt cleanup
   - still not enough

3. Main fix:
   - switched to the correct Qwen3.5 tool-calling template
   - parsed native `<tool_call>` outputs
   - aligned prompting with the model's real format

4. Final calibration:
   - less strict arg matching
   - more generous token budget
   - planner prompt tuning

Final metrics:

- `total_steps = 14`
- `step_success_rate = 0.7857`
- `json_valid_rate = 1.0`
- `case_success_rate = 0.75`
- `total_generated_tokens = 2693`
- `total_generation_time_s = 3252.3214`
- `tokens_per_second = 0.828`

Interpretation:

- The main failure was format and calibration, not fundamental model incapability.
- The model is now usable for local planner / tool-calling flows.

## 6. Local Demo Product

Current local demo includes:

- SSE streaming API endpoint,
- local frontend with live token rendering,
- usable browser demo for `general`, `code`, and `legal`.

Important files:

- API: `src/helix_proto/api.py`
- Frontend: `frontend/index.html`
- Local scripts:
  - `scripts/run-local-backend.ps1`
  - `scripts/run-local-frontend.ps1`
  - `scripts/start-local-demo.ps1`

Notes:

- TTFT and perceived responsiveness were improved.
- Thinking leaks were filtered aggressively.
- UX is much better than the raw model latency would suggest.
- The model still runs on CPU, so latency is real even when the UI feels better.

## 7. Verification of Josh's Repos

### 7.1 helix-substrate

Report:

- `verification/substrate-results.json`

Outcome:

- Compression works.
- Compression ratio is real and high.
- Packaging and practical integration are still rough.
- Fidelity on small models is worse than the attractive headline might suggest.

Key numbers:

- `conversion_seconds_cpu = 361.6767`
- `manifest_reported_ratio = 7.31`
- `adjusted_ratio_excluding_attn_bias_buffers = 6.6016`
- `output_total_bytes = 75487316`
- `logits.max_abs_err = 33.2527`
- `logits.mean_abs_err = 7.3328`
- `logits.cosine_similarity = 0.9961`
- `perplexity original = 1588.6476`
- `perplexity reconstructed = 2918.5029`
- `next_token_match = true`

Verdict:

- Real compression, but not clean enough or faithful enough to just drop in blindly.

### 7.2 helix-online-kv

Report:

- `verification/online-kv-results.json`

Outcome:

- Did not install from PyPI as documented.
- Worked only from source.
- Preserved output.
- Did not provide real memory savings in the verified implementation.
- Added very large time overhead.

Key numbers:

- `pip_install_from_pypi_works = false`
- `baseline_seconds = 2.7202`
- `compressed_seconds = 31.1496`
- `time_overhead_ratio = 11.4512`
- `output_match_exact = true`
- `baseline_exact_cache_bytes = 17326080`
- `effective_total_bytes_with_current_impl = 19156480`
- `effective_savings_vs_baseline = 0.9045`
- `memory_report_shows_real_savings = false`

Verdict:

- The operational claim did not hold in the verified state.
- We did not integrate it.

### 7.3 Direct comparison vs our turbo-int8

Report:

- `verification/kv-comparison.json`

Outcome:

- Our `turbo-int8` delivered the best real KV reduction among the verified options.

Key numbers:

- `helix_proto_fp32.kv_cache_bytes = 17399808`
- `helix_proto_turbo_int8.kv_cache_bytes = 4485888`
- `kv_compression_ratio_vs_helix_proto_fp32 = 3.8788`
- `output_match_vs_helix_proto_fp32 = true`
- `time_ratio_vs_helix_proto_fp32 = 1.0018`

## 8. KV Compression R&D

### 8.1 Original turbo-int8

Initial prototype characteristics:

- fixed orthogonal rotation,
- symmetric int8 quantization,
- on-the-fly dequantization,
- real `~3.56x` to `~3.88x` KV compression,
- very low error,
- identical output,
- essentially no overhead inside our runtime.

This was the base for the TurboQuant-style evolution.

### 8.1.1 Standalone package and policy benchmark

New repo-local package:

- `helix_kv`
- `helix-kv/pyproject.toml`
- `helix-kv/README.md`
- `helix-kv/tests/test_public_api.py`

Main public objects now exposed:

- `KVConfig`
- `CompressedKVCache`
- `AdaptiveKVPolicy`
- `build_adaptive_config(...)`
- `build_asymmetric_config(...)`
- `run_kv_landscape(...)`
- `run_adaptive_policy_benchmark(...)`

New local benchmark command:

- `python -m helix_proto.cli benchmark-gpt-kv-policy --output verification/gpt-kv-policy-benchmark`

Current local tiny-GPT2 policy benchmark:

- `verification/gpt-kv-policy-benchmark/benchmark_report.json`

Current result snapshot:

- `static-fp32`: `time_s=2.1128`, `kv_cache_bytes=5120`, `session_total_bytes=7431`
- `static-turbo-int8-hadamard`: `time_s=0.4646`, `kv_cache_bytes=3008`, `session_total_bytes=7682`
- `static-turbo-4bit`: `time_s=0.4812`, `kv_cache_bytes=2624`, `session_total_bytes=7950`
- `adaptive-policy`: `time_s=0.4728`, `kv_cache_bytes=2624`, `session_total_bytes=8647`, `switch_count=0`

Interpretation:

- the adaptive policy path is now implemented and benchmarked locally,
- on the tiny benchmark it stays stable and parseable,
- but this tiny workload does not force real mode upgrades/downgrades yet, so the result is infrastructure validation, not a final quality claim.

### 8.2 TurboQuant reference direction

Reference repo used for reading only:

- [OnlyTerp/turboquant](https://github.com/OnlyTerp/turboquant)

We used it as reference for:

- Hadamard rotation,
- Lloyd-Max scalar codebooks,
- QJL residual logic,
- overall algorithm shape.

We did **not** integrate it as a dependency.

### 8.3 Hadamard vs QR

Consolidated report:

- `verification/turboquant-evolution-benchmark.json`

Main finding:

- Hadamard improved fidelity materially at the same KV size.

Real GPT2 export benchmark:

- `turbo-int8-qr`
  - `kv_cache_bytes = 380160`
  - `max_abs_err = 0.6770`
  - `mean_abs_err = 0.5845`
  - `cosine_similarity = 0.9999997616`
  - `generated_match_vs_baseline = true`

- `turbo-int8-hadamard`
  - `kv_cache_bytes = 380160`
  - `max_abs_err = 0.1800`
  - `mean_abs_err = 0.0945`
  - `cosine_similarity = 0.9999998808`
  - `generated_match_vs_baseline = true`

Result:

- Hadamard became the default.

### 8.4 turbo-4bit

Status:

- Implemented.
- Experimental but functional.

Synthetic benchmark:

- `kv_cache_bytes = 13056`
- `compression_ratio_vs_fp32 = 7.5294`
- `max_abs_err = 0.04056`
- `mean_abs_err = 0.00761`
- `cosine_similarity = 0.99541`

Tiny GPT2 smoke:

- `verification/turboquant-cli-smoke-qjl-v3/benchmark_report.json`
- `kv_cache_bytes = 960`
- `max_abs_err = 0.014999`
- `cosine_similarity = 0.998512`
- `generated_match_vs_baseline = true`

Targeted GPT2 export:

- `verification/turboquant-qjl-targeted-gpt2.json`
- `kv_cache_bytes = 97920`
- `max_abs_err = 17.5399`
- `cosine_similarity = 0.9999039`
- `generated_match_vs_baseline = false`

Interpretation:

- Strong compression and promising practical behavior.
- Still not safe as the universal default.

### 8.5 turbo-qjl

#### First pass

Originally QJL was used as full residual vector reconstruction.

That was too noisy for our runtime path.

Historical tiny smoke:

- `verification/turboquant-cli-smoke/benchmark_report.json`
- `cosine_similarity = 0.7427`
- `max_abs_err = 0.3234`
- `generated_match_vs_baseline = false`

#### Current pass

Current changes:

- Gaussian QJL matrix,
- unit residual encoding,
- score correction on `K`,
- stable base decode for vectors instead of injecting the full residual into `V` materialization.

Improved tiny smoke:

- `verification/turboquant-cli-smoke-qjl-v3/benchmark_report.json`
- `cosine_similarity = 0.998512`
- `max_abs_err = 0.014994`
- `generated_match_vs_baseline = true`

Targeted GPT2 export:

- `verification/turboquant-qjl-targeted-gpt2.json`
- `kv_cache_bytes = 126720`
- `max_abs_err = 18.6271`
- `cosine_similarity = 0.9999075`
- `generated_match_vs_baseline = false`

Interpretation:

- QJL is much healthier than before.
- It no longer catastrophically breaks the tiny smoke.
- It still does not clearly beat `turbo-4bit` on the more representative GPT2 export benchmark.

### 8.6 Hot-window exact cache

New architecture:

- compressed cold prefix,
- exact float32 hot tail,
- exposed through `--kv-hot-window`.

Purpose:

- keep the most recent tokens exact,
- compress only the distant past,
- improve fidelity where attention is most sensitive,
- prepare for future partial-decoding / top-K refinement work.

Tiny smoke with `hot_window=4`:

- `verification/turboquant-cli-hotwindow4/benchmark_report.json`

Results:

- `turbo-4bit`
  - `kv_cache_bytes = 2624`
  - `prompt_perplexity = 61.743441`
  - `cosine_similarity = 0.999895`
  - `max_abs_err = 0.004684`
  - `generated_match_vs_baseline = true`

- `turbo-qjl`
  - `kv_cache_bytes = 2912`
  - `prompt_perplexity = 61.743441`
  - `cosine_similarity = 0.999895`
  - `max_abs_err = 0.004668`
  - `generated_match_vs_baseline = true`

Targeted GPT2 export with `hot_window=4`:

- `verification/turboquant-hotwindow4-targeted-gpt2.json`

Results:

- `turbo-4bit-hot4`
  - `kv_cache_bytes = 353664`
  - `max_abs_err = 10.5114`
  - `mean_abs_err = 5.0249`
  - `cosine_similarity = 0.9999177`
  - `generated_match_vs_baseline = false`

- `turbo-qjl-hot4`
  - `kv_cache_bytes = 370944`
  - `max_abs_err = 10.4571`
  - `mean_abs_err = 4.9736`
  - `cosine_similarity = 0.9999175`
  - `generated_match_vs_baseline = false`

Interpretation:

- Hot-window improves fidelity noticeably.
- It still does not give real speedup, because we still materialize the compressed cold prefix in full.
- The architecture is now ready for the next real acceleration step.

### 8.7 Layer-adaptive compression

New capability:

- profile `K` and `V` kurtosis per layer during calibration,
- bucket layers by thresholds,
- persist the profile inside session metadata,
- restore the same per-layer decisions on resume.

Current thresholds:

- kurtosis `>= 10` -> `fp32`
- kurtosis `>= 3` and `< 10` -> `turbo-int8`
- kurtosis `< 3` -> `turbo-4bit`

Primary benchmark:

- `verification/layer-adaptive-benchmark-v2/benchmark_report.json`

Results on the current 4-layer tiny GPT2 export:

- `adaptive`
  - `kv_cache_bytes = 19456`
  - `total_time_s = 8.9909`
  - `cosine_similarity = 1.0`
  - `max_abs_err = 0.000065`
  - `generated_match_vs_baseline = true`
  - selected layer modes:
    - layer 0 -> `turbo-int8`
    - layer 1 -> `turbo-int8`
    - layer 2 -> `turbo-int8`
    - layer 3 -> `turbo-int8`

Interpretation:

- The adaptive machinery works.
- Session save/load now preserves the kurtosis profile and chosen modes.
- On this random tiny GPT2, all layers landed in the medium bucket, so adaptive currently collapses to all-int8.
- The logic is ready; bigger or less uniform models are more likely to show mixed per-layer modes.

### 8.8 Asymmetric K/V compression

New capability:

- `K` and `V` can now use different compression modes while keeping the same runtime/session format.
- Current benchmarked variant: `K = turbo-4bit`, `V = turbo-int8`.

Primary benchmark:

- `verification/layer-adaptive-benchmark-v2/benchmark_report.json`

Results:

- `turbo-int8-hadamard`
  - `kv_cache_bytes = 19456`
  - `total_time_s = 8.3622`
  - `cosine_similarity = 1.0`
  - `max_abs_err = 0.000079`
  - `generated_match_vs_baseline = true`

- `turbo-4bitk-int8v`
  - `kv_cache_bytes = 16384`
  - `total_time_s = 9.5852`
  - `cosine_similarity = 0.9999990`
  - `max_abs_err = 0.000520`
  - `generated_match_vs_baseline = true`

Interpretation:

- The asymmetric path is a real memory win over pure int8 (`19456 -> 16384`, about `1.19x` smaller).
- Fidelity stayed effectively perfect on this benchmark.
- It is not yet a speed win on CPU/NumPy; it is slightly slower than symmetric int8 in this runtime.

## 9. Current Compression Status

Supported KV modes today:

- `fp32`
- `turbo-int8`
- `turbo-4bit`
- `turbo-qjl`
- `adaptive`

Current practical recommendation:

- Stable default: `turbo-int8` + Hadamard
- Experimental but promising: `turbo-4bit`
- Experimental R&D path: `turbo-qjl`
- Experimental but useful: asymmetric `K/V` (`K=4bit`, `V=int8`)
- Adaptive path: implemented and persistent, but still waiting for richer per-layer separation on larger models
- New architecture layer ready: `--kv-hot-window`

## 10. Important Files to Read First

If you need to resume coding, start here:

1. `docs/project-context.md`
2. `docs/helix-project-stack-summary.txt`
3. `src/helix_proto/hf.py`
4. `src/helix_proto/cli.py`
5. `tests/test_hf_kv_quant.py`

If you need product/demo context:

1. `src/helix_proto/api.py`
2. `frontend/index.html`
3. `docs/session-handoff-local-demo.md`

If you need benchmark evidence:

1. `benchmark-output/qwen35-4b-q4-bench/benchmark_report.json`
2. `benchmark-output/qwen35-4b-q4-toolcall/tool_call_report.json`
3. `verification/turboquant-evolution-benchmark.json`
4. `verification/turboquant-cli-hotwindow4/benchmark_report.json`
5. `verification/turboquant-hotwindow4-targeted-gpt2.json`

## 11. Commands Worth Remembering

PowerShell backend:

```powershell
Set-Location "C:\Users\Big Duck\proyectos\helix-backend-repo"
$env:PYTHONPATH = "src"
python -m helix_proto.cli serve-api --workspace-root workspace --host 127.0.0.1 --port 8000
```

PowerShell frontend:

```powershell
Set-Location "C:\Users\Big Duck\proyectos\helix-backend-repo"
python -m http.server 3000 --bind 127.0.0.1 --directory frontend
```

KV benchmark smoke:

```powershell
$env:PYTHONPATH = "src"
python -m helix_proto.cli benchmark-gpt-kv-modes --output verification/turboquant-cli-hotwindow4 --kv-hot-window 4 --include-qjl
```

Selective attention benchmark:

```powershell
$env:PYTHONPATH = "src"
python -m helix_proto.cli benchmark-gpt-kv-modes --output verification/selective-attention-benchmark --kv-hot-window 4 --kv-topk 8 --include-qjl
```

Resume smoke for hot-window:

```powershell
$env:PYTHONPATH = "src"
python -m helix_proto.cli demo-gpt-resume --output verification/turboquant-hotwindow-resume-smoke --kv-mode turbo-4bit --kv-hot-window 2
```

Session-size benchmark:

```powershell
$env:PYTHONPATH = "src"
python -m helix_proto.cli benchmark-gpt-session-size --output verification/session-size-benchmark-v2 --num-layers 1 --num-heads 2 --hidden-size 32 --prompt-lengths 128 512 1024 --max-new-tokens 1 --kv-hot-window 4
```

## 12. Tests Last Verified

Last verified green:

- `pytest tests/test_hf_kv_quant.py -q` -> `34 passed`
- `pytest tests/test_cli_smoke.py -q` -> `7 passed`

## 13. What Is Stable vs Experimental

Stable enough for practical product work:

- local model registration and workspace aliasing,
- GGUF backend through llama-cpp-python,
- assistants API + SSE streaming,
- local browser demo,
- tool-calling benchmark path,
- `turbo-int8` with Hadamard default,
- adaptive layer profiling + session persistence,
- asymmetric `K/V` runtime/session support,
- compressed session format v2 with codec artifacts persisted on disk,
- hot-window session save/resume path.
- inter-step selective shortlist reuse with periodic refresh.
- block scoring as an opt-in selective path that preserves `match=True` in the tuned int8 benchmark.
- incremental block-summary caches for compressed KV append paths.

Still experimental / research:

- `turbo-4bit` as a universal default,
- `turbo-qjl` as a meaningful win over `turbo-4bit`,
- selective attention (topk decode) as a default - functional but needs production benchmarks,
- adaptive per-layer mode selection on real exported models beyond tiny GPT2.

## 14. Selective Attention Architecture (Implemented)

The selective attention system implements the recommended next step: **approximate scoring on the cold compressed prefix → top-K selection → exact decode only on selected tokens → combine with exact hot window**.

### Pipeline per step

1. **Approximate scores**: Compute `Q·K^T` directly in the compressed/rotated domain, without materializing the full cold prefix to float32. For turbo-int8; `(R·Q) · (int8_data * scales)`. For turbo-4bit: `(R·Q) · (codebook_dequant * norms)`. For turbo-qjl: base 4-bit scores + QJL score correction.

2. **Top-K Selection**: Use `np.argpartition` (O(N) average) to select the top-K most relevant cold tokens per attention head.

3. **Selective Materialization**: Materialize only the selected cold tokens to float32 — for int8 this avoids full inverse rotation on unselected tokens.

4. **Exact Attention**: Compute exact attention on `[selected_cold, hot_window, new_token]`, bypassing the vast majority of the cold prefix.

5. **Fallback**: When `kv_topk=0` (default) or `kv_topk >= cold_length`, falls back to full materialization (exact behavior preserved).

### New CLI parameter

- `--kv-topk N` — activates selective attention with top-N cold tokens per head.
- Only meaningful when `--kv-hot-window > 0` and a compressed KV mode is active.
- `--kv-index-refresh-interval N` — forces a full cold-prefix rescan every `N` selective steps; in between, the runtime reuses the previous shortlist and only adds newly spilled cold tokens.
- `--kv-block-size N` — enables coarse block scoring on the cold prefix before exact token-level shortlist refine.

### Key Implementation Details

- `_gpt2_step_with_selective_kv()` in `hf.py` — the core selective attention step function
- `approximate_scores()` and `materialize_indices()` methods on all KV array classes
- `_HotWindowKVArray.supports_selective`, `.cold_approximate_scores()`, `.cold_materialize_indices()`
- Benchmark variants `turbo-int8-topk{N}`, `turbo-4bit-topk{N}`, `turbo-qjl-topk{N}` auto-added when `--kv-topk > 0`
- `turbo-4bit` selective materialization now gathers packed rows first and unpacks only the selected top-K rows, instead of scanning the full cold prefix before gather
- `turbo-4bit` / `turbo-qjl` selective routing now uses an exact-refine shortlist: approximate top-candidates are materialized, rescored exactly, and only then trimmed to the final top-K set
- Selective activation is now mode-aware: pure `turbo-4bit` uses a more conservative cold-prefix threshold on CPU/NumPy, while `turbo-int8` and `turbo-qjl` activate earlier
- Inter-step shortlist reuse is now implemented: after a full cold-prefix scan, the runtime caches the shortlist indices and rescoring only touches the cached candidates plus the newly spilled cold tokens until the next refresh
- Block scoring is now implemented for selective runs: the runtime can first score block summaries, keep only the best blocks, and then do exact shortlist refinement inside those blocks
- Session resume now validates KV metadata (`precision`, `seed`, `rotation`, `hot_window`, `topk`, `index_refresh_interval`, `block_size`) before loading persisted cache state
- Important current nuance: when `kv_topk > 0` but the adaptive threshold disables sparse decode, the engine still follows the incremental hot-window append path. This is not bit-identical to the older dense re-store path, and should be benchmarked as its own behavior rather than assumed equivalent

### End-to-End Speedup Optimizations

Initial implementations of selective attention achieved speedups in isolation but failed to deliver end-to-end pipeline speedups due to framework overhead. Two critical optimizations were implemented to solve this:

1. **Zero-Copy Cache Append**: The step function (`_gpt2_step_with_selective_kv`) now appends new generated tokens directly to the `_HotWindowKVArray` without materializing or re-compressing the cold prefix. This `append_token()` strategy ensures the compression cost is strictly O(1) per step.
2. **Gather-before-rotation for 4-bit**: Materializing sparse indices from `_Turbo4BitKVArray` now unpacks the nibbles to rotated float32s, gathers the selected subset, and *only then* applies the expensive inverse Hadamard rotation (`O(topk)` instead of `O(N)`).

### E2E Benchmarks & Results

A new comprehensive suite evaluates sequence lengths from 32 up to 512 tokens with various `topk` sizes. The results demonstrate massive speedups:
- **Int8**: Up to **~11.8x speedup** on 512-token prompts with `topk=4`, and **~10x** with `topk=8` (retaining 0.70 cosine fidelity).
- **4-bit**: Up to **~3.3x speedup** on 512-token prompts with `topk=4` (limited by bitwise unpacking scanning the full prefix, but saved by the gather-before-rotation optimization).
- `verification/selective-threshold-probe-v2/benchmark_report.json`: shortlist refine repaired `turbo-4bit-topk4` back to `match=True` on the focused 32-token probe.
- `verification/selective-long56-v3/benchmark_report.json`: long-prefix probe for the current adaptive-gating implementation; useful for comparing `turbo-int8`, pure `turbo-4bit`, and selective variants on the same exported tiny GPT2.
- `verification/selective-threshold-equivalence-probe.json`: confirms that pure `turbo-4bit` with `kv_topk=8` kept sparse decode disabled (`selective_ever_enabled=false`) for a 56-token prompt, but still diverged numerically from the older full re-store path because the incremental hot-window append path is different.
- `verification/selective-index-cache-compare/benchmark_report.json`: controlled comparison of full-refresh-every-step vs inter-step shortlist reuse on the same export and prompt.

### Inter-step index cache result

On the controlled `turbo-int8-topk8` comparison:

- `refresh=1`
  - `total_time_s = 16.2655`
  - `cosine_similarity = 0.5738`
  - `max_abs_err = 0.3501`
  - `generated_match_vs_baseline = false`
  - `full_refreshes = 112`
  - `reuse_hits = 0`

- `refresh=8`
  - `total_time_s = 15.8204`
  - `cosine_similarity = 0.9772`
  - `max_abs_err = 0.0779`
  - `generated_match_vs_baseline = true`
  - `full_refreshes = 16`
  - `reuse_hits = 96`

Interpretation:

- Reusing the shortlist is now materially changing the selective path, not just bookkeeping.
- On this benchmark it sharply reduces full cold-prefix rescans and improves fidelity.
- The speed gain is modest in Python/NumPy, but the architectural win is real: most selective steps now avoid a full cold scan.

### Block scoring result

Primary new reports:

- `verification/block-scoring-compare-tuned/benchmark_report.json`
- `verification/block-scoring-long80/benchmark_report.json`
- `verification/block-summary-cache-compare/benchmark_report.json`
- `verification/block-summary-cache-compare/benchmark_report_long112.json`
- `verification/gpt-kv-landscape/benchmark_report.json`

Tuned focused comparison on `turbo-int8-topk8-refresh8`:

- baseline selective, no block scoring
  - `total_time_s = 12.3119`
  - `cosine_similarity = 0.9772`
  - `max_abs_err = 0.0779`
  - `generated_match_vs_baseline = true`

- `block_size = 8`
  - `total_time_s = 15.7024`
  - `cosine_similarity = 0.9806`
  - `max_abs_err = 0.0713`
  - `generated_match_vs_baseline = true`
  - `block_pruned_steps = 16`
  - `block_rows_scored = 88`

- `block_size = 16`
  - `total_time_s = 13.8180`
  - `cosine_similarity = 0.9817`
  - `max_abs_err = 0.0672`
  - `generated_match_vs_baseline = true`
  - `block_pruned_steps = 16`
  - `block_rows_scored = 48`

Longer prompt probe:

- baseline selective, no block scoring
  - `total_time_s = 16.0764`
  - `cosine_similarity = 0.9639`
  - `max_abs_err = 0.0831`
  - `generated_match_vs_baseline = true`

- `block_size = 16`
  - `total_time_s = 21.3233`
  - `cosine_similarity = 0.9632`
  - `max_abs_err = 0.1000`
  - `generated_match_vs_baseline = true`
  - `block_pruned_steps = 28`
  - `block_rows_scored = 104`

Interpretation:

- Block scoring is now functionally stable on the tuned int8 selective path.
- It prunes cold-prefix work and preserves `match=True` after exact shortlist refine.
- In the current CPU Python/NumPy runtime it is still not a net speed win end-to-end.
- That makes it a valid experimental architecture, but not a default recommendation yet.

### Incremental coarse-summary cache result

New capability:

- block summaries now persist across `append_compressed()` instead of being dropped on every spill from the hot window to the cold prefix,
- the append path updates only the affected tail block plus any new full blocks,
- this is implemented for `turbo-int8`, `turbo-4bit`, and `turbo-qjl`.

Primary new reports:

- `verification/block-summary-cache-compare/benchmark_report.json`
- `verification/block-summary-cache-compare/benchmark_report_long112.json`

Key result:

- On the `112` token probe:
  - `turbo-int8-topk8-refresh8`
    - `total_time_s = 20.6210`
    - `generated_match_vs_baseline = false`
    - `cosine_similarity = 0.3937`
  - `turbo-int8-topk8-refresh8-block8`
    - `total_time_s = 20.9188`
    - `generated_match_vs_baseline = true`
    - `cosine_similarity = 0.9362`
  - `turbo-int8-topk8-refresh8-block16`
    - `total_time_s = 17.0889`
    - `generated_match_vs_baseline = true`
    - `cosine_similarity = 0.8931`

Interpretation:

- Incremental cached coarse stats materially changed the block-scoring tradeoff.
- `block16` is now faster than the non-block selective path on a longer prefix, while also restoring `match=True` in this probe.
- The win is still workload-sensitive: shorter prompts may remain neutral or slightly slower.
- This is the first sign that cached coarse statistics can produce real end-to-end savings in the CPU runtime, not just architectural cleanliness.

### Tests

New tests added (all 36 passing in the current file; CLI smoke 9 passing):

- `test_turbo_int8_approximate_scores_match_exact_scores`
- `test_turbo_4bit_approximate_scores_match_exact_scores`
- `test_turbo_int8_materialize_indices_returns_correct_subset`
- `test_hot_window_supports_selective_with_compressed_cold`
- `test_cli_parser_accepts_kv_topk_flag`
- `test_hot_window_append_token_preserves_cold`
- `test_turbo_4bit_optimized_materialize_matches_full`
- `test_turbo_4bit_materialize_indices_unpacks_only_selected_rows`
- `test_selective_helpers_use_thresholds_and_shortlist_expansion`
- `test_merge_selective_candidate_indices_adds_new_cold_tokens`
- `test_selective_index_cache_records_reuse_hits`
- `test_selective_index_refresh_interval_one_disables_reuse`
- `test_turbo_int8_approximate_block_scores_shape`
- `test_expand_block_indices_expands_each_selected_block`
- `test_selective_block_scoring_records_pruned_steps`
- `test_turbo_int8_append_preserves_block_summary_cache`
- `test_turbo_4bit_append_preserves_block_summary_cache`
- `test_turbo_qjl_append_preserves_block_summary_cache`
- `test_cross_layer_overlap_records_adjacent_pair_stats`
- `test_cross_layer_share_records_share_hits`

## 15. Cross-Layer Overlap and Sharing (Block 5)

New capability:

- the selective attention runner now records adjacent-layer Jaccard overlap for the final selected cold-prefix top-K,
- `kv_layer_share_stride` is wired end-to-end through the runtime, CLI, benchmark reports, and session metadata,
- the unified landscape benchmark can append explicit `shareN` rows when inter-layer sharing is enabled.

Primary new reports:

- `verification/cross-layer-overlap-probe/benchmark_report.json`
- `verification/gpt-kv-landscape-share4/benchmark_report.json`

Key overlap measurement (no sharing, 4-layer tiny GPT2, prompt length 112, `topk=8`, `refresh=8`, `block16`):

- `turbo-int8-topk8-refresh8-block16`
  - `total_time_s = 11.1168`
  - `generated_match_vs_baseline = true`
  - `mean_jaccard = 0.3557`
  - `high_overlap_rate = 0.0342`
  - adjacent-pair means:
    - layer `0→1`: `0.3369`
    - layer `1→2`: `0.3581`
    - layer `2→3`: `0.3721`

Interpretation:

- On this toy GPT2, the natural overlap between adjacent layers is far below the `~70%` reported by IndexCache for DSA-style models.
- That means cross-layer sharing is **not** yet justified here as a default optimization on the strength of overlap alone.

Experimental sharing result (`share4`) on the same probe:

- `turbo-int8-topk8-refresh8-block16-share4`
  - `total_time_s = 11.0509`
  - `generated_match_vs_baseline = true`
  - `full_refreshes = 10` vs `40` without sharing
  - `cross_layer_share_hits = 234`
  - `cross_layer_share_candidate_rows = 3000`

Interpretation:

- The current sharing path is stable and can reduce full rescans substantially.
- But its success here comes from reusing prior-layer shortlist structure plus exact rescore, not from naturally high adjacent-layer overlap.
- Treat `share4` as an experimental variant, not a promoted default.

Consolidated landscape with sharing:

- `verification/gpt-kv-landscape-share4/benchmark_report.json` now includes 12 rows:
  - the original 10-mode story matrix,
  - plus `turbo-int8-topk8-refresh8-block16-share4`,
  - plus `turbo-4bitk-int8v-topk8-refresh8-block16-share4`.

Most relevant rows from that report:

- `turbo-int8-topk8-refresh8`
  - `total_time_s = 9.5286`
  - `speedup_vs_fp32 = 1.0167`
  - `generated_match_vs_baseline = true`
  - `mean_jaccard = 0.3992`
- `turbo-int8-topk8-refresh8-block16`
  - `total_time_s = 10.0279`
  - `speedup_vs_fp32 = 0.9661`
  - `generated_match_vs_baseline = true`
  - `mean_jaccard = 0.3871`
- `turbo-int8-topk8-refresh8-block16-share4`
  - `total_time_s = 10.5511`
  - `speedup_vs_fp32 = 0.9182`
  - `generated_match_vs_baseline = true`
  - `mean_jaccard = 0.9154` (after sharing is imposed)

Landscape interpretation:

- On this benchmark, plain `refresh8` selective int8 remains the strongest selective default.
- `share4` preserves correctness but is slower end-to-end on this workload.
- The overlap instrumentation is still valuable because it now gives a clean decision rule: only promote sharing on workloads/models where the measured natural adjacent-layer Jaccard is actually high.

### Research Background

Inspired by:

- **TurboQuant (Google, 2026)** — PolarQuant + QJL 3-bit with 8x speedup. Our turbo-4bit + QJL is an adaptation.
- **RocketKV (NVIDIA, 2025)** — Two-stage coarse→fine attention with Hierarchical Sparse Attention.
- **Expected Attention (2025)** — Analytical importance scoring from future query distributions.
- **`np.argpartition`** — O(N) top-K selection, critical for CPU efficiency.

## 16. Next Recommended Technical Step

The selective attention architecture now includes block-wise selection, shortlist reuse, and incremental coarse-summary caching. The next steps are:

- **Validate overlap on a less toy-like model** - the tiny GPT2 probe does not show IndexCache-style `70%+` overlap; repeat on a larger or more structured model before promoting sharing.
- **Promote cached coarse stats selectively** - keep validating where `block16` is now a real win and where it is still neutral or worse.
- **Unify exact hot-window update paths**: the incremental append path used by selective/hot-window runs is not numerically identical to the older dense re-store path; decide whether to promote incremental append as the canonical exact path.
- **Cross-layer / cached coarse statistics** - if overlap is high, reuse shortlist structure or block summaries across adjacent layers and nearby generation steps.
- **Llama.cpp integration**: Qwen3.5-4B uses GGUF which bypasses the Python engine. Apply these architectural findings (coarse approximate scores + exact hot window) directly in C++ for `llama.cpp` to accelerate the agentic tool-use loops.

## 17. Bottom-Line Summary

Helix is already a usable local product stack with:

- a promoted local base model,
- working tool calling,
- a browser demo,
- a custom runtime,
- real benchmark evidence,
- a live KV-compression R&D track that already outperformed the currently verified external alternatives in practice,
- and a selective attention system that can produce real speedups through approximate scoring + top-K sparse decode.

The product is real. The compression is interesting. The selective attention is the first step toward true inference speedups from compressed KV caches.

## 18. Transformers-Native KV Benchmark

There are now two explicitly separated benchmarking tracks:

- `benchmark-transformers-kv` is the **real-model Hugging Face track**. It uses `AutoModelForCausalLM.from_pretrained()` directly and passes a Transformers-native compressed cache implementation as `past_key_values`, so it no longer depends on exporting the model into the GPT-only Helix runtime.
- `benchmark-gpt-landscape` remains the **legacy/export runtime track** for the custom GPT2-style engine, selective attention research, and session-format experiments.

Key files:

- [helix_kv/transformers_cache.py](/Users/Big Duck/proyectos/helix-backend-repo/helix_kv/transformers_cache.py)
- [src/helix_proto/cli.py](/Users/Big Duck/proyectos/helix-backend-repo/src/helix_proto/cli.py)
- [tools/remote_benchmark.py](/Users/Big Duck/proyectos/helix-backend-repo/tools/remote_benchmark.py)

Current scope of the Transformers-native benchmark:

- Supports dense compressed-cache variants:
  - `fp32`
  - `turbo-int8-qr`
  - `turbo-int8-hadamard`
  - `turbo-4bit`
  - `adaptive`
  - `turbo-int8k-4bitv`
  - optional `turbo-qjl`
- Reports:
  - prompt perplexity
  - last-logit comparison vs baseline
  - generated-match vs baseline
  - timing / tokens per second
  - logical KV cache bytes
  - per-layer kurtosis profile for `adaptive`
- Does **not** currently provide selective-attention acceleration through Hugging Face attention kernels, so:
  - `kv_topk`
  - `block_scoring`
  - `layer_share`
  are marked unsupported in this track.

Smoke validation:

- Local CLI smoke passes on a tiny local GPT2 HF model and writes:
  - [verification/transformers-kv-smoke/benchmark_report.json](/Users/Big Duck/proyectos/helix-backend-repo/verification/transformers-kv-smoke/benchmark_report.json)
- Real-model remote validation now exists for:
  - [verification/remote-qwen25-1.5b-transformers.json](/Users/Big Duck/proyectos/helix-backend-repo/verification/remote-qwen25-1.5b-transformers.json)
  - It proves the new benchmark path works on a real Qwen `Transformers` model without export/fallback.
  - Important caveat: that specific remote run executed on `cpu`, not `cuda`, because the rented host exposed an old-enough driver / small-enough disk combination that prevented a clean CUDA-12.8 PyTorch reinstall. So fidelity / compression numbers are still useful there, but speed numbers are not representative of GPU performance yet.

Recommended use:

- Use `benchmark-transformers-kv` for real HF models like Qwen2.5 / SmolLM / Phi.
- Keep GGUF + `llama.cpp` on the separate product path for the browser demo and assistants.
