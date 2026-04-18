# HeliX Session OS Reliability Gauntlet

This pass tests whether the Session OS layer behaves like operational infrastructure, not just a neat demo.

## What It Covers

- `MemoryCatalog` concurrency under SQLite WAL.
- Memory selection under tight token budgets with `importance` and `decay_score`.
- `.hlx` layer-slice chaos testing with full verification before injection.
- A short Rust/Python save-load soak for obvious RSS drift.
- Optional real AirLLM smoke, skipped cleanly unless the dependency and a local model path already exist.

## Commands

```powershell
python tools\run_local_session_os_reliability.py --suite all --output-dir verification
python tools\run_local_session_os_reliability.py --suite ffi-soak --duration-seconds 900 --output-dir verification
python tools\run_local_airllm_real_smoke.py --local-files-only --output-dir verification
```

## Artifacts

- `verification/local-memory-catalog-concurrency.json`
- `verification/local-memory-decay-selection.json`
- `verification/local-hlx-layer-chaos.json`
- `verification/local-rust-python-layer-slice-soak.json`
- `verification/local-airllm-real-smoke.json`

## Reading The Results

- `lost_observations=0` and `write_errors=0` means the local WAL pattern preserved concurrent catalog writes.
- `critical_retained_all=true` means the recall budget kept the highest-priority memories.
- `full_verify_blocked_injection=true` means the layer injector did not return a corrupted `.hlx` slice under full verification.
- `rss_delta_bytes` is a practical drift signal, not a formal allocator proof.
- `local-airllm-real-smoke.json` is allowed to be skipped on this laptop. A skip is not a failed HeliX claim.

## Claim Boundaries

Verified claims should use `verify_policy="full"` or a completed deferred audit. `receipt-only` remains the hot-path mode for speed, but it does not prove cryptographic integrity by itself.

The AirLLM lane remains adapter-first. This pass does not fork AirLLM, inject raw PyTorch pointers, or claim giant-model execution.
