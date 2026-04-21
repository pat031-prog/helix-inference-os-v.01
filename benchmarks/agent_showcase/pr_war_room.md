# HeliX PR War Room Fixture

Incident: `rust-hlx` sessions restore correctly, but saves are still slow.

Current evidence:
- GPT-2 session restore is deterministic: `hash_match=true`, `generated_ids_match=true`, `max_abs_logit_delta=0.0`.
- Qwen2.5-1.5B restore is deterministic with the same checks.
- Zamba2 hybrid restore also matches on the short local lane.
- Hypervisor v0 switched 5 agents across 2 rounds with `all_restore_hash_matches=true`.

Performance clue:
- The old `.hlx` path writes NumPy arrays to temporary raw files, then Rust reads those files and writes `kv_cache.hlx`.
- The new target is `rust-hlx-buffered`: Python passes memoryview/PyBuffer byte views to Rust, and Rust writes the bundle directly.
- The public claim must not say "5ms" unless the artifact proves p50 save time is at or below 5ms.

Mini diff:
```diff
- contiguous.tofile(staging_dir / relative)
- receipt = module.pack_hlx(str(staging_dir), str(session_json), str(destination))
+ spec = {"name": name, "dtype": str(array.dtype), "buffer": memoryview(array).cast("B")}
+ receipt = module.pack_hlx_buffers(meta_json, str(destination), specs)
```

Release question:
Can we ship this as a local public build story without overclaiming speed?
