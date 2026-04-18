# HeliX Stress Missions v1

Stress Missions v1 is the local, artifact-backed stress lab for the hybrid Zamba2 runtime. It is intentionally budgeted for the `laptop-12gb` profile: CPU, serial runs, compact JSON/JSONL receipts, and no dense tensor dumps.

The goal is not to add another compression feature. The goal is to connect memory savings with operational behavior: persistence, restore integrity, deterministic continuation, and graceful fallback when INT4 is too aggressive.

## Commands

Run the full local stress suite:

```powershell
python tools/run_local_hybrid_stress.py --mission all --profile laptop-12gb --output-dir verification --device cpu --local-files-only
```

Run only the restore equivalence probe:

```powershell
python tools/run_local_hybrid_stress.py --mission restore-equivalence --profile laptop-12gb --output-dir verification --device cpu --local-files-only
```

Open the Rust playback TUI after building `helix-watch`:

```powershell
.\crates\helix-watch\target\x86_64-pc-windows-gnullvm\release\helix-watch.exe .\verification\local-zamba2-stress-dashboard.json
```

If the Rust target directory was cleaned, rebuild it first:

```powershell
cargo +stable-x86_64-pc-windows-gnullvm build --release --target x86_64-pc-windows-gnullvm --manifest-path crates\helix-watch\Cargo.toml
```

## Mission Results

| Mission | Artifact | Result | Correct reading |
| --- | --- | --- | --- |
| Long-Context Coder | [`verification/local-zamba2-stress-long-context.json`](../verification/local-zamba2-stress-long-context.json) | `1.42x` best runtime-cache reduction, `1.04x` best speedup, `0/2` identifier recall | Compression works, but this is not a quality claim yet. The model did not recover the hidden identifiers. |
| State Juggler | [`verification/local-zamba2-stress-state-juggler.json`](../verification/local-zamba2-stress-state-juggler.json) | `hash_match=true`, `41,287,089` byte session, `3585.86ms` save, `748.61ms` load | The serialized/restored hybrid snapshot is bit-perfect. This proves integrity, not semantic understanding by itself. |
| Context Switcher | [`verification/local-zamba2-stress-context-switcher.json`](../verification/local-zamba2-stress-context-switcher.json) | `8,040,800` promoted blocks, `logits_finite=true`, ratio dropped to `1.43x` | The runtime sacrificed compression to stay numerically safe. That is graceful degradation, not a failure. |
| Restore Equivalence | [`verification/local-zamba2-stress-restore-equivalence.json`](../verification/local-zamba2-stress-restore-equivalence.json) | `hash_match=true`, `generated_ids_match=true`, `top1_match_all=true`, `max_abs_logit_delta=0.0` | A short deterministic continuation from the restored state matched the pre-restore continuation. This is still a probe, not a broad semantic eval. |

The dashboard artifact aggregates all four:

- [`verification/local-zamba2-stress-dashboard.json`](../verification/local-zamba2-stress-dashboard.json)

## What The Hash Proves

The SHA-256 hash match proves that the serialized snapshot and the restored snapshot are bit-perfect identical under the current snapshot format.

That matters because the hybrid session includes both memory regimes:

- Transformer KV cache for Transformer layers.
- Recurrent Mamba state for Mamba layers.

The hash does not prove that the model understands the same thing after restore. It proves that the stored/restored state is identical. The restore-equivalence mission then adds a behavioral probe: a deterministic continuation from the restored state produced the same generated ids, top-1 ids, finite logits, and zero measured logit delta.

## What Restore Equivalence Proves

Restore Equivalence is stronger than hash-only integrity, but still narrow:

- It builds a hybrid state.
- It saves a snapshot.
- It generates a short deterministic continuation from the in-memory cache.
- It loads the snapshot in a new Python process.
- It generates the same continuation from the restored cache.
- It compares generated token ids, top-1 ids, finitude, and logit deltas.

Current result:

```json
{
  "hash_match": true,
  "generated_ids_match": true,
  "top1_match_all": true,
  "max_abs_logit_delta": 0.0,
  "mean_abs_logit_delta": 0.0,
  "finite_before": true,
  "finite_after": true
}
```

This supports a precise public claim: HeliX can restore a complete hybrid session and reproduce a short deterministic continuation in the measured local probe.

It does not yet support a broad claim about long-context reasoning, semantic memory, or agent performance.

## Receipts

Mission receipts are compact compressed JSONL files. They are hash-chained and contain telemetry, not tensor dumps:

- `run_id`
- `layer_index`
- `token_index`
- `state_kind`
- `dense_bytes`
- `compressed_bytes`
- `ratio`
- `rel_rmse`
- `clip_pct`
- `finite_before`
- `finite_after`
- `fallback_precision`
- `fallback_reason`
- `block_count`
- `int4_block_count`
- `int8_block_count`
- `dense_block_count`
- `promoted_block_count`
- `max_abs_value`
- `state_norm`
- `prev_hash`
- `receipt_hash`

This keeps the evidence small enough for the laptop while still making failures inspectable.

## Limits

- The target is `Zyphra/Zamba2-1.2B-Instruct-v2` on CPU under the `laptop-12gb` profile.
- HXQ is not reopened in this suite; it remains a blocked lane until its local loader/runtime issue is fixed.
- `Zamba2-2.7B` is out of local scope for this machine.
- Long-Context Coder is intentionally staged, not a full 100k-token or GPU long-context benchmark.
- `helix-watch` is playback-only in v1. It reads artifacts; it does not run inference or stream live telemetry.
