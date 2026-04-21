# Transcript: bounded-context-under-depth

- Run ID: `infinite-depth-memory-baseline-smoke-02`
- Judge requested: `local/deterministic-measurer`
- Judge actual: `local/deterministic-measurer`
- Auditor requested: `local/deterministic-scorer`
- Auditor actual: `local/deterministic-scorer`

## Expected / Ground Truth

```json
{
  "depth": 128,
  "anchor_memory_id": "mem-bounded-context-under-depth-000121",
  "budget_tokens": 800,
  "limit": 5,
  "max_bounded_context_ms": 1000.0,
  "max_context_to_full_text_ratio": 0.05
}
```

## Visible Contract

```json
{
  "deterministic_suite": true,
  "case": "bounded-context-under-depth",
  "protocol": {
    "null_hypothesis": "Relevant retrieval over a deep store must emit context proportional to total memory depth.",
    "alternative_hypothesis": "Relevant retrieval returns a small, budget-bounded context while preserving the target anchor."
  }
}
```

## Judge Output

```json
{
  "mode": "rare-anchor-bounded-context",
  "context_memory_ids": [
    "mem-bounded-context-under-depth-000121"
  ],
  "context_tokens": 62,
  "context_chars": 244,
  "full_text_chars": 21823,
  "compression_ratio": 0.01118086,
  "timing": {
    "repeats": 1,
    "min_ms": 0.158,
    "median_ms": 0.158,
    "max_ms": 0.158,
    "rounded_min_ms_2dp": 0.16,
    "rounded_median_ms_2dp": 0.16,
    "raw_ns": [
      158000
    ]
  },
  "measured_latency_ms": 0.158
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
