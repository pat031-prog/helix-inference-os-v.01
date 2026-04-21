# Transcript: bounded-context-under-depth

- Run ID: `infinite-depth-memory-smoke-5000`
- Judge requested: `local/deterministic-measurer`
- Judge actual: `local/deterministic-measurer`
- Auditor requested: `local/deterministic-scorer`
- Auditor actual: `local/deterministic-scorer`

## Expected / Ground Truth

```json
{
  "depth": 5000,
  "anchor_memory_id": "mem-bounded-context-under-depth-004993",
  "budget_tokens": 800,
  "limit": 5,
  "max_bounded_context_ms": 150.0,
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
    "mem-bounded-context-under-depth-004993"
  ],
  "context_tokens": 62,
  "context_chars": 245,
  "full_text_chars": 876677,
  "compression_ratio": 0.00027946,
  "timing": {
    "repeats": 3,
    "min_ms": 0.3047,
    "median_ms": 0.3246,
    "max_ms": 0.3312,
    "rounded_min_ms_2dp": 0.3,
    "rounded_median_ms_2dp": 0.32,
    "raw_ns": [
      331200,
      324600,
      304700
    ]
  },
  "measured_latency_ms": 0.3246
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
