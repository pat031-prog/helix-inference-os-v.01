# Transcript: bounded-context-under-depth

- Run ID: `infinite-depth-memory-20260420-133040`
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
    "repeats": 7,
    "min_ms": 0.2639,
    "median_ms": 0.3025,
    "max_ms": 0.6094,
    "rounded_min_ms_2dp": 0.26,
    "rounded_median_ms_2dp": 0.3,
    "raw_ns": [
      302500,
      264800,
      263900,
      266500,
      532800,
      609400,
      395300
    ]
  },
  "measured_latency_ms": 0.3025
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
