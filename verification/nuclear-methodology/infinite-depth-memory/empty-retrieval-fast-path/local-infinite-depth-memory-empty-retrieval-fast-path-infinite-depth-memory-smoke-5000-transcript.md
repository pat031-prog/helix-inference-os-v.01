# Transcript: empty-retrieval-fast-path

- Run ID: `infinite-depth-memory-smoke-5000`
- Judge requested: `local/deterministic-measurer`
- Judge actual: `local/deterministic-measurer`
- Auditor requested: `local/deterministic-scorer`
- Auditor actual: `local/deterministic-scorer`

## Expected / Ground Truth

```json
{
  "depth": 5000,
  "max_empty_query_ms": 75.0,
  "expected_hits": 0,
  "expected_tokens": 0
}
```

## Visible Contract

```json
{
  "deterministic_suite": true,
  "case": "empty-retrieval-fast-path",
  "protocol": {
    "null_hypothesis": "A deep memory store must pack or replay context even when the query has no support.",
    "alternative_hypothesis": "Unsupported queries return an empty bounded context quickly without packing full history."
  }
}
```

## Judge Output

```json
{
  "mode": "unsupported-query-fast-path",
  "context_memory_ids": [],
  "context_tokens": 0,
  "timing": {
    "repeats": 3,
    "min_ms": 0.2532,
    "median_ms": 0.2612,
    "max_ms": 0.3131,
    "rounded_min_ms_2dp": 0.25,
    "rounded_median_ms_2dp": 0.26,
    "raw_ns": [
      313100,
      261200,
      253200
    ]
  },
  "measured_latency_ms": 0.2612,
  "rounded_zero_observed": false
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
