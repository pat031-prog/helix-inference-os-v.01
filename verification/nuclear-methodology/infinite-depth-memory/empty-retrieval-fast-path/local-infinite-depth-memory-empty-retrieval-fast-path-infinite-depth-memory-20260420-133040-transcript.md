# Transcript: empty-retrieval-fast-path

- Run ID: `infinite-depth-memory-20260420-133040`
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
    "repeats": 7,
    "min_ms": 0.1856,
    "median_ms": 0.2388,
    "max_ms": 0.3029,
    "rounded_min_ms_2dp": 0.19,
    "rounded_median_ms_2dp": 0.24,
    "raw_ns": [
      247100,
      196400,
      185600,
      200100,
      241600,
      238800,
      302900
    ]
  },
  "measured_latency_ms": 0.2388,
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
