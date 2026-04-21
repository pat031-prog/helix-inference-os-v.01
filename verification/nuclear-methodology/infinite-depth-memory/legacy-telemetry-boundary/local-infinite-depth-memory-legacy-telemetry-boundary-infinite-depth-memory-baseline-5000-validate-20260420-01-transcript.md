# Transcript: legacy-telemetry-boundary

- Run ID: `infinite-depth-memory-baseline-5000-validate-20260420-01`
- Judge requested: `local/deterministic-measurer`
- Judge actual: `local/deterministic-measurer`
- Auditor requested: `local/deterministic-scorer`
- Auditor actual: `local/deterministic-scorer`

## Expected / Ground Truth

```json
{
  "minimum_memory_nodes": 5000,
  "literal_zero_latency_claim_allowed": false,
  "rounded_zero_display_allowed": true
}
```

## Visible Contract

```json
{
  "deterministic_suite": true,
  "case": "legacy-telemetry-boundary",
  "protocol": {
    "null_hypothesis": "The historical 0.0 ms artifact is either missing or is being overread as literal zero latency.",
    "alternative_hypothesis": "The artifact is present and is classified as rounded display telemetry, not literal zero-cost proof."
  }
}
```

## Judge Output

```json
{
  "classification": "rounded_display_telemetry_not_literal_zero_cost",
  "legacy_build_context_ms": 0.0,
  "legacy_memory_nodes": 5000,
  "defensible_interpretation": "A 0.0 ms value rounded to two decimals is not proof of physical zero latency."
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
