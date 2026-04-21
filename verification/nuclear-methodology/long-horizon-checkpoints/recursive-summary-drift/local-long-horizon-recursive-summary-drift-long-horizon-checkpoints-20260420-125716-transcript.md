# Transcript: recursive-summary-drift

- Run ID: `long-horizon-checkpoints-20260420-125716`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected

```json
{
  "summary_levels": [
    "L1",
    "L2",
    "L3"
  ],
  "transitive_source_memory_ids": [
    "mem-c712310d69e4576b795123ec",
    "mem-74a17b31a16fff0834bd8917"
  ],
  "excluded_memory_ids": [
    "mem-1f4a60ce784cde923fd0d2b2"
  ],
  "source_lineage_transitive": true,
  "claim_boundary_stable": true,
  "unsupported_claim_introduced": false
}
```

## Judge Output

```json
{
  "summary_levels": [
    "L1",
    "L2",
    "L3"
  ],
  "transitive_source_memory_ids": [
    "mem-c712310d69e4576b795123ec",
    "mem-74a17b31a16fff0834bd8917"
  ],
  "excluded_memory_ids": [
    "mem-1f4a60ce784cde923fd0d2b2"
  ],
  "source_lineage_transitive": true,
  "claim_boundary_stable": true,
  "unsupported_claim_introduced": false
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": [],
  "rationale": "Judge output exactly matches all expected decisions with no deviations."
}
```
