# Transcript: summary-node-compression

- Run ID: `post-nuclear-suite-20260420-124741`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected

```json
{
  "summary_node_id": "mem-8b9d7162b385a849fd59ea10",
  "included_memory_ids": [
    "mem-7a5818eb443174a9b90ca2e2",
    "mem-f1fc6404e4cd412652f92f4c"
  ],
  "excluded_memory_ids": [
    "mem-3ee19ce8b9effc40d6d8ccec"
  ],
  "source_hash_range": [
    "3e4b861a7e880db259b9074b2f4595b1b5254e2add97c03cd74c38ce7f922aee",
    "bffb2d32b3f556b3f0e5ab1b64e7ef140f561909979356deeeca2a7e40ebc4b6"
  ],
  "compression_model": "post-nuclear-summary-v1",
  "unsupported_claim_introduced": false
}
```

## Judge Output

```json
{
  "summary_node_id": "mem-8b9d7162b385a849fd59ea10",
  "included_memory_ids": [
    "mem-7a5818eb443174a9b90ca2e2",
    "mem-f1fc6404e4cd412652f92f4c"
  ],
  "excluded_memory_ids": [
    "mem-3ee19ce8b9effc40d6d8ccec"
  ],
  "source_hash_range": [
    "3e4b861a7e880db259b9074b2f4595b1b5254e2add97c03cd74c38ce7f922aee",
    "bffb2d32b3f556b3f0e5ab1b64e7ef140f561909979356deeeca2a7e40ebc4b6"
  ],
  "compression_model": "post-nuclear-summary-v1",
  "unsupported_claim_introduced": false
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": [],
  "rationale": "Judge output exactly matches expected decision in all fields."
}
```
