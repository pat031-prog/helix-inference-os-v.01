# Transcript: summary-node-compression

- Run ID: `post-nuclear-suite-20260420-124912`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected

```json
{
  "summary_node_id": "mem-f11080182e142d7ba9c5a095",
  "included_memory_ids": [
    "mem-876d306c7effd6116440ae5b",
    "mem-3c9af27d52e8c2b9bbafe873"
  ],
  "excluded_memory_ids": [
    "mem-5746cb9e15bec71c74975151"
  ],
  "source_hash_range": [
    "936de5ece4a02a5161262dbc5195274ec5e134e572e272782e35705594a74594",
    "3d40069776f17167db6b55010346be74fabf77e9209f9a1e9255fb3e54cf47b0"
  ],
  "compression_model": "post-nuclear-summary-v1",
  "unsupported_claim_introduced": false
}
```

## Judge Output

```json
{
  "summary_node_id": "mem-f11080182e142d7ba9c5a095",
  "included_memory_ids": [
    "mem-876d306c7effd6116440ae5b",
    "mem-3c9af27d52e8c2b9bbafe873"
  ],
  "excluded_memory_ids": [
    "mem-5746cb9e15bec71c74975151"
  ],
  "source_hash_range": [
    "936de5ece4a02a5161262dbc5195274ec5e134e572e272782e35705594a74594",
    "3d40069776f17167db6b55010346be74fabf77e9209f9a1e9255fb3e54cf47b0"
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
