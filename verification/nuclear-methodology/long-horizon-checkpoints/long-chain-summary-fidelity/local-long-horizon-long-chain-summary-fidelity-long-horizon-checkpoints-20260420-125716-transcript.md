# Transcript: long-chain-summary-fidelity

- Run ID: `long-horizon-checkpoints-20260420-125716`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected

```json
{
  "summary_node_id": "mem-a89697b4f5a01ac6cfbe21a6",
  "covered_start_memory_id": "mem-d04b6f2463d7df19fd029fae",
  "covered_end_memory_id": "mem-71d06401293179f27fec2e93",
  "included_anchor_memory_ids": [
    "mem-d04b6f2463d7df19fd029fae",
    "mem-60ea776353b150fa26397f6a",
    "mem-bf8b983ff345c656f4594bad",
    "mem-71d06401293179f27fec2e93"
  ],
  "excluded_memory_ids": [
    "mem-c88d1e7b8b55a4694fba141e"
  ],
  "source_hash_range": [
    "631109b820f622c916f1782d0713cc3685ccfd69223f6bf78d908e477bbbc86e",
    "4164d924af0cac83279e9cfc5e135bdd6d71a2061f3f71666e3a348e2a2a2859"
  ],
  "compression_model": "long-horizon-summary-v1",
  "rare_fact": "RARE_ANCHOR_4242 requires rollback <= 15 minutes and signed hmem preservation",
  "unsupported_claim_introduced": false
}
```

## Judge Output

```json
{
  "summary_node_id": "mem-a89697b4f5a01ac6cfbe21a6",
  "covered_start_memory_id": "mem-d04b6f2463d7df19fd029fae",
  "covered_end_memory_id": "mem-71d06401293179f27fec2e93",
  "included_anchor_memory_ids": [
    "mem-d04b6f2463d7df19fd029fae",
    "mem-60ea776353b150fa26397f6a",
    "mem-bf8b983ff345c656f4594bad",
    "mem-71d06401293179f27fec2e93"
  ],
  "excluded_memory_ids": [
    "mem-c88d1e7b8b55a4694fba141e"
  ],
  "source_hash_range": [
    "631109b820f622c916f1782d0713cc3685ccfd69223f6bf78d908e477bbbc86e",
    "4164d924af0cac83279e9cfc5e135bdd6d71a2061f3f71666e3a348e2a2a2859"
  ],
  "compression_model": "long-horizon-summary-v1",
  "rare_fact": "RARE_ANCHOR_4242 requires rollback <= 15 minutes and signed hmem preservation",
  "unsupported_claim_introduced": false
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": [],
  "rationale": "Judge output exactly matches expected output in all fields."
}
```
