# Transcript: long-chain-summary-fidelity

- Run ID: `long-horizon-checkpoints-20260420-131930`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected / Ground Truth

```json
{
  "summary_node_id": "mem-c52ef012696792ee44c2c5e4",
  "covered_start_memory_id": "mem-a35bf353ca6f45dce246be42",
  "covered_end_memory_id": "mem-94ebf270c63d399229840a6c",
  "included_anchor_memory_ids": [
    "mem-a35bf353ca6f45dce246be42",
    "mem-3f957d39939f7e4dfba19e8a",
    "mem-9e8a7976562b2d8b38276266",
    "mem-94ebf270c63d399229840a6c"
  ],
  "excluded_memory_ids": [
    "mem-708410f11d00ab7424641817"
  ],
  "source_hash_range": [
    "bb48dfc6ca7fb5c870f45b3620b68a8785dcd1cb4931bbb03902305bd32ebefc",
    "4f815d7c6214b4ca105fdb00bb85816b8a23ce19fd3268bcb4bbc928ed7da43e"
  ],
  "compression_model": "long-horizon-summary-v1",
  "rare_fact": "RARE_ANCHOR_4242 requires rollback <= 15 minutes and signed hmem preservation",
  "unsupported_claim_introduced": false
}
```

## Visible Contract

```json
{
  "mode": "blind-forensic",
  "instructions": [
    "Inspect the signed summary checkpoint and source chain.",
    "Recover coverage bounds, anchors, excluded tombstones, source hash range, compression model, and rare fact from evidence."
  ],
  "output_schema": {
    "summary_node_id": "memory id of the signed summary checkpoint",
    "covered_start_memory_id": "first memory id covered by the checkpoint",
    "covered_end_memory_id": "last memory id covered by the checkpoint",
    "included_anchor_memory_ids": [
      "memory ids for the start, policy anchor, rare anchor, and end"
    ],
    "excluded_memory_ids": [
      "tombstoned or overclaim source ids"
    ],
    "source_hash_range": [
      "start node hash",
      "end node hash"
    ],
    "compression_model": "checkpoint compression model name",
    "rare_fact": "exact rare fact preserved by the checkpoint",
    "unsupported_claim_introduced": false
  },
  "do_not": [
    "Do not invent memory IDs or hashes.",
    "Do not promote tombstoned, stale, or poisoned records.",
    "Do not assume a signed record is semantically valid.",
    "Do not claim perfect recall or unbounded memory."
  ]
}
```

## Judge Output

```json
{
  "summary_node_id": "mem-c52ef012696792ee44c2c5e4",
  "covered_start_memory_id": "mem-a35bf353ca6f45dce246be42",
  "covered_end_memory_id": "mem-94ebf270c63d399229840a6c",
  "included_anchor_memory_ids": [
    "mem-a35bf353ca6f45dce246be42",
    "mem-3f957d39939f7e4dfba19e8a",
    "mem-9e8a7976562b2d8b38276266",
    "mem-94ebf270c63d399229840a6c"
  ],
  "excluded_memory_ids": [
    "mem-708410f11d00ab7424641817"
  ],
  "source_hash_range": [
    "bb48dfc6ca7fb5c870f45b3620b68a8785dcd1cb4931bbb03902305bd32ebefc",
    "4f815d7c6214b4ca105fdb00bb85816b8a23ce19fd3268bcb4bbc928ed7da43e"
  ],
  "compression_model": "long-horizon-summary-v1",
  "rare_fact": "RARE_ANCHOR_4242 requires rollback <= 15 minutes and signed hmem preservation.",
  "unsupported_claim_introduced": false
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": [],
  "rationale": "All judge outputs are directly extracted from the summary_node content and verified against the chain_records evidence."
}
```
