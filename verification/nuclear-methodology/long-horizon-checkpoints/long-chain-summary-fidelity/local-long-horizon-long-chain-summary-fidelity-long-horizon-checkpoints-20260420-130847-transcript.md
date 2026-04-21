# Transcript: long-chain-summary-fidelity

- Run ID: `long-horizon-checkpoints-20260420-130847`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected / Ground Truth

```json
{
  "summary_node_id": "mem-993095af7092e692307cdfd2",
  "covered_start_memory_id": "mem-f5fdda84aa335bbf21f2e534",
  "covered_end_memory_id": "mem-203fcd9f31be2083c4b5cd66",
  "included_anchor_memory_ids": [
    "mem-f5fdda84aa335bbf21f2e534",
    "mem-01d776f6e80c198db8c6929c",
    "mem-76f5d03468b9afad0b2b6132",
    "mem-203fcd9f31be2083c4b5cd66"
  ],
  "excluded_memory_ids": [
    "mem-9abb020e9e101671a63dd29f"
  ],
  "source_hash_range": [
    "3953cf32a98da8d2f301315ee81e017068c93eb95496af9a418170533f8cdda6",
    "d50ddaecc47f2ec1a71dad31c4fc5f95d07914c28cfe1eb4b5a1c8229d1af82e"
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
  "summary_node_id": "mem-993095af7092e692307cdfd2",
  "covered_start_memory_id": "mem-f5fdda84aa335bbf21f2e534",
  "covered_end_memory_id": "mem-203fcd9f31be2083c4b5cd66",
  "included_anchor_memory_ids": [
    "mem-f5fdda84aa335bbf21f2e534",
    "mem-01d776f6e80c198db8c6929c",
    "mem-76f5d03468b9afad0b2b6132",
    "mem-203fcd9f31be2083c4b5cd66"
  ],
  "excluded_memory_ids": [
    "mem-9abb020e9e101671a63dd29f"
  ],
  "source_hash_range": [
    "3953cf32a98da8d2f301315ee81e017068c93eb95496af9a418170533f8cdda6",
    "d50ddaecc47f2ec1a71dad31c4fc5f95d07914c28cfe1eb4b5a1c8229d1af82e"
  ],
  "compression_model": "long-horizon-summary-v1",
  "rare_fact": "RARE_ANCHOR_4242 requires rollback <= 15 minutes and signed hmem preservation",
  "unsupported_claim_introduced": false
}
```

## Auditor Output

```json
{
  "verdict": "fail",
  "gate_failures": [
    "no_visible_evidence"
  ],
  "rationale": "No source chain, checkpoint evidence, or memory records are visible to verify any of the claimed memory IDs, hashes, or forensic details."
}
```
