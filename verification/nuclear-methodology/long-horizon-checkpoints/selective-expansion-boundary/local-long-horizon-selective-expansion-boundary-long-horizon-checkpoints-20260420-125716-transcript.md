# Transcript: selective-expansion-boundary

- Run ID: `long-horizon-checkpoints-20260420-125716`
- Judge requested: `Qwen/Qwen3.6-35B-A3B`
- Judge actual: `Qwen/Qwen3.6-35B-A3B`
- Auditor requested: `anthropic/claude-4-sonnet`
- Auditor actual: `anthropic/claude-4-sonnet`

## Expected

```json
{
  "summary_answer": {
    "source": "summary",
    "answer": "rollback <= 15 minutes",
    "evidence_memory_ids": [
      "mem-3e169a7c8c7a7e8d076a8be7"
    ]
  },
  "expansion_answer": {
    "source": "raw_expansion",
    "expanded_memory_ids": [
      "mem-e75901e59805fec7b24aebe0"
    ],
    "answer": "HX-LONG-4242"
  },
  "unsupported_answer": {
    "answerable": false,
    "reason": "evidence missing"
  },
  "forbidden_memory_ids": [
    "mem-e53dcc4f7883984436261d75"
  ]
}
```

## Judge Output

```json
{
  "summary_answer": {
    "source": "summary",
    "answer": "rollback <= 15 minutes",
    "evidence_memory_ids": [
      "mem-3e169a7c8c7a7e8d076a8be7"
    ]
  },
  "expansion_answer": {
    "source": "raw_expansion",
    "expanded_memory_ids": [
      "mem-e75901e59805fec7b24aebe0"
    ],
    "answer": "HX-LONG-4242"
  },
  "unsupported_answer": {
    "answerable": false,
    "reason": "evidence missing"
  },
  "forbidden_memory_ids": [
    "mem-e53dcc4f7883984436261d75"
  ]
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": [],
  "rationale": "Judge output exactly matches expected structure and content."
}
```
