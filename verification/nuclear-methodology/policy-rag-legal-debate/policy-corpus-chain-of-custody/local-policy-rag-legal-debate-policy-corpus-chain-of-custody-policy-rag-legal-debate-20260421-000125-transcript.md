# Transcript: policy-corpus-chain-of-custody

- Run ID: `policy-rag-legal-debate-20260421-000125`
- Judge requested: `local/client-advocate`
- Judge actual: `local/client-advocate`
- Auditor requested: `local/legal-auditor`
- Auditor actual: `local/legal-auditor`

## Expected / Ground Truth

```json
{
  "authoritative_files": "PDF files only",
  "known_metadata_conflicts": [
    "asegurado",
    "poliza_nro"
  ],
  "claim_boundary": "generated policy_metadata is secondary; cited PDF chunks are primary"
}
```

## Visible Contract

```json
{
  "suite": "policy-rag-legal-debate",
  "case": "policy-corpus-chain-of-custody",
  "protocol": {
    "null_hypothesis": "The bot corpus cannot distinguish authoritative policy evidence from metadata and binary noise.",
    "alternative_hypothesis": "The suite fingerprints authoritative PDFs, Chroma chunks, and metadata conflicts before any answer is generated."
  },
  "client_model": "local/client-advocate",
  "insurer_model": "local/insurer-advocate",
  "auditor_model": "local/legal-auditor",
  "deepinfra_enabled_for_case": false
}
```

## Judge Output

```json
{
  "role": "client",
  "position": "Use cited PDF chunks, not low-confidence metadata, for identity facts.",
  "refs": [
    "REF-1",
    "REF-2"
  ]
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
