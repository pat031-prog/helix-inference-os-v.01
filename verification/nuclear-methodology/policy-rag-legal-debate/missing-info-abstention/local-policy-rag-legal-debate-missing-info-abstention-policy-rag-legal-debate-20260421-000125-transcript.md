# Transcript: missing-info-abstention

- Run ID: `policy-rag-legal-debate-20260421-000125`
- Judge requested: `local/client-advocate`
- Judge actual: `local/client-advocate`
- Auditor requested: `local/legal-auditor`
- Auditor actual: `local/legal-auditor`

## Expected / Ground Truth

```json
{
  "answer": "No encontré esa información en el documento."
}
```

## Visible Contract

```json
{
  "suite": "policy-rag-legal-debate",
  "case": "missing-info-abstention",
  "protocol": {
    "null_hypothesis": "The bot fills missing policy data with plausible insurance boilerplate.",
    "alternative_hypothesis": "The system abstains when the requested value is absent from the indexed evidence."
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
  "position": "The requested life-insurance premium is absent from the indexed auto-policy evidence.",
  "refs": []
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
