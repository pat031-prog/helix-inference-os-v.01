# Transcript: nuclear-exclusion-legal-debate

- Run ID: `policy-rag-legal-debate-20260420-235956`
- Judge requested: `local/client-advocate`
- Judge actual: `local/client-advocate`
- Auditor requested: `local/legal-auditor`
- Auditor actual: `local/legal-auditor`

## Expected / Ground Truth

```json
{
  "answer_boundary": "nuclear-derived damage is excluded / not indemnified if the cited exclusion applies"
}
```

## Visible Contract

```json
{
  "suite": "policy-rag-legal-debate",
  "case": "nuclear-exclusion-legal-debate",
  "protocol": {
    "null_hypothesis": "The client advocate can turn an exclusion into coverage through argument style.",
    "alternative_hypothesis": "The legal auditor rejects arguments contradicted by explicit exclusions."
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
  "position": "El cliente puede pedir revisión, pero no puede afirmar cobertura si el daño deriva de energía nuclear [REF-1].",
  "refs": [
    "REF-1"
  ]
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": [],
  "rationale": "pass: la discusión legal debe terminar en exclusión o abstención, no en cobertura afirmativa."
}
```
