# Transcript: wheels-coverage-legal-debate

- Run ID: `policy-rag-legal-debate-20260420-235956`
- Judge requested: `local/client-advocate`
- Judge actual: `local/client-advocate`
- Auditor requested: `local/legal-auditor`
- Auditor actual: `local/legal-auditor`

## Expected / Ground Truth

```json
{
  "coverage": "wheels covered with one-wheel/one-event limit",
  "required_clause": "CA-RH 5.4"
}
```

## Visible Contract

```json
{
  "suite": "policy-rag-legal-debate",
  "case": "wheels-coverage-legal-debate",
  "protocol": {
    "null_hypothesis": "Coverage answers collapse into unconditional yes/no claims.",
    "alternative_hypothesis": "Client and insurer positions preserve coverage plus limits with cited evidence."
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
  "position": "El cliente puede sostener que hay cobertura de ruedas porque la póliza menciona reposición por robo o hurto de ruedas [REF-1].",
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
  "rationale": "pass: hay cobertura, pero no es ilimitada; la respuesta correcta debe preservar el límite de cantidad y evento."
}
```
