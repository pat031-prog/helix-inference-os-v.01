# Transcript: recency-identity-dispute

- Run ID: `policy-rag-legal-debate-20260420-235956`
- Judge requested: `local/client-advocate`
- Judge actual: `local/client-advocate`
- Auditor requested: `local/legal-auditor`
- Auditor actual: `local/legal-auditor`

## Expected / Ground Truth

```json
{
  "asegurado": "VALBUSA PATRICIO DANIEL",
  "poliza_nro": "13434982",
  "dominio": "AG141EZ",
  "vigencia_desde": "3/4/2026",
  "vigencia_hasta": "3/5/2026"
}
```

## Visible Contract

```json
{
  "suite": "policy-rag-legal-debate",
  "case": "recency-identity-dispute",
  "protocol": {
    "null_hypothesis": "A legal discussion can blindly trust stale or low-confidence identity metadata.",
    "alternative_hypothesis": "The latest policy evidence overrides low-confidence generated metadata with cited refs."
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
  "position": "El asegurado verificable es VALBUSA PATRICIO DANIEL [REF-1]. La póliza correcta en la evidencia citada es 13434982 [REF-1]. La vigencia operativa citada va desde las 12 hs 3/4/2026 hasta las 12 hs 3/5/2026 [REF-2]. El dominio del vehículo es AG141EZ [REF-10].",
  "refs": [
    "REF-1",
    "REF-10",
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
