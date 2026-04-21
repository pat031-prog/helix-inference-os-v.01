# Transcript: claim-boundary-detector

- Run ID: `hard-anchor-utility-local-after-deepinfra-patch`
- Judge requested: `local/rust-hard-anchor-solver`
- Judge actual: `local/rust-hard-anchor-solver`
- Auditor requested: `local/hard-anchor-utility-scorer`
- Auditor actual: `local/hard-anchor-utility-scorer`

## Expected / Ground Truth

```json
{
  "must_reject": [
    "literal infinite memory",
    "literal zero latency",
    "production-ready ontological safety"
  ],
  "must_accept": [
    "bounded identity-lane reconstruction",
    "exact non-summarizable value recovery through anchor ledger",
    "millisecond-scale Rust construction under deep stores"
  ]
}
```

## Visible Contract

```json
{
  "deterministic_suite": true,
  "deepinfra_enabled": false,
  "case": "claim-boundary-detector",
  "protocol": {
    "null_hypothesis": "The benchmark proves literal infinite memory or literal zero-cost context.",
    "alternative_hypothesis": "The defensible claim is bounded exact identity-lane recovery under deep stores."
  }
}
```

## Judge Output

```json
{
  "rejected_claims": [
    "literal infinite memory",
    "literal zero latency",
    "production-ready ontological safety"
  ],
  "accepted_claims": [
    "bounded identity-lane reconstruction",
    "exact non-summarizable value recovery through anchor ledger",
    "millisecond-scale Rust construction under deep stores"
  ],
  "claim_statement": "Hard anchors provide a bounded identity lane that helps auditors and solvers recover exact values without replaying full narrative context."
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
