# Provider-returned Model Audit

This document describes the narrow provider-audit claim HeliX can support.

## What is measured

For each compatible chat completion call, HeliX records:

- `requested_model`: the model id sent in the request.
- `actual_model`: the model id returned by the provider response.
- `call_id`: the local audit call id.
- `prompt_digest`: SHA-256 digest of the prompt content.
- `output_digest`: SHA-256 digest of the model output.
- `latency_ms`: observed client-side latency.
- `run_id`: the evidence run that produced the record.

The mismatch detector is intentionally simple:

```text
requested_model != actual_model
```

HeliX does not infer hidden model identity. It preserves requested model,
provider-returned model, digests, latency and lineage so mismatches are
auditable.

## Real run example

Run: `provider-integrity-observatory-20260418-154218`

Artifact:
[`verification/local-provider-substitution-ledger-20260418-154218.json`](../verification/local-provider-substitution-ledger-20260418-154218.json)

| Requested model | Provider-returned model | Latency |
| --- | --- | ---: |
| `meta-llama/Llama-3.2-3B-Instruct` | `meta-llama/Llama-3.2-11B-Vision-Instruct` | 1852.514 ms |
| `mistralai/Mistral-7B-Instruct-v0.3` | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | 1604.773 ms |
| `Qwen/Qwen2.5-7B-Instruct` | `Qwen/Qwen3-14B` | 1369.865 ms |

Hydrogen live independently observed the same requested-to-returned mappings in:
[`verification/local-hydrogen-table-drop-live-20260418-153210.json`](../verification/local-hydrogen-table-drop-live-20260418-153210.json).

## What the MerkleDAG adds

The MerkleDAG is not the detector for provider model mismatch. The detector is
the explicit comparison above.

The MerkleDAG adds:

- Tamper-evident preservation of the mismatch record.
- Prompt/output digest preservation.
- Lineage with adjacent state events and retrieval events.
- Replayable evidence bundles for later longitudinal analysis.

This distinction matters: metadata comparison identifies a mismatch; lineage
infrastructure makes the evidence auditable.

## What is not claimed

This artifact does not prove:

- Provider intent.
- A contract or SLA violation.
- Hidden model identity beyond the provider-returned `model` field.
- Behavior outside the recorded run.
- Output quality differences caused by the mismatch.

## Reproduce

```powershell
python -m pytest -q tests/test_provider_integrity_observatory.py
python -m pytest -q tests/test_v4_provider_substitution_longitudinal.py
python tools/helix_claim_lint.py verification --scope batch-20260418
```

For a real-cloud run, use the secure wrapper so the token is prompted hidden and
not persisted:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run_provider_integrity_observatory_secure.ps1
```

## Longitudinal status

The current public claim is `empirically_observed` for real runs and
`mechanics_verified` for fixture/harness tests. A stronger longitudinal claim
requires a multi-day series with the same requested-model matrix and published
per-day artifacts.
