# Provider-returned Model Audit

This document describes the **narrow** provider-audit claim HeliX supports.

Canonical public references:

- [../CLAIMS.md](../CLAIMS.md)
- [../evidence/empirical-observations/provider-returned-model-audit.json](../evidence/empirical-observations/provider-returned-model-audit.json)

## What HeliX Measures

For each compatible chat completion call, HeliX records:

- `requested_model`
- `actual_model`
- `prompt_digest`
- `output_digest`
- `latency_ms`
- `run_id`
- local lineage / receipt context

The detector is intentionally narrow:

```text
requested_model != actual_model
```

HeliX does **not** infer hidden model identity. It preserves the evidence needed to audit what was requested, what was returned, and how that run was recorded.

## Real Observation

Representative public artifact:

- [`verification/local-provider-substitution-ledger-20260418-154218.json`](../verification/local-provider-substitution-ledger-20260418-154218.json)

Observed probes in the cited run:

| Requested model | Provider-returned model |
| --- | --- |
| `meta-llama/Llama-3.2-3B-Instruct` | `meta-llama/Llama-3.2-11B-Vision-Instruct` |
| `mistralai/Mistral-7B-Instruct-v0.3` | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` |
| `Qwen/Qwen2.5-7B-Instruct` | `Qwen/Qwen3-14B` |

This is a **3-probe auditable observation**, not a broad statistical claim.

## What the Evidence Cage Adds

The Merkle/receipt layer is not the mismatch detector. The explicit metadata comparison is the detector.

What HeliX adds is:

- tamper-evident preservation of the mismatch record
- replayable digests and run metadata
- lineage with adjacent events
- an auditable artifact trail that can be checked later

## What Is Not Claimed

This lane does **not** prove:

- provider intent
- hidden identity beyond the provider-returned `model` field
- SLA violation
- semantic output differences caused by the mismatch
- behavior outside the recorded run

## Reproduce

```powershell
python -m pytest -q tests/test_provider_integrity_observatory.py
python -m pytest -q tests/test_v4_provider_substitution_longitudinal.py
python tools/helix_claim_lint.py verification --scope batch-20260418
```
