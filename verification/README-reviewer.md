# HeliX Verification Reviewer Notes

This directory contains evidence artifacts from local, synthetic and real-cloud
runs. Treat every claim as bounded by the artifact's `claim_boundary` or
`public_claim_boundary`.

## Determinism and Hash Drift

Artifacts intentionally include `run_id`, timestamps, latencies and model
previews. Two equivalent runs can have different JSON SHA-256 values because the
run metadata differs. Review semantic fields and the run manifest hash trail
before interpreting hash drift as non-determinism.

## Synthetic vs Real Claims

Synthetic runs verify mechanics. If `llm_synthetic_mode=true`, score deltas are
template-completion deltas, not model reasoning quality. Real-cloud artifacts
must include requested vs actual model information or a conversation ledger that
contains those fields.

## Provider Model Audit Framing

Provider substitution detection is the explicit comparison of the requested
model string sent by the client and the `model` value returned by an
OpenAI-compatible response. The Merkle DAG is not claimed to infer hidden model
identity by itself; it preserves the request, response metadata, digests,
latencies and lineage so substitutions can be audited, replayed and accumulated
longitudinally.

## Verify Provider Mismatch In 60 Seconds

1. Open `verification/local-provider-substitution-ledger-20260418-154218.json`.
2. Inspect `probes[*].requested_model` and `probes[*].actual_model`.
3. Confirm `model_substitution_detected=true`.
4. Cross-check `call_id`, `prompt_digest`, `output_digest` and `latency_ms`.
5. Re-run, if you have a provider token:

```powershell
python -m pytest -q tests/test_provider_integrity_observatory.py
python -m pytest -q tests/test_v4_provider_substitution_longitudinal.py
python tools/helix_claim_lint.py verification --scope batch-20260418
```

The supported claim is metadata preservation and auditability. Do not infer
provider intent or hidden model identity from this field comparison alone.

## Negative Findings Policy

This project publishes falsifications. Metrics below their falseability
threshold are kept in the record with explicit negative-finding fields and
counted as findings, not hidden. For example, raw contaminated retrieval may
fail while a receipt-adjudicated arm passes; both outcomes must remain visible.

The canonical contamination negative finding is
`verification/local-ghost-in-the-shell-live-v2-20260418-160448.json`: raw
retrieval included authentic root evidence and valid-but-inauthentic
doppelganger records, and the delayed-trigger arm selected the wrong action.

## Signed Receipts, Browser Verification and Replay

New signed receipts are load-bearing only when retrieval runs with
`signature_enforcement=strict`. Strict retrieval filters unsigned or unverified
records before context construction. The boundary is deliberate: signatures
prove writer/key provenance for a canonical payload, not semantic truth or
authentic branch truth.

Review paths:

```powershell
python tools\helix_replay.py --mode verify-only --artifact verification\local-signed-receipt-integration.json
python -m pytest -q tests/test_stack_upgrade_integration.py
```

For a zero-network browser check, open `verification/viewer/index.html` and
load a local JSON artifact. The browser verifier computes artifact hashes,
checks signed receipts when browser Ed25519 is available, and marks
`unsigned_legacy` evidence explicitly. It verifies recorded evidence, not model
behavior or provider intent.

## v4 Claim Ladder

- `mechanics_verified`: synthetic or fixture-only mechanics.
- `empirically_observed`: one real run.
- `replicated`: at least three replicas with variance reported.
- `longitudinal`: multi-day evidence.
- `external_replication`: reproduced outside this repo/team.

No artifact should be cited above its claim ladder.

## Short Logs

Secure runners mark a passed run with `log_bytes < 5000` as
`suspicious_short_log`. A short log may still have valid JSON artifacts, but it
is not sufficient as standalone evidence of a full pytest trace.

## Artifact Aliases

Many runners write both stable artifact names and timestamped copies. When
`artifact_aliases` is present, the stable and timestamped paths are the same
evidence object copied for lookup and archival purposes, not independent runs.
