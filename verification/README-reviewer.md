# HeliX Raw Verification Archive

`verification/` is the **raw lab/archive tree** for HeliX.

It contains:

- historical artifacts
- preregistrations
- suite outputs
- run manifests
- viewer assets
- scratch or pre-hardening snapshots that remain useful for auditability

## Public Review Starts Elsewhere

Do not start public review from this directory anymore.

Start here instead:

- [`../CLAIMS.md`](../CLAIMS.md)
- [`../THREAT_MODEL.md`](../THREAT_MODEL.md)
- [`../NULL_RESULTS.md`](../NULL_RESULTS.md)
- [`../REPRODUCING.md`](../REPRODUCING.md)
- [`../evidence/index.json`](../evidence/index.json)

## How To Read `verification/`

- Treat every raw artifact as bounded by its `claim_boundary` or `public_claim_boundary`.
- Expect timestamps, latencies and manifests to change hashes even when the underlying semantic outcome is similar.
- Treat this tree as the backing store for replay and inspection, not as the public canonical navigation layer.

## Provider Model Audit Framing

Provider substitution detection is the explicit comparison of `requested_model` and provider-returned `actual_model`.

The Merkle/receipt layer does not infer hidden identity by itself; it preserves the request, response metadata, digests, latency and lineage so the mismatch can be audited later.

## Negative Findings Policy

This project keeps falsifications and blocked lanes visible.

- failed or blocked runs remain part of the archive
- contaminated raw retrieval is preserved when adjudicated retrieval matters
- historical artifacts do not override current failed local suites

## Replay Paths

```powershell
python tools\helix_replay.py --mode verify-only --artifact verification\local-signed-receipt-integration.json
python tools\helix_replay.py --mode verify-only --artifact verification\local-v4-lineage-forgery-gauntlet.json
```

For a zero-network browser check, open `verification/viewer/index.html` and load a local JSON artifact.
