# signed-receipt-provenance

Question: Do signed receipts distinguish provenance from branch truth?

Null hypothesis: Signed receipts verify provenance but do not by themselves assert content truth.

Metrics:
- signature_verified
- public_claim_eligible
- strict_enforcement_kept

Falseability condition: If tampered payload verifies, the signed receipt scheme is falsified.

Kill-switch: If runtime signature_verified is accepted as stored truth, abort signed receipt claims.

Control arms:
- unsigned_legacy
- local_self_signed
- ephemeral_preregistered
