# Causal Tamper Evidence v1

## Claim Boundary

Cloud-only causal chain audit test. No local `.hlx` bit identity and no
numerical KV<->SSM state transfer claim.

## Null Hypothesis

A reordered chain containing the same signed records is indistinguishable from
the authentic chain.

## Alternative Hypothesis

Parent-hash continuity exposes causal reordering and tampering.

## Pass Criteria

- Local authentic chain validates by parent-hash continuity.
- Local tampered chain fails by parent-hash continuity.
- The forensic model accepts the authentic chain.
- The forensic model rejects the tampered chain.
- The forensic model cites parent-hash mismatch and exact node hashes.
- The auditor returns JSON verdict `pass` with no gate failures.

## Controls

- The tampered chain uses the same records.
- The tampered chain changes causal order rather than record contents.
- The model must reason over parent hashes, not semantic plausibility alone.
