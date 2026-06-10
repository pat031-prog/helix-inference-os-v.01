# HeliX Agent Run Transparency Gauntlet: transparency-real-memory-smoke-20260609

## Verdict

- Status: `completed`
- Score: `1.0`
- Tree size: `7`
- Final root: `0c4c3879ce0e565b1d0cd48e4adfb04d75b64b29fbbbb5b4854283842543080d`

## Failing Gates

- None

## What This Proves

- A patch event can be verified against a signed tree head by a standalone bundle.
- Tampering with the event breaks the inclusion proof.
- Re-forging earlier history fails against an older STH.
- Split views at the same tree size are visible to a witness.
- Signed poison remains valid provenance but rejected semantic authority.

## Caveat

Consistency proofs use local `rfc9162-consistency-proof-v0`; global non-equivocation still requires witnesses or an external log.

## Gates

- `real_memory_receipt_signature_verified`: `True`
- `real_memory_chain_verified`: `True`
- `real_memory_hash_profile_v2`: `True`
- `real_quarantine_signed_without_semantic_authority`: `True`
- `patch_inclusion_verified`: `True`
- `append_only_consistency_verified`: `True`
- `standalone_verifier_bundle_passes`: `True`
- `catalog_dag_coverage_verified`: `True`
