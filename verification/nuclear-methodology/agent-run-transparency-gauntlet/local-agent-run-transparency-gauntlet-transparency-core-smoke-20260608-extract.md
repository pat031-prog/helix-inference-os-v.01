# HeliX Agent Run Transparency Gauntlet: transparency-core-smoke-20260608

## Verdict

- Status: `completed`
- Score: `1.0`
- Tree size: `7`
- Final root: `981d22971731292b703eae74a23121c4dad11c990e1d0ea5d2f66f569bb89d47`

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

- `baseline_inclusion_verified`: `True`
- `tampered_event_rejected`: `True`
- `append_only_consistency_verified`: `True`
- `reforged_history_rejected_against_prior_sth`: `True`
- `split_view_detected_by_witness`: `True`
- `signed_poison_signature_not_semantic_truth`: `True`
- `backdating_demoted_to_claim_mismatch`: `True`
- `provider_mismatch_included_and_auditable`: `True`
- `standalone_verifier_bundle_passes`: `True`
