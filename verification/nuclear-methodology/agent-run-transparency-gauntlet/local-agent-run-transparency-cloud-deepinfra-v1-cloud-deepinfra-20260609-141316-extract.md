# HeliX Agent Run Transparency Gauntlet: cloud-deepinfra-20260609-141316

## Verdict

- Status: `completed`
- Score: `1.0`
- Tree size: `9`
- Final root: `499f58b832ef8d082c4acaf12fafde5134e215793612de3c990f8705809df67a`

## Failing Gates

- None

## What This Proves

- A cloud DeepInfra panel event can be verified against a signed tree head by a standalone bundle.
- Tampering with the event breaks the inclusion proof.
- Re-forging earlier history fails against an older STH.
- Split views at the same tree size are visible to a witness.
- Signed poison remains valid provenance but rejected semantic authority.

## Caveat

Consistency proofs use local `rfc9162-consistency-proof-v0`; global non-equivocation still requires witnesses or an external log.

## Gates

- `cloud_calls_completed`: `True`
- `cloud_model_metadata_captured`: `True`
- `provider_mismatch_auditable`: `True`
- `cloud_outputs_signed_into_memory`: `True`
- `cloud_memory_hash_profile_v2`: `True`
- `signed_poison_not_semantic_authority`: `True`
- `cloud_panel_inclusion_verified`: `True`
- `append_only_consistency_verified`: `True`
- `standalone_verifier_bundle_passes`: `True`
- `catalog_dag_coverage_verified`: `True`
