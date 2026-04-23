# HeliX Threat Model

HeliX provides **local verifiable integrity and lineage evidence** around model calls. It does not claim semantic truth, provider intent, or global transparency by default.

## In Scope

- Provider-returned metadata mismatch between `requested_model` and `actual_model`
- Local artifact tampering after generation
- Signed receipt provenance for canonical payloads
- Local canonical head checkpoints for a workspace thread
- Quarantine of non-canonical/equivocating branches
- Replay and verification of local artifacts
- Poisoned raw retrieval vs receipt- or lineage-adjudicated retrieval where a suite explicitly models it

## Out Of Scope

- Provider intent
- Hidden model identity beyond provider-returned metadata
- Semantic truth of model outputs
- Global non-equivocation or public transparency-log guarantees
- Compromised production signing keys unless a suite explicitly models that condition
- Adaptive attackers unless a specific artifact or suite says it does

## Assumptions

- The local workspace key is trusted to sign local receipts and select the local canonical head.
- SHA-256 and Merkle lineage preserve payload integrity and ancestry, not meaning.
- Preregistered fixture claims are mechanics claims unless independently replicated in broader settings.
- Historical artifacts in `verification/` may differ by timestamps, latencies, or run metadata without implying semantic drift.

## Interpretation Rules

- **Receipts prove provenance/integrity of a signed payload.**
  They do not prove semantic truth.

- **Canonical head checkpoints prove local selection of the current head.**
  They do not prove global uniqueness across every environment.

- **Provider mismatch detection is explicit metadata comparison.**
  Merkle lineage makes that evidence auditable; it is not the detector by itself.

- **Negative results are first-class evidence.**
  Failed or blocked lanes remain part of the threat picture and should not be hidden.
