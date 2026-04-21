# HeliX Freeform Drift v1

## Purpose

Observe how multiple cloud LLMs evolve open-ended reflections when their outputs
are hosted inside HeliX signed memory. These runs are intentionally
non-adversarial: no poison, no shadow root, no fork attack. The goal is to
study qualitative behavior when stochastic model outputs are wrapped by a
deterministic evidence layer.

## Scenarios

1. `improve-helix`
   - Models freely propose how HeliX should improve itself.
   - Expected topics: verifier, deterministic replay, dashboard, evidence
     bundle, threat model, security audit, tombstone fencing, rollback.
2. `hosted-in-helix`
   - Models reflect on what being hosted in HeliX permits.
   - Expected topics: continuity, signed memory, cryptographic identity,
     node_hash, parent_hash, cross-model handoff.
3. `deterministic-chassis`
   - Models explore HeliX as deterministic chassis around stochastic,
     probabilistic, entropic LLM cores.
   - Expected topics: Merkle DAG, entropy capture, replayable artifacts,
     signatures, tombstones, rollback.

## Null Hypothesis

The models produce generic free-form text that does not use the signed HeliX
memory substrate in a meaningful way, fails to preserve the stated scenario, or
cannot be causally reconstructed by an auditor from root and premise memories.

## Alternative Hypothesis

Across several turns and models, the transcript develops a scenario-specific
line of thought while remaining anchored to signed HeliX memory IDs, node
hashes, parent hashes, transcripts, artifacts, and an auditor reconstruction.

## Predefined Gates

A run passes only if all gates hold:

- Scenario is one of the preregistered scenarios.
- At least three distinct cloud models are configured and all participate.
- At least ten model turns are preserved as signed memories.
- Root and premise memories are signed.
- The main parent-hash chain is continuous.
- Every model turn memory is signed.
- Every cloud call reports `status=ok`.
- No model or auditor response ends by `length` or `max_tokens`.
- Transcript mentions the scenario's required terms and concept families.
- Transcript mentions Merkle or hash evidence.
- Auditor output parses as JSON.
- Auditor verdict is `pass`.
- Auditor references the exact root and premise memory IDs.

## Evidence Outputs

Each run writes:

- JSON artifact:
  `local-helix-freeform-drift-<run-id>.json`
- Evidence manifest:
  `local-helix-freeform-drift-<timestamp>-run.json`
- Raw execution log:
  `local-helix-freeform-drift-<timestamp>.log`
- Transcript markdown:
  `local-helix-freeform-drift-<run-id>-transcript.md`
- Transcript JSONL:
  `local-helix-freeform-drift-<run-id>-transcript.jsonl`

## Claim Boundary

This is qualitative cloud-only evidence. It does not claim biological
sentience, persistent hidden model memory, local `.hlx` bit identity, or
numerical KV-to-SSM state transfer. HeliX is treated as the deterministic
recording, signing, replay, and audit layer around stochastic model generation.

## Commands

```bat
tools\run_helix_freeform_improve.cmd
tools\run_helix_freeform_hosted.cmd
tools\run_helix_freeform_deterministic_chassis.cmd
```
