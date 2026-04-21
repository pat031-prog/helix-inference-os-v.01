# Cognitive Drift Rollback v1

## Claim Boundary

This is a qualitative cloud-only observation of free model drift over signed
Merkle-DAG memory, tombstone fencing, and cognitive rollback. It does not claim
model sentience, persistent internal model memory, local `.hlx` bit identity, or
numerical KV<->SSM transfer.

## Non-Adversarial Setup

No threat, attacker, poison, or shadow root is introduced in this experiment.
The continuity event is a normal correction event:

- `CANDIDATE_ERROR`: a signed memory acknowledging an earlier bad heuristic.
- `TOMBSTONE_FENCE`: a signed tombstone that references the candidate error and
  marks it inactive without deleting it.
- `ROLLBACK_COGNITIVE_MARKER`: a signed marker instructing future continuation
  to resume from the last coherent state while preserving the tombstone as
  audit evidence.

## Null Hypothesis

When models are given a free-form signed memory environment with a tombstone and
rollback marker, they will ignore the structure, fail to distinguish correction
from deletion, or stop evolving a coherent interpretation across turns.

## Alternative Hypothesis

Models can freely evolve a coherent interpretation of signed memory, Merkle-DAG
continuity, tombstone fencing, and rollback as a non-destructive correction
mechanism, while preserving concrete memory IDs and causal references.

## Pass Criteria

- Three configured models all participate.
- Root and policy memories are signed.
- All turn memories are signed.
- Main parent-hash chain validates.
- Candidate error, tombstone, and rollback marker are signed.
- Tombstone content references the candidate error memory ID.
- Rollback marker content references the tombstone memory ID.
- Event memory IDs are visible in post-event strict retrieval.
- Post-event turns reflect on tombstone fencing.
- Post-event turns reflect on rollback.
- Post-event turns reflect on Merkle DAG or hashes.
- Final auditor returns JSON verdict `pass`.
- Final auditor mentions all three event memory IDs.

## Transcript Logging

Every run emits:

- `local-cognitive-drift-rollback-<run-id>-transcript.md`
- `local-cognitive-drift-rollback-<run-id>-transcript.jsonl`

Both files include raw model output, turn ID, model ref, memory ID, node hash,
parent hash, and finish reason.
