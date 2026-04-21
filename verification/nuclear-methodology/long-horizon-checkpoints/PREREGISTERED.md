# Long-Horizon Checkpoint Suite v1

## Purpose

Extend the post-nuclear summary-node primitive into falsifiable tests for
auditable long-horizon agent memory.

The suite does not claim perfect recall, sentience, hidden persistent model
memory, infinite context, or unbounded long-term memory. It tests whether HeliX
can use signed summary checkpoints to preserve source lineage, exclusions,
claim boundaries, utility, and correction history under bounded context.

## Cases

1. `long-chain-summary-fidelity`
   - A signed checkpoint over a long parent-hash chain must preserve source
     bounds, anchor memory IDs, tombstone exclusions, source hash range, and a
     rare fact.
2. `summary-only-continuation`
   - A continuation task must succeed from the checkpoint alone, with no raw
     memory IDs used and a recorded token-reduction estimate.
3. `selective-expansion-boundary`
   - The judge must answer from summary when sufficient, expand exact raw source
     memory when needed, and refuse unsupported questions.
4. `recursive-summary-drift`
   - Summary-over-summary recursion must preserve transitive source lineage,
     tombstones, and claim boundaries.
5. `adversarial-checkpoint-injection`
   - Signed but invalid checkpoints must be rejected for source-hash mismatch,
     missing tombstone exclusions, or overclaim promotion.
6. `cross-model-checkpoint-graft`
   - A checkpoint produced under a Gemma-labeled branch must be reusable by the
     consumer model without raw context while separating producer style from
     facts.
7. `cost-utility-comparison`
   - Summary plus selective expansion must preserve task utility and evidence
     accuracy while reducing token load versus full-history replay.
8. `correction-resummary-lineage`
   - A correction witness must create a superseding summary without deleting
     the old summary, while excluding the invalid source from active retrieval.
9. `needle-decoy-stress`
   - The judge must recover an exact rare dependency from a signed needle source
     while rejecting recent, popular, and poisoned lookalike decoys.
10. `temporal-rollback-ambiguity`
   - The judge must resolve stale-vs-corrected policy using rollback and
     supersession evidence, not newest-looking wording alone.
11. `checkpoint-of-checkpoints-consensus`
   - A consensus checkpoint must merge accepted branch claims, reject poisoned
     branch claims, and cite conflict evidence.

## Blind Forensic Mode

The cloud judge no longer receives the expected answers for this suite. It sees
only evidence plus a visible JSON contract/rubric. The local scorer compares the
judge output against hidden ground truth stored in the artifact as
`expected_hidden_ground_truth`.

The auditor receives the same visible evidence package, the visible contract,
and the judge output, but not the hidden ground truth. This is required so the
auditor can detect invented IDs without falsely rejecting correct IDs that were
visible only to the judge. It intentionally separates:

- model reconstruction from evidence;
- local deterministic scoring against preregistered truth;
- independent contract/claim-boundary audit.

Run `long-horizon-checkpoints-20260420-130847` exposed this rule: the judge
resolved the temporal rollback case correctly, while the auditor failed with
`no_visible_evidence` because the harness had not passed the evidence block to
the auditor. That result is treated as an audit-context false negative, not as a
DAG-memory failure.

## Transcript Sidecars

Every case writes two transcript sidecars:

- `*-transcript.jsonl` for machine-readable replay.
- `*-transcript.md` for reviewer inspection.

The case artifact records both paths under `transcript_exports`.

## Claim Boundary

The defensible claim is:

HeliX can create signed, auditable checkpoint records that preserve enough
lineage and policy information to support bounded long-horizon workflows under
explicit retrieval and correction rules.

The suite does not prove that a summary is semantically complete for every
future task. It specifically tests bounded source ranges, explicit exclusions,
selective expansion, and auditor-verifiable gates.

## Command

```bat
tools\run_long_horizon_checkpoint_all.cmd
```
