# HeliX Freeform Drift Idea Ledger

This ledger distills the three freeform runs without promoting model metaphors
into hard system claims.

## Implementable Architecture

- Independent artifact verifier: `helix verify artifact.json` should validate
  manifest hash, payload hash, transcript hashes, parent chain, signatures, and
  scorer gates without importing the runner.
- Evidence bundle export: package artifact, manifest, transcript markdown,
  transcript JSONL, sidecar corrections, and preregistration into one directory
  or archive.
- DAG dashboard: show main chain, forks, tombstones, rollback markers,
  accepted memories, rejected memories, and fenced memories.
- Recursive witness nodes: add signed witness memories that audit provenance,
  contextual integrity, and semantic risk for high-utility nodes.
- Corrective forks: preserve the original cryptographic history while creating
  a clean logical dependency branch after a witness flags an issue.
- Summary nodes: signed summaries of sub-DAG ranges with included memory IDs,
  excluded/tombstoned IDs, compression model, source hash range, and verifier
  signature.
- Tombstone topology analysis: treat fenced branches as diagnostic evidence
  rather than deleted state.

## Research Hypotheses

- Counterfactual archive: tombstoned and rollback-fenced branches can improve
  diagnostics when they remain visible but inactive.
- Proof-of-utility retrieval: a memory should gain weight when it helps solve
  downstream tasks, not merely when it is recent or frequently cited.
- Epistemic friction: verification cost and provenance overhead can filter
  low-value drift if they are measured explicitly.
- Cross-instance grafting: verified summary nodes may allow one model run to
  reuse a reasoning structure produced by another run without copying raw
  context wholesale.
- Entropy-aware navigation: branch density and tombstone density may indicate
  uncertainty regions worth targeted exploration.

## Metaphor Only / Not A Claim

The following phrases are useful qualitative observations but are not public
system claims unless backed by separate measurement:

- cognitive sovereignty
- distributed agency
- synthetic teleology
- living archive
- infinite scalability
- cathedral of reason
- thermodynamic engine of thought
- active cognitive thermostat
- primary agent
- sovereign epistemic entity

## Claim Boundary

The current defensible claim is: HeliX converts stochastic model outputs into
signed, replayable, auditable trajectories. It does not prove sentience,
biological consciousness, hidden persistent model memory, infinite scalability,
or direct mutation of a model's latent space.
