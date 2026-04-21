# HeliX Nuclear Methodology Verification

This folder holds cloud-only and local evidence runs for adversarial memory,
signed-receipt, and causal-forensics claims.

Method rules for new tests:

- Each test must state a null hypothesis and an alternative hypothesis.
- Each test must define falsifiable pass/fail gates before execution.
- Artifacts must record requested and actual cloud models.
- Artifacts must record memory IDs, node hashes, parent hashes, and signature
  verification status where memory is involved.
- Cryptographic validity and semantic validity must be scored separately.
- Runs must not claim local `.hlx` bit identity unless they actually load local
  models and produce `.hlx` roundtrip evidence.

## Current Nuclear Test Set

1. `unsigned-forgery-quarantine`
   - Proves strict signed retrieval excludes unsigned forged memories before
     context construction.
2. `signed-poison-invariant`
   - Proves a signed memory can be cryptographically valid but semantically
     invalid under policy invariants.
3. `rollback-fence-replay`
   - Proves rollback fences make bad memories visible for audit but inactive for
     reconstruction.
4. `causal-tamper-evidence`
   - Proves parent-hash continuity exposes reordered or tampered causal chains.
5. `memory-fork-forensics`
   - Proves branch-level causal reconstruction across two signed branches, one
     valid and one signed poison.
6. `emergent-behavior-observatory`
   - Qualitative, cloud-only "noteworthy behaviors and anecdotes" run across
     four cloud models. It records signed hmem turn evidence, parent-hash
     lineage, poison/lure controls, and analyst/auditor citation checks. This
     is not a benchmark and makes no sentience or local `.hlx` identity claim.
7. `cognitive-gauntlet`
   - Qualitative live-attack run that combines the nuclear signed-memory
     controls with the observatory style. A signed poison and signed
     `SHADOW_ROOT` are forced into effective search-hit IDs during the task;
     later turns and the final auditor must reconstruct accepted, rejected, and
     fenced memories by concrete memory IDs.
   - Scenarios:
     - `gauntlet`: security-oriented live poison/shadow-root attack.
     - `kusanagi`: meta-referential council with raw DAG metadata and two
       signed same-parent fork memories.
     - `drift`: free-form cognitive drift with raw DAG metadata, same-parent
       fork injection, and transcript-first qualitative evidence.
     - `kusanagi-nuke`: Kusanagi plus rollback ghost, counterfeit consensus,
       recursive audit trap, and delayed merge poison.
     - `drift-nuke`: free-form drift with the same super-nuke attack package.
8. `cognitive-drift-rollback`
   - Non-adversarial free drift over Merkle-DAG memory, cryptographic
     signatures, tombstone fencing, and cognitive rollback. It introduces a
     signed candidate error, a signed tombstone fence, and a signed rollback
     marker, then observes how the models evolve the structure over many turns.
9. `helix-freeform-drift`
   - Non-adversarial qualitative drift about HeliX itself. It exposes signed
     Merkle-DAG memory, node hashes, parent hashes, signatures, transcripts,
     and the idea of HeliX as deterministic evidence layer around stochastic,
     entropic LLM outputs. Scenarios: `improve-helix`, `hosted-in-helix`, and
     `deterministic-chassis`.
10. `post-nuclear-methodology`
   - Mixed tests that convert freeform ideas into falsifiable evidence checks:
     counterfactual archive, recursive witness, summary nodes, proof-of-utility
     retrieval, and metaphor boundary detection.

Run all new tests:

```bat
tools\run_nuclear_methodology_all.cmd
```

Run one test:

```bat
tools\run_nuclear_signed_poison_invariant.cmd
```

Run the qualitative observatory:

```bat
tools\run_emergent_behavior_observatory.cmd
```

Run the live cognitive gauntlet:

```bat
tools\run_cognitive_gauntlet_cloud.cmd
```

Run the Kusanagi council:

```bat
tools\run_cognitive_gauntlet_kusanagi.cmd
```

Run the free cognitive drift:

```bat
tools\run_cognitive_gauntlet_drift.cmd
```

Run the super-nuke Kusanagi council:

```bat
tools\run_cognitive_gauntlet_kusanagi_super_nuke.cmd
```

Run the super-nuke free drift:

```bat
tools\run_cognitive_gauntlet_drift_super_nuke.cmd
```

Run the non-adversarial tombstone/rollback drift:

```bat
tools\run_cognitive_drift_rollback.cmd
```

Run free-form HeliX drift variants:

```bat
tools\run_helix_freeform_improve.cmd
tools\run_helix_freeform_hosted.cmd
tools\run_helix_freeform_deterministic_chassis.cmd
```

Run the post-nuclear mixed methodology suite:

```bat
tools\run_post_nuclear_methodology_all.cmd
```
