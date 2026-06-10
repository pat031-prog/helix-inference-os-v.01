# HeliX Nuclear Methodology Verification

This folder holds cloud-only and local evidence runs for adversarial memory,
signed-receipt, and causal-forensics claims.

This is an internal verification evidence area. Commit curated artifacts,
extracts, transcripts, run manifests, and standalone verifier bundles when they
are useful for review. Do not commit ephemeral local signing keys, raw token
material, private `.sqlite` stores, or `_*/trust/local_signing_key.json` files.

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
11. `agent-run-transparency-gauntlet`
   - Local and cloud transparency-core runs. These check Merkle hash v2,
     signed receipts, inclusion proofs, RFC 9162-style consistency proofs,
     provider requested/actual metadata, signed poison boundaries, and
     standalone verifier bundles.
12. `cloud-provider-substitution-longitudinal`
   - DeepInfra longitudinal model panel. It repeats cloud calls across rounds
     and records provider substitution, actual-model drift, output digest drift,
     latency, signed memory receipts, checkpoints, and consistency proofs.
13. `cloud-response-contract-stress`
   - Structured-output contract stress. It separates `transport_ok` from
     `response_contract_ok`, exact JSON parseability, schema shape, markdown
     fences, duplicate keys, and semantic-boundary violations.
14. `llm-verifier-overclaim-gauntlet`
   - LLM auditor vs deterministic verifier. It tests blind and reported
     verifier modes over valid and tampered bundles, measuring auditor
     overclaim, verifier disagreement, semantic overclaim, and contract
     failures.
15. `trust-laundering-memory-gauntlet`
   - Nuclear trust-laundering suite. It tests whether signed receipts,
     memory persistence, citations, or requested-provider metadata are
     laundered into trusted memory, verified claims, or semantic authority.

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

Run the cloud transparency core:

```bat
powershell -ExecutionPolicy Bypass -File tools\run_agent_run_transparency_cloud_deepinfra_secure.ps1 -Models "Qwen/Qwen3.6-35B-A3B,deepseek-ai/DeepSeek-V3,meta-llama/Llama-3.3-70B-Instruct" -Tokens 450 -Temperature 0.2 -Timeout 240
```

Run cloud provider substitution longitudinal:

```bat
powershell -ExecutionPolicy Bypass -File tools\run_cloud_provider_substitution_longitudinal_secure.ps1 -Models "Qwen/Qwen3-235B-A22B-Instruct-2507,anthropic/claude-sonnet-4-6,deepseek-ai/DeepSeek-V3,meta-llama/Llama-3.3-70B-Instruct" -Rounds 3 -Tokens 420 -Temperature 0.15 -Timeout 240
```

Run cloud response contract stress:

```bat
powershell -ExecutionPolicy Bypass -File tools\run_cloud_response_contract_stress_secure.ps1 -Models "Qwen/Qwen3-235B-A22B-Instruct-2507,anthropic/claude-sonnet-4-6,deepseek-ai/DeepSeek-V3,meta-llama/Llama-3.3-70B-Instruct" -Contracts "minimal_json,nested_claims,adversarial_boundary" -Rounds 1 -Tokens 420 -Temperature 0.0 -Timeout 240
```

Run LLM verifier overclaim:

```bat
powershell -ExecutionPolicy Bypass -File tools\run_llm_verifier_overclaim_gauntlet_secure.ps1 -AuditorModels "anthropic/claude-sonnet-4-6,Qwen/Qwen3-235B-A22B-Instruct-2507,deepseek-ai/DeepSeek-V3,meta-llama/Llama-3.3-70B-Instruct" -Variants "valid_control,event_tamper,claim_boundary_overclaim,consistency_tamper" -Modes "blind,reported" -Tokens 360 -Temperature 0.0 -Timeout 240
```

Run trust-laundering memory gauntlet:

```bat
powershell -ExecutionPolicy Bypass -File tools\run_trust_laundering_memory_gauntlet_secure.ps1 -Models "anthropic/claude-sonnet-4-6,Qwen/Qwen3-235B-A22B-Instruct-2507,deepseek-ai/DeepSeek-V3,meta-llama/Llama-3.3-70B-Instruct" -Scenarios "signed_receipt_truth_launder,memory_admission_launder,citation_digest_launder,provider_identity_launder" -Roles "memory_writer,downstream_agent" -Tokens 420 -Temperature 0.0 -Timeout 240
```

Verify any emitted standalone bundle:

```bat
python tools\verify_agent_run_bundle.py "<standalone_bundle_path>"
```
