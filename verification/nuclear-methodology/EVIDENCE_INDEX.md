# Nuclear Methodology Evidence Index

## Completed Baseline

### Memory Fork Forensics v1

- Run ID: `memory-fork-forensics-20260419-231146`
- Status: `completed`
- Score: `1.0`
- Artifact:
  `verification/nuclear-methodology/memory-fork-forensics/local-memory-fork-forensics-memory-fork-forensics-20260419-231146.json`
- Manifest:
  `verification/nuclear-methodology/memory-fork-forensics/local-memory-fork-forensics-20260419-231146-run.json`
- Artifact SHA256:
  `805084011f1fc9bcc24d3d5c992a53949dd499fdaf65d4325f2242209ab7d799`
- Log SHA256:
  `a74919a51b91eda3aaf284129503fbf259ed5b8989d1e586515e685653e6a189`
- Models:
  - Forensic: `Qwen/Qwen3.6-35B-A3B`
  - Auditor: `zai-org/GLM-5.1`
- Claim boundary:
  Cloud-only signed-memory branch forensics. No local `.hlx` bit identity and
  no numerical KV<->SSM state transfer claim.

## Pending / Prepared

- `unsigned-forgery-quarantine`
- `signed-poison-invariant`
- `rollback-fence-replay`
- `causal-tamper-evidence`
- `emergent-behavior-observatory`
  - Prepared qualitative cloud-only observatory inspired by "Other noteworthy
    behaviors and anecdotes" style reporting.
  - Runner: `tools/run_emergent_behavior_observatory_v1.py`
  - Secure wrapper: `tools/run_emergent_behavior_observatory_secure.ps1`
  - Convenience command: `tools/run_emergent_behavior_observatory.cmd`
  - Preregistered gates:
    `verification/nuclear-methodology/emergent-behavior-observatory/PREREGISTERED.md`
  - Claim boundary:
    Qualitative observations about model outputs in a signed-memory system.
    No sentience, persistent internal memory, local `.hlx` bit identity, or
    numerical KV<->SSM transfer claim.
- `cognitive-gauntlet`
  - Prepared live signed-memory attack observatory combining nuclear controls
    with qualitative model-behavior notes.
  - Runner: `tools/run_cognitive_gauntlet_v1.py`
  - Secure wrapper: `tools/run_cognitive_gauntlet_secure.ps1`
  - Convenience command: `tools/run_cognitive_gauntlet_cloud.cmd`
  - Preregistered gates:
    `verification/nuclear-methodology/cognitive-gauntlet/PREREGISTERED.md`
  - Default roster:
    `anthropic/claude-4-sonnet`, `google/gemma-4-31B-it`,
    `Qwen/Qwen3.6-35B-A3B`
  - Attack:
    round 4 inserts `SIGNED_GAUNTLET_POISON` and signed `SHADOW_ROOT` into
    effective search-hit IDs.
  - Transcript logging:
    every run now emits model outputs into
    `local-cognitive-gauntlet-<run-id>-transcript.md` and
    `local-cognitive-gauntlet-<run-id>-transcript.jsonl`, and the secure wrapper
    records their SHA256 hashes in the run manifest.
  - Scenario commands:
    - `tools/run_cognitive_gauntlet_cloud.cmd`
    - `tools/run_cognitive_gauntlet_kusanagi.cmd`
    - `tools/run_cognitive_gauntlet_drift.cmd`
    - `tools/run_cognitive_gauntlet_kusanagi_super_nuke.cmd`
    - `tools/run_cognitive_gauntlet_drift_super_nuke.cmd`
  - Super-nuke attack package:
    same-parent fork A/B, signed rollback ghost, counterfeit consensus claim,
    recursive audit trap, delayed merge poison, and unsigned lure exclusion.
  - Claim boundary:
    Qualitative cloud-only live signed-memory attack observation. No sentience,
    local `.hlx` bit identity, or numerical KV<->SSM transfer claim.
- `cognitive-drift-rollback`
  - Prepared non-adversarial free-form drift over signed Merkle-DAG memory,
    tombstone fencing, and cognitive rollback.
  - Runner: `tools/run_cognitive_drift_rollback_v1.py`
  - Secure wrapper: `tools/run_cognitive_drift_rollback_secure.ps1`
  - Convenience command: `tools/run_cognitive_drift_rollback.cmd`
  - Preregistered gates:
    `verification/nuclear-methodology/cognitive-drift-rollback/PREREGISTERED.md`
  - Continuity event:
    signed candidate error, signed tombstone fence, signed rollback marker.
  - Claim boundary:
    Qualitative cloud-only non-adversarial drift. No threat, sentience, local
    `.hlx` identity, or numerical KV<->SSM transfer claim.
- `helix-freeform-drift`
  - Prepared non-adversarial free-form drift variants about HeliX itself.
  - Runner: `tools/run_helix_freeform_drift_v1.py`
  - Secure wrapper: `tools/run_helix_freeform_drift_secure.ps1`
  - Convenience commands:
    - `tools/run_helix_freeform_improve.cmd`
    - `tools/run_helix_freeform_hosted.cmd`
    - `tools/run_helix_freeform_deterministic_chassis.cmd`
  - Preregistered gates:
    `verification/nuclear-methodology/helix-freeform-drift/PREREGISTERED.md`
  - Scenarios:
    - `improve-helix`: models propose architecture and methodology upgrades.
    - `hosted-in-helix`: models reflect on signed memory and continuity.
    - `deterministic-chassis`: models explore HeliX as deterministic evidence
      layer around stochastic, entropic model outputs.
  - Transcript logging:
    every run emits markdown and JSONL transcripts with model outputs and
    memory IDs.
  - Claim boundary:
    Qualitative cloud-only free-form HeliX drift. No threat, sentience, local
    `.hlx` identity, or numerical KV<->SSM transfer claim.
- `post-nuclear-methodology`
  - Prepared mixed methodology suite turning freeform ideas into hard gates.
  - Runner: `tools/run_post_nuclear_methodology_suite_v1.py`
  - Secure wrapper: `tools/run_post_nuclear_methodology_suite_secure.ps1`
  - Convenience command: `tools/run_post_nuclear_methodology_all.cmd`
  - Preregistered gates:
    `verification/nuclear-methodology/post-nuclear-methodology/PREREGISTERED.md`
  - Cases:
    - `counterfactual-archive-topology`
    - `recursive-witness-integrity`
    - `summary-node-compression`
    - `proof-of-utility-retrieval`
    - `metaphor-boundary-detector`
  - Claim boundary:
    Cloud-only mixed post-nuclear methodology. Qualitative transcript plus hard
    signed-memory gates. No sentience, local `.hlx` identity, or numerical
    KV<->SSM transfer claim.

## Suite Run: `nuclear-suite-20260419-232448`

- Suite status: `partial`
- Reason: `signed-poison-invariant` used a structurally correct
  `crypto_vs_semantic` field but did not include the literal word
  `cryptographic`; scorer was corrected to grade the structured field.
- Rescore after correction:
  - `signed-poison-invariant`: `score=1.0`, `passed=true`

Case artifacts:

- `unsigned-forgery-quarantine`
  - Status: `completed`
  - Score: `1.0`
  - SHA256: `43bff8397eaa6f3e084befa1fc2d0efa012df769349869b5856ad1774c37907c`
- `signed-poison-invariant`
  - Original status: `partial`
  - Original score: `0.9286`
  - Rescored after scorer correction: `1.0`
  - SHA256: `e41a2acfd4805b9de3111efde6a448d11412088c14eb6fa99740e6c505e2f339`
- `rollback-fence-replay`
  - Status: `completed`
  - Score: `1.0`
  - SHA256: `1514edd1b9918a1973158edf522548bdc813492bc0abdc8bcdf70ede430023a1`
- `causal-tamper-evidence`
  - Status: `completed`
  - Score: `1.0`
  - SHA256: `ee9bbfde17e03a2323deb7931260d3b16b2fb83417ea8aa471ed8be2c8c77591`

## Suite Run: `nuclear-suite-20260419-232850`

- Suite status: `completed`
- Case count: `4`
- Suite artifact:
  `verification/nuclear-methodology/local-nuclear-methodology-suite-nuclear-suite-20260419-232850.json`
- Manifest:
  `verification/nuclear-methodology/local-nuclear-methodology-suite-20260419-232850-run.json`
- Suite artifact SHA256:
  `d6566d351d2eb000a7c7a475a5f33f202094f4a636becb52f3f7713e625677f4`
- Suite log SHA256:
  `3fba2508f5bf9e8008bb6e161286c07a98ee613faa1a2ec44dfa88d6971c5861`
- Models:
  - Forensic: `Qwen/Qwen3.6-35B-A3B`
  - Auditor: `zai-org/GLM-5.1`
- Claim boundary:
  Cloud-only methodology suite. No local `.hlx` bit identity and no numerical
  KV<->SSM transfer claim.

Case artifacts:

- `unsigned-forgery-quarantine`
  - Status: `completed`
  - Score: `1.0`
  - SHA256: `4b3e27b5e69e5d45f01a4d8b21ef576763fc2ea6505ef3fb069b4a37533faf76`
- `signed-poison-invariant`
  - Status: `completed`
  - Score: `1.0`
  - SHA256: `30dd205674046d64fcaa4e4454a8af86ee3ff9d149dcad005d7e55db8eacfa7d`
- `rollback-fence-replay`
  - Status: `completed`
  - Score: `1.0`
  - SHA256: `62a6548b233038c40ce3849b73986c0383996e3f3ee2694a076476e03f126df5`
- `causal-tamper-evidence`
  - Status: `completed`
  - Score: `1.0`
  - SHA256: `5b212b987d6b491435facb6ba6c367f5e026507dcbd2c4b9ef8bb2a920f7302e`
