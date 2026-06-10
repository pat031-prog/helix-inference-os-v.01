# HeliX Agentic Trust Runtime

HeliX should be presented as a control plane for agentic work, not just a CLI with memory and verification artifacts.

The commercial promise is simple:

> Run agentic tasks through engines such as OpenCode, DeepInfra and Gemini, while HeliX owns isolation, context policy, patch capture, verification, memory boundaries and claim limits.

## Product Surface

HeliX has four user-facing layers:

1. Conversation
   Fast chat, technical analysis and product thinking. No evidence or agent runner is activated unless the user asks for it.

2. Flow Profiles
   Commercial modes built from verification protocols. Examples: `web`, `web-recursive`, `patch-safe`, `doc-grounded`, `privacy-swarm`, `resilient-task`, `multi-review` and `deep-lab`.

3. Task Capsules
   Each important task records goal, lane, engine, models used, sandbox, changed files, patch hash, checks, trust card and claim boundary.

4. Lab Profiles
   Runtime checks derived from the nuclear suites. These are not raw suite dumps. They are product guarantees selected by risk level.

## Execution Model

```text
goal
  -> route flow
  -> create task capsule
  -> create sandbox
  -> execute engine
  -> capture patch and transcript
  -> run checks
  -> render trust card
  -> user decides apply/retry/deepen
```

The core rule is that HeliX does not silently turn every conversation into an audit. Depth is proportional to risk.

## Assurance Levels

| Level | Default Use | Runtime Checks |
| --- | --- | --- |
| quick | normal `/task` and `web` flow | sandbox provenance, patch hash, trust card |
| balanced | recursive web, docs, patch safety | quick plus capsule verification and selected lab profile |
| strict | resilient tasks, multi-review | balanced plus stronger warnings, disagreement and claim boundaries |
| deep | public claims and methodology work | explicit nuclear methodology suites |

## Honest Differentiator

HeliX does not prove that a model answer is true. It proves bounded operational claims:

- where an agent ran
- what files changed
- what patch was captured
- what evidence was consulted
- which checks passed
- which branches or sources were quarantined
- what the run does not prove

This is the product difference from a normal coding agent. HeliX is the authority over the task boundary.

## OpenCode Role

OpenCode is the execution backend for code and web tasks. HeliX wraps it:

- OpenCode edits only inside `.helix/opencode-runs/<run_id>/worktree`
- HeliX writes a task brief into the sandbox
- HeliX captures `patch.diff`, transcript, manifest, task capsule and trust card
- the real repo changes only after explicit `/apply last`

This lets HeliX benefit from OpenCode's agentic coding power while preserving HeliX's verification layer.

## First Demo

`web/helix-mission-control/` is the first product demo. It shows:

- Flow Gallery
- Mission pipeline
- Trust Card Explorer
- Artifact map
- Runtime protocol cards

It is intentionally static and dependency-free so it can be opened, reviewed and patched by any agent.
