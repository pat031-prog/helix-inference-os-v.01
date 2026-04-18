# Hardware Follow-Up for Development

## Local disk constraint

Current local development has only **15 GB of free disk** available.

This changes development priorities:

- Do **not** download large models for day-to-day development.
- Use the tiny/test models that already exist in the workspace or are built locally in code.
- Validate architecture first: SSE streaming, tools, agent loop, routing, sessions, and frontend token rendering.
- Treat model quality as secondary until the system works end-to-end.

## Development policy

For local development and testing:

- Prefer `tiny-gpt2` and the tiny local Hugging Face fixtures already supported by the CLI.
- Keep using dummy models for feature validation.
- Avoid pulling anything large enough to threaten disk stability.

For a real local model smoke test:

- `Qwen 3 0.6B` in a compressed format is the upper practical target.
- `Qwen 3.5 0.8B` is the maximum acceptable local test size for now.
- Anything bigger should be tested on a remote server instead of the local machine.

## What matters right now

A small model that produces low-quality answers but proves all of this is more valuable at the current stage than a big model that does not fit:

- token streaming works
- tools are invoked correctly
- the agent loop completes
- the frontend renders live output
- sessions and traces are persisted

The goal is to make the system feel like "magic" even with tiny models by tightening orchestration, tools, UX, and streaming behavior.

## Storage notes

- Workspace artifacts like sessions, knowledge chunks, and traces are relatively small.
- The real disk pressure comes from model weights and export artifacts.
- Development decisions should optimize for model footprint first.
