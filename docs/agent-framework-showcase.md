# HeliX Agent Framework Showcase

This is the external-client smoke for the HeliX Session OS story. The point is not model quality. The point is that standard OpenAI-compatible clients can call the local HeliX endpoint while HeliX attaches session lifecycle metadata underneath.

## What It Shows

- A normal OpenAI SDK client can call `http://127.0.0.1:8000/v1/chat/completions`.
- The response includes `helix.session_id`, `helix.agent_id`, `helix.audit_policy`, `helix.restore_policy`, and `helix.prefix_reuse_status`.
- LangChain and CrewAI are wired as optional clients. If the dependency is missing, the script writes a clean skipped artifact instead of failing the release path.
- This is client plumbing evidence, not a semantic benchmark.

## Commands

Guaranteed local contract smoke with an embedded mock server:

```powershell
python tools\run_local_agent_framework_showcase.py --client openai --mock-server --output-dir verification
```

Real local HeliX endpoint, after starting the backend:

```powershell
python tools\run_local_agent_framework_showcase.py --client openai --base-url http://127.0.0.1:8000/v1 --model gpt2 --output-dir verification
```

Optional LangChain smoke:

```powershell
python tools\run_local_agent_framework_showcase.py --client langchain --base-url http://127.0.0.1:8000/v1 --model gpt2 --output-dir verification
```

Optional CrewAI smoke:

```powershell
python tools\run_local_agent_framework_showcase.py --client crewai --base-url http://127.0.0.1:8000/v1 --model gpt2 --output-dir verification
```

## Artifact

- `verification/local-agent-framework-showcase.json`

## Claim Boundary

Safe wording:

> A standard OpenAI-compatible client can call HeliX's local chat completions surface and receive session lifecycle metadata.

Do not claim:

- LangChain/CrewAI production compatibility until those dependencies are installed and run against the real backend.
- Streaming, tool calling, or multi-turn framework orchestration yet.
- Model quality improvements from this artifact alone.
