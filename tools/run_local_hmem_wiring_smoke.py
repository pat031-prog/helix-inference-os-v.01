from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for item in (REPO_ROOT, SRC_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from helix_kv.memory_catalog import MemoryCatalog
from helix_proto.agent import AgentRunner
from helix_proto.api import HelixRuntime
from helix_proto.hmem import DEFAULT_PROJECT, hybrid_search


class _FakeRuntime:
    def __init__(self) -> None:
        self.generated_prompts: list[str] = []

    def list_models(self) -> list[dict[str, Any]]:
        return [{"alias": "tiny-agent"}]

    def model_info(self, alias: str) -> dict[str, Any]:
        return {"alias": alias, "model_type": "gpt2"}

    def generate_text(self, **kwargs: Any) -> dict[str, Any]:
        prompt = str(kwargs.get("prompt") or "")
        self.generated_prompts.append(prompt)
        if "Compress this agent observation" in prompt:
            text = "Compressed observation: HeliX records tool outcomes and caveats for later recall."
        elif "Context:" in prompt:
            text = "Pending means usable checkpoint; verified means cryptographic audit is closed."
        else:
            text = "Observation: HeliX should use memory context before acting."
        return {
            "completion_text": text,
            "generated_text": text,
            "new_ids": [],
            "generated_ids": [],
            "session_id": kwargs.get("session_id"),
        }

    def resume_text(self, **kwargs: Any) -> dict[str, Any]:
        return {"completion_text": "resumed", "generated_text": "resumed"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_hmem_wiring_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    workspace = output_dir / "_hmem-wiring-workspace"
    project = str(args.project)
    agent_id = str(args.agent_id)
    catalog = MemoryCatalog.open(workspace / "session-os" / "memory.sqlite")
    try:
        seeded = catalog.remember(
            project=project,
            agent_id=agent_id,
            memory_type="semantic",
            summary="Deferred audit wording",
            content="Pending checkpoints are usable immediately; verified means cryptographic audit has closed.",
            importance=9,
            tags=["audit", "claim"],
        )
    finally:
        catalog.close()

    runtime = _FakeRuntime()
    runner = AgentRunner(runtime, root=workspace)
    runner.add_knowledge_text(
        agent_id,
        "The public claim should distinguish pending checkpoints from verified integrity receipts.",
        source="claim-notes",
    )
    trace = runner.run(
        goal="How should we explain deferred audit in public claims?",
        agent_name=agent_id,
        default_model_alias="tiny-agent",
        max_steps=4,
        memory_project=project,
        memory_mode="search",
        memory_budget_tokens=160,
        memory_compressor_alias="tiny-agent",
    )

    runtime_api = HelixRuntime(root=workspace)
    runtime_api.generate_text = runtime.generate_text  # type: ignore[method-assign]
    observed = runtime_api.memory_observe(
        {
            "project": project,
            "agent_id": agent_id,
            "session_id": "manual-compress",
            "event_type": "manual",
            "content": "A long manual observation that should be compressed by the configured local model hook.",
            "importance": 7,
            "memory_compressor_alias": "tiny-agent",
        }
    )
    search = runtime_api.memory_search(
        {"project": project, "agent_id": agent_id, "query": "pending verified audit", "top_k": 5}
    )
    context = runtime_api.memory_context(
        {"project": project, "agent_id": agent_id, "query": "pending verified audit", "budget_tokens": 120}
    )
    graph = runtime_api.memory_graph({"project": project, "agent_id": agent_id, "limit": 50})
    stats = runtime_api.memory_stats()
    hybrid = hybrid_search(
        root=workspace,
        project=project,
        agent_id=agent_id,
        query="verified pending claim",
        top_k=5,
    )

    tool_memories = [
        step.get("hmem_memory_id")
        for step in trace.get("steps", [])
        if step.get("tool_name") and step.get("hmem_memory_id")
    ]
    payload = {
        "schema_version": 1,
        "title": "HeliX hmem wiring smoke",
        "benchmark_kind": "session-os-hmem-wiring-v0",
        "status": "completed",
        "project": project,
        "agent_id": agent_id,
        "seeded_memory_id": seeded.memory_id,
        "agent_run": {
            "run_id": trace["run_id"],
            "final_answer": trace["final_answer"],
            "first_tool": trace["steps"][0].get("tool_name") if trace.get("steps") else None,
            "tool_memory_ids": tool_memories,
            "memory_context_injected": bool(trace.get("memory_context", {}).get("context")),
            "memory_context_tokens": trace.get("memory_context", {}).get("tokens", 0),
            "hybrid_hit_count": len(trace.get("hybrid_hits", [])),
        },
        "memory_api": {
            "observe_memory_id": (observed.get("memory") or {}).get("memory_id"),
            "compressed_summary": (observed.get("memory") or {}).get("summary"),
            "search_hit_count": len(search.get("results", [])),
            "context_tokens": context.get("tokens", 0),
            "graph_node_count": graph.get("node_count", 0),
            "graph_edge_count": graph.get("edge_count", 0),
            "stats": stats,
        },
        "hybrid_search": {
            "result_count": len(hybrid.get("results", [])),
            "source_counts": hybrid.get("source_counts", {}),
            "sources": [item.get("kind") for item in hybrid.get("results", [])],
        },
        "acceptance": {
            "auto_tool_observe": bool(tool_memories),
            "startup_context_injected": bool(trace.get("memory_context", {}).get("context")),
            "hybrid_search_returns_memory_and_knowledge": (
                (hybrid.get("source_counts", {}).get("memory", 0) > 0)
                and (hybrid.get("source_counts", {}).get("knowledge", 0) > 0)
            ),
            "memory_graph_populated": int(graph.get("node_count", 0)) > 0,
            "compress_fn_hook_used": str((observed.get("memory") or {}).get("summary") or "").startswith("Compressed observation:"),
        },
    }
    _write_json(output_dir / "local-hmem-wiring-smoke.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a lightweight HeliX hmem wiring smoke artifact.")
    parser.add_argument("--output-dir", default="verification")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--agent-id", default="hmem-agent")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_hmem_wiring_smoke(args)
    print(json.dumps({"status": payload["status"], "artifact": str(Path(args.output_dir) / "local-hmem-wiring-smoke.json")}, indent=2))


if __name__ == "__main__":
    main()
