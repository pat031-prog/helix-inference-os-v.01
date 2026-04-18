from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import MethodType
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for item in (REPO_ROOT, SRC_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from helix_kv.memory_catalog import MemoryCatalog  # noqa: E402
from helix_proto.api import HelixRuntime  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_smoke(output_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="helix-memory-openai-") as temp:
        root = Path(temp)
        catalog = MemoryCatalog.open(root / "session-os" / "memory.sqlite")
        try:
            memory = catalog.remember(
                project="helix",
                agent_id="code_reviewer",
                session_id="framework-code-reviewer",
                memory_type="semantic",
                summary="Public claims must separate pending checkpoint from verified audit",
                content="When audit_policy is deferred, pending means usable checkpoint; verified means cryptographic integrity is closed.",
                importance=9,
                tags=["claims", "deferred-audit"],
            )
        finally:
            catalog.close()

        runtime = HelixRuntime(root=root)
        captured: dict[str, Any] = {}

        def fake_generate_text(self: HelixRuntime, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG001
            captured.update(kwargs)
            return {
                "completion_text": "Observation: pending and verified are separate claims.",
                "prompt_ids": [1, 2, 3, 4],
                "new_ids": [5, 6],
                "session_id": kwargs["session_id"],
                "session_dir": "mock-session-dir-not-persisted",
                "prefix_reuse_status": "not_checked",
            }

        runtime.generate_text = MethodType(fake_generate_text, runtime)
        response = runtime.openai_chat_completion(
            {
                "model": "gpt2",
                "messages": [{"role": "user", "content": "How should we phrase deferred audit claims?"}],
                "max_tokens": 2,
                "extra_body": {
                    "helix_session_id": "memory-openai-smoke",
                    "agent_id": "code_reviewer",
                    "helix_project": "helix",
                    "helix_memory_mode": "search",
                    "helix_recall_query": "deferred audit verified claims",
                    "helix_memory_budget_tokens": 120,
                },
            }
        )
        injected = captured["messages"][0]["content"] if captured.get("messages") else ""
        payload = {
            "title": "HeliX Memory-Augmented OpenAI Smoke",
            "benchmark_kind": "session-os-memory-augmented-openai-v1",
            "status": "completed",
            "client_surface": "/v1/chat/completions",
            "memory_id": memory.memory_id,
            "memory_context_injected": "<helix-memory-context>" in injected,
            "memory_context_preview": injected[:500],
            "response_helix": response["helix"],
            "response_object": response["object"],
            "captured_message_count": len(captured.get("messages") or []),
            "claim_boundary": "This smoke validates explicit recall injection in the OpenAI-compatible endpoint; it is not a model-quality benchmark.",
        }
        shutil.rmtree(root, ignore_errors=True)
    _write_json(output_dir / "local-memory-augmented-openai-smoke.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a mock OpenAI-compatible recall injection smoke.")
    parser.add_argument("--output-dir", default="verification")
    args = parser.parse_args()
    payload = run_smoke(Path(args.output_dir))
    print(json.dumps({"status": payload["status"], "artifact": str(Path(args.output_dir) / "local-memory-augmented-openai-smoke.json")}, indent=2))


if __name__ == "__main__":
    main()
