from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helix_proto.api import HelixRuntime
from tools.run_local_hybrid_stress import _json_ready, _write_json


class _SmokeRuntime(HelixRuntime):
    def generate_text(self, **kwargs: Any) -> dict[str, Any]:  # noqa: D401
        return {
            "completion_text": "Observation: HeliX session lifecycle is available through an OpenAI-shaped response.",
            "prompt_ids": [1, 2, 3, 4],
            "new_ids": [5, 6],
            "session_id": kwargs.get("session_id"),
            "session_dir": str(Path(self.root) / "models" / str(kwargs.get("alias")) / "sessions" / str(kwargs.get("session_id"))),
        }


def run_openai_smoke(args: argparse.Namespace) -> dict[str, Any]:
    runtime = _SmokeRuntime(root=Path(args.output_dir))
    response = runtime.openai_chat_completion(
        {
            "model": str(args.model),
            "messages": [{"role": "user", "content": "Give one safe HeliX Session OS claim."}],
            "max_tokens": 2,
            "extra_body": {
                "helix_session_id": "openai-smoke-session",
                "agent_id": "openai-smoke-agent",
                "audit_policy": "deferred",
                "compression_mode": "turbo-int8-hadamard",
                "restore_policy": "session",
            },
        }
    )
    payload = {
        "title": "HeliX OpenAI-Compatible API Smoke",
        "benchmark_kind": "session-os-openai-compatible-smoke-v0",
        "status": "completed",
        "object": response.get("object"),
        "model": response.get("model"),
        "session_recorded": bool((response.get("helix") or {}).get("session_id")),
        "response": response,
        "claim_boundary": "This smoke validates the local response contract; model-quality benchmarking is covered by separate runners.",
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "local-openai-compatible-smoke.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a lightweight OpenAI-compatible response smoke artifact.")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--output-dir", default="verification")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_openai_smoke(args)
    print(json.dumps(_json_ready(payload), indent=2))


if __name__ == "__main__":
    main()
