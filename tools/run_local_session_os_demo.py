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

from tools.run_local_hybrid_stress import _json_ready, _write_json
from tools.run_local_multimodel_hypervisor import build_parser as build_hypervisor_parser
from tools.run_local_multimodel_hypervisor import run_multimodel_hypervisor


def run_session_os_demo(args: argparse.Namespace) -> dict[str, Any]:
    hypervisor_args = build_hypervisor_parser().parse_args(
        [
            "--scenario",
            "coder-writer",
            "--profile",
            str(args.profile),
            "--code-model",
            str(args.code_model),
            "--writer-model",
            str(args.writer_model),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--prompt-tokens",
            str(args.prompt_tokens),
            "--codec",
            str(args.codec),
            "--audit-policy",
            str(args.audit_policy),
            "--output-dir",
            str(args.output_dir),
            "--artifact-name",
            "local-session-os-demo.json",
            "--device",
            str(args.device),
        ]
        + (["--local-files-only"] if bool(args.local_files_only) else [])
    )
    payload = run_multimodel_hypervisor(hypervisor_args)
    payload["title"] = "HeliX Local Session OS Demo"
    payload["benchmark_kind"] = "session-os-scheduler-demo-v0"
    payload["session_os_layers"] = [
        "model_registry",
        "model_lifecycle",
        "session_catalog",
        "prefix_resolver",
        "policy_scheduler",
        "deferred_audit",
    ]
    payload["claim_boundary"] = (
        "This demo exercises the Session OS routing/catalog layer with short local tasks. "
        "It is not a long agent-quality benchmark."
    )
    _write_json(Path(args.output_dir) / "local-session-os-demo.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a short HeliX Session OS scheduler demo.")
    parser.add_argument("--profile", default="laptop-12gb")
    parser.add_argument("--code-model", default="qwen-1.5b")
    parser.add_argument("--writer-model", default="gpt2-fast")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--prompt-tokens", type=int, default=192)
    parser.add_argument("--codec", default="rust-hlx-buffered-flat", choices=["rust-hlx", "rust-hlx-buffered", "rust-hlx-buffered-flat", "python-npz"])
    parser.add_argument("--audit-policy", default="deferred", choices=["blocking", "deferred"])
    parser.add_argument("--output-dir", default="verification")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_session_os_demo(args)
    print(json.dumps(_json_ready(payload), indent=2))


if __name__ == "__main__":
    main()
