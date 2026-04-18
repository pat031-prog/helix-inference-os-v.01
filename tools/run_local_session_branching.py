from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helix_kv.session_os import SessionCatalog, token_hash
from tools.run_local_hybrid_stress import _json_ready, _write_json


AGENTS = {
    "bug_hunter": list(range(1000, 1016)),
    "perf_engineer": list(range(2000, 2020)),
    "claims_editor": list(range(3000, 3012)),
}


def _write_segment(
    path: Path,
    *,
    token_ids: list[int],
    parent_session_id: str | None,
    segment_kind: str,
    base_token_count: int,
    delta_token_count: int,
) -> int:
    path.mkdir(parents=True, exist_ok=True)
    (path / "session.json").write_text(
        json.dumps(
            {
                "session_token_ids": token_ids,
                "parent_session_id": parent_session_id,
                "session_codec": "rust-hlx-buffered-flat",
                "segment_kind": segment_kind,
                "base_token_count": base_token_count,
                "delta_token_count": delta_token_count,
                "composed_token_count": len(token_ids),
                "branching_metadata_v1": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Metadata smoke only: physical token-slice storage remains a later optimization.
    (path / "kv_cache.hlx").write_bytes(f"HLX-BRANCH-{segment_kind}-{len(token_ids)}".encode("utf-8"))
    return int(sum(item.stat().st_size for item in path.iterdir() if item.is_file()))


def run_session_branching(args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="helix-session-branching-") as temp:
        root = Path(temp)
        catalog = SessionCatalog.open(root / "catalog.sqlite")
        base_tokens = list(range(int(args.base_tokens)))
        base_path = root / "sessions" / "base"
        base_bytes = _write_segment(
            base_path,
            token_ids=base_tokens,
            parent_session_id=None,
            segment_kind="base",
            base_token_count=len(base_tokens),
            delta_token_count=0,
        )
        base = catalog.record_session(
            session_id="shared-base",
            model_id=str(args.model_id),
            agent_id="shared",
            model_ref=str(args.model_ref),
            arch="transformer",
            path=base_path,
            token_ids=base_tokens,
            session_bytes=base_bytes,
            codec="rust-hlx-buffered-flat",
            audit_status="verified",
            session_hash=token_hash(base_tokens),
        )
        branches: list[dict[str, Any]] = []
        full_rewrite_bytes = 0
        delta_metadata_bytes = 0
        for agent_id, delta_tokens in AGENTS.items():
            composed = base_tokens + delta_tokens
            branch_path = root / "sessions" / agent_id
            branch_bytes = _write_segment(
                branch_path,
                token_ids=composed,
                parent_session_id=base.session_id,
                segment_kind="delta",
                base_token_count=len(base_tokens),
                delta_token_count=len(delta_tokens),
            )
            branch = catalog.record_session(
                session_id=f"{agent_id}-delta",
                model_id=str(args.model_id),
                agent_id=agent_id,
                model_ref=str(args.model_ref),
                arch="transformer",
                path=branch_path,
                token_ids=composed,
                session_bytes=branch_bytes,
                codec="rust-hlx-buffered-flat",
                audit_status="verified",
                session_hash=token_hash(composed),
                parent_session_id=base.session_id,
            )
            full_rewrite_bytes += branch_bytes
            delta_metadata_bytes += len(json.dumps({"agent_id": agent_id, "delta_tokens": delta_tokens}).encode("utf-8"))
            branches.append(
                {
                    "agent_id": agent_id,
                    "session_id": branch.session_id,
                    "parent_session_id": branch.parent_session_id,
                    "segment_kind": "delta",
                    "base_token_count": len(base_tokens),
                    "delta_token_count": len(delta_tokens),
                    "composed_token_count": len(composed),
                    "session_bytes": branch_bytes,
                    "verify_chain_status": "verified" if catalog.parent_chain(branch.session_id)[-1].session_id == base.session_id else "failed",
                }
            )
        shared_strategy_bytes = base_bytes + delta_metadata_bytes
        payload = {
            "title": "HeliX Session Branching Summary",
            "benchmark_kind": "session-os-branching-v1",
            "status": "completed",
            "model_id": str(args.model_id),
            "model_ref": str(args.model_ref),
            "branch_count": len(branches),
            "base_session": {
                "session_id": base.session_id,
                "segment_kind": "base",
                "base_token_count": len(base_tokens),
                "session_bytes": base_bytes,
            },
            "branches": branches,
            "full_rewrite_bytes": full_rewrite_bytes,
            "shared_strategy_bytes": shared_strategy_bytes,
            "rewrite_avoided_bytes_estimate": max(0, full_rewrite_bytes - shared_strategy_bytes),
            "verify_chain_status": "verified" if all(branch["verify_chain_status"] == "verified" for branch in branches) else "failed",
            "claim_boundary": "Branching v1 is metadata plus measured avoided rewrite estimate; physical token-slice storage is not implemented yet.",
        }
        catalog.close()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "local-session-branching-summary.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a local HeliX session branching smoke artifact.")
    parser.add_argument("--model-id", default="gpt2")
    parser.add_argument("--model-ref", default="gpt2")
    parser.add_argument("--base-tokens", type=int, default=128)
    parser.add_argument("--output-dir", default="verification")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_session_branching(args)
    print(json.dumps(_json_ready(payload), indent=2))


if __name__ == "__main__":
    main()
