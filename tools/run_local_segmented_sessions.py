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


def _write_segment(path: Path, *, token_ids: list[int], parent_session_id: str | None) -> int:
    path.mkdir(parents=True, exist_ok=True)
    (path / "session.json").write_text(
        json.dumps(
            {
                "session_token_ids": token_ids,
                "parent_session_id": parent_session_id,
                "session_codec": "rust-hlx-buffered-flat",
                "segment_metadata_only_v1": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # A tiny placeholder keeps the v1 artifact shaped like a .hlx session without pretending to benchmark I/O.
    (path / "kv_cache.hlx").write_bytes(b"HLX-SEGMENT-METADATA-V1")
    return int(sum(item.stat().st_size for item in path.iterdir() if item.is_file()))


def run_segmented_sessions(args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="helix-segmented-session-") as temp:
        root = Path(temp)
        catalog = SessionCatalog.open(root / "catalog.sqlite")
        base_tokens = list(range(128))
        delta_1 = list(range(128, 144))
        delta_2 = list(range(144, 160))
        base_path = root / "sessions" / "base"
        delta_1_path = root / "sessions" / "delta-1"
        delta_2_path = root / "sessions" / "delta-2"
        base_bytes = _write_segment(base_path, token_ids=base_tokens, parent_session_id=None)
        base = catalog.record_session(
            session_id="base-prefix",
            model_id="gpt2",
            agent_id="segment-agent",
            model_ref="gpt2",
            arch="transformer",
            path=base_path,
            token_ids=base_tokens,
            session_bytes=base_bytes,
            codec="rust-hlx-buffered-flat",
            audit_status="verified",
            session_hash=token_hash(base_tokens),
        )
        delta_1_bytes = _write_segment(delta_1_path, token_ids=base_tokens + delta_1, parent_session_id=base.session_id)
        first = catalog.record_session(
            session_id="turn-1",
            model_id="gpt2",
            agent_id="segment-agent",
            model_ref="gpt2",
            arch="transformer",
            path=delta_1_path,
            token_ids=base_tokens + delta_1,
            session_bytes=delta_1_bytes,
            codec="rust-hlx-buffered-flat",
            audit_status="verified",
            session_hash=token_hash(base_tokens + delta_1),
            parent_session_id=base.session_id,
        )
        delta_2_bytes = _write_segment(delta_2_path, token_ids=base_tokens + delta_1 + delta_2, parent_session_id=first.session_id)
        second = catalog.record_session(
            session_id="turn-2",
            model_id="gpt2",
            agent_id="segment-agent",
            model_ref="gpt2",
            arch="transformer",
            path=delta_2_path,
            token_ids=base_tokens + delta_1 + delta_2,
            session_bytes=delta_2_bytes,
            codec="rust-hlx-buffered-flat",
            audit_status="verified",
            session_hash=token_hash(base_tokens + delta_1 + delta_2),
            parent_session_id=first.session_id,
        )
        chain = catalog.parent_chain(second.session_id)
        verify_chain_status = "verified" if [item.session_id for item in chain] == ["turn-2", "turn-1", "base-prefix"] else "failed"
        full_rewrite_bytes = base_bytes + delta_1_bytes + delta_2_bytes
        metadata_delta_bytes = len(json.dumps({"delta_1": delta_1, "delta_2": delta_2}).encode("utf-8"))
        payload = {
            "title": "HeliX Segmented Sessions v1 Smoke",
            "benchmark_kind": "session-os-segmented-sessions-v1",
            "status": "completed",
            "segment_scope": "metadata_chain_v1",
            "model_id": "gpt2",
            "agent_id": "segment-agent",
            "base_session_bytes": base_bytes,
            "delta_session_bytes": delta_1_bytes + delta_2_bytes,
            "full_rewrite_bytes": full_rewrite_bytes,
            "metadata_delta_bytes": metadata_delta_bytes,
            "rewrite_avoided_bytes": max(0, full_rewrite_bytes - (base_bytes + metadata_delta_bytes)),
            "segment_chain_length": len(chain),
            "verify_chain_status": verify_chain_status,
            "sessions": [item.to_dict() for item in chain],
            "claim_boundary": "Segmented sessions v1 proves metadata parent chaining; physical token-slice storage is a later optimization.",
        }
        catalog.close()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "local-segmented-session-summary.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a lightweight HeliX segmented-session metadata smoke artifact.")
    parser.add_argument("--output-dir", default="verification")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_segmented_sessions(args)
    print(json.dumps(_json_ready(payload), indent=2))


if __name__ == "__main__":
    main()
