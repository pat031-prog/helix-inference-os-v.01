from __future__ import annotations

import argparse
import json
import tempfile
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helix_kv.session_os import PrefixResolver, SessionCatalog
from tools.run_local_hybrid_stress import _json_ready, _write_json


def run_catalog_smoke(args: argparse.Namespace) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="helix-session-catalog-smoke-") as temp:
        root = Path(temp)
        catalog = SessionCatalog.open(root / "catalog.sqlite")
        session_dir = root / "sessions" / "gpt2-agent" / "v0001"
        session_dir.mkdir(parents=True)
        (session_dir / "session.json").write_text('{"session_token_ids":[1,2,3,4]}', encoding="utf-8")
        session = catalog.record_session(
            session_id="gpt2-agent-v0001",
            model_id="gpt2",
            agent_id="agent-a",
            model_ref="gpt2",
            arch="transformer",
            path=session_dir,
            token_ids=[1, 2, 3, 4],
            session_bytes=128,
            codec="rust-hlx-buffered-flat",
            audit_status="verified",
            session_hash="demo",
            merkle_root="demo-root",
        )
        latest = catalog.find_latest("gpt2", "agent-a")
        prefix = PrefixResolver(catalog).find_best_prefix(
            model_id="gpt2",
            agent_id="agent-a",
            token_ids=[1, 2, 3, 9],
            arch="transformer",
        )
        traversal_rejected = False
        try:
            catalog.record_session(
                session_id="bad",
                model_id="gpt2",
                agent_id="agent-a",
                model_ref="gpt2",
                arch="transformer",
                path="../bad",
            )
        except ValueError:
            traversal_rejected = True
        payload = {
            "title": "HeliX Session Catalog Smoke",
            "benchmark_kind": "session-os-catalog-smoke-v0",
            "status": "completed",
            "session_recorded": session.to_dict(),
            "latest_session_id": None if latest is None else latest.session_id,
            "prefix_match": prefix.to_dict(),
            "traversal_rejected": traversal_rejected,
            "catalog_stats": catalog.stats(),
        }
        catalog.close()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "local-session-catalog-smoke.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a lightweight HeliX SessionCatalog smoke artifact.")
    parser.add_argument("--output-dir", default="verification")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_catalog_smoke(args)
    print(json.dumps(_json_ready(payload), indent=2))


if __name__ == "__main__":
    main()
