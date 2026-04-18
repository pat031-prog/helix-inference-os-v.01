from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for item in (REPO_ROOT, SRC_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from helix_kv.memory_catalog import MemoryCatalog, privacy_filter  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_smoke(output_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="helix-memory-catalog-") as temp:
        root = Path(temp)
        catalog = MemoryCatalog.open(root / "memory.sqlite")
        try:
            observation = catalog.observe(
                project="helix",
                agent_id="code_reviewer",
                session_id="session-a",
                observation_type="working",
                summary="Reviewed auth middleware and found a rate-limit seam",
                content="Use jose middleware. api_key=sk-proj-abcdefghijklmnopqrstuvwxyz should be hidden.",
                tags=["auth", "rate-limit"],
            )
            memory = catalog.remember(
                project="helix",
                agent_id="code_reviewer",
                session_id="session-a",
                memory_type="episodic",
                summary="JWT auth uses jose middleware",
                content="The code reviewer confirmed auth uses jose middleware and tests cover token validation.",
                importance=8,
                tags=["auth", "jwt", "jose"],
            )
            link = catalog.link_session_memory(session_id="session-a", memory_id=memory.memory_id)
            hits = catalog.search(project="helix", agent_id="code_reviewer", query="jose token validation", limit=3)
            context = catalog.build_context(
                project="helix",
                agent_id="code_reviewer",
                query="How is auth implemented?",
                budget_tokens=80,
                mode="search",
            )
            payload = {
                "title": "HeliX Agent Memory Catalog Smoke",
                "benchmark_kind": "session-os-agent-memory-catalog-v1",
                "status": "completed",
                "dependency_policy": "stdlib-sqlite-no-vector-db",
                "fts_enabled": catalog.fts_enabled,
                "privacy_redaction_ok": "[REDACTED_SECRET]" in observation["content"],
                "privacy_filter_example": privacy_filter("<private>secret</private> token=sk-ant-abcdefghijklmnopqrstuvwxyz"),
                "observation": observation,
                "memory": memory.to_dict(),
                "link": link,
                "search_hit_count": len(hits),
                "top_hit_memory_id": hits[0]["memory_id"] if hits else None,
                "context_tokens": context["tokens"],
                "context_memory_ids": context["memory_ids"],
                "context_preview": context["context"][:500],
                "stats": catalog.stats(),
                "claim_boundary": "This smoke validates local semantic recall plumbing; it does not benchmark embedding/vector quality.",
            }
        finally:
            catalog.close()
        output_path = output_dir / "local-agent-memory-catalog-smoke.json"
        _write_json(output_path, payload)
        shutil.rmtree(root, ignore_errors=True)
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a dependency-free HeliX MemoryCatalog smoke.")
    parser.add_argument("--output-dir", default="verification")
    args = parser.parse_args()
    payload = run_smoke(Path(args.output_dir))
    print(json.dumps({"status": payload["status"], "artifact": str(Path(args.output_dir) / "local-agent-memory-catalog-smoke.json")}, indent=2))


if __name__ == "__main__":
    main()
