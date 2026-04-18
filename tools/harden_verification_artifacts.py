from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from helix_proto.verification_hardening import patch_artifact, validate_artifact


def iter_targets(root: Path, *, scope: str) -> list[Path]:
    files = sorted(path for path in root.rglob("*.json") if path.is_file())
    if scope == "batch-20260418":
        files = [path for path in files if path.name.startswith("local-")]
    return [
        path
        for path in files
        if "pre-hardening-20260418" not in path.parts
        and not path.name.startswith("local-verification-hardening-")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch HeliX verification artifacts with claim hardening fields.")
    parser.add_argument("--verification-dir", default="verification")
    parser.add_argument("--backup", default="verification/pre-hardening-20260418")
    parser.add_argument("--scope", default="batch-20260418", choices=["all", "batch-20260418"])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    verification = Path(args.verification_dir)
    backup = Path(args.backup)
    results = []
    for path in iter_targets(verification, scope=args.scope):
        try:
            results.append(patch_artifact(path, backup_dir=backup, write=args.write))
        except Exception as exc:  # noqa: BLE001
            results.append({"path": str(path), "changed": False, "error": f"{type(exc).__name__}: {exc}"})

    remaining_issues = []
    for path in iter_targets(verification, scope=args.scope):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        remaining_issues.extend(issue.__dict__ for issue in validate_artifact(path, payload))

    manifest = {
        "artifact": "local-verification-hardening-20260418",
        "generated_ms": int(time.time() * 1000),
        "scope": args.scope,
        "write": bool(args.write),
        "backup_dir": str(backup),
        "patched_count": sum(1 for row in results if row.get("changed")),
        "results": results,
        "remaining_issue_count": len(remaining_issues),
        "remaining_issues": remaining_issues[:200],
        "claim_boundary": "Hardening patches preserve originals in backup and record hash transitions.",
    }
    out = verification / "local-verification-hardening-20260418.json"
    verification.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
