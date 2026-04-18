from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from helix_proto.verification_hardening import validate_artifact


def iter_json_files(root: Path, *, scope: str) -> list[Path]:
    files = sorted(
        path
        for path in root.rglob("*.json")
        if path.is_file()
        and "pre-hardening-20260418" not in path.parts
        and not path.name.startswith("local-verification-hardening-")
    )
    if scope == "batch-20260418":
        return [path for path in files if path.name.startswith("local-")]
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Lint HeliX verification artifacts for claim safety.")
    parser.add_argument("path", nargs="?", default="verification")
    parser.add_argument("--scope", default="all", choices=["all", "batch-20260418"])
    args = parser.parse_args()

    root = Path(args.path)
    issues = []
    for path in iter_json_files(root, scope=args.scope):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:  # noqa: BLE001
            issues.append({"path": str(path), "code": "invalid_json", "message": str(exc)})
            continue
        issues.extend(issue.__dict__ for issue in validate_artifact(path, payload))

    report = {
        "artifact": "helix-claim-lint-report",
        "path": str(root),
        "scope": args.scope,
        "issue_count": len(issues),
        "issues": issues,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
