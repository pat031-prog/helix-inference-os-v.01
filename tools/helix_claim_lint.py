from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from helix_proto.verification_hardening import validate_artifact


PUBLIC_ENTRY_FIELDS = (
    "claim",
    "tier",
    "status",
    "canonical_entry",
    "source_artifact",
    "sha256",
    "falsifier",
    "threat_model_scope",
    "public_note",
)


def iter_json_files(root: Path, *, scope: str) -> list[Path]:
    files = sorted(
        path
        for path in root.rglob("*.json")
        if path.is_file()
        and "pre-hardening-20260418" not in path.parts
        and not any(part.startswith("_pytest") for part in path.parts)
        and not path.name.startswith("local-verification-hardening-")
    )
    if scope == "batch-20260418":
        return [path for path in files if path.name.startswith("local-")]
    return files


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_public_evidence_index(index_path: Path) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # noqa: BLE001
        return [{"path": str(index_path), "code": "invalid_public_index", "message": str(exc)}]

    buckets = payload.get("buckets")
    if not isinstance(buckets, list):
        return [{"path": str(index_path), "code": "missing_buckets", "message": "public evidence index must contain a buckets array"}]

    for bucket in buckets:
        bucket_name = str(bucket.get("bucket") or "")
        entries = bucket.get("entries") or []
        if not isinstance(entries, list):
            issues.append(
                {
                    "path": str(index_path),
                    "code": "invalid_bucket_entries",
                    "message": f"bucket {bucket_name or '<unknown>'} must contain a list of entries",
                }
            )
            continue
        for entry in entries:
            claim = str(entry.get("claim") or "<unknown>")
            for field in PUBLIC_ENTRY_FIELDS:
                if not str(entry.get(field) or "").strip():
                    issues.append(
                        {
                            "path": str(index_path),
                            "code": "missing_public_field",
                            "message": f"{claim}: missing required field {field}",
                        }
                    )
            canonical_entry = REPO_ROOT / str(entry.get("canonical_entry") or "")
            if not canonical_entry.is_file():
                issues.append(
                    {
                        "path": str(index_path),
                        "code": "missing_canonical_entry",
                        "message": f"{claim}: canonical entry does not exist at {canonical_entry}",
                    }
                )
            source_artifact = REPO_ROOT / str(entry.get("source_artifact") or "")
            if not source_artifact.is_file():
                issues.append(
                    {
                        "path": str(index_path),
                        "code": "missing_source_artifact",
                        "message": f"{claim}: source artifact does not exist at {source_artifact}",
                    }
                )
                continue
            expected_hash = str(entry.get("sha256") or "").strip().lower()
            actual_hash = _sha256(source_artifact)
            if expected_hash != actual_hash:
                issues.append(
                    {
                        "path": str(index_path),
                        "code": "public_hash_mismatch",
                        "message": f"{claim}: expected sha256 {expected_hash}, got {actual_hash}",
                    }
                )
    return issues


def validate_public_docs(root: Path) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    required_files = (
        root / "README.md",
        root / "CLAIMS.md",
        root / "THREAT_MODEL.md",
        root / "NULL_RESULTS.md",
        root / "REPRODUCING.md",
        root / "evidence" / "README.md",
        root / "verification" / "README-reviewer.md",
    )
    for path in required_files:
        if not path.is_file():
            issues.append({"path": str(path), "code": "missing_public_doc", "message": "required public document is missing"})

    if issues:
        return issues

    readme = (root / "README.md").read_text(encoding="utf-8-sig")
    claims = (root / "CLAIMS.md").read_text(encoding="utf-8-sig")
    threat_model = (root / "THREAT_MODEL.md").read_text(encoding="utf-8-sig")
    null_results = (root / "NULL_RESULTS.md").read_text(encoding="utf-8-sig")
    reproducing = (root / "REPRODUCING.md").read_text(encoding="utf-8-sig")
    reviewer = (root / "verification" / "README-reviewer.md").read_text(encoding="utf-8-sig")

    required_readme_markers = (
        "deterministic evidence cage",
        "CLAIMS.md",
        "evidence/",
    )
    for marker in required_readme_markers:
        if marker not in readme:
            issues.append({"path": str(root / "README.md"), "code": "missing_readme_marker", "message": f"README missing marker: {marker}"})

    if "Provider-returned model mismatch is auditable" not in claims:
        issues.append({"path": str(root / "CLAIMS.md"), "code": "missing_claim_marker", "message": "CLAIMS.md is missing the provider audit anchor claim"})
    if "Out Of Scope" not in threat_model:
        issues.append({"path": str(root / "THREAT_MODEL.md"), "code": "missing_threat_model_section", "message": "THREAT_MODEL.md is missing Out Of Scope"})
    if "Active Memory AB current local run" not in null_results:
        issues.append({"path": str(root / "NULL_RESULTS.md"), "code": "missing_null_result", "message": "NULL_RESULTS.md is missing the Active Memory AB null result"})
    if "36 passed, 2 failed, 2 skipped" not in reproducing:
        issues.append({"path": str(root / "REPRODUCING.md"), "code": "missing_repro_snapshot", "message": "REPRODUCING.md is missing the current public-claim batch snapshot"})
    if "raw lab/archive tree" not in reviewer:
        issues.append({"path": str(root / "verification/README-reviewer.md"), "code": "missing_archive_boundary", "message": "verification reviewer notes must describe verification/ as the raw lab/archive tree"})

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Lint HeliX verification artifacts and curated public evidence.")
    parser.add_argument("path", nargs="?")
    parser.add_argument("--scope", default="all", choices=["all", "batch-20260418"])
    parser.add_argument("--public-evidence")
    parser.add_argument("--public-docs")
    args = parser.parse_args()

    issues: list[dict[str, Any]] = []
    lint_artifacts = bool(args.path) or (not args.public_evidence and not args.public_docs)
    if lint_artifacts:
        root = Path(args.path or "verification")
        for path in iter_json_files(root, scope=args.scope):
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception as exc:  # noqa: BLE001
                issues.append({"path": str(path), "code": "invalid_json", "message": str(exc)})
                continue
            issues.extend(issue.__dict__ for issue in validate_artifact(path, payload))
    else:
        root = None

    if args.public_evidence:
        issues.extend(validate_public_evidence_index(Path(args.public_evidence)))
    if args.public_docs:
        issues.extend(validate_public_docs(Path(args.public_docs)))

    report = {
        "artifact": "helix-claim-lint-report",
        "path": str(root) if root is not None else None,
        "scope": args.scope,
        "public_evidence": args.public_evidence,
        "public_docs": args.public_docs,
        "issue_count": len(issues),
        "issues": issues,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
