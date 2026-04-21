from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_DIR = Path("verification/nuclear-methodology/helix-freeform-drift")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _manifest_for(artifact: dict[str, Any], root: Path) -> Path | None:
    run_id = str(artifact.get("run_id") or "")
    suffix = run_id.rsplit("-", 2)
    if len(suffix) < 3:
        return None
    stamp = f"{suffix[-2]}-{suffix[-1]}"
    candidate = root / f"local-helix-freeform-drift-{stamp}-run.json"
    return candidate if candidate.exists() else None


def create_correction(artifact_path: Path) -> Path:
    artifact = _read_json(artifact_path)
    manifest_path = _manifest_for(artifact, artifact_path.parent)
    manifest = _read_json(manifest_path) if manifest_path else {}
    actual_hash = _sha256_path(artifact_path)
    embedded_hash = artifact.get("artifact_sha256")
    manifest_hash = manifest.get("artifact_sha256")
    correction_path = artifact_path.with_name(f"{artifact_path.stem}-integrity-correction.json")
    correction = {
        "artifact": "helix-integrity-correction-v1",
        "target_artifact_path": str(artifact_path),
        "target_run_id": artifact.get("run_id"),
        "target_scenario": artifact.get("scenario"),
        "status": "completed",
        "correction_type": "stale_embedded_artifact_sha256",
        "evidence_content_rewritten": False,
        "stale_embedded_artifact_sha256": embedded_hash,
        "actual_final_file_sha256": actual_hash,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_artifact_sha256": manifest_hash,
        "manifest_matches_actual_file": manifest_hash == actual_hash,
        "transcript_markdown_sha256": manifest.get("transcript_markdown_sha256"),
        "transcript_jsonl_sha256": manifest.get("transcript_jsonl_sha256"),
        "authoritative_hash_source": "run_manifest",
        "explanation": (
            "The target artifact was written, hashed, then rewritten after adding "
            "transcript paths. The run manifest records the final file hash. This "
            "sidecar preserves historical evidence and does not modify the target artifact."
        ),
    }
    _write_json(correction_path, correction)
    return correction_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create sidecar integrity corrections for stale HeliX artifacts")
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    args = parser.parse_args(argv)
    root = Path(args.dir)
    paths = sorted(root.glob("local-helix-freeform-drift-helix-freeform-*.json"))
    paths = [path for path in paths if not path.name.endswith("-integrity-correction.json")]
    for path in paths:
        print(create_correction(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
