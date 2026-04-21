from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


INTEGRITY_EXCLUDED_TOP_LEVEL_KEYS = [
    "artifact_sha256",
    "artifact_payload_sha256",
    "artifact_hash_kind",
    "integrity",
]


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def artifact_payload_sha256(artifact: dict[str, Any]) -> str:
    """Hash the artifact payload while excluding self-referential integrity fields."""

    payload = copy.deepcopy(artifact)
    for key in INTEGRITY_EXCLUDED_TOP_LEVEL_KEYS:
        payload.pop(key, None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def finalize_artifact(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    """Write an artifact with a stable payload hash and explicit self-hash policy.

    The final file hash belongs in the run manifest. Embedding a final SHA256 in
    the same JSON file is self-referential and will be stale after the file is
    rewritten.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    artifact["artifact_path"] = str(path)
    payload_hash = artifact_payload_sha256(artifact)
    artifact["artifact_payload_sha256"] = payload_hash
    artifact["artifact_sha256"] = payload_hash
    artifact["artifact_hash_kind"] = "canonical_payload_excluding_integrity"
    artifact["integrity"] = {
        "self_hash_policy": "external_manifest_only",
        "artifact_payload_sha256": payload_hash,
        "artifact_payload_hash_excludes": INTEGRITY_EXCLUDED_TOP_LEVEL_KEYS,
        "final_file_sha256_source": "run_manifest",
        "note": (
            "The run manifest records the final file SHA256. The artifact "
            "records a canonical payload hash excluding self-referential "
            "integrity fields."
        ),
    }
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    return artifact
