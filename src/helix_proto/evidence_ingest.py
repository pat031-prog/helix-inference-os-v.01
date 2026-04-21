from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from helix_proto import hmem
from helix_proto.artifact_replay import sha256_file, verify_artifact_file


RUN_STAMP_RE = re.compile(r"(20\d{6}-\d{6})")
DEFAULT_MAX_SCAN = 240


def _utc_sort_key(path: Path, payload: dict[str, Any] | None = None) -> str:
    payload = payload or {}
    for value in (
        payload.get("run_id"),
        payload.get("started_at_utc"),
        payload.get("completed_at_utc"),
        path.name,
        str(path),
    ):
        match = RUN_STAMP_RE.search(str(value or ""))
        if match:
            return match.group(1)
    try:
        return f"{path.stat().st_mtime_ns:020d}"
    except OSError:
        return "0"


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _iter_candidate_manifests(evidence_root: Path) -> Iterable[Path]:
    if not evidence_root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(evidence_root, onerror=lambda _error: None):
        dirnames[:] = [
            item
            for item in dirnames
            if not item.startswith("_pytest") and item not in {"__pycache__", ".git", ".venv", "node_modules"}
        ]
        base = Path(dirpath)
        for filename in filenames:
            if filename.endswith("-run.json") or filename == "public-evidence-index.json":
                yield base / filename


def _normalize_path(value: Any, *, repo_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = repo_root / path
    return path


def _compact_cases(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    cases = artifact.get("cases")
    if not isinstance(cases, list):
        return []
    compact = []
    for item in cases[:20]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "case_id": item.get("case_id"),
                "status": item.get("status"),
                "score": item.get("score"),
                "artifact_path": item.get("artifact_path"),
                "artifact_payload_sha256": item.get("artifact_payload_sha256"),
                "transcript_exports": item.get("transcript_exports"),
            }
        )
    return compact


def _collect_transcripts(artifact: dict[str, Any], *, repo_root: Path, artifact_path: Path | None) -> list[str]:
    paths: list[str] = []

    def add_exports(exports: Any) -> None:
        if not isinstance(exports, dict):
            return
        for key in ("jsonl_path", "md_path"):
            path = _normalize_path(exports.get(key), repo_root=repo_root)
            if path is not None:
                paths.append(str(path))

    add_exports(artifact.get("transcript_exports"))
    for case in artifact.get("cases") or []:
        if isinstance(case, dict):
            add_exports(case.get("transcript_exports"))

    if artifact_path and artifact_path.exists():
        try:
            for sidecar in artifact_path.parent.glob("*transcript*.jsonl"):
                paths.append(str(sidecar))
            for sidecar in artifact_path.parent.glob("*transcript*.md"):
                paths.append(str(sidecar))
        except OSError:
            pass

    deduped: list[str] = []
    seen: set[str] = set()
    for item in paths:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _query_matches(record: dict[str, Any], query: str | None) -> bool:
    if not query:
        return True
    terms = [term.lower() for term in re.findall(r"[A-Za-z0-9_\-]{3,}", str(query))]
    if not terms:
        return True
    blob = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str).lower()
    return any(term in blob for term in terms)


def _memory_id_for_record(record: dict[str, Any]) -> str:
    source = (
        record.get("artifact_file_sha256")
        or record.get("artifact_payload_sha256")
        or record.get("artifact_path")
        or record.get("manifest_path")
    )
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()
    return f"mem-evidence-{digest[:24]}"


def _record_from_manifest(manifest_path: Path, *, repo_root: Path) -> dict[str, Any] | None:
    manifest = _safe_read_json(manifest_path)
    if not manifest:
        return None

    artifact_path = _normalize_path(manifest.get("artifact_path"), repo_root=repo_root)
    artifact = _safe_read_json(artifact_path) if artifact_path and artifact_path.exists() else None
    replay_report: dict[str, Any] | None = None
    artifact_file_sha256: str | None = None
    verification_error: str | None = None

    if artifact_path and artifact_path.exists() and artifact is not None:
        try:
            replay_report = verify_artifact_file(artifact_path)
            artifact_file_sha256 = str(replay_report.get("artifact_file_sha256") or sha256_file(artifact_path))
        except Exception as exc:  # noqa: BLE001
            verification_error = str(exc)
    elif artifact_path and artifact_path.exists():
        try:
            artifact_file_sha256 = sha256_file(artifact_path)
        except OSError as exc:
            verification_error = str(exc)

    artifact_payload = artifact or {}
    suite_id = (
        artifact_payload.get("suite_id")
        or artifact_payload.get("suite")
        or manifest.get("suite_id")
        or manifest_path.parent.name
    )
    run_id = artifact_payload.get("run_id") or manifest.get("run_id") or manifest_path.stem.replace("-run", "")
    record = {
        "evidence_kind": "helix_artifact",
        "suite_id": suite_id,
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "artifact_path": str(artifact_path) if artifact_path else None,
        "status": artifact_payload.get("status") or manifest.get("status"),
        "case_count": artifact_payload.get("case_count") or manifest.get("case_count"),
        "score": artifact_payload.get("score") or manifest.get("score"),
        "artifact_payload_sha256": artifact_payload.get("artifact_payload_sha256"),
        "artifact_file_sha256": artifact_file_sha256,
        "manifest_artifact_sha256": manifest.get("artifact_sha256") or manifest.get("artifact_file_sha256"),
        "replay_status": replay_report.get("status") if replay_report else None,
        "signature_verified_count": replay_report.get("signature_verified_count") if replay_report else 0,
        "signature_failed_count": replay_report.get("signature_failed_count") if replay_report else 0,
        "chain_verified_count": replay_report.get("chain_verified_count") if replay_report else 0,
        "claim_boundaries": (replay_report or {}).get("claim_boundaries") or [],
        "verification_error": verification_error,
        "cases": _compact_cases(artifact_payload),
        "transcript_paths": _collect_transcripts(artifact_payload, repo_root=repo_root, artifact_path=artifact_path),
        "discovered_sort_key": _utc_sort_key(manifest_path, manifest),
    }
    record["memory_id"] = _memory_id_for_record(record)
    return record


def discover_evidence_records(
    *,
    repo_root: Path,
    evidence_root: Path | None = None,
    query: str | None = None,
    limit: int = 8,
    max_scan: int = DEFAULT_MAX_SCAN,
) -> list[dict[str, Any]]:
    evidence_root = evidence_root or (repo_root / "verification")
    manifests = sorted(
        _iter_candidate_manifests(evidence_root),
        key=lambda path: _utc_sort_key(path),
        reverse=True,
    )
    records: list[dict[str, Any]] = []
    for manifest_path in manifests[: max(int(max_scan), int(limit))]:
        record = _record_from_manifest(manifest_path, repo_root=repo_root)
        if not record or not _query_matches(record, query):
            continue
        records.append(record)
        if len(records) >= int(limit):
            break
    return records


@contextmanager
def _evidence_receipt_signing(record: dict[str, Any]):
    previous = {
        "HELIX_RECEIPT_SIGNING_MODE": os.environ.get("HELIX_RECEIPT_SIGNING_MODE"),
        "HELIX_RECEIPT_SIGNER_ID": os.environ.get("HELIX_RECEIPT_SIGNER_ID"),
        "HELIX_RECEIPT_SIGNING_SEED": os.environ.get("HELIX_RECEIPT_SIGNING_SEED"),
    }
    seed_source = record.get("artifact_file_sha256") or record.get("artifact_path") or record.get("manifest_path")
    os.environ["HELIX_RECEIPT_SIGNING_MODE"] = "ephemeral_preregistered"
    os.environ["HELIX_RECEIPT_SIGNER_ID"] = "helix-evidence-bridge"
    os.environ["HELIX_RECEIPT_SIGNING_SEED"] = f"helix-evidence:{record.get('memory_id')}:{seed_source}"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _summary_for_record(record: dict[str, Any]) -> str:
    artifact_hash = str(record.get("artifact_file_sha256") or record.get("artifact_payload_sha256") or "")
    short_hash = artifact_hash[:12] if artifact_hash else "no-hash"
    return (
        f"Evidence artifact {record.get('suite_id')} {record.get('run_id')}: "
        f"status={record.get('status')}, cases={record.get('case_count')}, hash={short_hash}"
    )


def ingest_evidence_records(
    *,
    root: str | Path | None,
    project: str,
    agent_id: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    catalog = hmem.open_catalog(root)
    ingested: list[dict[str, Any]] = []
    try:
        for record in records:
            memory_id = str(record["memory_id"])
            existing = catalog.get_memory(memory_id)
            if existing is None:
                content = json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2)
                with _evidence_receipt_signing(record):
                    catalog.remember(
                        project=project,
                        agent_id=agent_id,
                        session_id="helix-evidence",
                        memory_type="semantic",
                        summary=_summary_for_record(record),
                        content=content,
                        importance=9,
                        tags=[
                            "evidence",
                            "verify",
                            "artifact",
                            f"suite:{record.get('suite_id')}",
                            f"run:{record.get('run_id')}",
                        ],
                        memory_id=memory_id,
                    )
            node_hash = catalog.get_memory_node_hash(memory_id)
            receipt = catalog.get_memory_receipt(memory_id)
            chain = catalog.verify_chain(node_hash) if node_hash else {"status": "missing_node_hash"}
            enriched = {
                **record,
                "memory_id": memory_id,
                "node_hash": node_hash,
                "receipt": receipt,
                "signature_verified": bool((receipt or {}).get("signature_verified")),
                "key_provenance": (receipt or {}).get("key_provenance"),
                "chain_status": chain.get("status"),
                "chain_len": chain.get("chain_len"),
            }
            ingested.append(enriched)
    finally:
        catalog.close()
    return ingested


def ingest_artifact_file(
    *,
    root: str | Path | None,
    project: str,
    agent_id: str,
    repo_root: Path,
    artifact_path: Path,
) -> dict[str, Any]:
    artifact_path = artifact_path.resolve()
    artifact = _safe_read_json(artifact_path) or {}
    replay_report = verify_artifact_file(artifact_path)
    record = {
        "evidence_kind": "helix_artifact",
        "suite_id": artifact.get("suite_id") or artifact.get("suite") or artifact_path.parent.name,
        "run_id": artifact.get("run_id") or artifact_path.stem,
        "manifest_path": None,
        "artifact_path": str(artifact_path),
        "status": artifact.get("status"),
        "case_count": artifact.get("case_count"),
        "score": artifact.get("score"),
        "artifact_payload_sha256": artifact.get("artifact_payload_sha256"),
        "artifact_file_sha256": replay_report.get("artifact_file_sha256") or sha256_file(artifact_path),
        "manifest_artifact_sha256": None,
        "replay_status": replay_report.get("status"),
        "signature_verified_count": replay_report.get("signature_verified_count"),
        "signature_failed_count": replay_report.get("signature_failed_count"),
        "chain_verified_count": replay_report.get("chain_verified_count"),
        "claim_boundaries": replay_report.get("claim_boundaries") or [],
        "verification_error": None,
        "cases": _compact_cases(artifact),
        "transcript_paths": _collect_transcripts(artifact, repo_root=repo_root, artifact_path=artifact_path),
        "discovered_sort_key": _utc_sort_key(artifact_path, artifact),
    }
    record["memory_id"] = _memory_id_for_record(record)
    return ingest_evidence_records(root=root, project=project, agent_id=agent_id, records=[record])[0]


def refresh_evidence(
    *,
    root: str | Path | None,
    project: str,
    agent_id: str,
    repo_root: Path,
    evidence_root: Path | None = None,
    query: str | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    records = discover_evidence_records(
        repo_root=repo_root,
        evidence_root=evidence_root,
        query=query,
        limit=limit,
    )
    ingested = ingest_evidence_records(root=root, project=project, agent_id=agent_id, records=records)
    return {
        "source": "helix-evidence-bridge",
        "query": query,
        "repo_root": str(repo_root),
        "evidence_root": str(evidence_root or (repo_root / "verification")),
        "record_count": len(ingested),
        "records": ingested,
    }


def list_ingested_evidence(
    *,
    root: str | Path | None,
    project: str,
    agent_id: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    catalog = hmem.open_catalog(root)
    try:
        memories = catalog.list_memories(project=project, agent_id=agent_id, limit=max(int(limit) * 6, 40))
        evidence = [item for item in memories if "evidence" in {str(tag) for tag in item.get("tags", [])}]
        for item in evidence:
            memory_id = str(item.get("memory_id") or "")
            item["node_hash"] = catalog.get_memory_node_hash(memory_id)
            item["receipt"] = catalog.get_memory_receipt(memory_id)
            item["chain"] = catalog.verify_chain(item["node_hash"]) if item.get("node_hash") else None
        return evidence[: int(limit)]
    finally:
        catalog.close()
