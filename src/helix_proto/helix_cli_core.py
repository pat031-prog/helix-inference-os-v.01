from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _exe_name() -> str:
    return "helix-cli-core.exe" if os.name == "nt" else "helix-cli-core"


def rust_core_binary() -> Path | None:
    configured = os.environ.get("HELIX_CLI_CORE_BIN")
    candidates = [Path(configured)] if configured else []
    root = _repo_root()
    candidates.extend(
        [
            root / "crates" / "helix-cli-core" / "target" / "release" / _exe_name(),
            root / "crates" / "helix-cli-core" / "target" / "debug" / _exe_name(),
            root / "target" / "release" / _exe_name(),
            root / "target" / "debug" / _exe_name(),
        ]
    )
    for candidate in candidates:
        try:
            if candidate and candidate.exists():
                return candidate.resolve()
        except OSError:
            continue
    return None


def rust_core_status() -> dict[str, Any]:
    binary = rust_core_binary()
    return {
        "available": binary is not None,
        "binary": str(binary) if binary else None,
        "crate": str(_repo_root() / "crates" / "helix-cli-core"),
        "build_hint": "cargo build --manifest-path crates/helix-cli-core/Cargo.toml --release",
    }


def _run_core(args: list[str], *, timeout: float = 2.0) -> dict[str, Any] | None:
    binary = rust_core_binary()
    if binary is None:
        return None
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(binary), *args],
            cwd=_repo_root(),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - caller falls back to Python.
        return {
            "status": "error",
            "source": "rust-core",
            "error": f"{type(exc).__name__}: {exc}",
            "rust_core_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        payload = {"status": "error", "error": completed.stdout[-800:]}
    if not isinstance(payload, dict):
        payload = {"status": "error", "error": "rust core returned non-object JSON"}
    payload.setdefault("source", "rust-core")
    payload.setdefault("exit_code", completed.returncode)
    payload.setdefault("rust_core_ms", round((time.perf_counter() - started) * 1000, 3))
    if completed.returncode != 0 and "stderr" not in payload:
        payload["stderr"] = completed.stderr[-1200:]
    return payload


def route(
    text: str,
    *,
    latency_mode: str = "fast",
    interaction_mode: str = "balanced",
    timeout: float = 0.25,
) -> dict[str, Any]:
    payload = _run_core(
        [
            "route",
            "--text",
            str(text or ""),
            "--latency-mode",
            latency_mode,
            "--interaction-mode",
            interaction_mode,
        ],
        timeout=timeout,
    )
    if payload and payload.get("status") == "ok":
        return payload
    fallback = _route_fallback(text, latency_mode=latency_mode, interaction_mode=interaction_mode)
    if payload:
        fallback["rust_core_error"] = payload.get("error") or payload.get("stderr")
    return fallback


def suite_index_refresh(*, evidence_root: Path, repo_root: Path, timeout: float = 10.0) -> dict[str, Any]:
    payload = _run_core(
        [
            "evidence-index",
            "--evidence-root",
            str(evidence_root),
            "--repo-root",
            str(repo_root),
        ],
        timeout=timeout,
    )
    if payload and payload.get("status") == "ok":
        return payload
    fallback = _suite_index_refresh_python(evidence_root=evidence_root, repo_root=repo_root)
    if payload:
        fallback["rust_core_error"] = payload.get("error") or payload.get("stderr")
    return fallback


def suite_list(*, evidence_root: Path, repo_root: Path, timeout: float = 1.0) -> dict[str, Any]:
    payload = _run_core(["suite-list", "--evidence-root", str(evidence_root)], timeout=timeout)
    if payload and payload.get("status") == "ok":
        return payload
    fallback = _suite_list_from_index(evidence_root=evidence_root, repo_root=repo_root)
    if payload and fallback.get("status") != "ok":
        fallback["rust_core_error"] = payload.get("error") or payload.get("stderr")
    return fallback


def suite_search(*, evidence_root: Path, repo_root: Path, query: str, limit: int = 12, timeout: float = 1.0) -> dict[str, Any]:
    payload = _run_core(
        ["suite-search", "--evidence-root", str(evidence_root), "--query", query, "--limit", str(limit)],
        timeout=timeout,
    )
    if payload and payload.get("status") == "ok":
        return payload
    fallback = _suite_search_from_index(evidence_root=evidence_root, repo_root=repo_root, query=query, limit=limit)
    if payload and fallback.get("status") != "ok":
        fallback["rust_core_error"] = payload.get("error") or payload.get("stderr")
    return fallback


def latency_report() -> dict[str, Any]:
    payload = _run_core(["latency-report"], timeout=0.25)
    return payload if payload and payload.get("status") == "ok" else {"status": "ok", "rust_core": False, **rust_core_status()}


def opencode_status(*, opencode_bin: str | None = None, timeout: float = 0.3) -> dict[str, Any]:
    args = ["opencode-status"]
    if opencode_bin:
        args.extend(["--opencode-bin", opencode_bin])
    payload = _run_core(args, timeout=timeout)
    if payload and payload.get("status") == "ok":
        return payload
    return {
        "status": "ok",
        "available": False,
        "binary": None,
        "source": "python-wrapper",
        "rust_core": rust_core_status(),
        "rust_core_error": (payload or {}).get("error") or (payload or {}).get("stderr"),
    }


def opencode_run(
    *,
    repo_root: Path,
    goal: str,
    opencode_bin: str | None = None,
    run_id: str | None = None,
    evidence_root: Path | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    args = ["opencode-run", "--repo-root", str(repo_root), "--goal", str(goal or "")]
    if opencode_bin:
        args.extend(["--opencode-bin", opencode_bin])
    if run_id:
        args.extend(["--run-id", run_id])
    if evidence_root:
        args.extend(["--evidence-root", str(evidence_root)])
    payload = _run_core(args, timeout=timeout)
    if payload:
        return payload
    return {
        "status": "error",
        "engine": "opencode",
        "error": "Rust core is not available; build it with `cargo build --manifest-path crates/helix-cli-core/Cargo.toml --release`.",
        "rust_core": rust_core_status(),
    }


def verify_capsule(*, artifact_path: Path, timeout: float = 1.0) -> dict[str, Any]:
    payload = _run_core(["verify-capsule", "--artifact", str(artifact_path)], timeout=timeout)
    if payload:
        return payload
    return _verify_capsule_python(artifact_path)


def lab_profiles(timeout: float = 0.3) -> dict[str, Any]:
    payload = _run_core(["lab-profiles"], timeout=timeout)
    if payload and payload.get("status") == "ok":
        return payload
    return {
        "status": "ok",
        "kind": "helix-lab-profiles-v1",
        "source": "python-fallback",
        "profiles": [
            {"id": "patch-safety", "default_level": "quick", "description": "Sandbox, patch hash, trust card, and apply-readiness checks."},
            {"id": "doc-grounding", "default_level": "balanced", "description": "Hard-anchor style document grounding without context bloat."},
            {"id": "memory-isolation", "default_level": "balanced", "description": "Thread memory, quarantine, and branch contamination checks."},
            {"id": "provider-audit", "default_level": "strict", "description": "Provider/model identity and disagreement checks."},
            {"id": "deep-nuclear", "default_level": "deep", "description": "Explicit full nuclear suites only."},
        ],
    }


def lab_run(*, profile: str, evidence_root: Path, repo_root: Path, timeout: float = 2.0) -> dict[str, Any]:
    payload = _run_core(
        ["lab-run", "--profile", profile, "--evidence-root", str(evidence_root), "--repo-root", str(repo_root)],
        timeout=timeout,
    )
    if payload:
        return payload
    profiles = {item["id"] for item in lab_profiles().get("profiles", []) if isinstance(item, dict)}
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "", str(profile or "patch-safety")).lower()
    if normalized not in profiles:
        return {"status": "error", "error": f"unknown lab profile: {profile}", "profile": normalized}
    if normalized == "deep-nuclear":
        return {
            "status": "requires_explicit_deep",
            "profile": normalized,
            "message": "deep-nuclear is explicit and not run by lightweight lab profiles.",
        }
    return {
        "status": "ok",
        "kind": "helix-lab-run-v1",
        "source": "python-fallback",
        "profile": normalized,
        "repo_root": str(repo_root),
        "evidence_root": str(evidence_root),
        "checks": [{"id": "rust_core", "status": "warning", "summary": "Rust core unavailable; fallback reported readiness only."}],
        "claim_boundary": "Fallback lab run is a readiness summary, not deep verification.",
    }


def opencode_install(
    *,
    dry_run: bool = False,
    global_config: bool = False,
    force: bool = False,
    opencode_bin: str | None = None,
    config_path: Path | None = None,
    timeout: float = 2.0,
) -> dict[str, Any]:
    args = ["opencode-install"]
    if dry_run:
        args.append("--dry-run")
    if global_config:
        args.append("--global")
    if force:
        args.append("--force")
    if opencode_bin:
        args.extend(["--bin", opencode_bin])
    if config_path:
        args.extend(["--config-path", str(config_path)])
    payload = _run_core(args, timeout=timeout)
    if payload:
        return payload
    return {
        "status": "error",
        "engine": "opencode",
        "error": "Rust core is not available; cannot install OpenCode MCP config.",
        "rust_core": rust_core_status(),
    }


def _verify_capsule_python(artifact_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "artifact_path": str(artifact_path)}
    if not isinstance(artifact, dict):
        return {"status": "error", "error": "artifact is not a JSON object", "artifact_path": str(artifact_path)}
    patch_info = artifact.get("patch") if isinstance(artifact.get("patch"), dict) else {}
    patch_path = Path(str(patch_info.get("path") or Path(artifact_path).with_name("patch.diff")))
    try:
        patch_bytes = patch_path.read_bytes()
    except OSError:
        patch_bytes = b""
    actual = hashlib.sha256(patch_bytes).hexdigest()
    expected = str(patch_info.get("sha256") or "")
    match = bool(expected and expected == actual)
    status = "passed" if artifact.get("status") == "passed" and match else ("partial" if match else "failed")
    return {
        "status": status,
        "kind": "helix-capsule-verification-v1",
        "source": "python-fallback",
        "run_id": artifact.get("run_id"),
        "artifact_path": str(artifact_path),
        "patch_path": str(patch_path),
        "checks": [{"id": "patch_integrity", "status": "passed" if match else "failed", "expected_sha256": expected or None, "actual_sha256": actual}],
        "verify_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def _route_fallback(text: str, *, latency_mode: str, interaction_mode: str) -> dict[str, Any]:
    started = time.perf_counter()
    lowered = str(text or "").lower()

    def has(*terms: str) -> bool:
        return any(term in lowered for term in terms)

    suite = has("/verify", "suite", "suites", "artifact", "artefacto", "manifest", "transcript", "nuclear")
    repo = has("lee repo", "leer repo", "código", "codigo", "archivo", "src/", "pytest", "cargo", "patch", "diff", "implement", "refactor", "bug")
    evidence = has("evidencia", "evidence", "audit", "audita", "verifica", "demostra", "demostrá", "certifica", "receipt", "hash", "memoria", "memory")
    agentic = has("/task", "agentic", "agent", "multi archivo", "multi-file", "hacelo", "arreglalo", "terminal", "tools")
    url = "http://" in lowered or "https://" in lowered
    helix = "helix" in lowered or "hélix" in lowered
    if suite:
        path = "nuclear"
    elif agentic or repo:
        path = "agentic"
    elif latency_mode == "deep" or url or evidence or (latency_mode == "balanced" and helix and interaction_mode == "technical"):
        path = "grounded"
    else:
        path = "lightweight"
    return {
        "status": "ok",
        "source": "python-fallback",
        "path": path,
        "model_hint": {"lightweight": "chat", "grounded": "research", "agentic": "coder", "nuclear": "suite-run-analyst"}.get(path, "chat"),
        "tools_needed": [name for name, enabled in {"suite.index": suite, "web": url, "evidence": evidence, "repo": repo}.items() if enabled],
        "deep_required": path != "lightweight",
        "signals": {"helix": helix, "suite": suite, "repo": repo, "evidence": evidence, "agentic": agentic, "url": url},
        "routing_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def _index_path(evidence_root: Path) -> Path:
    return Path(evidence_root) / ".helix-index" / "suites.json"


def _index_jsonl_path(evidence_root: Path) -> Path:
    return Path(evidence_root) / ".helix-index" / "suites.jsonl"


def _read_index(evidence_root: Path) -> dict[str, Any] | None:
    path = _index_path(evidence_root)
    jsonl_path = _index_jsonl_path(evidence_root)
    if jsonl_path.exists():
        records = []
        try:
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
        except Exception:
            return None
        meta = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                meta = loaded if isinstance(loaded, dict) else {}
            except Exception:
                meta = {}
        return {
            "version": meta.get("version") or 1,
            "source": meta.get("source") or "jsonl-index",
            "evidence_root": meta.get("evidence_root") or str(evidence_root),
            "repo_root": meta.get("repo_root"),
            "generated_utc": meta.get("generated_utc"),
            "record_count": len(records),
            "records": records,
            "records_jsonl": str(jsonl_path),
        }
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _suite_list_from_index(*, evidence_root: Path, repo_root: Path) -> dict[str, Any]:
    index = _read_index(evidence_root)
    if not index:
        return _missing_index_payload(evidence_root)
    suites: dict[str, dict[str, Any]] = {}
    for record in index.get("records") or []:
        if not isinstance(record, dict) or record.get("catalog_scope") != "suite":
            continue
        suite_id = str(record.get("suite_id") or "").strip()
        if not suite_id:
            continue
        entry = suites.setdefault(
            suite_id,
            {
                "suite_id": suite_id,
                "path": f"verification/nuclear-methodology/{suite_id}",
                "registered": False,
                "script": None,
                "description": None,
                "preregistered_path": None,
                "counts": {},
                "latest": None,
            },
        )
        counts = entry["counts"]
        kind = str(record.get("kind") or "other")
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "preregistered":
            entry["preregistered_path"] = record.get("path")
        latest = entry.get("latest")
        if not latest or int(record.get("mtime_ns") or record.get("mtime_ms") or 0) > int(latest.get("mtime_ns") or latest.get("mtime_ms") or 0):
            entry["latest"] = record
    return {
        "status": "ok",
        "source": "python-index",
        "evidence_root": str(evidence_root),
        "index_path": str(_index_path(evidence_root)),
        "index_generated_utc": index.get("generated_utc"),
        "suite_count": len(suites),
        "suites": list(sorted(suites.values(), key=lambda item: item["suite_id"])),
    }


def _suite_search_from_index(*, evidence_root: Path, repo_root: Path, query: str, limit: int) -> dict[str, Any]:
    query_l = str(query or "").lower().strip()
    if not query_l:
        return {"status": "error", "error": "query is required", "results": []}
    index = _read_index(evidence_root)
    if not index:
        return _missing_index_payload(evidence_root) | {"query": query, "results": [], "result_count": 0}
    results: list[tuple[int, dict[str, Any]]] = []
    for record in index.get("records") or []:
        if not isinstance(record, dict):
            continue
        name_l = str(record.get("name") or "").lower()
        path_l = str(record.get("path") or "").lower()
        snippet_l = str(record.get("snippet") or "").lower()
        score = 0
        if name_l == query_l:
            score += 400
        elif query_l in name_l:
            score += 300
        if query_l in path_l:
            score += 250
        if query_l in snippet_l:
            score += 180
        if record.get("catalog_scope") == "global":
            score += 30
        if score:
            results.append((score, record))
    results.sort(key=lambda item: (item[0], int(item[1].get("mtime_ns") or item[1].get("mtime_ms") or 0)), reverse=True)
    rows = [dict(row) for _score, row in results[: max(1, int(limit or 12))]]
    return {"status": "ok", "source": "python-index", "query": query, "result_count": len(rows), "results": rows}


def _suite_index_refresh_python(*, evidence_root: Path, repo_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    evidence_root = Path(evidence_root)
    repo_root = Path(repo_root)
    nuclear_root = evidence_root / "nuclear-methodology"
    base_root = nuclear_root if nuclear_root.exists() else evidence_root
    records: list[dict[str, Any]] = []
    if base_root.exists():
        for path in sorted(base_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".md", ".log", ".txt"}:
                continue
            if any(part.startswith(".") or part.startswith("_") for part in path.relative_to(base_root).parts[:-1]):
                continue
            suite_dir = _suite_dir_for_path(path, base_root)
            records.append(_record_for(path, evidence_root=evidence_root, repo_root=repo_root, suite_dir=suite_dir, catalog_scope="suite"))
    for path in sorted(evidence_root.glob("*")) if evidence_root.exists() else []:
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".md", ".log", ".txt"}:
            records.append(_record_for(path, evidence_root=evidence_root, repo_root=repo_root, suite_dir=None, catalog_scope="global"))
    index = {
        "version": 1,
        "source": "python-index",
        "evidence_root": str(evidence_root),
        "repo_root": str(repo_root),
        "generated_utc": str(int(time.time() * 1000)),
        "record_count": len(records),
        "records": records,
    }
    path = _index_path(evidence_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")
    return {
        "status": "ok",
        "source": "python-index",
        "index_path": str(path),
        "evidence_root": str(evidence_root),
        "record_count": len(records),
        "index_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def _suite_dir_for_path(path: Path, base_root: Path) -> Path | None:
    try:
        rel = path.relative_to(base_root)
    except ValueError:
        return None
    return base_root / rel.parts[0] if rel.parts else None


def _record_for(path: Path, *, evidence_root: Path, repo_root: Path, suite_dir: Path | None, catalog_scope: str) -> dict[str, Any]:
    stat = path.stat()
    summary = _json_summary(path)
    if suite_dir is not None:
        suite_id = suite_dir.name
        try:
            rel_case = path.parent.relative_to(suite_dir)
            case_id = None if str(rel_case) == "." else str(rel_case).replace("\\", "/")
        except ValueError:
            case_id = None
    else:
        suite_id = summary.get("suite_id")
        try:
            rel_case = path.parent.relative_to(evidence_root)
            case_id = None if str(rel_case) == "." else str(rel_case).replace("\\", "/")
        except ValueError:
            case_id = None
    return {
        "suite_id": suite_id,
        "case_id": summary.get("case_id") or case_id,
        "catalog_scope": catalog_scope,
        "kind": _kind_for(path),
        "path": _rel(path, repo_root),
        "name": path.name,
        "bytes": stat.st_size,
        "updated_utc": str(int(stat.st_mtime * 1000)),
        "mtime_ns": stat.st_mtime_ns,
        "run_id": summary.get("run_id") or _timestamp_from_name(path),
        "status": summary.get("status"),
        "score": summary.get("score"),
        "case_count": summary.get("case_count"),
        "artifact_payload_sha256": summary.get("artifact_payload_sha256"),
        "transcript_exports": summary.get("transcript_exports"),
        "snippet": _snippet(path, summary),
    }


def _json_summary(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".json" or path.stat().st_size > 2_000_000:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _kind_for(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name == "preregistered.md":
        return "preregistered"
    if suffix == ".log":
        return "log"
    if suffix == ".jsonl":
        return "transcript_jsonl" if "transcript" in name else "jsonl"
    if suffix == ".md":
        return "transcript_md" if "transcript" in name else "markdown"
    if suffix == ".json":
        if name.endswith("-run.json"):
            return "manifest"
        if "integrity-correction" in name:
            return "integrity_correction"
        return "artifact"
    return "other"


def _snippet(path: Path, summary: dict[str, Any]) -> str:
    parts = []
    for key in ("run_id", "case_id", "status", "summary", "artifact_payload_sha256"):
        if summary.get(key) is not None:
            parts.append(f"{key}={summary.get(key)}")
    if parts:
        return " ".join(parts)[:900]
    if path.stat().st_size > 200_000:
        return ""
    try:
        return " ".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[:8])[:900]
    except Exception:
        return ""


def _timestamp_from_name(path: Path) -> str | None:
    match = re.search(r"(20\d{6}[-_]\d{6}|20\d{6}[-_]\d{2})", path.name)
    return match.group(1).replace("_", "-") if match else None


def _rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def _missing_index_payload(evidence_root: Path) -> dict[str, Any]:
    path = _index_path(evidence_root)
    return {
        "status": "index_missing",
        "source": "index-only",
        "evidence_root": str(evidence_root),
        "index_path": str(path),
        "warning": f"suite index missing; run `/suite index refresh` ({path})",
        "records_jsonl": str(_index_jsonl_path(evidence_root)),
        "suite_count": 0,
        "suites": [],
    }
