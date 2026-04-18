from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from helix_proto.workspace import slugify, workspace_root


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def agents_dir(root: str | Path | None = None) -> Path:
    path = workspace_root(root) / "agents"
    path.mkdir(parents=True, exist_ok=True)
    return path


def agent_dir(agent_name: str, root: str | Path | None = None) -> Path:
    path = agents_dir(root) / slugify(agent_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def knowledge_store_path(agent_name: str, root: str | Path | None = None) -> Path:
    return agent_dir(agent_name, root) / "knowledge.jsonl"


def memory_store_path(agent_name: str, root: str | Path | None = None) -> Path:
    return agent_dir(agent_name, root) / "memory.jsonl"


def runs_dir(agent_name: str, root: str | Path | None = None) -> Path:
    path = agent_dir(agent_name, root) / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(agent_name: str, run_id: str, root: str | Path | None = None) -> Path:
    path = runs_dir(agent_name, root) / slugify(run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunk_text(text: str, *, max_words: int = 120, overlap_words: int = 24) -> list[str]:
    words = text.split()
    if not words:
        return []
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words must be >= 0")

    chunks: list[str] = []
    step = max(1, max_words - overlap_words)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + max_words]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + max_words >= len(words):
            break
    return chunks


def add_knowledge_text(
    agent_name: str,
    text: str,
    *,
    source: str,
    root: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    max_words: int = 120,
    overlap_words: int = 24,
) -> dict[str, Any]:
    chunks = chunk_text(text, max_words=max_words, overlap_words=overlap_words)
    path = knowledge_store_path(agent_name, root)
    created_at = _utc_now_iso()
    rows = []
    for index, chunk in enumerate(chunks):
        row = {
            "id": f"{slugify(source)}-{int(datetime.now(timezone.utc).timestamp())}-{index}",
            "source": source,
            "text": chunk,
            "metadata": metadata or {},
            "chunk_index": index,
            "created_at_utc": created_at,
        }
        _append_jsonl(path, row)
        rows.append(row)
    return {
        "agent_name": agent_name,
        "source": source,
        "chunks_added": len(rows),
        "knowledge_path": str(path),
    }


def _read_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF ingestion needs optional dependency: pip install pypdf"
            ) from exc
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix in {".json"}:
        return json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2)
    return path.read_text(encoding="utf-8", errors="ignore")


def add_knowledge_file(
    agent_name: str,
    file_path: str | Path,
    *,
    root: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    max_words: int = 120,
    overlap_words: int = 24,
) -> dict[str, Any]:
    path = Path(file_path).resolve()
    text = _read_text_file(path)
    info = add_knowledge_text(
        agent_name,
        text,
        source=str(path),
        root=root,
        metadata=metadata,
        max_words=max_words,
        overlap_words=overlap_words,
    )
    info["file_path"] = str(path)
    return info


def _score_rows(rows: list[dict[str, Any]], query: str, *, top_k: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    doc_tokens = [Counter(_tokenize(row.get("text", ""))) for row in rows]
    doc_freq: Counter[str] = Counter()
    for tokens in doc_tokens:
        for token in tokens:
            doc_freq[token] += 1

    total_docs = len(rows)
    scored: list[dict[str, Any]] = []
    query_counts = Counter(query_tokens)
    for row, tokens in zip(rows, doc_tokens, strict=True):
        if not tokens:
            continue
        score = 0.0
        for token, q_count in query_counts.items():
            tf = tokens.get(token, 0)
            if tf <= 0:
                continue
            idf = math.log((1 + total_docs) / (1 + doc_freq[token])) + 1.0
            score += q_count * tf * idf
        if score <= 0:
            continue
        length_penalty = math.sqrt(sum(tokens.values()))
        scored.append(
            {
                **row,
                "score": score / max(1.0, length_penalty),
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def search_knowledge(
    agent_name: str,
    query: str,
    *,
    top_k: int = 5,
    root: str | Path | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(knowledge_store_path(agent_name, root))
    return {
        "query": query,
        "top_k": top_k,
        "results": _score_rows(rows, query, top_k=top_k),
    }


def append_memory_event(
    agent_name: str,
    *,
    kind: str,
    text: str,
    root: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "id": f"{slugify(kind)}-{int(datetime.now(timezone.utc).timestamp())}",
        "kind": kind,
        "text": text,
        "metadata": metadata or {},
        "created_at_utc": _utc_now_iso(),
    }
    path = memory_store_path(agent_name, root)
    _append_jsonl(path, row)
    return row


def search_memory(
    agent_name: str,
    query: str,
    *,
    top_k: int = 5,
    root: str | Path | None = None,
    exclude_run_id: str | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(memory_store_path(agent_name, root))
    if exclude_run_id is not None:
        rows = [row for row in rows if row.get("metadata", {}).get("run_id") != exclude_run_id]
    return {
        "query": query,
        "top_k": top_k,
        "results": _score_rows(rows, query, top_k=top_k),
    }


def save_run_trace(
    agent_name: str,
    run_id: str,
    trace: dict[str, Any],
    *,
    root: str | Path | None = None,
) -> Path:
    path = run_dir(agent_name, run_id, root) / "trace.json"
    path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return path
