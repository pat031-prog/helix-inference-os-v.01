from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from helix_kv.memory_catalog import MemoryCatalog
from helix_proto.memory import search_knowledge, search_memory as search_legacy_memory
from helix_proto.workspace import workspace_root


DEFAULT_PROJECT = "helix"
CompressFn = Callable[[str], str | dict[str, Any] | None]


def catalog_path(root: str | Path | None = None) -> Path:
    return workspace_root(root) / "session-os" / "memory.sqlite"


def open_catalog(root: str | Path | None = None) -> MemoryCatalog:
    return MemoryCatalog.open(catalog_path(root))


def _json_preview(value: Any, *, limit: int = 1200) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except TypeError:
        text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _summarize(text: str, *, limit: int = 240, compress_fn: CompressFn | None = None) -> str:
    clean = " ".join(str(text or "").split())
    if compress_fn is not None and clean:
        try:
            compressed = compress_fn(clean)
            if isinstance(compressed, dict):
                candidate = str(compressed.get("summary") or compressed.get("text") or "").strip()
            else:
                candidate = str(compressed or "").strip()
            if candidate:
                clean = " ".join(candidate.split())
        except Exception:  # noqa: BLE001
            # Compression is an optional recall optimization; it must never break agent execution.
            pass
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def runtime_compress_fn(runtime: Any, alias: str | None) -> CompressFn | None:
    if not alias:
        return None

    def _compress(text: str) -> str:
        prompt = (
            "Compress this agent observation into one concise operational memory. "
            "Preserve task decisions, errors, ids, file paths, and caveats. "
            "Return only the summary.\n\n"
            f"Observation:\n{text}"
        )
        result = runtime.generate_text(
            alias=str(alias),
            prompt=prompt,
            max_new_tokens=96,
            do_sample=False,
            cache_mode="session",
        )
        return str(result.get("completion_text") or result.get("generated_text") or "").strip()

    return _compress


def build_context(
    *,
    root: str | Path | None,
    project: str = DEFAULT_PROJECT,
    agent_id: str,
    query: str | None,
    budget_tokens: int = 800,
    mode: str = "search",
    limit: int = 5,
    exclude_memory_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build a context string for the agent from stored memories.

    Args:
        exclude_memory_ids: Memory IDs to hard-filter before any ranking.
            Pass rollback_manifest["fenced_memory_ids"] here to guarantee
            that a failed attempt's observations never leak into the retry's
            context window. Exclusion is pre-ranking, at the SQL WHERE level.
    """
    catalog = open_catalog(root)
    try:
        return catalog.build_context(
            project=project,
            agent_id=agent_id,
            query=query,
            budget_tokens=budget_tokens,
            mode=mode,
            limit=limit,
            exclude_memory_ids=exclude_memory_ids,
        )
    finally:
        catalog.close()


def observe_event(
    *,
    root: str | Path | None,
    project: str = DEFAULT_PROJECT,
    agent_id: str,
    session_id: str | None,
    event_type: str,
    content: str,
    summary: str | None = None,
    tags: list[str] | None = None,
    importance: int = 5,
    promote: bool = True,
    memory_type: str = "episodic",
    compress_fn: CompressFn | None = None,
) -> dict[str, Any]:
    compact_summary = _summarize(summary or content, compress_fn=compress_fn)
    catalog = open_catalog(root)
    try:
        observation = catalog.observe(
            project=project,
            agent_id=agent_id,
            session_id=session_id,
            observation_type=event_type,
            summary=compact_summary,
            content=content,
            importance=importance,
            tags=tags or [event_type],
        )
        memory = None
        if promote:
            memory_item = catalog.remember(
                project=project,
                agent_id=agent_id,
                session_id=session_id,
                memory_type=memory_type,
                summary=compact_summary,
                content=content,
                importance=importance,
                tags=tags or [event_type],
            )
            memory = memory_item.to_dict()
            if session_id:
                catalog.link_session_memory(session_id=session_id, memory_id=memory_item.memory_id, relation="observed")
        return {"observation": observation, "memory": memory}
    finally:
        catalog.close()


def fence_memory(
    *,
    root: str | Path | None,
    project: str = DEFAULT_PROJECT,
    agent_id: str,
    session_id: str | None,
    memory_id: str,
    reason: str,
) -> dict[str, Any]:
    """Write a rollback fence marker to hmem and return fencing metadata.

    This is a first-class OS primitive for the resilient-pipeline rollback
    mechanism. It does NOT alter the fenced memory in the store — the caller
    must pass ``memory_id`` to ``build_context`` via ``exclude_memory_ids``
    to activate the hard pre-ranking filter.

    The fence marker itself is written at importance=9 so the retry model
    receives it as high-priority context::

        "The previous attempt failed. Reason: <quality_gate_report>.
         Avoid the path that produced: <memory_id>."

    This turns a failure event into an analytical advantage for the fallback.

    Args:
        memory_id: The memory_id returned by the failed observe_event call.
        reason:    Human-readable quality gate failure report.

    Returns:
        dict with ``fence_memory_id`` (the marker itself) and
        ``fenced_memory_id`` (the original bad node, for the manifest).
    """
    result = observe_event(
        root=root,
        project=project,
        agent_id=agent_id,
        session_id=session_id,
        event_type="rollback_fence",
        content=(
            f"ROLLBACK FENCE\n"
            f"Fenced memory: {memory_id}\n"
            f"Reason: {reason}\n"
            f"Action: context excluded via exclude_memory_ids hard filter.\n"
            f"Instruction for retry model: the previous output at {memory_id} "
            f"failed the quality gate. Do NOT replicate that output pattern."
        ),
        summary=f"Rollback fence — quality gate fail: {reason[:180]}",
        tags=["rollback-fence", f"fenced:{memory_id}", "quality-gate-fail", "resilient-pipeline"],
        importance=9,
        promote=True,
        memory_type="episodic",
    )
    fence_memory_id = (result.get("memory") or {}).get("memory_id")
    return {
        "fence_memory_id": fence_memory_id,
        "fenced_memory_id": memory_id,
        "reason": reason,
        "agent_id": agent_id,
    }


def observe_tool_call(
    *,
    root: str | Path | None,
    project: str = DEFAULT_PROJECT,
    agent_id: str,
    run_id: str,
    step_index: int,
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
    planner: str | None = None,
    compress_fn: CompressFn | None = None,
) -> dict[str, Any]:
    preview = _json_preview(result, limit=1800)
    content = (
        f"Run: {run_id}\n"
        f"Step: {step_index}\n"
        f"Tool: {tool_name}\n"
        f"Planner: {planner or 'unknown'}\n"
        f"Arguments: {_json_preview(arguments, limit=600)}\n"
        f"Observation: {preview}"
    )
    summary = f"{tool_name} step {step_index}: {_summarize(preview, limit=180, compress_fn=compress_fn)}"
    return observe_event(
        root=root,
        project=project,
        agent_id=agent_id,
        session_id=run_id,
        event_type="tool_call",
        content=content,
        summary=summary,
        tags=["tool_call", tool_name, f"run:{run_id}"],
        importance=6 if "error" not in str(result).lower() else 8,
        promote=True,
        memory_type="episodic",
    )


def search(
    *,
    root: str | Path | None,
    project: str = DEFAULT_PROJECT,
    agent_id: str | None,
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:
    catalog = open_catalog(root)
    try:
        hits = catalog.search(project=project, agent_id=agent_id, query=query, limit=top_k)
    finally:
        catalog.close()
    results = [
        {
            **item,
            "kind": "memory",
            "source": f"hmem:{item.get('memory_id')}",
            "text": str(item.get("content") or item.get("summary") or ""),
        }
        for item in hits
    ]
    if results:
        return {"query": query, "top_k": top_k, "results": results, "source": "hmem"}
    legacy = search_legacy_memory(agent_id or "default-agent", query, top_k=top_k, root=root)
    legacy_results = [
        {
            **item,
            "kind": "legacy-memory",
            "source": f"legacy:{item.get('id')}",
            "text": str(item.get("text") or ""),
        }
        for item in legacy.get("results", [])
    ]
    return {"query": query, "top_k": top_k, "results": legacy_results, "source": "legacy-memory"}


def hybrid_search(
    *,
    root: str | Path | None,
    project: str = DEFAULT_PROJECT,
    agent_id: str,
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:
    memory_hits = search(root=root, project=project, agent_id=agent_id, query=query, top_k=top_k).get("results", [])
    knowledge = search_knowledge(agent_id, query, top_k=top_k, root=root).get("results", [])
    knowledge_hits = [
        {
            **item,
            "kind": "knowledge",
            "source": item.get("source", "knowledge"),
            "text": str(item.get("text") or ""),
        }
        for item in knowledge
    ]
    combined = [*memory_hits, *knowledge_hits]
    combined.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return {
        "query": query,
        "top_k": top_k,
        "results": combined[:top_k],
        "source_counts": {
            "memory": len(memory_hits),
            "knowledge": len(knowledge_hits),
        },
    }


def graph(
    *,
    root: str | Path | None,
    project: str | None = DEFAULT_PROJECT,
    agent_id: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    catalog = open_catalog(root)
    try:
        return catalog.graph(project=project, agent_id=agent_id, limit=limit)
    finally:
        catalog.close()


def stats(*, root: str | Path | None) -> dict[str, Any]:
    catalog = open_catalog(root)
    try:
        return catalog.stats()
    finally:
        catalog.close()

