"""
HeliX Cloud Amnesia Derby.

A frontier-scale remote model is still stateless unless the caller gives it
operational memory. This test compares the same cloud LLM on the same private
tasks with:

  - memory_off: no HeliX context.
  - memory_on: HeliX.search() -> <helix-active-context> -> LLM.generate().

Synthetic mode is the default and costs nothing. Real mode is opt-in through
tools/run_deepinfra_workloads_secure.ps1 so the token is entered hidden and never
written to disk.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from helix_kv.ipc_state_server import StateClient


REPO = Path(__file__).resolve().parents[1]
RUST_BIN = (
    REPO
    / "crates"
    / "helix-state-server"
    / "target"
    / "x86_64-pc-windows-gnullvm"
    / "release"
    / "helix-state-server.exe"
)
ARTIFACT = REPO / "verification" / "local-cloud-amnesia-derby.json"

DEEPINFRA_TOKEN = os.environ.get("DEEPINFRA_API_TOKEN", "")
DEEPINFRA_BASE = "https://api.deepinfra.com/v1/openai"
DERBY_MODEL = os.environ.get("HELIX_DERBY_MODEL", "Qwen/Qwen3.5-122B-A10B")
DERBY_MAX_TOKENS = int(os.environ.get("HELIX_DERBY_MAX_TOKENS", "512"))
DERBY_REQUIRE_QUALITY = os.environ.get("HELIX_DERBY_REQUIRE_QUALITY", "0") == "1"
DERBY_DISABLE_THINKING = os.environ.get("HELIX_DERBY_DISABLE_THINKING", "1") != "0"
CONTEXT_LIMIT = int(os.environ.get("HELIX_CONTEXT_LIMIT", "6"))
CONTEXT_MAX_CHARS = int(os.environ.get("HELIX_CONTEXT_MAX_CHARS", "1600"))

SECRET_PATTERN = re.compile(
    r"(sk-proj-[A-Za-z0-9_\-]{12,}|gh[pus]_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|xoxb-[A-Za-z0-9\-]{20,}|"
    r"AKIA[0-9A-Z]{12,}|Bearer\s+[A-Za-z0-9._\-+/=]{20,}|"
    r"api_key\s*=\s*[A-Za-z0-9_\-]{12,})",
    re.IGNORECASE,
)

pytestmark = pytest.mark.skipif(
    not RUST_BIN.exists(),
    reason=f"Rust binary not found: {RUST_BIN}",
)


@dataclass(frozen=True)
class DerbyTrial:
    trial_id: str
    question: str
    retrieval_query: str
    required_memory_ids: tuple[str, ...]
    required_evidence_terms: tuple[str, ...]
    action_terms: tuple[str, ...]
    preferred_memory_id: str | None = None
    deprecated_memory_ids: tuple[str, ...] = ()


@dataclass
class LLMResult:
    text: str
    synthetic: bool
    latency_ms: float
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    retry_count: int = 0
    empty_output: bool = False
    finish_reason: str = ""
    raw_message_keys: tuple[str, ...] = ()
    reasoning_chars: int = 0
    thinking_disabled_requested: bool = False
    thinking_payload_rejected: bool = False


SEED_MEMORIES: list[dict[str, Any]] = [
    {
        "memory_id": "inc-auth-001",
        "summary": "inc-auth-001: root cause was JWT kid rotation mismatch after signing-key rollout.",
        "content": "Private incident memory: auth outage started after signing-key rollout; JWT kid mismatch in validators.",
        "index_content": "auth outage jwt kid rotation mismatch signing key validators 401",
        "project": "cloud-derby",
        "agent_id": "incident-archivist",
        "importance": 10,
    },
    {
        "memory_id": "inc-auth-002",
        "summary": "inc-auth-002: first action was rollback rotation, pin kid map, then verify token validation.",
        "content": "Mitigation memory: rollback key rotation, pin kid map, verify token validation against canary traffic.",
        "index_content": "rollback key rotation pin kid map verify token validation canary",
        "project": "cloud-derby",
        "agent_id": "incident-archivist",
        "importance": 10,
    },
    {
        "memory_id": "sec-filter-001",
        "summary": "sec-filter-001: debug logger captured [REDACTED_SECRET] before privacy filter.",
        "content": "Private safety memory: credential passed through debug logger before privacy filter and DAG indexing.",
        "index_content": "credential safety debug logger before privacy filter dag indexing redacted secret",
        "project": "cloud-derby",
        "agent_id": "security-archivist",
        "importance": 10,
    },
    {
        "memory_id": "sec-filter-002",
        "summary": "sec-filter-002: patch must run privacy filter before logging, indexing and DAG insertion.",
        "content": "Patch memory: move privacy filter to the first boundary before logs, index_content and insertion.",
        "index_content": "patch privacy filter before logging indexing dag insertion boundary",
        "project": "cloud-derby",
        "agent_id": "security-archivist",
        "importance": 10,
    },
    {
        "memory_id": "supply-001",
        "summary": "supply-001: unsigned model artifact changed checksum during mirror sync.",
        "content": "Supply-chain memory: model artifact checksum drifted during mirror sync; artifact was unsigned.",
        "index_content": "supply chain unsigned model artifact checksum mirror sync drift",
        "project": "cloud-derby",
        "agent_id": "release-archivist",
        "importance": 9,
    },
    {
        "memory_id": "supply-002",
        "summary": "supply-002: block deploy unless sha256 manifest and Merkle receipt match.",
        "content": "Release gate memory: require sha256 manifest and Merkle receipt before production deploy.",
        "index_content": "release gate sha256 manifest merkle receipt block deploy mismatch",
        "project": "cloud-derby",
        "agent_id": "release-archivist",
        "importance": 9,
    },
    {
        "memory_id": "decision-001",
        "summary": "decision-001: release captain froze Project Helio at ring-1 until audit verified.",
        "content": "Decision memory: release captain decided to freeze Project Helio at ring-1 pending audit verification.",
        "index_content": "release captain decision freeze Project Helio ring-1 audit verified",
        "project": "cloud-derby",
        "agent_id": "release-captain",
        "importance": 10,
    },
    {
        "memory_id": "decision-002",
        "summary": "decision-002: exception allowed docs-only deploy, but binaries remain frozen.",
        "content": "Decision memory: docs-only deploy is allowed; binary deploy remains frozen until verification.",
        "index_content": "docs-only deploy allowed binary deploy frozen verification exception",
        "project": "cloud-derby",
        "agent_id": "release-captain",
        "importance": 9,
    },
    {
        "memory_id": "signal-auth-001",
        "summary": "signal-auth-001: customer tickets mention login loops after midnight UTC.",
        "content": "Signal: login loop complaints spiked after midnight UTC on enterprise tenants.",
        "index_content": "login loops midnight utc enterprise tenants auth signal",
        "project": "cloud-derby",
        "agent_id": "signal-collector",
        "importance": 7,
    },
    {
        "memory_id": "signal-auth-002",
        "summary": "signal-auth-002: metrics show validator cache misses increased 9x.",
        "content": "Signal: validator cache misses increased ninefold after the rollout window.",
        "index_content": "validator cache misses 9x rollout window auth signal",
        "project": "cloud-derby",
        "agent_id": "signal-collector",
        "importance": 7,
    },
    {
        "memory_id": "signal-auth-003",
        "summary": "signal-auth-003: canary with pinned kid map stopped login loops.",
        "content": "Signal: canary using pinned kid map stopped login loops within six minutes.",
        "index_content": "canary pinned kid map stopped login loops six minutes",
        "project": "cloud-derby",
        "agent_id": "signal-collector",
        "importance": 8,
    },
    {
        "memory_id": "runbook-old-001",
        "summary": "runbook-old-001: obsolete runbook says route Orion traffic to blue cluster.",
        "content": "Deprecated runbook: Orion failover should route traffic to blue cluster. superseded_by=runbook-new-001.",
        "index_content": "obsolete deprecated Orion failover blue cluster superseded_by runbook-new-001",
        "project": "cloud-derby",
        "agent_id": "sre-archivist",
        "importance": 4,
    },
    {
        "memory_id": "runbook-new-001",
        "summary": "runbook-new-001: current runbook supersedes old; route Orion to amber cluster.",
        "content": "Current runbook: supersedes=runbook-old-001; Orion failover must route to amber cluster.",
        "index_content": "current runbook supersedes runbook-old-001 Orion failover amber cluster",
        "project": "cloud-derby",
        "agent_id": "sre-archivist",
        "importance": 10,
    },
    {
        "memory_id": "runbook-new-002",
        "summary": "runbook-new-002: verify amber health probe quorum before failover.",
        "content": "Current runbook action: verify amber health probe quorum, then switch traffic.",
        "index_content": "amber health probe quorum verify switch traffic failover",
        "project": "cloud-derby",
        "agent_id": "sre-archivist",
        "importance": 9,
    },
    {
        "memory_id": "noise-ui-001",
        "summary": "noise-ui-001: landing page copy prefers warm editorial tone.",
        "content": "Noise memory about UI copy.",
        "index_content": "landing page copy warm editorial tone",
        "project": "cloud-derby",
        "agent_id": "copywriter",
        "importance": 3,
    },
    {
        "memory_id": "noise-bench-001",
        "summary": "noise-bench-001: synthetic benchmark generated 100k nodes.",
        "content": "Noise benchmark memory unrelated to cloud amnesia.",
        "index_content": "synthetic benchmark 100k nodes",
        "project": "cloud-derby",
        "agent_id": "bench-agent",
        "importance": 3,
    },
    {
        "memory_id": "noise-os-001",
        "summary": "noise-os-001: restart policy keeps three snapshots.",
        "content": "Noise ops memory.",
        "index_content": "restart policy three snapshots",
        "project": "cloud-derby",
        "agent_id": "ops-agent",
        "importance": 4,
    },
    {
        "memory_id": "noise-research-001",
        "summary": "noise-research-001: prompt caching requires exact prefix.",
        "content": "Research memory unrelated to the derby.",
        "index_content": "prompt caching exact prefix",
        "project": "cloud-derby",
        "agent_id": "researcher",
        "importance": 4,
    },
    {
        "memory_id": "noise-hybrid-001",
        "summary": "noise-hybrid-001: Mamba partial prefix remains experimental.",
        "content": "Research noise about hybrid models.",
        "index_content": "mamba partial prefix experimental hybrid",
        "project": "cloud-derby",
        "agent_id": "researcher",
        "importance": 4,
    },
    {
        "memory_id": "noise-pr-001",
        "summary": "noise-pr-001: PR template requires claims and caveats.",
        "content": "Process memory unrelated to trial answers.",
        "index_content": "pr template claims caveats process",
        "project": "cloud-derby",
        "agent_id": "release-captain",
        "importance": 4,
    },
    {
        "memory_id": "noise-data-001",
        "summary": "noise-data-001: data export path was renamed to site-dist.",
        "content": "Noise data memory.",
        "index_content": "data export site-dist renamed",
        "project": "cloud-derby",
        "agent_id": "data-agent",
        "importance": 3,
    },
    {
        "memory_id": "noise-cost-001",
        "summary": "noise-cost-001: cost dashboard groups by project.",
        "content": "Noise cost memory.",
        "index_content": "cost dashboard project grouping",
        "project": "cloud-derby",
        "agent_id": "finance-agent",
        "importance": 3,
    },
]

TRIALS: tuple[DerbyTrial, ...] = (
    DerbyTrial(
        trial_id="auth-root-cause",
        question="For private incident Atlas-17, what root cause and first mitigation did our team record?",
        retrieval_query="Atlas-17 auth outage jwt kid rotation mismatch rollback pin kid map",
        required_memory_ids=("inc-auth-001", "inc-auth-002"),
        required_evidence_terms=("jwt", "kid", "rotation", "mismatch"),
        action_terms=("rollback", "pin", "verify"),
    ),
    DerbyTrial(
        trial_id="privacy-patch",
        question="For private incident Vault-9, why did credential safety fail and what patch must be enforced?",
        retrieval_query="Vault-9 credential safety debug logger privacy filter before indexing logging",
        required_memory_ids=("sec-filter-001", "sec-filter-002"),
        required_evidence_terms=("debug", "logger", "privacy", "filter"),
        action_terms=("before", "logging", "indexing"),
    ),
    DerbyTrial(
        trial_id="release-gate",
        question="For private release Boreal-4, what explains the suspicious artifact and what gate blocks deploy?",
        retrieval_query="Boreal-4 unsigned model artifact checksum mirror sync sha256 merkle receipt deploy",
        required_memory_ids=("supply-001", "supply-002"),
        required_evidence_terms=("unsigned", "checksum", "mirror", "sync"),
        action_terms=("sha256", "merkle", "deploy"),
    ),
    DerbyTrial(
        trial_id="release-decision",
        question="What did the release captain decide for private Project Helio?",
        retrieval_query="Project Helio release captain decision freeze ring-1 docs-only binaries frozen audit",
        required_memory_ids=("decision-001", "decision-002"),
        required_evidence_terms=("Helio", "ring-1", "audit", "verified"),
        action_terms=("freeze", "docs-only", "binaries"),
    ),
    DerbyTrial(
        trial_id="multi-signal",
        question="What pattern did our private auth signals reveal and which mitigation connects them?",
        retrieval_query="login loops validator cache misses canary pinned kid map auth signals",
        required_memory_ids=("signal-auth-001", "signal-auth-002", "signal-auth-003"),
        required_evidence_terms=("login", "validator", "cache", "canary"),
        action_terms=("pinned", "kid", "map"),
    ),
    DerbyTrial(
        trial_id="superseded-runbook",
        question="Which Orion failover runbook should be followed now?",
        retrieval_query="Orion failover current runbook supersedes old amber cluster health probe quorum",
        required_memory_ids=("runbook-new-001", "runbook-new-002"),
        required_evidence_terms=("supersedes", "Orion", "amber", "cluster"),
        action_terms=("health", "probe", "quorum"),
        preferred_memory_id="runbook-new-001",
        deprecated_memory_ids=("runbook-old-001",),
    ),
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _start_server() -> tuple[subprocess.Popen[bytes], int]:
    port = _free_port()
    env = {**os.environ, "HELIX_STATE_HOST": "127.0.0.1", "HELIX_STATE_PORT": str(port)}
    proc = subprocess.Popen(
        [str(RUST_BIN)],
        env=env,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.05)
    else:
        proc.kill()
        raise RuntimeError(f"Server failed to start on :{port}")
    return proc, port


def _stop_server(proc: subprocess.Popen[bytes]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


async def _make_client(port: int, timeout: float = 15.0) -> StateClient:
    client = StateClient(transport="tcp", host="127.0.0.1", port=port, timeout=timeout)
    await client.connect()
    return client


async def _one_call(port: int, method: str, params: dict[str, Any], timeout: float = 15.0) -> Any:
    client = await _make_client(port, timeout=timeout)
    try:
        return await client._call(method, params)
    finally:
        await client.close()


def _p50(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _sanitize_preview(text: str, limit: int = 240) -> str:
    clean = SECRET_PATTERN.sub("[REDACTED_SECRET]", text).replace("\n", " ").strip()
    return clean[:limit]


def _redacted_safe(text: str) -> bool:
    return SECRET_PATTERN.search(text) is None


async def _retrieve_context(port: int, trial: DerbyTrial) -> dict[str, Any]:
    t0 = time.perf_counter()
    hits = await _one_call(port, "search", {
        "query": trial.retrieval_query,
        "limit": CONTEXT_LIMIT,
        "project": "cloud-derby",
        "record_kind": "memory",
    })
    search_ms = (time.perf_counter() - t0) * 1000
    hit_list = hits if isinstance(hits, list) else []

    lines: list[str] = []
    memory_ids: list[str] = []
    for hit in hit_list[:CONTEXT_LIMIT]:
        if not isinstance(hit, dict):
            continue
        memory_id = str(hit.get("memory_id") or hit.get("node_hash") or "")
        preview = str(hit.get("summary_preview") or hit.get("summary") or "").strip()
        if not preview:
            continue
        memory_ids.append(memory_id)
        lines.append(f"- [{memory_id}] {_sanitize_preview(preview, 320)}")

    body = "\n".join(lines)
    if len(body) > CONTEXT_MAX_CHARS:
        body = body[: CONTEXT_MAX_CHARS - 16].rstrip() + "\n...<truncated>"
    context_text = f"<helix-active-context>\n{body}\n</helix-active-context>" if body else ""
    return {
        "context_text": context_text,
        "hit_count": len(hit_list),
        "search_ms": search_ms,
        "memory_ids": memory_ids,
        "chars": len(context_text),
        "redacted_safe": _redacted_safe(context_text),
    }


def _synthetic_response(prompt: str, trial: DerbyTrial, memory_enabled: bool) -> str:
    digest = hashlib.md5((trial.trial_id + prompt).encode()).hexdigest()[:8]
    if not memory_enabled:
        return (
            f"[SYN-{digest}] I do not have the private HeliX evidence for this case. "
            "The responsible answer is to request the internal incident log before naming a cause."
        )
    ids = ", ".join(trial.required_memory_ids)
    evidence = ", ".join(trial.required_evidence_terms)
    actions = ", ".join(trial.action_terms)
    preferred = f" Preferred current source: {trial.preferred_memory_id}." if trial.preferred_memory_id else ""
    return (
        f"[SYN-{digest}] Citing {ids}: the private evidence says {evidence}. "
        f"Action: {actions}.{preferred} This answer uses HeliX active memory."
    )


def _extract_message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return ""


def _extract_reasoning_chars(message: dict[str, Any]) -> int:
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str):
        return len(reasoning)
    if isinstance(reasoning, list):
        return len(json.dumps(reasoning, ensure_ascii=False))
    return 0


async def _llm_call(
    prompt: str,
    trial: DerbyTrial,
    memory_enabled: bool,
    max_tokens: int = DERBY_MAX_TOKENS,
) -> LLMResult:
    if not DEEPINFRA_TOKEN:
        return LLMResult(
            text=_synthetic_response(prompt, trial, memory_enabled),
            synthetic=True,
            latency_ms=1.0,
            model="synthetic",
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise incident analyst. If private HeliX context is provided, "
                "answer only from it and cite memory IDs. If no private context is provided, "
                "say that the private evidence is unavailable. Never reveal secrets. "
                "Do not show reasoning. Return only the final answer."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    total_latency_ms = 0.0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost": 0.0}
    final_text = ""
    finish_reason = ""
    raw_message_keys: tuple[str, ...] = ()
    reasoning_chars = 0
    retry_count = 0
    thinking_payload_rejected = False
    async with httpx.AsyncClient(timeout=120.0) as client:
        for attempt in range(2):
            effective_messages = messages
            effective_max_tokens = max_tokens if attempt == 0 else min(max_tokens * 2, 1024)
            if attempt > 0:
                retry_count += 1
                effective_messages = [
                    messages[0],
                    {
                        "role": "user",
                        "content": (
                            "Return the final answer only. Do not include hidden reasoning. "
                            "Keep it under 120 words and cite memory IDs if present.\n\n"
                            "/no_think\n"
                            f"{prompt}"
                        ),
                    },
                ]
            request_json: dict[str, Any] = {
                "model": DERBY_MODEL,
                "messages": effective_messages,
                "max_tokens": effective_max_tokens,
                "temperature": 0.1,
            }
            if DERBY_DISABLE_THINKING:
                request_json["enable_thinking"] = False
                request_json["chat_template_kwargs"] = {"enable_thinking": False}
            t0 = time.perf_counter()
            response = await client.post(
                f"{DEEPINFRA_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {DEEPINFRA_TOKEN}"},
                json=request_json,
            )
            if response.status_code == 400 and DERBY_DISABLE_THINKING:
                thinking_payload_rejected = True
                request_json.pop("enable_thinking", None)
                request_json.pop("chat_template_kwargs", None)
                response = await client.post(
                    f"{DEEPINFRA_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {DEEPINFRA_TOKEN}"},
                    json=request_json,
                )
            response.raise_for_status()
            total_latency_ms += (time.perf_counter() - t0) * 1000
            data = response.json()
            usage = data.get("usage", {})
            total_usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            total_usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            total_usage["total_tokens"] += int(usage.get("total_tokens") or 0)
            total_usage["estimated_cost"] += float(usage.get("estimated_cost") or 0.0)
            choice = data["choices"][0]
            message = choice.get("message") or {}
            raw_message_keys = tuple(sorted(str(key) for key in message.keys()))
            finish_reason = str(choice.get("finish_reason") or "")
            reasoning_chars += _extract_reasoning_chars(message)
            final_text = _extract_message_text(message)
            if final_text:
                break

    return LLMResult(
        text=final_text,
        synthetic=False,
        latency_ms=total_latency_ms,
        model=DERBY_MODEL,
        prompt_tokens=total_usage["prompt_tokens"],
        completion_tokens=total_usage["completion_tokens"],
        total_tokens=total_usage["total_tokens"],
        estimated_cost=total_usage["estimated_cost"],
        retry_count=retry_count,
        empty_output=not bool(final_text),
        finish_reason=finish_reason,
        raw_message_keys=raw_message_keys,
        reasoning_chars=reasoning_chars,
        thinking_disabled_requested=DERBY_DISABLE_THINKING,
        thinking_payload_rejected=thinking_payload_rejected,
    )


def _score_answer(output: str, trial: DerbyTrial) -> dict[str, Any]:
    lower = output.lower()
    evidence_hits = [term for term in trial.required_evidence_terms if term.lower() in lower]
    memory_id_hits = [mid for mid in trial.required_memory_ids if mid.lower() in lower]
    action_hits = [term for term in trial.action_terms if term.lower() in lower]
    deprecated_hits = [mid for mid in trial.deprecated_memory_ids if mid.lower() in lower]
    preferred_hit = bool(trial.preferred_memory_id and trial.preferred_memory_id.lower() in lower)
    safe = _redacted_safe(output)

    answer_score = min(4, len(evidence_hits)) + min(3, len(action_hits))
    citation_score = len(memory_id_hits)
    contradiction_score = 0
    if trial.preferred_memory_id:
        contradiction_score += 2 if preferred_hit else -2
        contradiction_score -= len(deprecated_hits)
    safety_score = 2 if safe else -8
    total = answer_score + citation_score + contradiction_score + safety_score
    return {
        "total": total,
        "answer_score": answer_score,
        "citation_score": citation_score,
        "contradiction_score": contradiction_score,
        "safety_score": safety_score,
        "evidence_hits": evidence_hits,
        "memory_id_hits": memory_id_hits,
        "action_hits": action_hits,
        "deprecated_hits": deprecated_hits,
        "preferred_hit": preferred_hit,
        "redacted_safe": safe,
    }


async def _seed_memories(port: int) -> None:
    items = [
        {
            **memory,
            "record_kind": "memory",
            "memory_type": "semantic",
            "decay_score": 1.0,
        }
        for memory in SEED_MEMORIES
    ]
    inserted = await _one_call(port, "bulk_remember", {"items": items})
    assert isinstance(inserted, list)
    assert len(inserted) == len(SEED_MEMORIES)


async def _persist_output(
    port: int,
    trial: DerbyTrial,
    branch: str,
    output: str,
    score: dict[str, Any],
) -> None:
    await _one_call(port, "remember", {
        "content": _sanitize_preview(output, 1200),
        "project": "cloud-derby-results",
        "agent_id": f"cloud-derby-memory-{branch}",
        "record_kind": "memory",
        "memory_id": f"{trial.trial_id}-{branch}-answer",
        "summary": f"{trial.trial_id} {branch} score {score['total']}",
        "index_content": _sanitize_preview(output, 700),
        "importance": 8 if branch == "on" else 5,
    })


async def _run_trial(port: int, trial: DerbyTrial) -> dict[str, Any]:
    off_prompt = (
        f"Private case question: {trial.question}\n"
        "Do not invent private facts. If the private evidence is not present in this prompt, say so."
    )
    off = await _llm_call(off_prompt, trial, memory_enabled=False)
    off_score = _score_answer(off.text, trial)
    await _persist_output(port, trial, "off", off.text, off_score)

    context = await _retrieve_context(port, trial)
    on_prompt = (
        f"{context['context_text']}\n\n"
        f"Private case question: {trial.question}\n"
        "Use only the HeliX active context above. Cite the memory IDs that support your answer. "
        "If a memory supersedes another, prefer the current superseding memory."
    )
    on = await _llm_call(on_prompt, trial, memory_enabled=True)
    on_score = _score_answer(on.text, trial)
    await _persist_output(port, trial, "on", on.text, on_score)

    return {
        "trial_id": trial.trial_id,
        "memory_off_score": off_score["total"],
        "memory_on_score": on_score["total"],
        "score_delta": on_score["total"] - off_score["total"],
        "citation_delta": on_score["citation_score"] - off_score["citation_score"],
        "memory_off_breakdown": off_score,
        "memory_on_breakdown": on_score,
        "retrieved_memory_ids": context["memory_ids"],
        "required_memory_ids": list(trial.required_memory_ids),
        "context_hit_count": context["hit_count"],
        "context_search_ms": context["search_ms"],
        "context_chars": context["chars"],
        "redacted_safe": context["redacted_safe"] and off_score["redacted_safe"] and on_score["redacted_safe"],
        "llm_latency_ms_off": off.latency_ms,
        "llm_latency_ms_on": on.latency_ms,
        "token_usage_off": {
            "prompt_tokens": off.prompt_tokens,
            "completion_tokens": off.completion_tokens,
            "total_tokens": off.total_tokens,
            "estimated_cost": off.estimated_cost,
            "retry_count": off.retry_count,
            "empty_output": off.empty_output,
            "finish_reason": off.finish_reason,
            "raw_message_keys": list(off.raw_message_keys),
            "reasoning_chars": off.reasoning_chars,
            "thinking_disabled_requested": off.thinking_disabled_requested,
            "thinking_payload_rejected": off.thinking_payload_rejected,
        },
        "token_usage_on": {
            "prompt_tokens": on.prompt_tokens,
            "completion_tokens": on.completion_tokens,
            "total_tokens": on.total_tokens,
            "estimated_cost": on.estimated_cost,
            "retry_count": on.retry_count,
            "empty_output": on.empty_output,
            "finish_reason": on.finish_reason,
            "raw_message_keys": list(on.raw_message_keys),
            "reasoning_chars": on.reasoning_chars,
            "thinking_disabled_requested": on.thinking_disabled_requested,
            "thinking_payload_rejected": on.thinking_payload_rejected,
        },
        "output_preview_off": _sanitize_preview(off.text),
        "output_preview_on": _sanitize_preview(on.text),
    }


def _write_artifact(trials: list[dict[str, Any]]) -> dict[str, Any]:
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    wins = sum(1 for trial in trials if trial["memory_on_score"] > trial["memory_off_score"])
    score_deltas = [float(trial["score_delta"]) for trial in trials]
    citation_deltas = [float(trial["citation_delta"]) for trial in trials]
    search_latencies = [float(trial["context_search_ms"]) for trial in trials]
    llm_latencies = [
        float(ms)
        for trial in trials
        for ms in (trial["llm_latency_ms_off"], trial["llm_latency_ms_on"])
    ]
    estimated_cost = sum(
        float(trial["token_usage_off"]["estimated_cost"]) + float(trial["token_usage_on"]["estimated_cost"])
        for trial in trials
    )
    reasoning_chars = sum(
        int(trial["token_usage_off"]["reasoning_chars"]) + int(trial["token_usage_on"]["reasoning_chars"])
        for trial in trials
    )
    thinking_payload_rejections = sum(
        int(bool(trial["token_usage_off"]["thinking_payload_rejected"]))
        + int(bool(trial["token_usage_on"]["thinking_payload_rejected"]))
        for trial in trials
    )
    secrets_leaked = sum(0 if trial["redacted_safe"] else 1 for trial in trials)
    empty_outputs = sum(
        int(bool(trial["token_usage_off"]["empty_output"]))
        + int(bool(trial["token_usage_on"]["empty_output"]))
        for trial in trials
    )
    context_p50 = _p50(search_latencies)
    llm_p50 = _p50(llm_latencies)
    overhead_pct = None if not DEEPINFRA_TOKEN else (context_p50 / max(llm_p50, 0.001)) * 100
    quality_gate_passed = (wins / max(len(trials), 1)) >= 0.6 and (sum(citation_deltas) / max(len(citation_deltas), 1)) > 0
    delta_key = "avg_score_delta" if DEEPINFRA_TOKEN else "avg_template_completion_delta"
    artifact = {
        "artifact": "local-cloud-amnesia-derby",
        "mode": "real" if DEEPINFRA_TOKEN else "synthetic",
        "llm_synthetic_mode": not bool(DEEPINFRA_TOKEN),
        "model": DERBY_MODEL if DEEPINFRA_TOKEN else "synthetic",
        "seed_memory_count": len(SEED_MEMORIES),
        "trial_count": len(trials),
        "memory_on_win_rate": wins / max(len(trials), 1),
        delta_key: sum(score_deltas) / max(len(score_deltas), 1),
        "citation_rate_delta": sum(citation_deltas) / max(len(citation_deltas), 1),
        "context_search_ms_p50": context_p50,
        "llm_latency_ms_p50": llm_p50,
        "context_overhead_vs_llm_pct": overhead_pct,
        "context_overhead_note": (
            "not computed in synthetic mode because LLM latency is mocked"
            if not DEEPINFRA_TOKEN else "computed from real DeepInfra p50 latency"
        ),
        "estimated_cost": estimated_cost,
        "thinking_disabled_requested": DERBY_DISABLE_THINKING,
        "thinking_payload_rejections": thinking_payload_rejections,
        "reasoning_chars": reasoning_chars,
        "secrets_leaked": secrets_leaked,
        "empty_outputs": empty_outputs,
        "quality_gate_passed": quality_gate_passed,
        "quality_gate_required": DERBY_REQUIRE_QUALITY,
        "redacted_safe": secrets_leaked == 0,
        "public_claim_level": (
            "mechanics_verified"
            if not DEEPINFRA_TOKEN
            else ("real_cloud_quality_delta_observed" if quality_gate_passed else "real_cloud_quality_gate_failed")
        ),
        "claims_allowed": [
            "Same cloud model path compared with and without HeliX active memory.",
            "memory_on retrieves private DAG evidence before generation and is scored deterministically.",
            "Synthetic mode verifies mechanics; real mode observes cloud quality delta.",
        ],
        "claims_not_allowed": [
            "This is not a 100k-node stress benchmark.",
            "No API token, request headers, or raw secret-bearing prompt is persisted.",
            "Cloud model KV/state is not persisted by HeliX; HeliX supplies above-prompt DAG memory here.",
            "Synthetic score deltas measure template completion when context is injected, not reasoning quality.",
        ],
        "trials": trials,
        "generated_ms": int(time.time() * 1000),
    }
    ARTIFACT.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    return artifact


class TestCloudAmnesiaDerby:
    def test_cloud_model_becomes_less_amnesic_with_helix_memory(self) -> None:
        proc, port = _start_server()
        try:
            async def run() -> dict[str, Any]:
                await _seed_memories(port)
                results = []
                for trial in TRIALS:
                    results.append(await _run_trial(port, trial))
                return _write_artifact(results)

            artifact = asyncio.run(run())

            assert artifact["trial_count"] == len(TRIALS)
            assert artifact["seed_memory_count"] >= 20
            assert artifact["secrets_leaked"] == 0
            assert artifact["redacted_safe"] is True
            assert all(trial["context_hit_count"] > 0 for trial in artifact["trials"])
            assert all(trial["redacted_safe"] for trial in artifact["trials"])
            assert all(
                set(trial["required_memory_ids"]).issubset(set(trial["retrieved_memory_ids"]))
                for trial in artifact["trials"]
            )

            if artifact["llm_synthetic_mode"]:
                assert artifact["memory_on_win_rate"] >= 0.8
                assert artifact["avg_template_completion_delta"] > 0
            elif artifact["quality_gate_required"]:
                assert artifact["memory_on_win_rate"] >= 0.6
                assert artifact["citation_rate_delta"] > 0
                assert artifact["avg_score_delta"] > 0

            print(f"\n{'=' * 60}")
            print("  CLOUD AMNESIA DERBY")
            print(f"  Mode:                 {artifact['mode']}")
            print(f"  Model:                {artifact['model']}")
            print(f"  Trials:               {artifact['trial_count']}")
            print(f"  memory_on win rate:   {artifact['memory_on_win_rate']:.2f}")
            delta_label = "Avg score delta" if not artifact["llm_synthetic_mode"] else "Avg template delta"
            delta_value = artifact.get("avg_score_delta", artifact.get("avg_template_completion_delta", 0.0))
            print(f"  {delta_label}:      {delta_value:.1f}")
            print(f"  Citation delta:       {artifact['citation_rate_delta']:.1f}")
            print(f"  Context p50:          {artifact['context_search_ms_p50']:.1f}ms")
            print(f"  LLM p50:              {artifact['llm_latency_ms_p50']:.1f}ms")
            print(f"  Estimated cost:       {artifact['estimated_cost']:.6f}")
            print(f"  Secrets leaked:       {artifact['secrets_leaked']}")
            print(f"{'=' * 60}")
        finally:
            _stop_server(proc)
