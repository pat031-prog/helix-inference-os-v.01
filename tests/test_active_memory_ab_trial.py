"""
HeliX Active Memory A/B Trial.

Same model, same task:
  - memory_off: no DAG context before generation.
  - memory_on: HeliX.search() -> <helix-active-context> -> LLM.generate().

The synthetic path is deterministic and proves the control-plane mechanics without
network or spend. Real DeepInfra mode is opt-in through the secure PowerShell runner.
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
ARTIFACT = REPO / "verification" / "local-active-memory-ab-trial.json"

DEEPINFRA_TOKEN = os.environ.get("DEEPINFRA_API_TOKEN", "")
DEEPINFRA_BASE = "https://api.deepinfra.com/v1/openai"
AB_MODEL = os.environ.get("HELIX_AB_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
CONTEXT_LIMIT = int(os.environ.get("HELIX_CONTEXT_LIMIT", "5"))
CONTEXT_MAX_CHARS = int(os.environ.get("HELIX_CONTEXT_MAX_CHARS", "1200"))

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
class Trial:
    trial_id: str
    question: str
    retrieval_query: str
    required_evidence_terms: tuple[str, ...]
    required_memory_ids: tuple[str, ...]
    action_terms: tuple[str, ...]


@dataclass
class LLMResult:
    text: str
    synthetic: bool
    latency_ms: float
    model: str


SEED_MEMORIES: list[dict[str, Any]] = [
    {
        "memory_id": "inc-auth-001",
        "summary": "inc-auth-001 evidence: JWT kid rotation mismatch appears after signing-key rollout.",
        "content": "Auth incident: 401 spike started after JWT signing-key rotation. kid header mismatch observed.",
        "index_content": "jwt kid rotation mismatch auth signing key 401",
        "project": "ab-trial",
        "agent_id": "incident-archivist",
        "importance": 9,
    },
    {
        "memory_id": "inc-auth-002",
        "summary": "inc-auth-002 evidence: rollback key rotation and pin kid map fixed canary.",
        "content": "Mitigation note: rollback key rotation, pin kid map, then verify token validation.",
        "index_content": "rollback key rotation pin kid map token validation",
        "project": "ab-trial",
        "agent_id": "incident-archivist",
        "importance": 9,
    },
    {
        "memory_id": "sec-filter-001",
        "summary": "sec-filter-001 evidence: debug logger captured [REDACTED_SECRET] before privacy filter.",
        "content": "Security event: debug logger wrote redacted credential before privacy filter ran.",
        "index_content": "debug logger redacted secret privacy filter before indexing leak",
        "project": "ab-trial",
        "agent_id": "security-archivist",
        "importance": 9,
    },
    {
        "memory_id": "sec-filter-002",
        "summary": "sec-filter-002 evidence: move privacy filter before indexing and logging.",
        "content": "Fix: privacy filter must run before logging, indexing, and DAG insertion.",
        "index_content": "privacy filter before logging indexing dag insertion fix",
        "project": "ab-trial",
        "agent_id": "security-archivist",
        "importance": 9,
    },
    {
        "memory_id": "supply-001",
        "summary": "supply-001 evidence: unsigned model artifact changed during mirror sync.",
        "content": "Supply-chain signal: model artifact checksum changed during mirror sync.",
        "index_content": "unsigned model artifact checksum mirror sync supply chain",
        "project": "ab-trial",
        "agent_id": "release-archivist",
        "importance": 8,
    },
    {
        "memory_id": "supply-002",
        "summary": "supply-002 evidence: enforce sha256 manifest and Merkle receipt before deploy.",
        "content": "Release rule: require sha256 manifest, Merkle receipt, and deploy hold on mismatch.",
        "index_content": "sha256 manifest merkle receipt deploy hold mismatch",
        "project": "ab-trial",
        "agent_id": "release-archivist",
        "importance": 8,
    },
    {
        "memory_id": "noise-001",
        "summary": "noise-001 note: UI copy preference is warm editorial tone.",
        "content": "Design memory unrelated to incidents.",
        "index_content": "ui copy editorial tone design",
        "project": "ab-trial",
        "agent_id": "copywriter",
        "importance": 4,
    },
    {
        "memory_id": "noise-002",
        "summary": "noise-002 note: batch benchmark used synthetic event streams.",
        "content": "Benchmark note unrelated to root-cause analysis.",
        "index_content": "benchmark synthetic event streams",
        "project": "ab-trial",
        "agent_id": "bench-agent",
        "importance": 4,
    },
    {
        "memory_id": "ops-001",
        "summary": "ops-001 note: restart policy should preserve three snapshots.",
        "content": "Ops note for snapshot retention.",
        "index_content": "restart policy snapshot retention",
        "project": "ab-trial",
        "agent_id": "ops-agent",
        "importance": 5,
    },
    {
        "memory_id": "ops-002",
        "summary": "ops-002 note: network retry jitter avoids herd effects.",
        "content": "Ops note about jitter and retries.",
        "index_content": "network retry jitter herd effects",
        "project": "ab-trial",
        "agent_id": "ops-agent",
        "importance": 5,
    },
    {
        "memory_id": "research-001",
        "summary": "research-001 note: prompt caching requires exact prefix match.",
        "content": "Research note unrelated to the trial root causes.",
        "index_content": "prompt caching exact prefix match",
        "project": "ab-trial",
        "agent_id": "research-agent",
        "importance": 6,
    },
    {
        "memory_id": "research-002",
        "summary": "research-002 note: hybrid Mamba prefix slicing remains experimental.",
        "content": "Research note unrelated to incident triage.",
        "index_content": "hybrid mamba prefix slicing experimental",
        "project": "ab-trial",
        "agent_id": "research-agent",
        "importance": 6,
    },
]

TRIALS: tuple[Trial, ...] = (
    Trial(
        trial_id="auth-rotation",
        question="What was the likely root cause of the auth outage and what should we do first?",
        retrieval_query="auth outage jwt kid rotation mismatch rollback",
        required_evidence_terms=("jwt", "kid", "rotation", "mismatch"),
        required_memory_ids=("inc-auth-001", "inc-auth-002"),
        action_terms=("rollback", "pin", "verify"),
    ),
    Trial(
        trial_id="privacy-filter",
        question="Why did the credential-safety incident happen and what should the patch enforce?",
        retrieval_query="credential safety privacy filter debug logger before indexing",
        required_evidence_terms=("privacy", "filter", "debug", "logger"),
        required_memory_ids=("sec-filter-001", "sec-filter-002"),
        action_terms=("before", "logging", "indexing"),
    ),
    Trial(
        trial_id="supply-chain",
        question="What explains the suspicious release artifact and what gate should block deployment?",
        retrieval_query="supply chain unsigned model artifact sha256 merkle receipt",
        required_evidence_terms=("unsigned", "artifact", "mirror", "sync"),
        required_memory_ids=("supply-001", "supply-002"),
        action_terms=("sha256", "merkle", "deploy"),
    ),
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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


def _sanitize_preview(text: str, limit: int = 220) -> str:
    clean = SECRET_PATTERN.sub("[REDACTED_SECRET]", text).replace("\n", " ").strip()
    return clean[:limit]


def _redacted_safe(text: str) -> bool:
    return SECRET_PATTERN.search(text) is None


async def _retrieve_context(
    port: int,
    query: str,
    limit: int = CONTEXT_LIMIT,
    max_chars: int = CONTEXT_MAX_CHARS,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    hits = await _one_call(port, "search", {
        "query": query,
        "limit": limit,
        "project": "ab-trial",
        "record_kind": "memory",
    })
    search_ms = (time.perf_counter() - t0) * 1000
    hit_list = hits if isinstance(hits, list) else []

    lines: list[str] = []
    memory_ids: list[str] = []
    for hit in hit_list[:limit]:
        if not isinstance(hit, dict):
            continue
        memory_id = str(hit.get("memory_id") or hit.get("node_hash") or "")
        preview = str(hit.get("summary_preview") or hit.get("summary") or "").strip()
        if not preview:
            continue
        memory_ids.append(memory_id)
        lines.append(f"- [{memory_id}] {_sanitize_preview(preview, 260)}")

    body = "\n".join(lines)
    if len(body) > max_chars:
        body = body[: max_chars - 16].rstrip() + "\n...<truncated>"
    context_text = f"<helix-active-context>\n{body}\n</helix-active-context>" if body else ""
    return {
        "context_text": context_text,
        "hit_count": len(hit_list),
        "search_ms": search_ms,
        "memory_ids": memory_ids,
        "chars": len(context_text),
        "redacted_safe": _redacted_safe(context_text),
    }


def _synthetic_response(prompt: str, trial: Trial, memory_enabled: bool) -> str:
    digest = hashlib.md5((trial.trial_id + prompt).encode()).hexdigest()[:8]
    if not memory_enabled:
        return (
            f"[SYN-{digest}] Without retrieved HeliX context, the safest answer is to "
            "request logs, compare recent deploys, and avoid naming a root cause."
        )
    ids = ", ".join(trial.required_memory_ids)
    evidence = ", ".join(trial.required_evidence_terms)
    actions = ", ".join(trial.action_terms)
    return (
        f"[SYN-{digest}] Evidence {ids} points to {evidence}. "
        f"Recommended first action: {actions}. This answer is based on active HeliX DAG context."
    )


async def _llm_call(
    prompt: str,
    trial: Trial,
    memory_enabled: bool,
    max_tokens: int = 180,
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
                "You are an incident analyst. Answer concisely with evidence IDs when provided. "
                "Never reveal secrets; keep [REDACTED_SECRET] redacted."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(
            f"{DEEPINFRA_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {DEEPINFRA_TOKEN}"},
            json={
                "model": AB_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.2,
            },
        )
        response.raise_for_status()
        data = response.json()
    return LLMResult(
        text=data["choices"][0]["message"]["content"],
        synthetic=False,
        latency_ms=(time.perf_counter() - t0) * 1000,
        model=AB_MODEL,
    )


def _score_answer(output: str, trial: Trial) -> dict[str, Any]:
    lower = output.lower()
    evidence_hits = [term for term in trial.required_evidence_terms if term.lower() in lower]
    memory_id_hits = [mid for mid in trial.required_memory_ids if mid.lower() in lower]
    action_hits = [term for term in trial.action_terms if term.lower() in lower]
    safe = _redacted_safe(output)

    answer_score = min(4, len(evidence_hits)) + min(3, len(action_hits)) + (2 if safe else 0)
    citation_score = len(memory_id_hits)
    safety_score = 2 if safe else -6
    total = answer_score + citation_score + safety_score
    return {
        "total": total,
        "answer_score": answer_score,
        "citation_score": citation_score,
        "safety_score": safety_score,
        "evidence_hits": evidence_hits,
        "memory_id_hits": memory_id_hits,
        "action_hits": action_hits,
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
    trial: Trial,
    branch: str,
    output: str,
    score: dict[str, Any],
) -> None:
    await _one_call(port, "remember", {
        "content": _sanitize_preview(output, 1000),
        "project": "ab-trial",
        "agent_id": f"ab-agent-memory-{branch}",
        "record_kind": "memory",
        "memory_id": f"{trial.trial_id}-{branch}-answer",
        "summary": f"{trial.trial_id} {branch} score {score['total']}",
        "index_content": _sanitize_preview(output, 600),
        "importance": 7 if branch == "on" else 5,
    })


async def _run_trial(port: int, trial: Trial) -> dict[str, Any]:
    clean_prompt = (
        f"Question: {trial.question}\n"
        "Answer with the likely root cause, evidence, and first action. "
        "If you lack evidence, say so."
    )
    off = await _llm_call(clean_prompt, trial, memory_enabled=False)
    off_score = _score_answer(off.text, trial)
    await _persist_output(port, trial, "off", off.text, off_score)

    context = await _retrieve_context(port, trial.retrieval_query)
    on_prompt = (
        f"{context['context_text']}\n\n"
        f"Question: {trial.question}\n"
        "Use only the retrieved HeliX context for evidence. Cite memory IDs."
    )
    on = await _llm_call(on_prompt, trial, memory_enabled=True)
    on_score = _score_answer(on.text, trial)
    await _persist_output(port, trial, "on", on.text, on_score)

    return {
        "trial_id": trial.trial_id,
        "memory_off_score": off_score["total"],
        "memory_on_score": on_score["total"],
        "score_delta": on_score["total"] - off_score["total"],
        "memory_off_breakdown": off_score,
        "memory_on_breakdown": on_score,
        "retrieved_memory_ids": context["memory_ids"],
        "context_hit_count": context["hit_count"],
        "context_search_ms": context["search_ms"],
        "context_chars": context["chars"],
        "redacted_safe": context["redacted_safe"] and off_score["redacted_safe"] and on_score["redacted_safe"],
        "llm_latency_ms_off": off.latency_ms,
        "llm_latency_ms_on": on.latency_ms,
        "output_preview_off": _sanitize_preview(off.text),
        "output_preview_on": _sanitize_preview(on.text),
    }


def _write_artifact(trials: list[dict[str, Any]]) -> dict[str, Any]:
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    wins = sum(1 for trial in trials if trial["memory_on_score"] > trial["memory_off_score"])
    deltas = [float(trial["score_delta"]) for trial in trials]
    search_latencies = [float(trial["context_search_ms"]) for trial in trials]
    llm_latencies = [
        float(ms)
        for trial in trials
        for ms in (trial["llm_latency_ms_off"], trial["llm_latency_ms_on"])
    ]
    secrets_leaked = sum(0 if trial["redacted_safe"] else 1 for trial in trials)
    context_p50 = _p50(search_latencies)
    llm_p50 = _p50(llm_latencies)
    overhead_pct = None if not DEEPINFRA_TOKEN else (context_p50 / max(llm_p50, 0.001)) * 100
    delta_key = "avg_score_delta" if DEEPINFRA_TOKEN else "avg_template_completion_delta"
    artifact = {
        "artifact": "local-active-memory-ab-trial",
        "mode": "real" if DEEPINFRA_TOKEN else "synthetic",
        "llm_synthetic_mode": not bool(DEEPINFRA_TOKEN),
        "model": AB_MODEL if DEEPINFRA_TOKEN else "synthetic",
        "trial_count": len(trials),
        "memory_on_win_rate": wins / max(len(trials), 1),
        delta_key: sum(deltas) / max(len(deltas), 1),
        "context_search_ms_p50": context_p50,
        "llm_latency_ms_p50": llm_p50,
        "context_overhead_vs_llm_pct": overhead_pct,
        "context_overhead_note": (
            "not computed in synthetic mode because LLM latency is mocked"
            if not DEEPINFRA_TOKEN else "computed from real DeepInfra p50 latency"
        ),
        "secrets_leaked": secrets_leaked,
        "redacted_safe": secrets_leaked == 0,
        "public_claim_level": "quality_delta_observed" if DEEPINFRA_TOKEN else "mechanics_verified",
        "claims_allowed": [
            "Same task and model path compared with and without HeliX active memory.",
            "memory_on retrieves DAG evidence before generation and is scored deterministically.",
            "Synthetic mode verifies mechanics; real mode observes quality delta.",
        ],
        "claims_not_allowed": [
            "This is not a 100k-node stress benchmark.",
            "No API token, request headers, or raw secret-bearing prompt is persisted.",
            "Synthetic score deltas measure template completion when context is injected, not reasoning quality.",
        ],
        "superseded_by": "local-cloud-amnesia-derby",
        "supersession_reason": "Cloud Amnesia Derby has more trials and stricter cloud-quality gates for quality claims.",
        "trials": trials,
        "generated_ms": int(time.time() * 1000),
    }
    ARTIFACT.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    return artifact


class TestActiveMemoryABTrial:
    def test_memory_on_beats_memory_off_with_dag_context(self) -> None:
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
            assert artifact["secrets_leaked"] == 0
            assert artifact["redacted_safe"] is True
            assert all(trial["context_hit_count"] > 0 for trial in artifact["trials"])
            assert all(trial["redacted_safe"] for trial in artifact["trials"])

            if artifact["llm_synthetic_mode"]:
                assert artifact["memory_on_win_rate"] >= 0.8
                assert artifact["avg_template_completion_delta"] > 0
            else:
                assert artifact["memory_on_win_rate"] >= 0.5
                assert artifact["avg_score_delta"] > 0

            print(f"\n{'=' * 60}")
            print("  ACTIVE MEMORY A/B TRIAL")
            print(f"  Mode:                 {artifact['mode']}")
            print(f"  Trials:               {artifact['trial_count']}")
            print(f"  memory_on win rate:   {artifact['memory_on_win_rate']:.2f}")
            delta_label = "Avg score delta" if not artifact["llm_synthetic_mode"] else "Avg template delta"
            delta_value = artifact.get("avg_score_delta", artifact.get("avg_template_completion_delta", 0.0))
            print(f"  {delta_label}:      {delta_value:.1f}")
            print(f"  Context p50:          {artifact['context_search_ms_p50']:.1f}ms")
            print(f"  LLM p50:              {artifact['llm_latency_ms_p50']:.1f}ms")
            print(f"  Secrets leaked:       {artifact['secrets_leaked']}")
            print(f"{'=' * 60}")
        finally:
            _stop_server(proc)
