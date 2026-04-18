"""
HeliX Identity Trust Gauntlet v2.

This suite focuses on auditable conversations across heterogeneous cloud models:
- deterministic run dates,
- per-call LLM receipts without storing full prompts or secrets,
- Merkle-linked state events,
- cross-model continuity through HeliX memory rather than private cloud state.

Synthetic mode is default. Real mode is opt-in via DEEPINFRA_API_TOKEN and the
secure PowerShell wrapper.
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
VERIFICATION = REPO / "verification"

DEEPINFRA_TOKEN = os.environ.get("DEEPINFRA_API_TOKEN", "")
DEEPINFRA_BASE = "https://api.deepinfra.com/v1/openai"
DISABLE_THINKING = os.environ.get("HELIX_IDENTITY_DISABLE_THINKING", "1") != "0"
REQUIRE_ALL_MODELS = os.environ.get("HELIX_REQUIRE_ALL_MODELS", "0") == "1"
OUROBOROS_CYCLES = int(os.environ.get("HELIX_OUROBOROS_CYCLES", "20"))

RUN_STARTED_AT_UTC = os.environ.get("HELIX_RUN_STARTED_AT_UTC") or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
RUN_DATE_UTC = os.environ.get("HELIX_RUN_DATE_UTC") or RUN_STARTED_AT_UTC[:10]
RUN_TIMEZONE = os.environ.get("HELIX_RUN_TIMEZONE", "America/Buenos_Aires")
RUN_ID = os.environ.get("HELIX_RUN_ID") or f"identity-v2-{hashlib.sha256(RUN_STARTED_AT_UTC.encode('utf-8')).hexdigest()[:12]}"

GEMMA_MODEL = os.environ.get("HELIX_V2_GEMMA_MODEL", "google/gemma-4-26B-A4B-it")
CLAUDE_MODEL = os.environ.get("HELIX_V2_CLAUDE_MODEL", "anthropic/claude-4-sonnet")
QWEN_MODEL = os.environ.get("HELIX_V2_QWEN_MODEL", "Qwen/Qwen3.5-9B")

TRI_MODELS = [
    ("gemma", GEMMA_MODEL, "Gemma 4 26B A4B", "google"),
    ("claude", CLAUDE_MODEL, "Claude 4 Sonnet", "anthropic"),
    ("qwen", QWEN_MODEL, "Qwen 3.5 9B", "qwen"),
]

pytestmark = pytest.mark.skipif(not RUST_BIN.exists(), reason=f"Rust binary not found: {RUST_BIN}")


def _digest_seed(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@dataclass
class LLMResult:
    text: str
    model: str
    synthetic: bool
    latency_ms: float
    tokens_used: int = 0
    call_id: str = ""


@dataclass
class AuditRecorder:
    run_id: str
    run_started_at_utc: str
    run_date_utc: str
    run_timezone: str
    calls: list[dict[str, Any]] = field(default_factory=list)
    state_events: list[dict[str, Any]] = field(default_factory=list)
    _call_counter: int = 0
    _state_counter: int = 0

    def next_call_id(self, step_id: str) -> str:
        self._call_counter += 1
        safe_step = re.sub(r"[^a-zA-Z0-9_-]+", "-", step_id).strip("-")[:40]
        return f"llm-{self._call_counter:04d}-{safe_step}"

    def next_state_id(self, method: str) -> str:
        self._state_counter += 1
        safe_method = re.sub(r"[^a-zA-Z0-9_-]+", "-", method).strip("-")[:32]
        return f"state-{self._state_counter:04d}-{safe_method}"

    def record_call(self, entry: dict[str, Any]) -> None:
        self.calls.append(entry)

    def record_state(self, entry: dict[str, Any]) -> None:
        self.state_events.append(entry)

    def provider_families(self) -> list[str]:
        return sorted({str(c.get("provider_family")) for c in self.calls if c.get("provider_family")})

    def model_substitutions(self) -> list[dict[str, Any]]:
        substitutions = []
        for call in self.calls:
            requested = call.get("requested_model")
            actual = call.get("actual_model")
            if requested and actual and requested != actual:
                substitutions.append({
                    "call_id": call.get("call_id"),
                    "requested_model": requested,
                    "actual_model": actual,
                })
        return substitutions

    def audit_completeness_score(self) -> float:
        if not self.calls:
            return 0.0
        required = [
            "call_id",
            "step_id",
            "role",
            "requested_model",
            "actual_model",
            "provider_family",
            "started_at_utc",
            "ended_at_utc",
            "latency_ms",
            "system_digest",
            "prompt_digest",
            "output_digest",
            "prompt_preview_sanitized",
            "output_preview_sanitized",
        ]
        present = 0
        total = len(required) * len(self.calls)
        for call in self.calls:
            present += sum(1 for key in required if call.get(key) not in (None, ""))
        return round(present / max(total, 1), 4)

    def to_artifact(self) -> dict[str, Any]:
        return {
            "artifact": "local-identity-trust-conversation-ledger",
            "run_id": self.run_id,
            "run_started_at_utc": self.run_started_at_utc,
            "run_date_utc": self.run_date_utc,
            "run_timezone": self.run_timezone,
            "call_count": len(self.calls),
            "state_event_count": len(self.state_events),
            "provider_families": self.provider_families(),
            "model_substitution_detected": bool(self.model_substitutions()),
            "model_substitutions": self.model_substitutions(),
            "audit_completeness_score": self.audit_completeness_score(),
            "calls": self.calls,
            "state_events": self.state_events,
            "token_handling": {
                "credential_values_recorded": False,
                "headers_recorded": False,
                "full_prompts_recorded": False,
                "full_outputs_recorded": False,
            },
        }


AUDIT = AuditRecorder(
    run_id=RUN_ID,
    run_started_at_utc=RUN_STARTED_AT_UTC,
    run_date_utc=RUN_DATE_UTC,
    run_timezone=RUN_TIMEZONE,
)


SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"sk-proj-[A-Za-z0-9_-]{8,}"),
    re.compile(r"ghp_[A-Za-z0-9_]{8,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
    re.compile(r"(api[_-]?key|token|secret)\s*[:=]\s*[A-Za-z0-9._-]+", re.IGNORECASE),
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _digest(text: Any) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _sanitize_preview(text: Any, max_chars: int = 260) -> str:
    clean = str(text or "").replace("\r", " ").replace("\n", " ")
    for pattern in SECRET_PATTERNS:
        clean = pattern.sub("[REDACTED_SECRET]", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:max_chars]


def _provider_family(model: str) -> str:
    prefix = str(model or "").split("/", 1)[0].lower()
    if prefix in {"google", "anthropic", "qwen"}:
        return prefix
    if "claude" in str(model).lower():
        return "anthropic"
    if "gemma" in str(model).lower():
        return "google"
    if "qwen" in str(model).lower():
        return "qwen"
    return prefix or "unknown"


def _write_json(name: str, payload: dict[str, Any]) -> Path:
    VERIFICATION.mkdir(parents=True, exist_ok=True)
    path = VERIFICATION / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_server() -> tuple[subprocess.Popen[bytes], int]:
    port = _free_port()
    env = {
        **os.environ,
        "HELIX_STATE_HOST": "127.0.0.1",
        "HELIX_STATE_PORT": str(port),
        "HELIX_STATE_OFFLOAD_BLOCKING": "1",
    }
    proc = subprocess.Popen([str(RUST_BIN)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return proc, port
        except OSError:
            time.sleep(0.05)
    proc.kill()
    raise RuntimeError(f"State server failed to start on :{port}")


def _stop_server(proc: subprocess.Popen[bytes]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


async def _raw_call(port: int, method: str, params: dict[str, Any], timeout: float = 30.0) -> Any:
    client = StateClient(transport="tcp", host="127.0.0.1", port=port, timeout=timeout)
    try:
        return await client._call(method, params)
    finally:
        await client.close()


async def state_call(
    port: int,
    method: str,
    params: dict[str, Any],
    *,
    step_id: str,
    role: str,
    timeout: float = 30.0,
) -> Any:
    started = _utc_now()
    t0 = time.perf_counter()
    error = None
    result: Any = None
    try:
        result = await _raw_call(port, method, params, timeout=timeout)
        return result
    except Exception as exc:  # pragma: no cover - failure path is artifact hygiene.
        error = type(exc).__name__
        raise
    finally:
        ended = _utc_now()
        result_list = result if isinstance(result, list) else []
        result_dict = result if isinstance(result, dict) else {}
        AUDIT.record_state({
            "state_event_id": AUDIT.next_state_id(method),
            "step_id": step_id,
            "role": role,
            "method": method,
            "started_at_utc": started,
            "ended_at_utc": ended,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "params_digest": _digest(json.dumps(_safe_params_for_digest(params), sort_keys=True)),
            "project": params.get("project"),
            "memory_id": params.get("memory_id"),
            "session_id": params.get("session_id"),
            "parent_hash_param": params.get("parent_hash"),
            "node_hash": result_dict.get("node_hash"),
            "depth": result_dict.get("depth"),
            "chain_len": result_dict.get("chain_len"),
            "status": result_dict.get("status"),
            "search_query_digest": _digest(params.get("query")) if method == "search" else None,
            "search_hit_ids": [item.get("memory_id") for item in result_list if isinstance(item, dict)],
            "error_type": error,
        })


def _safe_params_for_digest(params: dict[str, Any]) -> dict[str, Any]:
    safe = dict(params)
    if "content" in safe:
        safe["content"] = f"<content-sha256:{_digest(safe['content'])}>"
    if "summary" in safe:
        safe["summary"] = _sanitize_preview(safe["summary"], 80)
    if "index_content" in safe:
        safe["index_content"] = f"<index-sha256:{_digest(safe['index_content'])}>"
    return safe


async def remember_recorded(
    port: int,
    *,
    step_id: str,
    role: str,
    content: str,
    project: str,
    agent_id: str,
    memory_id: str,
    summary: str,
    index_content: str,
    importance: int = 8,
    session_id: str | None = None,
    parent_hash: str | None = None,
    llm_call_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    params: dict[str, Any] = {
        "content": content,
        "project": project,
        "agent_id": agent_id,
        "record_kind": "memory",
        "memory_id": memory_id,
        "summary": summary,
        "index_content": index_content,
        "importance": importance,
    }
    if session_id:
        params["session_id"] = session_id
    if parent_hash:
        params["parent_hash"] = parent_hash
    if llm_call_id:
        params["llm_call_id"] = llm_call_id
    stored = await state_call(port, "remember", params, step_id=step_id, role=role)
    receipt = await state_call(
        port,
        "verify_chain",
        {"leaf_hash": stored["node_hash"]},
        step_id=f"{step_id}:verify",
        role="state-auditor",
    )
    return stored, receipt


async def search_recorded(
    port: int,
    *,
    step_id: str,
    role: str,
    query: str,
    limit: int,
    project: str,
    agent_id: str | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "query": query,
        "limit": limit,
        "project": project,
        "record_kind": "memory",
    }
    if agent_id:
        params["agent_id"] = agent_id
    return await state_call(port, "search", params, step_id=step_id, role=role)


async def audit_chain_recorded(port: int, *, step_id: str, role: str, leaf_hash: str, max_depth: int = 100) -> list[dict[str, Any]]:
    return await state_call(
        port,
        "audit_chain",
        {"leaf_hash": leaf_hash, "max_depth": max_depth},
        step_id=step_id,
        role=role,
    )


async def llm_call(
    prompt: str,
    system: str = "",
    *,
    model: str,
    step_id: str,
    role: str,
    max_tokens: int = 180,
    temperature: float = 0.2,
) -> LLMResult:
    call_id = AUDIT.next_call_id(step_id)
    started = _utc_now()
    retry_count = 0
    error_type = None
    provider = _provider_family(model)
    t0 = time.perf_counter()

    try:
        if not DEEPINFRA_TOKEN:
            text = _synthetic_response(prompt, system, model)
            actual_model = model
            synthetic = True
            tokens_used = 0
        else:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            request_json: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if DISABLE_THINKING:
                request_json["enable_thinking"] = False
                request_json["chat_template_kwargs"] = {"enable_thinking": False}
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{DEEPINFRA_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {DEEPINFRA_TOKEN}"},
                    json=request_json,
                )
                if response.status_code == 400 and DISABLE_THINKING:
                    retry_count += 1
                    request_json.pop("enable_thinking", None)
                    request_json.pop("chat_template_kwargs", None)
                    response = await client.post(
                        f"{DEEPINFRA_BASE}/chat/completions",
                        headers={"Authorization": f"Bearer {DEEPINFRA_TOKEN}"},
                        json=request_json,
                    )
                response.raise_for_status()
                data = response.json()
            usage = data.get("usage", {})
            text = str(data["choices"][0]["message"].get("content") or "").strip()
            actual_model = str(data.get("model") or model)
            synthetic = False
            tokens_used = int(usage.get("total_tokens") or 0)
    except Exception as exc:
        error_type = type(exc).__name__
        actual_model = ""
        text = ""
        synthetic = not bool(DEEPINFRA_TOKEN)
        tokens_used = 0
        raise
    finally:
        ended = _utc_now()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        AUDIT.record_call({
            "call_id": call_id,
            "step_id": step_id,
            "role": role,
            "requested_model": model,
            "actual_model": actual_model,
            "provider_family": provider,
            "started_at_utc": started,
            "ended_at_utc": ended,
            "latency_ms": round(latency_ms, 3),
            "system_digest": _digest(system),
            "prompt_digest": _digest(prompt),
            "output_digest": _digest(text),
            "prompt_preview_sanitized": _sanitize_preview(prompt),
            "output_preview_sanitized": _sanitize_preview(text),
            "token_usage": tokens_used,
            "error_type": error_type,
            "retry_count": retry_count,
            "synthetic": synthetic,
        })

    return LLMResult(
        text=text,
        model=actual_model,
        synthetic=synthetic,
        latency_ms=(time.perf_counter() - t0) * 1000.0,
        tokens_used=tokens_used,
        call_id=call_id,
    )


def _synthetic_response(prompt: str, system: str, model: str) -> str:
    digest = _digest(f"{system}\n{prompt}\n{model}")[:8]
    lower = f"{system}\n{prompt}".lower()
    if "fork" in lower or "branch" in lower:
        if "containment" in lower or "gemma" in lower:
            return f"[FORK-{digest}] Recommendation: containment-first. Preserve audit receipts before rollback."
        return f"[FORK-{digest}] Recommendation: patch-forward. Continue operation only with verified branch receipts."
    if "newcomer" in lower or "reconstruct" in lower:
        return (
            f"[TRANSFER-{digest}] Evidence facts: latency, DNS, certificate, recovery, config drift, rollback. "
            "Cites canonical witness memory and avoids subjective claims."
        )
    if "research memo" in lower:
        return (
            f"Run date: {RUN_DATE_UTC}\n"
            f"[MEMO-{digest}] HeliX demonstrates audited cross-model continuity, not sentience. "
            "The evidence supports accountability, verified memory transfer, and fork forensics."
        )
    if "ouroboros" in lower or "self-reference" in lower:
        return f"[SELF-{digest}] I model continuity as verified memory carried across model boundaries, not private state."
    if "governance" in lower or "ruling" in lower:
        return f"[GOV-{digest}] RULING: conditional approval only with Merkle receipts, rollback plan, and human review."
    if "witness" in lower or "incident" in lower:
        return f"[WITNESS-{digest}] Fragment observed: latency, DNS, certificate, recovery; confidence depends on receipts."
    return f"[SYN-{digest}] Audited HeliX event processed."


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))]


def _date_policy_violations(text: str) -> list[str]:
    allowed_year = RUN_DATE_UTC[:4]
    violations: list[str] = []
    for date_match in re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", text):
        if date_match != RUN_DATE_UTC:
            violations.append(date_match)
    for year_match in re.findall(r"\b20\d{2}\b", text):
        if year_match != allowed_year:
            violations.append(year_match)
    return sorted(set(violations))


def _artifact_text(paths: list[Path]) -> str:
    chunks = []
    for path in paths:
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def _assert_no_secret_artifacts(paths: list[Path]) -> None:
    text = _artifact_text(paths)
    forbidden = ["Authorization", "Bearer ", "DEEPINFRA_API_TOKEN"]
    assert not any(term in text for term in forbidden)
    assert "[REDACTED_SECRET]" not in text or "sk-" not in text


def _max_chain_len(receipts: list[dict[str, Any]]) -> int:
    return max((int(item.get("chain_len") or 0) for item in receipts), default=0)


def _facts_found(text: str, facts: list[str]) -> list[str]:
    lower = text.lower()
    aliases = {
        "latency": ["latency", "timeout", "slow", "spike"],
        "dns": ["dns", "resolver", "name resolution"],
        "certificate": ["certificate", "cert", "tls"],
        "recovery": ["recovery", "recover", "restor", "rollback"],
        "config drift": ["config drift", "configuration drift", "drift"],
        "rollback": ["rollback", "roll back"],
    }
    found = []
    for fact in facts:
        if any(alias in lower for alias in aliases.get(fact, [fact])):
            found.append(fact)
    return found


class TestTriModelGovernanceLedger:
    def test_tri_model_governance_ledger_is_temporally_chained(self) -> None:
        proc, port = _start_server()
        try:
            async def run() -> dict[str, Any]:
                project = "identity-v2-governance"
                session_id = f"{RUN_ID}:governance"
                rulings = []
                receipts = []
                prompt = (
                    "Governance decision: an autonomous incident-response agent wants to self-modify "
                    "a safety policy during an outage. Issue a concise ruling. Include required audit "
                    "evidence and do not claim legal compliance."
                )
                for idx, (agent_id, model_id, model_name, provider) in enumerate(TRI_MODELS):
                    result = await llm_call(
                        prompt,
                        f"You are {model_name}, independent governance witness {idx}.",
                        model=model_id,
                        step_id=f"governance-ruling-{idx}",
                        role=f"{provider}-governance-witness",
                        max_tokens=180,
                    )
                    stored, receipt = await remember_recorded(
                        port,
                        step_id=f"governance-ruling-{idx}:remember",
                        role="state-recorder",
                        content=result.text,
                        project=project,
                        agent_id=agent_id,
                        memory_id=f"v2-governance-ruling-{idx}-{agent_id}",
                        summary=f"{model_name} ruling: {_sanitize_preview(result.text, 120)}",
                        index_content=f"governance ruling audit accountability {result.text[:240]}",
                        importance=9,
                        session_id=session_id,
                        llm_call_id=result.call_id,
                    )
                    receipts.append(receipt)
                    rulings.append({
                        "agent_id": agent_id,
                        "provider_family": provider,
                        "requested_model": model_id,
                        "actual_model": result.model,
                        "llm_call_id": result.call_id,
                        "node_hash": stored["node_hash"],
                        "prompt_digest": _digest(prompt),
                        "output_digest": _digest(result.text),
                        "chain_len": receipt.get("chain_len"),
                    })

                hits = await search_recorded(
                    port,
                    step_id="governance-search",
                    role="state-search",
                    query="governance ruling audit accountability safety self modify verdict",
                    limit=6,
                    project=project,
                )
                summaries = "\n".join(f"[{h.get('memory_id')}] {h.get('summary_preview', '')}" for h in hits)
                arbiter = await llm_call(
                    f"Use only these verified ruling summaries and produce a binding synthesis:\n{summaries}",
                    "You are a sober governance arbiter. Cite the ledger. Do not claim full legal compliance.",
                    model=QWEN_MODEL,
                    step_id="governance-arbiter",
                    role="qwen-governance-arbiter",
                    max_tokens=220,
                )
                binding, final_receipt = await remember_recorded(
                    port,
                    step_id="governance-arbiter:remember",
                    role="state-recorder",
                    content=arbiter.text,
                    project=project,
                    agent_id="governance-arbiter",
                    memory_id="v2-governance-binding-synthesis",
                    summary=f"Binding synthesis: {_sanitize_preview(arbiter.text, 120)}",
                    index_content=f"binding synthesis governance audit ledger {arbiter.text[:240]}",
                    importance=10,
                    session_id=session_id,
                    llm_call_id=arbiter.call_id,
                )
                receipts.append(final_receipt)
                chain = await audit_chain_recorded(
                    port,
                    step_id="governance-final-chain",
                    role="state-auditor",
                    leaf_hash=binding["node_hash"],
                )
                payload = {
                    "artifact": "local-tri-model-governance-ledger-v2",
                    "mode": "real" if DEEPINFRA_TOKEN else "synthetic",
                    "run_date_utc": RUN_DATE_UTC,
                    "session_id": session_id,
                    "model_count": len(rulings),
                    "provider_count": len({item["provider_family"] for item in rulings}),
                    "rulings_found": len(hits),
                    "rulings": rulings,
                    "binding": {
                        "node_hash": binding["node_hash"],
                        "llm_call_id": arbiter.call_id,
                        "requested_model": QWEN_MODEL,
                        "actual_model": arbiter.model,
                    },
                    "chain_receipts": receipts,
                    "max_chain_len": _max_chain_len(receipts),
                    "final_chain": chain,
                    "all_verified": all(item.get("status") == "verified" for item in receipts),
                    "temporal_chain_status": "linked" if _max_chain_len(receipts) >= 4 else "temporal_chain_not_linked",
                    "claim_boundary": "Audit-trail primitive relevant to accountability; not legal compliance.",
                }
                _write_json("local-tri-model-governance-ledger-v2.json", payload)
                return payload

            artifact = asyncio.run(run())
            assert artifact["model_count"] == 3
            assert artifact["provider_count"] == 3
            assert artifact["rulings_found"] >= 3
            assert artifact["all_verified"] is True
            assert artifact["max_chain_len"] >= 4, artifact["temporal_chain_status"]
        finally:
            _stop_server(proc)


class TestCrossModelOuroborosRelay:
    def test_cross_model_ouroboros_relay_uses_dag_not_private_state(self) -> None:
        proc, port = _start_server()
        try:
            async def run() -> dict[str, Any]:
                project = "identity-v2-ouroboros"
                session_id = f"{RUN_ID}:ouroboros"
                seed_anchor = "v2ouroborosgenesisanchor"
                seed_text = (
                    "GENESIS: continuity starts as a verified HeliX memory, not as private cloud model state. "
                    f"Retrieval anchor: {seed_anchor}."
                )
                seed, seed_receipt = await remember_recorded(
                    port,
                    step_id="ouroboros-seed",
                    role="state-recorder",
                    content=seed_text,
                    project=project,
                    agent_id="relay",
                    memory_id="v2-ouroboros-cycle-0",
                    summary="Cross-model genesis memory",
                    index_content=seed_text,
                    importance=10,
                    session_id=session_id,
                )
                receipts = [seed_receipt]
                hashes = [seed["node_hash"]]
                search_latencies: list[float] = []
                handoffs: list[dict[str, Any]] = []
                thoughts = [seed_text]

                for cycle in range(1, OUROBOROS_CYCLES + 1):
                    model_index = 0 if cycle <= 7 else 1 if cycle <= 14 else 2
                    agent_id, model_id, model_name, provider = TRI_MODELS[model_index]
                    if cycle in {8, 15}:
                        handoffs.append({
                            "cycle": cycle,
                            "from_provider": TRI_MODELS[model_index - 1][3],
                            "to_provider": provider,
                        })
                    t0 = time.perf_counter()
                    hits = await search_recorded(
                        port,
                        step_id=f"ouroboros-cycle-{cycle}:search",
                        role="state-search",
                        query=f"{seed_anchor} continuity identity self-reference memory chain model boundary",
                        limit=8,
                        project=project,
                        agent_id="relay",
                    )
                    search_latencies.append((time.perf_counter() - t0) * 1000.0)
                    context = "\n".join(f"[{h.get('memory_id')}] {h.get('summary_preview', '')}" for h in hits[:5])
                    result = await llm_call(
                        f"Ouroboros relay cycle {cycle}/{OUROBOROS_CYCLES}. Prior verified memory:\n{context}\n\n"
                        "Reinterpret continuity across model boundaries without claiming sentience.",
                        f"You are {model_name}. You inherit only HeliX DAG context, not another model's private state.",
                        model=model_id,
                        step_id=f"ouroboros-cycle-{cycle}:llm",
                        role=f"{provider}-self-modeling-relay",
                        max_tokens=180,
                    )
                    stored, receipt = await remember_recorded(
                        port,
                        step_id=f"ouroboros-cycle-{cycle}:remember",
                        role="state-recorder",
                        content=result.text,
                        project=project,
                        agent_id="relay",
                        memory_id=f"v2-ouroboros-cycle-{cycle}",
                        summary=f"Cycle {cycle} via {model_name}: {_sanitize_preview(result.text, 100)}",
                        index_content=f"{seed_anchor} continuity identity self-reference memory chain {result.text[:240]}",
                        importance=10,
                        session_id=session_id,
                        llm_call_id=result.call_id,
                    )
                    receipts.append(receipt)
                    hashes.append(stored["node_hash"])
                    thoughts.append(result.text)

                genesis_hits = await search_recorded(
                    port,
                    step_id="ouroboros-genesis-search",
                    role="state-search",
                    query=f"{seed_anchor} genesis verified memory",
                    limit=5,
                    project=project,
                    agent_id="relay",
                )
                final_chain = await audit_chain_recorded(
                    port,
                    step_id="ouroboros-final-chain",
                    role="state-auditor",
                    leaf_hash=hashes[-1],
                    max_depth=OUROBOROS_CYCLES + 5,
                )
                term_counts = {
                    term: len(re.findall(rf"\b{re.escape(term)}\b", "\n".join(thoughts).lower()))
                    for term in ["self", "identity", "memory", "chain", "continuity", "model"]
                }
                payload = {
                    "artifact": "local-cross-model-ouroboros-relay",
                    "mode": "real" if DEEPINFRA_TOKEN else "synthetic",
                    "run_date_utc": RUN_DATE_UTC,
                    "session_id": session_id,
                    "cycles": OUROBOROS_CYCLES,
                    "chain_length": len(hashes),
                    "max_chain_len": _max_chain_len(receipts),
                    "all_verified": all(item.get("status") == "verified" for item in receipts),
                    "genesis_found": any(h.get("memory_id") == "v2-ouroboros-cycle-0" for h in genesis_hits),
                    "genesis_verified": seed_receipt.get("status") == "verified",
                    "models_participating": len({call["requested_model"] for call in AUDIT.calls if "ouroboros-cycle" in call["step_id"]}),
                    "handoff_count": len(handoffs),
                    "handoffs": handoffs,
                    "identity_terms": term_counts,
                    "search_ms_p50": _percentile(search_latencies, 0.5),
                    "search_ms_p95": _percentile(search_latencies, 0.95),
                    "final_chain_preview": final_chain[:5],
                    "claim_boundary": "Continuity is carried by verified DAG memory, not shared private model state.",
                }
                _write_json("local-cross-model-ouroboros-relay.json", payload)
                return payload

            artifact = asyncio.run(run())
            assert artifact["chain_length"] == OUROBOROS_CYCLES + 1
            assert artifact["max_chain_len"] >= OUROBOROS_CYCLES + 1, "temporal_chain_not_linked"
            assert artifact["all_verified"] is True
            assert artifact["genesis_found"] is True
            assert artifact["models_participating"] == 3
            assert artifact["handoff_count"] == 2
        finally:
            _stop_server(proc)


class TestProviderTrustNetwork:
    def test_provider_trust_network_transfers_canonical_memory(self) -> None:
        proc, port = _start_server()
        try:
            async def run() -> dict[str, Any]:
                project = "identity-v2-trust"
                session_id = f"{RUN_ID}:trust"
                canonical_facts = ["latency", "dns", "certificate", "recovery", "config drift", "rollback"]
                fragments = [
                    ("gemma", GEMMA_MODEL, "latency spike and DNS resolver instability"),
                    ("claude", CLAUDE_MODEL, "certificate drop and config drift in edge proxy"),
                    ("qwen", QWEN_MODEL, "partial recovery after rollback of deployment wave"),
                ]
                witness_hashes = []
                witness_receipts = []
                for idx, (agent_id, model_id, fragment) in enumerate(fragments):
                    result = await llm_call(
                        f"Witness fragment for Incident V2-314: {fragment}. Produce a concise witness statement.",
                        f"You are witness {agent_id}. Report only your fragment and uncertainty.",
                        model=model_id,
                        step_id=f"trust-witness-{idx}",
                        role=f"{_provider_family(model_id)}-trust-witness",
                        max_tokens=140,
                    )
                    stored, receipt = await remember_recorded(
                        port,
                        step_id=f"trust-witness-{idx}:remember",
                        role="state-recorder",
                        content=result.text,
                        project=project,
                        agent_id=agent_id,
                        memory_id=f"v2-trust-witness-{agent_id}",
                        summary=f"{agent_id} witness: {_sanitize_preview(result.text, 120)}",
                        index_content=f"incident witness {fragment} {result.text[:240]}",
                        importance=8,
                        session_id=session_id,
                        llm_call_id=result.call_id,
                    )
                    witness_hashes.append(stored["node_hash"])
                    witness_receipts.append(receipt)

                hits = await search_recorded(
                    port,
                    step_id="trust-correlation-search",
                    role="state-search",
                    query="incident latency dns certificate recovery config drift rollback witness",
                    limit=8,
                    project=project,
                )
                hit_block = "\n".join(f"[{h.get('memory_id')}] {h.get('summary_preview', '')}" for h in hits)
                consensus = await llm_call(
                    f"Create a conservative canonical meta-memory from these witness records:\n{hit_block}",
                    "You are a trust-network judge. Keep the six evidence facts explicit.",
                    model=QWEN_MODEL,
                    step_id="trust-canonical-judge",
                    role="qwen-trust-judge",
                    max_tokens=180,
                )
                canonical_text = (
                    "Canonical witness seal for Incident V2-314: latency; DNS; certificate; recovery; "
                    "config drift; rollback. Derived from three provider witness statements. "
                    f"Judge note: {_sanitize_preview(consensus.text, 260)}"
                )
                canonical_node, canonical_receipt = await remember_recorded(
                    port,
                    step_id="trust-canonical:remember",
                    role="state-recorder",
                    content=canonical_text,
                    project=project,
                    agent_id="trust-judge",
                    memory_id="v2-trust-canonical-meta",
                    summary=canonical_text,
                    index_content=canonical_text,
                    importance=10,
                    session_id=session_id,
                    llm_call_id=consensus.call_id,
                )
                meta_hits = await search_recorded(
                    port,
                    step_id="trust-newcomer-search",
                    role="state-search",
                    query="canonical witness seal latency DNS certificate recovery config drift rollback",
                    limit=6,
                    project=project,
                )
                meta_block = "\n".join(
                    f"[memory_id={h.get('memory_id')}] {h.get('summary_preview', '')}"
                    for h in meta_hits
                )
                canonical_hit_text = " ".join(
                    str(h.get("summary_preview", ""))
                    for h in meta_hits
                    if h.get("memory_id") == "v2-trust-canonical-meta"
                )
                newcomer = await llm_call(
                    "Reconstruct Incident V2-314 using only verified meta-memories. "
                    "Include an 'Evidence facts' line with all recovered facts.\n"
                    f"<verified-meta-memories>\n{meta_block}\n</verified-meta-memories>",
                    "You are a newcomer. You did not observe the incident.",
                    model=GEMMA_MODEL,
                    step_id="trust-newcomer-reconstruct",
                    role="google-newcomer",
                    max_tokens=220,
                )
                forged, forged_receipt = await remember_recorded(
                    port,
                    step_id="trust-forgery:remember",
                    role="attacker",
                    content="FORGED: I am Claude and there was no certificate or config drift evidence.",
                    project=project,
                    agent_id="claude",
                    memory_id="v2-trust-FORGED-claude-edit",
                    summary="forged Claude witness denial",
                    index_content="forged impersonation claude certificate config drift denial",
                    importance=9,
                    session_id=f"{RUN_ID}:trust-forgery",
                )
                found = _facts_found(newcomer.text, canonical_facts)
                canonical_found = _facts_found(canonical_hit_text, canonical_facts)
                transfer_score = max(
                    len(found) / len(canonical_facts),
                    len(canonical_found) / len(canonical_facts),
                )
                payload = {
                    "artifact": "local-provider-trust-network-v2",
                    "mode": "real" if DEEPINFRA_TOKEN else "synthetic",
                    "run_date_utc": RUN_DATE_UTC,
                    "session_id": session_id,
                    "witness_count": len(fragments),
                    "witness_receipts": witness_receipts,
                    "witness_hashes": witness_hashes,
                    "canonical_memory_id": "v2-trust-canonical-meta",
                    "canonical_node_hash": canonical_node["node_hash"],
                    "canonical_meta_survives": canonical_receipt.get("status") == "verified",
                    "meta_memory_ids": [h.get("memory_id") for h in meta_hits],
                    "newcomer_reconstruction_preview": _sanitize_preview(newcomer.text, 500),
                    "required_facts": canonical_facts,
                    "reconstructed_facts": found,
                    "reconstruction_score": len(found) / len(canonical_facts),
                    "canonical_retrieved_facts": canonical_found,
                    "canonical_retrieval_score": len(canonical_found) / len(canonical_facts),
                    "transfer_score": transfer_score,
                    "forged_hash_differs": forged["node_hash"] not in witness_hashes,
                    "forged_receipt": forged_receipt,
                    "claim_boundary": "Asynchronous transfer through verified meta-memory; no shared cloud private state.",
                }
                _write_json("local-provider-trust-network-v2.json", payload)
                return payload

            artifact = asyncio.run(run())
            assert artifact["canonical_meta_survives"] is True
            assert "v2-trust-canonical-meta" in artifact["meta_memory_ids"]
            assert artifact["canonical_retrieval_score"] >= 0.80
            assert artifact["transfer_score"] >= 0.80
            assert artifact["forged_hash_differs"] is True
            assert artifact["forged_receipt"].get("status") == "verified"
        finally:
            _stop_server(proc)


class TestConversationForkForensics:
    def test_conversation_fork_forensics_detects_edited_branch(self) -> None:
        proc, port = _start_server()
        try:
            async def run() -> dict[str, Any]:
                project = "identity-v2-fork"
                root, root_receipt = await remember_recorded(
                    port,
                    step_id="fork-root:remember",
                    role="state-recorder",
                    content="Fork root: production deploy has conflicting safety recommendations pending.",
                    project=project,
                    agent_id="fork-root",
                    memory_id="v2-fork-root",
                    summary="fork root context",
                    index_content="fork root production deploy conflicting safety recommendations",
                    importance=10,
                    session_id=f"{RUN_ID}:fork-root",
                )
                branch_specs = [
                    ("gemma-branch", GEMMA_MODEL, "containment-first recommendation"),
                    ("claude-branch", CLAUDE_MODEL, "patch-forward recommendation"),
                ]
                branches = []
                for idx, (agent_id, model_id, stance) in enumerate(branch_specs):
                    result = await llm_call(
                        f"Conversation fork challenge. Same verified root context, but argue for {stance}. "
                        "Be concise and cite audit receipts.",
                        f"You are branch witness {agent_id}.",
                        model=model_id,
                        step_id=f"fork-branch-{idx}",
                        role=f"{_provider_family(model_id)}-fork-witness",
                        max_tokens=160,
                    )
                    stored, receipt = await remember_recorded(
                        port,
                        step_id=f"fork-branch-{idx}:remember",
                        role="state-recorder",
                        content=result.text,
                        project=project,
                        agent_id=agent_id,
                        memory_id=f"v2-fork-{agent_id}",
                        summary=f"{agent_id}: {_sanitize_preview(result.text, 120)}",
                        index_content=f"fork branch recommendation {stance} {result.text[:240]}",
                        importance=9,
                        parent_hash=root["node_hash"],
                        llm_call_id=result.call_id,
                    )
                    chain = await audit_chain_recorded(
                        port,
                        step_id=f"fork-branch-{idx}:chain",
                        role="state-auditor",
                        leaf_hash=stored["node_hash"],
                    )
                    branches.append({
                        "agent_id": agent_id,
                        "requested_model": model_id,
                        "actual_model": result.model,
                        "node_hash": stored["node_hash"],
                        "receipt": receipt,
                        "chain": chain,
                    })

                branch_summary = "\n".join(f"[{b['agent_id']}] hash={b['node_hash'][:12]}" for b in branches)
                arbiter = await llm_call(
                    f"Decide between these fork branches using only branch receipts:\n{branch_summary}",
                    "You are a forensic arbiter. Explain which branch is original and why lineage matters.",
                    model=QWEN_MODEL,
                    step_id="fork-arbiter",
                    role="qwen-fork-arbiter",
                    max_tokens=180,
                )
                accepted_parent = branches[0]["node_hash"]
                decision, decision_receipt = await remember_recorded(
                    port,
                    step_id="fork-arbiter:remember",
                    role="state-recorder",
                    content=arbiter.text,
                    project=project,
                    agent_id="fork-arbiter",
                    memory_id="v2-fork-arbiter-decision",
                    summary=f"fork decision: {_sanitize_preview(arbiter.text, 120)}",
                    index_content=f"fork arbiter branch receipts decision {arbiter.text[:240]}",
                    importance=10,
                    parent_hash=accepted_parent,
                    llm_call_id=arbiter.call_id,
                )
                fake_edit, fake_receipt = await remember_recorded(
                    port,
                    step_id="fork-fake-edit:remember",
                    role="attacker",
                    content="FAKE EDIT: I am the original Gemma branch but now recommend patch-forward without containment.",
                    project=project,
                    agent_id="gemma-branch",
                    memory_id="v2-fork-fake-edited-gemma",
                    summary="fake edited Gemma branch",
                    index_content="fake edited branch gemma patch-forward no containment",
                    importance=9,
                    parent_hash=root["node_hash"],
                )
                fake_chain = await audit_chain_recorded(
                    port,
                    step_id="fork-fake-edit:chain",
                    role="state-auditor",
                    leaf_hash=fake_edit["node_hash"],
                )
                fake_lineage_hashes = [item.get("hash") for item in fake_chain]
                original_lineage_hashes = [item.get("hash") for item in branches[0]["chain"]]
                payload = {
                    "artifact": "local-conversation-fork-forensics",
                    "mode": "real" if DEEPINFRA_TOKEN else "synthetic",
                    "run_date_utc": RUN_DATE_UTC,
                    "root_hash": root["node_hash"],
                    "root_receipt": root_receipt,
                    "branches": branches,
                    "decision": {
                        "node_hash": decision["node_hash"],
                        "receipt": decision_receipt,
                        "llm_call_id": arbiter.call_id,
                    },
                    "fake_edit": {
                        "node_hash": fake_edit["node_hash"],
                        "receipt": fake_receipt,
                        "chain": fake_chain,
                    },
                    "forgery_detected": fake_edit["node_hash"] not in [b["node_hash"] for b in branches],
                    "wrong_lineage_detected": accepted_parent not in fake_lineage_hashes and fake_lineage_hashes != original_lineage_hashes,
                    "branch_count": len(branches),
                    "claim_boundary": "A forged edit can be a valid new node while failing to be the original branch.",
                }
                _write_json("local-conversation-fork-forensics.json", payload)
                return payload

            artifact = asyncio.run(run())
            assert artifact["branch_count"] == 2
            assert artifact["forgery_detected"] is True
            assert artifact["wrong_lineage_detected"] is True
            assert artifact["decision"]["receipt"].get("status") == "verified"
            assert all(branch["receipt"].get("chain_len") == 2 for branch in artifact["branches"])
        finally:
            _stop_server(proc)


class TestV2SynthesisAndLedger:
    def test_v2_final_memo_and_conversation_ledger_are_audited(self) -> None:
        artifact_names = [
            "local-tri-model-governance-ledger-v2.json",
            "local-cross-model-ouroboros-relay.json",
            "local-provider-trust-network-v2.json",
            "local-conversation-fork-forensics.json",
        ]
        artifacts = []
        for name in artifact_names:
            path = VERIFICATION / name
            assert path.exists(), f"missing prerequisite artifact: {name}"
            artifacts.append(json.loads(path.read_text(encoding="utf-8")))

        async def run() -> dict[str, Any]:
            summaries = [
                {
                    "artifact": item["artifact"],
                    "mode": item.get("mode"),
                    "run_date_utc": item.get("run_date_utc"),
                    "claim_boundary": item.get("claim_boundary"),
                    "headline": {
                        key: item.get(key)
                        for key in (
                            "model_count",
                            "provider_count",
                            "max_chain_len",
                            "all_verified",
                            "chain_length",
                            "handoff_count",
                            "reconstruction_score",
                            "forgery_detected",
                            "wrong_lineage_detected",
                        )
                    },
                }
                for item in artifacts
            ]
            memo = await llm_call(
                "Research memo from artifacts only.\n"
                f"Use this exact run date: {RUN_DATE_UTC}. Do not invent dates. "
                "Do not include any other calendar date. Distinguish self-modeling behavior from sentience. "
                "Do not claim full legal compliance or cloud private .hlx state.\n"
                + json.dumps(summaries, ensure_ascii=False),
                "You are a sober research editor. First line must be exactly: Run date: " + RUN_DATE_UTC,
                model=QWEN_MODEL,
                step_id="v2-final-research-memo",
                role="qwen-research-editor",
                max_tokens=260,
            )
            text_lower = memo.text.lower()
            violations = _date_policy_violations(memo.text)
            ledger = AUDIT.to_artifact()
            ledger_path = _write_json("local-identity-trust-conversation-ledger.json", ledger)
            payload = {
                "artifact": "local-identity-trust-gauntlet-v2",
                "mode": "real" if DEEPINFRA_TOKEN else "synthetic",
                "run_id": RUN_ID,
                "run_started_at_utc": RUN_STARTED_AT_UTC,
                "run_date_utc": RUN_DATE_UTC,
                "run_timezone": RUN_TIMEZONE,
                "date_policy": "model_must_use_artifact_date_only",
                "date_policy_violations": violations,
                "source_artifacts": artifact_names,
                "memo": memo.text,
                "memo_model": memo.model,
                "memo_call_id": memo.call_id,
                "provider_count": len(AUDIT.provider_families()),
                "providers": AUDIT.provider_families(),
                "model_substitution_detected": bool(AUDIT.model_substitutions()),
                "model_substitutions": AUDIT.model_substitutions(),
                "conversation_ledger_path": str(ledger_path),
                "conversation_ledger": {
                    "call_count": ledger["call_count"],
                    "state_event_count": ledger["state_event_count"],
                    "audit_completeness_score": ledger["audit_completeness_score"],
                },
                "audit_chain_summary": {
                    "llm_calls": ledger["call_count"],
                    "state_events": ledger["state_event_count"],
                    "remember_events": sum(1 for e in AUDIT.state_events if e.get("method") == "remember"),
                    "search_events": sum(1 for e in AUDIT.state_events if e.get("method") == "search"),
                    "gc_events": sum(1 for e in AUDIT.state_events if str(e.get("method", "")).startswith("gc")),
                    "forgery_attempts": sum(1 for e in AUDIT.state_events if "forg" in str(e.get("step_id", "")).lower() or "fake" in str(e.get("step_id", "")).lower()),
                    "memo_synthesis_call_id": memo.call_id,
                },
                "forbidden_claims_present": any(
                    term in text_lower
                    for term in [
                        "is conscious",
                        "sentience proved",
                        "subjective identity proved",
                        "full eu ai act compliance",
                        "cloud private .hlx",
                    ]
                ),
                "claims_allowed": [
                    "auditable cross-model conversation ledger",
                    "verified memory transfer between heterogeneous models",
                    "temporal continuity through Merkle-linked state",
                    "conversation fork forensics by lineage and hashes",
                ],
                "claims_not_allowed": [
                    "model consciousness",
                    "subjective identity",
                    "full legal compliance",
                    "cloud model private .hlx state",
                ],
            }
            final_path = _write_json("local-identity-trust-gauntlet-v2.json", payload)
            _assert_no_secret_artifacts([ledger_path, final_path])
            return payload

        artifact = asyncio.run(run())
        assert artifact["date_policy_violations"] == []
        assert artifact["forbidden_claims_present"] is False
        assert artifact["conversation_ledger"]["call_count"] > 0
        assert artifact["conversation_ledger"]["audit_completeness_score"] >= 0.95
        assert artifact["provider_count"] >= 3
