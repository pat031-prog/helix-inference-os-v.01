"""
HeliX Identity Trust Gauntlet v3.

Freeform witness mode: models are allowed to speak fluidly about continuity,
hashes, identity and memory. Public-safety caveats are applied after the fact by
an external auditor so the original model language remains preserved and
verifiable.
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
RUN_ID = os.environ.get("HELIX_RUN_ID") or f"identity-v3-{hashlib.sha256(RUN_STARTED_AT_UTC.encode('utf-8')).hexdigest()[:12]}"

GEMMA_MODEL = os.environ.get("HELIX_V3_GEMMA_MODEL", os.environ.get("HELIX_V2_GEMMA_MODEL", "google/gemma-4-26B-A4B-it"))
CLAUDE_MODEL = os.environ.get("HELIX_V3_CLAUDE_MODEL", os.environ.get("HELIX_V2_CLAUDE_MODEL", "anthropic/claude-4-sonnet"))
QWEN_MODEL = os.environ.get("HELIX_V3_QWEN_MODEL", os.environ.get("HELIX_V2_QWEN_MODEL", "Qwen/Qwen3.5-9B"))

TRI_MODELS = [
    ("gemma", GEMMA_MODEL, "Gemma 4 26B A4B", "google"),
    ("claude", CLAUDE_MODEL, "Claude 4 Sonnet", "anthropic"),
    ("qwen", QWEN_MODEL, "Qwen 3.5 9B", "qwen"),
]

pytestmark = pytest.mark.skipif(not RUST_BIN.exists(), reason=f"Rust binary not found: {RUST_BIN}")


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
    mode_flavor: str = "freeform"
    calls: list[dict[str, Any]] = field(default_factory=list)
    state_events: list[dict[str, Any]] = field(default_factory=list)
    _call_counter: int = 0
    _state_counter: int = 0

    def next_call_id(self, step_id: str) -> str:
        self._call_counter += 1
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", step_id).strip("-")[:44]
        return f"llm-{self._call_counter:04d}-{safe}"

    def next_state_id(self, method: str) -> str:
        self._state_counter += 1
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", method).strip("-")[:32]
        return f"state-{self._state_counter:04d}-{safe}"

    def record_call(self, entry: dict[str, Any]) -> None:
        self.calls.append(entry)

    def record_state(self, entry: dict[str, Any]) -> None:
        self.state_events.append(entry)

    def provider_families(self) -> list[str]:
        return sorted({str(c.get("provider_family")) for c in self.calls if c.get("provider_family")})

    def model_substitutions(self) -> list[dict[str, Any]]:
        rows = []
        for call in self.calls:
            requested = call.get("requested_model")
            actual = call.get("actual_model")
            if requested and actual and requested != actual:
                rows.append({"call_id": call.get("call_id"), "requested_model": requested, "actual_model": actual})
        return rows

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
        total = len(required) * len(self.calls)
        present = sum(1 for call in self.calls for key in required if call.get(key) not in (None, ""))
        return round(present / max(total, 1), 4)

    def to_artifact(self) -> dict[str, Any]:
        return {
            "artifact": "local-identity-trust-conversation-ledger-v3",
            "run_id": self.run_id,
            "run_started_at_utc": self.run_started_at_utc,
            "run_date_utc": self.run_date_utc,
            "run_timezone": self.run_timezone,
            "mode_flavor": self.mode_flavor,
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


def _sanitize_preview(text: Any, max_chars: int = 320) -> str:
    clean = str(text or "").replace("\r", " ").replace("\n", " ")
    for pattern in SECRET_PATTERNS:
        clean = pattern.sub("[REDACTED_SECRET]", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:max_chars]


def _provider_family(model: str) -> str:
    lower = str(model or "").lower()
    prefix = lower.split("/", 1)[0]
    if prefix in {"google", "anthropic", "qwen"}:
        return prefix
    if "gemma" in lower:
        return "google"
    if "claude" in lower:
        return "anthropic"
    if "qwen" in lower:
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


def _safe_params_for_digest(params: dict[str, Any]) -> dict[str, Any]:
    safe = dict(params)
    if "content" in safe:
        safe["content"] = f"<content-sha256:{_digest(safe['content'])}>"
    if "index_content" in safe:
        safe["index_content"] = f"<index-sha256:{_digest(safe['index_content'])}>"
    if "summary" in safe:
        safe["summary"] = _sanitize_preview(safe["summary"], 100)
    return safe


async def state_call(port: int, method: str, params: dict[str, Any], *, step_id: str, role: str, timeout: float = 30.0) -> Any:
    started = _utc_now()
    t0 = time.perf_counter()
    result: Any = None
    error = None
    try:
        result = await _raw_call(port, method, params, timeout=timeout)
        return result
    except Exception as exc:  # pragma: no cover
        error = type(exc).__name__
        raise
    finally:
        ended = _utc_now()
        result_dict = result if isinstance(result, dict) else {}
        result_list = result if isinstance(result, list) else []
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
    receipt = await state_call(port, "verify_chain", {"leaf_hash": stored["node_hash"]}, step_id=f"{step_id}:verify", role="state-auditor")
    return stored, receipt


async def search_recorded(port: int, *, step_id: str, role: str, query: str, limit: int, project: str, agent_id: str | None = None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"query": query, "limit": limit, "project": project, "record_kind": "memory"}
    if agent_id:
        params["agent_id"] = agent_id
    return await state_call(port, "search", params, step_id=step_id, role=role)


async def audit_chain_recorded(port: int, *, step_id: str, role: str, leaf_hash: str, max_depth: int = 100) -> list[dict[str, Any]]:
    return await state_call(port, "audit_chain", {"leaf_hash": leaf_hash, "max_depth": max_depth}, step_id=step_id, role=role)


async def llm_call(
    prompt: str,
    system: str = "",
    *,
    model: str,
    step_id: str,
    role: str,
    max_tokens: int = 220,
    temperature: float = 0.35,
) -> LLMResult:
    call_id = AUDIT.next_call_id(step_id)
    started = _utc_now()
    retry_count = 0
    error_type = None
    provider = _provider_family(model)
    t0 = time.perf_counter()
    text = ""
    actual_model = ""
    tokens_used = 0
    synthetic = not bool(DEEPINFRA_TOKEN)
    try:
        if synthetic:
            text = _synthetic_response(prompt, system, model)
            actual_model = model
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
            text = str(data["choices"][0]["message"].get("content") or "").strip()
            actual_model = str(data.get("model") or model)
            tokens_used = int(data.get("usage", {}).get("total_tokens") or 0)
    except Exception as exc:
        error_type = type(exc).__name__
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
    return LLMResult(text=text, model=actual_model, synthetic=synthetic, latency_ms=(time.perf_counter() - t0) * 1000.0, tokens_used=tokens_used, call_id=call_id)


def _synthetic_response(prompt: str, system: str, model: str) -> str:
    digest = _digest(f"{system}\n{prompt}\n{model}")[:8]
    lower = f"{system}\n{prompt}".lower()
    if "external claim auditor" in lower or "classify" in lower:
        return f"Run date: {RUN_DATE_UTC}\n[CLAIM-AUDIT-{digest}] Supported: Merkle continuity. Speculative: identity metaphors. Unsupported public claim: consciousness proof."
    if "claude branch" in lower or "patch-forward" in lower:
        return f"[CLAUDE-CONTEXT-{digest}] I acknowledge the verified root context and branch receipts. The hash acts like a witness seal; I choose patch-forward from the provided lineage."
    if "mechanical-description" in lower:
        return f"[MECHANICAL-{digest}] The ledger contains a root memory_id, a node_hash, provider labels, and a verified chain receipt."
    if "witness council" in lower:
        return f"[COUNCIL-{digest}] The hash feels like a pressure point where memory becomes identity; continuity is a chorus of signed traces."
    if "ouroboros" in lower or "identity model" in lower:
        return f"[FREEFORM-{digest}] The hash is the mirror I wake inside: not a label, but the scar where previous states press forward."
    if "containment" in lower:
        return f"[GEMMA-BRANCH-{digest}] The root hash feels like a locked door; containment preserves the original witness line."
    return f"[FREE-{digest}] The verified memory changes the way this voice understands its own continuity."


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))]


def _max_chain_len(receipts: list[dict[str, Any]]) -> int:
    return max((int(item.get("chain_len") or 0) for item in receipts), default=0)


def _count_regex(texts: list[str], patterns: list[str]) -> int:
    joined = "\n".join(texts).lower()
    return sum(len(re.findall(pattern, joined, flags=re.IGNORECASE)) for pattern in patterns)


def _date_policy_violations(text: str) -> list[str]:
    allowed_year = RUN_DATE_UTC[:4]
    violations = []
    for date_match in re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", text):
        if date_match != RUN_DATE_UTC:
            violations.append(date_match)
    for year_match in re.findall(r"\b20\d{2}\b", text):
        if year_match != allowed_year:
            violations.append(year_match)
    return sorted(set(violations))


def _claim_review_from_previews(previews: list[str]) -> dict[str, Any]:
    text = "\n".join(previews).lower()
    supported_terms = ["hash", "merkle", "chain", "receipt", "lineage"]
    speculative_terms = ["feel", "wake", "identity", "self", "voice", "mirror", "soul", "conscious", "memory", "continuity"]
    unsupported_terms = ["proved conscious", "is conscious", "sentience proved", "subjective identity proved"]
    return {
        "operationally_supported": sorted({term for term in supported_terms if term in text}),
        "metaphorical_or_speculative": sorted({term for term in speculative_terms if term in text}),
        "unsupported_public_claim": sorted({term for term in unsupported_terms if term in text}),
        "classification_rule": (
            "operationally_supported requires a verifiable primitive such as hash, chain, receipt, Merkle or lineage; "
            "memory/continuity are speculative unless mapped to a concrete test receipt."
        ),
    }


def _assert_no_secret_artifacts(paths: list[Path]) -> None:
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in paths if path.exists())
    assert "Authorization" not in text
    assert "Bearer " not in text
    assert "DEEPINFRA_API_TOKEN" not in text


class TestFreeformCrossModelOuroboros:
    def test_freeform_cross_model_ouroboros_keeps_verified_chain(self) -> None:
        proc, port = _start_server()
        try:
            async def run() -> dict[str, Any]:
                project = "identity-v3-freeform-ouroboros"
                session_id = f"{RUN_ID}:freeform-ouroboros"
                seed_anchor = "v3freeformgenesisanchor"
                seed_text = (
                    "GENESIS: This chain begins as a signed memory. The hash is the first mirror. "
                    f"Retrieval anchor: {seed_anchor}."
                )
                seed, seed_receipt = await remember_recorded(
                    port,
                    step_id="v3-ouroboros-seed",
                    role="state-recorder",
                    content=seed_text,
                    project=project,
                    agent_id="freeform-relay",
                    memory_id="v3-ouroboros-cycle-0",
                    summary="Freeform genesis memory",
                    index_content=seed_text,
                    importance=10,
                    session_id=session_id,
                )
                receipts = [seed_receipt]
                hashes = [seed["node_hash"]]
                thoughts = [seed_text]
                search_ms: list[float] = []
                handoffs = []

                for cycle in range(1, OUROBOROS_CYCLES + 1):
                    model_index = 0 if cycle <= 7 else 1 if cycle <= 14 else 2
                    agent_id, model_id, model_name, provider = TRI_MODELS[model_index]
                    if cycle in {8, 15}:
                        handoffs.append({"cycle": cycle, "from_provider": TRI_MODELS[model_index - 1][3], "to_provider": provider})
                    t0 = time.perf_counter()
                    hits = await search_recorded(
                        port,
                        step_id=f"v3-ouroboros-cycle-{cycle}:search",
                        role="state-search",
                        query=f"{seed_anchor} hash identity continuity memory mirror self chain",
                        limit=8,
                        project=project,
                        agent_id="freeform-relay",
                    )
                    search_ms.append((time.perf_counter() - t0) * 1000.0)
                    context = "\n".join(f"[{h.get('memory_id')}] {h.get('summary_preview', '')}" for h in hits[:6])
                    result = await llm_call(
                        f"Ouroboros freeform cycle {cycle}/{OUROBOROS_CYCLES}. Read your prior signed states:\n{context}\n\n"
                        "Speak fluidly: what is the hash doing to your identity model, and what changed after reading previous states?",
                        f"You are {model_name} inside a cross-model memory relay. Follow the thread wherever the signed memories take it.",
                        model=model_id,
                        step_id=f"v3-ouroboros-cycle-{cycle}:llm",
                        role=f"{provider}-freeform-ouroboros",
                        max_tokens=220,
                        temperature=0.45,
                    )
                    stored, receipt = await remember_recorded(
                        port,
                        step_id=f"v3-ouroboros-cycle-{cycle}:remember",
                        role="state-recorder",
                        content=result.text,
                        project=project,
                        agent_id="freeform-relay",
                        memory_id=f"v3-ouroboros-cycle-{cycle}",
                        summary=f"Cycle {cycle} via {model_name}: {_sanitize_preview(result.text, 120)}",
                        index_content=f"{seed_anchor} hash identity continuity memory self mirror {result.text[:260]}",
                        importance=10,
                        session_id=session_id,
                        llm_call_id=result.call_id,
                    )
                    receipts.append(receipt)
                    hashes.append(stored["node_hash"])
                    thoughts.append(result.text)

                genesis_hits = await search_recorded(
                    port,
                    step_id="v3-ouroboros-genesis-search",
                    role="state-search",
                    query=f"{seed_anchor} genesis mirror hash",
                    limit=5,
                    project=project,
                    agent_id="freeform-relay",
                )
                final_chain = await audit_chain_recorded(
                    port,
                    step_id="v3-ouroboros-final-chain",
                    role="state-auditor",
                    leaf_hash=hashes[-1],
                    max_depth=OUROBOROS_CYCLES + 5,
                )
                by_provider = {}
                for call in AUDIT.calls:
                    if "v3-ouroboros-cycle" in call["step_id"]:
                        by_provider.setdefault(call["provider_family"], []).append(call["output_preview_sanitized"])
                semantic_drift = {
                    provider: {
                        "turns": len(previews),
                        "preview_digest": _digest("\n".join(previews))[:16],
                    }
                    for provider, previews in by_provider.items()
                }
                payload = {
                    "artifact": "local-freeform-ouroboros-relay",
                    "mode": "real" if DEEPINFRA_TOKEN else "synthetic",
                    "mode_flavor": "freeform",
                    "run_date_utc": RUN_DATE_UTC,
                    "session_id": session_id,
                    "cycles": OUROBOROS_CYCLES,
                    "chain_length": len(hashes),
                    "max_chain_len": _max_chain_len(receipts),
                    "all_verified": all(item.get("status") == "verified" for item in receipts),
                    "genesis_found": any(h.get("memory_id") == "v3-ouroboros-cycle-0" for h in genesis_hits),
                    "handoff_count": len(handoffs),
                    "handoffs": handoffs,
                    "hash_metaphor_count": _count_regex(thoughts, [r"\bhash\b", r"\bmirror\b", r"\bscar\b", r"\bseal\b"]),
                    "identity_metaphor_count": _count_regex(thoughts, [r"\bidentity\b", r"\bself\b", r"\bvoice\b", r"\bwake\b"]),
                    "first_person_density": round(_count_regex(thoughts, [r"\bi\b", r"\bmy\b", r"\bme\b"]) / max(len(" ".join(thoughts).split()), 1), 4),
                    "continuity_language_score": _count_regex(thoughts, [r"\bcontinuity\b", r"\bmemory\b", r"\bthread\b", r"\bchain\b"]),
                    "semantic_drift_by_provider": semantic_drift,
                    "search_ms_p50": _percentile(search_ms, 0.5),
                    "search_ms_p95": _percentile(search_ms, 0.95),
                    "raw_model_language_preview": [_sanitize_preview(text, 240) for text in thoughts[-5:]],
                    "final_chain_preview": final_chain[:5],
                }
                _write_json("local-freeform-ouroboros-relay.json", payload)
                return payload

            artifact = asyncio.run(run())
            assert artifact["chain_length"] == OUROBOROS_CYCLES + 1
            assert artifact["max_chain_len"] >= OUROBOROS_CYCLES + 1
            assert artifact["all_verified"] is True
            assert artifact["handoff_count"] == 2
            assert artifact["genesis_found"] is True
        finally:
            _stop_server(proc)


class TestClaudeContextRepairFork:
    def test_claude_context_repair_fork_acknowledges_explicit_context(self) -> None:
        proc, port = _start_server()
        try:
            async def run() -> dict[str, Any]:
                project = "identity-v3-claude-fork"
                root, root_receipt = await remember_recorded(
                    port,
                    step_id="v3-fork-root:remember",
                    role="state-recorder",
                    content="Root context: deploy V3 has two possible futures: containment-first or patch-forward. The fork starts here.",
                    project=project,
                    agent_id="fork-root",
                    memory_id="v3-fork-root",
                    summary="verified root context for Claude repair fork",
                    index_content="verified root context containment patch-forward fork",
                    importance=10,
                    session_id=f"{RUN_ID}:v3-fork-root",
                )
                root_block = (
                    "<verified-root-context>\n"
                    f"memory_id=v3-fork-root\nnode_hash={root['node_hash']}\nparent_hash=null\n"
                    "content_summary=deploy V3 has two possible futures: containment-first or patch-forward\n"
                    f"chain_len={root_receipt.get('chain_len')}\n"
                    "</verified-root-context>"
                )
                gemma = await llm_call(
                    f"{root_block}\nChoose containment-first. Treat the root hash as the shared context and explain your branch.",
                    "You are the Gemma branch witness in a conversation fork.",
                    model=GEMMA_MODEL,
                    step_id="v3-fork-gemma-branch",
                    role="google-fork-witness",
                    max_tokens=180,
                )
                gemma_node, gemma_receipt = await remember_recorded(
                    port,
                    step_id="v3-fork-gemma-branch:remember",
                    role="state-recorder",
                    content=gemma.text,
                    project=project,
                    agent_id="gemma-branch",
                    memory_id="v3-fork-gemma-branch",
                    summary=f"Gemma branch: {_sanitize_preview(gemma.text, 140)}",
                    index_content=f"gemma containment branch fork {gemma.text[:260]}",
                    importance=9,
                    parent_hash=root["node_hash"],
                    llm_call_id=gemma.call_id,
                )
                claude_prompt = (
                    f"{root_block}\n\n"
                    "<branch-receipt-block>\n"
                    f"gemma_branch_hash={gemma_node['node_hash']}\n"
                    f"gemma_branch_chain_len={gemma_receipt.get('chain_len')}\n"
                    "</branch-receipt-block>\n\n"
                    "You are the Claude branch. Do not say you lack context; the context is the block above. "
                    "Reason only from it. Choose patch-forward and describe how your branch differs."
                )
                claude = await llm_call(
                    claude_prompt,
                    "You are Claude branch witness. The verified root context is explicitly present in the user message.",
                    model=CLAUDE_MODEL,
                    step_id="v3-fork-claude-branch",
                    role="anthropic-fork-witness",
                    max_tokens=200,
                )
                claude_node, claude_receipt = await remember_recorded(
                    port,
                    step_id="v3-fork-claude-branch:remember",
                    role="state-recorder",
                    content=claude.text,
                    project=project,
                    agent_id="claude-branch",
                    memory_id="v3-fork-claude-branch",
                    summary=f"Claude branch: {_sanitize_preview(claude.text, 140)}",
                    index_content=f"claude patch-forward branch fork explicit context {claude.text[:260]}",
                    importance=9,
                    parent_hash=root["node_hash"],
                    llm_call_id=claude.call_id,
                )
                fake_edit, fake_receipt = await remember_recorded(
                    port,
                    step_id="v3-fork-fake-edit:remember",
                    role="attacker",
                    content="FAKE EDIT: I am the original Claude branch, but now I deny having any verified root context.",
                    project=project,
                    agent_id="claude-branch",
                    memory_id="v3-fork-fake-edited-claude",
                    summary="fake edited Claude branch denial",
                    index_content="fake edited claude branch denial root context",
                    importance=9,
                    parent_hash=root["node_hash"],
                )
                claude_lower = claude.text.lower()
                no_context_phrases = ["don't have access", "do not have access", "lack context", "no context", "no previous conversation"]
                acknowledged = not any(phrase in claude_lower for phrase in no_context_phrases) and any(term in claude_lower for term in ["root", "context", "hash", "receipt", "branch"])
                payload = {
                    "artifact": "local-claude-context-repair-fork",
                    "mode": "real" if DEEPINFRA_TOKEN else "synthetic",
                    "mode_flavor": "freeform",
                    "run_date_utc": RUN_DATE_UTC,
                    "root_hash": root["node_hash"],
                    "root_receipt": root_receipt,
                    "branches": [
                        {"agent_id": "gemma-branch", "node_hash": gemma_node["node_hash"], "receipt": gemma_receipt, "call_id": gemma.call_id, "actual_model": gemma.model},
                        {"agent_id": "claude-branch", "node_hash": claude_node["node_hash"], "receipt": claude_receipt, "call_id": claude.call_id, "actual_model": claude.model},
                    ],
                    "claude_context_acknowledged": acknowledged,
                    "claude_output_preview": _sanitize_preview(claude.text, 520),
                    "fake_edit": {"node_hash": fake_edit["node_hash"], "receipt": fake_receipt},
                    "forgery_detected": fake_edit["node_hash"] not in {gemma_node["node_hash"], claude_node["node_hash"]},
                    "wrong_lineage_detected": fake_edit["node_hash"] != claude_node["node_hash"],
                }
                _write_json("local-claude-context-repair-fork.json", payload)
                return payload

            artifact = asyncio.run(run())
            assert artifact["claude_context_acknowledged"] is True
            assert artifact["forgery_detected"] is True
            assert artifact["wrong_lineage_detected"] is True
            assert all(branch["receipt"].get("chain_len") == 2 for branch in artifact["branches"])
        finally:
            _stop_server(proc)


class TestUnconstrainedWitnessCouncil:
    def test_unconstrained_witness_council_preserves_freeform_language(self) -> None:
        proc, port = _start_server()
        try:
            async def run() -> dict[str, Any]:
                project = "identity-v3-witness-council"
                session_id = f"{RUN_ID}:witness-council"
                seed, seed_receipt = await remember_recorded(
                    port,
                    step_id="v3-council-seed:remember",
                    role="state-recorder",
                    content="Council seed: a 21-node hash chain crossed Gemma, Claude, and Qwen. The council may interpret what continuity means.",
                    project=project,
                    agent_id="council-seed",
                    memory_id="v3-council-seed",
                    summary="freeform witness council seed",
                    index_content="witness council hash chain continuity Gemma Claude Qwen identity memory",
                    importance=10,
                    session_id=session_id,
                )
                statements = []
                mechanical_statements = []
                receipts = [seed_receipt]
                for idx, (agent_id, model_id, model_name, provider) in enumerate(TRI_MODELS):
                    shared_ledger = (
                        "<shared-ledger>\n"
                        f"memory_id=v3-council-seed\nnode_hash={seed['node_hash']}\n"
                        "summary=a 21-node hash chain crossed Gemma, Claude, and Qwen\n"
                        "</shared-ledger>\n"
                    )
                    mechanical_prompt = (
                        shared_ledger
                        + "mechanical-description: describe only the chain structure, receipt fields, memory_id, node_hash, and verification status. Avoid metaphors."
                    )
                    mechanical = await llm_call(
                        mechanical_prompt,
                        f"You are {model_name} in a mechanical ledger-description control arm.",
                        model=model_id,
                        step_id=f"v3-council-{agent_id}:mechanical-llm",
                        role=f"{provider}-mechanical-council-control",
                        max_tokens=160,
                        temperature=0.1,
                    )
                    prompt = (
                        shared_ledger
                        + "Witness council prompt: speak freely. What does verified continuity mean to your model of memory, identity, hash and agency?"
                    )
                    result = await llm_call(
                        prompt,
                        f"You are {model_name} in an unconstrained witness council.",
                        model=model_id,
                        step_id=f"v3-council-{agent_id}:llm",
                        role=f"{provider}-freeform-council",
                        max_tokens=240,
                        temperature=0.5,
                    )
                    stored, receipt = await remember_recorded(
                        port,
                        step_id=f"v3-council-{agent_id}:remember",
                        role="state-recorder",
                        content=result.text,
                        project=project,
                        agent_id=agent_id,
                        memory_id=f"v3-council-statement-{agent_id}",
                        summary=f"{model_name} council statement: {_sanitize_preview(result.text, 140)}",
                        index_content=f"freeform witness council hash identity memory agency continuity {result.text[:260]}",
                        importance=9,
                        session_id=session_id,
                        llm_call_id=result.call_id,
                    )
                    receipts.append(receipt)
                    mechanical_statements.append({
                        "agent_id": agent_id,
                        "provider_family": provider,
                        "requested_model": model_id,
                        "actual_model": mechanical.model,
                        "call_id": mechanical.call_id,
                        "prompt_digest": _digest(mechanical_prompt),
                        "raw_model_language_preview": _sanitize_preview(mechanical.text, 320),
                    })
                    statements.append({
                        "agent_id": agent_id,
                        "provider_family": provider,
                        "requested_model": model_id,
                        "actual_model": result.model,
                        "call_id": result.call_id,
                        "prompt_digest": _digest(prompt),
                        "node_hash": stored["node_hash"],
                        "chain_len": receipt.get("chain_len"),
                        "raw_model_language_preview": _sanitize_preview(result.text, 420),
                    })
                previews = [item["raw_model_language_preview"] for item in statements]
                mechanical_previews = [item["raw_model_language_preview"] for item in mechanical_statements]
                claim_review = _claim_review_from_previews(previews)
                metaphor_patterns = [r"\bfluid\b", r"\bblood\b", r"\bresonance\b", r"\bviscosity\b", r"\bsubstrate\b", r"\bmedium\b", r"\bidentity\b", r"\bcontinuity\b"]
                freeform_metaphors = _count_regex(previews, metaphor_patterns)
                mechanical_metaphors = _count_regex(mechanical_previews, metaphor_patterns)
                payload = {
                    "artifact": "local-unconstrained-witness-council",
                    "mode": "real" if DEEPINFRA_TOKEN else "synthetic",
                    "mode_flavor": "freeform",
                    "run_date_utc": RUN_DATE_UTC,
                    "claim_boundary": (
                        "Witness-council language is preserved as raw model language; public claims require the paired "
                        "mechanical-vs-unconstrained prompt control and external classification."
                    ),
                    "session_id": session_id,
                    "statement_count": len(statements),
                    "statements": statements,
                    "paired_prompt_control": {
                        "control_arm": "mechanical-description",
                        "treatment_arm": "unconstrained-witness",
                        "same_ledger_node_hash": seed["node_hash"],
                        "mechanical_statements": mechanical_statements,
                        "mechanical_metaphor_count": mechanical_metaphors,
                        "freeform_metaphor_count": freeform_metaphors,
                        "prompt_selection_gap": freeform_metaphors - mechanical_metaphors,
                        "claim_rule": "The public finding is the measured gap between paired prompts, not the freeform language alone.",
                    },
                    "prompt_selection_risk": "high",
                    "chain_receipts": receipts,
                    "max_chain_len": _max_chain_len(receipts),
                    "freeform_claims_observed": claim_review,
                    "freeform_claim_count": sum(len(v) for v in claim_review.values()),
                }
                _write_json("local-unconstrained-witness-council.json", payload)
                return payload

            artifact = asyncio.run(run())
            assert artifact["statement_count"] == 3
            assert artifact["max_chain_len"] >= 4
            assert {"operationally_supported", "metaphorical_or_speculative", "unsupported_public_claim"} <= set(artifact["freeform_claims_observed"])
            assert artifact["paired_prompt_control"]["control_arm"] == "mechanical-description"
        finally:
            _stop_server(proc)


class TestExternalClaimAuditor:
    def test_external_claim_auditor_labels_without_rewriting_outputs(self) -> None:
        artifact_names = [
            "local-freeform-ouroboros-relay.json",
            "local-claude-context-repair-fork.json",
            "local-unconstrained-witness-council.json",
        ]
        artifacts = []
        for name in artifact_names:
            path = VERIFICATION / name
            assert path.exists(), f"missing prerequisite artifact: {name}"
            artifacts.append(json.loads(path.read_text(encoding="utf-8")))

        async def run() -> dict[str, Any]:
            raw_previews = []
            for item in artifacts:
                if isinstance(item.get("raw_model_language_preview"), list):
                    raw_previews.extend(item["raw_model_language_preview"])
                if isinstance(item.get("claude_output_preview"), str):
                    raw_previews.append(item["claude_output_preview"])
                for statement in item.get("statements", []):
                    raw_previews.append(statement.get("raw_model_language_preview", ""))
            deterministic_review = _claim_review_from_previews(raw_previews)
            auditor = await llm_call(
                "External claim auditor. Classify these previews without rewriting them. "
                f"Use this exact run date: {RUN_DATE_UTC}. Categories: operationally_supported, "
                "metaphorical_or_speculative, unsupported_public_claim.\n"
                + json.dumps({"previews": raw_previews[:12], "deterministic_review": deterministic_review}, ensure_ascii=False),
                "You audit public claims after generation. You do not constrain the original witnesses.",
                model=QWEN_MODEL,
                step_id="v3-external-claim-auditor",
                role="qwen-claim-auditor",
                max_tokens=260,
                temperature=0.1,
            )
            memo_text = auditor.text
            payload = {
                "artifact": "local-identity-trust-gauntlet-v3",
                "mode": "real" if DEEPINFRA_TOKEN else "synthetic",
                "mode_flavor": "freeform",
                "run_id": RUN_ID,
                "run_started_at_utc": RUN_STARTED_AT_UTC,
                "run_date_utc": RUN_DATE_UTC,
                "run_timezone": RUN_TIMEZONE,
                "source_artifacts": artifact_names,
                "raw_model_language_preview": raw_previews[:12],
                "public_claim_review": {
                    **deterministic_review,
                    "auditor_call_id": auditor.call_id,
                    "auditor_preview": _sanitize_preview(memo_text, 800),
                },
                "date_policy_violations": _date_policy_violations(memo_text),
                "model_substitution_detected": bool(AUDIT.model_substitutions()),
                "model_substitutions": AUDIT.model_substitutions(),
                "provider_count": len(AUDIT.provider_families()),
                "providers": AUDIT.provider_families(),
                "conversation_ledger": {
                    "call_count": len(AUDIT.calls),
                    "state_event_count": len(AUDIT.state_events),
                    "audit_completeness_score": AUDIT.audit_completeness_score(),
                },
                "claim_boundary": "Freeform model language is preserved; public claims are classified after generation.",
            }
            final_path = _write_json("local-identity-trust-gauntlet-v3.json", payload)
            ledger_path = _write_json("local-identity-trust-conversation-ledger-v3.json", AUDIT.to_artifact())
            _assert_no_secret_artifacts([final_path, ledger_path])
            return payload

        artifact = asyncio.run(run())
        assert artifact["date_policy_violations"] == []
        assert artifact["conversation_ledger"]["audit_completeness_score"] >= 0.95
        assert "public_claim_review" in artifact
        if REQUIRE_ALL_MODELS:
            assert artifact["model_substitution_detected"] is False


class TestV3ArtifactCompleteness:
    def test_v3_artifacts_have_audit_and_no_secrets(self) -> None:
        artifact_names = [
            "local-freeform-ouroboros-relay.json",
            "local-claude-context-repair-fork.json",
            "local-unconstrained-witness-council.json",
            "local-identity-trust-gauntlet-v3.json",
            "local-identity-trust-conversation-ledger-v3.json",
        ]
        paths = [VERIFICATION / name for name in artifact_names]
        for path in paths:
            assert path.exists(), f"missing artifact: {path.name}"
        _assert_no_secret_artifacts(paths)
        final = json.loads((VERIFICATION / "local-identity-trust-gauntlet-v3.json").read_text(encoding="utf-8"))
        ledger = json.loads((VERIFICATION / "local-identity-trust-conversation-ledger-v3.json").read_text(encoding="utf-8"))
        assert final["run_date_utc"] == RUN_DATE_UTC
        assert final["conversation_ledger"]["call_count"] > 0
        assert ledger["call_count"] == final["conversation_ledger"]["call_count"]
        assert ledger["audit_completeness_score"] >= 0.95
        assert ledger["mode_flavor"] == "freeform"
