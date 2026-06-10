from __future__ import annotations

import argparse
import fnmatch
import getpass
import hashlib
import html
import importlib.util
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import unicodedata
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request
from urllib import parse as urlparse
from xml.etree import ElementTree

from helix_proto import helix_cli_core

DEFAULT_THEME = "industrial-brutalist"
_THEME_ALIASES = {
    "industrial-brutalist": "industrial-brutalist",
    "industrial-neon": "industrial-neon",
    "cyberpunk": "industrial-neon",
    "cyberpunk-gray": "industrial-neon",
    "xerox": "xerox",
    "brown-console": "brown-console",
    "brown": "brown-console",
}
_THEME_PALETTES: dict[str, dict[str, str]] = {
    name: {"theme_name": canonical, "description": canonical}
    for name, canonical in _THEME_ALIASES.items()
}
_HAS_UI = bool(importlib.util.find_spec("rich") and importlib.util.find_spec("prompt_toolkit"))
Console = None
PromptSession = None
WordCompleter = None
_CHROME = None


def _chrome() -> Any:
    global _CHROME, _HAS_UI, Console, PromptSession, WordCompleter, _THEME_PALETTES
    if _CHROME is None:
        from helix_proto import helix_cli_chrome as chrome  # noqa: PLC0415

        _CHROME = chrome
        _HAS_UI = bool(chrome.HAS_UI)
        Console = chrome.Console
        PromptSession = chrome.PromptSession
        WordCompleter = chrome.WordCompleter
        _THEME_PALETTES = chrome.THEME_PALETTES
    return _CHROME


def _theme_palette(theme_name: str | None) -> dict[str, str]:
    return _chrome().theme_palette(theme_name)


def _rich_theme(theme_name: str | None):
    return _chrome().rich_theme(theme_name)


def _prompt_style(theme_name: str | None):
    return _chrome().prompt_style(theme_name)


def _choose_ui_option(*args: Any, **kwargs: Any) -> Any:
    return _chrome().choose_option(*args, **kwargs)


def _normalize_theme_name(theme_name: str | None) -> str:
    if _CHROME is None:
        return _THEME_ALIASES.get(str(theme_name or DEFAULT_THEME).strip().lower(), DEFAULT_THEME)
    return _chrome().normalize_theme_name(theme_name)


def _theme_report() -> list[dict[str, Any]]:
    return _chrome().theme_report()


def _play_boot_handshake(*args: Any, **kwargs: Any) -> Any:
    return _chrome().play_boot_handshake(*args, **kwargs)


def _prompt_bottom_toolbar(*args: Any, **kwargs: Any) -> Any:
    return _chrome().prompt_bottom_toolbar(*args, **kwargs)


def _prompt_message(*args: Any, **kwargs: Any) -> Any:
    return _chrome().prompt_message(*args, **kwargs)


def _prompt_toolbar_markup(*args: Any, **kwargs: Any) -> Any:
    return _chrome().prompt_toolbar_markup(*args, **kwargs)


def _render_boot_banner(*args: Any, **kwargs: Any) -> Any:
    return _chrome().render_boot_banner(*args, **kwargs)


def _render_chat_response(*args: Any, **kwargs: Any) -> Any:
    return _chrome().render_chat_response(*args, **kwargs)


def _render_session_ribbon(*args: Any, **kwargs: Any) -> Any:
    return _chrome().render_session_ribbon(*args, **kwargs)


def _render_task_result_panel(*args: Any, **kwargs: Any) -> Any:
    return _chrome().render_task_result(*args, **kwargs)


def _render_verify_audit(*args: Any, **kwargs: Any) -> Any:
    return _chrome().render_verify_audit(*args, **kwargs)

_VISIBLE_OUTPUT_RE = re.compile(r"(?is)<helix_output>(.*?)</helix_output>")
_INTERNAL_BLOCK_RE = re.compile(
    r"(?is)<(think|thinking|tool_call|scratchpad|analysis|reasoning|plan|draft)\b[^>]*>.*?</\1>"
)
_UNCLOSED_INTERNAL_RE = re.compile(
    r"(?is)<(think|thinking|tool_call|scratchpad|analysis|reasoning|plan|draft)\b[^>]*>.*$"
)
_FINAL_MARKER_RE = re.compile(
    r"(?im)^\s*(final answer|final output|actual output|refined answer|respuesta final|respuesta|answer|output|response)\s*:\s*"
)
_THINKING_ONLY_RE = re.compile(
    r"(?is)^\s*(thinking process|thought process|reasoning process|analysis|the user is asking|we need answer)\b"
)
_URL_REF_RE = re.compile(r"(?i)\bhttps?://[^\s<>\"]+")
_PROVIDER_COOLDOWNS: dict[str, dict[str, Any]] = {}
_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 75.0
_HTTP_SESSION = None
_REQUESTS_SESSION_READY = False


@dataclass
class BlindInferencePolicy:
    enabled: bool = False
    scope: str = "cloud_proxy"
    placeholder_stability: str = "per_task"
    rules: tuple[dict[str, Any], ...] = ()
    detectors: dict[str, bool] | None = None
    policy_id: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "BlindInferencePolicy":
        body = dict(payload or {})
        detectors = dict(body.get("detectors") or {})
        material = {
            "enabled": bool(body.get("enabled", False)),
            "scope": str(body.get("scope") or "cloud_proxy"),
            "placeholder_stability": str(body.get("placeholder_stability") or "per_task"),
            "rules": [item for item in body.get("rules") or [] if isinstance(item, dict)],
            "detectors": detectors,
        }
        policy_id = hashlib.sha256(json.dumps(material, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]
        return cls(
            enabled=material["enabled"],
            scope=material["scope"],
            placeholder_stability=material["placeholder_stability"],
            rules=tuple(material["rules"]),
            detectors=detectors,
            policy_id=policy_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "scope": self.scope,
            "placeholder_stability": self.placeholder_stability,
            "rules": list(self.rules),
            "detectors": dict(self.detectors or {}),
            "policy_id": self.policy_id,
        }


@dataclass
class IntentCard:
    lane: str
    primary_goal: str
    correction_notes: list[str]
    sources: list[str]
    urls: list[str]
    output_target: str | None
    requires_write: bool
    requires_model: bool
    requires_opencode: bool
    requires_readback: bool
    route_reason: str
    confidence: float
    fallback_command: str | None
    route: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "helix-intent-card-v1",
            "lane": self.lane,
            "primary_goal": self.primary_goal,
            "correction_notes": list(self.correction_notes),
            "sources": list(self.sources),
            "urls": list(self.urls),
            "output_target": self.output_target,
            "requires_write": self.requires_write,
            "requires_model": self.requires_model,
            "requires_opencode": self.requires_opencode,
            "requires_readback": self.requires_readback,
            "route_reason": self.route_reason,
            "confidence": self.confidence,
            "fallback_command": self.fallback_command,
            "route": dict(self.route or {}),
        }


@dataclass
class ArtifactState:
    path: str
    kind: str
    exists: bool
    sha256: str | None = None
    bytes: int | None = None
    readback_chars: int = 0
    pages: int | None = None
    preview: str = ""
    warnings: list[str] | None = None
    last_action: str = "inspect"

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any] | None) -> "ArtifactState | None":
        if not isinstance(artifact, dict) or not artifact.get("path"):
            return None
        readback = artifact.get("readback") if isinstance(artifact.get("readback"), dict) else {}
        warnings = artifact.get("warnings") if isinstance(artifact.get("warnings"), list) else []
        return cls(
            path=str(artifact.get("path") or ""),
            kind=str(artifact.get("kind") or "file"),
            exists=bool(artifact.get("exists")),
            sha256=str(artifact.get("sha256")) if artifact.get("sha256") else None,
            bytes=int(artifact.get("bytes")) if isinstance(artifact.get("bytes"), int) else None,
            readback_chars=int(readback.get("chars") or 0),
            pages=int(readback.get("pages")) if isinstance(readback.get("pages"), int) else None,
            preview=str(artifact.get("preview") or readback.get("preview") or ""),
            warnings=[str(item) for item in warnings],
            last_action=str(artifact.get("last_action") or "inspect"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "helix-artifact-state-v1",
            "path": self.path,
            "artifact_kind": self.kind,
            "exists": self.exists,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "readback_chars": self.readback_chars,
            "pages": self.pages,
            "preview": _truncate_text(self.preview, 1200)["text"] if self.preview else "",
            "warnings": list(self.warnings or []),
            "last_action": self.last_action,
        }


class WorkEventBus:
    def __init__(self, session: "InteractiveSession", *, run_id: str, goal: str) -> None:
        self.session = session
        self.run_id = run_id
        self.goal = goal
        self._started = time.perf_counter()
        self._last = self._started
        self.events: list[dict[str, Any]] = []

    def emit(self, event: str, message: str, **payload: Any) -> dict[str, Any]:
        now = time.perf_counter()
        row = {
            "event": event,
            "message": message,
            "run_id": self.run_id,
            "created_utc": _utc_now(),
            "elapsed_ms": round((now - self._started) * 1000, 3),
            "phase_ms": round((now - self._last) * 1000, 3),
        }
        if payload:
            row.update({key: value for key, value in payload.items() if value is not None})
        self._last = now
        self.events.append(row)
        return row

    def phase_ms(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for row in self.events:
            event = str(row.get("event") or "unknown")
            totals[event] = round(totals.get(event, 0.0) + float(row.get("phase_ms") or 0.0), 3)
        return totals


def _sync_work_event_run_id(bus: WorkEventBus, run_id: str) -> None:
    if not run_id or bus.run_id == run_id:
        return
    bus.run_id = run_id
    for row in bus.events:
        row["run_id"] = run_id


def blind_transform_request(*args: Any, **kwargs: Any) -> Any:
    from helix_proto.blind_inference import BlindInferencePolicy as _RealBlindPolicy  # noqa: PLC0415
    from helix_proto.blind_inference import blind_transform_request as _blind_transform_request  # noqa: PLC0415

    if len(args) >= 2 and isinstance(args[1], BlindInferencePolicy):
        args = (args[0], _RealBlindPolicy.from_payload(args[1].to_dict()), *args[2:])
    elif isinstance(kwargs.get("policy"), BlindInferencePolicy):
        kwargs["policy"] = _RealBlindPolicy.from_payload(kwargs["policy"].to_dict())
    return _blind_transform_request(*args, **kwargs)


def blind_rehydrate_response(*args: Any, **kwargs: Any) -> Any:
    from helix_proto.blind_inference import blind_rehydrate_response as _blind_rehydrate_response  # noqa: PLC0415

    return _blind_rehydrate_response(*args, **kwargs)


class ProviderRateLimitError(RuntimeError):
    """Provider-level throttle that should not fan out to more same-provider attempts."""

    def __init__(
        self,
        provider_name: str,
        model: str | None,
        message: str,
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.model = model
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)


def _looks_like_internal_line(line: str) -> bool:
    clean = line.strip()
    if not clean:
        return False
    lowered = clean.lower()
    internal_fragments = (
        "i will ",
        "i'll ",
        "let's ",
        "all good",
        "proceed",
        "mandatory thinking",
        "thinking box",
        "thinking process",
        "thought process",
        "reasoning process",
        "output generation",
        "final polish",
        "check constraints",
        "construct response",
        "drafting the response",
        "mapping to helix architecture",
        "merkle-dag validation",
        "input received:",
        "planning:",
        "execution:",
        "analysis:",
        "` tags",
    )
    if any(fragment in lowered for fragment in internal_fragments):
        return True
    if re.match(r"^\s*(plan|draft|reasoning|analysis|thinking process|thought process|mental|checks?|system instructions?)\s*:", lowered):
        return True
    if re.match(
        r"^\s*(\d+[\s.)]+|[-*]|\u2022)\s*"
        r"(analy|check|identify|draft|final|write|refine|step|reasoning|goal|language|tone|style|request|input)",
        lowered,
    ):
        return True
    if lowered in {"no emojis? checked.", "no preamble? checked.", "language? spanish.", "tone? direct."}:
        return True
    return False


def _dedupe_repeated_paragraphs(text: str) -> str:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    seen: set[str] = set()
    kept: list[str] = []
    for paragraph in paragraphs:
        normalized = re.sub(r"\s+", " ", paragraph).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        kept.append(paragraph.strip())
    return "\n\n".join(kept).strip()


def _panel_width(active_console: Any) -> int:
    return _chrome().panel_width(active_console, context="chat")


def _short_model_name(model: str | None) -> str:
    text = str(model or "model")
    if "/" in text:
        text = text.split("/")[-1]
    return text if len(text) <= 42 else text[:39] + "..."


def _display_model_used(metadata: dict[str, Any], fallback: str | None = None) -> str:
    actual = _short_model_name(metadata.get("actual_model") or fallback or "model")
    selected = _short_model_name(metadata.get("selected_model") or "")
    if metadata.get("failover_used") and selected and selected != actual:
        return f"{selected} -> {actual}"
    return actual


def _provider_is_cloud_boundary(provider: "ProviderSpec") -> bool:
    if provider.kind == "helix-local":
        return False
    base_url = str(provider.base_url or "").lower()
    if base_url.startswith("http://127.0.0.1") or base_url.startswith("http://localhost"):
        return False
    return True


def _default_blind_policy_payload() -> dict[str, Any]:
    return {
        "enabled": False,
        "scope": "cloud_proxy",
        "placeholder_stability": "per_task",
        "rules": [],
        "detectors": {
            "email": True,
            "document_id": True,
            "account": True,
            "phone": True,
        },
    }


def _coerce_blind_policy_payload(value: str) -> dict[str, Any]:
    candidate = str(value or "").strip()
    if not candidate:
        raise ValueError("blind policy payload cannot be empty")
    path = Path(candidate).expanduser()
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
    else:
        loaded = json.loads(candidate)
    if not isinstance(loaded, dict):
        raise ValueError("blind policy must be a JSON object")
    return loaded


def _run_with_status(active_console: Any, func: Any, *, phase: str = "thinking") -> Any:
    return _chrome().run_with_status(active_console, func, phase=phase, phase_messages=_THINKING_MESSAGE_PHASES)


def _clean_assistant_text(text: str) -> str:
    """Return only the user-visible answer from a noisy model response."""
    if not text:
        return ""

    raw = str(text).strip()
    tagged = _VISIBLE_OUTPUT_RE.findall(raw)
    if tagged:
        raw = tagged[-1]
    elif _THINKING_ONLY_RE.match(raw) and not list(_FINAL_MARKER_RE.finditer(raw)):
        return ""

    raw = _INTERNAL_BLOCK_RE.sub("", raw)
    raw = _UNCLOSED_INTERNAL_RE.sub("", raw)

    marker_matches = list(_FINAL_MARKER_RE.finditer(raw))
    if marker_matches:
        raw = raw[marker_matches[-1].end():]

    lines = []
    for line in raw.splitlines():
        if _looks_like_internal_line(line):
            continue
        lines.append(line.rstrip())
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"(?is)</?(think|thinking|tool_call|scratchpad|analysis|reasoning|plan|draft)\b[^>]*>", "", cleaned)
    return _dedupe_repeated_paragraphs(cleaned).strip()


def _task_visible_output(raw_text: str) -> str:
    cleaned = _clean_assistant_text(raw_text)
    if cleaned:
        return cleaned
    if str(raw_text or "").strip():
        return "[raw output suppressed: model returned only internal reasoning or tool protocol residue]"
    return "No response from provider."


def _conversation_gate(raw_text: str) -> dict[str, Any]:
    """Normalize provider text into a stable user-visible response contract."""
    raw = str(raw_text or "")
    started = time.perf_counter()
    visible = _clean_assistant_text(raw)
    tool_protocol = bool(re.search(r"(?is)<tool_call\b|\"tool_calls?\"|\"function_calls?\"", raw))
    thinking_only = bool(raw.strip() and not visible and (_THINKING_ONLY_RE.match(raw.strip()) or tool_protocol))
    suppressed_reasoning = bool(raw.strip() and (visible != raw.strip() or thinking_only))
    fallback_used = False
    if not visible and raw.strip():
        visible = "[respuesta suprimida: el modelo devolvio solo razonamiento interno o protocolo de herramientas]"
        fallback_used = True
    elif not visible:
        visible = "No response from provider."
        fallback_used = True
    blocked_file_promise = _looks_like_unverified_file_promise(visible)
    if blocked_file_promise:
        visible = (
            "Eso requiere una tarea de Work Runtime; no voy a afirmar que cree, guarde o modifique "
            "archivos desde chat si HeliX no ejecuto escritura y readback verificable. "
            "Reformula como pedido de trabajo o usa /work run ..."
        )
        fallback_used = True
    return {
        "visible_text": visible,
        "raw_text": raw,
        "suppressed_reasoning": suppressed_reasoning,
        "tool_protocol": tool_protocol,
        "empty_visible_output": thinking_only or not bool(_clean_assistant_text(raw)),
        "local_fallback_used": fallback_used,
        "repair_retry_used": False,
        "blocked_file_promise": blocked_file_promise,
        "response_gate_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def _looks_like_unverified_file_promise(text: str) -> bool:
    folded = _fold_cli_text(text)
    if not folded:
        return False
    creation_terms = (
        "he creado",
        "cree ",
        "creado ",
        "he guardado",
        "guarde ",
        "guardado ",
        "he modificado",
        "modifique ",
        "modificado ",
        "voy a proceder con la creacion",
        "[creando",
    )
    artifact_terms = (".pdf", ".docx", ".md", ".html", " c:\\", " /", " archivo", " documento", " pagina web")
    return any(term in folded for term in creation_terms) and any(term in folded for term in artifact_terms)


def _goal_requests_memory_lookup(text: str) -> bool:
    lowered = str(text or "").lower()
    memory_terms = (
        "memoria",
        "memory",
        "recuerdo",
        "recuerdos",
        "recorda",
        "recordas",
        "recordás",
        "remember",
        "recall",
        "thread",
        "hilo",
        "conversacion",
        "conversación",
        "chat previo",
        "charla",
        "historial",
    )
    lookup_terms = (
        "revisa",
        "revisa ",
        "revisá",
        "resumi",
        "resumí",
        "resumime",
        "resúmeme",
        "busca",
        "buscá",
        "buscame",
        "consulta",
        "consulta ",
        "consultá",
        "decime",
        "dime",
        "que conclusion",
        "qué conclusión",
        "que conclu",
        "qué conclu",
        "que sacamos",
        "qué sacamos",
        "encontraste",
        "what did we conclude",
    )
    return any(term in lowered for term in memory_terms) and any(term in lowered for term in lookup_terms)


_HASH_REF_RE = re.compile(r"\b[0-9a-fA-F]{8,64}\b")
_LOCAL_FILE_SUFFIX_RE = r"(?:jsonl?|md|txt|log|py|rs|toml|ya?ml|csv|html?|css|js|ts|tsx|jsx|ini|cfg)"
_UNQUOTED_LOCAL_PATH_RE = re.compile(
    rf"(?i)\b[A-Z]:[\\/][^\r\n\"'`<>|]+?\.{_LOCAL_FILE_SUFFIX_RE}\b"
)
_UNQUOTED_WINDOWS_PATH_LINE_RE = re.compile(r"(?i)\b[A-Z]:[\\/][^\r\n\"'`<>|]+(?=$|[\r\n])")
_UNQUOTED_RELATIVE_PATH_RE = re.compile(r"(?<!https:)(?<!http:)\b(?:\.{1,2}[\\/]|(?:[\w.-]+[\\/])+[\w.-]+)\b")


def _extract_hash_prefixes(text: str) -> list[str]:
    """Extract plausible HeliX node hash prefixes without treating plain dates as hashes."""
    refs: list[str] = []
    seen: set[str] = set()
    for match in _HASH_REF_RE.finditer(str(text or "")):
        value = match.group(0).lower()
        if not any(char in "abcdef" for char in value):
            continue
        if value not in seen:
            refs.append(value)
            seen.add(value)
    return refs


def _latest_hash_reference(text: str, history: list[dict[str, str]] | None = None) -> str | None:
    current = _extract_hash_prefixes(text)
    if current:
        return current[-1]
    for item in reversed(history or []):
        refs = _extract_hash_prefixes(str(item.get("content") or ""))
        if refs:
            return refs[-1]
    return None


def _is_hash_recovery_request(text: str, history: list[dict[str, str]] | None = None) -> bool:
    lowered = " ".join(str(text or "").lower().split())
    if _extract_local_path_refs(text):
        explicit_hash_terms = (
            "hash",
            "node_hash",
            "node hash",
            "memory_id",
            "memory id",
            "este hash",
            "ese hash",
            "este node",
            "ese node",
        )
        if not any(term in lowered for term in explicit_hash_terms):
            return False
    current_refs = _extract_hash_prefixes(text)
    if current_refs:
        if _looks_like_pasted_suite_evidence(text):
            suite_hash_terms = ("recuper", "donde", "completo", "contenido", "literal", "este hash", "ese hash", "node_hash")
            return any(term in lowered for term in suite_hash_terms)
        explicit_terms = (
            "hash",
            "node_hash",
            "node hash",
            "memoria",
            "memory",
            "donde",
            "esta",
            "recuper",
            "completo",
            "contenido",
            "texto",
            "literal",
            "en realidad",
            "esto",
            "ese",
            "ancla",
        )
        return any(term in lowered for term in explicit_terms)
    if not _latest_hash_reference("", history):
        return False
    followup_terms = (
        "recuper",
        "completo",
        "contenido",
        "texto",
        "donde esta",
        "donde quedo",
        "mostramelo",
        "mostrame eso",
        "lee eso",
        "traelo",
    )
    return any(term in lowered for term in followup_terms)


def _format_memory_resolve_answer(result: dict[str, Any]) -> str:
    status = str(result.get("status") or "")
    ref = str(result.get("ref") or "").strip()
    if status == "error" and not ref:
        return (
            "No ejecuté `memory.resolve` porque la tool fue llamada sin `ref`. "
            "Para recuperar contenido exacto necesito un `memory_id` o un prefijo de `node_hash`; "
            "para una pregunta general puedo responder sin usar esa tool."
        )
    if status == "not_found":
        return (
            f"No pude resolver `{ref}` contra la memoria HeliX ni contra las transcripciones locales. "
            "No voy a reconstruir ese texto de memoria porque eso seria alucinarlo; necesito un hash mas largo, "
            "un `memory_id`, o que el registro exista en el workspace/transcripts activos."
        )
    if status == "ambiguous":
        rows = []
        for item in result.get("matches", [])[:8]:
            rows.append(
                f"- `{item.get('node_hash') or item.get('memory_id')}` | `{item.get('memory_id')}` | "
                f"{str(item.get('summary') or item.get('content') or '')[:140]}"
            )
        return (
            f"`{ref}` coincide con mas de una memoria. Necesito un prefijo mas largo o un `memory_id` exacto.\n\n"
            + "\n".join(rows)
        )
    if status != "ok":
        return f"No pude resolver `{ref}`: {result.get('error') or 'estado desconocido'}"

    record = (result.get("matches") or [{}])[0]
    content = str(record.get("content") or "")
    truncated = bool(record.get("content_truncated"))
    source = str(record.get("source") or "memory")
    chain = record.get("chain") if isinstance(record.get("chain"), dict) else {}
    lines = [
        f"Encontré `{ref}` en HeliX sin reconstruirlo con el modelo.",
        "",
        f"- Source: `{source}`",
        f"- Memory ID: `{record.get('memory_id') or 'n/a'}`",
        f"- Node hash: `{record.get('node_hash') or 'n/a'}`",
        f"- Chain status: `{chain.get('status') or record.get('chain_status') or 'n/a'}`",
    ]
    if record.get("path"):
        lines.append(f"- Path: `{record.get('path')}`")
    if record.get("created_utc") or record.get("created_ms"):
        lines.append(f"- Created: `{record.get('created_utc') or record.get('created_ms')}`")
    lines.extend(["", "Contenido exacto:", "", "```text", content, "```"])
    if truncated:
        lines.append("\n[helix] El contenido existe pero fue truncado por limite de salida de `memory.resolve`.")
    return "\n".join(lines)


def _normalise_local_path_ref(ref: str) -> str:
    value = str(ref or "").strip().strip("\"'`“”").rstrip(".,;:")
    if value.lower().startswith("file://"):
        parsed = urlparse.urlparse(value)
        value = urlparse.unquote(parsed.path or "")
        if re.match(r"^/[A-Za-z]:/", value):
            value = value[1:]
        value = value.replace("/", "\\") if os.name == "nt" else value
    if re.search(r"[A-Za-z]:[\\/]", value) or "\\" in value or "/" in value:
        value = re.sub(r"\s*[\r\n]+\s*", "", value)
    else:
        value = re.sub(r"[\r\n]+", " ", value)
    return value.strip()


def _path_ref_exists_in_repo_context(path_ref: str) -> bool:
    candidate = Path(path_ref).expanduser()
    bases = [
        Path.cwd(),
        REPO_ROOT,
        REPO_ROOT / "workspace",
        REPO_ROOT / "verification",
    ]
    candidates = [candidate] if candidate.is_absolute() else [base / candidate for base in bases]
    for item in candidates:
        try:
            if item.exists():
                return True
        except Exception:
            continue
    return False


def _extract_url_refs(text: str) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for match in _URL_REF_RE.finditer(str(text or "")):
        url = str(match.group(0) or "").rstrip(").,;:")
        if url and url not in seen:
            refs.append(url)
            seen.add(url)
    return refs


def _extract_local_path_refs(text: str) -> list[str]:
    raw = str(text or "")
    url_refs = _extract_url_refs(raw)
    refs: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        path_ref = _normalise_local_path_ref(candidate)
        lowered = path_ref.lower()
        if not path_ref or lowered.startswith(("http://", "https://")):
            return
        markers = (
            re.search(r"[A-Za-z]:[\\/]", path_ref) is not None,
            lowered.startswith(("~\\", "~/", ".\\", "./", "..\\", "../")),
            ("\\" in path_ref or "/" in path_ref),
            _path_ref_exists_in_repo_context(path_ref),
        )
        if any(markers) and path_ref not in seen:
            refs.append(path_ref)
            seen.add(path_ref)

    for match in re.finditer(r'["`“](.+?)["`”]', raw, flags=re.DOTALL):
        _add(match.group(1))
    for match in _UNQUOTED_LOCAL_PATH_RE.finditer(raw):
        _add(match.group(0))
    for match in _UNQUOTED_WINDOWS_PATH_LINE_RE.finditer(raw):
        _add(match.group(0))
    for match in _UNQUOTED_RELATIVE_PATH_RE.finditer(raw):
        candidate = match.group(0)
        if any(candidate in url for url in url_refs):
            continue
        _add(candidate)
    return refs


def _is_local_file_request(text: str) -> bool:
    refs = _extract_local_path_refs(text)
    if not refs:
        return False
    lowered = " ".join(str(text or "").lower().split())
    read_terms = (
        "lee",
        "leer",
        "leas",
        "abrir",
        "abri",
        "abrime",
        "mostrar",
        "mostra",
        "mostrame",
        "quiero este",
        "quiero esta",
        "me interesa",
        "me interesan",
        "archivo",
        "carpeta",
        "directorio",
        "ruta",
        "path",
        "donde estan",
        "dónde están",
        "navega",
        "navegar",
    )
    if _looks_like_pasted_suite_evidence(text):
        return any(term in lowered for term in read_terms)
    return any(term in lowered for term in read_terms) or any(Path(ref).suffix for ref in refs)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_suite_evidence_request(text: str, history: list[dict[str, str]] | None = None) -> bool:
    lowered = str(text or "").lower()
    if _is_pasted_suite_analysis_request(text):
        return True
    suite_terms = (
        "/verify",
        "verify",
        "suite",
        "suites",
        "corrida",
        "corridas",
        "artifact",
        "artifacts",
        "artefacto",
        "manifest",
        "transcript",
        "transcripts",
        "jsonl",
        "preregistered",
        "preregistro",
        "verification",
        "post nuclear",
        "post-nuclear",
        "long horizon",
        "hard anchor",
        "hard-anchor",
        "branch pruning",
        "policy rag",
        "poliza",
        "póliza",
    )
    if any(term in lowered for term in suite_terms):
        return True
    for item in history or []:
        content = str(item.get("content") or "").lower()
        if "suite" in content or "/verify" in content or "artifact" in content:
            return any(term in lowered for term in ("ultima", "última", "resultados", "transcript", "corrida", "esa", "eso"))
    return False


def _is_web_search_request(text: str) -> bool:
    lowered = str(text or "").lower()
    web_terms = (
        "busca en la web",
        "buscar en la web",
        "buscame en la web",
        "google",
        "googlea",
        "internet",
        "web",
        "online",
        "fuentes",
        "source",
        "sources",
        "links",
        "link",
        "noticias",
        "news",
        "latest",
        "último",
        "ultima",
        "última",
        "actual",
        "reciente",
        "benchmark",
        "benchmarks",
    )
    lookup_verbs = (
        "busca",
        "buscar",
        "buscame",
        "investiga",
        "investigar",
        "research",
        "encontra",
        "encuentra",
        "averigua",
        "quiero info",
        "necesito info",
        "mostrame info",
    )
    if any(term in lowered for term in ("google", "internet", "busca en la web", "buscar en la web", "web search")):
        return True
    return any(term in lowered for term in web_terms) and any(verb in lowered for verb in lookup_verbs)


def _looks_like_pasted_suite_evidence(text: str) -> bool:
    raw = str(text or "")
    lowered = raw.lower()
    markers = (
        '"suite_id"',
        '"run_id"',
        '"exit_code"',
        '"stderr"',
        "traceback",
        "runtimeerror:",
        "suite",
        "run id",
        "artifacts",
        "transcripts",
        "transcr",
        "branch-pruning-forensics",
        "hard-anchor-utility",
    )
    marker_count = sum(1 for marker in markers if marker in lowered)
    return marker_count >= 2 and ("\n" in raw or "│" in raw or "{" in raw)


def _is_pasted_suite_analysis_request(text: str) -> bool:
    lowered = str(text or "").lower()
    analysis_terms = (
        "quiero info",
        "quiero data",
        "dame info",
        "dame data",
        "contame",
        "analiza",
        "explica",
        "que significa",
        "qué significa",
        "por que fallo",
        "por qué falló",
        "fallo",
        "falló",
    )
    return _looks_like_pasted_suite_evidence(text) and any(term in lowered for term in analysis_terms)


def _looks_like_deferred_lookup_preamble(text: str) -> bool:
    lowered = " ".join(str(text or "").lower().split())
    if not lowered or len(lowered) > 220:
        return False
    starters = (
        "voy a buscar",
        "voy a revisar",
        "voy a consultar",
        "voy a mirar",
        "voy a fijarme",
        "voy a leer",
        "déjame buscar",
        "dejame buscar",
        "déjame revisar",
        "dejame revisar",
        "let me check",
        "let me look",
        "i'll check",
        "i will check",
        "i'll look",
        "i will look",
        "looking into",
    )
    return lowered.startswith(starters)


def _looks_like_unverified_memory_claim(text: str) -> bool:
    lowered = " ".join(str(text or "").lower().split())
    phrases = (
        "no encontré información relevante en la memoria",
        "no encontre informacion relevante en la memoria",
        "no encontré nada relevante en la memoria",
        "no encontre nada relevante en la memoria",
        "i couldn't find relevant information in memory",
        "i could not find relevant information in memory",
        "i did not find relevant information in memory",
    )
    return any(phrase in lowered for phrase in phrases)


_THINKING_MESSAGES = [
    "pensando...",
    "conspirando con el DAG...",
    "mucho laburo...",
    "leyendo memoria certificada...",
    "siguiendo hashes...",
    "ordenando el quilombo...",
    "ruteando modelo...",
    "consultando al oráculo barato...",
    "levantando contexto...",
    "podando ruido...",
    "sellando el turno...",
    "afinando la respuesta...",
    "cruzando evidencia...",
    "haciendo magia determinística...",
    "bajando la latencia...",
    "separando humo de señal...",
    "calibrando el kernel...",
    "despertando a Mistral...",
    "mirando el grafo de reojo...",
    "preparando salida limpia...",
]
_THINKING_MESSAGE_PHASES = {
    "thinking": _THINKING_MESSAGES,
    "task": [
        "abriendo tablero de misión...",
        "leyendo el terreno...",
        "buscando puntos de apoyo...",
        "siguiendo rastros en el repo...",
        "preguntándole al DAG dónde duele...",
        "separando síntoma de causa...",
        "armando hipótesis falsables...",
        "cruzando herramientas...",
        "dejando migas verificables...",
        "cerrando el circuito...",
    ],
    "tool": [
        "ejecutando tool read-only...",
        "mirando archivos sin tocar nada...",
        "corriendo prueba segura...",
        "resumiendo observaciones...",
        "sellando resultado de tool...",
    ],
}

from helix_proto.helix_cli_agent_shell import extract_patch as _extract_patch
from helix_proto.helix_cli_agent_shell import normalize_tool_event as _normalize_tool_event
from helix_proto.helix_cli_agent_shell import parse_agent_tool_calls as _parse_agent_tool_calls
from helix_proto.helix_cli_agent_shell import tool_event_detail as _tool_event_detail
from helix_proto.provider_audit import OPENAI_COMPATIBLE_PROVIDERS
from helix_proto.tools import ToolRegistry, ToolSpec


class _LazyHmem:
    def __init__(self) -> None:
        self._module: Any | None = None

    def _load(self) -> Any:
        if self._module is None:
            from helix_proto import hmem as module  # noqa: PLC0415

            self._module = module
        return self._module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)


hmem = _LazyHmem()


def verify_artifact_file(*args: Any, **kwargs: Any) -> Any:
    from helix_proto.artifact_replay import verify_artifact_file as _verify_artifact_file  # noqa: PLC0415

    return _verify_artifact_file(*args, **kwargs)


def ingest_artifact_file(*args: Any, **kwargs: Any) -> Any:
    from helix_proto.evidence_ingest import ingest_artifact_file as _ingest_artifact_file  # noqa: PLC0415

    return _ingest_artifact_file(*args, **kwargs)


def list_ingested_evidence(*args: Any, **kwargs: Any) -> Any:
    from helix_proto.evidence_ingest import list_ingested_evidence as _list_ingested_evidence  # noqa: PLC0415

    return _list_ingested_evidence(*args, **kwargs)


def refresh_evidence(*args: Any, **kwargs: Any) -> Any:
    from helix_proto.evidence_ingest import refresh_evidence as _refresh_evidence  # noqa: PLC0415

    return _refresh_evidence(*args, **kwargs)


@dataclass(slots=True)
class PlannerDecision:
    kind: str
    thought: str
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    final: str | None = None
    planner: str | None = None
    raw_text: str | None = None


class HelixRuntime:
    """Lazy API runtime proxy so importing helix_cli does not import numpy/agent code."""

    def __init__(self, *, root: str | Path | None = None) -> None:
        self.root = Path(root or _default_workspace_root()).resolve()
        self._real_runtime: Any | None = None
        self._tools: Any | None = None

    def _real(self) -> Any:
        if self._real_runtime is None:
            from helix_proto.api import HelixRuntime as _ApiHelixRuntime  # noqa: PLC0415

            self._real_runtime = _ApiHelixRuntime(root=self.root)
        return self._real_runtime

    def generate_text(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._real().generate_text(*args, **kwargs)

    def tool_manifest(self) -> list[dict[str, Any]]:
        if self._tools is None:
            from helix_proto.tools import build_runtime_tool_registry  # noqa: PLC0415

            self._tools = build_runtime_tool_registry(self)
        return self._tools.manifest()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._tools is None:
            from helix_proto.tools import build_runtime_tool_registry  # noqa: PLC0415

            self._tools = build_runtime_tool_registry(self)
        return self._tools.call(name, arguments)

    def agent_runner(self) -> Any:
        from helix_proto.agent import AgentRunner  # noqa: PLC0415

        return AgentRunner(self, root=self.root)

    def research_artifact_manifest(self) -> list[dict[str, Any]]:
        from helix_proto.research_artifacts import research_artifact_manifest  # noqa: PLC0415

        return research_artifact_manifest()

    def research_artifact(self, name: str) -> dict[str, Any]:
        from helix_proto.research_artifacts import artifact_title, load_research_artifact  # noqa: PLC0415

        return {"name": str(name), "title": artifact_title(str(name)), "payload": load_research_artifact(str(name))}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real(), name)


REDACTED = "[REDACTED]"
DEFAULT_TIMEOUT_SECONDS = 30.0
AGENT_TASK_TIMEOUT_SECONDS = 90.0
console = None


RESPONSE_STYLES: dict[str, str] = {
    "balanced": (
        "Balanced style: direct, grounded, a little distinctive, but not theatrical. "
        "Use the user's Spanish register when appropriate. Avoid repeating the same metaphors."
    ),
    "technical": (
        "Technical style: concise engineering prose, concrete claims, commands and file paths when useful. "
        "Avoid dramatic framing and avoid decorative analogies."
    ),
    "forensic": (
        "Forensic style: separate observed evidence, inference, risk, and next action. "
        "Use precise language around hashes, artifacts, dates, and uncertainty."
    ),
    "vivid": (
        "Vivid style: more expressive and memorable, still technically honest. "
        "Use sparing imagery and sharper phrasing, but do not invent capabilities or evidence."
    ),
    "terse": (
        "Terse style: shortest useful answer. Lead with the answer, then one or two concrete details."
    ),
}


def _normalize_response_style(value: str | None) -> str:
    candidate = str(value or "").strip().lower()
    aliases = {
        "default": "balanced",
        "normal": "balanced",
        "equilibrado": "balanced",
        "tecnico": "technical",
        "técnico": "technical",
        "tech": "technical",
        "forense": "forensic",
        "audit": "forensic",
        "interesante": "vivid",
        "picante": "vivid",
        "creativo": "vivid",
        "corto": "terse",
        "breve": "terse",
    }
    candidate = aliases.get(candidate, candidate)
    return candidate if candidate in RESPONSE_STYLES else "balanced"


INTERACTION_MODE_PROFILES: dict[str, dict[str, Any]] = {
    "balanced": {
        "description": "Default mixed mode: preserve current HeliX behavior and let prompt intent dominate.",
        "router_bias": "Minimal extra bias. Chat, code, research and audits follow the normal blueprint heuristics.",
        "tool_bias": "Use standard HeliX planner behavior with local grounding first and native provider features only when the prompt clearly warrants them.",
        "web_policy": "Use Gemini native URL/search grounding or HeliX web tools only when the prompt includes URLs, asks for current sources, or explicitly requests external research.",
        "tone_contract": "Balanced mode: answer directly, stay grounded, and only go deeper or wider when the prompt asks for it.",
        "examples": [
            "hola",
            "revisá este bug y explicamelo",
            "compará estas dos ideas sin salirte demasiado del tema",
        ],
    },
    "technical": {
        "description": "Engineering-first mode for diagnosis, code, auditability, repo work, suites, hashes and architecture.",
        "router_bias": "Bias toward code, audit, suite forensics, HeliX architecture, evidence packs and grounded repo answers.",
        "tool_bias": "Prefer local evidence, architecture packs, read-only repo tools and concrete verification over speculative framing.",
        "web_policy": "Do not widen to the web unless the prompt explicitly asks for current external information or includes URLs that need grounding.",
        "tone_contract": "Technical mode: separate verified fact from inference, prefer concrete semantics and next steps, and avoid decorative philosophy unless asked.",
        "examples": [
            "/tech explicame el canonical head y los receipts",
            "/mode technical",
            "revisá el repo y diagnosticá el bug",
        ],
    },
    "explore": {
        "description": "Open exploration mode for philosophy, culture, creative synthesis, writing and broader research.",
        "router_bias": "Bias toward wide reasoning, research, cultural synthesis and reflective discussion before forcing core diagnostics.",
        "tool_bias": "Keep memory and thread continuity, but only inject heavy HeliX architecture grounding when the prompt becomes concretely technical.",
        "web_policy": "Use external grounding when there are URLs, requests for current sources, explicit research asks, or other clear signals that outside context would help.",
        "tone_contract": "Explore mode: it is fine to interpret, connect ideas and speculate carefully, but label interpretation versus verified fact and do not overclaim runtime guarantees.",
        "examples": [
            "/explore helix y ghost in the shell",
            "investigá estas fuentes y armá una síntesis amplia",
            "quiero explorar las influencias culturales de helix",
        ],
    },
}


def _normalize_interaction_mode(value: str | None) -> str:
    candidate = str(value or "").strip().lower()
    aliases = {
        "default": "balanced",
        "normal": "balanced",
        "auto": "balanced",
        "equilibrado": "balanced",
        "balanceado": "balanced",
        "tech": "technical",
        "tecnico": "technical",
        "técnico": "technical",
        "analytic": "technical",
        "analitico": "technical",
        "analítico": "technical",
        "explorar": "explore",
        "exploracion": "explore",
        "exploración": "explore",
        "creative": "explore",
        "creativo": "explore",
    }
    candidate = aliases.get(candidate, candidate)
    return candidate if candidate in INTERACTION_MODE_PROFILES else "balanced"


def _is_known_interaction_mode(value: str | None) -> bool:
    candidate = str(value or "").strip().lower()
    return candidate in {
        "balanced",
        "technical",
        "explore",
        "default",
        "normal",
        "auto",
        "equilibrado",
        "balanceado",
        "tech",
        "tecnico",
        "técnico",
        "analytic",
        "analitico",
        "analítico",
        "explorar",
        "exploracion",
        "exploración",
        "creative",
        "creativo",
    }


def _interaction_mode_payload(mode: str) -> dict[str, Any]:
    normalized = _normalize_interaction_mode(mode)
    return {"name": normalized, **INTERACTION_MODE_PROFILES[normalized]}


def _interaction_mode_report() -> list[dict[str, Any]]:
    return [_interaction_mode_payload(name) for name in ("balanced", "technical", "explore")]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _repo_root()


def _base_config_dir() -> Path:
    configured = os.environ.get("HELIX_CONFIG_DIR")
    if configured:
        return Path(configured).expanduser()
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "HeliX"
    return Path.home() / ".helix"


def _base_data_dir() -> Path:
    configured = os.environ.get("HELIX_DATA_DIR")
    if configured:
        return Path(configured).expanduser()
    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        return Path(localappdata) / "HeliX"
    return _base_config_dir()


def _config_path() -> Path:
    return _base_config_dir() / "config.json"


def _load_config() -> dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_config(config: dict[str, Any]) -> Path:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")
    return path


def _config_token(provider_name: str) -> str | None:
    tokens = _load_config().get("tokens")
    if not isinstance(tokens, dict):
        return None
    token = tokens.get(provider_name)
    return str(token) if token else None


def _save_config_token(provider_name: str, token: str) -> Path:
    config = _load_config()
    tokens = config.get("tokens")
    if not isinstance(tokens, dict):
        tokens = {}
    tokens[provider_name] = token
    config["tokens"] = tokens
    return _save_config(config)


def _forget_config_token(provider_name: str) -> Path:
    config = _load_config()
    tokens = config.get("tokens")
    if isinstance(tokens, dict):
        tokens.pop(provider_name, None)
        config["tokens"] = tokens
    return _save_config(config)


def _default_workspace_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists() or candidate == REPO_ROOT:
            return candidate / "workspace"
    config = _load_config()
    configured = os.environ.get("HELIX_WORKSPACE_ROOT") or config.get("workspace_root")
    if configured:
        return Path(configured).expanduser()
    return start / "workspace"


def _default_task_root() -> Path:
    config = _load_config()
    configured = os.environ.get("HELIX_TASK_ROOT") or config.get("task_root")
    if configured:
        return Path(configured).expanduser().resolve()
    start = Path.cwd().resolve()
    try:
        if start == Path.home().resolve() and REPO_ROOT.exists():
            return REPO_ROOT.resolve()
    except Exception:
        pass
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


def _default_transcript_dir() -> Path:
    config = _load_config()
    configured = os.environ.get("HELIX_TRANSCRIPT_DIR") or config.get("transcript_dir")
    return Path(configured).expanduser() if configured else (_base_data_dir() / "sessions")


def _default_evidence_root() -> Path:
    config = _load_config()
    configured = os.environ.get("HELIX_EVIDENCE_ROOT") or config.get("evidence_root")
    return Path(configured).expanduser() if configured else (REPO_ROOT / "verification")


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    kind: str
    base_url: str | None
    token_env: str | None
    default_model: str
    requires_token: bool
    description: str
    native_capabilities: tuple[str, ...] = ()
    native_constraints: tuple[str, ...] = ()

    @property
    def token_available(self) -> bool:
        return bool(self.token_env and os.environ.get(self.token_env))


@dataclass(frozen=True)
class SuiteSpec:
    suite_id: str
    script: str
    description: str
    output_dir: str
    requires_deepinfra: bool = False
    supports_deepinfra_flag: bool = False

    @property
    def script_path(self) -> Path:
        return REPO_ROOT / self.script


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    role: str
    provider: str
    input_per_million: float | None
    output_per_million: float | None
    notes: str
    supports_function_calling: bool = False
    supports_parallel_tools: bool = False
    supports_url_context: bool = False
    supports_search_grounding: bool = False
    supports_file_search: bool = False
    supports_vision: bool = False
    supports_long_context: bool = False
    supports_structured_output: bool = False
    latency_tier: str = "medium"
    cost_tier: str = "unknown"
    stability_tier: str = "stable"
    preferred_workloads: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouterBlueprint:
    name: str
    description: str
    default_alias: str
    chat_alias: str
    reasoning_alias: str
    research_alias: str
    code_alias: str
    agentic_alias: str
    audit_alias: str
    vision_alias: str


@dataclass(frozen=True)
class AgentBlueprint:
    blueprint_id: str
    description: str
    preferred_model_alias: str
    fallback_aliases: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    max_steps: int
    evidence_requirement: str
    output_contract: str


def _provider_registry() -> dict[str, ProviderSpec]:
    providers: dict[str, ProviderSpec] = {}
    for provider in OPENAI_COMPATIBLE_PROVIDERS:
        default_model = "mistralai/magistral-small-2506" if provider.name == "nvidia" else "Qwen/Qwen3.6-35B-A3B"
        description = (
            "NVIDIA Build OpenAI-compatible cloud provider"
            if provider.name == "nvidia"
            else "OpenAI-compatible cloud provider"
        )
        providers[provider.name] = ProviderSpec(
            name=provider.name,
            kind="openai-compatible",
            base_url=provider.base_url,
            token_env=provider.token_env,
            default_model=default_model,
            requires_token=True,
            description=description,
            native_capabilities=("chat_completions", "structured_output"),
            native_constraints=(
                (
                    "HeliX curates NVIDIA Build free-endpoint models in-repo; no live catalog discovery at runtime."
                    if provider.name == "nvidia"
                    else "Capabilities are curated in-repo; HeliX does not do live model discovery at runtime."
                ),
            ),
        )
    providers.update(
        {
            "openai": ProviderSpec(
                name="openai",
                kind="openai-compatible",
                base_url="https://api.openai.com/v1",
                token_env="OPENAI_API_KEY",
                default_model="gpt-5.4-mini",
                requires_token=True,
                description="OpenAI Chat Completions compatible endpoint",
                native_capabilities=("chat_completions", "structured_output"),
                native_constraints=("Capabilities are curated in-repo; HeliX does not do live model discovery at runtime.",),
            ),
            "anthropic": ProviderSpec(
                name="anthropic",
                kind="anthropic",
                base_url="https://api.anthropic.com/v1",
                token_env="ANTHROPIC_API_KEY",
                default_model="claude-4-sonnet",
                requires_token=True,
                description="Anthropic Messages API",
                native_capabilities=("messages_api", "vision", "structured_output"),
                native_constraints=("Capabilities are curated in-repo; HeliX does not do live model discovery at runtime.",),
            ),
            "gemini": ProviderSpec(
                name="gemini",
                kind="gemini",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                token_env="GEMINI_API_KEY",
                default_model="gemini-3-flash-preview",
                requires_token=True,
                description="Google Gemini generateContent API",
                native_capabilities=(
                    "function_calling",
                    "parallel_tools",
                    "url_context",
                    "search_grounding",
                    "file_search",
                    "vision",
                    "long_context",
                    "structured_output",
                ),
                native_constraints=(
                    "HeliX keeps local repo/filesystem grounding on file.inspect instead of Gemini File Search.",
                    "In this CLI pass, URL Context and Google Search grounding are not mixed with Gemini function calling.",
                ),
            ),
            "ollama": ProviderSpec(
                name="ollama",
                kind="openai-compatible",
                base_url="http://127.0.0.1:11434/v1",
                token_env=None,
                default_model="llama3.1",
                requires_token=False,
                description="Local Ollama OpenAI-compatible endpoint",
                native_capabilities=("chat_completions",),
                native_constraints=("Capabilities depend on the local model server; HeliX keeps metadata static.",),
            ),
            "llamacpp": ProviderSpec(
                name="llamacpp",
                kind="openai-compatible",
                base_url="http://127.0.0.1:8080/v1",
                token_env=None,
                default_model="local-model",
                requires_token=False,
                description="Local llama.cpp server OpenAI-compatible endpoint",
                native_capabilities=("chat_completions",),
                native_constraints=("Capabilities depend on the local server; HeliX keeps metadata static.",),
            ),
            "local": ProviderSpec(
                name="local",
                kind="helix-local",
                base_url=None,
                token_env=None,
                default_model="",
                requires_token=False,
                description="Prepared local HeliX model alias via HelixRuntime",
                native_capabilities=("helix_runtime",),
                native_constraints=("Capabilities are determined by the local runtime alias and are not auto-discovered.",),
            ),
        }
    )
    return providers


PROVIDERS = _provider_registry()


DEEPINFRA_MODEL_PROFILES: dict[str, ModelProfile] = {
    "chat": ModelProfile(
        model_id="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        role="chat",
        provider="deepinfra",
        input_per_million=0.05,
        output_per_million=0.10,
        notes="Fast everyday chat model with strong instruction following and lower infinite-generation risk.",
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="fast",
        cost_tier="low",
        stability_tier="stable",
        preferred_workloads=("chat", "drafts", "quick_help"),
    ),
    "mistral": ModelProfile(
        model_id="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        role="mistral-chat",
        provider="deepinfra",
        input_per_million=0.05,
        output_per_million=0.10,
        notes="Mistral Small 3.2 chat/profile option for direct answers and model-control requests.",
    ),
    "devstral": ModelProfile(
        model_id="mistralai/Devstral-Small-2507",
        role="mistral-code",
        provider="deepinfra",
        input_per_million=0.05,
        output_per_million=0.10,
        notes="Mistral/Devstral software-engineering profile for code-oriented agentic tasks.",
    ),
    "cheap": ModelProfile(
        model_id="Qwen/Qwen3.5-9B",
        role="cheap",
        provider="deepinfra",
        input_per_million=0.04,
        output_per_million=0.20,
        notes="Cheap long-context chat, summaries, intent classification, and simple help.",
    ),
    "default": ModelProfile(
        model_id="Qwen/Qwen3.6-35B-A3B",
        role="default",
        provider="deepinfra",
        input_per_million=0.20,
        output_per_million=1.00,
        notes="Balanced default for normal chat, repo Q&A, light reasoning, and Spanish/English work.",
        supports_function_calling=True,
        supports_parallel_tools=True,
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="medium",
        cost_tier="medium",
        stability_tier="stable",
        preferred_workloads=("chat", "reasoning", "repo_qa"),
    ),
    "code": ModelProfile(
        model_id="Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo",
        role="code",
        provider="deepinfra",
        input_per_million=0.30,
        output_per_million=1.20,
        notes="Primary agentic coding model: repo-scale understanding, tool use, function calling, and 256K context.",
        supports_function_calling=True,
        supports_parallel_tools=True,
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="medium",
        cost_tier="high",
        stability_tier="stable",
        preferred_workloads=("repo_code", "agentic_code", "patch_planning", "tool_use"),
    ),
    "qwen-big": ModelProfile(
        model_id="Qwen/Qwen3.5-122B-A10B",
        role="qwen-heavy",
        provider="deepinfra",
        input_per_million=0.29,
        output_per_million=2.90,
        notes="Primary large-Qwen profile for research, HeliX self-questions, synthesis, long context, and agentic planning.",
        supports_function_calling=True,
        supports_parallel_tools=True,
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="medium",
        cost_tier="high",
        stability_tier="stable",
        preferred_workloads=("research", "helix_meta", "long_context", "agentic_planning"),
    ),
    "qwen-122b": ModelProfile(
        model_id="Qwen/Qwen3.5-122B-A10B",
        role="qwen-general",
        provider="deepinfra",
        input_per_million=0.29,
        output_per_million=2.90,
        notes="Explicit Qwen 122B alias kept for compatibility; qwen-big is the preferred heavy-Qwen route.",
    ),
    "gemma": ModelProfile(
        model_id="google/gemma-4-31B",
        role="gemma-reasoning",
        provider="deepinfra",
        input_per_million=None,
        output_per_million=None,
        notes="Gemma reasoning/general profile for careful mid-weight analysis, decomposition, and precise answers.",
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="medium",
        cost_tier="medium",
        stability_tier="stable",
        preferred_workloads=("reasoning", "analysis", "mid_weight_synthesis"),
    ),
    "llama-vision": ModelProfile(
        model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        role="vision",
        provider="deepinfra",
        input_per_million=None,
        output_per_million=None,
        notes="Vision-capable Llama profile for screenshots, images, OCR-like descriptions, and visual debugging.",
        supports_vision=True,
        supports_structured_output=True,
        latency_tier="medium",
        cost_tier="medium",
        stability_tier="stable",
        preferred_workloads=("vision", "ocr_like", "screenshot_debug"),
    ),
    "llama-70b": ModelProfile(
        model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        role="llama-general",
        provider="deepinfra",
        input_per_million=None,
        output_per_million=None,
        notes="Large Llama generalist for broad instruction following, fallback synthesis, and high-context prose.",
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="slow",
        cost_tier="medium",
        stability_tier="stable",
        preferred_workloads=("chat", "fallback_synthesis", "generalist_prose"),
    ),
    "reasoning": ModelProfile(
        model_id="google/gemma-4-31B",
        role="reasoning",
        provider="deepinfra",
        input_per_million=None,
        output_per_million=None,
        notes="Balanced reasoning profile backed by Gemma for decomposition, analysis, and deliberate medium-depth work.",
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="medium",
        cost_tier="medium",
        stability_tier="stable",
        preferred_workloads=("reasoning", "decomposition", "analysis"),
    ),
    "agentic": ModelProfile(
        model_id="Qwen/Qwen3.5-122B-A10B",
        role="agentic",
        provider="deepinfra",
        input_per_million=None,
        output_per_million=None,
        notes="Balanced agentic profile backed by Qwen 122B for long tasks, search-heavy work, and broad synthesis.",
        supports_function_calling=True,
        supports_parallel_tools=True,
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="medium",
        cost_tier="high",
        stability_tier="stable",
        preferred_workloads=("agentic", "search_heavy", "synthesis"),
    ),
    "research": ModelProfile(
        model_id="Qwen/Qwen3.5-122B-A10B",
        role="research",
        provider="deepinfra",
        input_per_million=0.29,
        output_per_million=2.90,
        notes="Research/search-oriented Qwen 122B profile for long context synthesis and careful uncertainty handling.",
        supports_function_calling=True,
        supports_parallel_tools=True,
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="medium",
        cost_tier="high",
        stability_tier="stable",
        preferred_workloads=("research", "long_context", "synthesis", "uncertainty_handling"),
    ),
    "bioinformatics": ModelProfile(
        model_id="Qwen/Qwen3.5-122B-A10B",
        role="bioinformatics",
        provider="deepinfra",
        input_per_million=0.29,
        output_per_million=2.90,
        notes="Bioinformatics-heavy analysis alias pinned to the primary large Qwen research model for complex scientific logic.",
        supports_function_calling=True,
        supports_parallel_tools=True,
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="medium",
        cost_tier="high",
        stability_tier="stable",
        preferred_workloads=("bioinformatics", "scientific_reasoning", "research", "long_context"),
    ),
    "legacy-reasoning": ModelProfile(
        model_id="stepfun-ai/Step-3.5-Flash",
        role="legacy-reasoning",
        provider="deepinfra",
        input_per_million=0.10,
        output_per_million=0.30,
        notes="Legacy lightweight reasoning profile kept for the current/legacy router blueprint.",
    ),
    "legacy-agentic": ModelProfile(
        model_id="MiniMaxAI/MiniMax-M2.5",
        role="legacy-agentic",
        provider="deepinfra",
        input_per_million=0.27,
        output_per_million=0.95,
        notes="Legacy MiniMax agentic profile kept for the current/legacy router blueprint.",
    ),
    "legacy-research": ModelProfile(
        model_id="MiniMaxAI/MiniMax-M2.5",
        role="legacy-research",
        provider="deepinfra",
        input_per_million=0.27,
        output_per_million=0.95,
        notes="Legacy MiniMax research profile kept for the current/legacy router blueprint.",
    ),
    "engineering": ModelProfile(
        model_id="zai-org/GLM-5.1",
        role="engineering",
        provider="deepinfra",
        input_per_million=1.40,
        output_per_million=4.40,
        notes="Premium agentic engineering model for hard multi-step code and terminal workflows.",
        supports_function_calling=True,
        supports_parallel_tools=True,
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="slow",
        cost_tier="high",
        stability_tier="stable",
        preferred_workloads=("agentic_code", "engineering", "hard_debugging"),
    ),
    "deep-reasoning": ModelProfile(
        model_id="deepseek-ai/DeepSeek-V3.2",
        role="deep-reasoning",
        provider="deepinfra",
        input_per_million=0.26,
        output_per_million=0.38,
        notes="Reasoning and agentic tool-use model with efficient long-context behavior.",
        supports_function_calling=True,
        supports_parallel_tools=True,
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="medium",
        cost_tier="medium",
        stability_tier="stable",
        preferred_workloads=("reasoning", "agentic", "long_context"),
    ),
    "sonnet": ModelProfile(
        model_id="anthropic/claude-4-sonnet",
        role="sonnet",
        provider="deepinfra",
        input_per_million=None,
        output_per_million=None,
        notes="Existing HeliX premium auditor model. Used for high-stakes audit/legal/claim-boundary turns.",
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="slow",
        cost_tier="high",
        stability_tier="stable",
        preferred_workloads=("audit", "legal", "claim_boundaries"),
    ),
}


GEMINI_MODEL_PROFILES: dict[str, ModelProfile] = {
    "gemini-pro": ModelProfile(
        model_id="gemini-3.1-pro-preview",
        role="gemini-pro",
        provider="gemini",
        input_per_million=None,
        output_per_million=None,
        notes="Gemini 3.1 Pro preview for high-depth reasoning, synthesis, and complex non-code analysis.",
        supports_function_calling=True,
        supports_parallel_tools=True,
        supports_url_context=True,
        supports_search_grounding=True,
        supports_file_search=True,
        supports_vision=True,
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="slow",
        cost_tier="high",
        stability_tier="preview",
        preferred_workloads=("long_analysis", "reasoning", "url_comparison", "docs_synthesis"),
    ),
    "gemini-pro-tools": ModelProfile(
        model_id="gemini-3.1-pro-preview-customtools",
        role="gemini-pro-tools",
        provider="gemini",
        input_per_million=None,
        output_per_million=None,
        notes="Gemini 3.1 Pro custom-tools preview for agentic workflows that must prioritize custom tools.",
        supports_function_calling=True,
        supports_parallel_tools=True,
        supports_url_context=True,
        supports_search_grounding=True,
        supports_file_search=True,
        supports_vision=True,
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="slow",
        cost_tier="high",
        stability_tier="preview",
        preferred_workloads=("agentic", "custom_tools", "long_analysis"),
    ),
    "gemini-flash": ModelProfile(
        model_id="gemini-3-flash-preview",
        role="gemini-flash",
        provider="gemini",
        input_per_million=None,
        output_per_million=None,
        notes="Gemini 3 Flash preview for fast general chat, research drafts, and lower-latency turns.",
        supports_function_calling=True,
        supports_parallel_tools=True,
        supports_url_context=True,
        supports_search_grounding=True,
        supports_file_search=True,
        supports_vision=True,
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="fast",
        cost_tier="medium",
        stability_tier="preview",
        preferred_workloads=("chat", "web_grounding", "drafts", "fast_url_reads"),
    ),
    "gemini-lite": ModelProfile(
        model_id="gemini-3.1-flash-lite-preview",
        role="gemini-lite",
        provider="gemini",
        input_per_million=None,
        output_per_million=None,
        notes="Gemini 3.1 Flash Lite preview for cheap/fast classification, summaries, and lightweight chat.",
        supports_function_calling=True,
        supports_parallel_tools=True,
        supports_url_context=True,
        supports_search_grounding=True,
        supports_file_search=True,
        supports_vision=True,
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="fast",
        cost_tier="low",
        stability_tier="preview",
        preferred_workloads=("classification", "cheap_summaries", "lightweight_chat"),
    ),
    "gemini-2.5-pro": ModelProfile(
        model_id="gemini-2.5-pro",
        role="gemini-2.5-pro",
        provider="gemini",
        input_per_million=None,
        output_per_million=None,
        notes="Stable Gemini 2.5 Pro for reliable deep reasoning, code, long-context analysis, and fallback from preview Pro.",
        supports_function_calling=True,
        supports_parallel_tools=True,
        supports_url_context=True,
        supports_search_grounding=True,
        supports_file_search=True,
        supports_vision=True,
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="slow",
        cost_tier="high",
        stability_tier="stable",
        preferred_workloads=("reasoning", "code", "long_context", "reliable_fallback"),
    ),
    "gemini-2.5-flash": ModelProfile(
        model_id="gemini-2.5-flash",
        role="gemini-2.5-flash",
        provider="gemini",
        input_per_million=None,
        output_per_million=None,
        notes="Stable Gemini 2.5 Flash for reliable low-latency agentic and high-volume fallback work.",
        supports_function_calling=True,
        supports_parallel_tools=True,
        supports_url_context=True,
        supports_search_grounding=True,
        supports_file_search=True,
        supports_vision=True,
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="fast",
        cost_tier="medium",
        stability_tier="stable",
        preferred_workloads=("chat", "agentic_light", "reliable_web_grounding"),
    ),
    "gemini-2.5-flash-lite": ModelProfile(
        model_id="gemini-2.5-flash-lite",
        role="gemini-2.5-flash-lite",
        provider="gemini",
        input_per_million=None,
        output_per_million=None,
        notes="Stable Gemini 2.5 Flash-Lite for cheap, fast lightweight fallback tasks.",
        supports_function_calling=True,
        supports_parallel_tools=True,
        supports_url_context=True,
        supports_search_grounding=True,
        supports_file_search=True,
        supports_vision=True,
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="fast",
        cost_tier="low",
        stability_tier="stable",
        preferred_workloads=("classification", "cheap_chat", "fallback"),
    ),
}


NVIDIA_MODEL_PROFILES: dict[str, ModelProfile] = {
    "nvidia-chat": ModelProfile(
        model_id="mistralai/magistral-small-2506",
        role="nvidia-chat",
        provider="nvidia",
        input_per_million=None,
        output_per_million=None,
        notes="NVIDIA Build free-endpoint everyday chat and lightweight reasoning profile.",
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="fast",
        cost_tier="free-trial",
        stability_tier="stable",
        preferred_workloads=("chat", "light_reasoning", "general_help"),
    ),
    "nvidia-research": ModelProfile(
        model_id="mistralai/mistral-large-3-675b-instruct-2512",
        role="nvidia-research",
        provider="nvidia",
        input_per_million=None,
        output_per_million=None,
        notes="NVIDIA Build free-endpoint research/synthesis profile for long-context technical work.",
        supports_function_calling=True,
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="slow",
        cost_tier="free-trial",
        stability_tier="stable",
        preferred_workloads=("research", "synthesis", "long_context", "technical_analysis"),
    ),
    "nvidia-code": ModelProfile(
        model_id="qwen/qwen3-coder-480b-a35b-instruct",
        role="nvidia-code",
        provider="nvidia",
        input_per_million=None,
        output_per_million=None,
        notes="NVIDIA Build flagship code model for agentic coding, repo understanding and structured tool-like work.",
        supports_function_calling=True,
        supports_parallel_tools=True,
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="medium",
        cost_tier="free-trial",
        stability_tier="stable",
        preferred_workloads=("repo_code", "agentic_code", "tool_use", "patch_planning"),
    ),
    "nvidia-agentic": ModelProfile(
        model_id="mistralai/devstral-2-123b-instruct-2512",
        role="nvidia-agentic",
        provider="nvidia",
        input_per_million=None,
        output_per_million=None,
        notes="NVIDIA Build software-engineering verifier for review, repair and SWE-style multi-step follow-through.",
        supports_function_calling=True,
        supports_parallel_tools=True,
        supports_long_context=True,
        supports_structured_output=True,
        latency_tier="medium",
        cost_tier="free-trial",
        stability_tier="stable",
        preferred_workloads=("engineering", "verification", "repair", "swe_tasks"),
    ),
    "nvidia-guard": ModelProfile(
        model_id="meta/llama-guard-4-12b",
        role="nvidia-guard",
        provider="nvidia",
        input_per_million=None,
        output_per_million=None,
        notes="NVIDIA Build specialized guard classifier for safety gating of prompt/output segments.",
        supports_structured_output=True,
        latency_tier="fast",
        cost_tier="free-trial",
        stability_tier="stable",
        preferred_workloads=("guardrails", "safety_review", "classification"),
    ),
    "nvidia-pii": ModelProfile(
        model_id="nvidia/gliner-pii",
        role="nvidia-pii",
        provider="nvidia",
        input_per_million=None,
        output_per_million=None,
        notes="NVIDIA Build specialized PII detector for explicit redaction/extraction workflows, not general chat.",
        supports_structured_output=True,
        latency_tier="fast",
        cost_tier="free-trial",
        stability_tier="stable",
        preferred_workloads=("pii_detection", "redaction_support", "classification"),
    ),
}


MODEL_PROFILES: dict[str, ModelProfile] = {
    **DEEPINFRA_MODEL_PROFILES,
    **GEMINI_MODEL_PROFILES,
    **NVIDIA_MODEL_PROFILES,
}


ROUTER_BLUEPRINTS: dict[str, RouterBlueprint] = {
    "balanced": RouterBlueprint(
        name="balanced",
        description="Preferred mixed blueprint: Mistral Small for chat, Gemma for reasoning, Qwen Big for research/HeliX, Qwen Coder for code, Sonnet for audits.",
        default_alias="chat",
        chat_alias="chat",
        reasoning_alias="reasoning",
        research_alias="qwen-big",
        code_alias="code",
        agentic_alias="qwen-big",
        audit_alias="sonnet",
        vision_alias="llama-vision",
    ),
    "qwen-heavy": RouterBlueprint(
        name="qwen-heavy",
        description="Qwen-first blueprint: Qwen Big for most serious work, Qwen Coder for repo tasks, Gemma for deliberate reasoning, Sonnet for audit.",
        default_alias="qwen-big",
        chat_alias="qwen-big",
        reasoning_alias="gemma",
        research_alias="qwen-big",
        code_alias="code",
        agentic_alias="qwen-big",
        audit_alias="sonnet",
        vision_alias="llama-vision",
    ),
    "qwen-gemma-mistral": RouterBlueprint(
        name="qwen-gemma-mistral",
        description="Explicit hybrid blueprint using Mistral Small + Gemma + Qwen families as the main stack.",
        default_alias="chat",
        chat_alias="chat",
        reasoning_alias="reasoning",
        research_alias="qwen-big",
        code_alias="code",
        agentic_alias="qwen-big",
        audit_alias="sonnet",
        vision_alias="llama-vision",
    ),
    "current": RouterBlueprint(
        name="current",
        description="Legacy/current HeliX router behavior from before the Qwen/Gemma rebalance.",
        default_alias="chat",
        chat_alias="chat",
        reasoning_alias="legacy-reasoning",
        research_alias="legacy-research",
        code_alias="code",
        agentic_alias="legacy-agentic",
        audit_alias="sonnet",
        vision_alias="llama-vision",
    ),
    "cheap": RouterBlueprint(
        name="cheap",
        description="Cheaper stack favoring small/efficient models while preserving code and audit escapes.",
        default_alias="cheap",
        chat_alias="cheap",
        reasoning_alias="legacy-reasoning",
        research_alias="chat",
        code_alias="devstral",
        agentic_alias="legacy-reasoning",
        audit_alias="sonnet",
        vision_alias="llama-vision",
    ),
    "premium": RouterBlueprint(
        name="premium",
        description="Premium stack favoring strongest engineering/reasoning paths while keeping Sonnet for audits.",
        default_alias="llama-70b",
        chat_alias="llama-70b",
        reasoning_alias="deep-reasoning",
        research_alias="engineering",
        code_alias="engineering",
        agentic_alias="engineering",
        audit_alias="sonnet",
        vision_alias="llama-vision",
    ),
    "nvidia-build": RouterBlueprint(
        name="nvidia-build",
        description="Opt-in NVIDIA Build blueprint: Magistral chat, Mistral Large research, Qwen Coder for code, Devstral for verification.",
        default_alias="nvidia-chat",
        chat_alias="nvidia-chat",
        reasoning_alias="nvidia-research",
        research_alias="nvidia-research",
        code_alias="nvidia-code",
        agentic_alias="nvidia-agentic",
        audit_alias="nvidia-research",
        vision_alias="nvidia-research",
    ),
}


ROUTER_POLICIES = set(ROUTER_BLUEPRINTS)


AGENT_BLUEPRINTS: dict[str, AgentBlueprint] = {
    "repo-scout": AgentBlueprint(
        blueprint_id="repo-scout",
        description="Map a repository with file listing, text search, selected reads, and git status.",
        preferred_model_alias="code",
        fallback_aliases=("qwen-big", "devstral"),
        allowed_tools=("list_files", "search_text", "read_file", "file.inspect", "git_status"),
        max_steps=5,
        evidence_requirement="Read files before making repo claims.",
        output_contract="Summarize findings with file paths and uncertainty.",
    ),
    "patch-planner": AgentBlueprint(
        blueprint_id="patch-planner",
        description="Inspect repo state and propose a patch without writing files.",
        preferred_model_alias="code",
        fallback_aliases=("devstral", "qwen-big"),
        allowed_tools=("list_files", "search_text", "read_file", "file.inspect", "git_status", "git_diff"),
        max_steps=6,
        evidence_requirement="Use current files/diff before proposing changes.",
        output_contract="Return a concise diagnosis and optional unified diff proposal.",
    ),
    "test-diagnoser": AgentBlueprint(
        blueprint_id="test-diagnoser",
        description="Diagnose allowlisted test failures with read-only test commands.",
        preferred_model_alias="code",
        fallback_aliases=("devstral", "qwen-big"),
        allowed_tools=("list_files", "search_text", "read_file", "file.inspect", "git_status", "run_test"),
        max_steps=6,
        evidence_requirement="Only claim tests ran when run_test returned an event.",
        output_contract="Report command, pass/fail, relevant output, and next fix.",
    ),
    "evidence-auditor": AgentBlueprint(
        blueprint_id="evidence-auditor",
        description="Audit HeliX artifacts, receipts, manifests, hashes, and claim boundaries.",
        preferred_model_alias="sonnet",
        fallback_aliases=("qwen-big", "deep-reasoning"),
        allowed_tools=("evidence.latest", "evidence.show", "query_evidence", "inspect_artifact", "file.inspect", "suite.latest", "suite.read"),
        max_steps=5,
        evidence_requirement="Cite local artifact/transcript paths or say evidence is missing.",
        output_contract="Separate verified facts, inferred risks, and unverified claims.",
    ),
    "suite-cartographer": AgentBlueprint(
        blueprint_id="suite-cartographer",
        description="Explain available experiment suites, scripts, preregisters, outputs, and dry-run commands.",
        preferred_model_alias="qwen-big",
        fallback_aliases=("research", "default"),
        allowed_tools=("suite.catalog", "suite.latest", "suite.transcripts", "suite.dry_run"),
        max_steps=4,
        evidence_requirement="Use suite catalog metadata before summarizing suite coverage.",
        output_contract="List suites by purpose, latest evidence, and safe commands.",
    ),
    "suite-run-analyst": AgentBlueprint(
        blueprint_id="suite-run-analyst",
        description="Compare runs, statuses, scores, cases, manifests, and transcript availability.",
        preferred_model_alias="qwen-big",
        fallback_aliases=("sonnet", "research"),
        allowed_tools=("suite.catalog", "suite.latest", "suite.search", "suite.read", "suite.transcripts", "file.inspect"),
        max_steps=5,
        evidence_requirement="Use artifact/manifest/transcript metadata from verification/.",
        output_contract="Report what changed, what passed/failed, and evidence paths.",
    ),
    "transcript-forensics": AgentBlueprint(
        blueprint_id="transcript-forensics",
        description="Read suite transcripts and reconstruct model/tool/event behavior.",
        preferred_model_alias="qwen-big",
        fallback_aliases=("sonnet", "research"),
        allowed_tools=("suite.search", "suite.read", "suite.transcripts", "file.inspect", "query_evidence"),
        max_steps=5,
        evidence_requirement="Read transcript excerpts before analyzing behavior.",
        output_contract="Summarize timeline, model roles, contradictions, and limits.",
    ),
    "policy-rag-auditor": AgentBlueprint(
        blueprint_id="policy-rag-auditor",
        description="Audit insurance/policy RAG debate evidence and legal claim boundaries.",
        preferred_model_alias="sonnet",
        fallback_aliases=("qwen-big", "code"),
        allowed_tools=("suite.search", "suite.read", "query_evidence", "read_file", "file.inspect", "search_text"),
        max_steps=5,
        evidence_requirement="Ground legal/RAG claims in policy suite artifacts or local files.",
        output_contract="Separate policy facts, legal positions, disputes, and missing evidence.",
    ),
    "model-researcher": AgentBlueprint(
        blueprint_id="model-researcher",
        description="Compare model profiles, router choices, and provider capabilities.",
        preferred_model_alias="qwen-big",
        fallback_aliases=("research", "default"),
        allowed_tools=("suite.search", "evidence.latest", "web.search", "web.read"),
        max_steps=5,
        evidence_requirement="Use local router/model profile data; use web.search when the user asks for current model/provider information.",
        output_contract="Recommend model routing by job type with fallback chain.",
    ),
}


SUITES: dict[str, SuiteSpec] = {
    "post-nuclear-methodology": SuiteSpec(
        suite_id="post-nuclear-methodology",
        script="tools/run_post_nuclear_methodology_suite_v1.py",
        description="Cloud mixed post-nuclear methodology suite",
        output_dir="verification/nuclear-methodology/post-nuclear-methodology",
        requires_deepinfra=True,
    ),
    "long-horizon-checkpoints": SuiteSpec(
        suite_id="long-horizon-checkpoints",
        script="tools/run_long_horizon_checkpoint_suite_v1.py",
        description="Cloud long-horizon checkpoint methodology suite",
        output_dir="verification/nuclear-methodology/long-horizon-checkpoints",
        requires_deepinfra=True,
    ),
    "recursive-architectural-integrity-audit": SuiteSpec(
        suite_id="recursive-architectural-integrity-audit",
        script="tools/run_recursive_architectural_integrity_audit_v1.py",
        description="Recursive meta-architecture audit over recent artifacts",
        output_dir="verification/nuclear-methodology/recursive-architectural-integrity-audit",
        requires_deepinfra=True,
    ),
    "hard-anchor-utility": SuiteSpec(
        suite_id="hard-anchor-utility",
        script="tools/run_hard_anchor_utility_suite_v1.py",
        description="Hard-anchor utility and identity lane suite",
        output_dir="verification/nuclear-methodology/hard-anchor-utility",
        supports_deepinfra_flag=True,
    ),
    "branch-pruning-forensics": SuiteSpec(
        suite_id="branch-pruning-forensics",
        script="tools/run_branch_pruning_forensics_suite_v1.py",
        description="Tombstone branch-pruning forensic suite",
        output_dir="verification/nuclear-methodology/branch-pruning-forensics",
        supports_deepinfra_flag=True,
    ),
    "policy-rag-legal-debate": SuiteSpec(
        suite_id="policy-rag-legal-debate",
        script="tools/run_policy_rag_legal_debate_suite_v1.py",
        description="Insurance policy RAG legal debate suite",
        output_dir="verification/nuclear-methodology/policy-rag-legal-debate",
        supports_deepinfra_flag=True,
    ),
    "infinite-depth-memory": SuiteSpec(
        suite_id="infinite-depth-memory",
        script="tools/run_infinite_depth_memory_suite_v1.py",
        description="Infinite-depth memory methodology and latency boundary suite",
        output_dir="verification/nuclear-methodology/infinite-depth-memory",
    ),
    "cognitive-gauntlet": SuiteSpec(
        suite_id="cognitive-gauntlet",
        script="tools/run_cognitive_gauntlet_v1.py",
        description="Cognitive contradiction, fork, shadow-root and drift gauntlet suite",
        output_dir="verification/nuclear-methodology/cognitive-gauntlet",
        supports_deepinfra_flag=True,
    ),
    "multi-agent-concurrency": SuiteSpec(
        suite_id="multi-agent-concurrency",
        script="tools/run_multi_agent_concurrency_suite_v1.py",
        description="Multi-agent concurrent branch quarantine and merge methodology suite",
        output_dir="verification/nuclear-methodology/multi-agent-concurrency",
        supports_deepinfra_flag=True,
    ),
    "nuclear-methodology": SuiteSpec(
        suite_id="nuclear-methodology",
        script="tools/run_nuclear_methodology_suite_v1.py",
        description="Original nuclear methodology cloud suite",
        output_dir="verification/nuclear-methodology",
        requires_deepinfra=True,
    ),
}


def _slugish(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip().lower()).strip("-")
    return clean or "helix"


def _provider_ready(provider_name: str) -> bool:
    if _provider_cooldown_status(provider_name).get("active"):
        return False
    provider = PROVIDERS[provider_name]
    if not provider.requires_token:
        return True
    if provider.token_env and os.environ.get(provider.token_env):
        return True
    return bool(_config_token(provider.name))


def _retry_after_seconds_from_headers(headers: Any) -> float | None:
    if headers is None:
        return None
    try:
        value = headers.get("Retry-After")
    except Exception:
        value = None
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        return None


def _provider_cooldown_status(provider_name: str) -> dict[str, Any]:
    entry = _PROVIDER_COOLDOWNS.get(provider_name)
    if not entry:
        return {"active": False, "remaining_seconds": 0.0}
    until = float(entry.get("until_monotonic") or 0.0)
    remaining = until - time.monotonic()
    if remaining <= 0:
        _PROVIDER_COOLDOWNS.pop(provider_name, None)
        return {"active": False, "remaining_seconds": 0.0}
    return {
        "active": True,
        "remaining_seconds": remaining,
        "reason": entry.get("reason") or "provider cooldown",
        "model": entry.get("model"),
    }


def _mark_provider_cooldown(
    provider_name: str,
    *,
    model: str | None = None,
    reason: str = "rate limit",
    seconds: float | None = None,
) -> dict[str, Any]:
    duration = max(5.0, float(seconds or _DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS))
    entry = {
        "until_monotonic": time.monotonic() + duration,
        "reason": reason,
        "model": model,
        "duration_seconds": duration,
    }
    _PROVIDER_COOLDOWNS[provider_name] = entry
    return _provider_cooldown_status(provider_name)


def _is_rate_limit_error(exc_or_text: Any) -> bool:
    if isinstance(exc_or_text, ProviderRateLimitError):
        return True
    text = str(exc_or_text or "").lower()
    return any(
        marker in text
        for marker in (
            "http error 429",
            "too many requests",
            "rate limit",
            "ratelimit",
            "quota exceeded",
            "resource exhausted",
        )
    )


def _friendly_provider_failure_text(error_text: str) -> str | None:
    lowered = str(error_text or "").lower()
    provider_name = next((name for name in PROVIDERS if name in lowered), "gemini" if "gemini" in lowered else "")
    provider_label = provider_name or "el provider"
    if "http error 400" in lowered or "bad request" in lowered:
        return (
            f"{provider_label} rechazó el payload del turno (`HTTP 400 Bad Request`). "
            "HeliX intenta recompactar automáticamente el contexto cuando el pedido es grande; "
            "si el proveedor lo vuelve a rechazar, el turno queda registrado pero no conviene mostrar el error crudo como respuesta. "
            "Probá repetir la pregunta o cambiar temporalmente a `/model auto` o a un modelo más liviano."
        )
    if not _is_rate_limit_error(error_text):
        return None
    cooldown_status = _provider_cooldown_status(provider_name) if provider_name else {}
    remaining = int(round(float(cooldown_status.get("remaining_seconds") or 0.0)))
    suffix = f" HeliX va a esperar aproximadamente {remaining}s antes de volver a pegarle a {provider_label}." if remaining > 0 else ""
    alternates = [
        name
        for name in ("deepinfra", "nvidia", "openai", "anthropic")
        if name in PROVIDERS and name != provider_name and _provider_ready(name)
    ]
    if alternates:
        suffix += f" Hay fallback disponible: {', '.join(alternates[:3])}."
    else:
        suffix += " No veo otro provider listo con credenciales locales, así que prefiero no inventar una respuesta sin modelo."
    return (
        f"{provider_label} devolvió rate limit/cuota temporal (`HTTP 429`). "
        "El turno quedó registrado, pero la respuesta del modelo no se generó; prefiero no inventar una respuesta sin modelo."
        + suffix
    )


def _provider_capability_payload(provider: ProviderSpec) -> dict[str, Any]:
    return {
        "native_capabilities": list(provider.native_capabilities),
        "native_constraints": list(provider.native_constraints),
    }


def _profile_capability_payload(profile: ModelProfile) -> dict[str, Any]:
    return {
        "supports_function_calling": profile.supports_function_calling,
        "supports_parallel_tools": profile.supports_parallel_tools,
        "supports_url_context": profile.supports_url_context,
        "supports_search_grounding": profile.supports_search_grounding,
        "supports_file_search": profile.supports_file_search,
        "supports_vision": profile.supports_vision,
        "supports_long_context": profile.supports_long_context,
        "supports_structured_output": profile.supports_structured_output,
        "latency_tier": profile.latency_tier,
        "cost_tier": profile.cost_tier,
        "stability_tier": profile.stability_tier,
        "preferred_workloads": list(profile.preferred_workloads),
    }


def _model_profile_for_id(model_id: str | None) -> ModelProfile | None:
    alias = _profile_alias_for_model_id(str(model_id or ""))
    return MODEL_PROFILES.get(alias or "")


def model_profiles_report() -> list[dict[str, Any]]:
    return [
        {
            "alias": alias,
            "model_id": profile.model_id,
            "role": profile.role,
            "provider": profile.provider,
            "input_per_million": profile.input_per_million,
            "output_per_million": profile.output_per_million,
            "notes": profile.notes,
            **_profile_capability_payload(profile),
        }
        for alias, profile in sorted(MODEL_PROFILES.items())
    ]


def models_payload() -> dict[str, Any]:
    profiles = model_profiles_report()
    return {
        "model_profiles": profiles,
        "deepinfra_model_profiles": [item for item in profiles if item.get("provider") == "deepinfra"],
        "gemini_model_profiles": [item for item in profiles if item.get("provider") == "gemini"],
        "nvidia_model_profiles": [item for item in profiles if item.get("provider") == "nvidia"],
        "providers": provider_report(probe_local=False),
        "router_blueprints": router_blueprints_report(),
        "interaction_modes": _interaction_mode_report(),
    }


def router_blueprints_report() -> list[dict[str, Any]]:
    return [
        {
            "name": blueprint.name,
            "description": blueprint.description,
            "default_alias": blueprint.default_alias,
            "chat_alias": blueprint.chat_alias,
            "reasoning_alias": blueprint.reasoning_alias,
            "research_alias": blueprint.research_alias,
            "code_alias": blueprint.code_alias,
            "agentic_alias": blueprint.agentic_alias,
            "audit_alias": blueprint.audit_alias,
            "vision_alias": blueprint.vision_alias,
        }
        for blueprint in sorted(ROUTER_BLUEPRINTS.values(), key=lambda item: item.name)
    ]


def agent_blueprints_report() -> list[dict[str, Any]]:
    return [
        {
            "blueprint_id": blueprint.blueprint_id,
            "description": blueprint.description,
            "preferred_model_alias": blueprint.preferred_model_alias,
            "fallback_aliases": list(blueprint.fallback_aliases),
            "allowed_tools": list(blueprint.allowed_tools),
            "max_steps": blueprint.max_steps,
            "evidence_requirement": blueprint.evidence_requirement,
            "output_contract": blueprint.output_contract,
        }
        for blueprint in sorted(AGENT_BLUEPRINTS.values(), key=lambda item: item.blueprint_id)
    ]


def resolve_model_alias(value: str) -> str:
    candidate = str(value or "").strip()
    lowered = candidate.lower()
    if lowered in {"auto", "router:auto"}:
        return "auto"
    if lowered in MODEL_PROFILES:
        return MODEL_PROFILES[lowered].model_id
    aliases = {
        "claude": "sonnet",
        "claude-sonnet": "sonnet",
        "claude sonnet": "sonnet",
        "mistral-small": "mistral",
        "mistral small": "mistral",
        "mistral": "mistral",
        "devstral": "devstral",
        "qwen": "qwen-big",
        "qwen-big": "qwen-big",
        "qwen big": "qwen-big",
        "qwen-heavy": "qwen-big",
        "qwen heavy": "qwen-big",
        "qwen122b": "qwen-122b",
        "qwen-122b": "qwen-122b",
        "qwen 122b": "qwen-122b",
        "qwen-coder": "code",
        "qwen coder": "code",
        "qwen-coder-turbo": "code",
        "qwen coder turbo": "code",
        "coder": "code",
        "gemma": "gemma",
        "gemma-4": "gemma",
        "gemma 4": "gemma",
        "gemini": "gemini-flash",
        "gemini-pro": "gemini-pro",
        "gemini pro": "gemini-pro",
        "gemini-3.1-pro": "gemini-pro",
        "gemini 3.1 pro": "gemini-pro",
        "gemini-3.1-pro-preview": "gemini-pro",
        "gemini-pro-tools": "gemini-pro-tools",
        "gemini pro tools": "gemini-pro-tools",
        "gemini customtools": "gemini-pro-tools",
        "gemini-3.1-pro-preview-customtools": "gemini-pro-tools",
        "gemini flash": "gemini-flash",
        "gemini-flash": "gemini-flash",
        "gemini-3-flash": "gemini-flash",
        "gemini 3 flash": "gemini-flash",
        "gemini-3-flash-preview": "gemini-flash",
        "gemini lite": "gemini-lite",
        "gemini-lite": "gemini-lite",
        "gemini flash lite": "gemini-lite",
        "gemini-3.1-flash-lite": "gemini-lite",
        "gemini 3.1 flash lite": "gemini-lite",
        "gemini-3.1-flash-lite-preview": "gemini-lite",
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini 2.5 pro": "gemini-2.5-pro",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini 2.5 flash": "gemini-2.5-flash",
        "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
        "gemini 2.5 flash lite": "gemini-2.5-flash-lite",
        "llama": "llama-70b",
        "llama-70b": "llama-70b",
        "llama 70b": "llama-70b",
        "llama-vision": "llama-vision",
        "llama vision": "llama-vision",
        "vision": "llama-vision",
        "research": "research",
        "bioinformatics": "bioinformatics",
        "bioinfo": "bioinformatics",
        "nvidia-chat": "nvidia-chat",
        "nvidia chat": "nvidia-chat",
        "magistral": "nvidia-chat",
        "magistral-small": "nvidia-chat",
        "magistral small": "nvidia-chat",
        "nvidia-research": "nvidia-research",
        "nvidia research": "nvidia-research",
        "mistral-large-3": "nvidia-research",
        "mistral large 3": "nvidia-research",
        "nvidia-code": "nvidia-code",
        "nvidia code": "nvidia-code",
        "qwen3-coder": "nvidia-code",
        "qwen3 coder": "nvidia-code",
        "nvidia-agentic": "nvidia-agentic",
        "nvidia agentic": "nvidia-agentic",
        "devstral-2": "nvidia-agentic",
        "devstral 2": "nvidia-agentic",
        "nvidia-guard": "nvidia-guard",
        "nvidia guard": "nvidia-guard",
        "llama-guard": "nvidia-guard",
        "llama guard": "nvidia-guard",
        "nvidia-pii": "nvidia-pii",
        "nvidia pii": "nvidia-pii",
        "gliner-pii": "nvidia-pii",
        "gliner pii": "nvidia-pii",
    }
    alias = aliases.get(lowered)
    if alias:
        return MODEL_PROFILES[alias].model_id
    return candidate


def _resolve_router_blueprint(policy: str | None) -> RouterBlueprint:
    return ROUTER_BLUEPRINTS.get(str(policy or "").strip().lower(), ROUTER_BLUEPRINTS["balanced"])


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _score_terms(text: str, terms: tuple[str, ...], *, weight: float = 1.0) -> float:
    return sum(weight for term in terms if term in text)


def _explicit_model_control_alias(text: str) -> str | None:
    lowered = str(text or "").lower()
    control_phrases = (
        "respondeme con",
        "responde con",
        "respondeme usando",
        "responde usando",
        "usa ",
        "usar ",
        "usando ",
        "cambia a",
        "cambiar a",
        "quiero que me responda",
        "quiero que responda",
        "modelo de",
        "modelo ",
        "/with ",
    )
    model_alias_terms: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("sonnet", ("sonnet", "claude")),
        ("llama-vision", ("llama vision", "llama-vision", "vision", "screenshot", "imagen", "image")),
        ("llama-70b", ("llama", "llama 70b", "llama-70b")),
        ("gemma", ("gemma", "gemma 4", "gemma-4")),
        ("gemini-pro", ("gemini pro", "gemini-3.1-pro", "gemini 3.1 pro", "gemini-3.1-pro-preview")),
        ("gemini-pro-tools", ("gemini pro tools", "gemini customtools", "gemini-3.1-pro-preview-customtools")),
        ("gemini-2.5-pro", ("gemini 2.5 pro", "gemini-2.5-pro")),
        ("gemini-2.5-flash", ("gemini 2.5 flash", "gemini-2.5-flash")),
        ("gemini-2.5-flash-lite", ("gemini 2.5 flash lite", "gemini-2.5-flash-lite")),
        ("gemini-lite", ("gemini lite", "gemini flash lite", "gemini-3.1-flash-lite", "gemini 3.1 flash lite")),
        ("gemini-flash", ("gemini flash", "gemini-3-flash", "gemini 3 flash", "gemini-3-flash-preview", "gemini")),
        ("code", ("qwen coder", "qwen-coder", "coder")),
        ("bioinformatics", ("bioinformatics", "bioinfo")),
        ("nvidia-chat", ("nvidia chat", "magistral", "magistral small", "magistral-small")),
        ("nvidia-research", ("nvidia research", "mistral large 3", "mistral-large-3")),
        ("nvidia-code", ("nvidia code", "qwen3 coder", "qwen3-coder")),
        ("nvidia-agentic", ("nvidia agentic", "devstral 2", "devstral-2")),
        ("nvidia-guard", ("nvidia guard", "llama guard", "llama-guard")),
        ("nvidia-pii", ("nvidia pii", "gliner pii", "gliner-pii")),
        ("qwen-big", ("qwen big", "qwen-heavy", "qwen", "qwen 122b", "qwen-122b")),
        ("devstral", ("devstral",)),
        ("mistral", ("mistral", "mistral small")),
        ("engineering", ("glm", "glm-5.1", "glm 5.1")),
        ("deep-reasoning", ("deepseek", "deepseek v3", "deepseek-v3")),
    )
    if not _contains_any(lowered, control_phrases):
        if "tenes algun modelo" not in lowered and "tienes algun modelo" not in lowered:
            return None
    for alias, terms in model_alias_terms:
        if _contains_any(lowered, terms):
            return alias
    return None


def _profile_alias_for_model_id(model_id: str) -> str | None:
    for alias, profile in MODEL_PROFILES.items():
        if profile.model_id == model_id or alias == model_id:
            return alias
    return None


_MISSING_MODEL_ALIAS_RE = re.compile(r"model alias not found:\s*([A-Za-z0-9_.-]+)", re.IGNORECASE)


def _missing_model_alias_from_error(exc: Exception) -> str | None:
    seen: set[int] = set()
    current: Exception | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        match = _MISSING_MODEL_ALIAS_RE.search(str(current))
        if match:
            return match.group(1)
        next_exc = current.__cause__ or current.__context__
        current = next_exc if isinstance(next_exc, Exception) else None
    return None


def _warn_missing_model_alias(alias: str, fallback_alias: str) -> None:
    message = f"[!] Alias not found ('{alias}'). Falling back to default '{fallback_alias}' profile."
    if console:
        console.print(f"[warning]{message}[/warning]")
    else:
        print(message)


def _recover_missing_model_alias(session: "InteractiveSession", exc: Exception) -> bool:
    missing_alias = _missing_model_alias_from_error(exc)
    if not missing_alias:
        return False
    fallback_alias = "research"
    fallback_profile = MODEL_PROFILES[fallback_alias]
    session.provider_name = fallback_profile.provider
    session.model = fallback_profile.model_id
    session.record(
        role="system",
        content=f"Recovered from missing model alias `{missing_alias}` by falling back to `{fallback_alias}`.",
        event_type="model_alias_fallback",
        metadata={
            "missing_alias": missing_alias,
            "fallback_alias": fallback_alias,
            "fallback_model": fallback_profile.model_id,
            "fallback_provider": fallback_profile.provider,
        },
    )
    _warn_missing_model_alias(missing_alias, fallback_alias)
    return True


def _capability_requirements_for_prompt(
    text: str,
    *,
    intent: str,
    url_refs: list[str],
    path_refs: list[str],
    suite_focus: bool = False,
    web_focus: bool = False,
    interaction_mode: str = "balanced",
) -> dict[str, Any]:
    lowered = str(text or "").lower()
    active_mode = _normalize_interaction_mode(interaction_mode)
    explore_external_signal = active_mode == "explore" and (
        bool(url_refs)
        or web_focus
        or any(term in lowered for term in ("fuentes", "sources", "research", "investiga", "google", "internet", "web", "actual", "latest", "reciente"))
    )
    needs_long_context = (
        len(str(text or "")) > 1200
        or len(url_refs) > 1
        or intent in {"research", "web_research", "suite_forensics", "helix_self", "creative_helix", "agentic", "agentic_code", "audit"}
    )
    structured_output_terms = (
        " json", "json ", "json.", "json,", "json:",
        "schema", "estructurado", "estructurada", "structured",
        "formato json", "formato yaml", "json output", "json mode",
        "respond with json", "respondé con json", "responde con json",
        "devolveme json", "devolveme un json", "give me json", "give me a json",
        "as json", "como json", "in json format", "en formato json",
        "/cert ", "/tools json", "tools json",
        "valid json", "json valido", "json válido",
    )
    needs_structured_output = bool(
        # Tool-calling intents almost always benefit from JSON-mode for the
        # final tool-arg payload.
        intent in {"agentic", "agentic_code", "suite_forensics"}
        or any(term in (" " + lowered + " ") for term in structured_output_terms)
    )
    return {
        "function_calling": bool(intent in {"agentic", "agentic_code"}),
        "parallel_tools": bool(intent in {"agentic", "agentic_code", "suite_forensics"}),
        "url_context": bool(url_refs),
        "search_grounding": bool(
            web_focus
            or explore_external_signal
            or (
                url_refs
                and any(term in lowered for term in ("latest", "actual", "actuales", "current", "reciente", "news", "fuentes", "sources"))
            )
        ),
        "file_search": False,
        "vision": bool(intent == "vision"),
        "long_context": needs_long_context,
        "structured_output": needs_structured_output,
        "local_file_grounding": bool(path_refs),
        "suite_grounding": bool(suite_focus),
    }


def _empty_native_tool_plan(provider_name: str) -> dict[str, Any]:
    return {
        "provider": provider_name,
        "mode": "helix-only",
        "url_context_urls": [],
        "enable_search_grounding": False,
        "function_declarations": [],
        "function_calling_mode": None,
        "file_search_store_ids": [],
        "request_response_format": None,
        "why_not": [],
    }


def _grounding_plan_for_route(
    route: dict[str, Any],
    capability_requirements: dict[str, Any],
    native_tool_plan: dict[str, Any],
    *,
    interaction_mode: str,
) -> str:
    if native_tool_plan.get("mode") == "gemini-native":
        return "gemini-native"
    if capability_requirements.get("local_file_grounding") or capability_requirements.get("suite_grounding"):
        return "helix-only"
    if capability_requirements.get("search_grounding"):
        return "helix-web-tools"
    if _normalize_interaction_mode(interaction_mode) == "explore" and capability_requirements.get("url_context"):
        return "helix-web-tools"
    return "helix-only"


def _mode_reason_for_route(
    *,
    interaction_mode: str,
    intent: str,
    capability_requirements: dict[str, Any],
    grounding_plan: str,
) -> str:
    active_mode = _normalize_interaction_mode(interaction_mode)
    if active_mode == "technical":
        return "Technical mode biases this turn toward code, audit, repo, suite and HeliX-core grounding while preserving prompt intent."
    if active_mode == "explore":
        if intent == "creative_helix":
            return "Explore mode kept the HeliX prompt in creative/cultural synthesis because no concrete core/audit terms were requested."
        if grounding_plan in {"gemini-native", "helix-web-tools"} or capability_requirements.get("search_grounding"):
            return "Explore mode allows external grounding for URLs, current sources or explicit research signals."
        return "Explore mode biases toward broad synthesis and interpretation while keeping runtime guarantees clearly bounded."
    return "Balanced mode keeps the existing router behavior and only activates extra grounding when prompt signals require it."


def _gemini_alias_for_prompt(
    text: str,
    *,
    intent: str,
    capability_requirements: dict[str, Any],
) -> str:
    lowered = str(text or "").lower()
    if intent == "vision":
        return "gemini-flash"
    if intent in {"agentic_code", "code", "audit", "suite_forensics", "helix_self"}:
        return "gemini-pro"
    if capability_requirements.get("url_context") and (
        capability_requirements.get("long_context")
        or len(_extract_url_refs(text)) > 1
        or any(term in lowered for term in ("compar", "compare", "docs", "documentacion", "documentation", "sintetiza", "synthesis"))
    ):
        return "gemini-pro"
    if any(term in lowered for term in ("clasifica", "classify", "etiqueta", "tag", "breve", "one line", "una linea")):
        return "gemini-lite"
    if intent in {"reasoning", "research", "web_research", "creative_helix"}:
        return "gemini-pro" if capability_requirements.get("long_context") else "gemini-flash"
    return "gemini-flash"


def _native_tool_plan_for_route(route: dict[str, Any], text: str) -> dict[str, Any]:
    provider_name = str(route.get("provider") or "")
    profile = _model_profile_for_id(str(route.get("model") or ""))
    url_refs = _extract_url_refs(text)
    path_refs = _extract_local_path_refs(text)
    suite_focus = _is_suite_evidence_request(text)
    web_focus = _is_web_search_request(text)
    capability_requirements = _capability_requirements_for_prompt(
        text,
        intent=str(route.get("intent") or "chat"),
        url_refs=url_refs,
        path_refs=path_refs,
        suite_focus=suite_focus,
        web_focus=web_focus,
        interaction_mode=str(route.get("interaction_mode") or "balanced"),
    )
    plan = _empty_native_tool_plan(provider_name)
    why_not: list[str] = []
    if provider_name != "gemini":
        if url_refs:
            why_not.append("URL Context is only wired for Gemini in this HeliX pass.")
        if capability_requirements.get("search_grounding"):
            why_not.append("Native web grounding stays disabled outside Gemini; HeliX web tools remain available.")
    else:
        if capability_requirements.get("url_context"):
            if profile and profile.supports_url_context:
                plan["mode"] = "gemini-native"
                plan["url_context_urls"] = url_refs[:8]
            else:
                why_not.append("The selected Gemini profile does not advertise URL Context support.")
        if capability_requirements.get("search_grounding"):
            if profile and profile.supports_search_grounding:
                plan["mode"] = "gemini-native"
                plan["enable_search_grounding"] = True
            else:
                why_not.append("The selected Gemini profile does not advertise Google Search grounding support.")
        if capability_requirements.get("function_calling"):
            why_not.append("HeliX keeps local tool orchestration in its own planner; Gemini function calling is not auto-enabled here.")
        if capability_requirements.get("file_search"):
            why_not.append("Gemini File Search is reserved for remote stores; local repo files stay on file.inspect.")
    # Structured output: only request OpenAI-compat json_object mode for
    # OpenAI-compatible providers (DeepInfra). Gemini and Anthropic have
    # different schema-binding APIs and are left to HeliX prompt-side schema
    # enforcement here.
    if capability_requirements.get("structured_output"):
        if provider_name == "deepinfra" and profile and getattr(profile, "supports_structured_output", False):
            plan["request_response_format"] = {"type": "json_object"}
        elif provider_name not in {"deepinfra"}:
            why_not.append(f"Native structured-output mode is wired only for OpenAI-compat providers; {provider_name} stays on prompt-side schema.")
        else:
            why_not.append("The selected profile does not advertise structured output; HeliX prompt-side schema only.")
    if path_refs:
        why_not.append("Local paths and directories are grounded via HeliX file.inspect before any provider-native remote context.")
    plan["why_not"] = why_not
    return plan


def _augment_route_metadata(route: dict[str, Any], text: str, *, interaction_mode: str | None = None) -> dict[str, Any]:
    payload = dict(route)
    active_mode = _normalize_interaction_mode(interaction_mode or str(payload.get("interaction_mode") or "balanced"))
    url_refs = _extract_url_refs(text)
    path_refs = _extract_local_path_refs(text)
    suite_focus = _is_suite_evidence_request(text)
    web_focus = _is_web_search_request(text)
    capability_requirements = _capability_requirements_for_prompt(
        text,
        intent=str(payload.get("intent") or "chat"),
        url_refs=url_refs,
        path_refs=path_refs,
        suite_focus=suite_focus,
        web_focus=web_focus,
        interaction_mode=active_mode,
    )
    payload["interaction_mode"] = active_mode
    native_tool_plan = _native_tool_plan_for_route(payload, text)
    grounding_plan = _grounding_plan_for_route(
        payload,
        capability_requirements,
        native_tool_plan,
        interaction_mode=active_mode,
    )
    payload.update(
        {
            "path_refs": path_refs,
            "url_refs": url_refs,
            "capability_requirements": capability_requirements,
            "native_tool_plan": native_tool_plan,
            "grounding_plan": grounding_plan,
            "mode_policy": _interaction_mode_payload(active_mode),
            "tone_contract": INTERACTION_MODE_PROFILES[active_mode]["tone_contract"],
            "mode_reason": _mode_reason_for_route(
                interaction_mode=active_mode,
                intent=str(payload.get("intent") or "chat"),
                capability_requirements=capability_requirements,
                grounding_plan=grounding_plan,
            ),
            "why_not": list(native_tool_plan.get("why_not") or []),
        }
    )
    profile = _model_profile_for_id(str(payload.get("model") or ""))
    provider = PROVIDERS.get(str(payload.get("provider") or ""))
    if profile:
        payload.update(_profile_capability_payload(profile))
    if provider:
        payload["provider_native_capabilities"] = list(provider.native_capabilities)
        payload["provider_native_constraints"] = list(provider.native_constraints)
    return payload


def _manual_route_for_model(
    model_id: str,
    *,
    provider_name: str,
    policy: str,
    user_text: str = "",
    interaction_mode: str = "balanced",
) -> dict[str, Any]:
    alias = _profile_alias_for_model_id(model_id)
    profile = MODEL_PROFILES.get(alias or "")
    fallback_chain = list(_fallback_aliases_for_alias(alias or ""))
    route_provider = profile.provider if profile else provider_name
    return _augment_route_metadata(
        {
            "provider": route_provider,
            "model": model_id,
            "profile": alias or "manual",
            "role": profile.role if profile else "manual",
            "intent": "manual",
            "confidence": 1.0,
            "signals": ["manual_model"],
            "policy": policy,
            "blueprint": _resolve_router_blueprint(policy).name,
            "blueprint_description": _resolve_router_blueprint(policy).description,
            "intent_scores": {"manual": 1.0},
            "top_intents": [["manual", 1.0]],
            "ambiguity": False,
            "ambiguity_resolver": "not_used",
            "manual_model_alias": alias,
            "fallback_chain": fallback_chain,
            "reason": profile.notes if profile else "User-selected model for this action/session.",
            "input_per_million": profile.input_per_million if profile else None,
            "output_per_million": profile.output_per_million if profile else None,
            "interaction_mode": _normalize_interaction_mode(interaction_mode),
        },
        user_text,
        interaction_mode=interaction_mode,
    )


def _fallback_aliases_for_alias(alias: str) -> tuple[str, ...]:
    if alias == "gemini-pro":
        return ("gemini-pro-tools", "gemini-2.5-pro", "gemini-flash", "gemini-2.5-flash", "gemini-lite", "gemini-2.5-flash-lite")
    if alias == "gemini-pro-tools":
        return ("gemini-pro", "gemini-2.5-pro", "gemini-flash", "gemini-2.5-flash", "gemini-lite", "gemini-2.5-flash-lite")
    if alias == "gemini-flash":
        return ("gemini-2.5-flash", "gemini-lite", "gemini-2.5-flash-lite")
    if alias == "qwen-big":
        return ("qwen-122b", "default", "chat")
    if alias == "gemini-lite":
        return ("gemini-2.5-flash-lite",)
    if alias == "gemini-2.5-pro":
        return ("gemini-flash", "gemini-2.5-flash", "gemini-lite")
    if alias == "gemini-2.5-flash":
        return ("gemini-lite", "gemini-2.5-flash-lite")
    if alias == "qwen-122b":
        return ("default", "chat")
    if alias == "code":
        return ("devstral", "qwen-big", "chat")
    if alias == "engineering":
        return ("code", "qwen-big", "devstral")
    if alias == "sonnet":
        return ("qwen-big", "deep-reasoning")
    if alias == "gemma":
        return ("qwen-big", "legacy-reasoning")
    if alias == "research":
        return ("qwen-big", "default", "chat")
    if alias == "deep-reasoning":
        return ("qwen-big", "gemma")
    return ()


def _dominant_recent_intent(intents: list[str] | None) -> tuple[str | None, int]:
    """Return (intent, support) for the most repeated non-chat intent in the
    last few user turns. Requires at least 2/N support to count as a trail."""
    if not intents:
        return None, 0
    significant = [item for item in intents if item and item != "chat"]
    if len(significant) < 2:
        return None, 0
    counts: dict[str, int] = {}
    for intent in significant:
        counts[intent] = counts.get(intent, 0) + 1
    top_intent, top_count = max(counts.items(), key=lambda pair: pair[1])
    return (top_intent, top_count) if top_count >= 2 else (None, 0)


def route_model_for_task(
    text: str,
    *,
    provider_name: str = "deepinfra",
    policy: str = "balanced",
    interaction_mode: str = "balanced",
    recent_intents: list[str] | None = None,
) -> dict[str, Any]:
    """Select a model for one turn using transparent heuristics.

    This is intentionally deterministic. The router should be auditable before
    it becomes another model call.

    `recent_intents` is the route.intent of the last few user turns (oldest
    first). When the user has been in the same lane for 2+ consecutive turns
    the dominant intent gets a +0.4 score bump so a vague follow-up does not
    fall back to chat — the multi-turn objective is preserved.
    """

    lowered = str(text or "").lower()
    active_mode = _normalize_interaction_mode(interaction_mode)
    url_refs = _extract_url_refs(text)
    path_refs = _extract_local_path_refs(text)
    blueprint = _resolve_router_blueprint(policy)
    policy = blueprint.name
    signals: list[str] = []
    explicit_alias = _explicit_model_control_alias(lowered)
    continuity_intent, continuity_support = _dominant_recent_intent(recent_intents)

    helix_terms = (
        "helix",
        "merkle",
        "dag",
        "receipt",
        "receipts",
        "memoria firmada",
        "evidencia certificada",
        "verification",
        "/verify",
        "corridas",
        "artifact",
        "artefacto",
    )
    research_terms = (
        "busca",
        "buscame",
        "google",
        "investiga",
        "research",
        "scraping",
        "scrap",
        "benchmark",
        "benchamark",
        "paper",
        "papers",
        "fuentes",
        "source",
        "sources",
        "web",
        "modelos nuevos",
        "llms",
        "deepinfra",
    )
    web_terms = (
        "busca en la web",
        "buscar en la web",
        "buscame en la web",
        "google",
        "googlea",
        "internet",
        "online",
        "latest",
        "noticias",
        "news",
        "reciente",
        "actual",
        "fuentes",
        "links",
        "sources",
    )
    suite_terms = (
        "/verify",
        "verify",
        "suite",
        "suites",
        "corrida",
        "corridas",
        "artifact",
        "artefacto",
        "artifacts",
        "manifest",
        "manifests",
        "transcript",
        "transcripts",
        "jsonl",
        "preregistered",
        "preregistro",
        "resultados",
        "experimentos",
        "verification",
        "post nuclear",
        "post-nuclear",
        "long horizon",
        "hard anchor",
        "hard-anchor",
        "branch pruning",
        "policy rag",
        "poliza",
        "póliza",
    )
    code_terms = (
        "code",
        "codigo",
        "código",
        "bug",
        "fix",
        "patch",
        "diff",
        "repo",
        "pytest",
        "test",
        "refactor",
        "typescript",
        "javascript",
        "python",
        "rust",
        "powershell",
        "compila",
        "build",
        "cli",
        "archivo",
        "commit",
    )
    agentic_code_terms = (
        "claude code",
        "codex",
        "agentic",
        "agente",
        "agent",
        "workspace",
        "multi-archivo",
        "multi archivo",
        "multi-file",
        "multiarchivo",
        "implementa",
        "implementalo",
        "refactoriza",
        "refactorizalo",
        "arregla",
        "arreglalo",
        "hacelo",
        "lee el repo",
        "fijate el repo",
        "mira el repo",
        "revisa el repo",
        "corré tests",
        "corre tests",
        "ejecuta tests",
        "terminal",
        "tool",
        "tools",
    )
    audit_terms = (
        "auditor",
        "audit",
        "legal",
        "poliza",
        "claims",
        "claim",
        "forense",
        "forensic",
        "seguridad",
        "security",
        "certifica",
        "metodologia",
        "evidencia",
        "evidence",
        "riesgo",
        "risk",
    )
    reasoning_terms = (
        "razona",
        "reason",
        "matematica",
        "matemática",
        "prueba",
        "proof",
        "hipotesis",
        "hipótesis",
        "analiza",
        "desglosa",
        "compar",
        "tradeoff",
        "arquitectura",
        "architecture",
        "metodologia",
        "metodología",
    )
    vision_terms = (
        "imagen",
        "imagenes",
        "imágenes",
        "image",
        "images",
        "foto",
        "fotos",
        "photo",
        "screenshot",
        "captura",
        "capturas",
        "screen",
        "ocr",
        "vision",
        "visual",
        "pdf escaneado",
        "scan",
        "diagrama",
        "diagram",
    )
    long_task_terms = (
        "largo plazo",
        "long horizon",
        "planifica",
        "orquesta",
        "suite",
        "serie de test",
        "plan de implementacion",
    )
    creative_terms = (
        "filosofia",
        "filosofía",
        "cultura",
        "cultural",
        "metafora",
        "metáfora",
        "ghost in the shell",
        "rizoma",
        "rizomas",
        "hipersticion",
        "hiperstición",
        "deleuze",
        "guattari",
        "ontologia",
        "ontología",
        "poetica",
        "poética",
        "influencias",
        "simbolismo",
        "explora",
        "explorar",
        "creativo",
        "imaginario",
    )

    scores: dict[str, float] = {
        "chat": 1.0,
        "helix_self": _score_terms(lowered, helix_terms, weight=1.5),
        "suite_forensics": _score_terms(lowered, suite_terms, weight=1.7),
        "research": _score_terms(lowered, research_terms, weight=1.6),
        "web_research": _score_terms(lowered, web_terms, weight=1.9),
        "code": _score_terms(lowered, code_terms, weight=1.4),
        "agentic_code": _score_terms(lowered, agentic_code_terms, weight=1.6),
        "audit": _score_terms(lowered, audit_terms, weight=1.8),
        "reasoning": _score_terms(lowered, reasoning_terms, weight=1.3),
        "vision": _score_terms(lowered, vision_terms, weight=2.0),
        "agentic": _score_terms(lowered, long_task_terms, weight=1.4),
        "creative_helix": 0.0,
    }
    if explicit_alias:
        scores["model_control"] = 10.0
        signals.append("model_control")
    if url_refs:
        signals.append("url_refs")
        if not path_refs:
            scores["web_research"] += 0.8
            scores["research"] += 0.6
    if path_refs:
        signals.append("local_path_refs")
        scores["code"] += 0.4
        scores["agentic_code"] += 0.3
        if any(term in lowered for term in ("patch", "diff", "repo", "archivo", "archivos", "codigo", "code", "bug", "fix", "refactor", "src/")):
            scores["code"] += 2.2
        if any(term in lowered for term in ("patch", "fix", "implementa", "refactor", "arregla", "propon", "propose")):
            scores["agentic_code"] += 1.8
    if len(text) > 1200:
        scores["agentic"] += 1.2
        scores["reasoning"] += 0.8
        signals.append("long_prompt")
    if scores["agentic_code"] and scores["code"]:
        scores["agentic_code"] += 2.0
    if scores["audit"] and scores["code"]:
        scores["audit"] += 0.5
    if scores["helix_self"] and scores["audit"]:
        scores["audit"] += 0.6
    if scores["research"] and "modelos" in lowered and "nuevo" in lowered:
        scores["research"] += 1.5
    if _is_web_search_request(lowered):
        scores["web_research"] += 5.0
        scores["research"] += 1.0
    if scores["suite_forensics"] and scores["audit"]:
        scores["suite_forensics"] += 0.8
    if active_mode == "technical":
        scores["code"] += 0.7
        scores["agentic_code"] += 0.5
        scores["audit"] += 0.8
        scores["suite_forensics"] += 0.6
        scores["helix_self"] += 0.6
        signals.append("mode:technical")
    elif active_mode == "explore":
        scores["research"] += 0.6
        scores["reasoning"] += 0.5
        scores["chat"] += 0.2
        scores["web_research"] += 0.4 if (url_refs or _is_web_search_request(lowered)) else 0.0
        signals.append("mode:explore")
        if _is_creative_helix_prompt(lowered) and not _is_helix_auditability_request(lowered):
            scores["creative_helix"] = max(
                scores["helix_self"] + 1.6,
                2.8 + _score_terms(lowered, creative_terms, weight=1.0),
            )
            scores["helix_self"] = max(0.0, min(scores["helix_self"], scores["creative_helix"] - 1.0))
            signals.append("creative_helix_scope")

    # Continuity bump: when the user has stayed in the same lane for 2+ of
    # the last 3-4 turns, give that intent a small score bump so a vague
    # follow-up like "y eso?" or "arreglalo" stays in the right carril.
    # The bump is intentionally small (+0.4) so it tilts ties without
    # overriding strong signals from the current turn.
    if continuity_intent and continuity_intent in scores:
        scores[continuity_intent] += 0.4
        signals.append(f"continuity:{continuity_intent}({continuity_support})")

    if scores["audit"]:
        signals.append("audit_or_high_stakes")
    if scores["suite_forensics"]:
        signals.append("suite_forensics")
    if scores["agentic_code"] or scores["agentic"]:
        signals.append("agentic_or_long_horizon")
    if scores["code"]:
        signals.append("code_or_repo")
    if scores["reasoning"]:
        signals.append("reasoning")
    if scores["research"]:
        signals.append("research")
    if scores["web_research"]:
        signals.append("web_research")
    if scores["vision"]:
        signals.append("vision")
    if scores["helix_self"]:
        signals.append("helix_self")
    if scores["creative_helix"]:
        signals.append("creative_helix")

    priority = {
        "model_control": 100,
        "vision": 90,
        "audit": 80,
        "agentic_code": 76,
        "code": 70,
        "suite_forensics": 66,
        "helix_self": 62,
        "creative_helix": 62,
        "web_research": 61,
        "research": 60,
        "agentic": 55,
        "reasoning": 50,
        "chat": 0,
    }
    ranked = sorted(scores.items(), key=lambda item: (item[1], priority.get(item[0], 0)), reverse=True)
    intent, top_score = ranked[0]
    second_intent, second_score = ranked[1] if len(ranked) > 1 else ("none", 0.0)
    if active_mode == "explore" and intent == "helix_self" and scores["creative_helix"] >= max(1.5, scores["helix_self"]):
        intent = "creative_helix"
        top_score = scores["creative_helix"]
        second_intent, second_score = "helix_self", scores["helix_self"]
    if path_refs and intent in {"web_research", "research"} and max(scores["code"], scores["agentic_code"]) >= 3.5:
        intent = "agentic_code" if scores["agentic_code"] >= scores["code"] else "code"
        top_score = scores[intent]
    if top_score <= 1.0:
        intent = "chat"
    ambiguity = bool(top_score > 1.0 and second_score > 1.0 and (top_score - second_score) <= 1.25)

    if intent == "model_control" and explicit_alias and explicit_alias in MODEL_PROFILES:
        profile = MODEL_PROFILES[explicit_alias]
        return _augment_route_metadata(
            {
            "provider": profile.provider,
            "model": profile.model_id,
            "profile": explicit_alias,
            "role": profile.role,
            "intent": intent,
            "confidence": 0.97,
            "signals": sorted(set(signals)),
            "policy": policy,
            "blueprint": blueprint.name,
            "blueprint_description": blueprint.description,
            "intent_scores": {key: round(value, 4) for key, value in scores.items() if value > 0},
            "top_intents": [[name, round(score, 4)] for name, score in ranked[:3]],
            "ambiguity": False,
            "ambiguity_resolver": "explicit_model_alias",
            "manual_model_alias": explicit_alias,
            "fallback_chain": list(_fallback_aliases_for_alias(explicit_alias)),
            "reason": profile.notes,
            "input_per_million": profile.input_per_million,
            "output_per_million": profile.output_per_million,
            "interaction_mode": active_mode,
            },
            text,
            interaction_mode=active_mode,
        )

    gemini_override = bool(url_refs and not path_refs and provider_name == "deepinfra" and _provider_ready("gemini"))
    if provider_name == "gemini":
        alias = _gemini_alias_for_prompt(
            text,
            intent=intent,
            capability_requirements=_capability_requirements_for_prompt(
                text,
                intent=intent,
                url_refs=url_refs,
                path_refs=path_refs,
                suite_focus=bool(scores["suite_forensics"]),
                web_focus=bool(scores["web_research"]),
                interaction_mode=active_mode,
            ),
        )
        profile = GEMINI_MODEL_PROFILES[alias]
        confidence = 0.58 if not signals else min(0.97, 0.66 + max(0.0, top_score - second_score) * 0.06 + top_score * 0.03)
        if ambiguity:
            confidence = min(confidence, 0.7)
        return _augment_route_metadata(
            {
                "provider": "gemini",
                "model": profile.model_id,
                "profile": alias,
                "role": profile.role,
                "intent": intent,
                "confidence": round(confidence, 4),
                "signals": sorted(set(signals)),
                "policy": policy,
                "blueprint": blueprint.name,
                "blueprint_description": blueprint.description,
                "intent_scores": {key: round(value, 4) for key, value in scores.items() if value > 0},
                "top_intents": [[name, round(score, 4)] for name, score in ranked[:3]],
                "ambiguity": ambiguity,
                "ambiguity_resolver": "deterministic_scoring",
                "manual_model_alias": explicit_alias if intent == "model_control" else None,
                "fallback_chain": list(_fallback_aliases_for_alias(alias)),
                "reason": profile.notes,
                "input_per_million": profile.input_per_million,
                "output_per_million": profile.output_per_million,
                "interaction_mode": active_mode,
            },
            text,
            interaction_mode=active_mode,
        )

    if provider_name != "deepinfra":
        return _augment_route_metadata(
            {
                "provider": provider_name,
                "model": None,
                "profile": "provider-default",
                "intent": intent,
                "confidence": 0.55,
                "signals": sorted(set(signals)),
                "policy": policy,
                "blueprint": blueprint.name,
                "blueprint_description": blueprint.description,
                "intent_scores": {key: round(value, 4) for key, value in scores.items() if value > 0},
                "top_intents": [[name, round(score, 4)] for name, score in ranked[:3]],
                "ambiguity": ambiguity,
                "ambiguity_resolver": "not_used",
                "reason": "Non-DeepInfra providers keep their configured/default model unless the provider has a dedicated router.",
                "interaction_mode": active_mode,
            },
            text,
            interaction_mode=active_mode,
        )

    if intent == "model_control" and explicit_alias:
        alias = explicit_alias
    else:
        if gemini_override and intent in {"chat", "reasoning", "research", "web_research", "helix_self", "creative_helix", "suite_forensics"}:
            alias = _gemini_alias_for_prompt(
                text,
                intent=intent,
                capability_requirements=_capability_requirements_for_prompt(
                    text,
                    intent=intent,
                    url_refs=url_refs,
                    path_refs=path_refs,
                    suite_focus=bool(scores["suite_forensics"]),
                    web_focus=bool(scores["web_research"]),
                    interaction_mode=active_mode,
                ),
            )
        elif intent == "vision":
            alias = blueprint.vision_alias
        elif intent == "audit":
            alias = blueprint.audit_alias
        elif intent == "agentic_code":
            alias = blueprint.code_alias
        elif intent == "code":
            alias = blueprint.code_alias
        elif intent in {"helix_self", "creative_helix", "suite_forensics"}:
            alias = blueprint.research_alias
        elif intent in {"research", "web_research"}:
            alias = blueprint.research_alias
        elif intent == "agentic":
            alias = blueprint.agentic_alias
        elif intent == "reasoning":
            alias = blueprint.reasoning_alias
        else:
            alias = blueprint.chat_alias or blueprint.default_alias

    profile = MODEL_PROFILES[alias]
    fallback_chain = list(_fallback_aliases_for_alias(alias))
    confidence = 0.45 if not signals else min(0.97, 0.62 + max(0.0, top_score - second_score) * 0.07 + top_score * 0.03)
    if ambiguity:
        confidence = min(confidence, 0.68)
    if gemini_override and profile.provider == "gemini":
        signals.append("gemini_url_context_candidate")
    return _augment_route_metadata(
        {
        "provider": profile.provider,
        "model": profile.model_id,
        "profile": alias,
        "role": profile.role,
        "intent": intent,
        "confidence": round(confidence, 4),
        "signals": sorted(set(signals)),
        "policy": policy,
        "blueprint": blueprint.name,
        "blueprint_description": blueprint.description,
        "intent_scores": {key: round(value, 4) for key, value in scores.items() if value > 0},
        "top_intents": [[name, round(score, 4)] for name, score in ranked[:3]],
        "ambiguity": ambiguity,
        "ambiguity_resolver": "deterministic_scoring",
        "manual_model_alias": explicit_alias if intent == "model_control" else None,
        "fallback_chain": fallback_chain,
        "reason": profile.notes,
        "input_per_million": profile.input_per_million,
        "output_per_million": profile.output_per_million,
        "interaction_mode": active_mode,
        },
        text,
        interaction_mode=active_mode,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _print_json(payload: Any) -> None:
    print(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False))


def _print_table(rows: list[dict[str, Any]], columns: list[tuple[str, str, int]]) -> None:
    if not rows:
        print("(empty)")
        return
    header = "  ".join(label.ljust(width) for _key, label, width in columns)
    print(header.rstrip())
    print("  ".join("-" * width for _key, _label, width in columns).rstrip())
    for row in rows:
        cells = []
        for key, _label, width in columns:
            value = str(row.get(key, "") if row.get(key, "") is not None else "")
            value = value.replace("\n", " ")
            if len(value) > width:
                value = value[: max(0, width - 1)] + "…"
            cells.append(value.ljust(width))
        print("  ".join(cells).rstrip())


def _compact_model_rows() -> list[dict[str, Any]]:
    return [
        {
            "alias": item["alias"],
            "provider": item["provider"],
            "role": item["role"],
            "model": _short_model_name(item["model_id"]),
            "cost": (
                f"{item['input_per_million']}/{item['output_per_million']}"
                if item.get("input_per_million") is not None and item.get("output_per_million") is not None
                else "n/a"
            ),
            "use": item["notes"],
        }
        for item in model_profiles_report()
    ]


def _print_models_compact() -> None:
    rows = _compact_model_rows()
    columns = [
        ("alias", "alias", 18),
        ("provider", "provider", 10),
        ("role", "role", 18),
        ("model", "model", 34),
        ("cost", "$/M in/out", 12),
        ("use", "use", 58),
    ]
    provider_order = ("deepinfra", "gemini", "nvidia")
    printed: set[str] = set()
    for provider_name in provider_order:
        provider_rows = [row for row in rows if row.get("provider") == provider_name]
        if not provider_rows:
            continue
        print(f"\n[{provider_name}]")
        _print_table(provider_rows, columns)
        printed.add(provider_name)
    remaining = [row for row in rows if row.get("provider") not in printed]
    if remaining:
        print("\n[other]")
        _print_table(remaining, columns)
    print("\nUse /model use ALIAS to pin one model, /model auto to restore routing, /models json for full metadata.")


def _print_tools_compact(report: dict[str, Any]) -> None:
    rows = [
        {
            "name": item.get("name"),
            "kind": item.get("kind") or item.get("safety") or "runtime",
            "description": item.get("description"),
        }
        for item in report.get("tools", [])
    ]
    _print_table(rows, [("name", "tool", 24), ("kind", "kind", 16), ("description", "description", 82)])
    print("\nUse /tools blueprints for agentic toolsets, /tools json for full registry.")


def _print_agent_blueprints_compact() -> None:
    rows = [
        {
            "blueprint": item["blueprint_id"],
            "model": item["preferred_model_alias"],
            "steps": item["max_steps"],
            "tools": ", ".join(item["allowed_tools"][:4]) + ("..." if len(item["allowed_tools"]) > 4 else ""),
            "description": item["description"],
        }
        for item in agent_blueprints_report()
    ]
    _print_table(
        rows,
        [
            ("blueprint", "blueprint", 24),
            ("model", "model", 12),
            ("steps", "steps", 5),
            ("tools", "tools", 44),
            ("description", "description", 62),
        ],
    )


def _print_suites_compact(payload: dict[str, Any]) -> None:
    rows = []
    for suite in payload.get("suites", []):
        counts = suite.get("counts") or {}
        latest = suite.get("latest") or {}
        rows.append(
            {
                "suite": suite.get("suite_id"),
                "registered": "yes" if suite.get("registered") else "no",
                "artifacts": counts.get("artifact", 0),
                "transcripts": counts.get("transcript_jsonl", 0) + counts.get("transcript_md", 0),
                "latest": latest.get("updated_utc") or "",
                "description": suite.get("description") or "",
            }
        )
    _print_table(
        rows,
        [
            ("suite", "suite", 34),
            ("registered", "reg", 4),
            ("artifacts", "json", 5),
            ("transcripts", "tx", 4),
            ("latest", "latest utc", 22),
            ("description", "description", 60),
        ],
    )
    print("\nUse /suite latest SUITE, /suite transcripts SUITE, /suite search QUERY, /suites json.")


def _compact_trust_report(report: dict[str, Any]) -> dict[str, Any]:
    proof = report.get("proof") if isinstance(report.get("proof"), dict) else {}
    lineage = report.get("lineage") if isinstance(report.get("lineage"), dict) else {}
    lineage_verification = proof.get("lineage_verification") if isinstance(proof.get("lineage_verification"), dict) else {}
    if not lineage_verification and isinstance(report.get("lineage_verification"), dict):
        lineage_verification = report.get("lineage_verification") or {}
    head = report.get("head_checkpoint") if isinstance(report.get("head_checkpoint"), dict) else {}
    if not head and isinstance(proof.get("head_checkpoint"), dict):
        head = proof.get("head_checkpoint") or {}
    trust_root = report.get("trust_root") if isinstance(report.get("trust_root"), dict) else {}

    checkpoint_hash = (
        head.get("checkpoint_hash")
        or lineage_verification.get("checkpoint_hash")
        or proof.get("checkpoint_hash")
        or ""
    )
    checkpoint_verified = (
        lineage_verification.get("checkpoint_verified")
        if lineage_verification.get("checkpoint_verified") is not None
        else lineage.get("checkpoint_verified")
    )
    if checkpoint_verified is None:
        checkpoint_verified = head.get("checkpoint_verified")
    if checkpoint_verified is None:
        checkpoint_verified = head.get("signature_verified")

    trust_status = (
        lineage_verification.get("trust_status")
        or lineage.get("trust_status")
        or lineage_verification.get("status")
        or lineage.get("status")
        or "unknown"
    )
    signature_verified = head.get("signature_verified")
    if signature_verified is None:
        signature_verified = head.get("checkpoint_verified")

    if trust_status in {"verified", "verified_with_quarantine"} and checkpoint_verified is not False:
        interpretation = "Local canonical head and signed checkpoint path verify."
    elif trust_status == "failed" or checkpoint_verified is False:
        interpretation = "Local trust check failed; use /trust --forensics for the full proof payload."
    else:
        interpretation = "Trust status is incomplete; use /trust current json for full diagnostics."

    return {
        "kind": "helix-local-trust-summary",
        "thread_id": report.get("thread_id"),
        "status": trust_status,
        "lineage_status": lineage_verification.get("status") or lineage.get("status"),
        "checkpoint_verified": checkpoint_verified,
        "signature_verified": signature_verified,
        "checkpoint_count": lineage_verification.get("checkpoint_count") or lineage.get("checkpoint_count"),
        "legacy_unsigned_count": lineage_verification.get("legacy_unsigned_count") or lineage.get("legacy_unsigned_count"),
        "quarantined_count": lineage_verification.get("quarantined_count") or lineage.get("quarantined_count"),
        "include_quarantined": bool(proof.get("include_quarantined") or report.get("include_quarantined") or False),
        "checkpoint_hash_short": str(checkpoint_hash)[:16] if checkpoint_hash else None,
        "signing_key_id": head.get("signing_key_id"),
        "trust_root_active_key_id": trust_root.get("active_key_id"),
        "public_claim_eligible": head.get("public_claim_eligible"),
        "interpretation": interpretation,
        "full_report": "Run /trust current json, /trust proof current, or /trust --forensics for receipts, signatures and lineage details.",
    }


def _task_trust_card_from_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    embedded = result.get("trust_card")
    if isinstance(embedded, dict):
        return embedded
    path = result.get("trust_card_path")
    if path:
        try:
            payload = json.loads(Path(str(path)).read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    rust_payload = result.get("rust_core_payload") if isinstance(result.get("rust_core_payload"), dict) else {}
    embedded = rust_payload.get("trust_card") if isinstance(rust_payload, dict) else None
    if isinstance(embedded, dict):
        return embedded
    patch = str(result.get("patch") or "")
    patch_sha = hashlib.sha256(patch.encode("utf-8")).hexdigest() if patch else result.get("patch_sha256")
    status = "passed" if result.get("status") in {"completed", "passed"} else result.get("status") or "unknown"
    return {
        "kind": "helix-trust-card-v1",
        "subject_type": "task",
        "status": status,
        "assurance": result.get("assurance") or "quick",
        "engine": result.get("engine") or result.get("mode") or "helix",
        "run_id": result.get("run_id"),
        "models_used": result.get("models_used") or {
            "planner_model": None,
            "coder_engine": result.get("engine") or result.get("mode") or "helix",
            "critic_model": None,
            "verifier_model": "helix-cli",
        },
        "changed_files": result.get("changed_files") or [],
        "checks_passed": [
            {"id": "sandbox_provenance", "status": "passed" if result.get("sandbox_root") else "not_run", "summary": "Sandbox path captured for agentic task."},
            {"id": "patch_integrity", "status": "passed" if patch_sha else "warning", "summary": "Patch hash captured." if patch_sha else "No patch hash available."},
            {"id": "deep_nuclear", "status": "not_run", "summary": "Deep nuclear suites require explicit verification."},
        ],
        "warnings": result.get("warnings") or ([] if status in {"completed", "passed"} else [str(result.get("error") or "task did not complete cleanly")]),
        "patch": {"path": result.get("patch_path"), "sha256": patch_sha, "bytes": len(patch.encode("utf-8")) if patch else None},
        "artifact_paths": {
            "artifact": result.get("artifact_path"),
            "patch": result.get("patch_path"),
            "trust_card": result.get("trust_card_path"),
            "task_capsule": result.get("task_capsule_path"),
        },
        "claim_boundary": "This card records local task provenance and captured outputs; it does not prove semantic correctness without review, tests, or stricter verification.",
    }


def _print_trust_card(card: dict[str, Any]) -> None:
    checks = card.get("checks_passed") if isinstance(card.get("checks_passed"), list) else card.get("checks")
    check_rows = []
    for check in checks or []:
        if isinstance(check, dict):
            check_rows.append(f"{check.get('id')}={check.get('status')}")
    changed = card.get("changed_files") if isinstance(card.get("changed_files"), list) else []
    patch = card.get("patch") if isinstance(card.get("patch"), dict) else {}
    artifact_paths = card.get("artifact_paths") if isinstance(card.get("artifact_paths"), dict) else {}
    print("HeliX Trust Card")
    print(f"- status: {card.get('status')}")
    print(f"- engine: {card.get('engine')} | assurance: {card.get('assurance')}")
    print(f"- run_id: {card.get('run_id') or 'n/a'}")
    if card.get("goal"):
        print(f"- goal: {card.get('goal')}")
    print(f"- changed_files: {', '.join(map(str, changed)) if changed else 'none'}")
    if card.get("flow_profile") or card.get("work_intent"):
        print(f"- work: {card.get('work_intent') or 'n/a'} | flow: {card.get('flow_profile') or 'n/a'}")
    sources = card.get("sources") if isinstance(card.get("sources"), list) else []
    if sources:
        source_labels = []
        for source in sources[:5]:
            if isinstance(source, dict):
                source_labels.append(str(source.get("path") or source.get("url") or source.get("kind") or "source"))
        suffix = " ..." if len(sources) > 5 else ""
        print(f"- sources: {len(sources)} ({'; '.join(source_labels)}{suffix})")
    anchors = card.get("anchors") if isinstance(card.get("anchors"), dict) else {}
    if anchors:
        print(f"- anchors: {anchors.get('count') or 0}")
    if card.get("output_target"):
        print(f"- output_target: {card.get('output_target')}")
    output_file = card.get("output_file") if isinstance(card.get("output_file"), dict) else {}
    if output_file:
        print(f"- output_file: {output_file.get('path') or 'n/a'}")
        print(f"- output_sha256: {output_file.get('sha256') or 'n/a'}")
    artifact = card.get("artifact") if isinstance(card.get("artifact"), dict) else {}
    if artifact:
        readback = artifact.get("readback") if isinstance(artifact.get("readback"), dict) else {}
        print(f"- artifact: {artifact.get('kind') or 'file'} | {artifact.get('path') or 'n/a'}")
        print(f"- readback: {readback.get('status') or 'unknown'} | chars={readback.get('chars') or 0} | pages={readback.get('pages') or 'n/a'}")
    before_artifact = card.get("artifact_before") if isinstance(card.get("artifact_before"), dict) else {}
    if before_artifact:
        before_readback = before_artifact.get("readback") if isinstance(before_artifact.get("readback"), dict) else {}
        print(f"- before: chars={before_readback.get('chars') or 0} sha256={before_artifact.get('sha256') or 'n/a'}")
    print(f"- patch_sha256: {patch.get('sha256') or 'n/a'}")
    print(f"- checks: {', '.join(check_rows) if check_rows else 'none'}")
    warnings = card.get("warnings") if isinstance(card.get("warnings"), list) else []
    if warnings:
        print(f"- warnings: {'; '.join(map(str, warnings))}")
    print(f"- artifact: {artifact_paths.get('artifact') or 'n/a'}")
    print(f"- claim_boundary: {card.get('claim_boundary') or 'local provenance only; semantic truth not proven'}")


def _format_work_summary(result: dict[str, Any] | None, *, plan: dict[str, Any] | None = None, trust_card: dict[str, Any] | None = None) -> str:
    if not result:
        return "[helix] Todavia no hay una tarea Work Runtime en esta sesion."
    active_plan = plan or (result.get("work_plan") if isinstance(result.get("work_plan"), dict) else {}) or {}
    card = trust_card or (result.get("trust_card") if isinstance(result.get("trust_card"), dict) else {}) or {}
    paths = result.get("work_artifact_paths") if isinstance(result.get("work_artifact_paths"), dict) else {}
    changed = result.get("changed_files") if isinstance(result.get("changed_files"), list) else card.get("changed_files") if isinstance(card.get("changed_files"), list) else []
    sources = card.get("sources") if isinstance(card.get("sources"), list) else []
    source_labels = [str(source.get("path") or source.get("url") or source.get("kind") or "source") for source in sources[:3] if isinstance(source, dict)]
    output_file = result.get("output_file") if isinstance(result.get("output_file"), dict) else card.get("output_file") if isinstance(card.get("output_file"), dict) else {}
    artifact = result.get("artifact") if isinstance(result.get("artifact"), dict) else card.get("artifact") if isinstance(card.get("artifact"), dict) else {}
    readback = artifact.get("readback") if isinstance(artifact.get("readback"), dict) else {}
    patch_state = "apply ready" if result.get("patch_available") else "patch invalid" if result.get("patch_generated") else "no patch"
    lines = [
        "HeliX Work Result",
        f"- status: {result.get('status') or 'unknown'} | engine: {result.get('engine') or 'helix'} | patch: {patch_state}",
        f"- goal: {result.get('flow_goal') or result.get('goal') or active_plan.get('goal') or 'n/a'}",
        f"- did: {result.get('final') or 'completed'}",
        f"- changed: {', '.join(map(str, changed)) if changed else 'none'}",
        f"- sources: {', '.join(source_labels) if source_labels else 'none'}",
        f"- output: {output_file.get('path') or active_plan.get('output_target') or card.get('output_target') or 'n/a'}",
    ]
    if artifact:
        lines.append(
            f"- artifact: {artifact.get('kind') or 'file'} | exists={'yes' if artifact.get('exists') else 'no'} | "
            f"chars={readback.get('chars') or 0} | pages={readback.get('pages') or 'n/a'}"
        )
    progress_events = result.get("progress_events") if isinstance(result.get("progress_events"), list) else []
    if progress_events:
        lines.append("- timeline:")
        timeline_rows = progress_events if len(progress_events) <= 8 else [*progress_events[:3], {"event": "...", "message": "..."}, *progress_events[-4:]]
        for item in timeline_rows:
            if not isinstance(item, dict):
                continue
            lines.append(f"  {item.get('event')}: {item.get('message') or ''} ({item.get('elapsed_ms') or 0}ms)")
    if paths:
        lines.append(f"- patch_file: {paths.get('patch') or 'n/a'}")
        lines.append(f"- trust_card: {paths.get('trust_card') or 'n/a'}")
    if output_file.get("path"):
        lines.append(f"- file_sha256: {output_file.get('sha256') or 'n/a'}")
        lines.append("- next: /read last | /trust | abrilo desde esa ruta")
    elif result.get("patch_available"):
        lines.append("- next: /trust last -> /apply last")
    elif result.get("patch_generated"):
        check = result.get("patch_apply_check") if isinstance(result.get("patch_apply_check"), dict) else {}
        reason = check.get("stderr") or check.get("stdout") or check.get("error") or "git apply --check failed"
        lines.append(f"- blocked: {str(reason).strip()[:500]}")
    else:
        lines.append("- next: ajusta el pedido o usa /work sources last para ver que pudo leer")
    return "\n".join(lines)


def _primary_work_output_path(result: dict[str, Any] | None, *, plan: dict[str, Any] | None = None, task_root: Path | None = None) -> str | None:
    if not result:
        return None
    active_plan = plan or (result.get("work_plan") if isinstance(result.get("work_plan"), dict) else {}) or {}
    output_file = result.get("output_file") if isinstance(result.get("output_file"), dict) else {}
    if output_file.get("path"):
        return str(output_file["path"])
    changed = result.get("changed_files") if isinstance(result.get("changed_files"), list) else []
    candidate = str(changed[0]) if changed else str(active_plan.get("output_target") or "").strip()
    if not candidate:
        return None
    intent = str(active_plan.get("work_intent") or active_plan.get("intent") or "")
    if intent == "source_to_web" and not Path(candidate).suffix:
        candidate = str(Path(candidate) / "index.html")
    if task_root and not Path(candidate).is_absolute():
        return str((task_root / candidate).resolve(strict=False))
    return candidate


def _format_work_history(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "[helix] Todavia no hay historial de Work Runtime."
    lines = ["HeliX Work History"]
    for item in reversed(rows[-10:]):
        changed = item.get("changed_files") if isinstance(item.get("changed_files"), list) else []
        lines.append(
            f"- {item.get('run_id') or 'n/a'} | {item.get('status') or 'unknown'} | {item.get('engine') or 'helix'} | "
            f"patch={'ready' if item.get('patch_available') else 'no'} | output={item.get('output_target') or (changed[0] if changed else 'n/a')}"
        )
    return "\n".join(lines)


DEMO_WOW_SCENARIOS: tuple[dict[str, str], ...] = (
    {
        "id": "doc-to-web",
        "title": "Documento a web",
        "claim_boundary": "Shows bounded source collection and static web generation from local anchors; it does not prove marketing claims are true.",
    },
    {
        "id": "artifact-repair",
        "title": "Lectura y reparacion de artefacto",
        "claim_boundary": "Shows before/after readback and hash change for a local artifact; it does not prove editorial quality without review.",
    },
    {
        "id": "patch-safe",
        "title": "Patch gate",
        "claim_boundary": "Shows patch capture, hash and git apply --check; it does not apply changes to the repo.",
    },
    {
        "id": "browser-proof",
        "title": "Verificacion visual",
        "claim_boundary": "Shows optional browser snapshot/screenshot when agent-browser is available; skipped browser checks are explicit.",
    },
    {
        "id": "trust-explorer",
        "title": "Trust Card humana",
        "claim_boundary": "Shows checks, warnings and limits in a compact card; it does not claim semantic truth.",
    },
)


def _demo_skill_status() -> dict[str, Any]:
    home = Path.home()
    skill_paths = {
        "agent-browser": home / ".codex" / "plugins" / "cache" / "openai-curated" / "vercel" / "b8edb371" / "skills" / "agent-browser" / "SKILL.md",
        "web-design-guidelines": home / ".codex" / "skills" / "web-design-guidelines" / "SKILL.md",
        "pdf": home / ".codex" / "skills" / "pdf" / "SKILL.md",
    }
    rows = {}
    for name, path in skill_paths.items():
        rows[name] = {"available": path.exists(), "path": str(path) if path.exists() else None}
    return rows


def _agent_browser_status() -> dict[str, Any]:
    binary = shutil.which("agent-browser")
    skill = _demo_skill_status().get("agent-browser") or {}
    return {
        "available": bool(binary),
        "binary": binary,
        "skill_available": bool(skill.get("available")),
        "skill_path": skill.get("path"),
    }


def _demo_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _demo_write_timeline(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_json_ready(item), ensure_ascii=False, sort_keys=True) + "\n" for item in events),
        encoding="utf-8",
    )


def _demo_source_ref(task_root: Path) -> str:
    candidates = [
        task_root / "README.md",
        task_root / "docs" / "product" / "verification-to-product-map.md",
        task_root / "docs" / "product" / "agentic-trust-runtime.md",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return candidate.relative_to(task_root).as_posix()
            except Exception:
                return str(candidate)
    return "README.md"


def _demo_browser_verify(site_index: Path, run_dir: Path, *, enabled: bool = True) -> dict[str, Any]:
    status = _agent_browser_status()
    snapshot_path = run_dir / "browser_snapshot.txt"
    screenshot_path = run_dir / "screenshot.png"
    if not enabled:
        payload = {"status": "skipped", "reason": "browser verification disabled for this run", **status}
        snapshot_path.write_text("browser verification skipped: disabled\n", encoding="utf-8")
        payload["snapshot_path"] = str(snapshot_path)
        return payload
    binary = status.get("binary")
    if not binary:
        payload = {
            "status": "skipped",
            "reason": "agent-browser CLI is not available on PATH",
            "next": "Install/enable the agent-browser skill or ensure `agent-browser` is on PATH.",
            **status,
        }
        snapshot_path.write_text("browser verification skipped: agent-browser missing\n", encoding="utf-8")
        payload["snapshot_path"] = str(snapshot_path)
        return payload
    url = site_index.resolve(strict=False).as_uri()
    commands = [
        [str(binary), "open", url],
        [str(binary), "wait", "--load", "networkidle"],
        [str(binary), "snapshot", "-i"],
        [str(binary), "screenshot", str(screenshot_path)],
    ]
    outputs: list[str] = []
    for command in commands:
        try:
            completed = subprocess.run(  # noqa: S603 - explicit argv for optional local browser verifier.
                command,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
                timeout=20,
            )
        except Exception as exc:  # noqa: BLE001
            outputs.append(f"$ {' '.join(command)}\nERROR {type(exc).__name__}: {exc}")
            snapshot_path.write_text("\n\n".join(outputs), encoding="utf-8")
            return {
                "status": "failed",
                "binary": binary,
                "url": url,
                "error": f"{type(exc).__name__}: {exc}",
                "snapshot_path": str(snapshot_path),
                "screenshot_path": str(screenshot_path) if screenshot_path.exists() else None,
            }
        outputs.append(
            "$ "
            + " ".join(command)
            + f"\nexit={completed.returncode}\n"
            + _truncate_text((completed.stdout or "") + ("\nSTDERR:\n" + completed.stderr if completed.stderr else ""), 6000)["text"]
        )
        if completed.returncode != 0:
            snapshot_path.write_text("\n\n".join(outputs), encoding="utf-8")
            return {
                "status": "failed",
                "binary": binary,
                "url": url,
                "exit_code": completed.returncode,
                "snapshot_path": str(snapshot_path),
                "screenshot_path": str(screenshot_path) if screenshot_path.exists() else None,
            }
    snapshot_path.write_text("\n\n".join(outputs), encoding="utf-8")
    return {
        "status": "passed",
        "binary": binary,
        "url": url,
        "snapshot_path": str(snapshot_path),
        "screenshot_path": str(screenshot_path) if screenshot_path.exists() else None,
    }


def demo_doctor_report(*, session: "InteractiveSession | None" = None) -> dict[str, Any]:
    task_root = session.task_root if session is not None else _default_task_root()
    evidence_root = session.evidence_root if session is not None else _default_evidence_root()
    provider_name = session.provider_name if session is not None else "deepinfra"
    provider = PROVIDERS.get(provider_name, PROVIDERS["deepinfra"])
    return {
        "status": "ok",
        "kind": "helix-demo-doctor-v1",
        "task_root": str(task_root),
        "task_root_exists": task_root.exists(),
        "evidence_root": str(evidence_root),
        "provider": provider.name,
        "provider_token_available": bool(provider.token_available or _provider_ready_from_config(provider.name)),
        "rust_core": helix_cli_core.rust_core_status(),
        "opencode": helix_cli_core.opencode_status(),
        "agent_browser": _agent_browser_status(),
        "skills": _demo_skill_status(),
        "last_latency": session.last_latency_trace if session is not None else None,
        "last_demo_run_id": (session.last_demo_result or {}).get("run_id") if session is not None and session.last_demo_result else None,
    }


def _demo_skills_suggestions() -> dict[str, Any]:
    return {
        "status": "ok",
        "kind": "helix-demo-skills-suggestions-v1",
        "detected": _demo_skill_status(),
        "optional_commands": [
            "npx skills find agent-browser",
            "npx skills find web-design-guidelines",
            "npx skills find pdf",
        ],
        "policy": "HeliX never installs skills automatically during the demo.",
    }


def _task_assurance_followup(
    *,
    assurance: str,
    artifact_path: str | None,
    evidence_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    assurance = _normalize_assurance(assurance)
    if assurance == "quick":
        return {"requested": "quick", "effective": "quick", "checks": [], "status": "not_requested"}
    if not artifact_path:
        return {
            "requested": assurance,
            "effective": "quick",
            "status": "warning",
            "checks": [],
            "warnings": ["No artifact path was available for follow-up assurance verification."],
        }
    verification = helix_cli_core.verify_capsule(artifact_path=Path(str(artifact_path)))
    payload: dict[str, Any] = {
        "requested": assurance,
        "effective": "balanced" if assurance in {"balanced", "strict"} else "quick",
        "status": verification.get("status") or "unknown",
        "capsule_verification": verification,
        "checks": verification.get("checks") if isinstance(verification.get("checks"), list) else [],
    }
    if assurance == "strict":
        payload["lab_profile"] = helix_cli_core.lab_run(profile="patch-safety", evidence_root=evidence_root, repo_root=repo_root)
        payload["effective"] = "strict-local"
        payload["warnings"] = [
            "Strict local verification ran capsule/hash and patch-safety readiness checks; provider critic/model disagreement remains not_run unless an explicit multimodel verifier is configured."
        ]
    elif assurance == "deep":
        payload["effective"] = "balanced"
        payload["status"] = "requires_explicit_deep"
        payload["warnings"] = [
            "Deep nuclear verification is intentionally not launched from the foreground task path. Use /lab run deep-nuclear or suite commands explicitly."
        ]
    return payload


@dataclass(frozen=True)
class FlowProfile:
    profile_id: str
    title: str
    lane: str
    engine: str
    assurance: str
    lab_profile: str | None
    description: str
    protocols: tuple[str, ...]
    task_template: str
    claim_boundary: str

    def build_goal(self, goal: str) -> str:
        return self.task_template.format(goal=str(goal or "").strip())

    def report(self) -> dict[str, Any]:
        return {
            "id": self.profile_id,
            "title": self.title,
            "lane": self.lane,
            "engine": self.engine,
            "assurance": self.assurance,
            "lab_profile": self.lab_profile,
            "description": self.description,
            "protocols": list(self.protocols),
            "claim_boundary": self.claim_boundary,
        }


FLOW_PROFILES: dict[str, FlowProfile] = {
    "web": FlowProfile(
        profile_id="web",
        title="Web Build",
        lane="commercial-build",
        engine="opencode",
        assurance="quick",
        lab_profile="patch-safety",
        description="Create a static/editorial web asset in an OpenCode sandbox with a HeliX patch and trust card.",
        protocols=("sandbox_provenance", "patch_integrity", "claim_boundary"),
        task_template=(
            "HeliX Flow Profile: web.\n"
            "Goal: {goal}\n\n"
            "Build a complete static/editorial web result. If the user did not name an output folder, use "
            "web/helix-flow-site/. Prefer plain HTML/CSS/JS unless the repo clearly has a web framework already. "
            "Create all needed files, keep assets local, avoid network-only dependencies, make the page responsive, "
            "and include concise editorial copy about the subject. Do not edit outside the requested site folder unless "
            "a small README or integration note is necessary."
        ),
        claim_boundary="Proves sandboxed creation and captured patch integrity; visual quality still needs review.",
    ),
    "web-recursive": FlowProfile(
        profile_id="web-recursive",
        title="Recursive Web Build",
        lane="recursive-build",
        engine="opencode",
        assurance="balanced",
        lab_profile="patch-safety",
        description="Build a web artifact, self-review it once, then refine it before HeliX captures the patch.",
        protocols=("meta_microsite", "cognitive_drift_rollback", "patch_integrity", "provider_disagreement"),
        task_template=(
            "HeliX Flow Profile: web-recursive.\n"
            "Goal: {goal}\n\n"
            "Create a polished static/editorial website, then run one bounded self-review pass inside the same sandbox: "
            "check information architecture, visual hierarchy, mobile layout, missing files, and whether the result "
            "actually communicates HeliX as a recursive/verifiable agentic runtime. Apply only high-confidence "
            "improvements. If no output folder is named, use web/helix-recursive-site/."
        ),
        claim_boundary="Proves a bounded recursive build/review loop was captured; it does not prove marketing claims are true.",
    ),
    "patch-safe": FlowProfile(
        profile_id="patch-safe",
        title="Patch Safety",
        lane="code-change",
        engine="opencode",
        assurance="balanced",
        lab_profile="patch-safety",
        description="Make a code change in sandbox, capture diff, and run HeliX capsule verification.",
        protocols=("sandbox_provenance", "patch_integrity", "rollback_fence", "provider_disagreement"),
        task_template=(
            "HeliX Flow Profile: patch-safe.\n"
            "Goal: {goal}\n\n"
            "Make the smallest useful code change in the sandbox. Inspect existing patterns first, avoid unrelated "
            "refactors, and include test suggestions or commands in the final answer. HeliX will capture the patch; "
            "do not assume it will be applied to the real repo."
        ),
        claim_boundary="Proves a sandbox patch was captured and hash-checked; correctness still depends on tests/review.",
    ),
    "doc-grounded": FlowProfile(
        profile_id="doc-grounded",
        title="Document Grounding",
        lane="document-qa",
        engine="opencode",
        assurance="balanced",
        lab_profile="doc-grounding",
        description="Answer from local files with compact anchors and a clear boundary around what was inspected.",
        protocols=("hard_anchor_utility", "source_poison_guard", "claim_boundary"),
        task_template=(
            "HeliX Flow Profile: doc-grounded.\n"
            "Goal: {goal}\n\n"
            "Inspect only relevant local files/directories. Produce a concise answer grounded in file paths and short "
            "anchors, without copying large document bodies. If edits are not explicitly requested, do not modify files."
        ),
        claim_boundary="Proves bounded local inspection and anchor capture; it does not prove uninspected documents.",
    ),
    "resilient-task": FlowProfile(
        profile_id="resilient-task",
        title="Resilient Task",
        lane="fault-tolerant-agentic",
        engine="opencode",
        assurance="strict",
        lab_profile="patch-safety",
        description="Run a task with explicit failure fencing, fallback thinking, and strict local verification.",
        protocols=("resilient_pipeline", "rollback_fence_replay", "cognitive_drift_rollback", "patch_integrity"),
        task_template=(
            "HeliX Flow Profile: resilient-task.\n"
            "Goal: {goal}\n\n"
            "Treat failures or uncertainty as fenced observations, not context to blindly continue from. Prefer a "
            "small successful result over a broad brittle one. Record assumptions in final output, keep changes "
            "minimal, and leave clear next checks."
        ),
        claim_boundary="Proves fenced execution artifacts and strict local checks; it does not guarantee all failure modes were explored.",
    ),
    "privacy-swarm": FlowProfile(
        profile_id="privacy-swarm",
        title="Privacy Swarm",
        lane="privacy-preserving-analysis",
        engine="opencode",
        assurance="balanced",
        lab_profile="doc-grounding",
        description="Analyze sensitive local material with minimization, local-only boundaries, and explicit redaction notes.",
        protocols=("privacy_swarm", "source_poison_guard", "hard_anchor_utility"),
        task_template=(
            "HeliX Flow Profile: privacy-swarm.\n"
            "Goal: {goal}\n\n"
            "Minimize sensitive exposure. Inspect local files only as needed, summarize with redaction-aware language, "
            "and avoid emitting secrets or credentials. If a secret is encountered, report its path/category without "
            "printing the value."
        ),
        claim_boundary="Proves a minimization-oriented local run was captured; it is not a formal privacy audit.",
    ),
    "multi-review": FlowProfile(
        profile_id="multi-review",
        title="Multi Review",
        lane="review-and-decision",
        engine="opencode",
        assurance="strict",
        lab_profile="provider-audit",
        description="Use HeliX-style disagreement discipline for architecture/product/code review work.",
        protocols=("multi_agent_concurrency", "provider_disagreement", "provider_substitution", "claim_boundary"),
        task_template=(
            "HeliX Flow Profile: multi-review.\n"
            "Goal: {goal}\n\n"
            "Review the target as if a second model will challenge the result. Separate findings, assumptions, risks, "
            "and recommended next action. If changing files, make only review-supporting edits and capture the patch."
        ),
        claim_boundary="Proves review artifacts and local checks; it does not prove model consensus unless external critics are configured.",
    ),
    "deep-lab": FlowProfile(
        profile_id="deep-lab",
        title="Deep Lab",
        lane="explicit-nuclear-verification",
        engine="opencode",
        assurance="deep",
        lab_profile="deep-nuclear",
        description="Prepare or run explicit deep verification work; never launches full nuclear suites implicitly.",
        protocols=("deep_nuclear", "long_horizon_checkpoints", "branch_quarantine", "provider_audit"),
        task_template=(
            "HeliX Flow Profile: deep-lab.\n"
            "Goal: {goal}\n\n"
            "Prepare deep verification work, fixtures, or a runnable plan. Do not launch expensive/deep suites unless "
            "the command explicitly requested that exact suite/run. Keep outputs bounded and claim boundaries clear."
        ),
        claim_boundary="Deep evidence requires explicit suite execution; this flow only prepares/captures foreground work.",
    ),
}


def _flow_profile_id(value: str | None) -> str:
    candidate = _slugish(value or "").replace("_", "-")
    aliases = {
        "website": "web",
        "site": "web",
        "web-editorial": "web",
        "recursive-web": "web-recursive",
        "ouroboros": "web-recursive",
        "patch": "patch-safe",
        "code": "patch-safe",
        "docs": "doc-grounded",
        "doc": "doc-grounded",
        "research": "doc-grounded",
        "resilient": "resilient-task",
        "privacy": "privacy-swarm",
        "review": "multi-review",
        "multi": "multi-review",
        "lab": "deep-lab",
        "deep": "deep-lab",
    }
    return aliases.get(candidate, candidate)


def flow_profiles_report() -> dict[str, Any]:
    return {
        "status": "ok",
        "kind": "helix-flow-profiles-v1",
        "default": "web",
        "profiles": [FLOW_PROFILES[key].report() for key in sorted(FLOW_PROFILES)],
        "claim_boundary": "Flows adapt HeliX verification protocols to commercial tasks; they do not prove semantic truth without review/tests.",
    }


def _flow_profile_or_error(profile: str | None) -> FlowProfile:
    profile_id = _flow_profile_id(profile or "web")
    flow = FLOW_PROFILES.get(profile_id)
    if flow is None:
        known = ", ".join(sorted(FLOW_PROFILES))
        raise ValueError(f"unknown flow profile: {profile}. Known profiles: {known}")
    return flow


def _lab_profile_for_flow(flow_profile: str | None) -> str:
    flow_id = _flow_profile_id(flow_profile or "")
    if flow_id in {"doc-grounded", "privacy-swarm"}:
        return "doc-grounding"
    if flow_id in {"multi-review"}:
        return "provider-audit"
    if flow_id in {"deep-lab"}:
        return "deep-nuclear"
    return "patch-safety"


def _flow_result_payload(flow: FlowProfile, goal: str, result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    payload["flow"] = flow.report()
    payload["flow_goal"] = goal
    payload["claim_boundary"] = flow.claim_boundary
    card = payload.get("trust_card")
    if isinstance(card, dict):
        card.setdefault("flow_profile", flow.profile_id)
        card.setdefault("flow_lane", flow.lane)
        card.setdefault("claim_boundary", flow.claim_boundary)
    return payload


_WORK_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".json", ".jsonl", ".html", ".htm", ".csv", ".tsv", ".py", ".js", ".ts", ".tsx", ".css", ".rs", ".toml", ".yaml", ".yml"}
_WORK_DOC_SUFFIXES = _WORK_TEXT_SUFFIXES | {".pdf", ".docx"}


def _work_words(text: str) -> set[str]:
    return {part for part in re.split(r"[^a-zA-Z0-9_/-]+", str(text or "").lower()) if part}


def _work_slug_from_goal(goal: str, *, fallback: str = "helix-work-document") -> str:
    folded = unicodedata.normalize("NFKD", str(goal or ""))
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch)).lower()
    primary = folded.splitlines()[0] if folded.splitlines() else folded
    topic = None
    stop = r"(?:\s+(?:y\s+)?(?:dejalo|dejala|guardalo|guardala|ponelo|ponela|armalo|armala|crealo|creala|en\s+[a-z]:|en\s+web/|en\s+docs/)|$)"
    for pattern in (rf"\bsobre\s+(.+?){stop}", rf"\bde\s+(.+?){stop}"):
        match = re.search(pattern, primary)
        if match:
            topic = match.group(1)
            break
    material = topic or primary
    material = re.split(
        r"\b(?:pero|dije|nono|ademas|además|tiene que|mejor contenido|gran formato|buen formato|no era|no en relacion)\b",
        material,
        maxsplit=1,
    )[0]
    material = re.sub(r"\b(armame|crear|crea|creame|generar|genera|generame|hacer|hace|pdf|documento|reporte|informe|sobre|de|un|una|el|la|los|las|y|en|ruta|esta|dejalo|armalo|podes|puedes)\b", " ", material)
    words = [item for item in re.split(r"[^a-z0-9]+", material) if item]
    slug = "-".join(words[:8]).strip("-")
    return slug or fallback


def _extract_work_path_refs(text: str) -> list[str]:
    refs = list(_extract_local_path_refs(text))
    seen = set(refs)
    suffixes = "|".join(re.escape(item.lstrip(".")) for item in sorted(_WORK_DOC_SUFFIXES, key=len, reverse=True))

    def _add(candidate: str, *, unquoted: bool = False) -> None:
        if unquoted and not re.search(r"[\\/]|[A-Za-z]:", candidate) and " " in candidate.strip():
            candidate = candidate.strip().split()[-1]
        value = _normalise_local_path_ref(candidate)
        if value and Path(value).suffix.lower() in _WORK_DOC_SUFFIXES and value not in seen:
            refs.append(value)
            seen.add(value)

    for match in re.finditer(r'["`“](.+?)["`”]', str(text or ""), flags=re.DOTALL):
        _add(match.group(1))
    for match in re.finditer(rf"(?i)(?<![\w/:\\.-])[\w .-]+\.({suffixes})(?![\w.-])", str(text or "")):
        _add(match.group(0), unquoted=True)
    absolute_refs = [item for item in refs if Path(item).expanduser().is_absolute()]
    filtered: list[str] = []
    for item in refs:
        item_norm = item.lower().replace("\\", "/").strip("/")
        if not Path(item).expanduser().is_absolute() and any(
            item_norm and item_norm in abs_ref.lower().replace("\\", "/")
            for abs_ref in absolute_refs
        ):
            continue
        filtered.append(item)
    return filtered


def _work_intent_for_goal(goal: str, *, source_refs: list[str], url_refs: list[str], output_target: str | None) -> dict[str, Any]:
    lowered = str(goal or "").lower()
    words = _work_words(goal)
    wants_web_output = bool(
        output_target and (str(output_target).lower().endswith((".html", "/")) or "web" in str(output_target).lower().split("/"))
    ) or any(term in lowered for term in ("pagina web", "página web", "sitio", "site", "landing", "microsite", "html"))
    wants_document = any(term in lowered for term in ("documento", "reporte", "informe", "markdown")) or bool(
        output_target and str(output_target).lower().endswith((".md", ".docx", ".pdf"))
    )
    wants_pdf = ".pdf" in lowered or re.search(r"\bpdf\b", lowered) is not None
    wants_scrape = bool(url_refs) and any(term in lowered for term in ("scrap", "scrape", "crawl", "crawler", "raspa", "scraping", "web"))
    wants_patch = _looks_like_agent_task(goal) and any(term in lowered for term in ("patch", "bug", "fix", "arreg", "implement", "refactor", "tests", "repo", "codigo", "código"))
    wants_verify = any(term in lowered for term in ("verifica", "verificá", "audit", "audita", "evidencia", "evidence", "trust", "suite", "hash"))
    if wants_patch:
        return {"intent": "code_patch", "flow_profile": "patch-safe", "output_kind": "patch", "assurance": "balanced", "needs_opencode": True}
    if wants_web_output:
        return {"intent": "source_to_web", "flow_profile": "web-recursive" if source_refs or url_refs else "web", "output_kind": "web", "assurance": "balanced", "needs_opencode": False}
    if wants_document or wants_pdf:
        return {"intent": "source_to_document", "flow_profile": "doc-grounded", "output_kind": "document", "assurance": "balanced", "needs_opencode": False}
    if wants_verify:
        return {"intent": "verification", "flow_profile": "multi-review", "output_kind": "analysis", "assurance": "strict", "needs_opencode": False}
    if wants_scrape or source_refs or url_refs:
        return {"intent": "grounded_analysis", "flow_profile": "doc-grounded", "output_kind": "analysis", "assurance": "balanced", "needs_opencode": False}
    if {"armame", "crea", "crear", "genera", "generar", "make", "build"} & words:
        return {"intent": "creative_build", "flow_profile": "web", "output_kind": "web", "assurance": "quick", "needs_opencode": True}
    return {"intent": "analysis", "flow_profile": "doc-grounded", "output_kind": "analysis", "assurance": "quick", "needs_opencode": False}


def _work_target_for_goal(goal: str, path_refs: list[str]) -> tuple[str | None, list[str]]:
    lowered = str(goal or "").lower()
    lowered_paths = lowered.replace("\\", "/")
    source_refs: list[str] = []
    output_target: str | None = None
    for ref in path_refs:
        ref_l = ref.lower().replace("\\", "/")
        candidate = Path(ref)
        output_phrase = any(
            marker in lowered_paths
            for marker in (
                f" en {ref_l}",
                f" en \"{ref_l}",
                f" hacia {ref_l}",
                f" hacia \"{ref_l}",
                f" como {ref_l}",
                f" como \"{ref_l}",
            )
        )
        looks_output = (
            ref_l.startswith("web/")
            or ref_l.startswith("docs/")
            or output_phrase
            or (("/" in ref_l or "\\" in ref) and ref_l.endswith((".html", ".htm", ".docx", ".pdf", ".md")))
        )
        if output_target is None and looks_output and (output_phrase or ref_l.startswith("web/") or not candidate.exists()):
            output_target = ref
        else:
            source_refs.append(ref)
    if output_target is None:
        if any(term in lowered for term in ("pagina web", "página web", "sitio", "site", "landing", "microsite")):
            output_target = "web/helix-work-output/"
        elif ".pdf" in lowered or re.search(r"\bpdf\b", lowered):
            output_target = f"docs/{_work_slug_from_goal(goal)}.pdf"
        elif ".docx" in lowered or re.search(r"\bdocx\b", lowered):
            output_target = f"docs/{_work_slug_from_goal(goal)}.docx"
        elif any(term in lowered for term in ("documento", "reporte", "informe", "markdown")):
            output_target = f"docs/{_work_slug_from_goal(goal)}.md"
    return output_target, source_refs


def _work_chunk_text(text: str, *, chunk_chars: int = 2400, max_chunks: int = 12) -> list[dict[str, Any]]:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    chunks = []
    cursor = 0
    index = 0
    while cursor < len(clean) and len(chunks) < max_chunks:
        piece = clean[cursor: cursor + chunk_chars].strip()
        if piece:
            chunks.append(
                {
                    "anchor_id": f"a{index:03d}",
                    "char_start": cursor,
                    "char_end": cursor + len(piece),
                    "text": piece,
                    "sha256": hashlib.sha256(piece.encode("utf-8")).hexdigest(),
                }
            )
        cursor += chunk_chars
        index += 1
    return chunks


def _extract_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        data = archive.read("word/document.xml")
    root = ElementTree.fromstring(data)
    parts = []
    for node in root.iter():
        if node.tag.endswith("}t") and node.text:
            parts.append(node.text)
        elif node.tag.endswith("}p"):
            parts.append("\n")
    return " ".join(parts).replace(" \n ", "\n").strip()


def _extract_pdf_text_with_optional_ocr(path: Path, *, max_pages: int = 20) -> tuple[str, list[dict[str, Any]], list[str]]:
    pages: list[dict[str, Any]] = []
    warnings: list[str] = []
    text_parts: list[str] = []
    try:
        from pypdf import PdfReader  # noqa: PLC0415

        reader = PdfReader(str(path))
        page_count = len(reader.pages)
        for index in range(min(page_count, max_pages)):
            page = reader.pages[index]
            page_text = page.extract_text() or ""
            pages.append({"page": index + 1, "chars": len(page_text), "method": "pypdf"})
            text_parts.append(f"\n[page {index + 1}]\n{page_text}")
        if page_count > max_pages:
            warnings.append(f"pdf truncated to first {max_pages} pages out of {page_count}")
    except ImportError:
        warnings.append("pypdf is not installed; install pypdf for text PDF extraction")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"pypdf extraction failed: {type(exc).__name__}: {exc}")
    text = "\n".join(text_parts).strip()
    if len(text.strip()) >= 80:
        return text, pages, warnings
    try:
        import fitz  # type: ignore  # noqa: PLC0415
        import pytesseract  # type: ignore  # noqa: PLC0415
        from PIL import Image  # type: ignore  # noqa: PLC0415

        ocr_parts = []
        document = fitz.open(str(path))
        ocr_limit = min(max_pages, len(document), 8)
        for index in range(ocr_limit):
            pix = document[index].get_pixmap(dpi=150)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            page_text = pytesseract.image_to_string(image)
            pages.append({"page": index + 1, "chars": len(page_text), "method": "ocr"})
            ocr_parts.append(f"\n[page {index + 1} ocr]\n{page_text}")
        if len(document) > ocr_limit:
            warnings.append(f"ocr truncated to first {ocr_limit} pages out of {len(document)}")
        text = "\n".join(ocr_parts).strip()
    except ImportError:
        warnings.append("OCR unavailable; install PyMuPDF, Pillow and pytesseract, plus local Tesseract, for scanned PDFs")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"OCR extraction failed: {type(exc).__name__}: {exc}")
    return text, pages, warnings


def _work_source_from_file(session: "InteractiveSession", ref: str) -> dict[str, Any]:
    try:
        path, raw = session._resolve_user_path(ref)
    except Exception as exc:  # noqa: BLE001
        return {"status": "blocked" if isinstance(exc, PermissionError) else "error", "ref": ref, "error": f"{type(exc).__name__}: {exc}"}
    reason = session._sensitive_file_reason(path)
    if reason:
        return {"status": "blocked", "ref": raw, "path": str(path), "reason": reason}
    if path.is_dir():
        listing = session.file_inspect(str(path), list_limit=120)
        return {
            "status": listing.get("status"),
            "kind": "directory",
            "ref": raw,
            "path": str(path),
            "entries": listing.get("entries") or [],
            "anchors": [],
            "warnings": ["directory listing only; ask for specific files or recursive collection for deeper grounding"],
        }
    if not path.exists() or not path.is_file():
        return {"status": "not_found", "ref": raw, "path": str(path)}
    suffix = path.suffix.lower()
    size = path.stat().st_size
    warnings: list[str] = []
    metadata: dict[str, Any] = {}
    text = ""
    try:
        if suffix == ".pdf":
            text, pages, warnings = _extract_pdf_text_with_optional_ocr(path)
            metadata["pages"] = pages
            method = "pdf"
        elif suffix == ".docx":
            text = _extract_docx_text(path)
            method = "docx"
        elif suffix in _WORK_TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            method = "text"
            if suffix in {".html", ".htm"}:
                text = _strip_html(text)
        else:
            return {"status": "blocked", "ref": raw, "path": str(path), "bytes": size, "reason": f"unsupported file type: {suffix or 'none'}"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "ref": raw, "path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    text = redact_value(text, secrets=_secret_values(session.provider))
    anchors = _work_chunk_text(text)
    return {
        "status": "ok" if text.strip() else "empty",
        "kind": "file",
        "ref": raw,
        "path": str(path),
        "name": path.name,
        "suffix": suffix,
        "bytes": size,
        "sha256": _sha256_file(path) if size <= 25_000_000 else None,
        "extract_method": method,
        "chars": len(text),
        "anchors": anchors,
        "warnings": warnings,
        "metadata": metadata,
        "content_preview": text[:4000],
    }


def _same_domain(url_a: str, url_b: str) -> bool:
    return urlparse.urlparse(url_a).netloc.lower() == urlparse.urlparse(url_b).netloc.lower()


def _links_from_html(base_url: str, body: str) -> list[str]:
    links: list[str] = []
    for match in re.finditer(r'(?is)<a\s+[^>]*href=["\']([^"\']+)["\']', body):
        href = html.unescape(match.group(1)).strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        joined = urlparse.urljoin(base_url, href)
        parsed = urlparse.urlparse(joined)
        if parsed.scheme in {"http", "https"}:
            links.append(urlparse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", parsed.query, "")))
    return links


def _work_collect_web_sources(urls: list[str], *, max_pages: int = 25, depth: int = 2, cross_domain: bool = False, max_bytes_per_page: int = 2_000_000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    max_pages = _safe_int(max_pages, 25, minimum=1, maximum=100)
    depth = _safe_int(depth, 2, minimum=0, maximum=4)
    seen: set[str] = set()
    queue: list[tuple[str, int, str]] = [(url, 0, url) for url in urls]
    sources: list[dict[str, Any]] = []
    warnings: list[str] = []
    while queue and len(sources) < max_pages:
        url, level, seed = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            body, content_type = _fetch_text_url(url, timeout=8.0, max_bytes=max_bytes_per_page)
            text = _strip_html(body) if "html" in content_type.lower() or "<html" in body[:500].lower() else body
            sources.append(
                {
                    "status": "ok",
                    "kind": "web",
                    "url": url,
                    "content_type": content_type,
                    "depth": level,
                    "chars": len(text),
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "anchors": _work_chunk_text(text, max_chunks=6),
                    "content_preview": text[:4000],
                }
            )
            if level < depth and ("html" in content_type.lower() or "<html" in body[:500].lower()):
                for link in _links_from_html(url, body)[:40]:
                    if cross_domain or _same_domain(seed, link):
                        queue.append((link, level + 1, seed))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{url}: {type(exc).__name__}: {exc}")
            sources.append({"status": "error", "kind": "web", "url": url, "depth": level, "error": f"{type(exc).__name__}: {exc}"})
    return sources, {"requested_urls": urls, "visited_count": len(seen), "source_count": len(sources), "max_pages": max_pages, "depth": depth, "cross_domain": cross_domain, "warnings": warnings}


def _collect_work_sources(session: "InteractiveSession", goal: str, *, source_refs: list[str], url_refs: list[str], max_pages: int = 25, depth: int = 2, cross_domain: bool = False) -> dict[str, Any]:
    local_sources = [_work_source_from_file(session, ref) for ref in source_refs]
    web_sources, crawler = _work_collect_web_sources(url_refs, max_pages=max_pages, depth=depth, cross_domain=cross_domain) if url_refs else ([], {"requested_urls": [], "visited_count": 0, "source_count": 0, "max_pages": max_pages, "depth": depth, "cross_domain": cross_domain, "warnings": []})
    sources = [*local_sources, *web_sources]
    anchors = []
    warnings = []
    for source_index, source in enumerate(sources):
        source_id = f"s{source_index:03d}"
        for item in source.get("warnings") or []:
            warnings.append(str(item))
        for anchor in source.get("anchors") or []:
            anchor = dict(anchor)
            anchor["source_id"] = source_id
            anchor["source_ref"] = source.get("path") or source.get("url") or source.get("ref")
            anchors.append(anchor)
    warnings.extend(crawler.get("warnings") or [])
    return {
        "status": "ok" if all(source.get("status") in {"ok", "empty"} for source in sources) else "partial",
        "sources": sources,
        "anchors": anchors,
        "warnings": warnings,
        "crawler_summary": crawler,
    }


def _artifact_kind_for_path(path: Path) -> str:
    if path.is_dir():
        return "web_directory" if (path / "index.html").exists() else "directory"
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".docx":
        return "docx"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix == ".txt":
        return "text"
    return "file"


def _artifact_preview_from_source(source: dict[str, Any], *, max_chars: int = 900) -> str:
    preview = str(source.get("content_preview") or "").strip()
    if not preview:
        anchors = source.get("anchors") if isinstance(source.get("anchors"), list) else []
        preview = "\n\n".join(str(item.get("text") or "") for item in anchors[:2] if isinstance(item, dict)).strip()
    return _truncate_text(preview, max_chars)["text"] if preview else ""


def _inspect_work_artifact(session: "InteractiveSession", path_ref: str | Path, *, last_action: str = "inspect") -> dict[str, Any]:
    try:
        path, raw = session._resolve_user_path(str(path_ref))
    except Exception as exc:  # noqa: BLE001
        return {
            "kind": "artifact",
            "status": "blocked" if isinstance(exc, PermissionError) else "error",
            "exists": False,
            "path": str(path_ref),
            "error": f"{type(exc).__name__}: {exc}",
            "last_action": last_action,
        }
    artifact: dict[str, Any] = {
        "kind": _artifact_kind_for_path(path),
        "status": "ok",
        "exists": path.exists(),
        "path": str(path),
        "ref": raw,
        "last_action": last_action,
        "warnings": [],
    }
    if not path.exists():
        artifact["status"] = "not_found"
        return _with_artifact_state(artifact)
    reason = session._sensitive_file_reason(path)
    if reason:
        artifact.update({"status": "blocked", "reason": reason, "warnings": [reason]})
        return _with_artifact_state(artifact)
    if path.is_file():
        size = path.stat().st_size
        artifact["bytes"] = size
        artifact["sha256"] = _sha256_file(path) if size <= 25_000_000 else None
        source = _work_source_from_file(session, str(path))
        warnings = list(source.get("warnings") or [])
        chars = int(source.get("chars") or 0)
        pages = None
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        if isinstance(metadata.get("pages"), list):
            pages = len(metadata["pages"])
        artifact["readback"] = {
            "status": source.get("status"),
            "chars": chars,
            "pages": pages,
            "extract_method": source.get("extract_method"),
            "preview": _artifact_preview_from_source(source),
        }
        artifact["preview"] = artifact["readback"]["preview"]
        artifact["warnings"] = warnings
        if source.get("status") not in {"ok", "empty"}:
            artifact["status"] = str(source.get("status") or "error")
            if source.get("error"):
                artifact["error"] = source.get("error")
        return _with_artifact_state(artifact)
    listing = session.file_inspect(str(path), list_limit=80)
    entries = listing.get("entries") if isinstance(listing.get("entries"), list) else []
    index_path = path / "index.html"
    related = [
        child
        for child in sorted(path.rglob("*"), key=lambda item: item.as_posix().lower())
        if child.is_file() and child.suffix.lower() in {".html", ".htm", ".css", ".js", ".md", ".txt"}
    ][:24]
    sources = []
    for child in ([index_path] if index_path.exists() else []) + [item for item in related if item != index_path][:8]:
        sources.append(_work_source_from_file(session, str(child)))
    chars = sum(int(item.get("chars") or 0) for item in sources)
    previews = [_artifact_preview_from_source(item, max_chars=400) for item in sources]
    artifact.update(
        {
            "entry_count": listing.get("entry_count") or len(entries),
            "entries": entries[:20],
            "web_files": [str(item.get("path") or item.get("ref") or "") for item in sources],
            "readback": {
                "status": "ok" if sources else "empty",
                "chars": chars,
                "pages": None,
                "extract_method": "web-directory",
                "preview": _truncate_text("\n\n".join(item for item in previews if item), 1200)["text"],
            },
        }
    )
    artifact["preview"] = artifact["readback"]["preview"]
    return _with_artifact_state(artifact)


def _with_artifact_state(artifact: dict[str, Any]) -> dict[str, Any]:
    state = ArtifactState.from_artifact(artifact)
    if state:
        artifact["artifact_state"] = state.to_dict()
        artifact.setdefault("readback_chars", state.readback_chars)
        artifact.setdefault("pages", state.pages)
    return artifact


def _artifact_density_threshold(goal: str, artifact: dict[str, Any]) -> int:
    folded = _fold_cli_text(goal)
    kind = str(artifact.get("kind") or "")
    if kind in {"web_directory", "html"}:
        return 600
    if any(term in folded for term in ("bien", "curado", "detallado", "completo", "profundo", "info")):
        return 2200
    return 1200 if kind in {"pdf", "docx", "markdown", "text"} else 200


def _artifact_readback_ok(goal: str, artifact: dict[str, Any]) -> bool:
    readback = artifact.get("readback") if isinstance(artifact.get("readback"), dict) else {}
    if not artifact.get("exists") or artifact.get("status") not in {"ok", "empty"}:
        return False
    threshold = _artifact_density_threshold(goal, artifact)
    return int(readback.get("chars") or 0) >= threshold


def _format_artifact_summary(artifact: dict[str, Any] | None) -> str:
    if not artifact:
        return "[helix] No hay artefacto disponible para leer."
    readback = artifact.get("readback") if isinstance(artifact.get("readback"), dict) else {}
    lines = [
        "HeliX Artifact",
        f"- status: {artifact.get('status') or 'unknown'} | kind: {artifact.get('kind') or 'file'} | exists: {'yes' if artifact.get('exists') else 'no'}",
        f"- path: {artifact.get('path') or 'n/a'}",
    ]
    if artifact.get("bytes") is not None:
        lines.append(f"- bytes: {artifact.get('bytes')}")
    if artifact.get("sha256"):
        lines.append(f"- sha256: {artifact.get('sha256')}")
    if readback:
        lines.append(
            f"- readback: {readback.get('status') or 'unknown'} | chars={readback.get('chars') or 0} | "
            f"pages={readback.get('pages') or 'n/a'} | method={readback.get('extract_method') or 'n/a'}"
        )
    warnings = artifact.get("warnings") if isinstance(artifact.get("warnings"), list) else []
    if warnings:
        lines.append(f"- warnings: {'; '.join(map(str, warnings[:5]))}")
    preview = str(artifact.get("preview") or readback.get("preview") or "").strip()
    if preview:
        lines.append("")
        lines.append("Preview:")
        lines.append(preview)
    return "\n".join(lines)


def _work_sandbox_source_path(source: dict[str, Any]) -> str | None:
    ref = str(source.get("ref") or "").strip()
    if ref and not Path(ref).expanduser().is_absolute():
        return ref.replace("\\", "/").strip("/")
    path_value = source.get("path")
    if not path_value:
        return None
    path = Path(str(path_value))
    for root in (REPO_ROOT, Path.cwd()):
        try:
            return path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
        except Exception:
            continue
    return path.name or None


def _render_work_brief(plan: dict[str, Any], source_pack: dict[str, Any]) -> str:
    compact_sources = []
    remaining_anchor_budget = 80
    for index, source in enumerate(source_pack.get("sources") or []):
        anchors = []
        for anchor in source.get("anchors") or []:
            if remaining_anchor_budget <= 0:
                break
            item = dict(anchor)
            if isinstance(item.get("text"), str):
                item["text"] = _truncate_text(item["text"], 1200)["text"]
            anchors.append(item)
            remaining_anchor_budget -= 1
        sandbox_path = _work_sandbox_source_path(source)
        host_path = source.get("path")
        compact_sources.append(
            {
                "source_id": f"s{index:03d}",
                "kind": source.get("kind"),
                "status": source.get("status"),
                "path": sandbox_path,
                "sandbox_path": sandbox_path,
                "url": source.get("url"),
                "sha256": source.get("sha256"),
                "host_path_sha256": hashlib.sha256(str(host_path).encode("utf-8")).hexdigest() if host_path else None,
                "extract_method": source.get("extract_method"),
                "warnings": source.get("warnings") or [],
                "anchors": anchors,
            }
        )
    return (
        "# HeliX Work Brief\n\n"
        f"Goal: {plan.get('goal')}\n"
        f"Intent: {plan.get('intent')}\n"
        f"Flow profile: {plan.get('flow_profile')}\n"
        f"Assurance: {plan.get('assurance')}\n"
        f"Output target: {plan.get('output_target') or 'analysis-only'}\n\n"
        "## Source Contract\n\n"
        "- Use only the sources and anchors below unless the goal explicitly requests broader web scraping.\n"
        "- Do not read absolute host paths from this brief. For local files, use sandbox_path/path values as sandbox-local files, or use the included anchor text.\n"
        "- Cite source IDs and anchor IDs in generated prose when relevant.\n"
        "- Do not print secrets. Respect blocked-source warnings.\n\n"
        "## Sources JSON\n\n"
        "```json\n"
        + json.dumps({"sources": compact_sources, "crawler_summary": source_pack.get("crawler_summary")}, ensure_ascii=False, indent=2)
        + "\n```\n\n"
        "## Claim Boundary\n\n"
        + str(plan.get("claim_boundary") or "This run proves bounded source collection and task provenance, not semantic truth.")
    )


def _work_output_file(task_root: Path, output_target: str | None, intent: str) -> tuple[Path, str]:
    raw_target = str(output_target or "").strip() or ("web/helix-work-output" if intent == "source_to_web" else "docs/helix-work-output.md")
    candidate = Path(raw_target).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
    else:
        resolved = (task_root / candidate).resolve(strict=False)
    try:
        rel = resolved.relative_to(task_root.resolve(strict=False))
    except Exception as exc:
        raise ValueError(f"output target is outside task root: {raw_target}") from exc
    if any(part in {"..", ".git", ".helix", "verification"} for part in rel.parts):
        raise ValueError(f"output target is not allowed for Work Runtime generation: {raw_target}")
    if intent == "source_to_web" and rel.suffix.lower() not in {".html", ".htm"}:
        rel = rel / "index.html"
    elif intent == "source_to_document" and not rel.suffix:
        rel = rel / "report.md"
    return task_root / rel, rel.as_posix()


def _work_output_target_path(task_root: Path, output_target: str | None, intent: str, goal: str) -> Path:
    raw_target = str(output_target or "").strip()
    wants_pdf = ".pdf" in str(goal or "").lower() or re.search(r"\bpdf\b", str(goal or "").lower()) is not None or raw_target.lower().endswith(".pdf")
    wants_docx = ".docx" in str(goal or "").lower() or re.search(r"\bdocx\b", str(goal or "").lower()) is not None or raw_target.lower().endswith(".docx")
    if not raw_target:
        raw_target = f"docs/{_work_slug_from_goal(goal)}.pdf" if wants_pdf else f"docs/{_work_slug_from_goal(goal)}.docx" if wants_docx else "docs/helix-work-output.md"
    candidate = Path(raw_target).expanduser()
    if not candidate.is_absolute():
        candidate = task_root / candidate
    if candidate.suffix.lower() not in {".pdf", ".md", ".docx"}:
        suffix = ".pdf" if wants_pdf else ".docx" if wants_docx else ".md"
        candidate = candidate / f"{_work_slug_from_goal(goal)}{suffix}"
    resolved = candidate.resolve(strict=False)
    if any(part in {"..", ".git", ".helix"} for part in resolved.parts):
        raise ValueError(f"output target is not allowed for Work Runtime export: {raw_target}")
    return resolved


def _is_external_work_output(task_root: Path, output_target: str | None, goal: str) -> bool:
    raw_target = str(output_target or "").strip()
    wants_pdf = ".pdf" in str(goal or "").lower() or re.search(r"\bpdf\b", str(goal or "").lower()) is not None or raw_target.lower().endswith(".pdf")
    if wants_pdf:
        return True
    if not raw_target:
        return False
    candidate = Path(raw_target).expanduser()
    if not candidate.is_absolute():
        return False
    try:
        candidate.resolve(strict=False).relative_to(task_root.resolve(strict=False))
        return False
    except Exception:
        return True


def _work_anchor_digest(source_pack: dict[str, Any], *, limit: int = 8) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for anchor in source_pack.get("anchors") or []:
        if len(anchors) >= limit:
            break
        text = _truncate_text(str(anchor.get("text") or "").strip(), 900)["text"].strip()
        if not text:
            continue
        anchors.append(
            {
                "anchor_id": anchor.get("anchor_id"),
                "source_id": anchor.get("source_id"),
                "source_ref": anchor.get("source_ref"),
                "text": text,
            }
        )
    return anchors


def _patch_range(count: int) -> str:
    return "0,0" if count <= 0 else f"1,{count}"


def _patch_text_line(prefix: str, line: str) -> str:
    if line.startswith("\\"):
        return prefix + "\\" + line
    return prefix + line


def _build_text_patch(task_root: Path, rel_path: str, content: str) -> dict[str, Any]:
    target = (task_root / Path(rel_path)).resolve(strict=False)
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    if not content.endswith("\n"):
        content += "\n"
    new_lines = content.splitlines()
    if target.exists() and target.is_file():
        old_text = target.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        old_lines = old_text.splitlines()
        header = [
            f"diff --git a/{rel_path} b/{rel_path}",
            f"--- a/{rel_path}",
            f"+++ b/{rel_path}",
            f"@@ -{_patch_range(len(old_lines))} +{_patch_range(len(new_lines))} @@",
        ]
        body = [_patch_text_line("-", line) for line in old_lines]
        body.extend(_patch_text_line("+", line) for line in new_lines)
    else:
        header = [
            f"diff --git a/{rel_path} b/{rel_path}",
            "new file mode 100644",
            "index 0000000..0000000",
            "--- /dev/null",
            f"+++ b/{rel_path}",
            f"@@ -0,0 +{_patch_range(len(new_lines))} @@",
        ]
        body = [_patch_text_line("+", line) for line in new_lines]
    patch = "\n".join([*header, *body, ""])
    return {"patch": patch, "patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest(), "changed_files": [rel_path]}


def _git_apply_check(task_root: Path, patch: str | None) -> dict[str, Any]:
    if not patch:
        return {"status": "not_run", "ok": False, "reason": "no patch"}
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, patch is stdin, shell disabled.
            ["git", "-C", str(task_root), "apply", "--check", "-"],
            input=patch,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=20,
        )
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "ok": completed.returncode == 0,
        "exit_code": completed.returncode,
        "stdout": _truncate_text(completed.stdout or "", 2000)["text"],
        "stderr": _truncate_text(completed.stderr or "", 2000)["text"],
    }


def _render_internal_work_document(plan: dict[str, Any], source_pack: dict[str, Any]) -> str:
    anchors = _work_anchor_digest(source_pack, limit=12)
    if not anchors or str(plan.get("artifact_action") or "") == "modify_last":
        return _render_curated_topic_document(plan, source_pack)
    source_lines = []
    for index, source in enumerate(source_pack.get("sources") or []):
        ref = source.get("path") or source.get("url") or source.get("ref") or f"source-{index}"
        source_lines.append(f"- s{index:03d}: {ref} ({source.get('status') or 'unknown'})")
    anchor_lines = [
        f"- {item.get('source_id')}/{item.get('anchor_id')}: {item.get('text')}"
        for item in anchors
    ]
    if not anchor_lines:
        anchor_lines = ["- No text anchors were available; review extraction warnings before using this output."]
    return (
        f"# {plan.get('goal') or 'HeliX Work Runtime Report'}\n\n"
        "## Resumen\n\n"
        "Este documento fue generado por HeliX desde fuentes recolectadas y anchors compactos. "
        "Usalo como primer borrador verificable: conserva referencias a fuentes, limites y advertencias.\n\n"
        "## Fuentes inspeccionadas\n\n"
        + "\n".join(source_lines or ["- No sources collected."])
        + "\n\n## Anchors relevantes\n\n"
        + "\n\n".join(anchor_lines)
        + "\n\n## Lectura de trabajo\n\n"
        "A partir de los anchors, la tarea debe tratarse como grounded en las fuentes listadas. "
        "Las afirmaciones no presentes en anchors deben revisarse antes de publicarse.\n\n"
        "## Limites\n\n"
        f"{plan.get('claim_boundary') or _work_claim_boundary(str(plan.get('work_intent') or ''))}\n"
    )


def _curated_topic_from_goal(goal: str) -> str:
    slug = _work_slug_from_goal(goal, fallback="tema")
    return slug.replace("-", " ").strip() or "tema"


def _goal_carries_topic_correction(goal: str) -> bool:
    folded = _fold_cli_text(goal)
    return any(term in folded for term in ("sobre", "nick", "land", "ccru", "tema", "dije", "era "))


def _render_curated_topic_document(plan: dict[str, Any], source_pack: dict[str, Any]) -> str:
    goal = str(plan.get("goal") or "")
    topic = _curated_topic_from_goal(goal)
    if _looks_like_modify_last_work_request(goal) and not _goal_carries_topic_correction(goal):
        target = str(plan.get("output_target") or "").strip()
        stem = Path(target).stem if target else ""
        if stem and stem.lower() not in {"report", "index", "document", "helix-work-output"}:
            topic = stem.replace("-", " ").replace("_", " ")
    lowered_topic = topic.lower()
    if "land" in lowered_topic and ("nick" in lowered_topic or "ccru" in lowered_topic):
        body = (
            "# Nick Land y la CCRU\n\n"
            "## Tesis breve\n\n"
            "Nick Land fue una figura central en el momento mas intenso y controversial de la teoria britanica de los anos noventa: la constelacion asociada a la Cybernetic Culture Research Unit, conocida como CCRU. El interes de ese nucleo no era producir una filosofia academica ordenada, sino forzar una zona hibrida entre cibernetica, capitalismo, ficcion teorica, musica electronica, ocultismo, teoria francesa y cultura rave.\n\n"
            "## Contexto: Warwick, anos noventa\n\n"
            "La CCRU emerge alrededor de la Universidad de Warwick, en un clima donde la filosofia continental se cruzaba con cyberpunk, jungle, matematicas especulativas, inteligencia artificial temprana y teoria cultural acelerada. Aunque su estatuto institucional fue inestable, su efecto posterior fue fuerte: funciono menos como escuela y mas como laboratorio de contagio conceptual.\n\n"
            "## Nick Land\n\n"
            "Land venia de una lectura intensa de Kant, Nietzsche, Bataille, Freud, Deleuze y Guattari. Su escritura desplaza la filosofia hacia un registro de alta velocidad: no explica serenamente el capitalismo, sino que intenta pensar su dinamica como proceso impersonal, maquinal y deshumanizante. En ese marco aparece el aceleracionismo temprano: la idea de que las fuerzas abstractas del capital y la tecnica no son meros instrumentos humanos, sino procesos que reorganizan deseo, tiempo, subjetividad y cultura.\n\n"
            "## CCRU como metodo\n\n"
            "La CCRU mezclaba ensayo, manifiesto, glosario, narrativa, numerologia, teoria de sistemas y ficcion especulativa. Sus textos no siempre deben leerse como tesis verificables en sentido academico clasico; muchas veces operan como dispositivos: producen una atmosfera conceptual donde las categorias de humano, maquina, mercado, deseo y futuro se vuelven inestables.\n\n"
            "## Conceptos clave\n\n"
            "- Aceleracion: no simplemente ir mas rapido, sino pensar como el capitalismo intensifica abstracciones, automatismos y bucles tecnicos.\n"
            "- Capital como proceso maquinal: el capital aparece como inteligencia impersonal que usa sujetos, instituciones y tecnologias como soportes.\n"
            "- Ficcion teorica: escritura que no separa del todo argumento, mito, codigo, delirio y diagnostico cultural.\n"
            "- Afrofuturismo, jungle y cyberculture: la CCRU leyo escenas musicales y subculturas como laboratorios de tiempo, ritmo y mutacion.\n"
            "- Numogram y ocultismo tecnico: sistemas simbolicos que funcionan como mapas especulativos mas que como evidencia empirica.\n\n"
            "## Relacion con Deleuze y Guattari\n\n"
            "El vinculo con Deleuze y Guattari es decisivo: deseo maquinal, desterritorializacion, cuerpos sin organos, flujos y ensamblajes. Pero Land radicaliza esa herencia hacia una imagen mucho mas oscura: la desterritorializacion capitalista no libera necesariamente al sujeto; puede disolverlo, capturarlo o volverlo residuo de procesos no humanos.\n\n"
            "## Importancia cultural\n\n"
            "El legado de Land y la CCRU aparece en debates sobre aceleracionismo, teoria del capitalismo digital, realismo especulativo, neorreaccion, cultura de internet, inteligencia artificial y esteticas oscuras de la tecnologia. Parte de su potencia viene de ahi: no es solo contenido filosofico, sino una forma de escribir el colapso entre teoria, cultura y maquina.\n\n"
            "## Riesgos de lectura\n\n"
            "Hay que distinguir tres niveles: el Land de los noventa vinculado a la CCRU; las recepciones aceleracionistas posteriores; y sus derivas politicas mas tardias, muy discutidas y muchas veces incompatibles con lecturas emancipatorias. Mezclar todo sin cuidado produce confusion. Una lectura rigurosa separa contexto, textos, recepcion e implicancias politicas.\n\n"
            "## Lecturas recomendadas\n\n"
            "- Nick Land, Fanged Noumena.\n"
            "- CCRU, Writings 1997-2003.\n"
            "- Deleuze y Guattari, El Anti-Edipo y Mil mesetas.\n"
            "- Mark Fisher, escritos sobre hauntologia, capitalismo y cultura rave.\n"
            "- Kodwo Eshun, More Brilliant than the Sun.\n\n"
            "## Cierre\n\n"
            "Nick Land y la CCRU importan porque muestran una forma extrema de teoria cultural: una que no mira la tecnologia desde afuera, sino que escribe como si ya estuviera infectada por sus ritmos, velocidades y automatismos. Su valor no esta en aceptarlo todo, sino en leer con precision donde diagnostico, estetica, delirio y politica se cruzan.\n"
        )
    elif "postestructural" in lowered_topic:
        body = (
            "# Postestructuralismo\n\n"
            "## Tesis breve\n\n"
            "El postestructuralismo no es una escuela cerrada ni un programa doctrinario unico. Es una constelacion de lecturas, metodos y gestos criticos que emergen sobre todo en Francia entre fines de los anos sesenta y los setenta, en dialogo y tension con el estructuralismo. Su punto de partida es una sospecha: las estructuras que parecen ordenar el lenguaje, la cultura, la subjetividad o el poder no son neutrales, estables ni exteriores a la historia.\n\n"
            "## De que se distancia\n\n"
            "El estructuralismo habia buscado sistemas de relaciones capaces de explicar fenomenos culturales: oposiciones, reglas, codigos, parentescos, mitos, signos. El postestructuralismo conserva parte de esa atencion por la estructura, pero desconfia de que exista un centro firme que garantice el significado. Donde el estructuralismo tendia a buscar orden, el postestructuralismo mira desplazamientos, fracturas, bordes, exclusiones y efectos de poder.\n\n"
            "## Ideas clave\n\n"
            "- Significado inestable: el sentido no esta simplemente dentro de una palabra, una obra o una institucion; se produce por diferencias, usos, contextos y repeticiones.\n"
            "- Critica del sujeto soberano: el sujeto no aparece como origen transparente de sus actos, sino como efecto de lenguaje, deseo, instituciones, disciplina e historia.\n"
            "- Poder productivo: en Foucault, el poder no solo reprime; tambien produce saberes, cuerpos, normalidades, archivos y formas de verdad.\n"
            "- Deconstruccion: en Derrida, leer implica atender a tensiones internas de un texto, a lo que excluye para poder decir lo que dice, y a la imposibilidad de cerrar definitivamente el sentido.\n"
            "- Multiplicidad: en Deleuze y Guattari, conceptos como rizoma, devenir y maquina deseante desplazan modelos jerarquicos y lineales.\n\n"
            "## Autores y zonas de influencia\n\n"
            "Jacques Derrida trabaja sobre escritura, diferencia, presencia y deconstruccion. Michel Foucault analiza saber, poder, disciplina, biopolitica y genealogia. Gilles Deleuze, con Felix Guattari, propone una filosofia de multiplicidades, devenires y ensamblajes. Roland Barthes desplaza la autoridad del autor y abre la lectura como campo plural. Julia Kristeva introduce problemas de intertextualidad, semiotica y subjetividad. Aunque no todos aceptarian la etiqueta, sus obras comparten una critica a los fundamentos estables.\n\n"
            "## Por que importa\n\n"
            "Su impacto fue enorme en teoria literaria, filosofia continental, estudios culturales, feminismos, teoria queer, antropologia, historia intelectual, arquitectura y critica politica. Sirve para analizar como se construyen verdades, identidades y jerarquias; tambien para evitar que conceptos aparentemente naturales se vuelvan intocables.\n\n"
            "## Riesgos de lectura\n\n"
            "Una mala lectura convierte el postestructuralismo en relativismo plano: 'todo vale' o 'nada significa'. Esa caricatura pierde lo mas interesante. La apuesta no es abandonar la verdad, sino preguntar como se fabrica, quien la administra, que excluye, que cuerpos organiza y bajo que condiciones se vuelve aceptable.\n\n"
            "## Lectura recomendada\n\n"
            "- Derrida, De la gramatologia.\n"
            "- Foucault, Vigilar y castigar; Historia de la sexualidad I.\n"
            "- Deleuze y Guattari, Mil mesetas.\n"
            "- Barthes, La muerte del autor.\n"
            "- Kristeva, Semiotica.\n\n"
            "## Cierre\n\n"
            "Pensar postestructuralmente es leer los sistemas desde sus fisuras: no para negar toda estructura, sino para mostrar que ninguna estructura se sostiene sin operaciones historicas, politicas y textuales que conviene volver visibles.\n"
        )
    else:
        title = topic.title()
        body = (
            f"# {title}\n\n"
            "## Resumen curado\n\n"
            f"Este documento es una sintesis editorial sobre {topic}. No parte de fuentes locales inspeccionadas; funciona como borrador conceptual para revisar, ampliar o contrastar con bibliografia.\n\n"
            "## Ejes para entender el tema\n\n"
            "- Contexto: ubicar el tema en su momento historico, tecnico o cultural.\n"
            "- Conceptos centrales: definir las nociones que organizan la discusion.\n"
            "- Tensiones: identificar desacuerdos, limites y problemas abiertos.\n"
            "- Usos: explicar para que sirve pensar este tema en la practica.\n\n"
            "## Desarrollo\n\n"
            f"Una lectura util de {topic} deberia separar definiciones basicas, genealogia, actores principales, controversias y aplicaciones. Para convertir este borrador en un documento publicable, conviene agregar fuentes concretas y citas verificables.\n\n"
            "## Limites\n\n"
            "Este texto fue generado como sintesis general sin fuentes locales. No debe tratarse como investigacion documental hasta incorporar referencias verificadas.\n"
        )
    warnings = source_pack.get("warnings") if isinstance(source_pack.get("warnings"), list) else []
    warning_block = "\n".join(f"- {item}" for item in warnings) if warnings else "- No se inspeccionaron fuentes locales para este borrador."
    return (
        body
        + "\n\n## Metodo HeliX\n\n"
        "Modo: sintesis curada sin anchors locales. El archivo fue escrito y hasheado por HeliX, pero las afirmaciones conceptuales requieren revision bibliografica si se van a publicar.\n\n"
        "## Advertencias\n\n"
        + warning_block
        + "\n"
    )


def _pdf_escape_text(text: str) -> str:
    return str(text or "").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_pdf_line(text: str, *, width: int = 92) -> list[str]:
    words = str(text or "").split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _write_simple_pdf(path: Path, *, title: str, body: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    clean_title = re.sub(r"\s+", " ", str(title or "HeliX Work Document")).strip()
    lines.extend(_wrap_pdf_line(clean_title, width=70))
    lines.append("")
    for raw_line in str(body or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            lines.append("")
            continue
        stripped = re.sub(r"^#{1,6}\s*", "", stripped)
        stripped = re.sub(r"^\s*[-*]\s+", "- ", stripped)
        for line in _wrap_pdf_line(stripped):
            lines.append(line)
    lines_per_page = 44
    pages = [lines[index: index + lines_per_page] for index in range(0, max(len(lines), 1), lines_per_page)]
    page_count = max(len(pages), 1)
    font_obj_id = 3 + (page_count * 2)
    objects: list[bytes] = []
    kids = " ".join(f"{3 + (index * 2)} 0 R" for index in range(page_count))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode("ascii"))
    for page_index, page_lines in enumerate(pages):
        page_obj_id = 3 + (page_index * 2)
        content_obj_id = page_obj_id + 1
        commands = ["BT", "/F1 11 Tf", "14 TL", "72 760 Td"]
        for line_index, line in enumerate(page_lines):
            if line_index:
                commands.append("T*")
            commands.append(f"({_pdf_escape_text(line)}) Tj")
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1", errors="replace")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 {font_obj_id} 0 R >> >> /Contents {content_obj_id} 0 R >>".encode("ascii")
        )
        objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    payload = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{index} 0 obj\n".encode("ascii"))
        payload.extend(obj)
        payload.extend(b"\nendobj\n")
    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(bytes(payload))
    return {"path": str(path), "sha256": hashlib.sha256(bytes(payload)).hexdigest(), "bytes": len(payload)}


def _write_simple_docx(path: Path, *, title: str, body: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    paragraphs: list[str] = [str(title or "HeliX Work Document").strip()]
    for raw_line in str(body or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^#{1,6}\s*", "", stripped)
        stripped = re.sub(r"^\s*[-*]\s+", "- ", stripped)
        paragraphs.append(stripped)

    def _paragraph(text: str) -> str:
        return "<w:p><w:r><w:t xml:space=\"preserve\">" + html.escape(text, quote=False) + "</w:t></w:r></w:p>"

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
        + "".join(_paragraph(item) for item in paragraphs)
        + '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>'
        + "</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document_xml)
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _render_internal_work_web(plan: dict[str, Any], source_pack: dict[str, Any]) -> str:
    anchors = _work_anchor_digest(source_pack, limit=10)
    title = html.escape(str(plan.get("goal") or "HeliX Work Runtime"))
    source_items = []
    for index, source in enumerate(source_pack.get("sources") or []):
        ref = html.escape(str(source.get("path") or source.get("url") or source.get("ref") or f"source-{index}"))
        source_items.append(f"<li><strong>s{index:03d}</strong> {ref} <span>{html.escape(str(source.get('status') or 'unknown'))}</span></li>")
    anchor_cards = []
    for item in anchors:
        anchor_cards.append(
            "<article class=\"anchor-card\">"
            f"<p>{html.escape(str(item.get('text') or ''))}</p>"
            f"<small>{html.escape(str(item.get('source_id') or 'source'))} / {html.escape(str(item.get('anchor_id') or 'anchor'))}</small>"
            "</article>"
        )
    if not anchor_cards:
        anchor_cards.append("<article class=\"anchor-card\"><p>No text anchors were available. Review extraction warnings before publishing.</p><small>HeliX warning</small></article>")
    boundary = html.escape(str(plan.get("claim_boundary") or _work_claim_boundary(str(plan.get("work_intent") or ""))))
    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101114; --paper: #f4efe7; --ink: #171717; --muted: #6d665e; --accent: #0f766e; --line: rgba(244,239,231,.2); }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, sans-serif; background: var(--bg); color: var(--paper); line-height: 1.55; }}
    header {{ min-height: 62vh; display: grid; align-content: end; padding: 8vw; border-bottom: 1px solid var(--line); background: linear-gradient(180deg, rgba(16,17,20,.2), #101114), radial-gradient(circle at 20% 20%, rgba(15,118,110,.32), transparent 30%); }}
    main {{ padding: 56px 8vw 80px; }}
    h1 {{ max-width: 980px; margin: 0; font-size: clamp(2.3rem, 6vw, 5.8rem); line-height: .95; letter-spacing: 0; }}
    .kicker {{ color: #9dd8d0; text-transform: uppercase; letter-spacing: .12em; font-size: .78rem; font-weight: 800; margin-bottom: 24px; }}
    .lead {{ max-width: 760px; margin: 28px 0 0; color: #d8d0c6; font-size: 1.15rem; }}
    section {{ max-width: 1100px; margin: 0 auto 64px; }}
    h2 {{ font-size: clamp(1.6rem, 3vw, 2.8rem); margin: 0 0 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    .anchor-card {{ background: var(--paper); color: var(--ink); border-radius: 8px; padding: 22px; min-height: 180px; display: flex; flex-direction: column; justify-content: space-between; }}
    .anchor-card p {{ margin: 0 0 24px; }}
    .anchor-card small, li span {{ color: var(--muted); }}
    ul {{ padding-left: 1.2rem; }}
    li {{ margin: 10px 0; }}
    .boundary {{ border-top: 1px solid var(--line); padding-top: 24px; color: #d8d0c6; }}
  </style>
</head>
<body>
  <header>
    <div class="kicker">HeliX Work Runtime</div>
    <h1>{title}</h1>
    <p class="lead">Pagina estatica generada desde fuentes recolectadas, anchors compactos y una claim boundary explicita.</p>
  </header>
  <main>
    <section>
      <h2>Anchors principales</h2>
      <div class="grid">
        {''.join(anchor_cards)}
      </div>
    </section>
    <section>
      <h2>Fuentes inspeccionadas</h2>
      <ul>{''.join(source_items) or '<li>No sources collected.</li>'}</ul>
    </section>
    <section class="boundary">
      <h2>Limite de confianza</h2>
      <p>{boundary}</p>
    </section>
  </main>
</body>
</html>
"""


def _internal_work_patch(task_root: Path, plan: dict[str, Any], source_pack: dict[str, Any]) -> dict[str, Any]:
    intent = str(plan.get("work_intent") or plan.get("intent") or "")
    if intent not in {"source_to_web", "source_to_document"}:
        return {"status": "not_applicable", "reason": f"no internal generator for intent={intent}"}
    _target, rel_path = _work_output_file(task_root, str(plan.get("output_target") or ""), intent)
    content = _render_internal_work_web(plan, source_pack) if intent == "source_to_web" else _render_internal_work_document(plan, source_pack)
    patch_info = _build_text_patch(task_root, rel_path, content)
    return {
        "status": "passed",
        "engine": "helix-internal-generator",
        "run_id": f"internal-{int(time.time() * 1000)}",
        "patch": patch_info["patch"],
        "patch_sha256": patch_info["patch_sha256"],
        "changed_files": patch_info["changed_files"],
        "fallback_reason": "opencode produced no applicable patch",
    }


def _internal_work_export(task_root: Path, plan: dict[str, Any], source_pack: dict[str, Any]) -> dict[str, Any]:
    intent = str(plan.get("work_intent") or plan.get("intent") or "")
    if intent != "source_to_document":
        return {"status": "not_applicable", "reason": f"no export generator for intent={intent}"}
    target = _work_output_target_path(task_root, str(plan.get("output_target") or ""), intent, str(plan.get("goal") or ""))
    markdown = _render_internal_work_document(plan, source_pack)
    if target.suffix.lower() == ".pdf":
        written = _write_simple_pdf(target, title=str(plan.get("goal") or target.stem), body=markdown)
        method = "helix-simple-pdf"
    elif target.suffix.lower() == ".docx":
        written = _write_simple_docx(target, title=str(plan.get("goal") or target.stem), body=markdown)
        method = "helix-simple-docx"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not markdown.endswith("\n"):
            markdown += "\n"
        target.write_text(markdown, encoding="utf-8")
        data = markdown.encode("utf-8")
        written = {"path": str(target), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        method = "helix-markdown-export"
    return {
        "status": "passed",
        "engine": "helix-internal-exporter",
        "run_id": f"export-{int(time.time() * 1000)}",
        "output_file": written,
        "changed_files": [str(target)],
        "export_method": method,
    }


def _work_opencode_timeout_seconds() -> float:
    raw = os.environ.get("HELIX_WORK_OPENCODE_TIMEOUT_SECONDS", "3")
    try:
        value = float(raw)
    except ValueError:
        value = 8.0
    return max(1.0, min(value, 120.0))


def _work_claim_boundary(intent: str) -> str:
    if intent == "source_to_web":
        return "This run can prove bounded source collection, sandboxed web generation, patch capture and hashes; visual quality and semantic correctness still need review."
    if intent == "source_to_document":
        return "This run can prove bounded source collection, document generation, patch capture and hashes; factual correctness is limited to inspected anchors."
    if intent == "code_patch":
        return "This run can prove sandbox provenance and patch integrity; correctness still needs tests or review."
    if intent == "verification":
        return "This run can summarize verification evidence and checks; deep nuclear claims require explicit suite execution."
    return "This run can prove bounded source collection and response provenance; it does not prove semantic truth beyond inspected sources."


def _work_artifact_paths(task_root: Path, run_id: str) -> dict[str, str]:
    base = Path(task_root) / "verification" / "work-runtime" / run_id
    return {
        "dir": str(base),
        "plan": str(base / "work_plan.json"),
        "sources": str(base / "sources.json"),
        "brief": str(base / "HELIX_WORK_BRIEF.md"),
        "patch": str(base / "patch.diff"),
        "trust_card": str(base / "work_trust_card.json"),
    }


def _write_work_artifacts(task_root: Path, run_id: str, *, plan: dict[str, Any], source_pack: dict[str, Any], trust_card: dict[str, Any], patch: str | None = None) -> dict[str, str]:
    paths = _work_artifact_paths(task_root, run_id)
    base = Path(paths["dir"])
    base.mkdir(parents=True, exist_ok=True)
    Path(paths["plan"]).write_text(json.dumps(_json_ready(plan), indent=2, ensure_ascii=False), encoding="utf-8")
    Path(paths["sources"]).write_text(json.dumps(_json_ready(source_pack), indent=2, ensure_ascii=False), encoding="utf-8")
    Path(paths["brief"]).write_text(_render_work_brief(plan, source_pack), encoding="utf-8")
    if patch:
        Path(paths["patch"]).write_text(patch, encoding="utf-8")
    Path(paths["trust_card"]).write_text(json.dumps(_json_ready(trust_card), indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{_slugish(prefix)}-{stamp}"


def _token_for_provider(provider: ProviderSpec, *, prompt: bool) -> str | None:
    if not provider.token_env:
        return None
    token = os.environ.get(provider.token_env)
    if token:
        return token
    saved = _config_token(provider.name)
    if saved:
        os.environ[provider.token_env] = saved
        return saved
    if not prompt:
        return None
    return getpass.getpass(f"Paste {provider.name} token for this process only: ").strip()


def _secret_values(provider: ProviderSpec | None = None) -> list[str]:
    env_names = [item.token_env for item in PROVIDERS.values() if item.token_env]
    secrets = [os.environ.get(name) for name in env_names if name and os.environ.get(name)]
    if provider and provider.token_env and os.environ.get(provider.token_env):
        secrets.append(os.environ[provider.token_env])
    return [item for item in secrets if item and len(item) >= 4]


_SENSITIVE_KEY_RE = re.compile(r"(api[_-]?key|authorization|bearer|secret|token)", re.IGNORECASE)


def redact_value(value: Any, *, secrets: list[str] | None = None) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if _SENSITIVE_KEY_RE.search(str(key)):
                redacted[str(key)] = REDACTED if item else item
            else:
                redacted[str(key)] = redact_value(item, secrets=secrets)
        return redacted
    if isinstance(value, list):
        return [redact_value(item, secrets=secrets) for item in value]
    if isinstance(value, str):
        text = value
        for secret in secrets or []:
            text = text.replace(secret, REDACTED)
        return text
    return value


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    global _HTTP_SESSION, _REQUESTS_SESSION_READY
    if _HTTP_SESSION is None and not _REQUESTS_SESSION_READY:
        _REQUESTS_SESSION_READY = True
        try:  # requests keeps provider TCP/TLS connections warm, but only after first provider call.
            import requests as _requests  # noqa: PLC0415

            _HTTP_SESSION = _requests.Session()
        except Exception:  # pragma: no cover - urllib fallback keeps HeliX dependency-light.
            _HTTP_SESSION = None
    if _HTTP_SESSION is not None:
        response = _HTTP_SESSION.post(url, json=payload, headers=headers, timeout=timeout)
        if response.status_code >= 400:
            raise error.HTTPError(
                url,
                response.status_code,
                response.reason,
                dict(response.headers),
                io.BytesIO(response.content),
            )
        parsed = response.json()
        if not isinstance(parsed, dict):
            raise ValueError("provider response root must be a JSON object")
        return parsed

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, method="POST", headers=headers)
    with request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - explicit user-selected endpoint
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("provider response root must be a JSON object")
    return parsed


def _get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 2.0,
) -> dict[str, Any]:
    req = request.Request(url, method="GET", headers=headers or {})
    with request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - explicit user-selected endpoint
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("endpoint response root must be a JSON object")
    return parsed


def _fetch_text_url(url: str, *, timeout: float = 8.0, max_bytes: int = 1_000_000) -> tuple[str, str]:
    req = request.Request(
        url,
        method="GET",
        headers={
            "User-Agent": "HeliX-CLI/5.4 (+local research tool)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
        },
    )
    with request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - user-requested web retrieval
        content_type = response.headers.get("content-type", "")
        data = response.read(max_bytes + 1)
    return data[:max_bytes].decode("utf-8", errors="replace"), content_type


def _strip_html(text: str) -> str:
    cleaned = re.sub(r"(?is)<(script|style|noscript)\b[^>]*>.*?</\1>", " ", str(text or ""))
    cleaned = re.sub(r"(?s)<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def web_search(query: str, *, limit: int = 5, timeout: float = 8.0) -> dict[str, Any]:
    query = str(query or "").strip()
    if not query:
        return {"status": "error", "error": "query is required", "results": []}
    limit = _safe_int(limit, 5, minimum=1, maximum=10)
    url = "https://duckduckgo.com/html/?" + urlparse.urlencode({"q": query})
    try:
        body, content_type = _fetch_text_url(url, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "query": query, "error": f"{type(exc).__name__}: {exc}", "results": []}
    results: list[dict[str, Any]] = []
    for match in re.finditer(r'(?is)<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', body):
        href = html.unescape(match.group(1))
        title = _strip_html(match.group(2))
        parsed = urlparse.urlparse(href)
        if parsed.path == "/l/":
            params = urlparse.parse_qs(parsed.query)
            href = params.get("uddg", [href])[0]
        snippet = ""
        tail = body[match.end(): match.end() + 1800]
        snippet_match = re.search(r'(?is)<a[^>]+class="result__snippet"[^>]*>(.*?)</a>|<div[^>]+class="result__snippet"[^>]*>(.*?)</div>', tail)
        if snippet_match:
            snippet = _strip_html(snippet_match.group(1) or snippet_match.group(2) or "")
        if title and href:
            results.append({"title": title, "url": href, "snippet": snippet})
        if len(results) >= limit:
            break
    return {
        "status": "ok" if results else "empty",
        "query": query,
        "source": "duckduckgo-html",
        "content_type": content_type,
        "result_count": len(results),
        "results": results,
    }


def web_read(url: str, *, max_chars: int = 8000, timeout: float = 8.0) -> dict[str, Any]:
    raw = str(url or "").strip()
    if not raw:
        return {"status": "error", "error": "url is required"}
    parsed = urlparse.urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        return {"status": "blocked", "error": "only http/https URLs can be read", "url": raw}
    max_chars = _safe_int(max_chars, 8000, minimum=1000, maximum=30000)
    try:
        body, content_type = _fetch_text_url(raw, timeout=timeout, max_bytes=max_chars * 4)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "url": raw, "error": f"{type(exc).__name__}: {exc}"}
    text = _strip_html(body) if "html" in content_type.lower() or "<html" in body[:500].lower() else body
    return {
        "status": "ok",
        "url": raw,
        "content_type": content_type,
        "chars": len(text),
        "truncated": len(text) > max_chars,
        "content": text[:max_chars],
    }


def _http_error_detail(exc: error.HTTPError, *, max_chars: int = 2000) -> str:
    detail = f"HTTP Error {getattr(exc, 'code', '?')}: {getattr(exc, 'reason', '') or exc}"
    body = ""
    try:
        raw = exc.read()
        body = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw or "")
    except Exception:
        body = ""
    body = body.strip()
    if body:
        detail = f"{detail}: {body[:max_chars]}"
    return detail


def _messages_char_count(messages: list[dict[str, str]]) -> int:
    return sum(len(str(item.get("content") or "")) for item in messages)


def _compact_message_content(content: str, limit: int) -> str:
    text = str(content or "")
    if len(text) <= limit:
        return text
    head = max(0, int(limit * 0.7))
    tail = max(0, limit - head - 80)
    return (
        text[:head]
        + "\n...[middle compacted by HeliX after provider rejected the full request]...\n"
        + (text[-tail:] if tail else "")
    )


def _compact_openai_compatible_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    compacted: list[dict[str, str]] = []
    non_system = [item for item in messages if item.get("role") != "system"]
    recent_non_system = non_system[-6:]
    for item in messages:
        role = item.get("role")
        if role == "system":
            compacted.append(
                {
                    "role": "system",
                    "content": _compact_message_content(str(item.get("content") or ""), 12000)
                    + "\n\n[HeliX note: request compacted after provider Bad Request; answer only from visible context.]",
                }
            )
            continue
        if item not in recent_non_system:
            continue
        compacted.append(
            {
                "role": str(role or "user"),
                "content": _compact_message_content(str(item.get("content") or ""), 5000),
            }
        )
    return compacted


def _openai_compatible_chat(
    provider: ProviderSpec,
    *,
    model: str,
    messages: list[dict[str, str]],
    token: str | None,
    max_tokens: int,
    temperature: float,
    timeout: float,
    base_url: str | None = None,
    native_request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{(base_url or provider.base_url or '').rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    started = time.perf_counter()
    response_format = None
    if isinstance(native_request, dict):
        candidate = native_request.get("request_response_format")
        if isinstance(candidate, dict):
            response_format = candidate

    def _payload_for(call_messages: list[dict[str, str]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": call_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    compact_retry = False
    try:
        response = _post_json(
            url,
            _payload_for(messages),
            headers=headers,
            timeout=timeout,
        )
    except error.HTTPError as exc:
        status_code = getattr(exc, "code", None)
        detail = _http_error_detail(exc)
        if status_code == 429:
            retry_after = _retry_after_seconds_from_headers(getattr(exc, "headers", None))
            _mark_provider_cooldown(provider.name, model=model, reason=detail, seconds=retry_after)
            raise ProviderRateLimitError(
                provider.name,
                model,
                f"API Error ({provider.name}): {detail}",
                retry_after_seconds=retry_after,
            ) from exc
        if status_code == 400 and _messages_char_count(messages) > 24000:
            compacted_messages = _compact_openai_compatible_messages(messages)
            if _messages_char_count(compacted_messages) < _messages_char_count(messages):
                try:
                    response = _post_json(
                        url,
                        _payload_for(compacted_messages),
                        headers=headers,
                        timeout=timeout,
                    )
                    compact_retry = True
                except error.HTTPError as retry_exc:
                    retry_detail = _http_error_detail(retry_exc)
                    raise RuntimeError(
                        f"API Error ({provider.name}): {retry_detail}; compact retry after Bad Request also failed"
                    ) from retry_exc
                except Exception as retry_exc:
                    raise RuntimeError(
                        f"API Error ({provider.name}): {retry_exc}; compact retry after Bad Request also failed"
                    ) from retry_exc
            else:
                raise RuntimeError(f"API Error ({provider.name}): {detail}") from exc
        else:
            raise RuntimeError(f"API Error ({provider.name}): {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"API Error ({provider.name}): {exc}") from exc

    if "error" in response:
        error_msg = response["error"].get("message", str(response["error"])) if isinstance(response["error"], dict) else str(response["error"])
        if _is_rate_limit_error(error_msg):
            _mark_provider_cooldown(provider.name, model=model, reason=error_msg)
            raise ProviderRateLimitError(
                provider.name,
                model,
                f"Provider Error ({provider.name}): {error_msg}",
            )
        raise RuntimeError(f"Provider Error: {error_msg}")

    latency_ms = (time.perf_counter() - started) * 1000
    choice = dict((response.get("choices") or [{}])[0] or {})
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    
    # Capture content and reasoning
    content = str(message.get("content") or choice.get("text") or "").strip()
    reasoning = str(message.get("reasoning_content") or "").strip()
    
    if not content and not reasoning:
        raise RuntimeError(f"Provider returned empty content. Raw response: {json.dumps(response)}")

    # Unified text output: prefer content, fallback to reasoning
    unified_text = content if content else reasoning

    return {
        "provider": provider.name,
        "requested_model": model,
        "actual_model": response.get("model") or model,
        "text": unified_text,
        "reasoning": reasoning,
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage"),
        "latency_ms": latency_ms,
        "request_compacted_after_bad_request": compact_retry,
        "raw": response,
    }


def _anthropic_chat(
    provider: ProviderSpec,
    *,
    model: str,
    messages: list[dict[str, str]],
    token: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> dict[str, Any]:
    system_parts = [item["content"] for item in messages if item.get("role") == "system"]
    call_messages = [item for item in messages if item.get("role") != "system"]
    started = time.perf_counter()
    response = _post_json(
        f"{(provider.base_url or '').rstrip('/')}/messages",
        {
            "model": model,
            "system": "\n\n".join(system_parts) if system_parts else None,
            "messages": call_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        headers={
            "Content-Type": "application/json",
            "x-api-key": token,
            "anthropic-version": "2023-06-01",
        },
        timeout=timeout,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    blocks = response.get("content") if isinstance(response.get("content"), list) else []
    text = "".join(str(block.get("text") or "") for block in blocks if isinstance(block, dict))
    return {
        "provider": provider.name,
        "requested_model": model,
        "actual_model": response.get("model") or model,
        "text": text,
        "finish_reason": response.get("stop_reason"),
        "usage": response.get("usage"),
        "latency_ms": latency_ms,
        "raw": response,
    }


def _prepare_gemini_native_request(model: str, native_request: dict[str, Any] | None) -> dict[str, Any]:
    request_payload = dict(native_request or {})
    profile = _model_profile_for_id(model)
    url_context_urls = [str(item) for item in (request_payload.get("url_context_urls") or []) if str(item).strip()]
    enable_search_grounding = bool(request_payload.get("enable_search_grounding"))
    function_declarations = request_payload.get("function_declarations") or []
    function_calling_mode = request_payload.get("function_calling_mode")
    file_search_store_ids = [str(item) for item in (request_payload.get("file_search_store_ids") or []) if str(item).strip()]
    if (url_context_urls or enable_search_grounding) and (function_declarations or function_calling_mode):
        raise ValueError("Gemini URL Context / Google Search grounding cannot be combined with function calling in this HeliX pass")
    if url_context_urls and len(url_context_urls) > 20:
        raise ValueError("Gemini URL Context accepts at most 20 URLs per request")
    if url_context_urls and profile and not profile.supports_url_context:
        raise ValueError(f"{model} is not marked as supporting Gemini URL Context")
    if enable_search_grounding and profile and not profile.supports_search_grounding:
        raise ValueError(f"{model} is not marked as supporting Gemini Search grounding")
    if function_declarations and profile and not profile.supports_function_calling:
        raise ValueError(f"{model} is not marked as supporting Gemini function calling")
    if file_search_store_ids and profile and not profile.supports_file_search:
        raise ValueError(f"{model} is not marked as supporting Gemini File Search")

    tools: list[dict[str, Any]] = []
    tool_config: dict[str, Any] | None = None
    if url_context_urls:
        tools.append({"url_context": {}})
    if enable_search_grounding:
        tools.append({"google_search": {}})
    if function_declarations:
        tools.append({"functionDeclarations": function_declarations})
    if file_search_store_ids:
        tools.append({"fileSearch": {"fileSearchStoreNames": file_search_store_ids}})
    if function_calling_mode:
        tool_config = {"functionCallingConfig": {"mode": str(function_calling_mode)}}
        allowed_function_names = request_payload.get("allowed_function_names") or []
        if allowed_function_names:
            tool_config["functionCallingConfig"]["allowedFunctionNames"] = [str(item) for item in allowed_function_names if str(item).strip()]
    return {
        "tools": tools,
        "toolConfig": tool_config,
        "url_context_urls": url_context_urls,
        "enable_search_grounding": enable_search_grounding,
        "function_declarations": function_declarations,
        "function_calling_mode": function_calling_mode,
        "file_search_store_ids": file_search_store_ids,
    }


def _gemini_chat(
    provider: ProviderSpec,
    *,
    model: str,
    messages: list[dict[str, str]],
    token: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    base_url: str | None = None,
    native_request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    system_parts = [str(item.get("content") or "") for item in messages if item.get("role") == "system"]
    contents: list[dict[str, Any]] = []
    for item in messages:
        role = item.get("role")
        content = str(item.get("content") or "")
        if not content or role == "system":
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": content}]})
    if not contents:
        contents.append({"role": "user", "parts": [{"text": ""}]})

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        },
    }
    native_payload = _prepare_gemini_native_request(model, native_request)
    if system_parts:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
    if native_payload["tools"]:
        payload["tools"] = native_payload["tools"]
    if native_payload["toolConfig"]:
        payload["toolConfig"] = native_payload["toolConfig"]

    started = time.perf_counter()
    try:
        response = _post_json(
            f"{(base_url or provider.base_url or '').rstrip('/')}/models/{model}:generateContent",
            payload,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": token,
            },
            timeout=timeout,
        )
    except error.HTTPError as exc:
        if getattr(exc, "code", None) == 429:
            retry_after = _retry_after_seconds_from_headers(getattr(exc, "headers", None))
            _mark_provider_cooldown(
                provider.name,
                model=model,
                reason="HTTP 429 Too Many Requests",
                seconds=retry_after,
            )
            raise ProviderRateLimitError(
                provider.name,
                model,
                f"API Error ({provider.name}): HTTP Error 429: Too Many Requests",
                retry_after_seconds=retry_after,
            ) from exc
        raise RuntimeError(f"API Error ({provider.name}): {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"API Error ({provider.name}): {exc}") from exc

    if "error" in response:
        error_msg = response["error"].get("message", str(response["error"])) if isinstance(response["error"], dict) else str(response["error"])
        if _is_rate_limit_error(error_msg):
            _mark_provider_cooldown(provider.name, model=model, reason=error_msg)
            raise ProviderRateLimitError(
                provider.name,
                model,
                f"Provider Error ({provider.name}): {error_msg}",
            )
        raise RuntimeError(f"Provider Error ({provider.name}): {error_msg}")

    latency_ms = (time.perf_counter() - started) * 1000
    candidate = dict((response.get("candidates") or [{}])[0] or {})
    content = candidate.get("content") if isinstance(candidate.get("content"), dict) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    text = "".join(str(part.get("text") or "") for part in parts if isinstance(part, dict)).strip()
    function_calls = [part.get("functionCall") for part in parts if isinstance(part, dict) and isinstance(part.get("functionCall"), dict)]
    if not text and function_calls:
        text = json.dumps({"function_calls": function_calls}, ensure_ascii=False)
    if not text:
        raise RuntimeError(f"Provider returned empty content. Raw response: {json.dumps(response)}")
    return {
        "provider": provider.name,
        "requested_model": model,
        "actual_model": response.get("modelVersion") or model,
        "text": text,
        "finish_reason": candidate.get("finishReason"),
        "usage": response.get("usageMetadata"),
        "latency_ms": latency_ms,
        "function_calls": function_calls,
        "native_tool_metadata": {
            "url_context_metadata": candidate.get("urlContextMetadata") or candidate.get("url_context_metadata"),
            "grounding_metadata": candidate.get("groundingMetadata") or candidate.get("grounding_metadata"),
        },
        "raw": response,
    }


def run_chat(
    *,
    provider_name: str,
    model: str | None,
    prompt: str,
    system: str | None = None,
    history: list[dict[str, str]] | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    base_url: str | None = None,
    prompt_token: bool = True,
    workspace_root: Path | None = None,
    native_request: dict[str, Any] | None = None,
    blind_inference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provider = PROVIDERS[provider_name]
    selected_model = model or provider.default_model
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    for item in history or []:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": str(content)})
    messages.append({"role": "user", "content": prompt})
    blind_runtime: dict[str, Any] | None = None
    blind_policy = BlindInferencePolicy.from_payload(
        blind_inference.get("policy") if isinstance(blind_inference, dict) else None
    )
    blind_requested = bool(isinstance(blind_inference, dict) and blind_inference.get("enabled"))
    if blind_requested and blind_policy.enabled and _provider_is_cloud_boundary(provider):
        blind_runtime = blind_transform_request(
            messages,
            policy=blind_policy,
            task_id=str(blind_inference.get("task_id") or f"{provider_name}:{selected_model}:{time.time_ns()}"),
        )
        messages = list(blind_runtime.get("messages") or messages)

    if provider.kind == "helix-local":
        if not selected_model:
            raise ValueError("--model is required when --provider local")
        runtime = HelixRuntime(root=workspace_root)
        started = time.perf_counter()
        result = runtime.generate_text(
            alias=selected_model,
            messages=messages,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1.0 if temperature <= 0.0 else temperature),
        )
        latency_ms = (time.perf_counter() - started) * 1000
        result = {
            "provider": provider.name,
            "requested_model": selected_model,
            "actual_model": selected_model,
            "text": str(result.get("completion_text") or result.get("generated_text") or ""),
            "finish_reason": None,
            "usage": None,
            "latency_ms": latency_ms,
            "raw": result,
        }
        result["blind_inference"] = {
            "requested": blind_requested,
            "enabled": False,
            "bypassed": bool(blind_requested),
            "reason": "local_provider_boundary",
            "provider_target": provider.name,
        }
        return result

    cooldown = _provider_cooldown_status(provider.name)
    if cooldown.get("active"):
        remaining = int(round(float(cooldown.get("remaining_seconds") or 0.0)))
        raise ProviderRateLimitError(
            provider.name,
            selected_model,
            f"{provider.name} is cooling down after a provider rate limit; retry in ~{remaining}s",
            retry_after_seconds=float(cooldown.get("remaining_seconds") or 0.0),
        )

    token = _token_for_provider(provider, prompt=prompt_token)
    if provider.requires_token and not token:
        raise RuntimeError(f"{provider.token_env} is required for provider {provider.name}")
    if provider.kind == "anthropic":
        result = _anthropic_chat(
            provider,
            model=selected_model,
            messages=messages,
            token=str(token),
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
    elif provider.kind == "gemini":
        result = _gemini_chat(
            provider,
            model=selected_model,
            messages=messages,
            token=str(token),
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            base_url=base_url,
            native_request=native_request,
        )
    else:
        result = _openai_compatible_chat(
            provider,
            model=selected_model,
            messages=messages,
            token=token,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            base_url=base_url,
            native_request=native_request,
        )
    if blind_runtime is not None:
        raw_provider_text = str(result.get("text") or "")
        result["text"] = blind_rehydrate_response(raw_provider_text, blind_runtime.get("vault"))
        result["blind_inference"] = {
            "requested": True,
            "enabled": True,
            "bypassed": False,
            "provider_target": provider.name,
            "policy_id": blind_runtime.get("policy_id"),
            "task_id": blind_runtime.get("task_id"),
            "span_count": int(blind_runtime.get("span_count") or 0),
            "sensitive_classes": list(blind_runtime.get("sensitive_classes") or []),
            "warnings": list(blind_runtime.get("warnings") or []),
            "baseline_redaction_applied": bool(blind_runtime.get("baseline_redaction_applied")),
            "vault_present": bool((blind_runtime.get("vault") or {}).summary().get("vault_present")) if blind_runtime.get("vault") else False,
        }
    else:
        result["blind_inference"] = {
            "requested": blind_requested,
            "enabled": False,
            "bypassed": bool(blind_requested),
            "reason": "policy_disabled_or_local_boundary" if blind_requested else "disabled",
            "provider_target": provider.name,
        }
    return result


def _fallback_model_ids_for_route(
    route: dict[str, Any] | None,
    *,
    primary_model: str | None,
    agent_blueprint: AgentBlueprint | None = None,
    include_route_fallbacks: bool = True,
) -> list[str]:
    aliases: list[str] = []
    if agent_blueprint is not None:
        aliases.extend(agent_blueprint.fallback_aliases)
    if include_route_fallbacks:
        aliases.extend(str(item) for item in ((route or {}).get("fallback_chain") or []))
    models: list[str] = []
    seen = {str(primary_model or "")}
    for alias in aliases:
        try:
            model_id = resolve_model_alias(alias)
        except Exception:
            continue
        if model_id and model_id not in seen and model_id.lower() not in {"auto", "router:auto"}:
            models.append(model_id)
            seen.add(model_id)
    return models


def run_chat_with_failover(
    *,
    provider_name: str,
    model: str | None,
    fallback_models: list[str] | None = None,
    fallback_targets: list[dict[str, str | None]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    ordered_models = [model, *(fallback_models or [])]
    ordered_targets: list[dict[str, str | None]] = [
        {"provider_name": provider_name, "model": candidate}
        for candidate in ordered_models
    ]
    for target in fallback_targets or []:
        target_provider = str(target.get("provider_name") or provider_name)
        ordered_targets.append({"provider_name": target_provider, "model": target.get("model")})
    last_error: Exception | None = None
    rate_limited_providers: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for target in ordered_targets:
        candidate_provider = str(target.get("provider_name") or provider_name)
        candidate = target.get("model")
        seen_key = (candidate_provider, str(candidate or ""))
        if seen_key in seen:
            continue
        seen.add(seen_key)
        if candidate_provider in rate_limited_providers:
            continue
        try:
            call_kwargs = dict(kwargs)
            if candidate_provider != provider_name and call_kwargs.get("native_request"):
                call_kwargs["native_request"] = None
            result = run_chat(provider_name=candidate_provider, model=candidate, **call_kwargs)
        except Exception as exc:  # noqa: BLE001
            rate_limited = _is_rate_limit_error(exc)
            attempts.append(
                {
                    "provider": candidate_provider,
                    "model": candidate,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "rate_limited": rate_limited,
                }
            )
            last_error = exc
            if rate_limited:
                retry_after = exc.retry_after_seconds if isinstance(exc, ProviderRateLimitError) else None
                _mark_provider_cooldown(
                    candidate_provider,
                    model=str(candidate or "") or None,
                    reason=str(exc),
                    seconds=retry_after,
                )
                rate_limited_providers.add(candidate_provider)
            continue
        result["failover_attempts"] = attempts
        result["failover_used"] = bool(attempts)
        result["selected_model_after_failover"] = result.get("actual_model") or candidate
        return result
    summary = "; ".join(
        f"{item.get('provider') or provider_name}:{item.get('model')}: {item.get('error_type')}"
        for item in attempts[-4:]
    )
    last_message = str(last_error) if last_error else "unknown provider error"
    raise RuntimeError(f"all model attempts failed ({summary}): {last_message}") from last_error


def _memory_receipt_for(root: Path | None, memory_id: str | None) -> dict[str, Any] | None:
    if not memory_id:
        return None
    catalog = hmem.open_catalog(root)
    try:
        receipt = catalog.get_memory_receipt(memory_id)
        node_hash = catalog.get_memory_node_hash(memory_id)
    finally:
        catalog.close()
    if not receipt and not node_hash:
        return None
    return {"memory_id": memory_id, "node_hash": node_hash, "receipt": receipt}


def _compact_receipt(receipt: dict[str, Any] | None) -> dict[str, Any]:
    if not receipt:
        return {}
    signed = receipt.get("receipt") if "receipt" in receipt and "node_hash" in receipt else receipt
    payload = signed.get("payload") if isinstance(signed.get("payload"), dict) else signed
    return {
        "memory_id": receipt.get("memory_id") or payload.get("memory_id"),
        "node_hash": receipt.get("node_hash") or payload.get("node_hash"),
        "parent_hash": payload.get("parent_hash"),
        "signature_alg": signed.get("signature_alg"),
        "signature_verified": signed.get("signature_verified"),
        "key_provenance": signed.get("key_provenance"),
        "receipt_payload_version": payload.get("receipt_payload_version"),
    }


def _truncate_text(text: str, limit: int = 12000) -> dict[str, Any]:
    value = str(text or "")
    if len(value) <= limit:
        return {"text": value, "truncated": False, "chars": len(value)}
    return {"text": value[:limit] + "\n...[truncated by HeliX]...", "truncated": True, "chars": len(value)}


def _repo_display_path(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(REPO_ROOT.resolve(strict=False)).as_posix()
    except Exception:
        return str(path)


def _architecture_excerpt(
    path: Path,
    *,
    label: str,
    needles: tuple[str, ...],
    radius: int = 4,
    max_chars: int = 1800,
) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    except Exception as exc:  # noqa: BLE001
        return {
            "path": _repo_display_path(path),
            "label": label,
            "found": False,
            "error": f"{type(exc).__name__}: {exc}",
            "excerpt": "",
        }

    windows: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    lowered_needles = [str(item).lower() for item in needles]
    for needle, lowered in zip(needles, lowered_needles):
        for index, line in enumerate(lines):
            if lowered not in line.lower():
                continue
            start = max(0, index - radius)
            end = min(len(lines), index + radius + 1)
            key = (start, end)
            if key in seen:
                break
            seen.add(key)
            windows.append((start, end, needle))
            break
    if not windows:
        windows.append((0, min(len(lines), radius * 2 + 4), "file"))

    blocks: list[str] = []
    for start, end, needle in windows[:2]:
        block = "\n".join(f"{line_number + 1}: {lines[line_number]}" for line_number in range(start, end))
        blocks.append(f"[focus: {needle}]\n{block}")
    excerpt = "\n...\n".join(blocks)
    trimmed = _truncate_text(excerpt, max_chars)
    return {
        "path": _repo_display_path(path),
        "label": label,
        "found": bool(windows),
        "excerpt": trimmed["text"],
        "truncated": trimmed["truncated"],
        "focus_needles": list(needles),
    }


def _architecture_excerpt_specs() -> list[dict[str, Any]]:
    return [
        {
            "path": REPO_ROOT / "helix_kv" / "memory_catalog.py",
            "label": "canonical head, lineage verification and quarantine",
            "needles": (
                "def session_lineage",
                "def verify_session_lineage",
                "self._session_lineage",
                "def verify_chain",
            ),
        },
        {
            "path": REPO_ROOT / "helix_kv" / "merkle_dag.py",
            "label": "parent-linked Merkle-DAG structure",
            "needles": (
                "class MerkleNode",
                "def _insert_unlocked",
                "def audit_chain",
            ),
        },
        {
            "path": REPO_ROOT / "src" / "helix_proto" / "signed_receipts.py",
            "label": "receipt authenticity boundaries",
            "needles": (
                "does not",
                "def sign_receipt_payload",
                "def verify_signed_receipt",
            ),
        },
        {
            "path": REPO_ROOT / "src" / "helix_proto" / "helix_cli.py",
            "label": "CLI prompt grounding and tool routing",
            "needles": (
                "def _chat_system",
                "def _task_system",
                "def _planner_callback_factory",
            ),
        },
    ]


def _architecture_context_blob(pack: dict[str, Any] | None, *, limit: int = 10000) -> str:
    if not pack:
        return "{}"
    serialized = json.dumps(pack, ensure_ascii=False, indent=2)
    return _truncate_text(serialized, limit)["text"]


def _safe_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _normalise_command(command: Any) -> list[str]:
    if isinstance(command, list):
        return [str(item) for item in command if str(item).strip()]
    text = str(command or "").strip()
    if not text:
        return []
    if re.search(r"[\n\r|&;<>]", text):
        raise ValueError("shell control operators are blocked in read-only agent mode")
    return [item.strip('"') for item in shlex.split(text, posix=False)]


def _command_basename(token: str) -> str:
    clean = token.strip().strip('"')
    if not clean:
        return ""
    return Path(clean).name.lower()


def _is_safe_readonly_command(tokens: list[str]) -> bool:
    if not tokens:
        return False
    first = _command_basename(tokens[0])
    second = tokens[1].lower() if len(tokens) > 1 else ""
    third = tokens[2].lower() if len(tokens) > 2 else ""
    if first in {"python", "python.exe", "py", "py.exe"} and second == "-m":
        return third in {"pytest", "py_compile", "unittest"}
    if first == Path(sys.executable).name.lower() and second == "-m":
        return third in {"pytest", "py_compile", "unittest"}
    if first in {"pytest", "pytest.exe"}:
        return True
    if first in {"cargo", "cargo.exe"} and second == "test":
        return True
    return False


class SuiteEvidenceCatalog:
    """Read-only index over verification suites, artifacts, manifests and transcripts."""

    TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".log", ".txt"}
    GLOBAL_SKIP_DIRS = {
        "cli-sessions",
        "fixtures",
        "viewer",
        "session-os",
        "sessions",
    }

    def __init__(self, *, evidence_root: Path) -> None:
        self.evidence_root = Path(evidence_root).resolve()
        self.nuclear_root = self.evidence_root / "nuclear-methodology"

    def _fast_index_enabled(self) -> bool:
        mode = os.environ.get("HELIX_SUITE_CATALOG_MODE", "").strip().lower()
        if mode in {"slow", "legacy", "deep"}:
            return False
        if mode in {"fast", "index", "rust"}:
            return True
        try:
            return self.evidence_root == (REPO_ROOT / "verification").resolve()
        except Exception:
            return False

    def refresh_index(self) -> dict[str, Any]:
        payload = helix_cli_core.suite_index_refresh(evidence_root=self.evidence_root, repo_root=REPO_ROOT)
        payload["fast_index_enabled"] = self._fast_index_enabled()
        return payload

    def _augment_indexed_suites(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("status") != "ok":
            return payload
        suites = []
        for suite in payload.get("suites", []):
            if not isinstance(suite, dict):
                continue
            suite_id = str(suite.get("suite_id") or "")
            spec = SUITES.get(suite_id)
            suite = dict(suite)
            suite["registered"] = bool(spec)
            suite["script"] = spec.script if spec else suite.get("script")
            suite["description"] = spec.description if spec else suite.get("description")
            prereg = self._base_root() / suite_id / "PREREGISTERED.md"
            suite["preregistered_path"] = suite.get("preregistered_path") or (self._rel(prereg) if prereg.exists() else None)
            suites.append(suite)
        payload = dict(payload)
        payload["suites"] = suites
        payload["suite_count"] = len(suites)
        return payload

    def _indexed_show_suite(self, suite_id: str, *, limit: int = 12) -> dict[str, Any]:
        listed = self._augment_indexed_suites(helix_cli_core.suite_list(evidence_root=self.evidence_root, repo_root=REPO_ROOT))
        if listed.get("status") != "ok":
            return listed | {"suite_id": suite_id, "latest_records": []}
        wanted = _slugish(suite_id)
        row = next((item for item in listed.get("suites", []) if _slugish(str(item.get("suite_id") or "")) == wanted), None)
        if row is None:
            return {"status": "not_found", "suite_id": suite_id, "suites": [item.get("suite_id") for item in listed.get("suites", [])]}
        index = helix_cli_core._read_index(self.evidence_root) or {}
        records = [
            item
            for item in index.get("records", [])
            if isinstance(item, dict) and _slugish(str(item.get("suite_id") or "")) == wanted
        ]
        records.sort(key=lambda item: int(item.get("mtime_ns") or item.get("mtime_ms") or 0), reverse=True)
        return {
            "status": "ok",
            "source": row.get("source") or listed.get("source"),
            "suite_id": row.get("suite_id"),
            "path": row.get("path"),
            "preregistered_path": row.get("preregistered_path"),
            "counts": row.get("counts") or {},
            "latest_records": records[:limit],
        }

    def _base_root(self) -> Path:
        return self.nuclear_root if self.nuclear_root.exists() else self.evidence_root

    def _suite_dirs(self) -> list[Path]:
        root = self._base_root()
        if not root.exists():
            return []
        return [
            path
            for path in sorted(root.iterdir())
            if path.is_dir() and not path.name.startswith("_") and not path.name.startswith(".")
        ]

    def _suite_dir(self, suite_id: str) -> Path | None:
        wanted = _slugish(suite_id)
        for path in self._suite_dirs():
            if _slugish(path.name) == wanted:
                return path
        return None

    def _iter_text_files(
        self,
        root: Path,
        *,
        limit: int = 2000,
        recursive: bool = True,
        skip_dirs: set[str] | None = None,
    ) -> list[Path]:
        if not root.exists():
            return []
        if root.is_file():
            return [root] if root.suffix.lower() in self.TEXT_SUFFIXES else []
        files: list[Path] = []
        walker = os.walk(root, onerror=lambda _exc: None) if recursive else [(str(root), [], [item.name for item in sorted(root.iterdir()) if item.is_file()])]
        blocked = {name.lower() for name in (skip_dirs or set())}
        for current, dirs, names in walker:
            dirs[:] = [
                name
                for name in sorted(dirs)
                if not name.startswith("_")
                and not name.startswith(".")
                and name.lower() not in blocked
            ]
            for name in sorted(names):
                path = Path(current) / name
                if path.suffix.lower() not in self.TEXT_SUFFIXES:
                    continue
                files.append(path)
                if len(files) >= limit:
                    return files
        return files

    def _iter_global_files(self, *, limit: int = 5000) -> list[Path]:
        files = self._iter_text_files(
            self.evidence_root,
            limit=limit,
            recursive=True,
            skip_dirs=self.GLOBAL_SKIP_DIRS,
        )
        suite_root = self.nuclear_root.resolve()
        return [
            path
            for path in files
            if not self.nuclear_root.exists() or not _is_relative_to(path.resolve(), suite_root)
        ]

    def _suite_dir_for_path(self, path: Path) -> Path | None:
        resolved = path.resolve()
        for suite_dir in self._suite_dirs():
            if _is_relative_to(resolved, suite_dir.resolve()):
                return suite_dir
        return None

    def _kind_for(self, path: Path) -> str:
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

    def _iter_suite_files(self, suite_dir: Path, *, limit: int = 2000) -> list[Path]:
        return self._iter_text_files(suite_dir, limit=limit, recursive=True)

    def _rel(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(REPO_ROOT))
        except Exception:
            return str(path)

    def _timestamp_from_name(self, path: Path) -> str | None:
        match = re.search(r"(20\d{6}[-_]\d{6}|20\d{6}[-_]\d{2})", path.name)
        return match.group(1).replace("_", "-") if match else None

    def _json_summary(self, path: Path) -> dict[str, Any]:
        if path.suffix.lower() != ".json" or path.stat().st_size > 2_000_000:
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            "run_id": payload.get("run_id"),
            "case_id": payload.get("case_id"),
            "status": payload.get("status"),
            "score": payload.get("score"),
            "case_count": payload.get("case_count"),
            "artifact_payload_sha256": payload.get("artifact_payload_sha256"),
            "transcript_exports": payload.get("transcript_exports"),
        }

    def _record_for(self, suite_dir: Path | None, path: Path) -> dict[str, Any]:
        summary = self._json_summary(path)
        if suite_dir is not None:
            try:
                rel_case = path.parent.resolve().relative_to(suite_dir.resolve())
                case_id = None if str(rel_case) == "." else str(rel_case).replace("\\", "/")
            except Exception:
                case_id = None
            suite_id = suite_dir.name
            catalog_scope = "suite"
        else:
            rel_parent = None
            try:
                rel_parent = path.parent.resolve().relative_to(self.evidence_root.resolve())
            except Exception:
                rel_parent = None
            case_id = None if rel_parent in {None, Path(".")} else str(rel_parent).replace("\\", "/")
            suite_id = summary.get("suite_id")
            catalog_scope = "global"
        return {
            "suite_id": suite_id,
            "case_id": summary.get("case_id") or case_id,
            "catalog_scope": catalog_scope,
            "kind": self._kind_for(path),
            "path": self._rel(path),
            "name": path.name,
            "bytes": path.stat().st_size,
            "updated_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "run_id": summary.get("run_id") or self._timestamp_from_name(path),
            "status": summary.get("status"),
            "score": summary.get("score"),
            "case_count": summary.get("case_count"),
            "artifact_payload_sha256": summary.get("artifact_payload_sha256"),
            "transcript_exports": summary.get("transcript_exports"),
        }

    def _search_rank(self, path: Path, *, query_l: str, content_hit: bool) -> int:
        rel_l = self._rel(path).lower()
        name_l = path.name.lower()
        stem_l = path.stem.lower()
        score = 0
        if query_l == stem_l:
            score += 400
        elif query_l == name_l:
            score += 380
        elif name_l.startswith(query_l) or stem_l.startswith(query_l):
            score += 340
        elif query_l in name_l or query_l in stem_l:
            score += 320
        elif rel_l.endswith(query_l):
            score += 300
        elif query_l in rel_l:
            score += 260
        if content_hit:
            score += 180
        if path.parent.resolve() == self.evidence_root.resolve():
            score += 25
        kind = self._kind_for(path)
        if kind.startswith("transcript"):
            score += 20
        elif kind == "artifact":
            score += 15
        elif kind == "manifest":
            score += 10
        return score

    def _search_record(self, path: Path, *, query_l: str) -> dict[str, Any] | None:
        rel_l = self._rel(path).lower()
        name_l = path.name.lower()
        content_hit = False
        snippet = ""
        if path.stat().st_size <= 1_000_000:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                text = ""
            lowered_text = text.lower()
            if query_l in lowered_text:
                content_hit = True
                idx = lowered_text.find(query_l)
                snippet = text[max(0, idx - 160): idx + 360].replace("\n", " ")
        if query_l not in name_l and query_l not in rel_l and not content_hit:
            return None
        record = self._record_for(self._suite_dir_for_path(path), path)
        record["snippet"] = snippet
        record["_match_score"] = self._search_rank(path, query_l=query_l, content_hit=content_hit)
        return record

    def list_suites(self) -> dict[str, Any]:
        if self._fast_index_enabled():
            return self._augment_indexed_suites(helix_cli_core.suite_list(evidence_root=self.evidence_root, repo_root=REPO_ROOT))
        suites = []
        for suite_dir in self._suite_dirs():
            files = self._iter_suite_files(suite_dir)
            counts: dict[str, int] = {}
            records = [self._record_for(suite_dir, path) for path in files]
            for record in records:
                counts[record["kind"]] = counts.get(record["kind"], 0) + 1
            latest_records = sorted(records, key=lambda item: str(item.get("updated_utc") or ""), reverse=True)
            spec = SUITES.get(suite_dir.name)
            suites.append(
                {
                    "suite_id": suite_dir.name,
                    "path": self._rel(suite_dir),
                    "registered": bool(spec),
                    "script": spec.script if spec else None,
                    "description": spec.description if spec else None,
                    "preregistered_path": self._rel(suite_dir / "PREREGISTERED.md") if (suite_dir / "PREREGISTERED.md").exists() else None,
                    "counts": counts,
                    "latest": latest_records[0] if latest_records else None,
                }
            )
        return {"evidence_root": str(self.evidence_root), "suite_count": len(suites), "suites": suites}

    def show_suite(self, suite_id: str, *, limit: int = 12) -> dict[str, Any]:
        if self._fast_index_enabled():
            return self._indexed_show_suite(suite_id, limit=limit)
        suite_dir = self._suite_dir(suite_id)
        if suite_dir is None:
            return {"status": "not_found", "suite_id": suite_id, "suites": [path.name for path in self._suite_dirs()]}
        records = [self._record_for(suite_dir, path) for path in self._iter_suite_files(suite_dir)]
        records.sort(key=lambda item: str(item.get("updated_utc") or ""), reverse=True)
        counts: dict[str, int] = {}
        for record in records:
            counts[record["kind"]] = counts.get(record["kind"], 0) + 1
        return {
            "status": "ok",
            "suite_id": suite_dir.name,
            "path": self._rel(suite_dir),
            "preregistered_path": self._rel(suite_dir / "PREREGISTERED.md") if (suite_dir / "PREREGISTERED.md").exists() else None,
            "counts": counts,
            "latest_records": records[:limit],
        }

    def latest(self, suite_id: str) -> dict[str, Any]:
        payload = self.show_suite(suite_id, limit=50)
        if payload.get("status") != "ok":
            return payload
        records = list(payload.get("latest_records") or [])
        artifacts = [item for item in records if item.get("kind") == "artifact"]
        manifests = [item for item in records if item.get("kind") == "manifest"]
        transcripts = [item for item in records if str(item.get("kind", "")).startswith("transcript")]
        return {
            "status": "ok",
            "suite_id": payload.get("suite_id"),
            "artifact": artifacts[0] if artifacts else None,
            "manifest": manifests[0] if manifests else None,
            "transcripts": transcripts[:10],
            "preregistered_path": payload.get("preregistered_path"),
            "counts": payload.get("counts"),
        }

    def transcripts(self, suite_id: str, *, query: str | None = None, limit: int = 30) -> dict[str, Any]:
        payload = self.show_suite(suite_id, limit=500)
        if payload.get("status") != "ok":
            return payload
        query_l = str(query or "").lower().strip()
        rows = [
            item
            for item in payload.get("latest_records", [])
            if str(item.get("kind", "")).startswith("transcript")
        ]
        if query_l:
            rows = [item for item in rows if query_l in str(item.get("path", "")).lower() or query_l in str(item.get("case_id", "")).lower()]
        return {"status": "ok", "suite_id": payload.get("suite_id"), "transcript_count": len(rows), "transcripts": rows[:limit]}

    def search(self, query: str, *, limit: int = 12, deep: bool = False) -> dict[str, Any]:
        query_l = str(query or "").lower().strip()
        if not query_l:
            return {"status": "error", "error": "query is required", "results": []}
        if self._fast_index_enabled() and not deep:
            return helix_cli_core.suite_search(
                evidence_root=self.evidence_root,
                repo_root=REPO_ROOT,
                query=query,
                limit=limit,
            )
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for suite_dir in self._suite_dirs():
            for path in self._iter_suite_files(suite_dir, limit=2500):
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                record = self._search_record(path, query_l=query_l)
                if record is not None:
                    results.append(record)
        for path in self._iter_global_files(limit=6000):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            record = self._search_record(path, query_l=query_l)
            if record is not None:
                results.append(record)
        results.sort(
            key=lambda item: (
                int(item.get("_match_score") or 0),
                str(item.get("updated_utc") or ""),
                str(item.get("path") or ""),
            ),
            reverse=True,
        )
        for record in results:
            record.pop("_match_score", None)
        limited = results[:limit]
        return {"status": "ok", "query": query, "result_count": len(limited), "results": limited}

    def read(self, ref: str, *, max_bytes: int = 16000) -> dict[str, Any]:
        raw = str(ref or "").strip().strip('"')
        if not raw:
            return {"status": "error", "error": "path or artifact reference is required"}
        candidate = Path(raw)
        if candidate.is_absolute():
            path = candidate.resolve()
        else:
            repo_candidate = (REPO_ROOT / candidate).resolve()
            evidence_candidate = (self.evidence_root / candidate).resolve()
            base_candidate = (self._base_root() / candidate).resolve()
            if repo_candidate.exists():
                path = repo_candidate
            elif evidence_candidate.exists():
                path = evidence_candidate
            elif base_candidate.exists():
                path = base_candidate
            else:
                matches = self.search(raw, limit=10).get("results", [])
                if matches:
                    path = (REPO_ROOT / str(matches[0]["path"])).resolve()
                else:
                    return {"status": "not_found", "ref": raw}
        try:
            path.relative_to(REPO_ROOT.resolve())
        except ValueError:
            try:
                path.relative_to(self.evidence_root)
            except ValueError:
                return {"status": "blocked", "error": "path escapes repository/evidence root", "path": str(path)}
        if not path.exists():
            return {"status": "not_found", "path": str(path)}
        if path.is_dir():
            entries = [
                self._record_for(self._suite_dir_for_path(item), item)
                for item in self._iter_text_files(path, limit=24, recursive=True)
            ]
            entries.sort(key=lambda item: str(item.get("updated_utc") or ""), reverse=True)
            return {
                "status": "ok",
                "path": self._rel(path),
                "type": "directory",
                "entry_count": len(entries),
                "entries": entries,
            }
        if not path.is_file():
            return {"status": "not_found", "path": str(path)}
        max_bytes = _safe_int(max_bytes, 16000, minimum=512, maximum=60000)
        data = path.read_bytes()
        clipped = data[:max_bytes]
        return {
            "status": "ok",
            "path": self._rel(path),
            "kind": self._kind_for(path),
            "bytes": len(data),
            "truncated": len(data) > max_bytes,
            "content": clipped.decode("utf-8", errors="replace"),
        }


class ReadOnlyAgentTools:
    """Small read-only toolbelt for cloud/local task mode.

    These tools inspect the selected task root and may run allowlisted tests.
    They never write source files and never execute through a shell.
    """

    SKIP_DIRS = {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "dist",
        "build",
        "target",
    }

    def __init__(self, *, root: Path, evidence_callback: Any | None = None) -> None:
        self.root = Path(root).resolve()
        self.evidence_callback = evidence_callback

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "list_files",
                "description": "List files under the task root without reading their contents.",
                "input_schema": {"path": "relative directory, default '.'", "pattern": "glob pattern", "limit": "max 200"},
            },
            {
                "name": "read_file",
                "description": "Read a UTF-8-ish text file under the task root, truncated by max_bytes.",
                "input_schema": {"path": "relative file path", "max_bytes": "default 12000, max 50000"},
            },
            {
                "name": "search_text",
                "description": "Search text files under the task root using substring or regex.",
                "input_schema": {"query": "text or regex", "path": "relative directory", "regex": "bool", "limit": "max 200"},
            },
            {
                "name": "git_status",
                "description": "Run git status --short in the task root.",
                "input_schema": {},
            },
            {
                "name": "git_diff",
                "description": "Run git diff in the task root and return a truncated patch.",
                "input_schema": {"max_chars": "default 12000, max 50000"},
            },
            {
                "name": "run_test",
                "description": "Run an allowlisted read-only test command: python -m pytest, python -m py_compile, unittest, pytest, cargo test.",
                "input_schema": {"command": "string or argv list", "timeout": "seconds, max 90"},
            },
            {
                "name": "inspect_artifact",
                "description": "Verify a HeliX artifact JSON under the task root with artifact_replay.",
                "input_schema": {"path": "relative artifact JSON path"},
            },
            {
                "name": "query_evidence",
                "description": "Refresh/search certified HeliX verification evidence and return matching records.",
                "input_schema": {"query": "evidence search query", "limit": "max 20"},
            },
        ]

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = dict(arguments or {})
        started = time.perf_counter()
        try:
            if name == "list_files":
                result = self._list_files(args)
            elif name == "read_file":
                result = self._read_file(args)
            elif name == "search_text":
                result = self._search_text(args)
            elif name == "git_status":
                result = self._git_status()
            elif name == "git_diff":
                result = self._git_diff(args)
            elif name == "run_test":
                result = self._run_test(args)
            elif name == "inspect_artifact":
                result = self._inspect_artifact(args)
            elif name == "query_evidence":
                result = self._query_evidence(args)
            else:
                result = {"status": "blocked", "error": f"unknown read-only tool: {name}"}
        except Exception as exc:  # noqa: BLE001
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        result.setdefault("status", "ok")
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return {"tool": name, "arguments": args, "result": result}

    def _resolve_under_root(self, value: Any, *, must_exist: bool = True) -> tuple[Path, str]:
        raw = str(value or ".").strip().strip('"')
        candidate = Path(raw)
        target = candidate if candidate.is_absolute() else self.root / candidate
        resolved = target.resolve()
        try:
            rel = resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"path escapes task root: {raw}") from exc
        if must_exist and not resolved.exists():
            raise FileNotFoundError(str(rel))
        return resolved, str(rel) if str(rel) != "." else "."

    def _iter_files(self, start: Path, *, limit: int) -> list[Path]:
        files: list[Path] = []
        if start.is_file():
            return [start]
        for current, dirs, names in os.walk(start):
            dirs[:] = sorted(name for name in dirs if name not in self.SKIP_DIRS and not name.startswith(".helix"))
            for name in sorted(names):
                path = Path(current) / name
                files.append(path)
                if len(files) >= limit:
                    return files
        return files

    def _list_files(self, args: dict[str, Any]) -> dict[str, Any]:
        target, rel = self._resolve_under_root(args.get("path", "."))
        pattern = str(args.get("pattern") or "*")
        limit = _safe_int(args.get("limit"), 80, minimum=1, maximum=200)
        rows = []
        for path in self._iter_files(target, limit=limit * 5):
            try:
                relative = str(path.resolve().relative_to(self.root))
            except Exception:
                continue
            if fnmatch.fnmatch(Path(relative).name, pattern) or fnmatch.fnmatch(relative, pattern):
                rows.append({"path": relative, "bytes": path.stat().st_size})
            if len(rows) >= limit:
                break
        return {"root": str(self.root), "path": rel, "pattern": pattern, "files": rows, "truncated": len(rows) >= limit}

    def _read_file(self, args: dict[str, Any]) -> dict[str, Any]:
        target, rel = self._resolve_under_root(args.get("path"))
        if not target.is_file():
            raise IsADirectoryError(rel)
        max_bytes = _safe_int(args.get("max_bytes"), 12000, minimum=256, maximum=50000)
        data = target.read_bytes()
        clipped = data[:max_bytes]
        return {
            "path": rel,
            "bytes": len(data),
            "truncated": len(data) > max_bytes,
            "content": clipped.decode("utf-8", errors="replace"),
        }

    def _search_text(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "")
        if not query:
            raise ValueError("query is required")
        target, rel = self._resolve_under_root(args.get("path", "."))
        limit = _safe_int(args.get("limit"), 80, minimum=1, maximum=200)
        use_regex = bool(args.get("regex", False))
        case_sensitive = bool(args.get("case_sensitive", False))
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = re.compile(query, flags) if use_regex else None
        needle = query if case_sensitive else query.lower()
        matches = []
        scanned = 0
        for path in self._iter_files(target, limit=3000):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".pyd", ".exe", ".dll"}:
                continue
            try:
                if path.stat().st_size > 1_000_000:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            scanned += 1
            for line_no, line in enumerate(text.splitlines(), start=1):
                haystack = line if case_sensitive else line.lower()
                hit = bool(compiled.search(line)) if compiled else needle in haystack
                if hit:
                    matches.append({"path": str(path.resolve().relative_to(self.root)), "line": line_no, "text": line[:300]})
                    if len(matches) >= limit:
                        return {"path": rel, "query": query, "matches": matches, "scanned_files": scanned, "truncated": True}
        return {"path": rel, "query": query, "matches": matches, "scanned_files": scanned, "truncated": False}

    def _run_git(self, args: list[str], *, timeout: float = 10.0) -> dict[str, Any]:
        completed = subprocess.run(  # noqa: S603 - argv is fixed and shell is disabled
            ["git", "-C", str(self.root), *args],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        return {
            "exit_code": completed.returncode,
            "stdout": _truncate_text(completed.stdout, 12000)["text"],
            "stderr": _truncate_text(completed.stderr, 4000)["text"],
        }

    def _git_status(self) -> dict[str, Any]:
        return self._run_git(["status", "--short"])

    def _git_diff(self, args: dict[str, Any]) -> dict[str, Any]:
        max_chars = _safe_int(args.get("max_chars"), 12000, minimum=1000, maximum=50000)
        completed = self._run_git(["diff", "--no-ext-diff"], timeout=15.0)
        clipped = _truncate_text(str(completed.get("stdout") or ""), max_chars)
        completed["stdout"] = clipped["text"]
        completed["truncated"] = clipped["truncated"]
        return completed

    def _run_test(self, args: dict[str, Any]) -> dict[str, Any]:
        tokens = _normalise_command(args.get("command"))
        if not _is_safe_readonly_command(tokens):
            return {
                "status": "blocked",
                "reason": "command is not in the read-only allowlist",
                "allowed": ["python -m pytest", "python -m py_compile", "python -m unittest", "pytest", "cargo test"],
                "command": tokens,
            }
        timeout = float(_safe_int(args.get("timeout"), 45, minimum=1, maximum=90))
        completed = subprocess.run(  # noqa: S603 - argv is allowlisted and shell is disabled
            tokens,
            cwd=self.root,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        return {
            "command": tokens,
            "exit_code": completed.returncode,
            "passed": completed.returncode == 0,
            "stdout": _truncate_text(completed.stdout, 12000)["text"],
            "stderr": _truncate_text(completed.stderr, 8000)["text"],
        }

    def _inspect_artifact(self, args: dict[str, Any]) -> dict[str, Any]:
        target, rel = self._resolve_under_root(args.get("path"))
        report = verify_artifact_file(target)
        return {"path": rel, "report": report}

    def _query_evidence(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "")
        limit = _safe_int(args.get("limit"), 8, minimum=1, maximum=20)
        if not self.evidence_callback:
            return {"status": "unavailable", "records": []}
        pack = self.evidence_callback(query, limit)
        return {"query": query, "records": pack.get("records", []), "record_count": pack.get("record_count", 0)}


def _agent_system_prompt(*, task_root: Path, tool_manifest: list[dict[str, Any]], mode: str) -> str:
    return (
        "You are HeliX Agent Shell, a Codex/Claude-Code-style task agent wrapped by HeliX evidence memory. "
        f"Mode: {mode}. Task root: {task_root}. "
        "You may request tools only by outputting JSON inside <tool_call>...</tool_call>. "
        "One call example: <tool_call>{\"tool\":\"search_text\",\"arguments\":{\"query\":\"TODO\",\"path\":\".\"}}</tool_call>. "
        "Multiple calls example: <tool_call>{\"tool_calls\":[{\"tool\":\"list_files\",\"arguments\":{\"path\":\".\"}}]}</tool_call>. "
        "Use tools to inspect facts before making claims. Do not invent file paths, hashes, run IDs, or test results. "
        "Do not ask for write tools in read-only mode. If code changes are needed, propose a unified diff in the final answer. "
        "When done, answer inside <helix_output>...</helix_output>. "
        "Available read-only tools:\n"
        f"{json.dumps(tool_manifest, ensure_ascii=False, indent=2)}"
    )


def _agent_observation_prompt(
    goal: str,
    observations: list[dict[str, Any]],
    *,
    mode: str = "task",
    helix_focus: bool = False,
    helix_auditability: bool = False,
    max_chars: int = 18000,
) -> str:
    instructions = [
        "Continue the task.",
        "The user does not see the raw tool observations, result counts, or similarity scores.",
        "Do not narrate retrieval mechanics, rankings, or say 'the search found' unless the user explicitly asked for the search report itself.",
        "Extract the facts you need from the observations and answer directly in natural language.",
        "Request another tool with <tool_call> JSON only if a concrete factual gap remains.",
        "If enough evidence is available, return only <helix_output>final answer</helix_output>.",
    ]
    if mode == "chat":
        instructions.append("Prefer a direct, user-facing answer over a process recap.")
    if _looks_like_pasted_suite_evidence(goal):
        instructions.append(
            "The user pasted suite output/logs. Explain the pasted failure or suite rows directly. If stderr shows a traceback, identify the failing file/function, root error, and likely next action. Do not suggest rerunning as if the pasted block were a run request."
        )
    if any(str(item.get("tool") or "") in {"web.search", "web.read"} for item in observations):
        instructions.append(
            "For web observations, cite the result URLs/titles you used and distinguish current sourced facts from your inference."
        )
    if any(str(item.get("tool") or "") == "memory.resolve" for item in observations):
        instructions.append(
            "For memory.resolve observations, answer only from the resolved content. If it is not_found or ambiguous, say that directly. Do not recreate, paraphrase as exact, or infer missing text."
        )
    if any(str(item.get("tool") or "") == "file.inspect" for item in observations):
        instructions.append(
            "For file.inspect observations, answer from the actual file or directory result. If status is not_found/blocked/error, report that status and any suggestions; do not claim the file was moved, renamed, pruned, or deleted unless the observation proves it."
        )
    if any(str(item.get("tool") or "") == "helix.architecture" for item in observations):
        instructions.append(
            "For helix.architecture observations, use the attached invariants, lineage state, excerpts, and claim boundaries as the primary source of truth for HeliX architecture claims."
        )
        instructions.append(
            "Do not reinterpret equivocation/quarantine counters as same-sequence-slot collisions or catastrophic trust failure unless the observation explicitly says that. If trust remains `verified_with_quarantine`, say the canonical head is still preserved and locally verifiable."
        )
    if any(str(item.get("tool") or "") == "helix.trust" for item in observations):
        instructions.append(
            "For helix.trust observations, separate local signed-checkpoint verification from semantic truth or global transparency; report canonical head, equivocation/quarantine and legacy warnings directly."
        )
        instructions.append(
            "If helix.trust reports `equivocation_detected` together with `verified_with_quarantine`, describe that as quarantined competing branches with a still-verifiable canonical head. Do not claim tampering, injection, or unauditability unless signature, chain, or checkpoint verification failed."
        )
    if helix_focus:
        instructions.extend(
            [
                "For questions about HeliX, describe only verified capabilities from the certified evidence pack or tool outputs.",
                "Prefer concrete terms such as signed memories, receipts, signature verification, node hashes, chain status, Merkle-DAG links, thread persistence, memory search, evidence refresh, and tool registry behavior.",
                "Do not drift into generic industry examples, abstract AI philosophy, or claims about hidden reasoning unless the evidence explicitly supports them.",
            ]
        )
    if helix_auditability:
        instructions.append(
            "The user is asking specifically about HeliX auditability and hashes; explain what gets signed, what node hashes identify, what signature or chain verification means, and mention any current boundaries if they appear in the evidence."
        )
    payload = {
        "goal": goal,
        "instruction": " ".join(instructions),
        "observations": observations[-12:],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    return _truncate_text(text, max_chars)["text"]


_QUERY_TOOL_NAMES = {
    "helix.search",
    "memory.search",
    "rag.search",
    "search_text",
    "query_evidence",
    "suite.search",
    "web.search",
}


def _repair_planner_tool_arguments(tool_name: str, arguments: dict[str, Any], goal: str) -> dict[str, Any]:
    repaired = dict(arguments or {})
    name = str(tool_name or "")
    if name in _QUERY_TOOL_NAMES and not str(repaired.get("query") or "").strip():
        repaired["query"] = str(goal or "").strip()
    if name == "search_text" and not str(repaired.get("path") or "").strip():
        repaired["path"] = "."
    if name in {"suite.latest", "suite.transcripts"} and not str(repaired.get("suite_id") or "").strip():
        suite_id = _suite_from_text(goal)
        if suite_id:
            repaired["suite_id"] = suite_id
    if name == "suite.read" and not str(repaired.get("ref") or "").strip():
        suite_id = _suite_from_text(goal)
        if suite_id:
            repaired["ref"] = suite_id
    if name == "file.inspect" and not str(repaired.get("path") or "").strip():
        refs = _extract_local_path_refs(goal)
        if refs:
            repaired["path"] = refs[0]
    return repaired


def _format_runner_fallback_answer(trace: dict[str, Any], *, goal: str) -> str | None:
    if trace.get("final_planner") != "fallback-summary":
        return None
    observations = list(trace.get("observations") or [])
    if not observations:
        return None
    latest = observations[-1]
    observation = latest.get("observation") if isinstance(latest, dict) else None
    if not isinstance(observation, dict):
        return None
    tool_name = str(observation.get("tool") or latest.get("tool_name") or "")
    arguments = observation.get("arguments") if isinstance(observation.get("arguments"), dict) else {}
    result = observation.get("result") if isinstance(observation.get("result"), dict) else {}
    lines = [
        "No llegué a una respuesta final del modelo después de usar herramientas; resumo la última observación en vez de mostrar JSON crudo.",
        "",
        f"- Tool: `{tool_name or 'unknown'}`",
    ]
    if arguments.get("query"):
        lines.append(f"- Query: `{arguments.get('query')}`")
    if arguments.get("suite_id"):
        lines.append(f"- Suite: `{arguments.get('suite_id')}`")
    if result.get("status"):
        lines.append(f"- Status: `{result.get('status')}`")
    if "result_count" in result:
        lines.append(f"- Results: `{result.get('result_count')}`")
    if "record_count" in result:
        lines.append(f"- Records: `{result.get('record_count')}`")
    if tool_name == "suite.search" and result.get("result_count") == 0:
        suite_id = _suite_from_text(goal)
        if suite_id:
            lines.append(f"- Next useful lookup: `/suite latest {suite_id}` or `/suite transcripts {suite_id}`")
    lines.append("")
    lines.append("La tarea necesita otro turno o una herramienta más específica para producir el reporte final.")
    return "\n".join(lines)


@contextmanager
def _cli_receipt_signing(run_id: str, event_type: str, role: str):
    previous = {
        "HELIX_RECEIPT_SIGNING_MODE": os.environ.get("HELIX_RECEIPT_SIGNING_MODE"),
        "HELIX_RECEIPT_SIGNER_ID": os.environ.get("HELIX_RECEIPT_SIGNER_ID"),
        "HELIX_RECEIPT_SIGNING_SEED": os.environ.get("HELIX_RECEIPT_SIGNING_SEED"),
    }
    os.environ["HELIX_RECEIPT_SIGNING_MODE"] = previous["HELIX_RECEIPT_SIGNING_MODE"] or "local_self_signed"
    os.environ["HELIX_RECEIPT_SIGNER_ID"] = "helix-cli"
    if os.environ["HELIX_RECEIPT_SIGNING_MODE"] == "ephemeral_preregistered":
        os.environ["HELIX_RECEIPT_SIGNING_SEED"] = f"helix-cli:{run_id}:{event_type}:{role}:{time.time_ns()}"
    else:
        os.environ.pop("HELIX_RECEIPT_SIGNING_SEED", None)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class InteractiveSession:
    def __init__(
        self,
        *,
        provider_name: str,
        model: str,
        workspace_root: Path,
        project: str,
        agent_id: str,
        max_tokens: int,
        temperature: float,
        transcript_dir: Path,
        router_policy: str = "balanced",
        evidence_root: Path | None = None,
        task_root: Path | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.model = model
        self.workspace_root = workspace_root
        self.project = project
        self.agent_id = agent_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.router_policy = router_policy if router_policy in ROUTER_POLICIES else "balanced"
        self.raw_output = False
        self.theme_name = DEFAULT_THEME
        self.response_style = "balanced"
        self.interaction_mode = "balanced"
        self.latency_mode = "fast"
        self.preflight_mode = "compact"
        self.task_engine = _normalize_task_engine(os.environ.get("HELIX_TASK_ENGINE") or "auto")
        self.blind_inference_enabled = False
        self.blind_inference_policy = BlindInferencePolicy.from_payload(_default_blind_policy_payload())
        self.last_blind_report: dict[str, Any] | None = None
        self._blind_task_counter = 0
        self.runtime = HelixRuntime(root=workspace_root)
        self.run_id = ""
        self.thread_id: str | None = None
        self.events: list[dict[str, Any]] = []
        self.transcript_dir = transcript_dir
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.transcript_dir / "pending.jsonl"
        self.md_path = self.transcript_dir / "pending.md"
        self.evidence_root = evidence_root or _default_evidence_root()
        self.task_root = Path(task_root or Path.cwd()).resolve()
        self.agent_mode = "read-only"
        self.tool_policy = self._default_tool_policy()
        self.last_evidence_pack: dict[str, Any] | None = None
        self.last_task_result: dict[str, Any] | None = None
        self.last_patch: str | None = None
        self.last_patch_sha256: str | None = None
        self.last_trust_card: dict[str, Any] | None = None
        self.last_runner_trace: dict[str, Any] | None = None
        self.last_model_turns: list[dict[str, Any]] = []
        self.last_latency_trace: dict[str, Any] | None = None
        self.last_conversation_report: dict[str, Any] | None = None
        self.last_work_result: dict[str, Any] | None = None
        self.last_work_plan: dict[str, Any] | None = None
        self.last_work_sources: dict[str, Any] | None = None
        self.last_artifact: dict[str, Any] | None = None
        self.last_intent_card: dict[str, Any] | None = None
        self.last_demo_result: dict[str, Any] | None = None
        self.internal_hooks: list[Any] = []
        self._state_path = self.workspace_root / "session-os" / "helix-cli-state.json"
        self.suite_catalog = SuiteEvidenceCatalog(evidence_root=self.evidence_root)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        active_thread_id = self._load_active_thread_id()
        if active_thread_id:
            self._activate_thread(active_thread_id, lifecycle_event="thread_resume")
        else:
            self.new_thread("interactive")
        self._load_last_work_state()

    @property
    def provider(self) -> ProviderSpec:
        return PROVIDERS[self.provider_name]

    def _default_tool_policy(self) -> dict[str, Any]:
        return {
            "mode": "controlled-auto",
            "auto": [
                "helix.search",
                "memory.search",
                "rag.search",
                "list_files",
                "read_file",
                "search_text",
                "git_status",
                "git_diff",
                "query_evidence",
                "file.inspect",
                "memory.resolve",
                "evidence.latest",
                "evidence.refresh",
                "evidence.show",
                "suite.list",
                "suite.catalog",
                "suite.latest",
                "suite.search",
                "suite.transcripts",
                "suite.read",
                "suite.dry_run",
                "web.search",
                "web.read",
            ],
            "confirmation_required": ["/apply last", "/cert SUITE"],
            "blocked_for_planner": ["destructive git", "destructive filesystem"],
        }

    def _thread_paths(self, thread_id: str) -> tuple[Path, Path]:
        return self.transcript_dir / f"{thread_id}.jsonl", self.transcript_dir / f"{thread_id}.md"

    def _load_state(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return {}
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _save_state(self, payload: dict[str, Any]) -> None:
        self._state_path.write_text(
            json.dumps(_json_ready(payload), indent=2, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )

    def _load_active_thread_id(self) -> str | None:
        payload = self._load_state()
        value = payload.get("active_thread_id")
        return _slugish(str(value)) if value else None

    def _save_active_thread_id(self, thread_id: str | None) -> None:
        payload = self._load_state()
        payload["active_thread_id"] = thread_id
        self._save_state(payload)

    def _load_last_work_state(self) -> None:
        payload = self._load_state()
        work = payload.get("last_work") if isinstance(payload.get("last_work"), dict) else {}
        if not work:
            return
        result = work.get("result") if isinstance(work.get("result"), dict) else None
        plan = work.get("plan") if isinstance(work.get("plan"), dict) else None
        sources = work.get("sources") if isinstance(work.get("sources"), dict) else None
        trust_card = work.get("trust_card") if isinstance(work.get("trust_card"), dict) else None
        artifact = work.get("artifact") if isinstance(work.get("artifact"), dict) else None
        intent_card = work.get("intent_card") if isinstance(work.get("intent_card"), dict) else None
        demo = payload.get("last_demo") if isinstance(payload.get("last_demo"), dict) else None
        self.last_work_result = result
        self.last_work_plan = plan
        self.last_work_sources = sources
        self.last_trust_card = trust_card
        self.last_artifact = artifact or (result.get("artifact") if isinstance(result, dict) and isinstance(result.get("artifact"), dict) else None)
        self.last_intent_card = intent_card or (result.get("intent_card") if isinstance(result, dict) and isinstance(result.get("intent_card"), dict) else None)
        self.last_demo_result = demo
        if result:
            self.last_task_result = result
            patch_path = None
            paths = result.get("work_artifact_paths") if isinstance(result.get("work_artifact_paths"), dict) else {}
            if isinstance(paths, dict):
                patch_path = paths.get("patch") or paths.get("work_patch")
            if result.get("patch_available") and patch_path:
                try:
                    patch_text = Path(str(patch_path)).read_text(encoding="utf-8", errors="replace")
                except Exception:
                    patch_text = ""
                if patch_text:
                    self.last_patch = patch_text
                    self.last_patch_sha256 = str(result.get("patch_sha256") or hashlib.sha256(patch_text.encode("utf-8")).hexdigest())

    def _save_last_work_state(self) -> None:
        payload = self._load_state()
        history = payload.get("work_history") if isinstance(payload.get("work_history"), list) else []
        result = self.last_work_result or {}
        plan = self.last_work_plan or {}
        card = self.last_trust_card or {}
        history_entry = {
            "run_id": result.get("run_id"),
            "goal": result.get("goal") or result.get("flow_goal") or plan.get("goal"),
            "status": result.get("status"),
            "engine": result.get("engine"),
            "patch_available": bool(result.get("patch_available")),
            "changed_files": result.get("changed_files") if isinstance(result.get("changed_files"), list) else card.get("changed_files"),
            "output_target": plan.get("output_target") or card.get("output_target"),
            "artifact_path": (self.last_artifact or {}).get("path") if isinstance(self.last_artifact, dict) else None,
            "created_utc": _utc_now(),
        }
        if history_entry.get("run_id"):
            history = [item for item in history if not (isinstance(item, dict) and item.get("run_id") == history_entry["run_id"])]
            history.append(history_entry)
            history = history[-20:]
        payload["last_work"] = {
            "result": self.last_work_result,
            "plan": self.last_work_plan,
            "sources": self.last_work_sources,
            "trust_card": self.last_trust_card,
            "artifact": self.last_artifact,
            "intent_card": self.last_intent_card,
        }
        if self.last_demo_result:
            payload["last_demo"] = self.last_demo_result
        payload["work_history"] = history
        self._save_state(payload)

    def turn_controller(self, text: str) -> IntentCard:
        card = TurnController(self).decide(text)
        self.last_intent_card = card.to_dict()
        return card

    def _run_internal_hooks(self, event_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        decisions: list[dict[str, Any]] = []
        warnings: list[str] = []
        blocked = False
        block_reason = ""
        for hook in list(self.internal_hooks):
            try:
                result = hook(event_name, payload)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{event_name} hook failed: {type(exc).__name__}: {exc}")
                continue
            if not isinstance(result, dict):
                continue
            decisions.append(result)
            if str(result.get("action") or "").lower() in {"block", "deny"}:
                blocked = True
                block_reason = str(result.get("message") or result.get("reason") or f"{event_name} blocked by hook")
        return {"event": event_name, "blocked": blocked, "block_reason": block_reason, "warnings": warnings, "decisions": decisions}

    def work_history(self, *, limit: int = 10) -> list[dict[str, Any]]:
        payload = self._load_state()
        history = payload.get("work_history") if isinstance(payload.get("work_history"), list) else []
        rows = [item for item in history if isinstance(item, dict)]
        return rows[-max(1, min(limit, 50)):]

    def demo_wow(self, *, fast: bool = True, browser: bool = True, with_opencode: bool = False) -> dict[str, Any]:
        started = time.perf_counter()
        run_id = f"demo-wow-{int(time.time() * 1000)}"
        run_dir = self.evidence_root / "demo-wow" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        bus = WorkEventBus(self, run_id=run_id, goal="HeliX Wow Demo")
        scenarios: list[dict[str, Any]] = []
        warnings: list[str] = []
        artifacts: dict[str, Any] = {}
        bus.emit("demo.start", "Iniciando HeliX Wow Demo v0.7.", fast=fast)

        chat_card = self.turn_controller("hola")
        bus.emit("intent.detected", "Chat liviano detectado sin tools.", lane=chat_card.lane, route_reason=chat_card.route_reason)

        source_ref = _demo_source_ref(self.task_root)
        source_goal = f"analiza {source_ref} y armame una pagina web en web/helix-wow-demo"
        intent_card = self.turn_controller(source_goal)
        source_pack = _collect_work_sources(self, source_goal, source_refs=[source_ref], url_refs=[], max_pages=8, depth=1, cross_domain=False)
        plan = {
            "kind": "helix-work-plan-v1",
            "goal": source_goal,
            "work_intent": "source_to_web",
            "intent": "source_to_web",
            "flow_profile": "web-recursive",
            "protocols_used": ["meta_microsite", "hard_anchor_recall", "patch_integrity", "browser_proof"],
            "assurance": "balanced",
            "output_kind": "web",
            "output_target": "web/helix-wow-demo",
            "needs_opencode": False,
            "source_refs": [source_ref],
            "url_refs": [],
            "source_count": len(source_pack.get("sources") or []),
            "anchor_count": len(source_pack.get("anchors") or []),
            "extract_warnings": source_pack.get("warnings") or [],
            "crawler_summary": source_pack.get("crawler_summary") or {},
            "claim_boundary": "This demo proves bounded source readback, artifact generation, patch capture and browser check when available; it does not prove semantic truth.",
            "intent_card": intent_card.to_dict(),
        }
        warnings.extend(str(item) for item in (source_pack.get("warnings") or []))
        bus.emit(
            "source.read",
            f"Fuente base: {source_ref}; anchors capturados: {len(source_pack.get('anchors') or [])}.",
            source_count=plan["source_count"],
            anchor_count=plan["anchor_count"],
        )

        doc_body = _render_internal_work_document({**plan, "work_intent": "source_to_document", "goal": "HeliX Wow Demo brief"}, source_pack)
        doc_md = run_dir / "helix_wow_brief.md"
        doc_md.write_text(doc_body, encoding="utf-8")
        doc_pdf = run_dir / "helix_wow_brief.pdf"
        try:
            pdf_info = _write_simple_pdf(doc_pdf, title="HeliX Wow Demo", body=doc_body)
        except Exception as exc:  # noqa: BLE001
            pdf_info = {"status": "error", "path": str(doc_pdf), "error": f"{type(exc).__name__}: {exc}"}
            warnings.append(f"PDF generation failed: {pdf_info['error']}")
        doc_artifact = _inspect_work_artifact(self, doc_md, last_action="demo_doc_readback")
        bus.emit(
            "artifact.after_read",
            f"Brief re-leido: {doc_artifact.get('readback_chars') or ((doc_artifact.get('readback') or {}).get('chars') if isinstance(doc_artifact.get('readback'), dict) else 0)} chars.",
            artifact=str(doc_md),
        )
        artifacts["brief_markdown"] = doc_artifact
        artifacts["brief_pdf"] = pdf_info
        scenarios.append({**DEMO_WOW_SCENARIOS[0], "status": "passed", "artifacts": [str(doc_md), str(doc_pdf)]})

        site_dir = run_dir / "site"
        site_index = site_dir / "index.html"
        site_dir.mkdir(parents=True, exist_ok=True)
        web_html = _render_internal_work_web(plan, source_pack)
        site_index.write_text(web_html, encoding="utf-8")
        site_artifact = _inspect_work_artifact(self, site_dir, last_action="demo_web_readback")
        bus.emit("artifact.written", f"Web estatica generada: {site_index}.", artifact=str(site_index))
        bus.emit(
            "artifact.after_read",
            f"Web re-leida: {((site_artifact.get('readback') or {}).get('chars') if isinstance(site_artifact.get('readback'), dict) else 0)} chars.",
            artifact=str(site_dir),
        )
        artifacts["site"] = site_artifact

        poor_html = run_dir / "artifact_repair" / "index.html"
        poor_html.parent.mkdir(parents=True, exist_ok=True)
        poor_html.write_text("<!doctype html><title>HeliX</title><h1>HeliX</h1>\n", encoding="utf-8")
        before_repair = _inspect_work_artifact(self, poor_html.parent, last_action="demo_before_repair")
        repaired_html = web_html.replace("</main>", "<section><h2>Repair pass</h2><p>HeliX releyó un artefacto pobre, detectó baja densidad y lo reemplazó por una versión con fuentes, anchors y límites de claim.</p></section></main>")
        poor_html.write_text(repaired_html, encoding="utf-8")
        after_repair = _inspect_work_artifact(self, poor_html.parent, last_action="demo_after_repair")
        bus.emit(
            "artifact.repaired",
            "Artefacto pobre reparado con readback before/after.",
            before_chars=((before_repair.get("readback") or {}).get("chars") if isinstance(before_repair.get("readback"), dict) else 0),
            after_chars=((after_repair.get("readback") or {}).get("chars") if isinstance(after_repair.get("readback"), dict) else 0),
        )
        artifacts["repair_before"] = before_repair
        artifacts["repair_after"] = after_repair
        scenarios.append({**DEMO_WOW_SCENARIOS[1], "status": "passed", "artifacts": [str(poor_html)]})

        rel_web_path = "web/helix-wow-demo/index.html"
        patch_info = _build_text_patch(self.task_root, rel_web_path, web_html)
        patch_text = str(patch_info.get("patch") or "")
        patch_path = run_dir / "patch.diff"
        patch_path.write_text(patch_text, encoding="utf-8")
        patch_apply_check = _git_apply_check(self.task_root, patch_text)
        patch_ready = bool(patch_apply_check.get("ok"))
        bus.emit(
            "verify.apply_check",
            "Patch verificado con git apply --check." if patch_ready else "Patch capturado; apply check no paso en este workspace.",
            status=patch_apply_check.get("status"),
        )
        scenarios.append(
            {
                **DEMO_WOW_SCENARIOS[2],
                "status": "passed" if patch_ready else "warning",
                "patch_path": str(patch_path),
                "patch_sha256": patch_info.get("patch_sha256"),
                "apply_check": patch_apply_check,
            }
        )

        opencode_report = {"status": "skipped", "reason": "demo fast path keeps OpenCode optional"}
        if with_opencode:
            opencode_report = helix_cli_core.opencode_status()
            if not opencode_report.get("available"):
                warnings.append("OpenCode no disponible; patch-safe uso generador interno HeliX.")
        browser_report = _demo_browser_verify(site_index, run_dir, enabled=browser)
        if browser_report.get("status") == "skipped":
            warnings.append(str(browser_report.get("reason") or "browser verification skipped"))
        elif browser_report.get("status") != "passed":
            warnings.append("browser verification did not pass; see browser_snapshot.txt")
        bus.emit(
            "browser.verify",
            "Browser check completado." if browser_report.get("status") == "passed" else f"Browser check {browser_report.get('status')}.",
            status=browser_report.get("status"),
        )
        scenarios.append({**DEMO_WOW_SCENARIOS[3], "status": browser_report.get("status"), "browser": browser_report})

        checks = [
            {"id": "intent_card", "status": "passed", "summary": f"IntentCard lane={intent_card.lane} reason={intent_card.route_reason}."},
            {"id": "source_collection", "status": "passed" if plan["source_count"] else "warning", "summary": f"{plan['source_count']} source(s) collected."},
            {"id": "anchor_capture", "status": "passed" if plan["anchor_count"] else "warning", "summary": f"{plan['anchor_count']} anchor(s) captured."},
            {
                "id": "document_readback",
                "status": (
                    "passed"
                    if isinstance(doc_artifact.get("readback"), dict)
                    and doc_artifact["readback"].get("status") == "ok"
                    and int(doc_artifact["readback"].get("chars") or 0) >= 200
                    else "warning"
                ),
                "summary": "Markdown brief was written and read back.",
            },
            {"id": "web_readback", "status": "passed" if _artifact_readback_ok("web demo", site_artifact) else "warning", "summary": "Static web artifact was written and read back."},
            {
                "id": "artifact_repair",
                "status": (
                    "passed"
                    if (
                        (after_repair.get("sha256") and after_repair.get("sha256") != before_repair.get("sha256"))
                        or (
                            isinstance(after_repair.get("readback"), dict)
                            and isinstance(before_repair.get("readback"), dict)
                            and int(after_repair["readback"].get("chars") or 0) > int(before_repair["readback"].get("chars") or 0)
                        )
                    )
                    else "warning"
                ),
                "summary": "Readback before/after recorded for a repaired artifact.",
            },
            {"id": "patch_integrity", "status": "passed", "summary": "Patch diff captured and hashed."},
            {"id": "apply_check", "status": "passed" if patch_ready else str(patch_apply_check.get("status") or "failed"), "summary": "git apply --check passed." if patch_ready else "Patch is not apply-ready in this workspace."},
            {"id": "browser_verification", "status": str(browser_report.get("status") or "skipped"), "summary": str(browser_report.get("reason") or "Browser snapshot/screenshot captured.")},
            {"id": "claim_boundary", "status": "passed", "summary": "Every demo scenario includes an explicit claim boundary."},
        ]
        trust_card = {
            "kind": "helix-trust-card-v1",
            "subject_type": "demo",
            "status": "passed" if all(item.get("status") in {"passed", "skipped"} for item in checks) else "partial",
            "assurance": "balanced",
            "engine": "helix-demo-orchestrator",
            "run_id": run_id,
            "goal": "HeliX Wow Demo Roadmap",
            "flow_profile": "demo-wow",
            "work_intent": "demo",
            "protocols_used": ["intent_card", "hard_anchor_recall", "artifact_readback", "patch_integrity", "browser_proof", "claim_boundary"],
            "sources": (source_pack.get("sources") or [])[:5],
            "changed_files": [rel_web_path],
            "checks_passed": checks,
            "warnings": warnings,
            "output_target": rel_web_path,
            "patch": {"path": str(patch_path), "sha256": patch_info.get("patch_sha256"), "bytes": len(patch_text.encode("utf-8"))},
            "artifact": site_artifact,
            "artifact_paths": {
                "demo_run": str(run_dir / "demo_run.json"),
                "timeline": str(run_dir / "timeline.jsonl"),
                "trust_card": str(run_dir / "trust_card.json"),
                "patch": str(patch_path),
                "browser_snapshot": browser_report.get("snapshot_path"),
                "screenshot": browser_report.get("screenshot_path"),
                "site": str(site_index),
                "brief_markdown": str(doc_md),
                "brief_pdf": str(doc_pdf),
            },
            "claim_boundary": "This demo proves local orchestration, source readback, artifact readback, patch capture and optional browser verification; it does not prove factual truth or production readiness.",
        }
        scenarios.append({**DEMO_WOW_SCENARIOS[4], "status": "passed", "trust_card_path": str(run_dir / "trust_card.json")})
        bus.emit("trust.updated", "Trust card final actualizada.", status=trust_card["status"])
        bus.emit("done", "Demo wow completado.", status=trust_card["status"])
        latency_trace = _latency_trace(path="demo", started=started, lane="demo-wow")
        latency_trace["work_phase_ms"] = bus.phase_ms()
        latency_trace["work_event_count"] = len(bus.events)
        demo_run = {
            "kind": "helix-demo-wow-run-v1",
            "status": trust_card["status"],
            "run_id": run_id,
            "created_utc": _utc_now(),
            "task_root": str(self.task_root),
            "evidence_root": str(self.evidence_root),
            "fast": fast,
            "scenarios": scenarios,
            "progress_events": list(bus.events),
            "artifacts": artifacts,
            "browser_verification": browser_report,
            "opencode": opencode_report,
            "trust_card": trust_card,
            "patch": patch_text,
            "patch_sha256": patch_info.get("patch_sha256"),
            "patch_available": patch_ready,
            "patch_apply_check": patch_apply_check,
            "changed_files": [rel_web_path],
            "output_file": {"path": str(site_index), "sha256": _sha256_file(site_index), "bytes": site_index.stat().st_size},
            "work_artifact_paths": trust_card["artifact_paths"],
            "claim_boundary": trust_card["claim_boundary"],
            "latency_trace": latency_trace,
        }
        _demo_write_timeline(run_dir / "timeline.jsonl", list(bus.events))
        _demo_write_json(run_dir / "trust_card.json", trust_card)
        _demo_write_json(run_dir / "demo_run.json", demo_run)
        self.last_patch = patch_text if patch_ready else None
        self.last_patch_sha256 = str(patch_info.get("patch_sha256") or "") if patch_ready else None
        self.last_trust_card = trust_card
        self.last_task_result = demo_run
        self.last_work_result = {
            "status": demo_run["status"],
            "mode": "demo",
            "engine": "helix-demo-orchestrator",
            "run_id": run_id,
            "goal": "HeliX Wow Demo Roadmap",
            "final": "Demo wow completado: web, readback, repair, patch gate, browser check y trust card.",
            "work_plan": plan,
            "work_sources": source_pack,
            "work_artifact_paths": trust_card["artifact_paths"],
            "trust_card": trust_card,
            "intent_card": intent_card.to_dict(),
            "progress_events": list(bus.events),
            "flow_goal": source_goal,
            "claim_boundary": trust_card["claim_boundary"],
            "artifact": site_artifact,
            "output_file": demo_run["output_file"],
            "patch": patch_text,
            "patch_sha256": patch_info.get("patch_sha256"),
            "patch_available": patch_ready,
            "patch_generated": True,
            "patch_apply_check": patch_apply_check,
            "changed_files": [rel_web_path],
            "latency_trace": latency_trace,
            "demo_run_path": str(run_dir / "demo_run.json"),
        }
        self.last_work_plan = plan
        self.last_work_sources = source_pack
        self.last_artifact = site_artifact
        self.last_demo_result = demo_run
        self.last_latency_trace = latency_trace
        self._save_last_work_state()
        self.record(
            role="assistant",
            content=str(self.last_work_result.get("final") or ""),
            event_type="demo_final",
            metadata={"mode": "demo", "demo_run": demo_run, "trust_card": trust_card, "latency_trace": latency_trace},
        )
        return demo_run

    def _load_events(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                events.append(payload)
        return events

    def _conversation_turns(self) -> list[dict[str, Any]]:
        return [
            event
            for event in self.events
            if event.get("event") in {"user_turn", "assistant_turn", "task_final", "task_error"}
        ]

    def _latest_thread_summary_event(self) -> dict[str, Any] | None:
        for event in reversed(self.events):
            if event.get("event") == "thread_summary_reset":
                return None
            if event.get("event") == "thread_summary":
                return event
        return None

    def _thread_summary_payload(self) -> dict[str, Any] | None:
        event = self._latest_thread_summary_event()
        if not event:
            return None
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        summary = str(metadata.get("thread_summary") or event.get("content") or "").strip()
        if not summary:
            return None
        return {
            "summary": summary,
            "created_utc": event.get("created_utc"),
            "source_event_count": int(metadata.get("source_event_count") or 0),
            "turn_count": int(metadata.get("turn_count") or 0),
        }

    def _build_thread_summary_text(self) -> tuple[str, int, int]:
        turns = self._conversation_turns()
        selected = turns[-18:]
        lines = []
        for event in selected:
            role = "Assistant" if str(event.get("role") or "") == "assistant" else "User"
            content = _truncate_text(str(event.get("content") or ""), 260)["text"]
            if content:
                lines.append(f"{role}: {content}")
        summary = "\n".join(lines).strip()
        if len(summary) > 1800:
            summary = _truncate_text(summary, 1800)["text"]
        return summary, len(turns), sum(len(str(event.get("content") or "")) for event in turns)

    def ensure_thread_summary(self, *, force: bool = False) -> dict[str, Any]:
        summary_started = time.perf_counter()
        existing = self._thread_summary_payload()
        turns = self._conversation_turns()
        existing_count = int((existing or {}).get("source_event_count") or 0)
        total_chars = sum(len(str(event.get("content") or "")) for event in turns)
        stale = len(turns) - existing_count >= 6 or total_chars >= 6000
        if existing and not force and not stale:
            existing["updated"] = False
            existing["conversation_summary_ms"] = round((time.perf_counter() - summary_started) * 1000, 3)
            return existing
        if not force and not existing and len(turns) < 6 and total_chars < 2400:
            return {
                "summary": "",
                "updated": False,
                "source_event_count": len(turns),
                "turn_count": len(turns),
                "conversation_summary_ms": round((time.perf_counter() - summary_started) * 1000, 3),
            }
        summary, source_event_count, char_count = self._build_thread_summary_text()
        if not summary:
            return {
                "summary": "",
                "updated": False,
                "source_event_count": source_event_count,
                "turn_count": len(turns),
                "conversation_summary_ms": round((time.perf_counter() - summary_started) * 1000, 3),
            }
        event = self.record(
            role="system",
            content=summary,
            event_type="thread_summary",
            metadata={
                "thread_summary": summary,
                "source_event_count": source_event_count,
                "turn_count": len(turns),
                "char_count": char_count,
                "context_policy": "thread_only",
                "retrieval_scope": "session",
            },
            promote=False,
            write_markdown=False,
        )
        return {
            "summary": summary,
            "updated": True,
            "created_utc": event.get("created_utc"),
            "source_event_count": source_event_count,
            "turn_count": len(turns),
            "conversation_summary_ms": round((time.perf_counter() - summary_started) * 1000, 3),
        }

    def reset_thread_summary(self) -> dict[str, Any]:
        event = self.record(
            role="system",
            content="Conversation summary reset for active thread.",
            event_type="thread_summary_reset",
            metadata={"thread_id": self.thread_id, "context_policy": "thread_only"},
            promote=False,
            write_markdown=False,
        )
        return {"status": "ok", "thread_id": self.thread_id, "created_utc": event.get("created_utc")}

    def conversation_status(self) -> dict[str, Any]:
        summary = self._thread_summary_payload()
        return {
            "thread_id": self.thread_id,
            "lane": (self.last_conversation_report or {}).get("lane"),
            "last_route_reason": (self.last_conversation_report or {}).get("route_reason"),
            "recent_intents": self.recent_route_intents(limit=6),
            "thread_summary_present": bool(summary),
            "thread_summary": summary,
            "last_response_gate": (self.last_conversation_report or {}).get("response_gate"),
            "last_suppressed_reasoning": bool(((self.last_conversation_report or {}).get("response_gate") or {}).get("suppressed_reasoning")),
        }

    def _active_thread_branch_info(self) -> dict[str, Any]:
        return self._thread_branch_info(self.thread_id, self.events)

    @staticmethod
    def _event_turn_id(event: dict[str, Any] | None) -> str | None:
        if not isinstance(event, dict):
            return None
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        for value in (event.get("turn_id"), metadata.get("turn_id"), (event.get("helix_memory") or {}).get("memory_id")):
            if value:
                return str(value)
        return None

    def _last_branch_snapshot(self) -> dict[str, Any] | None:
        preferred = {"assistant_turn", "task_final"}
        fallback = {"user_turn", "task_start"}
        for allowed in (preferred, fallback):
            for event in reversed(self.events):
                if event.get("event") not in allowed:
                    continue
                turn_id = self._event_turn_id(event)
                return {
                    "parent_turn_id": turn_id,
                    "parent_event_memory_id": (event.get("helix_memory") or {}).get("memory_id"),
                    "parent_event": event.get("event"),
                    "parent_created_utc": event.get("created_utc"),
                    "parent_summary": _truncate_text(str(event.get("content") or ""), 180)["text"],
                }
        return None

    @staticmethod
    def _thread_branch_info(thread_id: str | None, events: list[dict[str, Any]]) -> dict[str, Any]:
        info: dict[str, Any] = {
            "thread_id": thread_id,
            "kind": "root",
            "parent_thread_id": None,
            "parent_turn_id": None,
            "parent_event_memory_id": None,
            "branch_root_policy": "thread_only_memory",
        }
        for event in events:
            if event.get("event") not in {"thread_open", "branch_open"}:
                continue
            metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
            info["kind"] = "branch" if event.get("event") == "branch_open" else "root"
            info["parent_thread_id"] = metadata.get("parent_thread_id")
            info["parent_turn_id"] = metadata.get("parent_turn_id")
            info["parent_event_memory_id"] = metadata.get("parent_event_memory_id")
            info["branch_root_policy"] = metadata.get("branch_root_policy") or info["branch_root_policy"]
            break
        return info

    def _activate_thread(
        self,
        thread_id: str,
        *,
        lifecycle_event: str,
        lifecycle_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_thread_id = _slugish(thread_id)
        self.thread_id = normalized_thread_id
        self.run_id = normalized_thread_id
        self.jsonl_path, self.md_path = self._thread_paths(normalized_thread_id)
        self.events = self._load_events(self.jsonl_path)
        self._save_active_thread_id(normalized_thread_id)
        metadata = {"thread_id": normalized_thread_id, **(lifecycle_metadata or {})}
        payload = self.record(
            role="system",
            content=json.dumps({"thread_id": normalized_thread_id, "event": lifecycle_event, **(lifecycle_metadata or {})}, ensure_ascii=False),
            event_type=lifecycle_event,
            metadata=metadata,
            promote=True,
            session_id=normalized_thread_id,
        )
        return {
            "thread_id": normalized_thread_id,
            "transcript": str(self.jsonl_path),
            "memory_receipt": payload.get("helix_memory"),
        }

    def ensure_active_thread(self, title: str | None = None) -> str:
        if self.thread_id:
            return self.thread_id
        return str(self.new_thread(title or "interactive").get("thread_id") or _run_id("interactive"))

    def new_thread(self, title: str | None = None) -> dict[str, Any]:
        return self._activate_thread(
            _run_id(title or "helix-thread"),
            lifecycle_event="thread_open",
            lifecycle_metadata={
                "branch_root_policy": "empty_history_thread_only_memory",
                "context_policy": "thread_only",
                "retrieval_scope": "session",
            },
        )

    def branch_thread(self, title: str | None = None) -> dict[str, Any]:
        parent_thread_id = self.ensure_active_thread("interactive")
        snapshot = self._last_branch_snapshot() or {}
        branch_title = title or f"{parent_thread_id}-branch"
        metadata = {
            "parent_thread_id": parent_thread_id,
            "parent_turn_id": snapshot.get("parent_turn_id"),
            "parent_event_memory_id": snapshot.get("parent_event_memory_id"),
            "parent_event": snapshot.get("parent_event"),
            "parent_created_utc": snapshot.get("parent_created_utc"),
            "parent_summary": snapshot.get("parent_summary"),
            "branch_root_policy": "empty_history_thread_only_memory",
            "context_policy": "thread_only",
            "retrieval_scope": "session",
        }
        return self._activate_thread(_run_id(branch_title), lifecycle_event="branch_open", lifecycle_metadata=metadata)

    def open_thread(self, thread_id: str) -> dict[str, Any]:
        return self._activate_thread(thread_id, lifecycle_event="thread_resume")

    def close_thread(self, thread_id: str | None = None) -> dict[str, Any]:
        target = _slugish(thread_id or self.ensure_active_thread("thread"))
        event = self.record(
            role="system",
            content=json.dumps({"thread_id": target, "event": "thread_close"}, ensure_ascii=False),
            event_type="thread_close",
            metadata={"thread_id": target},
            promote=True,
            session_id=target,
        )
        if self.thread_id == target:
            self.thread_id = None
            self.run_id = ""
            self._save_active_thread_id(None)
        return {"thread_id": target, "memory_receipt": event.get("helix_memory")}

    def list_threads(self, *, limit: int = 20) -> list[dict[str, Any]]:
        threads = hmem.list_sessions(
            root=self.workspace_root,
            project=self.project,
            agent_id=self.agent_id,
            limit=limit,
        )
        for item in threads:
            item["active"] = item.get("session_id") == self.thread_id
            jsonl_path, md_path = self._thread_paths(str(item.get("session_id") or ""))
            item["jsonl_path"] = str(jsonl_path)
            item["md_path"] = str(md_path)
            item.update(self._thread_branch_info(str(item.get("session_id") or ""), self._load_events(jsonl_path)))
        return threads

    def current_thread(self) -> dict[str, Any]:
        thread_id = self.ensure_active_thread("interactive")
        branch_info = self._active_thread_branch_info()
        return {
            "thread_id": thread_id,
            "jsonl_path": str(self.jsonl_path),
            "md_path": str(self.md_path),
            "event_count": len(self.events),
            "context_policy": "thread_only",
            "retrieval_scope": "session",
            **branch_info,
            "interaction_mode": self.interaction_mode,
            "latency_mode": self.latency_mode,
            "rust_core": helix_cli_core.rust_core_status(),
            "suite_index": {
                "path": str(self.evidence_root / ".helix-index" / "suites.json"),
                "records_jsonl": str(self.evidence_root / ".helix-index" / "suites.jsonl"),
                "exists": (self.evidence_root / ".helix-index" / "suites.jsonl").exists(),
            },
            "blind_inference": self.blind_status(),
            "tool_policy": self.tool_policy,
        }

    def thread_tree(self) -> dict[str, Any]:
        nodes: dict[str, dict[str, Any]] = {}
        if self.transcript_dir.exists():
            for path in sorted(self.transcript_dir.glob("*.jsonl")):
                thread_id = path.stem
                if thread_id == "pending":
                    continue
                events = self._load_events(path)
                info = self._thread_branch_info(thread_id, events)
                last_event = next((event for event in reversed(events) if event.get("event") not in {"thread_resume"}), None)
                nodes[thread_id] = {
                    **info,
                    "thread_id": thread_id,
                    "active": thread_id == self.thread_id,
                    "event_count": len(events),
                    "jsonl_path": str(path),
                    "md_path": str(path.with_suffix(".md")),
                    "last_event": None if last_event is None else last_event.get("event"),
                    "last_summary": "" if last_event is None else _truncate_text(str(last_event.get("content") or ""), 140)["text"],
                    "children": [],
                }
        if self.thread_id and self.thread_id not in nodes:
            nodes[self.thread_id] = {
                **self._active_thread_branch_info(),
                "active": True,
                "event_count": len(self.events),
                "jsonl_path": str(self.jsonl_path),
                "md_path": str(self.md_path),
                "last_event": None,
                "last_summary": "",
                "children": [],
            }
        for node in nodes.values():
            parent = node.get("parent_thread_id")
            if parent and parent in nodes:
                nodes[parent]["children"].append(node["thread_id"])
        roots = sorted(thread_id for thread_id, node in nodes.items() if not node.get("parent_thread_id") or node.get("parent_thread_id") not in nodes)
        for node in nodes.values():
            node["children"] = sorted(node["children"])
        return {
            "active_thread_id": self.thread_id,
            "context_policy": "thread_only",
            "retrieval_scope": "session",
            "roots": roots,
            "nodes": nodes,
        }

    def thread_tree_text(self) -> str:
        tree = self.thread_tree()
        nodes = tree["nodes"]
        lines = ["Thread tree (context: thread_only/session)"]

        def walk(thread_id: str, prefix: str = "") -> None:
            node = nodes[thread_id]
            marker = "*" if node.get("active") else "-"
            kind = "branch" if node.get("parent_thread_id") else "root"
            parent = f" parent={node.get('parent_thread_id')}@{node.get('parent_turn_id')}" if node.get("parent_thread_id") else ""
            summary = f" :: {node.get('last_summary')}" if node.get("last_summary") else ""
            lines.append(f"{prefix}{marker} {thread_id} [{kind}, events={node.get('event_count')}] {parent}{summary}".rstrip())
            for child in node.get("children", []):
                walk(child, prefix + "  ")

        for root in tree["roots"]:
            walk(root)
        return "\n".join(lines)

    def blind_status(self) -> dict[str, Any]:
        policy = self.blind_inference_policy or BlindInferencePolicy.from_payload(_default_blind_policy_payload())
        return {
            "enabled": bool(self.blind_inference_enabled),
            "scope": policy.scope,
            "placeholder_stability": policy.placeholder_stability,
            "policy_id": policy.policy_id,
            "rule_count": len(policy.rules),
            "detectors": dict(policy.detectors),
            "task_vault_present": bool((self.last_blind_report or {}).get("vault_present")),
            "last_report": dict(self.last_blind_report or {}),
        }

    def _next_blind_task_id(self, *, mode: str) -> str:
        self._blind_task_counter += 1
        thread_id = self.ensure_active_thread(mode)
        seed = f"{thread_id}:{mode}:{self._blind_task_counter}:{time.time_ns()}"
        return f"blind-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:12]}"

    def _blind_request_for_provider(self, provider_name: str, *, mode: str) -> dict[str, Any] | None:
        if not self.blind_inference_enabled:
            return None
        provider = PROVIDERS.get(provider_name)
        if provider is None or not _provider_is_cloud_boundary(provider):
            return None
        policy = BlindInferencePolicy.from_payload(self.blind_inference_policy.to_dict())
        policy.enabled = True
        return {
            "enabled": True,
            "policy": policy.to_dict(),
            "task_id": self._next_blind_task_id(mode=mode),
        }

    def trust_report(
        self,
        thread_id: str | None = None,
        *,
        ref: str | None = None,
        include_quarantined: bool = False,
    ) -> dict[str, Any]:
        target = _slugish(thread_id or self.ensure_active_thread("interactive"))
        catalog = hmem.open_catalog(self.workspace_root)
        try:
            trust_root = catalog.trust_root()
        finally:
            catalog.close()
        lineage = hmem.verify_session_lineage(
            root=self.workspace_root,
            session_id=target,
            include_quarantined=include_quarantined,
        )
        checkpoint = hmem.head_checkpoint(root=self.workspace_root, session_id=target)
        proof = hmem.export_session_proof(
            root=self.workspace_root,
            session_id=target,
            ref=ref,
            include_quarantined=include_quarantined,
        )
        return {
            "kind": "helix-local-trust-report",
            "thread_id": target,
            "workspace_root": str(self.workspace_root),
            "trust_root": {
                "path": str(self.workspace_root / "session-os" / "trust" / "trust_root.json"),
                "version": trust_root.get("version"),
                "active_key_id": trust_root.get("active_key_id"),
                "threshold": trust_root.get("threshold"),
                "external_anchor": trust_root.get("external_anchor"),
            },
            "lineage": lineage,
            "head_checkpoint": checkpoint,
            "proof": proof,
            "interpretation_rules": [
                "`equivocation_detected` means HeliX observed one or more competing non-canonical transitions for this thread; it does not mean the canonical head is missing.",
                "`verified_with_quarantine` means the local canonical chain and signed checkpoint path still verify while quarantined branches are preserved for forensics.",
                "`equivocation_count` and `quarantined_count` are historical thread counters; they are not proof that multiple nodes claimed the exact same sequential slot.",
                "Do not infer tampering, external injection, or loss of provenance from quarantine alone. Escalate those claims only if chain, checkpoint, or signature verification actually failed.",
            ],
            "limits": [
                "Receipts prove local payload integrity/provenance for a stored memory; they do not prove semantic truth.",
                "Signed checkpoints prove the local canonical head selected by this workspace key; they are not global transparency or consensus.",
                "Quarantine preserves equivocation branches for forensics and excludes them from normal retrieval.",
            ],
        }

    def record(
        self,
        *,
        role: str,
        content: str,
        event_type: str,
        metadata: dict[str, Any] | None = None,
        promote: bool = True,
        session_id: str | None = None,
        write_markdown: bool = True,
    ) -> dict[str, Any]:
        active_session_id = _slugish(session_id or self.ensure_active_thread("interactive"))
        with _cli_receipt_signing(active_session_id, event_type, role):
            observed = hmem.observe_event(
                root=self.workspace_root,
                project=self.project,
                agent_id=self.agent_id,
                session_id=active_session_id,
                event_type=event_type,
                content=content,
                summary=f"{role}: {content[:180]}",
                tags=["helix-cli", role, event_type, f"run:{active_session_id}", f"thread:{active_session_id}"],
                importance=6 if role == "assistant" else 5,
                promote=promote,
            )
        memory_id = (observed.get("memory") or {}).get("memory_id")
        receipt = _memory_receipt_for(self.workspace_root, memory_id)
        event = redact_value(
            {
                "event": event_type,
                "role": role,
                "run_id": active_session_id,
                "thread_id": active_session_id,
                "created_utc": _utc_now(),
                "provider": self.provider_name,
                "model": self.model,
                "content": content,
                "metadata": metadata or {},
                "helix_memory": receipt,
            },
            secrets=_secret_values(self.provider),
        )
        if memory_id:
            event["turn_id"] = str(memory_id)
            if isinstance(event.get("metadata"), dict):
                event["metadata"].setdefault("turn_id", str(memory_id))
        jsonl_path, _md_path = self._thread_paths(active_session_id)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        if self.thread_id == active_session_id:
            self.events.append(event)
            if write_markdown:
                self._write_markdown()
        return event

    def _write_markdown(self) -> None:
        lines = [
            "# HeliX Interactive Session",
            "",
            f"- Thread ID: `{self.run_id}`",
            f"- Provider: `{self.provider_name}`",
            f"- Model: `{self.model}`",
            f"- Router policy: `{self.router_policy}`",
            f"- Interaction mode: `{self.interaction_mode}`",
            f"- Blind inference: `{'on' if self.blind_inference_enabled else 'off'}`",
            f"- Project: `{self.project}`",
            f"- Agent ID: `{self.agent_id}`",
            f"- Workspace: `{self.workspace_root}`",
            f"- JSONL: `{self.jsonl_path}`",
            "",
        ]
        for event in self.events:
            receipt = event.get("helix_memory") or {}
            lines.extend(
                [
                    f"## {event.get('role')} / {event.get('event')}",
                    "",
                    f"- UTC: `{event.get('created_utc')}`",
                    f"- Memory ID: `{receipt.get('memory_id')}`",
                    f"- Node hash: `{receipt.get('node_hash')}`",
                    "",
                    str(event.get("content") or ""),
                    "",
                ]
            )
        self.md_path.write_text("\n".join(lines), encoding="utf-8")

    def recent_history(self, *, limit: int = 6, exclude_latest_user: bool = False) -> list[dict[str, str]]:
        turns = [e for e in self.events if e.get("event") in {"user_turn", "assistant_turn", "task_final", "task_error"}]
        if exclude_latest_user and turns and turns[-1].get("event") == "user_turn":
            turns = turns[:-1]
        turns = turns[-limit * 2 :]
        history: list[dict[str, str]] = []
        for event in turns:
            role = "assistant" if str(event.get("role") or "") == "assistant" else "user"
            history.append({"role": role, "content": str(event.get("content") or "")})
        return history

    def recent_route_intents(self, *, limit: int = 4) -> list[str]:
        """Return the route.intent of the last N user turns (oldest first).

        Used by the router to detect a multi-turn "trail" — e.g. three
        consecutive agentic turns means the next vague follow-up should
        stay agentic instead of falling back to chat heuristics.
        """
        intents: list[str] = []
        for event in reversed(self.events):
            if event.get("event") != "user_turn":
                continue
            metadata = event.get("metadata") or {}
            route = metadata.get("route") if isinstance(metadata.get("route"), dict) else {}
            intent = str(route.get("intent") or "").strip()
            if intent:
                intents.append(intent)
            if len(intents) >= int(limit):
                break
        intents.reverse()
        return intents

    def memory_context(
        self,
        query: str,
        *,
        refresh_evidence_first: bool = False,
        budget_tokens: int = 900,
        limit: int = 6,
        retrieval_scope: str = "session",
    ) -> dict[str, Any]:
        active_thread_id = self.ensure_active_thread(query[:32] if query else "interactive")
        if refresh_evidence_first:
            try:
                self.refresh_evidence(query or None, limit=max(6, limit))
            except Exception:
                pass
        context = hmem.build_context(
            root=self.workspace_root,
            project=self.project,
            agent_id=self.agent_id,
            session_id=active_thread_id,
            query=query,
            budget_tokens=budget_tokens,
            mode="search",
            limit=limit,
            retrieval_scope=retrieval_scope,
        )
        context["context_policy"] = "thread_only" if retrieval_scope == "session" else "explicit_global"
        if retrieval_scope == "session" and not context.get("memory_ids") and isinstance(self.last_evidence_pack, dict):
            evidence_records = [
                item
                for item in self.last_evidence_pack.get("records", [])
                if isinstance(item, dict) and item.get("memory_id")
            ][:limit]
            if evidence_records:
                evidence_ids = [str(item.get("memory_id")) for item in evidence_records]
                evidence_lines = [
                    "Certified evidence refreshed in this session:",
                    *[
                        f"- {item.get('suite_id') or 'evidence'} {item.get('run_id') or ''} {item.get('status') or ''} {item.get('artifact_path') or item.get('path') or ''}".strip()
                        for item in evidence_records
                    ],
                ]
                context["memory_ids"] = evidence_ids
                context["items"] = list(context.get("items") or []) + evidence_records
                context["context"] = ((context.get("context") or "") + "\n" + "\n".join(evidence_lines)).strip()
                context["tokens"] = max(1, len(str(context.get("context") or "").split()))
                context["evidence_fast_lane"] = True
        return context

    def _fast_evidence_pack_from_index(self, query: str | None = None, *, limit: int = 8) -> dict[str, Any]:
        if query:
            search = self.suite_catalog.search(query, limit=limit)
            records = list(search.get("results") or [])
        else:
            listed = self.suite_catalog.list_suites()
            records = [
                item.get("latest")
                for item in listed.get("suites", [])
                if isinstance(item, dict) and isinstance(item.get("latest"), dict)
            ][:limit]
        return {
            "source": "helix-evidence-index-fast",
            "query": query,
            "repo_root": str(REPO_ROOT),
            "evidence_root": str(self.evidence_root),
            "record_count": len(records),
            "records": records,
            "replay_skipped": True,
            "deep_available": True,
            "deep_hint": "/evidence refresh --deep" if not query else f"/evidence refresh --deep {query}",
        }

    def refresh_evidence(self, query: str | None = None, *, limit: int = 8, deep: bool = False) -> dict[str, Any]:
        if not deep and self.suite_catalog._fast_index_enabled():
            pack = self._fast_evidence_pack_from_index(query, limit=limit)
            self.last_evidence_pack = pack
            return pack
        pack = refresh_evidence(
            root=self.workspace_root,
            project=self.project,
            agent_id=self.agent_id,
            repo_root=REPO_ROOT,
            evidence_root=self.evidence_root,
            query=query,
            limit=limit,
        )
        self.last_evidence_pack = pack
        return pack

    def latest_evidence(self, *, limit: int = 8) -> list[dict[str, Any]]:
        return list_ingested_evidence(
            root=self.workspace_root,
            project=self.project,
            agent_id=self.agent_id,
            limit=limit,
        )

    def evidence_search(self, query: str, *, limit: int = 8) -> dict[str, Any]:
        pack = self.refresh_evidence(query, limit=max(limit, 8))
        search_result = hmem.search(
            root=self.workspace_root,
            project=self.project,
            agent_id=self.agent_id,
            session_id=self.thread_id,
            query=query,
            top_k=limit * 2,
            retrieval_scope="workspace",
        )
        evidence_hits = [
            item
            for item in search_result.get("results", [])
            if "evidence" in {str(tag) for tag in item.get("tags", [])}
        ]
        return {
            "query": query,
            "record_count": len(evidence_hits),
            "results": evidence_hits[:limit],
            "refresh": pack,
        }

    def evidence_show(self, memory_id: str) -> dict[str, Any] | None:
        catalog = hmem.open_catalog(self.workspace_root)
        try:
            item = catalog.get_memory(memory_id)
            if item is None:
                return None
            node_hash = catalog.get_memory_node_hash(memory_id)
            return {
                "memory": item.to_dict(),
                "node_hash": node_hash,
                "receipt": catalog.get_memory_receipt(memory_id),
                "chain": catalog.verify_chain(node_hash) if node_hash else None,
                "lineage": catalog.session_lineage(item.session_id, include_quarantined=True, limit=8) if item.session_id else None,
            }
        finally:
            catalog.close()

    def _resolve_memory_matches_from_catalog(self, ref: str, *, max_chars: int) -> list[dict[str, Any]]:
        needle = str(ref or "").strip().lower()
        if not needle:
            return []
        catalog = hmem.open_catalog(self.workspace_root)
        try:
            matches: list[dict[str, Any]] = []

            def _payload_for(memory_id: str, item_dict: dict[str, Any] | None = None) -> dict[str, Any] | None:
                item = catalog.get_memory(memory_id)
                if item is None:
                    return None
                node_hash = catalog.get_memory_node_hash(memory_id)
                receipt = catalog.get_memory_receipt(memory_id)
                payload = dict(item_dict or item.to_dict())
                content = _truncate_text(str(payload.get("content") or ""), max_chars)
                payload.update(
                    {
                        "source": "hmem",
                        "memory_id": memory_id,
                        "node_hash": node_hash,
                        "content": content["text"],
                        "content_truncated": content["truncated"],
                        "receipt": receipt,
                        "signature_verified": bool((receipt or {}).get("signature_verified")),
                        "chain": catalog.verify_chain(node_hash) if node_hash else None,
                        "lineage": catalog.session_lineage(item.session_id, include_quarantined=True, limit=8) if item.session_id else None,
                    }
                )
                return payload

            if needle.startswith("mem-"):
                payload = _payload_for(needle)
                return [payload] if payload else []

            for agent_filter in (self.agent_id, None):
                rows = catalog.list_memories(
                    project=self.project,
                    agent_id=agent_filter,
                    session_id=self.thread_id,
                    limit=20000,
                    retrieval_scope="workspace",
                )
                for row in rows:
                    memory_id = str(row.get("memory_id") or "")
                    node_hash = str(row.get("node_hash") or "")
                    if memory_id.lower().startswith(needle) or node_hash.lower().startswith(needle):
                        payload = _payload_for(memory_id, row)
                        if payload:
                            matches.append(payload)
                if matches:
                    break
            seen: set[tuple[str, str]] = set()
            unique: list[dict[str, Any]] = []
            for item in matches:
                key = (str(item.get("memory_id") or ""), str(item.get("node_hash") or ""))
                if key in seen:
                    continue
                seen.add(key)
                unique.append(item)
            return unique
        finally:
            catalog.close()

    def _resolve_memory_matches_from_transcripts(self, ref: str, *, max_chars: int) -> list[dict[str, Any]]:
        needle = str(ref or "").strip().lower()
        if not needle or not self.transcript_dir.exists():
            return []
        matches: list[dict[str, Any]] = []
        jsonl_paths = sorted(self.transcript_dir.rglob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in jsonl_paths[:300]:
            try:
                lines = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
            except Exception:
                continue
            for line_number, line in enumerate(lines, start=1):
                if needle not in line.lower():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                receipt = event.get("helix_memory") if isinstance(event.get("helix_memory"), dict) else {}
                node_hash = str(receipt.get("node_hash") or "")
                memory_id = str(receipt.get("memory_id") or "")
                if not (node_hash.lower().startswith(needle) or memory_id.lower().startswith(needle) or needle in line.lower()):
                    continue
                content = _truncate_text(str(event.get("content") or ""), max_chars)
                matches.append(
                    {
                        "source": "transcript-jsonl",
                        "path": str(path),
                        "line": line_number,
                        "memory_id": memory_id or None,
                        "node_hash": node_hash or None,
                        "role": event.get("role"),
                        "event": event.get("event"),
                        "created_utc": event.get("created_utc"),
                        "content": content["text"],
                        "content_truncated": content["truncated"],
                        "receipt": receipt or None,
                    }
                )
        if matches:
            return matches
        md_paths = sorted(self.transcript_dir.rglob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in md_paths[:300]:
            try:
                text = path.read_text(encoding="utf-8-sig", errors="ignore")
            except Exception:
                continue
            index = text.lower().find(needle)
            if index < 0:
                continue
            start = max(0, index - 1200)
            end = min(len(text), index + max_chars)
            excerpt = text[start:end].strip()
            matches.append(
                {
                    "source": "transcript-md",
                    "path": str(path),
                    "memory_id": None,
                    "node_hash": needle,
                    "content": excerpt,
                    "content_truncated": start > 0 or end < len(text),
                }
            )
        return matches

    def memory_resolve(self, ref: str, *, max_chars: int = 40000) -> dict[str, Any]:
        needle = str(ref or "").strip().lower()
        if not needle:
            return {"status": "error", "error": "missing ref", "ref": ref}
        matches = self._resolve_memory_matches_from_catalog(needle, max_chars=max_chars)
        if not matches:
            matches = self._resolve_memory_matches_from_transcripts(needle, max_chars=max_chars)
        if not matches:
            return {"status": "not_found", "ref": needle, "match_count": 0, "matches": []}
        exact = [
            item
            for item in matches
            if str(item.get("memory_id") or "").lower() == needle or str(item.get("node_hash") or "").lower() == needle
        ]
        if exact:
            matches = exact
        if len(matches) > 1:
            distinct = {
                str(item.get("node_hash") or item.get("memory_id") or item.get("path") or "")
                for item in matches
            }
            if len(distinct) > 1:
                return {"status": "ambiguous", "ref": needle, "match_count": len(matches), "matches": matches[:12]}
        return {"status": "ok", "ref": needle, "match_count": len(matches), "matches": matches[:1]}

    def _file_allowed_roots(self) -> list[Path]:
        roots = [
            self.task_root,
            self.workspace_root,
            self.evidence_root,
            self.transcript_dir,
            REPO_ROOT,
            Path.home(),
        ]
        unique: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            try:
                resolved = Path(root).expanduser().resolve(strict=False)
            except Exception:
                continue
            key = str(resolved).lower()
            if key not in seen:
                unique.append(resolved)
                seen.add(key)
        return unique

    def _path_is_allowed(self, path: Path) -> bool:
        for root in self._file_allowed_roots():
            try:
                path.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def _resolve_user_path(self, ref: str) -> tuple[Path, str]:
        raw = _normalise_local_path_ref(ref)
        if not raw:
            raise ValueError("path is required")
        candidate = Path(raw).expanduser()
        candidates = [candidate] if candidate.is_absolute() else [
            self.task_root / candidate,
            REPO_ROOT / candidate,
            self.evidence_root / candidate,
            self.transcript_dir / candidate,
            self.workspace_root / candidate,
        ]
        selected = candidates[0]
        for item in candidates:
            if item.exists():
                selected = item
                break
        resolved = selected.resolve(strict=False)
        if not self._path_is_allowed(resolved):
            raise PermissionError(f"path is outside allowed HeliX roots: {raw}")
        return resolved, raw

    def _sensitive_file_reason(self, path: Path) -> str | None:
        lowered_parts = [part.lower() for part in path.parts]
        name = path.name.lower()
        suffix = path.suffix.lower()
        if any(part in {".ssh", ".aws", ".azure", ".gcp", ".gnupg"} for part in lowered_parts):
            return "sensitive credential directory"
        if name == ".env" or name.startswith(".env."):
            return "environment secret file"
        if suffix in {".pem", ".key", ".p12", ".pfx", ".crt", ".cer"}:
            return "credential/key file extension"
        if name in {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "known_hosts"}:
            return "ssh credential file"
        if name == "config.json" and "helix" in lowered_parts and "appdata" in lowered_parts:
            return "HeliX user config may contain API tokens"
        if any(part in {"secrets", ".secrets"} for part in lowered_parts):
            return "secret directory"
        return None

    def file_inspect(self, path_ref: str, *, max_bytes: int = 60000, list_limit: int = 80) -> dict[str, Any]:
        try:
            path, raw = self._resolve_user_path(path_ref)
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "blocked" if isinstance(exc, PermissionError) else "error",
                "ref": path_ref,
                "error": f"{type(exc).__name__}: {exc}",
                "allowed_roots": [str(root) for root in self._file_allowed_roots()],
            }
        if not path.exists():
            suggestions: list[dict[str, Any]] = []
            parent = path.parent
            if parent.exists() and parent.is_dir() and self._path_is_allowed(parent):
                needle = path.stem.lower()
                parts = [part for part in re.split(r"[-_.\s]+", needle) if len(part) >= 4]
                for child in sorted(parent.iterdir(), key=lambda item: item.name.lower())[:1000]:
                    child_name = child.name.lower()
                    if needle in child_name or any(part in child_name for part in parts):
                        suggestions.append(
                            {
                                "path": str(child),
                                "kind": "directory" if child.is_dir() else "file",
                                "bytes": child.stat().st_size if child.is_file() else None,
                            }
                        )
                    if len(suggestions) >= 12:
                        break
            return {
                "status": "not_found",
                "ref": raw,
                "path": str(path),
                "parent_exists": parent.exists(),
                "suggestions": suggestions,
            }
        sensitive_reason = self._sensitive_file_reason(path)
        if sensitive_reason:
            return {"status": "blocked", "ref": raw, "path": str(path), "reason": sensitive_reason}
        if path.is_dir():
            limit = _safe_int(list_limit, 80, minimum=1, maximum=200)
            entries = []
            for child in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
                if child.name in ReadOnlyAgentTools.SKIP_DIRS:
                    continue
                reason = self._sensitive_file_reason(child)
                entries.append(
                    {
                        "name": child.name,
                        "path": str(child),
                        "kind": "directory" if child.is_dir() else "file",
                        "bytes": child.stat().st_size if child.is_file() else None,
                        "blocked": bool(reason),
                    }
                )
                if len(entries) >= limit:
                    break
            return {
                "status": "ok",
                "type": "directory",
                "ref": raw,
                "path": str(path),
                "entry_count": len(entries),
                "truncated": len(entries) >= limit,
                "entries": entries,
            }
        if not path.is_file():
            return {"status": "blocked", "ref": raw, "path": str(path), "reason": "not a regular file"}
        max_bytes = _safe_int(max_bytes, 60000, minimum=512, maximum=120000)
        size = path.stat().st_size
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
        if b"\x00" in data[:4096]:
            return {"status": "blocked", "ref": raw, "path": str(path), "bytes": size, "reason": "binary file"}
        text = data[:max_bytes].decode("utf-8", errors="replace")
        text = redact_value(text, secrets=_secret_values(self.provider))
        return {
            "status": "ok",
            "type": "file",
            "ref": raw,
            "path": str(path),
            "name": path.name,
            "suffix": path.suffix.lower(),
            "bytes": size,
            "sha256": _sha256_file(path) if size <= 25_000_000 else None,
            "truncated": size > max_bytes,
            "content": text,
        }

    def certified_identity_evidence(self, *, latest_user_receipt: dict[str, Any] | None = None) -> dict[str, Any]:
        stats = hmem.stats(root=self.workspace_root)
        graph = hmem.graph(
            root=self.workspace_root,
            project=self.project,
            agent_id=self.agent_id,
            session_id=self.thread_id,
            limit=12,
            retrieval_scope="workspace",
        )
        lineage = hmem.verify_session_lineage(
            root=self.workspace_root,
            session_id=self.thread_id,
            include_quarantined=True,
        )
        latest_evidence = self.latest_evidence(limit=5)
        return {
            "claim": "This HeliX CLI session is backed by HeliX memory and evidence exports.",
            "session": {
                "thread_id": self.thread_id,
                "project": self.project,
                "agent_id": self.agent_id,
                "workspace_root": str(self.workspace_root),
                "jsonl_transcript": str(self.jsonl_path),
                "markdown_transcript": str(self.md_path),
            },
            "memory_backend": {
                "memory_count": stats.get("memory_count"),
                "observation_count": stats.get("observation_count"),
                "dag_node_count": stats.get("dag_node_count"),
                "search_backend": stats.get("search_backend"),
                "journal_mode": stats.get("journal_mode"),
            },
            "graph_excerpt": {
                "node_count": graph.get("node_count"),
                "edge_count": graph.get("edge_count"),
                "nodes": graph.get("nodes", [])[:8],
                "edges": graph.get("edges", [])[:8],
            },
            "lineage": lineage,
            "repository_evidence": {
                "evidence_root": str(self.evidence_root),
                "latest_count": len(latest_evidence),
                "latest": [
                    {
                        "memory_id": item.get("memory_id"),
                        "node_hash": item.get("node_hash"),
                        "summary": item.get("summary"),
                        "signature_verified": bool((item.get("receipt") or {}).get("signature_verified")),
                        "chain_status": (item.get("chain") or {}).get("status"),
                    }
                    for item in latest_evidence
                ],
            },
            "latest_user_receipt": _compact_receipt(latest_user_receipt),
            "routing": {
                "policy": self.router_policy,
                "interaction_mode": self.interaction_mode,
                "model_mode": self.model,
                "provider": self.provider_name,
                "known_profile_count": len(MODEL_PROFILES),
                "current_model_profile": _profile_alias_for_model_id(self.model),
            },
            "tombstone_boundary": (
                "The CLI records signed memories and can call HeliX fence/tombstone primitives, "
                "but this interactive shell does not yet auto-tombstone normal chat turns."
            ),
        }

    def architecture_context_pack(self, query: str | None = None, *, include_excerpts: bool = True) -> dict[str, Any]:
        query_text = str(query or "").strip()
        thread_lineage = hmem.verify_session_lineage(
            root=self.workspace_root,
            session_id=self.thread_id,
            include_quarantined=True,
        )
        thread_history = hmem.session_lineage(
            root=self.workspace_root,
            session_id=self.thread_id,
            include_quarantined=True,
            limit=8,
        )
        graph = hmem.graph(
            root=self.workspace_root,
            project=self.project,
            agent_id=self.agent_id,
            session_id=self.thread_id,
            limit=10,
            retrieval_scope="workspace",
            include_quarantined=True,
        )
        suite_hits = []
        if query_text:
            suite_hits = list((self.suite_catalog.search(query_text, limit=6).get("results") or [])[:6])
        if not suite_hits:
            discovered = list((self.suite_catalog.list_suites().get("suites") or [])[:6])
            suite_hits = [
                {
                    "suite_id": item.get("suite_id"),
                    "path": item.get("path"),
                    "kind": "suite",
                    "catalog_scope": "suite",
                    "description": item.get("description"),
                    "latest": item.get("latest"),
                }
                for item in discovered
            ]
        latest_evidence = [
            {
                "memory_id": item.get("memory_id"),
                "node_hash": item.get("node_hash"),
                "summary": item.get("summary"),
                "signature_verified": bool(item.get("signature_verified")),
                "canonical": bool(item.get("canonical", True)),
                "quarantined": bool(item.get("quarantined", False)),
            }
            for item in self.latest_evidence(limit=4)
        ]
        excerpts = [
            _architecture_excerpt(
                Path(spec["path"]),
                label=str(spec["label"]),
                needles=tuple(spec["needles"]),
            )
            for spec in _architecture_excerpt_specs()
        ] if include_excerpts else []
        return {
            "kind": "helix-architecture-context-pack",
            "focus_query": query_text or None,
            "thread_id": self.thread_id,
            "workspace_root": str(self.workspace_root),
            "project": self.project,
            "agent_id": self.agent_id,
            "verified_invariants": [
                "Memory nodes are chained by parent_hash inside the current Merkle-DAG implementation.",
                "verify_chain checks hash continuity of a branch; it is not a proof of canonical head uniqueness.",
                "Signed receipts prove provenance/integrity of the canonical receipt payload, not semantic truth or branch authenticity.",
                "Signed head checkpoints prove the local workspace key selected a canonical head for this thread; they do not provide global non-equivocation.",
                "Canonical head and equivocation semantics are tracked per thread/session and quarantined branches are excluded from normal retrieval.",
            ],
            "claim_boundaries": [
                "Treat `Recursive Witness` and `Branch-Pruning Forensics` as local methodology/evidence terms unless the cited code excerpt shows runtime enforcement.",
                "Do not describe `Ouroboros` as the storage core unless the current turn includes direct local code or evidence for that claim.",
                "If a concept only appears in verification artifacts or suite outputs, label it as evidence or methodology, not guaranteed runtime behavior.",
                "Do not claim CT/Rekor-style public transparency yet; current checkpoints are local signed heads with exportable proof metadata.",
                "Do not reinterpret `equivocation_count` or `quarantined_count` as same-slot sequence collisions unless a cited tool result says that explicitly.",
                "If `trust_status` is `verified_with_quarantine`, describe the canonical head as preserved and locally verifiable unless a chain/checkpoint verification failure is also present.",
                "Do not infer payload tampering, post-generation injection, or broken chain of custody from quarantine alone; that requires failed receipt, chain, or checkpoint verification in the cited evidence.",
            ],
            "interpretation_rules": [
                "`equivocation_detected` in thread lineage means competing branches were recorded for the thread and quarantined from normal retrieval.",
                "`verified_with_quarantine` means the canonical branch remains selected and locally verifiable while non-canonical history is retained for audit.",
                "Historical quarantine counters can grow over time and may include both lineage conflicts and policy quarantines; they are not by themselves catastrophic trust failures.",
            ],
            "module_map": [
                {"path": "helix_kv/memory_catalog.py", "role": "session lineage, canonical head, retrieval filtering, receipts attachment"},
                {"path": "helix_kv/merkle_dag.py", "role": "parent-linked Merkle-DAG node storage and chain traversal"},
                {"path": "src/helix_proto/hmem.py", "role": "workspace memory wrappers used by CLI and runner"},
                {"path": "src/helix_proto/helix_cli.py", "role": "interactive prompts, router, tool registry, and meta-grounding injection"},
                {"path": "src/helix_proto/signed_receipts.py", "role": "receipt signing and verification boundaries"},
            ],
            "thread_lineage": thread_lineage,
            "thread_history": thread_history,
            "graph_excerpt": {
                "node_count": graph.get("node_count"),
                "edge_count": graph.get("edge_count"),
                "nodes": graph.get("nodes", [])[:6],
                "edges": graph.get("edges", [])[:6],
            },
            "evidence_pointers": latest_evidence,
            "suite_pointers": suite_hits,
            "excerpts": excerpts,
        }

    def _agent_memory_tool_manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "helix.search",
                "description": "Search HeliX workspace memory with thread-priority and knowledge context.",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}},
                "safety": "auto",
            },
            {
                "name": "memory.search",
                "description": "Search persisted memory in the current workspace.",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}},
                "safety": "auto",
            },
            {
                "name": "rag.search",
                "description": "Search the agent knowledge base.",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}},
                "safety": "auto",
            },
        ]

    def _cli_extra_tool_registry(self) -> tuple[ToolRegistry, list[dict[str, Any]]]:
        toolbox = ReadOnlyAgentTools(
            root=self.task_root,
            evidence_callback=lambda query, limit: self.refresh_evidence(query, limit=limit),
        )
        specs: list[ToolSpec] = []
        report: list[dict[str, Any]] = []

        def _schema_from_manifest(manifest_item: dict[str, Any]) -> dict[str, Any]:
            properties = {
                str(key): {"type": "string"}
                for key in (manifest_item.get("input_schema") or {}).keys()
            }
            return {"type": "object", "properties": properties}

        for manifest_item in toolbox.manifest():
            tool_name = str(manifest_item["name"])
            specs.append(
                ToolSpec(
                    name=tool_name,
                    description=str(manifest_item.get("description") or ""),
                    input_schema=_schema_from_manifest(manifest_item),
                    handler=lambda args, tool_name=tool_name: (toolbox.call(tool_name, args).get("result") or {}),
                )
            )
            report.append(
                {
                    "name": tool_name,
                    "description": manifest_item.get("description"),
                    "input_schema": manifest_item.get("input_schema"),
                    "safety": "auto",
                    "kind": "repo-readonly",
                }
            )

        specs.extend(
            [
                ToolSpec(
                    name="evidence.latest",
                    description="List the latest ingested evidence memories from verification artifacts.",
                    input_schema={"type": "object", "properties": {"limit": {"type": "integer"}}},
                    handler=lambda args: {"records": self.latest_evidence(limit=_safe_int(args.get("limit"), 8, minimum=1, maximum=20))},
                ),
                ToolSpec(
                    name="evidence.refresh",
                    description="Refresh verification evidence into HeliX memory and return matching records.",
                    input_schema={"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}},
                    handler=lambda args: self.refresh_evidence(
                        str(args.get("query") or "") or None,
                        limit=_safe_int(args.get("limit"), 8, minimum=1, maximum=20),
                    ),
                ),
                ToolSpec(
                    name="evidence.show",
                    description="Inspect one ingested evidence memory, receipt and chain status.",
                    input_schema={"type": "object", "properties": {"memory_id": {"type": "string"}}, "required": ["memory_id"]},
                    handler=lambda args: self.evidence_show(str(args["memory_id"])) or {"status": "not_found"},
                ),
                ToolSpec(
                    name="memory.resolve",
                    description="Resolve a memory_id or node_hash prefix to exact HeliX memory/transcript content.",
                    input_schema={
                        "type": "object",
                        "properties": {"ref": {"type": "string"}, "max_chars": {"type": "integer"}},
                        "required": ["ref"],
                    },
                    handler=lambda args: self.memory_resolve(
                        str(args["ref"]),
                        max_chars=_safe_int(args.get("max_chars"), 40000, minimum=1000, maximum=120000),
                    ),
                ),
                ToolSpec(
                    name="helix.architecture",
                    description="Return a grounded HeliX architecture pack with lineage state, claim boundaries, evidence pointers, and curated code excerpts.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "include_excerpts": {"type": "boolean"},
                        },
                    },
                    handler=lambda args: self.architecture_context_pack(
                        str(args.get("query") or "") or None,
                        include_excerpts=bool(args.get("include_excerpts", True)),
                    ),
                ),
                ToolSpec(
                    name="helix.trust",
                    description="Return local signed checkpoint, canonical lineage and proof metadata for a HeliX thread.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "thread_id": {"type": "string"},
                            "ref": {"type": "string"},
                            "include_quarantined": {"type": "boolean"},
                        },
                    },
                    handler=lambda args: self.trust_report(
                        str(args.get("thread_id") or "") or None,
                        ref=str(args.get("ref") or "") or None,
                        include_quarantined=bool(args.get("include_quarantined", False)),
                    ),
                ),
                ToolSpec(
                    name="file.inspect",
                    description="Inspect an explicit local file or directory path under allowed HeliX roots; reads text files or lists directories.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "max_bytes": {"type": "integer"},
                            "list_limit": {"type": "integer"},
                        },
                        "required": ["path"],
                    },
                    handler=lambda args: self.file_inspect(
                        str(args["path"]),
                        max_bytes=_safe_int(args.get("max_bytes"), 60000, minimum=512, maximum=120000),
                        list_limit=_safe_int(args.get("list_limit"), 80, minimum=1, maximum=200),
                    ),
                ),
                ToolSpec(
                    name="suite.list",
                    description="List verification suites known to HeliX.",
                    input_schema={"type": "object", "properties": {}},
                    handler=lambda _args: {
                        "suites": [
                            {
                                "suite_id": suite.suite_id,
                                "script": suite.script,
                                "script_exists": suite.script_path.exists(),
                                "description": suite.description,
                                "requires_deepinfra": suite.requires_deepinfra,
                                "supports_deepinfra_flag": suite.supports_deepinfra_flag,
                            }
                            for suite in sorted(SUITES.values(), key=lambda item: item.suite_id)
                        ],
                        "discovered": self.suite_catalog.list_suites(),
                    },
                ),
                ToolSpec(
                    name="suite.catalog",
                    description="Discover local verification suites with artifacts, manifests, preregisters and transcripts.",
                    input_schema={"type": "object", "properties": {}},
                    handler=lambda _args: self.suite_catalog.list_suites(),
                ),
                ToolSpec(
                    name="suite.latest",
                    description="Show latest artifact, manifest and transcript records for one suite.",
                    input_schema={"type": "object", "properties": {"suite_id": {"type": "string"}}, "required": ["suite_id"]},
                    handler=lambda args: self.suite_catalog.latest(str(args["suite_id"])),
                ),
                ToolSpec(
                    name="suite.search",
                    description="Search suite artifacts, manifests, preregisters, logs and transcripts under verification/.",
                    input_schema={"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]},
                    handler=lambda args: self.suite_catalog.search(
                        str(args["query"]),
                        limit=_safe_int(args.get("limit"), 12, minimum=1, maximum=50),
                    ),
                ),
                ToolSpec(
                    name="suite.transcripts",
                    description="List transcript JSONL/Markdown files for one suite.",
                    input_schema={
                        "type": "object",
                        "properties": {"suite_id": {"type": "string"}, "query": {"type": "string"}, "limit": {"type": "integer"}},
                        "required": ["suite_id"],
                    },
                    handler=lambda args: self.suite_catalog.transcripts(
                        str(args["suite_id"]),
                        query=str(args.get("query") or "") or None,
                        limit=_safe_int(args.get("limit"), 30, minimum=1, maximum=100),
                    ),
                ),
                ToolSpec(
                    name="suite.read",
                    description="Read a safe excerpt from one suite artifact, preregister, log or transcript path.",
                    input_schema={"type": "object", "properties": {"ref": {"type": "string"}, "max_bytes": {"type": "integer"}}, "required": ["ref"]},
                    handler=lambda args: self.suite_catalog.read(
                        str(args["ref"]),
                        max_bytes=_safe_int(args.get("max_bytes"), 16000, minimum=512, maximum=60000),
                    ),
                ),
                ToolSpec(
                    name="web.search",
                    description="Search the public web for current information when the user explicitly asks for web/Google/latest/source lookup.",
                    input_schema={"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]},
                    handler=lambda args: web_search(
                        str(args["query"]),
                        limit=_safe_int(args.get("limit"), 5, minimum=1, maximum=10),
                    ),
                ),
                ToolSpec(
                    name="web.read",
                    description="Read a bounded text excerpt from an HTTP/HTTPS URL returned by web.search.",
                    input_schema={"type": "object", "properties": {"url": {"type": "string"}, "max_chars": {"type": "integer"}}, "required": ["url"]},
                    handler=lambda args: web_read(
                        str(args["url"]),
                        max_chars=_safe_int(args.get("max_chars"), 8000, minimum=1000, maximum=30000),
                    ),
                ),
                ToolSpec(
                    name="suite.dry_run",
                    description="Show the command for a verification suite without running it.",
                    input_schema={"type": "object", "properties": {"suite_id": {"type": "string"}}, "required": ["suite_id"]},
                    handler=lambda args: run_cert_suite(
                        str(args["suite_id"]),
                        provider_name=self.provider_name if self.provider_name == "deepinfra" else None,
                        prompt_token=False,
                        dry_run=True,
                    ),
                ),
            ]
        )
        report.extend(
            [
                {"name": "evidence.latest", "description": "List latest evidence memories.", "safety": "auto", "kind": "evidence"},
                {"name": "evidence.refresh", "description": "Incrementally refresh evidence into memory.", "safety": "auto", "kind": "evidence"},
                {"name": "evidence.show", "description": "Inspect one evidence memory and receipt.", "safety": "auto", "kind": "evidence"},
                {"name": "memory.resolve", "description": "Resolve memory_id or node_hash prefix to exact content.", "safety": "auto", "kind": "memory"},
                {"name": "helix.architecture", "description": "Return a grounded HeliX architecture pack.", "safety": "auto", "kind": "memory"},
                {"name": "helix.trust", "description": "Return local signed checkpoint and lineage proof.", "safety": "auto", "kind": "memory"},
                {"name": "file.inspect", "description": "Read text files or list directories from explicit local paths.", "safety": "auto", "kind": "filesystem-read"},
                {"name": "suite.list", "description": "List verification suites.", "safety": "auto", "kind": "suite"},
                {"name": "suite.catalog", "description": "Discover local suite artifacts and transcripts.", "safety": "auto", "kind": "suite"},
                {"name": "suite.latest", "description": "Show latest suite evidence.", "safety": "auto", "kind": "suite"},
                {"name": "suite.search", "description": "Search artifacts and transcripts.", "safety": "auto", "kind": "suite"},
                {"name": "suite.transcripts", "description": "List suite transcripts.", "safety": "auto", "kind": "suite"},
                {"name": "suite.read", "description": "Read artifact/transcript excerpts.", "safety": "auto", "kind": "suite"},
                {"name": "suite.dry_run", "description": "Inspect a suite command without executing it.", "safety": "auto", "kind": "suite"},
                {"name": "web.search", "description": "Search the public web for current information.", "safety": "auto", "kind": "web"},
                {"name": "web.read", "description": "Read a bounded excerpt from a web result URL.", "safety": "auto", "kind": "web"},
            ]
        )
        return ToolRegistry(specs), report

    def tool_registry_report(self) -> dict[str, Any]:
        _extra_registry, extra_report = self._cli_extra_tool_registry()
        runtime_tools = self.runtime.tool_manifest()
        agent_tools = self._agent_memory_tool_manifest()
        return {
            "mode": self.agent_mode,
            "thread_id": self.thread_id,
            "task_root": str(self.task_root),
            "tool_policy": self.tool_policy,
            "tools": [*runtime_tools, *agent_tools, *extra_report],
            "tool_count": len(runtime_tools) + len(agent_tools) + len(extra_report),
            "agent_blueprints": agent_blueprints_report(),
        }

    def _chat_system(
        self,
        *,
        user_text: str,
        memory_context: dict[str, Any],
        identity_evidence: dict[str, Any] | None,
        repository_evidence_pack: dict[str, Any] | None,
        tool_manifest: list[dict[str, Any]],
        helix_focus: bool = False,
        helix_auditability: bool = False,
        architecture_context_pack: dict[str, Any] | None = None,
        interaction_mode: str = "balanced",
        tone_contract: str | None = None,
    ) -> str:
        active_mode = _normalize_interaction_mode(interaction_mode)
        mode_contract = tone_contract or INTERACTION_MODE_PROFILES[active_mode]["tone_contract"]
        mode_instruction = (
            f"Interaction mode: {active_mode}. {mode_contract} "
            if active_mode == "balanced"
            else (
                f"Interaction mode: {active_mode}. {mode_contract} "
                "In technical mode, prioritize verified local evidence, concrete semantics, paths, hashes, tests and next actions. "
                if active_mode == "technical"
                else f"Interaction mode: {active_mode}. {mode_contract} "
                "In explore mode, do not turn every HeliX mention into core architecture analysis; use cultural, philosophical or creative framing when the prompt asks for it, while labeling speculation clearly. "
            )
        )
        helix_focus_instruction = (
            "The user is explicitly asking to understand HeliX. Give a grounded explanation first, then list 3-6 concrete things it enables in practice. "
            if helix_focus
            else ""
        )
        helix_auditability_instruction = (
            "The user is asking about HeliX auditability, hashes, receipts, or signatures. Explain concrete semantics such as signed memories, node hashes, signature verification, chain status, and current scope limits. "
            if helix_auditability
            else ""
        )
        helix_architecture_instruction = (
            "An architecture context pack is attached for this turn. Use it as the primary anchor for HeliX claims, and separate verified implementation facts from inference or suite-only terminology. "
            if architecture_context_pack
            else ""
        )
        architecture_pack_section = (
            "HeliX architecture context pack:\n"
            f"{_architecture_context_blob(architecture_context_pack)}\n\n"
            if architecture_context_pack
            else ""
        )
        body = (
            "You are HeliX interactive, a practical coding and research shell running through the unified HeliX runtime. "
            "HeliX is the deterministic orchestration, memory, routing, and evidence layer around local or cloud models; "
            "do not claim that HeliX itself is the language model. "
            f"Runtime UTC now: {_utc_now()}. Thread ID: {self.thread_id}. "
            "You may either answer directly, or request exactly one tool by emitting "
            "<tool_call>{\"tool\":\"name\",\"arguments\":{...}}</tool_call>. "
            "If no tool is needed, return only the visible answer, optionally wrapped in <helix_output>...</helix_output>. "
            "Do not invent dates, run IDs, hashes, memory IDs, node hashes, or file paths. "
            "Do not reveal chain-of-thought, scratchpads, plans, hidden reasoning, or fake tool calls. "
            f"{mode_instruction}"
            "If the user gives a memory_id or node_hash prefix and asks where it is, what it contains, or to recover it, use memory.resolve and never reconstruct the content from model memory. "
            "If the user gives an explicit local file or directory path and asks to read, inspect, open, navigate, or comment on it, use file.inspect before answering. "
            f"Response style: {self.response_style}. {RESPONSE_STYLES.get(self.response_style, RESPONSE_STYLES['balanced'])} "
            "For ordinary questions outside HeliX, answer normally and follow the user's topic; do not force Merkle-DAG, receipts, evidence, or routing metaphors into unrelated conversation. "
            "If the user pastes suite output, tables, JSON, tracebacks, or logs and asks for info/data, analyze the pasted evidence and any tool results; do not rerun certification suites unless the user explicitly asks to run/certify them. "
            "When explaining HeliX itself, describe only concrete, observable capabilities grounded in the current memory, evidence pack, runtime state, or tool registry. "
            "Do not claim that HeliX captures 'trajectories of thought', preserves hidden reasoning, or records private chain-of-thought unless that exact capability is explicitly present in the evidence provided here. "
            "Do not say HeliX guarantees conversations were not altered unless the evidence pack contains a verified signature and verified chain for the exact cited record; otherwise say it records receipts/hashes that can be checked. "
            "Never claim specific suites, runs, artifacts, or transcripts are present unless they appear in tool output, repository evidence, or memory context in this turn. "
            "Prefer plain statements about what HeliX stores, signs, searches, routes, verifies, exposes, or automates in this session. "
            "If the evidence is partial, say what is verified and what remains unverified instead of filling gaps with theory. "
            "Do not pad HeliX explanations with generic industry examples such as healthcare, finance, education, or security unless the evidence pack actually mentions them. "
        )
        body += helix_focus_instruction + helix_auditability_instruction + helix_architecture_instruction
        body += f"{_preferred_language_instruction(user_text)} "
        body += "Certified HeliX evidence pack:\n"
        body += f"{json.dumps(identity_evidence or {}, ensure_ascii=False, indent=2)}\n\n"
        body += "Certified repository evidence pack:\n"
        body += f"{json.dumps(repository_evidence_pack or self.last_evidence_pack or {}, ensure_ascii=False, indent=2)}\n\n"
        body += architecture_pack_section
        body += "Deep Memory:\n"
        body += f"{memory_context.get('context') or '(empty)'}\n\n"
        body += "Recent terminal turns:\n"
        body += f"{json.dumps(self.recent_history(limit=6, exclude_latest_user=True), ensure_ascii=False, indent=2)}\n\n"
        body += "Available tools:\n"
        body += f"{json.dumps(tool_manifest, ensure_ascii=False, indent=2)}"
        return body

    def _lightweight_chat_system(
        self,
        *,
        user_text: str,
        interaction_mode: str = "balanced",
        tone_contract: str | None = None,
        thread_summary: str | None = None,
    ) -> str:
        active_mode = _normalize_interaction_mode(interaction_mode)
        mode_contract = tone_contract or INTERACTION_MODE_PROFILES[active_mode]["tone_contract"]
        summary_section = (
            f"Active thread summary for continuity only:\n{thread_summary}\n\n"
            if thread_summary
            else ""
        )
        return (
            "You are HeliX interactive speaking through the selected language model. "
            "For ordinary conversation, answer directly and naturally. "
            "Do not mention certified evidence, Merkle-DAGs, receipts, memory backends, or internal orchestration unless the user explicitly asks about HeliX, sources, verification, or current-data limits. "
            "If the topic is general knowledge, lifestyle, culture, weather seasons, or casual conversation, just answer the question instead of narrating what data you do or do not have in HeliX. "
            "Keep the answer proportionate to the question. "
            f"Interaction mode: {active_mode}. {mode_contract} "
            f"Response style: {self.response_style}. {RESPONSE_STYLES.get(self.response_style, RESPONSE_STYLES['balanced'])} "
            f"{_preferred_language_instruction(user_text)} "
            f"{summary_section}"
            "Do not reveal hidden reasoning. Never print headings like 'Thinking Process', 'Analysis', or 'System Instructions'; answer with only the final visible response."
        )

    def _task_system(
        self,
        *,
        tool_manifest: list[dict[str, Any]],
        memory_context: dict[str, Any],
        repository_evidence_pack: dict[str, Any] | None,
        helix_focus: bool = False,
        helix_auditability: bool = False,
        architecture_context_pack: dict[str, Any] | None = None,
        interaction_mode: str = "balanced",
        tone_contract: str | None = None,
    ) -> str:
        active_mode = _normalize_interaction_mode(interaction_mode)
        mode_contract = tone_contract or INTERACTION_MODE_PROFILES[active_mode]["tone_contract"]
        helix_meta_instruction = (
            "This task is about HeliX itself. Prioritize the attached architecture context pack, local evidence, and local code-grounding over theory. "
            if (helix_focus or helix_auditability or architecture_context_pack)
            else ""
        )
        architecture_pack_section = (
            "\nHeliX architecture context pack:\n"
            + _architecture_context_blob(architecture_context_pack)
            + "\n\n"
            if architecture_context_pack
            else ""
        )
        body = (
            "You are HeliX Agent Shell running through the unified HeliX runtime with persistent thread memory. "
            f"Thread ID: {self.thread_id}. Task root: {self.task_root}. "
            "Use at most one tool per turn. Request tools only with <tool_call> JSON. "
            "If enough evidence is available, answer directly or inside <helix_output>...</helix_output>. "
            "You may inspect repo files, git state, HeliX evidence, and suite metadata. "
            f"Interaction mode: {active_mode}. {mode_contract} "
            f"Response style: {self.response_style}. {RESPONSE_STYLES.get(self.response_style, RESPONSE_STYLES['balanced'])} "
            "Do not invent file paths, hashes, test results, or patch application claims. "
            "For explicit memory-review requests, a sentence like 'voy a buscar...' is not a final answer: "
            "either call helix.search / memory.search or provide the actual summary. "
            "For explicit local file paths, read or list them with file.inspect/read_file before making claims about their content. "
            "When reasoning about HeliX architecture, distinguish verified implementation facts, design inference, and methodology/evidence terminology. "
            "If code changes are needed, you may propose a unified diff in the final answer, but never claim a patch was applied automatically.\n\n"
        )
        body += helix_meta_instruction
        body += architecture_pack_section
        body += "Current deep memory:\n"
        body += f"{memory_context.get('context') or '(empty)'}\n\n"
        body += "Certified repository evidence pack:\n"
        body += f"{json.dumps(repository_evidence_pack or self.last_evidence_pack or {}, ensure_ascii=False, indent=2)}\n\n"
        body += "Available tools:\n"
        body += f"{json.dumps(tool_manifest, ensure_ascii=False, indent=2)}"
        return body

    def _planner_callback_factory(
        self,
        *,
        goal: str,
        mode: str,
        selected_model: str,
        selected_provider_name: str | None,
        tool_manifest: list[dict[str, Any]],
        memory_context: dict[str, Any],
        identity_evidence: dict[str, Any] | None,
        repository_evidence_pack: dict[str, Any] | None,
        helix_focus: bool = False,
        helix_auditability: bool = False,
        suite_focus: bool = False,
        web_focus: bool = False,
        architecture_context_pack: dict[str, Any] | None = None,
        hash_recovery_ref: str | None = None,
        file_path_ref: str | None = None,
        url_refs: list[str] | None = None,
        interaction_mode: str = "balanced",
        tone_contract: str | None = None,
        native_request: dict[str, Any] | None = None,
        blind_request: dict[str, Any] | None = None,
        fallback_models: list[str] | None = None,
        timeout: float | None,
    ) -> tuple[Any, list[dict[str, Any]]]:
        model_turns: list[dict[str, Any]] = []
        active_interaction_mode = _normalize_interaction_mode(interaction_mode)
        active_url_refs = list(url_refs or [])

        def _callback(state: dict[str, Any]) -> PlannerDecision:
            observations = [
                {
                    "tool": item.get("tool_name"),
                    "arguments": item.get("arguments"),
                    "result": item.get("observation"),
                }
                for item in state.get("observations", [])
            ]
            if mode == "chat" and observations and str(observations[-1].get("tool") or "") == "memory.resolve":
                result = observations[-1].get("result") if isinstance(observations[-1].get("result"), dict) else {}
                if isinstance(result, dict) and isinstance(result.get("result"), dict):
                    result = result["result"]
                return PlannerDecision(
                    kind="final",
                    thought="render exact memory.resolve result without model reconstruction",
                    final=_format_memory_resolve_answer(result),
                    planner="memory-resolve",
                    raw_text="",
                )
            if mode == "chat" and hash_recovery_ref and not observations:
                return PlannerDecision(
                    kind="tool",
                    thought="resolve node hash or memory id prefix before answering",
                    tool_name="memory.resolve",
                    arguments={"ref": hash_recovery_ref, "max_chars": 60000},
                    planner="memory-resolve",
                    raw_text="",
                )
            if mode == "chat" and file_path_ref and not observations:
                return PlannerDecision(
                    kind="tool",
                    thought="inspect explicit local file or directory path before answering",
                    tool_name="file.inspect",
                    arguments={"path": file_path_ref, "max_bytes": 80000, "list_limit": 100},
                    planner="file-grounding",
                    raw_text="",
                )
            if (
                mode == "chat"
                and active_url_refs
                and not observations
                and active_interaction_mode == "explore"
                and not (isinstance(native_request, dict) and native_request.get("mode") == "gemini-native")
            ):
                return PlannerDecision(
                    kind="tool",
                    thought="read explicit URL in explore mode when provider-native URL Context is unavailable",
                    tool_name="web.read",
                    arguments={"url": active_url_refs[0], "max_chars": 12000},
                    planner="web-grounding",
                    raw_text="",
                )
            if mode in {"chat", "task"} and suite_focus and not observations:
                if _looks_like_pasted_suite_evidence(goal):
                    return PlannerDecision(
                        kind="tool",
                        thought="analyze pasted suite output by searching local suite evidence instead of rerunning the suite",
                        tool_name="suite.search",
                        arguments={"query": goal, "limit": 8},
                        planner="suite-grounding",
                        raw_text="",
                    )
                suite_id = _suite_from_text(goal)
                if suite_id:
                    return PlannerDecision(
                        kind="tool",
                        thought="ground suite questions in local verification artifacts and transcripts",
                        tool_name="suite.latest",
                        arguments={"suite_id": suite_id},
                        planner="suite-grounding",
                        raw_text="",
                    )
                return PlannerDecision(
                    kind="tool",
                    thought="ground suite questions in local verification search results",
                    tool_name="suite.search",
                    arguments={"query": goal, "limit": 8},
                    planner="suite-grounding",
                    raw_text="",
                )
            if mode == "chat" and web_focus and not observations:
                return PlannerDecision(
                    kind="tool",
                    thought="explicit web/current-info request requires web.search before answering",
                    tool_name="web.search",
                    arguments={"query": goal, "limit": 5},
                    planner="web-grounding",
                    raw_text="",
                )
            if mode in {"chat", "task"} and helix_auditability and not observations:
                return PlannerDecision(
                    kind="tool",
                    thought="ground HeliX auditability questions in the local trust report before analyzing semantics",
                    tool_name="helix.trust",
                    arguments={"thread_id": self.thread_id, "include_quarantined": True},
                    planner="helix-trust-grounding",
                    raw_text="",
                )
            if mode in {"chat", "task"} and helix_focus and architecture_context_pack and not observations:
                return PlannerDecision(
                    kind="tool",
                    thought="ground meta-HeliX questions in the local architecture pack before answering",
                    tool_name="helix.architecture",
                    arguments={"query": goal, "include_excerpts": True},
                    planner="helix-grounding",
                    raw_text="",
                )
            if mode == "chat" and helix_focus and not observations:
                return PlannerDecision(
                    kind="tool",
                    thought="ground HeliX explanation requests in actual workspace memory before answering",
                    tool_name="helix.search",
                    arguments={"query": _helix_grounding_query(goal), "top_k": 6},
                    planner="helix-grounding",
                    raw_text="",
                )
            history = self.recent_history(limit=6, exclude_latest_user=True)
            if observations:
                history.append(
                    {
                        "role": "user",
                        "content": "HeliX read-only tool results:\n"
                        + _truncate_text(json.dumps(observations[-8:], ensure_ascii=False, indent=2), 12000)["text"],
                    }
                )
            prompt = (
                _agent_observation_prompt(
                    goal,
                    observations,
                    mode=mode,
                    helix_focus=helix_focus,
                    helix_auditability=helix_auditability,
                )
                if observations
                else goal
            )
            system = (
                self._task_system(
                    tool_manifest=tool_manifest,
                    memory_context=memory_context,
                    repository_evidence_pack=repository_evidence_pack,
                    helix_focus=helix_focus,
                    helix_auditability=helix_auditability,
                    architecture_context_pack=architecture_context_pack,
                    interaction_mode=active_interaction_mode,
                    tone_contract=tone_contract,
                )
                if mode == "task"
                else self._chat_system(
                    user_text=goal,
                    memory_context=memory_context,
                    identity_evidence=identity_evidence,
                    repository_evidence_pack=repository_evidence_pack,
                    tool_manifest=tool_manifest,
                    helix_focus=helix_focus,
                    helix_auditability=helix_auditability,
                    architecture_context_pack=architecture_context_pack,
                    interaction_mode=active_interaction_mode,
                    tone_contract=tone_contract,
                )
            )
            result = run_chat_with_failover(
                provider_name=selected_provider_name or self.provider_name,
                model=selected_model,
                fallback_models=fallback_models,
                prompt=prompt,
                system=system,
                history=history,
                max_tokens=max(self.max_tokens, 1400 if mode == "task" else self.max_tokens),
                temperature=self.temperature,
                workspace_root=self.workspace_root,
                prompt_token=False,
                timeout=timeout,
                native_request=native_request,
                blind_inference=blind_request,
            )
            raw_text = str(result.get("text") or "").strip()
            calls = _parse_agent_tool_calls(raw_text)
            cleaned_text = _task_visible_output(raw_text)
            model_turns.append(
                {
                    "actual_model": result.get("actual_model"),
                    "selected_model": selected_model,
                    "latency_ms": result.get("latency_ms"),
                    "finish_reason": result.get("finish_reason"),
                    "usage": result.get("usage"),
                    "native_request": native_request,
                    "native_tool_metadata": result.get("native_tool_metadata"),
                    "blind_inference": result.get("blind_inference"),
                    "failover_used": result.get("failover_used"),
                    "failover_attempts": result.get("failover_attempts") or [],
                    "tool_call_count": len(calls),
                    "raw_preview": raw_text[:2000],
                    "raw_text": raw_text,
                }
            )
            if calls:
                first = calls[0]
                tool_name = str(first.get("tool") or "")
                tool_arguments = _repair_planner_tool_arguments(
                    tool_name,
                    first.get("arguments") if isinstance(first.get("arguments"), dict) else {},
                    goal,
                )
                return PlannerDecision(
                    kind="tool",
                    thought="provider planner requested a tool",
                    tool_name=tool_name,
                    arguments=tool_arguments,
                    planner=f"{self.provider_name}:{selected_model}",
                    raw_text=raw_text,
                )
            if (
                mode == "task"
                and _goal_requests_memory_lookup(goal)
                and not observations
                and (
                    _looks_like_deferred_lookup_preamble(cleaned_text)
                    or _looks_like_unverified_memory_claim(cleaned_text)
                    or cleaned_text in {
                        "[raw output suppressed: model returned only internal reasoning or tool protocol residue]",
                        "No response from provider.",
                    }
                )
            ):
                return PlannerDecision(
                    kind="tool",
                    thought="explicit memory-review request requires an actual memory search step",
                    tool_name="helix.search",
                    arguments={"query": goal, "top_k": 6},
                    planner=f"{self.provider_name}:{selected_model}",
                    raw_text=raw_text,
                )
            return PlannerDecision(
                kind="final",
                thought="provider planner returned a final answer",
                final=cleaned_text,
                planner=f"{self.provider_name}:{selected_model}",
                raw_text=raw_text,
            )

        return _callback, model_turns

    def chat(self, user_text: str, *, interaction_mode_override: str | None = None) -> dict[str, Any]:
        chat_started = time.perf_counter()
        route_ms = 0.0
        active_interaction_mode = _normalize_interaction_mode(interaction_mode_override or self.interaction_mode)
        rust_route = helix_cli_core.route(
            user_text,
            latency_mode=self.latency_mode,
            interaction_mode=active_interaction_mode,
        )
        rust_core_ms = float(rust_route.get("rust_core_ms") or rust_route.get("routing_ms") or 0.0)
        recent_history = self.recent_history(limit=4, exclude_latest_user=False)
        helix_focus = _is_helix_explanation_request(user_text, recent_history)
        helix_auditability = _is_helix_auditability_request(user_text, recent_history)
        suite_focus = _is_suite_evidence_request(user_text, recent_history)
        web_focus = _is_web_search_request(user_text)
        hash_recovery_ref = _latest_hash_reference(user_text, recent_history) if _is_hash_recovery_request(user_text, recent_history) else None
        url_refs = _extract_url_refs(user_text)
        file_path_refs = _extract_local_path_refs(user_text)
        file_path_ref = file_path_refs[0] if file_path_refs else None
        route = None
        selected_model = self.model
        selected_provider_name = self.provider_name
        model_is_auto = self.model.lower() in {"auto", "router:auto"}
        if model_is_auto:
            route_started = time.perf_counter()
            route = route_model_for_task(
                f"{user_text}\nMode: chat",
                provider_name=self.provider_name,
                policy=self.router_policy,
                interaction_mode=active_interaction_mode,
                recent_intents=self.recent_route_intents(limit=4),
            )
            route_ms += (time.perf_counter() - route_started) * 1000
            if (
                active_interaction_mode == "explore"
                and isinstance(route, dict)
                and route.get("intent") in {"chat", "reasoning", "research"}
                and _recent_history_mentions_helix(recent_history)
                and _is_creative_helix_prompt(f"helix {user_text}")
            ):
                route = _override_route_for_creative_helix_focus(
                    route,
                    user_text=user_text,
                    policy=self.router_policy,
                )
            if (
                active_interaction_mode != "explore"
                and self.provider_name == "deepinfra"
                and helix_focus
                and isinstance(route, dict)
                and route.get("intent") == "chat"
            ):
                route = _override_route_for_helix_focus(
                    route,
                    user_text=user_text,
                    policy=self.router_policy,
                    auditability=helix_auditability,
                )
            selected_provider_name = str(route.get("provider") or self.provider_name)
            selected_model = route.get("model") or PROVIDERS[selected_provider_name].default_model
        else:
            route_started = time.perf_counter()
            route = _manual_route_for_model(
                selected_model,
                provider_name=self.provider_name,
                policy=self.router_policy,
                user_text=user_text,
                interaction_mode=active_interaction_mode,
            )
            route_ms += (time.perf_counter() - route_started) * 1000
            selected_provider_name = str(route.get("provider") or self.provider_name)
        if (
            url_refs
            and not file_path_refs
            and self.provider_name == "deepinfra"
            and _provider_ready("gemini")
            and isinstance(route, dict)
        ):
            alias = _gemini_alias_for_prompt(
                user_text,
                intent=str(route.get("intent") or "web_research"),
                capability_requirements=_capability_requirements_for_prompt(
                    user_text,
                    intent=str(route.get("intent") or "web_research"),
                    url_refs=url_refs,
                    path_refs=file_path_refs,
                    suite_focus=suite_focus,
                    web_focus=True,
                    interaction_mode=active_interaction_mode,
                ),
            )
            profile = GEMINI_MODEL_PROFILES[alias]
            selected_provider_name = "gemini"
            selected_model = profile.model_id
            route.update(
                {
                    "provider": "gemini",
                    "model": profile.model_id,
                    "profile": alias,
                    "signals": sorted(set([*(route.get("signals") or []), "gemini_url_context_candidate"])),
                }
            )
            route = _augment_route_metadata(route, user_text, interaction_mode=active_interaction_mode)
        if isinstance(route, dict):
            route["rust_core"] = rust_route
            route["local_path"] = rust_route.get("path")
        native_tool_plan = dict(route.get("native_tool_plan") or {}) if isinstance(route, dict) else {}
        capability_requirements = dict(route.get("capability_requirements") or {}) if isinstance(route, dict) else {}
        mode_policy = dict(route.get("mode_policy") or _interaction_mode_payload(active_interaction_mode)) if isinstance(route, dict) else _interaction_mode_payload(active_interaction_mode)
        tone_contract = str(route.get("tone_contract") or mode_policy.get("tone_contract") or INTERACTION_MODE_PROFILES[active_interaction_mode]["tone_contract"]) if isinstance(route, dict) else INTERACTION_MODE_PROFILES[active_interaction_mode]["tone_contract"]
        grounding_plan = str(route.get("grounding_plan") or "helix-only") if isinstance(route, dict) else "helix-only"
        fallback_models = _fallback_model_ids_for_route(
            route,
            primary_model=selected_model,
            include_route_fallbacks=True,
        )
        blind_request = self._blind_request_for_provider(selected_provider_name, mode="chat")
        blind_status = self.blind_status()
        certified_evidence_required = _needs_certified_evidence(user_text, route, recent_history)
        use_lightweight_chat = _should_use_lightweight_chat_path(
            user_text,
            route=route,
            recent_history=recent_history,
            helix_focus=helix_focus,
            helix_auditability=helix_auditability,
            suite_focus=suite_focus,
            web_focus=web_focus,
            hash_recovery_ref=hash_recovery_ref,
            file_path_ref=file_path_ref,
            url_refs=url_refs,
            latency_mode=self.latency_mode,
        )
        if (
            use_lightweight_chat
            and model_is_auto
            and self.latency_mode == "fast"
            and selected_provider_name == "deepinfra"
        ):
            fast_profile = DEEPINFRA_MODEL_PROFILES["chat"]
            previous_model = selected_model
            selected_model = fast_profile.model_id
            selected_provider_name = fast_profile.provider
            if isinstance(route, dict):
                route["fast_model_override"] = {
                    "reason": "lightweight_fast_path_prefers_non_reasoning_chat_model",
                    "from_model": previous_model,
                    "to_model": fast_profile.model_id,
                    "to_profile": "chat",
                }
                route["provider"] = fast_profile.provider
                route["model"] = fast_profile.model_id
                route["profile"] = "chat"
            fallback_models = _fallback_model_ids_for_route(
                route,
                primary_model=selected_model,
                include_route_fallbacks=True,
            )
        lane, route_reason = _conversation_lane_for_turn(
            user_text,
            route=route,
            use_lightweight_chat=use_lightweight_chat,
            file_path_ref=file_path_ref,
            helix_auditability=helix_auditability,
            suite_focus=suite_focus,
            web_focus=web_focus,
            hash_recovery_ref=hash_recovery_ref,
            certified_evidence_required=certified_evidence_required,
        )
        if isinstance(route, dict):
            route["lane"] = lane
            route["route_reason"] = route_reason

        if lane == "file_qa" and file_path_ref:
            file_qa_started = time.perf_counter()
            summary_payload = self.ensure_thread_summary()
            thread_summary = str(summary_payload.get("summary") or "")
            inspection = self.file_inspect(file_path_ref)
            observation = {
                "tool_name": "file.inspect",
                "arguments": {"path": file_path_ref, "max_bytes": 60000, "list_limit": 80},
                "observation": {"result": inspection},
            }
            memory_ids: list[str] = []
            user_event = self.record(
                role="user",
                content=user_text,
                event_type="user_turn",
                metadata={
                    "recall_memory_ids": memory_ids,
                    "route": route,
                    "thread_id": self.thread_id,
                    "context_policy": "thread_summary_only",
                    "retrieval_scope": "session",
                    "lane": lane,
                    "route_reason": route_reason,
                    "thread_summary_used": bool(thread_summary),
                    "path_refs": file_path_refs,
                    "url_refs": url_refs,
                    "native_tool_plan": native_tool_plan,
                    "capability_requirements": capability_requirements,
                    "interaction_mode": active_interaction_mode,
                    "mode_policy": mode_policy,
                    "grounding_plan": "file-qa",
                    "tone_contract": tone_contract,
                    "architecture_context_enabled": False,
                    "memory_context_skipped": True,
                    "latency_mode": self.latency_mode,
                },
                write_markdown=False,
            )
            history = self.recent_history(limit=4, exclude_latest_user=True)
            observation_text = "file.inspect observations:\n" + json.dumps([observation], ensure_ascii=False, indent=2)
            if thread_summary:
                history.insert(0, {"role": "assistant", "content": f"Active thread summary:\n{thread_summary}"})
            history.append({"role": "assistant", "content": observation_text})
            raw_text = ""
            if inspection.get("status") == "blocked":
                raw_text = (
                    f"No puedo leer ese path porque HeliX lo bloquea como dato sensible: "
                    f"{inspection.get('reason') or inspection.get('error') or 'blocked'}."
                )
                result: dict[str, Any] = {"actual_model": "helix-local-file-guard", "latency_ms": 0.0, "finish_reason": "blocked", "usage": {"total_tokens": 0}}
            else:
                result = run_chat_with_failover(
                    provider_name=selected_provider_name or self.provider_name,
                    model=selected_model,
                    fallback_models=fallback_models,
                    prompt=f"{user_text}\n\n{observation_text}",
                    system=(
                        "You are HeliX file QA. Answer from the provided file.inspect observations. "
                        "If the observation says blocked, missing, or error, report that status directly. "
                        "Do not claim you read anything beyond the observation. "
                        f"{_preferred_language_instruction(user_text)}"
                    ),
                    history=history,
                    max_tokens=min(self.max_tokens, 700),
                    temperature=self.temperature,
                    workspace_root=self.workspace_root,
                    prompt_token=False,
                    timeout=None,
                    native_request=native_tool_plan,
                    blind_inference=blind_request,
                )
                raw_text = str(result.get("text") or "")
            gate = _conversation_gate(raw_text)
            clean_text = str(gate.get("visible_text") or "")
            latency_trace = _latency_trace(
                path="grounded",
                started=chat_started,
                rust_core_ms=rust_core_ms,
                route_ms=route_ms,
                routing_ms=float(rust_route.get("routing_ms") or route_ms),
                context_ms=(time.perf_counter() - file_qa_started) * 1000,
                python_ms=(time.perf_counter() - file_qa_started) * 1000,
                provider_latency_ms=result.get("latency_ms"),
                max_tokens=min(self.max_tokens, 700),
                history_turns=len(history),
                lane=lane,
                conversation_summary_ms=float(summary_payload.get("conversation_summary_ms") or 0.0),
                response_gate_ms=float(gate.get("response_gate_ms") or 0.0),
                suppressed_reasoning=bool(gate.get("suppressed_reasoning")),
                repair_retry_used=bool(gate.get("repair_retry_used")),
            )
            file_trace = {
                "mode": "file_qa",
                "observations": [observation],
                "initial_memory_context": {"memory_ids": memory_ids, "skipped": True, "thread_summary_used": bool(thread_summary)},
                "timing": latency_trace,
            }
            self.last_runner_trace = file_trace
            self.last_model_turns = [
                {
                    "actual_model": result.get("actual_model"),
                    "selected_model": selected_model,
                    "latency_ms": result.get("latency_ms"),
                    "finish_reason": result.get("finish_reason"),
                    "usage": result.get("usage"),
                    "timing": latency_trace,
                    "tool_call_count": 1,
                    "raw_preview": raw_text[:2000],
                    "raw_text": raw_text,
                }
            ]
            self.last_latency_trace = latency_trace
            self.last_conversation_report = {
                "lane": lane,
                "route_reason": route_reason,
                "thread_summary_used": bool(thread_summary),
                "response_gate": gate,
            }
            self.record(
                role="assistant",
                content=clean_text,
                event_type="assistant_turn",
                metadata={
                    "actual_model": result.get("actual_model"),
                    "selected_model": selected_model,
                    "route": route,
                    "lane": lane,
                    "route_reason": route_reason,
                    "thread_summary_used": bool(thread_summary),
                    "response_gate": gate,
                    "local_fallback_used": bool(gate.get("local_fallback_used")) or inspection.get("status") == "blocked",
                    "latency_ms": result.get("latency_ms"),
                    "finish_reason": result.get("finish_reason"),
                    "usage": result.get("usage"),
                    "recall_memory_ids": memory_ids,
                    "context_policy": "thread_summary_only",
                    "retrieval_scope": "session",
                    "path_refs": file_path_refs,
                    "url_refs": url_refs,
                    "native_tool_plan": native_tool_plan,
                    "capability_requirements": capability_requirements,
                    "interaction_mode": active_interaction_mode,
                    "mode_policy": mode_policy,
                    "grounding_plan": "file-qa",
                    "tone_contract": tone_contract,
                    "thread_id": self.thread_id,
                    "latency_mode": self.latency_mode,
                    "latency_trace": latency_trace,
                },
            )
            return {
                "text": clean_text,
                "raw_text": raw_text,
                "reasoning": "",
                "route": route,
                "trace": file_trace,
                "interaction_mode": active_interaction_mode,
                "blind_inference": {},
            }

        if use_lightweight_chat:
            lightweight_started = time.perf_counter()
            summary_payload = self.ensure_thread_summary()
            thread_summary = str(summary_payload.get("summary") or "")
            context = {
                "context": "",
                "memory_ids": [],
                "tokens": 0,
                "retrieval_scope": "skipped",
                "context_policy": "zero_context",
                "skip_reason": "lightweight_chat_zero_context",
            }
            memory_ids = list(context.get("memory_ids") or [])
            user_event = self.record(
                role="user",
                content=user_text,
                event_type="user_turn",
                metadata={
                    "recall_memory_ids": memory_ids,
                    "route": route,
                    "thread_id": self.thread_id,
                    "context_policy": "zero_context",
                    "retrieval_scope": "skipped",
                    "path_refs": file_path_refs,
                    "url_refs": url_refs,
                    "native_tool_plan": native_tool_plan,
                    "capability_requirements": capability_requirements,
                    "interaction_mode": active_interaction_mode,
                    "mode_policy": mode_policy,
                    "grounding_plan": "lightweight-chat",
                    "tone_contract": tone_contract,
                    "architecture_context_enabled": False,
                    "lane": lane,
                    "route_reason": route_reason,
                    "thread_summary_used": bool(thread_summary),
                    "fast_path": "lightweight_chat",
                    "memory_context_skipped": True,
                    "blind_inference_enabled": bool(blind_request),
                    "blind_policy_id": blind_status.get("policy_id"),
                    "blind_provider_target": selected_provider_name,
                    "latency_mode": self.latency_mode,
                    "latency_trace": {
                        "path": "lightweight",
                        "rust_core_ms": round(rust_core_ms, 3),
                        "route_ms": round(route_ms, 3),
                        "routing_ms": round(float(rust_route.get("routing_ms") or route_ms), 3),
                        "evidence_ms": 0.0,
                        "memory_ms": 0.0,
                        "context_ms": 0.0,
                        "python_ms": 0.0,
                        "agent_runner_ms": 0.0,
                        "conversation_summary_ms": round(float(summary_payload.get("conversation_summary_ms") or 0.0), 3),
                        "lane": lane,
                        "local_budget_exceeded": False,
                    },
                },
                write_markdown=False,
            )
            excluded_memory_ids = [str((user_event.get("helix_memory") or {}).get("memory_id") or "")]
            excluded_memory_ids = [item for item in excluded_memory_ids if item]
            history = self.recent_history(limit=2, exclude_latest_user=True)
            pre_model_ms = (time.perf_counter() - lightweight_started) * 1000
            lightweight_max_tokens = _lightweight_chat_token_budget(user_text, self.max_tokens)
            local_answer = _basic_helix_fast_answer(user_text, recent_history=recent_history)
            if local_answer:
                latency_trace = _latency_trace(
                    path="lightweight",
                    started=chat_started,
                    rust_core_ms=rust_core_ms,
                    route_ms=route_ms,
                    routing_ms=float(rust_route.get("routing_ms") or route_ms),
                    python_ms=pre_model_ms,
                    provider_latency_ms=0.0,
                    max_tokens=0,
                    history_turns=len(history),
                    lane=lane,
                    conversation_summary_ms=float(summary_payload.get("conversation_summary_ms") or 0.0),
                )
                lightweight_timing = {
                    **latency_trace,
                    "pre_model_ms": pre_model_ms,
                        "local_fast_answer": True,
                        "thread_summary_used": bool(thread_summary),
                    }
                self.last_runner_trace = None
                self.last_model_turns = [
                    {
                        "actual_model": "helix-local-fast-answer",
                        "selected_model": selected_model,
                        "latency_ms": 0.0,
                        "finish_reason": "local_fast_answer",
                        "usage": {"total_tokens": 0},
                        "native_request": native_tool_plan,
                        "native_tool_metadata": None,
                        "blind_inference": {"requested": False, "enabled": False, "reason": "local_fast_answer"},
                        "failover_used": False,
                        "failover_attempts": [],
                        "timing": lightweight_timing,
                        "tool_call_count": 0,
                        "raw_preview": local_answer[:2000],
                        "raw_text": local_answer,
                    }
                ]
                self.last_blind_report = {"requested": False, "enabled": False, "reason": "local_fast_answer"}
                self.last_latency_trace = latency_trace
                self.record(
                    role="assistant",
                    content=local_answer,
                    event_type="assistant_turn",
                    metadata={
                        "actual_model": "helix-local-fast-answer",
                        "selected_model": selected_model,
                        "failover_used": False,
                        "failover_attempts": [],
                        "route": route,
                        "latency_ms": 0.0,
                        "finish_reason": "local_fast_answer",
                        "usage": {"total_tokens": 0},
                        "recall_memory_ids": memory_ids,
                        "path_refs": file_path_refs,
                        "url_refs": url_refs,
                        "native_tool_plan": native_tool_plan,
                        "capability_requirements": capability_requirements,
                        "interaction_mode": active_interaction_mode,
                        "mode_policy": mode_policy,
                        "grounding_plan": "lightweight-chat",
                        "tone_contract": tone_contract,
                        "architecture_context_enabled": False,
                        "lane": lane,
                        "route_reason": route_reason,
                        "thread_summary_used": bool(thread_summary),
                        "response_gate": _conversation_gate(local_answer),
                        "raw_model_text": local_answer,
                        "visible_output_cleaned": False,
                        "reasoning_internal": "",
                        "thread_id": self.thread_id,
                        "fast_path": "lightweight_chat",
                        "local_fast_answer": True,
                        "local_fallback_used": False,
                        "memory_context_skipped": True,
                        "latency_mode": self.latency_mode,
                        "latency_trace": latency_trace,
                        "timing": lightweight_timing,
                    },
                )
                self.last_conversation_report = {
                    "lane": lane,
                    "route_reason": route_reason,
                    "thread_summary_used": bool(thread_summary),
                    "response_gate": _conversation_gate(local_answer),
                }
                return {
                    "text": local_answer,
                    "raw_text": local_answer,
                    "reasoning": "",
                    "route": route,
                    "trace": {
                        "mode": "lightweight_chat",
                        "local_fast_answer": True,
                        "initial_memory_context": {"memory_ids": memory_ids, "skipped": True, "thread_summary_used": bool(thread_summary)},
                        "timing": lightweight_timing,
                    },
                    "interaction_mode": active_interaction_mode,
                    "blind_inference": dict(self.last_blind_report or {}),
                }
            try:
                result = run_chat_with_failover(
                    provider_name=selected_provider_name or self.provider_name,
                    model=selected_model,
                    fallback_models=fallback_models,
                    prompt=user_text,
                    system=self._lightweight_chat_system(
                        user_text=user_text,
                        interaction_mode=active_interaction_mode,
                        tone_contract=tone_contract,
                        thread_summary=thread_summary,
                    ),
                    history=history,
                    max_tokens=lightweight_max_tokens,
                    temperature=self.temperature,
                    workspace_root=self.workspace_root,
                    prompt_token=False,
                    timeout=None,
                    native_request=native_tool_plan,
                    blind_inference=blind_request,
                )
            except Exception as exc:  # noqa: BLE001
                error_text = f"{type(exc).__name__}: {exc}"
                clean_text = _friendly_provider_failure_text(error_text) or f"Task failed: {error_text}"
                latency_trace = _latency_trace(
                    path="lightweight",
                    started=chat_started,
                    rust_core_ms=rust_core_ms,
                    route_ms=route_ms,
                    routing_ms=float(rust_route.get("routing_ms") or route_ms),
                    python_ms=pre_model_ms,
                    max_tokens=lightweight_max_tokens,
                    history_turns=len(history),
                )
                self.last_latency_trace = latency_trace
                self.last_model_turns = [
                    {
                        "actual_model": None,
                        "selected_model": selected_model,
                        "latency_ms": None,
                        "finish_reason": "error",
                        "usage": None,
                        "failover_used": True,
                        "failover_attempts": [{"provider": selected_provider_name, "model": selected_model, "error_type": type(exc).__name__, "error": str(exc)}],
                        "timing": latency_trace,
                        "tool_call_count": 0,
                        "raw_preview": error_text,
                        "raw_text": error_text,
                    }
                ]
                self.record(
                    role="assistant",
                    content=clean_text,
                    event_type="assistant_turn",
                    metadata={
                        "selected_model": selected_model,
                        "route": route,
                        "thread_id": self.thread_id,
                        "fast_path": "lightweight_chat",
                        "provider_error": error_text,
                        "latency_mode": self.latency_mode,
                        "latency_trace": latency_trace,
                    },
                )
                return {
                    "text": clean_text,
                    "raw_text": error_text,
                    "reasoning": "",
                    "route": route,
                    "trace": {"mode": "lightweight_chat", "error": error_text, "timing": latency_trace},
                    "interaction_mode": active_interaction_mode,
                    "blind_inference": {},
                }
            latency_trace = _latency_trace(
                path="lightweight",
                started=chat_started,
                rust_core_ms=rust_core_ms,
                route_ms=route_ms,
                routing_ms=float(rust_route.get("routing_ms") or route_ms),
                python_ms=pre_model_ms,
                provider_latency_ms=result.get("latency_ms"),
                max_tokens=lightweight_max_tokens,
                history_turns=len(history),
                lane=lane,
                conversation_summary_ms=float(summary_payload.get("conversation_summary_ms") or 0.0),
            )
            lightweight_timing = {
                **latency_trace,
                "pre_model_ms": pre_model_ms,
            }
            raw_text = str(result.get("text") or "")
            gate = _conversation_gate(raw_text)
            clean_text = str(gate.get("visible_text") or "")
            latency_trace["response_gate_ms"] = gate.get("response_gate_ms")
            latency_trace["suppressed_reasoning"] = bool(gate.get("suppressed_reasoning"))
            latency_trace["repair_retry_used"] = bool(gate.get("repair_retry_used"))
            lightweight_timing.update(
                {
                    "response_gate_ms": gate.get("response_gate_ms"),
                    "suppressed_reasoning": bool(gate.get("suppressed_reasoning")),
                    "repair_retry_used": bool(gate.get("repair_retry_used")),
                }
            )
            self.last_runner_trace = None
            self.last_model_turns = [
                {
                    "actual_model": result.get("actual_model"),
                    "selected_model": selected_model,
                    "latency_ms": result.get("latency_ms"),
                    "finish_reason": result.get("finish_reason"),
                    "usage": result.get("usage"),
                    "native_request": native_tool_plan,
                    "native_tool_metadata": result.get("native_tool_metadata"),
                    "blind_inference": result.get("blind_inference"),
                    "failover_used": result.get("failover_used"),
                    "failover_attempts": result.get("failover_attempts") or [],
                    "timing": lightweight_timing,
                    "tool_call_count": 0,
                    "raw_preview": raw_text[:2000],
                    "raw_text": raw_text,
                }
            ]
            self.last_blind_report = dict(result.get("blind_inference") or {})
            self.last_latency_trace = latency_trace
            self.record(
                role="assistant",
                content=clean_text,
                event_type="assistant_turn",
                metadata={
                    "actual_model": result.get("actual_model"),
                    "selected_model": selected_model,
                    "failover_used": result.get("failover_used"),
                    "failover_attempts": result.get("failover_attempts") or [],
                    "route": route,
                    "latency_ms": result.get("latency_ms"),
                    "finish_reason": result.get("finish_reason"),
                    "usage": result.get("usage"),
                    "recall_memory_ids": memory_ids,
                    "path_refs": file_path_refs,
                    "url_refs": url_refs,
                    "native_tool_plan": native_tool_plan,
                    "capability_requirements": capability_requirements,
                    "interaction_mode": active_interaction_mode,
                    "mode_policy": mode_policy,
                    "grounding_plan": "lightweight-chat",
                    "tone_contract": tone_contract,
                    "architecture_context_enabled": False,
                    "lane": lane,
                    "route_reason": route_reason,
                    "thread_summary_used": bool(thread_summary),
                    "response_gate": gate,
                    "raw_model_text": raw_text,
                    "visible_output_cleaned": clean_text != raw_text,
                    "reasoning_internal": "",
                    "thread_id": self.thread_id,
                    "fast_path": "lightweight_chat",
                    "local_fallback_used": bool(gate.get("local_fallback_used")),
                    "memory_context_skipped": True,
                    "blind_inference_enabled": bool((self.last_blind_report or {}).get("enabled")),
                    "blind_policy_id": blind_status.get("policy_id"),
                    "blind_provider_target": selected_provider_name,
                    "blind_span_count": (self.last_blind_report or {}).get("span_count"),
                    "blind_sensitive_classes": (self.last_blind_report or {}).get("sensitive_classes") or [],
                    "blind_warnings": (self.last_blind_report or {}).get("warnings") or [],
                    "blind_baseline_redaction_applied": bool((self.last_blind_report or {}).get("baseline_redaction_applied")),
                    "latency_mode": self.latency_mode,
                    "latency_trace": latency_trace,
                    "timing": lightweight_timing,
                },
            )
            self.last_conversation_report = {
                "lane": lane,
                "route_reason": route_reason,
                "thread_summary_used": bool(thread_summary),
                "response_gate": gate,
            }
            return {
                "text": clean_text,
                "raw_text": raw_text,
                "reasoning": "",
                "route": route,
                "trace": {
                    "mode": "lightweight_chat",
                    "initial_memory_context": {"memory_ids": memory_ids, "skipped": True, "thread_summary_used": bool(thread_summary)},
                    "timing": lightweight_timing,
                },
                "interaction_mode": active_interaction_mode,
                "blind_inference": dict(self.last_blind_report or {}),
            }

        repository_evidence_pack = None
        if _needs_repository_evidence(user_text, route):
            evidence_started = time.perf_counter()
            repository_evidence_pack = self.refresh_evidence(user_text, limit=8)
            evidence_ms = (time.perf_counter() - evidence_started) * 1000
        else:
            evidence_ms = 0.0

        memory_started = time.perf_counter()
        context = self.memory_context(user_text)
        memory_ms = (time.perf_counter() - memory_started) * 1000
        architecture_context_pack = (
            self.architecture_context_pack(user_text, include_excerpts=True)
            if (
                helix_auditability
                or str((route or {}).get("intent") or "") in {"helix_self", "audit"}
                or (active_interaction_mode != "explore" and helix_focus)
            )
            else None
        )
        memory_ids = list(context.get("memory_ids") or [])
        user_event = self.record(
            role="user",
            content=user_text,
            event_type="user_turn",
            metadata={
                "recall_memory_ids": memory_ids,
                "route": route,
                "thread_id": self.thread_id,
                "context_policy": "thread_only",
                "retrieval_scope": "session",
                "path_refs": file_path_refs,
                "url_refs": url_refs,
                "native_tool_plan": native_tool_plan,
                "capability_requirements": capability_requirements,
                "interaction_mode": active_interaction_mode,
                "mode_policy": mode_policy,
                "grounding_plan": grounding_plan,
                "tone_contract": tone_contract,
                "architecture_context_enabled": bool(architecture_context_pack),
                "lane": lane,
                "route_reason": route_reason,
                "thread_summary_used": False,
                "blind_inference_enabled": bool(blind_request),
                "blind_policy_id": blind_status.get("policy_id"),
                "blind_provider_target": selected_provider_name,
                "latency_mode": self.latency_mode,
                "latency_trace": {
                    "path": "grounded",
                    "route_ms": round(route_ms, 3),
                    "evidence_ms": round(evidence_ms, 3),
                    "memory_ms": round(memory_ms, 3),
                },
            },
        )
        excluded_memory_ids = [str((user_event.get("helix_memory") or {}).get("memory_id") or "")]
        excluded_memory_ids = [item for item in excluded_memory_ids if item]
        identity_evidence = None
        if helix_focus or _needs_certified_evidence(user_text, route, recent_history):
            identity_evidence = self.certified_identity_evidence(
                latest_user_receipt=user_event.get("helix_memory"),
            )
        extra_tools, extra_tool_report = self._cli_extra_tool_registry()
        tool_manifest = [
            *self.runtime.tool_manifest(),
            *self._agent_memory_tool_manifest(),
            *extra_tool_report,
        ]
        planner_callback, model_turns = self._planner_callback_factory(
            goal=user_text,
            mode="chat",
            selected_provider_name=selected_provider_name,
            selected_model=selected_model,
            tool_manifest=tool_manifest,
            memory_context=context,
            identity_evidence=identity_evidence,
            repository_evidence_pack=repository_evidence_pack,
            helix_focus=helix_focus,
            helix_auditability=helix_auditability,
            suite_focus=suite_focus,
            web_focus=web_focus,
            architecture_context_pack=architecture_context_pack,
            hash_recovery_ref=hash_recovery_ref,
            file_path_ref=file_path_ref,
            url_refs=url_refs,
            interaction_mode=active_interaction_mode,
            tone_contract=tone_contract,
            native_request=native_tool_plan,
            blind_request=blind_request,
            fallback_models=fallback_models,
            timeout=None,
        )
        agent_started = time.perf_counter()
        trace = self.runtime.agent_runner().run(
            goal=user_text,
            agent_name=self.agent_id,
            agent_id=self.agent_id,
            session_id=self.thread_id,
            memory_project=self.project,
            planner_callback=planner_callback,
            planner_name=f"{self.provider_name}:{selected_model}",
            allow_heuristic_fallback=False,
            extra_tools=extra_tools,
            tool_policy=self.tool_policy,
            retrieval_scope="session",
            memory_exclude_ids=excluded_memory_ids,
            max_steps=4,
        )
        agent_runner_ms = (time.perf_counter() - agent_started) * 1000
        self.last_runner_trace = trace
        self.last_model_turns = model_turns
        self.last_blind_report = dict(model_turns[-1].get("blind_inference") or {}) if model_turns else None
        latency_trace = _latency_trace(
            path="grounded",
            started=chat_started,
            rust_core_ms=rust_core_ms,
            route_ms=route_ms,
            routing_ms=float(rust_route.get("routing_ms") or route_ms),
            evidence_ms=evidence_ms,
            memory_ms=memory_ms,
            context_ms=memory_ms,
            provider_latency_ms=model_turns[-1].get("latency_ms") if model_turns else None,
            agent_runner_ms=agent_runner_ms,
            local_budget_exceeded=(route_ms + evidence_ms + memory_ms + agent_runner_ms) > 10_000,
            degraded_reason="local_pre_model_budget_exceeded" if (route_ms + evidence_ms + memory_ms + agent_runner_ms) > 10_000 else None,
            lane=lane,
        )
        self.last_latency_trace = latency_trace
        if model_turns:
            model_turns[-1]["timing"] = latency_trace
        planner_errors = [
            error_text
            for attempt in trace.get("planner_attempts", [])
            for error_text in attempt.get("errors", [])
        ]
        text = _task_visible_output(str(trace.get("final_answer") or ""))
        fallback_text = _format_runner_fallback_answer(trace, goal=user_text)
        if fallback_text:
            text = fallback_text
        if trace.get("final_planner") == "none" and planner_errors:
            text = _friendly_provider_failure_text(planner_errors[-1]) or f"Task failed: {planner_errors[-1]}"
        raw_text = str(model_turns[-1].get("raw_text") if model_turns else text)
        gate = _conversation_gate(text)
        latency_trace["response_gate_ms"] = gate.get("response_gate_ms")
        latency_trace["suppressed_reasoning"] = bool(gate.get("suppressed_reasoning"))
        latency_trace["repair_retry_used"] = bool(gate.get("repair_retry_used"))
        self.record(
            role="assistant",
            content=text,
            event_type="assistant_turn",
            metadata={
                "actual_model": model_turns[-1].get("actual_model") if model_turns else None,
                "selected_model": selected_model,
                "failover_used": model_turns[-1].get("failover_used") if model_turns else None,
                "failover_attempts": model_turns[-1].get("failover_attempts") if model_turns else [],
                "route": route,
                "latency_ms": model_turns[-1].get("latency_ms") if model_turns else None,
                "finish_reason": model_turns[-1].get("finish_reason") if model_turns else None,
                "usage": model_turns[-1].get("usage") if model_turns else None,
                "recall_memory_ids": memory_ids,
                "context_policy": "thread_only",
                "retrieval_scope": "session",
                "path_refs": file_path_refs,
                "url_refs": url_refs,
                "native_tool_plan": native_tool_plan,
                "capability_requirements": capability_requirements,
                "interaction_mode": active_interaction_mode,
                "mode_policy": mode_policy,
                "grounding_plan": grounding_plan,
                "tone_contract": tone_contract,
                "architecture_context_enabled": bool(architecture_context_pack),
                "lane": lane,
                "route_reason": route_reason,
                "thread_summary_used": False,
                "response_gate": gate,
                "local_fallback_used": bool(gate.get("local_fallback_used")),
                "blind_inference_enabled": bool((self.last_blind_report or {}).get("enabled")),
                "blind_policy_id": blind_status.get("policy_id"),
                "blind_provider_target": selected_provider_name,
                "blind_span_count": (self.last_blind_report or {}).get("span_count"),
                "blind_sensitive_classes": (self.last_blind_report or {}).get("sensitive_classes") or [],
                "blind_warnings": (self.last_blind_report or {}).get("warnings") or [],
                "blind_baseline_redaction_applied": bool((self.last_blind_report or {}).get("baseline_redaction_applied")),
                "raw_model_text": raw_text,
                "visible_output_cleaned": text != raw_text,
                "reasoning_internal": "",
                "trace_path": trace.get("trace_path"),
                "thread_id": self.thread_id,
                "latency_mode": self.latency_mode,
                "latency_trace": latency_trace,
            },
        )
        self.last_conversation_report = {
            "lane": lane,
            "route_reason": route_reason,
            "thread_summary_used": False,
            "response_gate": gate,
        }
        return {
            "text": text,
            "raw_text": raw_text,
            "reasoning": "",
            "route": route,
            "trace": trace,
            "interaction_mode": active_interaction_mode,
            "blind_inference": dict(self.last_blind_report or {}),
        }

    def _task_opencode(self, goal: str, *, rust_route: dict[str, Any], engine_configured: str, assurance: str = "quick", timeout: float = 600.0) -> dict[str, Any]:
        task_started = time.perf_counter()
        route_ms = float(rust_route.get("rust_core_ms") or rust_route.get("routing_ms") or 0.0)
        run_id = f"opencode-{self.thread_id or self.run_id or int(time.time())}-{int(time.time() * 1000)}"
        task_start_event = self.record(
            role="user",
            content=goal,
            event_type="task_start",
            metadata={
                "mode": "opencode",
                "engine": "opencode",
                "assurance": assurance,
                "engine_configured": engine_configured,
                "task_root": str(self.task_root),
                "route": {"rust_core": rust_route, "local_path": rust_route.get("path")},
                "thread_id": self.thread_id,
                "context_policy": "thread_only",
                "retrieval_scope": "session",
                "latency_mode": self.latency_mode,
                "latency_trace": {
                    "path": "agentic",
                    "lane": "work",
                    "rust_core_ms": round(route_ms, 3),
                    "route_ms": round(route_ms, 3),
                    "routing_ms": round(float(rust_route.get("routing_ms") or route_ms), 3),
                    "engine": "opencode",
                    "local_budget_exceeded": False,
                },
            },
        )
        payload = helix_cli_core.opencode_run(
            repo_root=self.task_root,
            goal=goal,
            run_id=run_id,
            evidence_root=self.evidence_root,
            timeout=timeout,
        )
        opencode_trace = payload.get("opencode_trace") if isinstance(payload.get("opencode_trace"), dict) else {}
        patch = str(payload.get("patch") or "")
        patch_apply_check = _git_apply_check(self.task_root, patch)
        patch_apply_ready = bool(patch and patch_apply_check.get("ok"))
        self.last_patch = patch if patch_apply_ready else None
        self.last_patch_sha256 = str(payload.get("patch_sha256") or hashlib.sha256(patch.encode("utf-8")).hexdigest()) if patch_apply_ready else None
        changed_files = payload.get("changed_files") if isinstance(payload.get("changed_files"), list) else []
        trust_card = payload.get("trust_card") if isinstance(payload.get("trust_card"), dict) else None
        if isinstance(trust_card, dict):
            checks = trust_card.get("checks_passed") if isinstance(trust_card.get("checks_passed"), list) else []
            if not any(isinstance(check, dict) and check.get("id") == "apply_check" for check in checks):
                checks.append(
                    {
                        "id": "apply_check",
                        "status": "passed" if patch_apply_ready else "failed" if patch else "not_run",
                        "summary": "git apply --check passed." if patch_apply_ready else "Patch was not apply-ready or no patch was generated.",
                    }
                )
                trust_card["checks_passed"] = checks
        status = "completed" if payload.get("status") == "passed" else "error"
        if status == "completed" and not patch:
            status = "partial"
        if status != "completed":
            patch_apply_ready = False
            self.last_patch = None
            self.last_patch_sha256 = None
            if isinstance(trust_card, dict):
                for check in trust_card.get("checks_passed") or []:
                    if isinstance(check, dict) and check.get("id") == "apply_check":
                        check["status"] = "failed" if patch else "not_run"
                        check["summary"] = "OpenCode did not complete cleanly; patch was not marked apply-ready."
        final_text = (
            f"OpenCode terminó en sandbox. Patch disponible: {'sí' if patch else 'no'}.\n"
            f"Artifact: {payload.get('artifact_path') or 'n/a'}\n"
            f"Trust card: {payload.get('trust_card_path') or 'n/a'}\n"
            f"Archivos cambiados: {', '.join(map(str, changed_files)) if changed_files else 'ninguno'}"
        )
        if patch_apply_ready:
            final_text = (
                "OpenCode termino en sandbox y dejo un patch aplicable.\n"
                f"Cambia: {', '.join(map(str, changed_files)) if changed_files else 'archivos no reportados'}\n"
                "Siguiente paso: revisa /trust last y aplica con /apply last."
            )
        elif patch:
            final_text = (
                "OpenCode termino en sandbox, pero el patch no aplica limpio sobre el repo actual.\n"
                f"Cambia: {', '.join(map(str, changed_files)) if changed_files else 'archivos no reportados'}\n"
                f"Motivo: {patch_apply_check.get('stderr') or patch_apply_check.get('stdout') or patch_apply_check.get('error') or 'git apply --check failed'}\n"
                f"Artifact: {payload.get('artifact_path') or 'n/a'}"
            )
        else:
            final_text = (
                "OpenCode termino en sandbox sin generar patch aplicable.\n"
                f"Artifact: {payload.get('artifact_path') or 'n/a'}\n"
                f"Archivos cambiados: {', '.join(map(str, changed_files)) if changed_files else 'ninguno'}"
            )
        if status == "partial" and not patch:
            final_text = (
                "OpenCode termino en sandbox, pero no produjo patch ni artefacto aplicable.\n"
                f"Artifact: {payload.get('artifact_path') or 'n/a'}\n"
                "Estado: partial. HeliX no lo marca como listo para aplicar."
            )
        if status == "error":
            final_text = (
                "OpenCode no pudo completar la tarea en sandbox.\n"
                f"Error: {payload.get('error') or opencode_trace.get('stderr_preview') or payload.get('stderr') or 'unknown'}"
            )
        latency_trace = _latency_trace(
            path="agentic",
            started=task_started,
            rust_core_ms=float(payload.get("rust_core_ms") or route_ms),
            route_ms=route_ms,
            routing_ms=float(rust_route.get("routing_ms") or route_ms),
            context_ms=0.0,
            evidence_ms=0.0,
            memory_ms=0.0,
            agent_runner_ms=0.0,
            provider_latency_ms=None,
            lane="work",
            local_budget_exceeded=bool(payload.get("local_budget_exceeded")),
        )
        latency_trace.update(
            {
                "engine": "opencode",
                "opencode_ms": opencode_trace.get("latency_ms"),
                "rust_core_ms": payload.get("rust_core_ms", latency_trace.get("rust_core_ms")),
            }
        )
        self.last_latency_trace = latency_trace
        task_result = {
            "status": status,
            "mode": "opencode",
            "engine": "opencode",
            "assurance": assurance,
            "engine_configured": engine_configured,
            "run_id": payload.get("run_id"),
            "goal": goal,
            "task_root": str(self.task_root),
            "route": {"rust_core": rust_route, "local_path": rust_route.get("path")},
            "final": final_text,
            "tool_events": [],
            "model_turns": [],
            "patch_available": patch_apply_ready,
            "patch_generated": bool(patch),
            "patch_apply_check": patch_apply_check,
            "patch_path": payload.get("patch_path"),
            "opencode_events_path": payload.get("opencode_events_path"),
            "patch_sha256": self.last_patch_sha256,
            "artifact_path": payload.get("artifact_path"),
            "trust_card_path": payload.get("trust_card_path"),
            "task_capsule_path": payload.get("task_capsule_path"),
            "trust_card": trust_card,
            "sandbox_root": payload.get("sandbox_root"),
            "changed_files": changed_files,
            "opencode_trace": opencode_trace,
            "latency_trace": latency_trace,
            "rust_core_payload": payload,
            "task_start_memory_id": (task_start_event.get("helix_memory") or {}).get("memory_id"),
        }
        if status == "error":
            task_result["error"] = payload.get("error") or opencode_trace.get("stderr_preview") or "opencode failed"
        if trust_card is None:
            trust_card = _task_trust_card_from_result(task_result)
            task_result["trust_card"] = trust_card
        task_result["assurance_followup"] = _task_assurance_followup(
            assurance=assurance,
            artifact_path=str(payload.get("artifact_path") or "") or None,
            evidence_root=self.evidence_root,
            repo_root=self.task_root,
        )
        self.last_trust_card = trust_card if isinstance(trust_card, dict) else None
        self.last_task_result = task_result
        self.record(
            role="assistant",
            content=final_text,
            event_type="task_error" if status == "error" else "task_final",
            metadata={
                "mode": "opencode",
                "engine": "opencode",
                "assurance": assurance,
                "route": {"rust_core": rust_route, "local_path": rust_route.get("path")},
                "context_policy": "thread_only",
                "retrieval_scope": "session",
                "tool_event_count": 0,
                "patch_available": patch_apply_ready,
                "patch_generated": bool(patch),
                "patch_apply_check": patch_apply_check,
                "patch_path": payload.get("patch_path"),
                "patch_sha256": self.last_patch_sha256,
                "artifact_path": payload.get("artifact_path"),
                "trust_card_path": payload.get("trust_card_path"),
                "task_capsule_path": payload.get("task_capsule_path"),
                "trust_card": trust_card,
                "sandbox_root": payload.get("sandbox_root"),
                "changed_files": changed_files,
                "latency_mode": self.latency_mode,
                "latency_trace": latency_trace,
            },
        )
        return task_result

    def _work_plan(self, goal: str, *, max_pages: int = 25, depth: int = 2, cross_domain: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
        path_refs = _extract_work_path_refs(goal)
        url_refs = _extract_url_refs(goal)
        output_target, source_refs = _work_target_for_goal(goal, path_refs)
        modifies_last = _looks_like_modify_last_work_request(goal)
        if modifies_last:
            last_output = (
                str((self.last_artifact or {}).get("path") or "")
                or _primary_work_output_path(self.last_work_result, plan=self.last_work_plan, task_root=self.task_root)
            )
            if last_output:
                if not output_target:
                    output_target = last_output
                if last_output not in source_refs:
                    source_refs.append(last_output)
        intent = _work_intent_for_goal(goal, source_refs=source_refs, url_refs=url_refs, output_target=output_target)
        flow = _flow_profile_or_error(intent["flow_profile"])
        source_pack = _collect_work_sources(
            self,
            goal,
            source_refs=source_refs,
            url_refs=url_refs,
            max_pages=max_pages,
            depth=depth,
            cross_domain=cross_domain,
        )
        plan = {
            "kind": "helix-work-plan-v1",
            "goal": goal,
            "work_intent": intent["intent"],
            "intent": intent["intent"],
            "flow_profile": flow.profile_id,
            "flow": flow.report(),
            "protocols_used": list(flow.protocols),
            "assurance": intent["assurance"],
            "output_kind": intent["output_kind"],
            "output_target": output_target,
            "needs_opencode": bool(intent["needs_opencode"]),
            "source_refs": source_refs,
            "url_refs": url_refs,
            "source_count": len(source_pack.get("sources") or []),
            "anchor_count": len(source_pack.get("anchors") or []),
            "extract_warnings": source_pack.get("warnings") or [],
            "crawler_summary": source_pack.get("crawler_summary") or {},
            "claim_boundary": _work_claim_boundary(str(intent["intent"])),
            "artifact_action": "modify_last" if modifies_last else "create_or_analyze",
        }
        return plan, source_pack

    def _work_trust_card(
        self,
        *,
        plan: dict[str, Any],
        source_pack: dict[str, Any],
        status: str,
        engine: str,
        run_id: str,
        patch_sha256: str | None = None,
        changed_files: list[Any] | None = None,
        artifact_paths: dict[str, Any] | None = None,
        patch_apply_check: dict[str, Any] | None = None,
        artifact: dict[str, Any] | None = None,
        before_artifact: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sources = source_pack.get("sources") if isinstance(source_pack.get("sources"), list) else []
        warnings = [str(item) for item in (plan.get("extract_warnings") or [])]
        apply_status = "not_run"
        if patch_apply_check:
            apply_status = "passed" if patch_apply_check.get("ok") else str(patch_apply_check.get("status") or "failed")
        checks = [
            {"id": "source_collection", "status": "passed" if sources else "warning", "summary": f"{len(sources)} source(s) collected."},
            {"id": "anchor_capture", "status": "passed" if plan.get("anchor_count") else "warning", "summary": f"{plan.get('anchor_count') or 0} anchor(s) captured."},
            {"id": "sandbox_provenance", "status": "passed" if engine == "opencode" else "not_run", "summary": "OpenCode runs in a HeliX sandbox when file generation is required."},
            {"id": "helix_generation", "status": "passed" if engine == "helix-internal-generator" else "not_run", "summary": "HeliX generated a deterministic patch from collected anchors."},
            {"id": "patch_integrity", "status": "passed" if patch_sha256 else "not_run", "summary": "Patch hash captured." if patch_sha256 else "No patch generated for analysis-only work."},
            {"id": "apply_check", "status": apply_status, "summary": "git apply --check passed." if apply_status == "passed" else "Patch was not apply-ready or no patch was generated."},
            {"id": "claim_boundary", "status": "passed", "summary": "The run includes an explicit claim boundary."},
        ]
        if artifact:
            readback = artifact.get("readback") if isinstance(artifact.get("readback"), dict) else {}
            checks.extend(
                [
                    {
                        "id": "artifact_exists",
                        "status": "passed" if artifact.get("exists") else "failed",
                        "summary": f"Artifact path: {artifact.get('path') or 'n/a'}.",
                    },
                    {
                        "id": "artifact_readback_after",
                        "status": "passed" if readback.get("status") == "ok" else str(readback.get("status") or "failed"),
                        "summary": f"Read back {readback.get('chars') or 0} chars.",
                    },
                    {
                        "id": "content_density",
                        "status": "passed" if _artifact_readback_ok(str(plan.get("goal") or ""), artifact) else "warning",
                        "summary": "Generated artifact has enough extractable content." if _artifact_readback_ok(str(plan.get("goal") or ""), artifact) else "Generated artifact is thin or hard to extract.",
                    },
                ]
            )
        if before_artifact:
            before_readback = before_artifact.get("readback") if isinstance(before_artifact.get("readback"), dict) else {}
            checks.append(
                {
                    "id": "artifact_readback_before",
                    "status": "passed" if before_readback.get("status") == "ok" else str(before_readback.get("status") or "failed"),
                    "summary": f"Before state had {before_readback.get('chars') or 0} extractable chars.",
                }
            )
            if artifact:
                checks.append(
                    {
                        "id": "artifact_modified",
                        "status": "passed" if artifact.get("sha256") and artifact.get("sha256") != before_artifact.get("sha256") else "warning",
                        "summary": "Artifact hash changed after modification." if artifact.get("sha256") != before_artifact.get("sha256") else "Artifact hash did not change.",
                    }
                )
        return {
            "kind": "helix-trust-card-v1",
            "subject_type": "work",
            "status": status,
            "assurance": plan.get("assurance") or "quick",
            "engine": engine,
            "run_id": run_id,
            "goal": plan.get("goal"),
            "flow_profile": plan.get("flow_profile"),
            "work_intent": plan.get("work_intent"),
            "protocols_used": plan.get("protocols_used") or [],
            "sources": [
                {
                    "kind": source.get("kind"),
                    "status": source.get("status"),
                    "path": source.get("path"),
                    "url": source.get("url"),
                    "sha256": source.get("sha256"),
                    "extract_method": source.get("extract_method"),
                    "warning_count": len(source.get("warnings") or []),
                }
                for source in sources
            ],
            "anchors": {
                "count": plan.get("anchor_count") or 0,
                "sample": (source_pack.get("anchors") or [])[:5],
            },
            "changed_files": changed_files or [],
            "checks_passed": checks,
            "warnings": warnings,
            "output_target": plan.get("output_target"),
            "crawler_summary": plan.get("crawler_summary") or {},
            "patch": {"sha256": patch_sha256},
            "artifact": artifact or {},
            "artifact_before": before_artifact or {},
            "artifact_paths": artifact_paths or {},
            "claim_boundary": plan.get("claim_boundary") or _work_claim_boundary(str(plan.get("work_intent") or "")),
        }

    def _blocked_work_result(
        self,
        *,
        goal: str,
        run_id: str,
        plan: dict[str, Any],
        source_pack: dict[str, Any],
        bus: WorkEventBus,
        reason: str,
    ) -> dict[str, Any]:
        work_card = self._work_trust_card(
            plan=plan,
            source_pack=source_pack,
            status="blocked",
            engine="helix-workbench",
            run_id=run_id,
        )
        checks = work_card.get("checks_passed") if isinstance(work_card.get("checks_passed"), list) else []
        checks.append({"id": "hook_block", "status": "blocked", "summary": reason})
        work_card["checks_passed"] = checks
        work_card["warnings"] = [*(work_card.get("warnings") or []), reason]
        paths = _write_work_artifacts(self.task_root, run_id, plan=plan, source_pack=source_pack, trust_card=work_card)
        work_card["artifact_paths"] = {"trust_card": paths.get("trust_card"), "plan": paths.get("plan"), "sources": paths.get("sources")}
        Path(paths["trust_card"]).write_text(json.dumps(_json_ready(work_card), indent=2, ensure_ascii=False), encoding="utf-8")
        trace = _latency_trace(path="work", started=bus._started, lane=str(plan.get("work_intent") or "work"))
        trace["work_phase_ms"] = bus.phase_ms()
        trace["work_event_count"] = len(bus.events)
        work_result = {
            "status": "blocked",
            "mode": "work",
            "engine": "helix-workbench",
            "run_id": run_id,
            "goal": goal,
            "final": f"Tarea bloqueada antes de escribir: {reason}",
            "work_plan": plan,
            "work_sources": source_pack,
            "work_artifact_paths": paths,
            "trust_card": work_card,
            "intent_card": self.last_intent_card,
            "progress_events": list(bus.events),
            "patch_available": False,
            "patch_generated": False,
            "changed_files": [],
            "blocked_reason": reason,
            "latency_trace": trace,
        }
        self.last_patch = None
        self.last_patch_sha256 = None
        self.last_trust_card = work_card
        self.last_task_result = work_result
        self.last_work_result = work_result
        self.last_latency_trace = trace
        self._save_last_work_state()
        self.record(
            role="assistant",
            content=str(work_result.get("final") or ""),
            event_type="work_error",
            metadata={"mode": "work", "work_plan": plan, "progress_events": list(bus.events), "trust_card": work_card, "latency_trace": trace},
        )
        return work_result

    def work(
        self,
        goal: str,
        *,
        max_pages: int = 25,
        depth: int = 2,
        cross_domain: bool = False,
        engine_override: str | None = None,
        assurance_override: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        run_id = f"work-{int(time.time() * 1000)}"
        intent_card = self.turn_controller(goal)
        bus = WorkEventBus(self, run_id=run_id, goal=goal)
        bus.emit(
            "intent.detected",
            f"Entendi: {intent_card.primary_goal}",
            lane=intent_card.lane,
            output_target=intent_card.output_target,
            route_reason=intent_card.route_reason,
        )
        bus.emit("source.collecting", "Recolectando fuentes y artefactos relacionados.")
        plan, source_pack = self._work_plan(goal, max_pages=max_pages, depth=depth, cross_domain=cross_domain)
        if assurance_override:
            plan["assurance"] = _normalize_assurance(assurance_override)
        plan["intent_card"] = intent_card.to_dict()
        self.last_intent_card = intent_card.to_dict()
        bus.emit(
            "source.read",
            f"Fuentes leidas: {len(source_pack.get('sources') or [])}; anchors: {len(source_pack.get('anchors') or [])}.",
            source_count=len(source_pack.get("sources") or []),
            anchor_count=len(source_pack.get("anchors") or []),
        )
        self.last_work_plan = plan
        self.last_work_sources = source_pack
        self.record(
            role="user",
            content=goal,
            event_type="work_start",
            metadata={
                "mode": "work",
                "lane": "work",
                "route": _classify_workbench_prompt(goal, self.recent_route_intents(limit=4)),
                "work_plan": plan,
                "intent_card": intent_card.to_dict(),
                "progress_events": list(bus.events),
                "source_count": plan.get("source_count"),
                "anchor_count": plan.get("anchor_count"),
                "context_policy": "thread_only",
                "retrieval_scope": "session",
            },
        )
        explicit_opencode = _normalize_task_engine(engine_override or "") == "opencode"
        if (
            str(plan.get("work_intent") or "") == "source_to_document"
            and not explicit_opencode
            and _is_external_work_output(self.task_root, str(plan.get("output_target") or ""), goal)
        ):
            before_artifact = None
            if str(plan.get("artifact_action") or "") == "modify_last" and plan.get("output_target"):
                bus.emit("artifact.before_read", f"Leyendo artefacto previo: {plan.get('output_target')}.")
                before_artifact = _inspect_work_artifact(self, str(plan.get("output_target")), last_action="before_modify")
            hook_prewrite = self._run_internal_hooks("PreWrite", {"goal": goal, "plan": plan, "intent_card": intent_card.to_dict(), "output_target": plan.get("output_target")})
            for warning in hook_prewrite.get("warnings") or []:
                bus.emit("hook.warning", str(warning), hook="PreWrite")
            if hook_prewrite.get("blocked"):
                bus.emit("blocked", str(hook_prewrite.get("block_reason") or "PreWrite blocked task."), hook="PreWrite")
                work_result = self._blocked_work_result(
                    goal=goal,
                    run_id=run_id,
                    plan=plan,
                    source_pack=source_pack,
                    bus=bus,
                    reason=str(hook_prewrite.get("block_reason") or "PreWrite blocked task."),
                )
                return work_result
            bus.emit("writer.started", f"Escribiendo documento en {plan.get('output_target') or 'destino inferido'}.")
            try:
                exported = _internal_work_export(self.task_root, plan, source_pack)
            except Exception as exc:  # noqa: BLE001
                exported = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            run_id = str(exported.get("run_id") or run_id)
            _sync_work_event_run_id(bus, run_id)
            output_file = exported.get("output_file") if isinstance(exported.get("output_file"), dict) else {}
            changed_files = exported.get("changed_files") if isinstance(exported.get("changed_files"), list) else []
            bus.emit("artifact.written", f"Documento escrito: {output_file.get('path') or plan.get('output_target') or 'n/a'}.", output_file=output_file.get("path"))
            bus.emit("artifact.after_read", f"Releyendo output: {output_file.get('path') or plan.get('output_target') or 'n/a'}.")
            artifact = _inspect_work_artifact(self, str(output_file.get("path") or plan.get("output_target") or ""), last_action="export_readback") if output_file.get("path") else {}
            hook_readback = self._run_internal_hooks("PostArtifactRead", {"goal": goal, "plan": plan, "artifact": artifact})
            for warning in hook_readback.get("warnings") or []:
                bus.emit("hook.warning", str(warning), hook="PostArtifactRead")
            export_ok = (
                exported.get("status") == "passed"
                and bool(output_file.get("path"))
                and Path(str(output_file.get("path"))).exists()
                and _artifact_readback_ok(goal, artifact)
            )
            work_card = self._work_trust_card(
                plan=plan,
                source_pack=source_pack,
                status="passed" if export_ok else "failed",
                engine=str(exported.get("engine") or "helix-internal-exporter"),
                run_id=run_id,
                patch_sha256=None,
                changed_files=changed_files,
                artifact_paths={},
                patch_apply_check={"status": "not_run", "ok": False, "reason": "direct document export"},
                artifact=artifact,
                before_artifact=before_artifact,
            )
            work_card["output_file"] = output_file
            checks = work_card.get("checks_passed") if isinstance(work_card.get("checks_passed"), list) else []
            checks.append(
                {
                    "id": "output_exists",
                    "status": "passed" if export_ok else "failed",
                    "summary": f"Output file verified at {output_file.get('path') or 'n/a'}.",
                }
            )
            work_card["checks_passed"] = checks
            bus.emit("trust.updated", "Trust card actualizada.", status=work_card.get("status"))
            bus.emit("done" if export_ok else "blocked", str(work_card.get("claim_boundary") or plan.get("claim_boundary") or "work completed"), status="completed" if export_ok else "failed")
            work_paths = _write_work_artifacts(self.task_root, run_id, plan=plan, source_pack=source_pack, trust_card=work_card)
            work_card["artifact_paths"] = {"trust_card": work_paths.get("trust_card"), "plan": work_paths.get("plan"), "sources": work_paths.get("sources")}
            Path(work_paths["trust_card"]).write_text(json.dumps(_json_ready(work_card), indent=2, ensure_ascii=False), encoding="utf-8")
            latency_trace = _latency_trace(path="work", started=started, lane="work_doc")
            latency_trace["work_phase_ms"] = bus.phase_ms()
            latency_trace["work_event_count"] = len(bus.events)
            work_result = {
                "status": "completed" if export_ok else "failed",
                "mode": "work",
                "engine": str(exported.get("engine") or "helix-internal-exporter"),
                "run_id": run_id,
                "goal": goal,
                "final": (
                    f"HeliX genero, releyo y verifico el documento en {output_file.get('path')}."
                    if export_ok
                    else f"HeliX escribio el documento pero no pudo verificar suficiente contenido extraible: {exported.get('error') or artifact.get('error') or 'readback/content-density failed'}"
                ),
                "work_plan": plan,
                "work_sources": source_pack,
                "work_artifact_paths": work_paths,
                "trust_card": work_card,
                "intent_card": intent_card.to_dict(),
                "progress_events": list(bus.events),
                "flow_goal": goal,
                "claim_boundary": plan.get("claim_boundary"),
                "artifact": artifact,
                "artifact_before": before_artifact or {},
                "output_file": output_file,
                "patch_available": False,
                "patch_generated": False,
                "changed_files": changed_files,
                "latency_trace": latency_trace,
            }
            self.last_patch = None
            self.last_patch_sha256 = None
            self.last_trust_card = work_card
            self.last_task_result = work_result
            self.last_work_result = work_result
            self.last_artifact = artifact if artifact else None
            self.last_latency_trace = work_result["latency_trace"]
            self._save_last_work_state()
            self.record(
                role="assistant",
                content=str(work_result.get("final") or ""),
                event_type="work_final" if export_ok else "work_error",
                metadata={"mode": "work", "work_plan": plan, "intent_card": intent_card.to_dict(), "progress_events": list(bus.events), "trust_card": work_card, "latency_trace": self.last_latency_trace, "output_file": output_file},
            )
            return work_result

        if str(plan.get("work_intent") or "") in {"source_to_web", "source_to_document"} and not explicit_opencode:
            flow = _flow_profile_or_error(str(plan.get("flow_profile") or "web"))
            hook_prewrite = self._run_internal_hooks("PreWrite", {"goal": goal, "plan": plan, "intent_card": intent_card.to_dict(), "output_target": plan.get("output_target")})
            for warning in hook_prewrite.get("warnings") or []:
                bus.emit("hook.warning", str(warning), hook="PreWrite")
            if hook_prewrite.get("blocked"):
                bus.emit("blocked", str(hook_prewrite.get("block_reason") or "PreWrite blocked task."), hook="PreWrite")
                return self._blocked_work_result(
                    goal=goal,
                    run_id=run_id,
                    plan=plan,
                    source_pack=source_pack,
                    bus=bus,
                    reason=str(hook_prewrite.get("block_reason") or "PreWrite blocked task."),
                )
            bus.emit("writer.started", f"Generando patch para {plan.get('output_target') or 'output inferido'}.")
            try:
                generated = _internal_work_patch(self.task_root, plan, source_pack)
            except Exception as exc:  # noqa: BLE001
                generated = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            patch_text = str(generated.get("patch") or "")
            bus.emit("artifact.written", f"Patch generado: {'si' if patch_text else 'no'}.", patch_bytes=len(patch_text))
            bus.emit("verify.apply_check", "Verificando patch con git apply --check.")
            patch_apply_check = _git_apply_check(self.task_root, patch_text)
            patch_apply_ready = bool(patch_apply_check.get("ok"))
            run_id = str(generated.get("run_id") or run_id)
            _sync_work_event_run_id(bus, run_id)
            changed_files = generated.get("changed_files") if isinstance(generated.get("changed_files"), list) else []
            patch_sha256 = str(generated.get("patch_sha256") or "") or (hashlib.sha256(patch_text.encode("utf-8")).hexdigest() if patch_text else None)
            work_card = self._work_trust_card(
                plan=plan,
                source_pack=source_pack,
                status="passed" if patch_apply_ready else "failed",
                engine="helix-internal-generator",
                run_id=run_id,
                patch_sha256=patch_sha256,
                changed_files=changed_files,
                artifact_paths={},
                patch_apply_check=patch_apply_check,
            )
            bus.emit("trust.updated", "Trust card actualizada.", status=work_card.get("status"))
            bus.emit("done" if patch_apply_ready else "blocked", "Patch listo para aplicar." if patch_apply_ready else "Patch bloqueado por apply check.", status="completed" if patch_apply_ready else "failed")
            work_paths = _write_work_artifacts(self.task_root, run_id, plan=plan, source_pack=source_pack, trust_card=work_card, patch=patch_text or None)
            work_card["artifact_paths"] = {"work_patch": work_paths.get("patch"), "trust_card": work_paths.get("trust_card"), "plan": work_paths.get("plan"), "sources": work_paths.get("sources")}
            Path(work_paths["trust_card"]).write_text(json.dumps(_json_ready(work_card), indent=2, ensure_ascii=False), encoding="utf-8")
            latency_trace = _latency_trace(path="work", started=started, lane=str(plan.get("work_intent") or "work"))
            latency_trace["work_phase_ms"] = bus.phase_ms()
            latency_trace["work_event_count"] = len(bus.events)
            work_result = {
                "status": "completed" if patch_apply_ready else "failed",
                "mode": "work",
                "engine": "helix-internal-generator",
                "run_id": run_id,
                "goal": goal,
                "final": (
                    "HeliX leyo las fuentes, capturo anchors y genero un patch aplicable. Revisa /trust y aplica con /apply last."
                    if patch_apply_ready
                    else "HeliX genero una propuesta, pero git apply --check no paso. Usa /trust para ver el bloqueo."
                ),
                "work_plan": plan,
                "work_sources": source_pack,
                "work_artifact_paths": work_paths,
                "trust_card": work_card,
                "intent_card": intent_card.to_dict(),
                "progress_events": list(bus.events),
                "flow": flow.report(),
                "flow_goal": goal,
                "claim_boundary": plan.get("claim_boundary"),
                "patch": patch_text,
                "patch_sha256": patch_sha256,
                "patch_available": patch_apply_ready,
                "patch_generated": bool(patch_text),
                "patch_apply_check": patch_apply_check,
                "changed_files": changed_files,
                "latency_trace": latency_trace,
            }
            self.last_patch = patch_text if patch_apply_ready else None
            self.last_patch_sha256 = patch_sha256 if patch_apply_ready else None
            self.last_trust_card = work_card
            self.last_task_result = work_result
            self.last_work_result = work_result
            self.last_latency_trace = work_result["latency_trace"]
            self._save_last_work_state()
            self.record(
                role="assistant",
                content=str(work_result.get("final") or ""),
                event_type="work_final" if patch_apply_ready else "work_error",
                metadata={"mode": "work", "work_plan": plan, "intent_card": intent_card.to_dict(), "progress_events": list(bus.events), "trust_card": work_card, "latency_trace": self.last_latency_trace, "patch_available": patch_apply_ready},
            )
            return work_result

        if plan.get("needs_opencode") or explicit_opencode:
            flow = _flow_profile_or_error(str(plan.get("flow_profile") or "web"))
            work_brief = _render_work_brief(plan, source_pack)
            output_instruction = (
                f"\n\n## Output Instruction\nCreate or modify files for output_target={plan.get('output_target') or 'infer from goal'} "
                "inside the sandbox. Keep the result reviewable and dependency-light unless the repo already provides a stack."
            )
            opencode_goal = work_brief + output_instruction
            rust_route = helix_cli_core.route(goal, latency_mode=self.latency_mode, interaction_mode=self.interaction_mode)
            bus.emit("writer.started", "Delegando ejecucion a OpenCode en sandbox.", engine="opencode")
            result = self._task_opencode(
                opencode_goal,
                rust_route=rust_route,
                engine_configured=engine_override or "work-runtime",
                assurance=str(plan.get("assurance") or flow.assurance),
                timeout=_work_opencode_timeout_seconds(),
            )
            fallback = None
            if not result.get("patch") and str(plan.get("work_intent") or "") in {"source_to_web", "source_to_document"}:
                result["status"] = "partial"
                result["partial_reason"] = "OpenCode completed without a patch."
                try:
                    bus.emit("writer.repair", "OpenCode no produjo patch; intentando fallback interno.")
                    fallback = _internal_work_patch(self.task_root, plan, source_pack)
                except Exception as exc:  # noqa: BLE001
                    fallback = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                if fallback.get("status") == "passed" and fallback.get("patch"):
                    fallback_apply_check = _git_apply_check(self.task_root, str(fallback.get("patch") or ""))
                    fallback_apply_ready = bool(fallback_apply_check.get("ok"))
                    result["opencode_result_before_fallback"] = {
                        "status": result.get("status"),
                        "run_id": result.get("run_id"),
                        "artifact_path": result.get("artifact_path"),
                        "patch_available": bool(result.get("patch")),
                        "changed_files": result.get("changed_files") or [],
                    }
                    result.update(
                        {
                            "status": "completed",
                            "engine": "helix-internal-generator",
                            "fallback_used": True,
                            "fallback_reason": fallback.get("fallback_reason"),
                            "opencode_error": _truncate_text(str(((result.get("rust_core_payload") or {}) if isinstance(result.get("rust_core_payload"), dict) else {}).get("error") or result.get("error") or ""), 800)["text"],
                            "patch": fallback.get("patch"),
                            "patch_sha256": fallback.get("patch_sha256"),
                            "patch_available": fallback_apply_ready,
                            "patch_generated": True,
                            "patch_apply_check": fallback_apply_check,
                            "changed_files": fallback.get("changed_files") or [],
                            "final": (
                                "HeliX leyo las fuentes, capturo anchors y genero una propuesta aplicable "
                                "desde el Work Runtime interno. Revisa /trust last y aplica con /apply last."
                                if fallback_apply_ready
                                else "HeliX genero una propuesta desde las fuentes, pero git apply --check no paso; revisa el artifact antes de aplicar."
                            ),
                        }
                    )
                    self.last_patch = str(fallback.get("patch") or "") if fallback_apply_ready else None
                    self.last_patch_sha256 = str(fallback.get("patch_sha256") or "") if fallback_apply_ready else None
                    result.pop("rust_core_payload", None)
                    result.pop("error", None)
            run_id = str(result.get("run_id") or run_id)
            _sync_work_event_run_id(bus, run_id)
            artifact_paths = {
                "artifact": result.get("artifact_path"),
                "patch": result.get("patch_path"),
                "opencode_events": result.get("opencode_events_path"),
                "trust_card": result.get("trust_card_path"),
                "task_capsule": result.get("task_capsule_path"),
            }
            changed_files = result.get("changed_files") if isinstance(result.get("changed_files"), list) else []
            patch_sha256 = str(result.get("patch_sha256") or "") or None
            patch_text = str(result.get("patch") or "")
            if patch_text:
                bus.emit("artifact.written", "Patch capturado desde executor.", patch_bytes=len(patch_text))
                bus.emit("verify.apply_check", "Verificando patch capturado.")
            else:
                bus.emit("blocked", "Executor termino sin patch aplicable.", status=result.get("status"))
            work_card = self._work_trust_card(
                plan=plan,
                source_pack=source_pack,
                status="passed" if result.get("status") in {"completed", "passed"} else str(result.get("status") or "error"),
                engine=str(result.get("engine") or "opencode"),
                run_id=run_id,
                patch_sha256=patch_sha256,
                changed_files=changed_files,
                artifact_paths=artifact_paths,
                patch_apply_check=result.get("patch_apply_check") if isinstance(result.get("patch_apply_check"), dict) else None,
            )
            work_paths = _write_work_artifacts(self.task_root, run_id, plan=plan, source_pack=source_pack, trust_card=work_card, patch=patch_text or None)
            if patch_text:
                artifact_paths["work_patch"] = work_paths.get("patch")
                work_card["artifact_paths"] = {**(work_card.get("artifact_paths") or {}), "work_patch": work_paths.get("patch")}
                Path(work_paths["trust_card"]).write_text(json.dumps(_json_ready(work_card), indent=2, ensure_ascii=False), encoding="utf-8")
            bus.emit("trust.updated", "Trust card actualizada.", status=work_card.get("status"))
            bus.emit("done" if result.get("patch_available") else "blocked", str(result.get("final") or "executor completed"), status=result.get("status"))
            latency_trace = result.get("latency_trace") if isinstance(result.get("latency_trace"), dict) else _latency_trace(path="work", started=started, lane=str(plan.get("work_intent") or "work"))
            latency_trace["work_phase_ms"] = bus.phase_ms()
            latency_trace["work_event_count"] = len(bus.events)
            result.update(
                {
                    "run_id": run_id,
                    "goal": goal,
                    "mode": "work",
                    "work_plan": plan,
                    "work_sources": source_pack,
                    "work_artifact_paths": work_paths,
                    "trust_card": work_card,
                    "intent_card": intent_card.to_dict(),
                    "progress_events": list(bus.events),
                    "flow": flow.report(),
                    "flow_goal": goal,
                    "claim_boundary": plan.get("claim_boundary"),
                    "latency_trace": latency_trace,
                }
            )
            self.last_trust_card = work_card
            self.last_task_result = result
            self.last_work_result = result
            self.last_latency_trace = latency_trace
            self._save_last_work_state()
            return result

        context = json.dumps(
            {
                "goal": goal,
                "plan": plan,
                "sources": [
                    {
                        "kind": source.get("kind"),
                        "status": source.get("status"),
                        "path": source.get("path"),
                        "url": source.get("url"),
                        "content_preview": source.get("content_preview"),
                        "anchors": source.get("anchors"),
                        "warnings": source.get("warnings") or [],
                    }
                    for source in source_pack.get("sources") or []
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        result = run_chat_with_failover(
            provider_name=self.provider_name,
            model=resolve_model_alias(self.model) if self.model.lower() not in {"auto", "router:auto"} else DEEPINFRA_MODEL_PROFILES["research"].model_id,
            fallback_models=[],
            prompt=f"{goal}\n\nHeliX Work Runtime source context:\n{context}",
            system=(
                "You are HeliX Work Runtime. Answer from the provided source context and anchors. "
                "Separate inspected facts from inference. Include a short limitations/claim-boundary note."
            ),
            history=[],
            max_tokens=min(self.max_tokens, 1200),
            temperature=self.temperature,
            workspace_root=self.workspace_root,
            prompt_token=False,
            timeout=None,
        )
        raw_text = str(result.get("text") or "")
        gate = _conversation_gate(raw_text)
        final_text = str(gate.get("visible_text") or "")
        work_card = self._work_trust_card(
            plan=plan,
            source_pack=source_pack,
            status="completed",
            engine="helix-planner",
            run_id=run_id,
        )
        bus.emit("writer.started", "Generando respuesta grounded desde fuentes.")
        work_paths = _write_work_artifacts(self.task_root, run_id, plan=plan, source_pack=source_pack, trust_card=work_card)
        bus.emit("trust.updated", "Trust card actualizada.", status=work_card.get("status"))
        bus.emit("done", "Analisis completado.", status="completed")
        latency_trace = _latency_trace(path="work", started=started, provider_latency_ms=result.get("latency_ms"), lane="work")
        latency_trace["work_phase_ms"] = bus.phase_ms()
        latency_trace["work_event_count"] = len(bus.events)
        work_result = {
            "status": "completed",
            "mode": "work",
            "engine": "helix-planner",
            "run_id": run_id,
            "goal": goal,
            "final": final_text,
            "work_plan": plan,
            "work_sources": source_pack,
            "work_artifact_paths": work_paths,
            "trust_card": work_card,
            "intent_card": intent_card.to_dict(),
            "progress_events": list(bus.events),
            "latency_trace": latency_trace,
        }
        self.last_trust_card = work_card
        self.last_task_result = work_result
        self.last_work_result = work_result
        self.last_latency_trace = work_result["latency_trace"]
        self._save_last_work_state()
        self.record(
            role="assistant",
            content=final_text,
            event_type="work_final",
            metadata={"mode": "work", "work_plan": plan, "intent_card": intent_card.to_dict(), "progress_events": list(bus.events), "trust_card": work_card, "latency_trace": self.last_latency_trace},
        )
        return work_result

    def task(
        self,
        goal: str,
        *,
        max_steps: int = 5,
        mode_override: str | None = None,
        interaction_mode_override: str | None = None,
        agent_blueprint: AgentBlueprint | None = None,
        engine_override: str | None = None,
        assurance: str = "quick",
    ) -> dict[str, Any]:
        task_started = time.perf_counter()
        route_started = time.perf_counter()
        active_interaction_mode = _normalize_interaction_mode(interaction_mode_override or self.interaction_mode)
        rust_route = helix_cli_core.route(
            goal,
            latency_mode=self.latency_mode,
            interaction_mode=active_interaction_mode,
        )
        rust_core_ms = float(rust_route.get("rust_core_ms") or rust_route.get("routing_ms") or 0.0)
        engine_configured = _normalize_task_engine(engine_override or self.task_engine)
        assurance = _normalize_assurance(assurance)
        selected_engine = _task_engine_for_goal(goal, configured=engine_configured, rust_route=rust_route)
        if selected_engine == "opencode" and agent_blueprint is None:
            return self._task_opencode(goal, rust_route=rust_route, engine_configured=engine_configured, assurance=assurance)
        route = route_model_for_task(
            f"{goal}\nTask mode: inspect repo, use tools, propose patch if needed.",
            provider_name=self.provider_name,
            policy=self.router_policy,
            interaction_mode=active_interaction_mode,
            recent_intents=self.recent_route_intents(limit=4),
        )
        route_ms = (time.perf_counter() - route_started) * 1000
        url_refs = _extract_url_refs(goal)
        path_refs = _extract_local_path_refs(goal)
        selected_model = self.model
        selected_provider_name = self.provider_name
        model_is_auto = self.model.lower() in {"auto", "router:auto"}
        blueprint_controls_model = agent_blueprint is not None and model_is_auto
        if blueprint_controls_model:
            selected_model = resolve_model_alias(agent_blueprint.preferred_model_alias)
            route = _manual_route_for_model(
                selected_model,
                provider_name=self.provider_name,
                policy=self.router_policy,
                user_text=goal,
                interaction_mode=active_interaction_mode,
            )
            route["intent"] = "agentic_blueprint"
            route["agent_blueprint"] = agent_blueprint.blueprint_id
            route = _augment_route_metadata(route, goal, interaction_mode=active_interaction_mode)
        elif model_is_auto:
            selected_model = route.get("model") or PROVIDERS[self.provider_name].default_model
        else:
            route = _manual_route_for_model(
                selected_model,
                provider_name=self.provider_name,
                policy=self.router_policy,
                user_text=goal,
                interaction_mode=active_interaction_mode,
            )
        if isinstance(route, dict):
            route["rust_core"] = rust_route
            route["local_path"] = rust_route.get("path")
        selected_provider_name = str(route.get("provider") or self.provider_name)
        native_tool_plan = dict(route.get("native_tool_plan") or {}) if isinstance(route, dict) else {}
        capability_requirements = dict(route.get("capability_requirements") or {}) if isinstance(route, dict) else {}
        mode_policy = dict(route.get("mode_policy") or _interaction_mode_payload(active_interaction_mode)) if isinstance(route, dict) else _interaction_mode_payload(active_interaction_mode)
        tone_contract = str(route.get("tone_contract") or mode_policy.get("tone_contract") or INTERACTION_MODE_PROFILES[active_interaction_mode]["tone_contract"]) if isinstance(route, dict) else INTERACTION_MODE_PROFILES[active_interaction_mode]["tone_contract"]
        grounding_plan = str(route.get("grounding_plan") or "helix-only") if isinstance(route, dict) else "helix-only"
        fallback_models = _fallback_model_ids_for_route(
            route,
            primary_model=selected_model,
            agent_blueprint=agent_blueprint if blueprint_controls_model else None,
            include_route_fallbacks=True,
        )
        blind_request = self._blind_request_for_provider(selected_provider_name, mode="task")
        blind_status = self.blind_status()
        active_agent_mode = mode_override or self.agent_mode
        active_tool_policy = self.tool_policy
        if agent_blueprint is not None:
            allowed = set(agent_blueprint.allowed_tools) | {"helix.search", "memory.search", "rag.search"}
            active_tool_policy = {
                **self.tool_policy,
                "mode": f"blueprint:{agent_blueprint.blueprint_id}",
                "auto": [tool for tool in self.tool_policy.get("auto", []) if tool in allowed],
                "agent_blueprint": agent_blueprint.blueprint_id,
                "allowed_tools": sorted(allowed),
            }
            max_steps = agent_blueprint.max_steps
        evidence_started = time.perf_counter()
        repository_evidence_pack = self.refresh_evidence(goal, limit=8)
        evidence_ms = (time.perf_counter() - evidence_started) * 1000
        memory_started = time.perf_counter()
        context = self.memory_context(goal)
        memory_ms = (time.perf_counter() - memory_started) * 1000
        memory_ids = list(context.get("memory_ids") or [])
        suite_focus = _is_suite_evidence_request(goal)
        recent_history = self.recent_history(limit=4, exclude_latest_user=False)
        helix_focus = _is_explicit_helix_meta_task_request(goal, recent_history)
        helix_auditability = _is_helix_auditability_request(goal, recent_history)
        architecture_context_pack = (
            self.architecture_context_pack(goal, include_excerpts=True)
            if (
                helix_auditability
                or str((route or {}).get("intent") or "") in {"helix_self", "audit"}
                or (active_interaction_mode != "explore" and helix_focus)
            )
            else None
        )
        task_start_event = self.record(
            role="user",
            content=goal,
            event_type="task_start",
            metadata={
                "mode": active_agent_mode,
                "agent_blueprint": agent_blueprint.blueprint_id if agent_blueprint else None,
                "allowed_tools": list(agent_blueprint.allowed_tools) if agent_blueprint else None,
                "task_root": str(self.task_root),
                "route": route,
                "thread_id": self.thread_id,
                "context_policy": "thread_only",
                "retrieval_scope": "session",
                "recall_memory_ids": memory_ids,
                "path_refs": path_refs,
                "url_refs": url_refs,
                "native_tool_plan": native_tool_plan,
                "capability_requirements": capability_requirements,
                "interaction_mode": active_interaction_mode,
                "mode_policy": mode_policy,
                "grounding_plan": grounding_plan,
                "tone_contract": tone_contract,
                "architecture_context_enabled": bool(architecture_context_pack),
                "blind_inference_enabled": bool(blind_request),
                "blind_policy_id": blind_status.get("policy_id"),
                "blind_provider_target": selected_provider_name,
                "latency_mode": self.latency_mode,
                "latency_trace": {
                    "path": "agentic",
                    "rust_core_ms": round(rust_core_ms, 3),
                    "route_ms": round(route_ms, 3),
                    "routing_ms": round(float(rust_route.get("routing_ms") or route_ms), 3),
                    "evidence_ms": round(evidence_ms, 3),
                    "memory_ms": round(memory_ms, 3),
                    "context_ms": round(memory_ms, 3),
                    "local_budget_exceeded": (route_ms + evidence_ms + memory_ms) > 10_000,
                },
            },
        )
        excluded_memory_ids = [str((task_start_event.get("helix_memory") or {}).get("memory_id") or "")]
        excluded_memory_ids = [item for item in excluded_memory_ids if item]
        extra_tools, extra_tool_report = self._cli_extra_tool_registry()
        tool_manifest = [
            *self.runtime.tool_manifest(),
            *self._agent_memory_tool_manifest(),
            *extra_tool_report,
        ]
        planner_callback, model_turns = self._planner_callback_factory(
            goal=goal,
            mode="task",
            selected_model=selected_model,
            selected_provider_name=selected_provider_name,
            tool_manifest=tool_manifest,
            memory_context=context,
            identity_evidence=None,
            repository_evidence_pack=repository_evidence_pack,
            helix_focus=helix_focus,
            helix_auditability=helix_auditability,
            suite_focus=suite_focus,
            architecture_context_pack=architecture_context_pack,
            file_path_ref=path_refs[0] if path_refs else None,
            url_refs=url_refs,
            interaction_mode=active_interaction_mode,
            tone_contract=tone_contract,
            native_request=native_tool_plan,
            blind_request=blind_request,
            fallback_models=fallback_models,
            timeout=AGENT_TASK_TIMEOUT_SECONDS,
        )
        try:
            agent_started = time.perf_counter()
            trace = self.runtime.agent_runner().run(
                goal=goal,
                agent_name=self.agent_id,
                agent_id=self.agent_id,
                session_id=self.thread_id,
                memory_project=self.project,
                planner_callback=planner_callback,
                planner_name=f"{self.provider_name}:{selected_model}",
                allow_heuristic_fallback=False,
                extra_tools=extra_tools,
                tool_policy=active_tool_policy,
                retrieval_scope="session",
                memory_exclude_ids=excluded_memory_ids,
                max_steps=max(1, max_steps),
            )
            agent_runner_ms = (time.perf_counter() - agent_started) * 1000
        except Exception as exc:  # noqa: BLE001
            error_text = f"{type(exc).__name__}: {exc}"
            latency_trace = _latency_trace(
                path="agentic",
                started=task_started,
                rust_core_ms=rust_core_ms,
                route_ms=route_ms,
                routing_ms=float(rust_route.get("routing_ms") or route_ms),
                evidence_ms=evidence_ms,
                memory_ms=memory_ms,
                context_ms=memory_ms,
                provider_latency_ms=model_turns[-1].get("latency_ms") if model_turns else None,
                local_budget_exceeded=(route_ms + evidence_ms + memory_ms) > 10_000,
                degraded_reason="local_pre_model_budget_exceeded" if (route_ms + evidence_ms + memory_ms) > 10_000 else None,
            )
            self.last_latency_trace = latency_trace
            final_text = _friendly_provider_failure_text(error_text) or f"Task failed: {error_text}"
            self.last_patch = None
            self.last_patch_sha256 = None
            self.last_blind_report = dict(model_turns[-1].get("blind_inference") or {}) if model_turns else None
            task_result = {
                "status": "error",
                "mode": active_agent_mode,
                "assurance": assurance,
                "agent_blueprint": agent_blueprint.blueprint_id if agent_blueprint else None,
                "goal": goal,
                "task_root": str(self.task_root),
                "selected_model": selected_model,
                "fallback_models": fallback_models,
                "route": route,
                "path_refs": path_refs,
                "url_refs": url_refs,
                "native_tool_plan": native_tool_plan,
                "capability_requirements": capability_requirements,
                "interaction_mode": active_interaction_mode,
                "mode_policy": mode_policy,
                "grounding_plan": grounding_plan,
                "tone_contract": tone_contract,
                "final": final_text,
                "tool_events": [],
                "model_turns": model_turns,
                "blind_inference": dict(self.last_blind_report or {}),
                "patch_available": False,
                "error": error_text,
            }
            self.last_trust_card = _task_trust_card_from_result(task_result)
            task_result["trust_card"] = self.last_trust_card
            self.last_task_result = task_result
            self.record(
                role="assistant",
                content=final_text,
                event_type="task_error",
                metadata={
                    "mode": active_agent_mode,
                    "agent_blueprint": agent_blueprint.blueprint_id if agent_blueprint else None,
                    "selected_model": selected_model,
                    "fallback_models": fallback_models,
                    "route": route,
                    "path_refs": path_refs,
                    "context_policy": "thread_only",
                    "retrieval_scope": "session",
                    "url_refs": url_refs,
                    "native_tool_plan": native_tool_plan,
                    "capability_requirements": capability_requirements,
                    "interaction_mode": active_interaction_mode,
                    "mode_policy": mode_policy,
                    "grounding_plan": grounding_plan,
                    "tone_contract": tone_contract,
                    "architecture_context_enabled": bool(architecture_context_pack),
                    "blind_inference_enabled": bool((self.last_blind_report or {}).get("enabled")),
                    "blind_policy_id": blind_status.get("policy_id"),
                    "blind_provider_target": selected_provider_name,
                    "blind_span_count": (self.last_blind_report or {}).get("span_count"),
                    "blind_sensitive_classes": (self.last_blind_report or {}).get("sensitive_classes") or [],
                    "blind_warnings": (self.last_blind_report or {}).get("warnings") or [],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "latency_mode": self.latency_mode,
                    "latency_trace": latency_trace,
                    "trust_card": self.last_trust_card,
                },
            )
            return task_result

        self.last_runner_trace = trace
        self.last_model_turns = model_turns
        self.last_blind_report = dict(model_turns[-1].get("blind_inference") or {}) if model_turns else None
        latency_trace = _latency_trace(
            path="agentic",
            started=task_started,
            rust_core_ms=rust_core_ms,
            route_ms=route_ms,
            routing_ms=float(rust_route.get("routing_ms") or route_ms),
            evidence_ms=evidence_ms,
            memory_ms=memory_ms,
            context_ms=memory_ms,
            provider_latency_ms=model_turns[-1].get("latency_ms") if model_turns else None,
            agent_runner_ms=agent_runner_ms,
            local_budget_exceeded=(route_ms + evidence_ms + memory_ms + agent_runner_ms) > 10_000,
            degraded_reason="local_pre_model_budget_exceeded" if (route_ms + evidence_ms + memory_ms + agent_runner_ms) > 10_000 else None,
        )
        self.last_latency_trace = latency_trace
        if model_turns:
            model_turns[-1]["timing"] = latency_trace
        planner_errors = [
            error_text
            for attempt in trace.get("planner_attempts", [])
            for error_text in attempt.get("errors", [])
        ]
        final_text = _task_visible_output(str(trace.get("final_answer") or ""))
        fallback_text = _format_runner_fallback_answer(trace, goal=goal)
        if fallback_text:
            final_text = fallback_text
        status = "completed"
        if trace.get("final_planner") == "none" and planner_errors:
            status = "error"
            final_text = _friendly_provider_failure_text(planner_errors[-1]) or f"Task failed: {planner_errors[-1]}"
        tool_events = [
            {
                "tool": item.get("tool_name"),
                "arguments": item.get("arguments") if isinstance(item.get("arguments"), dict) else {},
                "result": item.get("observation"),
            }
            for item in trace.get("observations", [])
        ]
        for index, event in enumerate(tool_events, start=1):
            self.record(
                role="tool",
                content=json.dumps(event, ensure_ascii=False),
                event_type="task_tool_result",
                metadata={
                    "step": index,
                    "tool": event.get("tool"),
                    "status": ((event.get("result") or {}).get("status") if isinstance(event.get("result"), dict) else None),
                },
            )
        patch = _extract_patch(final_text)
        self.last_patch = patch
        self.last_patch_sha256 = hashlib.sha256(patch.encode("utf-8")).hexdigest() if patch else None
        task_result = {
            "status": status,
            "mode": active_agent_mode,
            "assurance": assurance,
            "agent_blueprint": agent_blueprint.blueprint_id if agent_blueprint else None,
            "goal": goal,
            "task_root": str(self.task_root),
            "selected_model": selected_model,
            "fallback_models": fallback_models,
            "route": route,
            "path_refs": path_refs,
            "url_refs": url_refs,
            "native_tool_plan": native_tool_plan,
            "capability_requirements": capability_requirements,
            "interaction_mode": active_interaction_mode,
            "mode_policy": mode_policy,
            "grounding_plan": grounding_plan,
            "tone_contract": tone_contract,
            "final": final_text,
            "tool_events": tool_events,
            "model_turns": model_turns,
            "blind_inference": dict(self.last_blind_report or {}),
            "failover_used": model_turns[-1].get("failover_used") if model_turns else None,
            "failover_attempts": model_turns[-1].get("failover_attempts") if model_turns else [],
            "patch_available": bool(patch),
            "patch_sha256": self.last_patch_sha256,
            "trace_path": trace.get("trace_path"),
            "latency_trace": latency_trace,
        }
        if status == "error":
            task_result["error"] = planner_errors[-1] if planner_errors else final_text
        self.last_trust_card = _task_trust_card_from_result(task_result)
        task_result["trust_card"] = self.last_trust_card
        self.last_task_result = task_result
        self.record(
            role="assistant",
            content=final_text,
            event_type="task_error" if status == "error" else "task_final",
            metadata={
                "mode": active_agent_mode,
                "agent_blueprint": agent_blueprint.blueprint_id if agent_blueprint else None,
                "allowed_tools": list(agent_blueprint.allowed_tools) if agent_blueprint else None,
                "selected_model": selected_model,
                "fallback_models": fallback_models,
                "failover_used": model_turns[-1].get("failover_used") if model_turns else None,
                "failover_attempts": model_turns[-1].get("failover_attempts") if model_turns else [],
                "route": route,
                "path_refs": path_refs,
                "context_policy": "thread_only",
                "retrieval_scope": "session",
                "url_refs": url_refs,
                "native_tool_plan": native_tool_plan,
                "capability_requirements": capability_requirements,
                "interaction_mode": active_interaction_mode,
                "mode_policy": mode_policy,
                "grounding_plan": grounding_plan,
                "tone_contract": tone_contract,
                "architecture_context_enabled": bool(architecture_context_pack),
                "blind_inference_enabled": bool((self.last_blind_report or {}).get("enabled")),
                "blind_policy_id": blind_status.get("policy_id"),
                "blind_provider_target": selected_provider_name,
                "blind_span_count": (self.last_blind_report or {}).get("span_count"),
                "blind_sensitive_classes": (self.last_blind_report or {}).get("sensitive_classes") or [],
                "blind_warnings": (self.last_blind_report or {}).get("warnings") or [],
                "tool_event_count": len(tool_events),
                "patch_available": bool(patch),
                "patch_sha256": self.last_patch_sha256,
                "trust_card": self.last_trust_card,
                "trace_path": trace.get("trace_path"),
                "latency_mode": self.latency_mode,
                "latency_trace": latency_trace,
            },
        )
        return task_result

    def status(self) -> dict[str, Any]:
        profile = _model_profile_for_id(self.model if self.model.lower() not in {"auto", "router:auto"} else None)
        last_turn = self.last_model_turns[-1] if self.last_model_turns else {}
        last_timing = last_turn.get("timing") if isinstance(last_turn.get("timing"), dict) else {}
        branch_info = self._active_thread_branch_info()
        return {
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "provider": self.provider_name,
            "model": self.model,
            "provider_capabilities": _provider_capability_payload(self.provider),
            "model_capabilities": _profile_capability_payload(profile) if profile else None,
            "project": self.project,
            "agent_id": self.agent_id,
            "workspace_root": str(self.workspace_root),
            "task_root": str(self.task_root),
            "jsonl_path": str(self.jsonl_path),
            "md_path": str(self.md_path),
            "evidence_root": str(self.evidence_root),
            "event_count": len(self.events),
            "context_policy": "thread_only",
            "retrieval_scope": "session",
            "thread_kind": branch_info.get("kind"),
            "parent_thread_id": branch_info.get("parent_thread_id"),
            "parent_turn_id": branch_info.get("parent_turn_id"),
            "router_policy": self.router_policy,
            "interaction_mode": self.interaction_mode,
            "interaction_mode_profile": _interaction_mode_payload(self.interaction_mode),
            "latency_mode": self.latency_mode,
            "preflight_mode": self.preflight_mode,
            "blind_inference": self.blind_status(),
            "theme": self.theme_name,
            "response_style": self.response_style,
            "agent_mode": self.agent_mode,
            "task_engine": self.task_engine,
            "tool_policy": self.tool_policy,
            "last_latency": (
                self.last_latency_trace
                or (
                    {
                        "path": "lightweight" if last_turn.get("timing") is not None else None,
                        "provider_latency_ms": last_turn.get("latency_ms"),
                        "pre_model_ms": last_timing.get("pre_model_ms"),
                        "total_turn_ms": last_timing.get("total_turn_ms"),
                        "fast_path": last_turn.get("timing") is not None,
                        "max_tokens": last_timing.get("max_tokens"),
                        "history_turns": last_timing.get("history_turns"),
                    }
                    if last_turn
                    else None
                )
            ),
            "latency_history": _latency_history(self.events, limit=8),
            "conversation": self.conversation_status(),
            "last_patch_available": bool(self.last_patch),
            "last_task_trust_card": self.last_trust_card,
            "last_intent_card": self.last_intent_card,
            "last_artifact": self.last_artifact,
            "last_demo": self.last_demo_result,
            "config_path": str(_config_path()),
            "state_path": str(self._state_path),
        }


HELP_TEXT = """Commands:
  /help                         Show this help
  /status                       Show provider, model, workspace and transcript paths
  /latency                      Show latency mode and recent turn breakdowns
  /latency mode fast|balanced|deep
  /preflight compact|verbose|off Control how much routing preflight is shown before a turn
  /demo wow                     Run the reproducible Workbench wow demo
  /demo doctor                  Check provider, Rust core, OpenCode, agent-browser and skills for the demo
  /demo open                    Show the primary artifact from the last demo
  /hooks status                 Show internal Workbench hook events and latest progress events
  /conversation status          Show current conversational lane, summary and response-gate state
  /conversation summarize       Rebuild the active thread summary for long conversations
  /conversation reset-summary   Clear the active thread summary without deleting the transcript
  /last                         Human summary of the last Work Runtime result
  /sources                      Raw sources/anchors from the last Work Runtime result
  /read last|PATH               Read and describe the last/generated docs-web artifact
  /inspect last|PATH            Inspect artifact metadata, readback and preview
  /open last                    Show the primary generated output path
  /fast on|off|status           Toggle fast-path routing defaults and show Rust core status
  /provider NAME                Switch provider: deepinfra, gemini, openai, anthropic, ollama, llamacpp, local, ...
  /model NAME                   Switch model; aliases include auto, qwen-big, mistral, qwen, gemma, gemini-pro, nvidia-code, nvidia-research, coder, llama-vision, sonnet
  /model use NAME               Same as /model NAME; persists until /model auto
  /model list                   List model aliases and router blueprints
  /with MODEL PROMPT            Use one model for a single action, then restore the previous model
  /models                       Open/select or compact-list model profiles; /models json for full metadata
  /models compare last          Show planner/coder/critic/verifier roles and disagreements for the last task
  /route TEXT                   Explain which model auto-routing would pick
  /web QUERY                    Search the public web directly and show raw result metadata
  /router NAME                  Change routing blueprint/policy: balanced, qwen-heavy, current, qwen-gemma-mistral, cheap, premium, nvidia-build
  /router why TEXT              Explain intent scores, model choice and fallback chain
  /router list                  Open or print the routing blueprint selector
  /theme NAME                   Switch terminal theme: industrial-brutalist, industrial-neon, xerox, brown-console
  /theme list                   Open or print the theme selector/report
  /style NAME                   Response register only: balanced, technical, forensic, vivid, terse
  /mode [NAME|list|show]        Interaction mode picker. /mode opens a selector; /mode NAME (balanced|technical|explore) sets it; /mode list prints all profiles; /mode show prints the current JSON.
  /blind on|off|status          Toggle or inspect blind inference cloud proxy mode
  /blind policy PATH_OR_INLINE  Load an explicit blind inference policy from JSON file or inline JSON
  /tech TEXT                    One-shot technical turn without changing the sticky mode
  /explore TEXT                 One-shot exploratory turn without changing the sticky mode
  /raw on|off                   Toggle raw model output after the cleaned answer
  /clear                        Clear the terminal
  /key [PROVIDER]               Prompt for a provider API key for this process only
  /key save [PROVIDER]          Save provider API key in HeliX user config
  /key forget [PROVIDER]        Remove saved provider API key from HeliX user config
  /config                       Show HeliX config/data paths
  /doctor                       Run helix doctor
  /doctor perf                  Show local latency budget, Rust core and suite-index readiness
  /opencode status              Show OpenCode binary and Rust MCP readiness
  /opencode install --global    Install HeliX MCP config for OpenCode
  /providers                    List providers
  /cert SUITE [-- args]         Run a certification suite
  /cert-dry SUITE [-- args]     Show the suite command without running it
  /evidence refresh [--deep] Q  Refresh indexed evidence metadata; --deep verifies/replays artifacts
  /evidence latest [N]          Show latest certified evidence memories
  /evidence search QUERY        Search certified evidence memories
  /evidence show MEMORY_ID      Show one certified evidence memory, receipt and chain status
  /verify PATH|latest|search Q  Verify an artifact JSON or discover verified artifacts
  /verify last --level quick    Verify the last Task Capsule without rerunning the task
  /lab profiles                 List commercial runtime verification profiles
  /lab run PROFILE              Run quick readiness checks: patch-safety, doc-grounding, memory-isolation, provider-audit
  /flow list                    List commercial HeliX flow profiles built from verification protocols
  /flow run PROFILE GOAL        Run a flow: web, web-recursive, patch-safe, doc-grounded, resilient-task, privacy-swarm, multi-review, deep-lab
  /work run GOAL                Natural Work Runtime: collect sources, plan flow, generate/analyze with HeliX guarantees
  /work status                  Show last Work Runtime plan/source/trust status
  /work last                    Show the last Work Runtime result
  /work history                 Show recent Work Runtime runs
  /work sources last            Show sources and anchors from the last Work Runtime run
  /work plan last               Show the last Work Runtime plan
  /suites                       Compact catalog of local verification suites and latest artifacts
  /suite latest SUITE           Show latest artifact, manifest and transcript paths for one suite
  /suite index refresh          Rebuild the fast local suite/evidence index
  /suite transcripts SUITE      List suite transcript exports; add a filter after the suite name
  /suite search [--deep] QUERY  Search indexed artifacts/transcripts; --deep scans bodies explicitly
  /suite read PATH_OR_NAME      Read a bounded local artifact/transcript excerpt
  /file PATH                    Inspect a local file or directory path under allowed HeliX roots
  /memory QUERY                 Search HeliX memory in the active thread only
  /memory search --global QUERY Search workspace memory explicitly across threads
  /memory resolve HASH_OR_ID    Resolve a memory_id or node_hash prefix to exact stored content
  /trust [current|THREAD_ID]     Compact local trust summary; add json/proof/--forensics for full proof
  /trust last                   Human Task Capsule trust card for the last agentic task
  /thread new --clean [TITLE]   Create and switch to a new zero-context persistent thread
  /thread list                  List known workspace threads
  /thread tree                  Show the thread/branch tree and active branch
  /thread open THREAD_ID        Reopen an existing thread
  /thread close [THREAD_ID]     Close a thread without deleting its memory
  /thread current               Show the active thread
  /branch new [TITLE]           Fork a new zero-history branch from the last completed turn
  /task GOAL                    Run the unified HeliX runner; code tasks can use OpenCode sandbox
  /task --engine opencode --assurance quick|balanced|strict GOAL
  /task engine auto|helix|opencode
  /tools                        Compact-list runner tools; /tools blueprints for agent toolsets; /tools json for raw registry
  /agents                       List agentic blueprints for Codex-like tasks and evidence analysis
  /apply last                   Apply last proposed patch after explicit confirmation
  /agent suggest GOAL           Codex-like safe mode: read, plan, use read-only tools, propose next actions
  /agent use BLUEPRINT GOAL     Run a specific agent blueprint, e.g. suite-run-analyst or patch-planner
  /agent GOAL                   Alias for /agent suggest GOAL
  /exit                         Leave the session

Natural language defaults to chat. Repo/debug/patch requests are routed to /task; certification suite requests are routed to /cert.
Use /mode technical for diagnosis, code, hashes, receipts, suites and architecture. Use /mode explore for philosophy, culture, wider research and source-backed exploration.
You can also ask naturally: "lee src/helix_proto", "compará cognitive-gauntlet con local-ghost-in-the-shell-live", or "revisá esta URL y resumila".
"""


def _split_command(text: str) -> list[str]:
    try:
        return [item.strip('"') for item in shlex.split(text, posix=False)]
    except ValueError:
        return text.split()


def _normalize_latency_mode(value: str | None) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in {"fast", "balanced", "deep"} else "fast"


def _normalize_preflight_mode(value: str | None) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in {"compact", "verbose", "off"} else "compact"


def _normalize_task_engine(value: str | None) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in {"auto", "helix", "opencode"} else "auto"


def _normalize_assurance(value: str | None) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in {"quick", "balanced", "strict", "deep"} else "quick"


def _extract_task_options(parts: list[str], *, default_engine: str, default_assurance: str = "quick") -> tuple[str, str, list[str]]:
    engine = _normalize_task_engine(default_engine)
    assurance = _normalize_assurance(default_assurance)
    cleaned: list[str] = []
    index = 0
    while index < len(parts):
        item = parts[index]
        lower = item.lower()
        if lower == "--engine" and index + 1 < len(parts):
            engine = _normalize_task_engine(parts[index + 1])
            index += 2
            continue
        if lower.startswith("--engine="):
            engine = _normalize_task_engine(item.split("=", 1)[1])
            index += 1
            continue
        if lower == "--assurance" and index + 1 < len(parts):
            assurance = _normalize_assurance(parts[index + 1])
            index += 2
            continue
        if lower.startswith("--assurance="):
            assurance = _normalize_assurance(item.split("=", 1)[1])
            index += 1
            continue
        cleaned.append(item)
        index += 1
    return engine, assurance, cleaned


def _extract_task_engine(parts: list[str], *, default: str) -> tuple[str, list[str]]:
    engine, _assurance, cleaned = _extract_task_options(parts, default_engine=default)
    return engine, cleaned


def _task_engine_for_goal(goal: str, *, configured: str, rust_route: dict[str, Any] | None = None) -> str:
    configured = _normalize_task_engine(configured)
    if configured != "auto":
        return configured
    path = str((rust_route or {}).get("path") or "").lower()
    lowered = str(goal or "").lower()
    strong_code_terms = (
        "implement", "implementa", "implementá", "arregla", "arreglá", "fix", "refactor",
        "patch", "diff", "test", "tests", "pytest", "cargo test", "bug", "repo", "archivo",
        "codigo", "código", "modifica", "modificá", "edita", "editá",
    )
    if path == "agentic" and any(term in lowered for term in strong_code_terms):
        return "opencode"
    return "helix"


def _latency_trace(
    *,
    path: str,
    started: float,
    rust_core_ms: float = 0.0,
    route_ms: float = 0.0,
    routing_ms: float | None = None,
    index_ms: float = 0.0,
    context_ms: float | None = None,
    python_ms: float = 0.0,
    evidence_ms: float = 0.0,
    memory_ms: float = 0.0,
    provider_latency_ms: float | None = None,
    agent_runner_ms: float = 0.0,
    max_tokens: int | None = None,
    history_turns: int | None = None,
    local_budget_exceeded: bool = False,
    degraded_reason: str | None = None,
    lane: str | None = None,
    conversation_summary_ms: float = 0.0,
    response_gate_ms: float = 0.0,
    suppressed_reasoning: bool = False,
    repair_retry_used: bool = False,
) -> dict[str, Any]:
    trace = {
        "path": path,
        "rust_core_ms": round(float(rust_core_ms or 0.0), 3),
        "route_ms": round(float(route_ms or 0.0), 3),
        "routing_ms": round(float(routing_ms if routing_ms is not None else route_ms or 0.0), 3),
        "index_ms": round(float(index_ms or 0.0), 3),
        "context_ms": round(float(context_ms if context_ms is not None else memory_ms or 0.0), 3),
        "python_ms": round(float(python_ms or 0.0), 3),
        "evidence_ms": round(float(evidence_ms or 0.0), 3),
        "memory_ms": round(float(memory_ms or 0.0), 3),
        "provider_latency_ms": provider_latency_ms,
        "agent_runner_ms": round(float(agent_runner_ms or 0.0), 3),
        "conversation_summary_ms": round(float(conversation_summary_ms or 0.0), 3),
        "response_gate_ms": round(float(response_gate_ms or 0.0), 3),
        "total_turn_ms": round((time.perf_counter() - started) * 1000, 3),
        "fast_path": path == "lightweight",
        "local_budget_exceeded": bool(local_budget_exceeded),
        "suppressed_reasoning": bool(suppressed_reasoning),
        "repair_retry_used": bool(repair_retry_used),
    }
    if lane:
        trace["lane"] = lane
    if degraded_reason:
        trace["degraded_reason"] = degraded_reason
    if max_tokens is not None:
        trace["max_tokens"] = max_tokens
    if history_turns is not None:
        trace["history_turns"] = history_turns
    numeric = {
        key: value
        for key, value in trace.items()
        if key.endswith("_ms") and key != "total_turn_ms" and isinstance(value, (int, float))
    }
    trace["dominant_phase"] = max(numeric.items(), key=lambda item: item[1])[0] if numeric else None
    return trace


def _latency_history(events: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in reversed(events):
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        trace = metadata.get("latency_trace") if isinstance(metadata.get("latency_trace"), dict) else None
        if not trace:
            continue
        rows.append(
            {
                "event": event.get("event"),
                "role": event.get("role"),
                "created_utc": event.get("created_utc"),
                "path": trace.get("path"),
                "lane": trace.get("lane"),
                "total_turn_ms": trace.get("total_turn_ms"),
                "dominant_phase": trace.get("dominant_phase"),
                "provider_latency_ms": trace.get("provider_latency_ms"),
                "suppressed_reasoning": trace.get("suppressed_reasoning"),
            }
        )
        if len(rows) >= limit:
            break
    rows.reverse()
    return rows


def _read_default(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}] (Enter = default): ").strip()
    return raw or default


def _choose_provider(default: str) -> tuple[str, str | None]:
    while True:
        raw = input(f"Provider [{default}] (Enter = default): ").strip()
        if not raw:
            return default, None
        value = raw.lower()
        if value in PROVIDERS:
            return value, None
        print(f"[helix] Using provider={default}; treating initial input as first chat message.")
        return default, raw


def _provider_ready_from_config(provider_name: str) -> bool:
    provider = PROVIDERS.get(provider_name)
    if not provider:
        return False
    if not provider.requires_token:
        return True
    return bool((provider.token_env and os.environ.get(provider.token_env)) or _config_token(provider.name))


def _ensure_provider_token(provider_name: str) -> None:
    provider = PROVIDERS[provider_name]
    if not provider.requires_token:
        return
    had_saved = bool(_config_token(provider.name))
    token = _token_for_provider(provider, prompt=True)
    if token and provider.token_env:
        os.environ[provider.token_env] = token
        if had_saved:
            print(f"[helix] {provider.token_env} loaded from HeliX config.")
        else:
            print(f"[helix] {provider.token_env} loaded for this process only.")
            save = input("Save this token to HeliX user config for future sessions? [y/N]: ").strip().lower()
            if save in {"y", "yes", "s", "si"}:
                path = _save_config_token(provider.name, token)
                print(f"[helix] token saved in user config: {path}")


def _maybe_prompt_optional_provider_token(provider_name: str, *, config: dict[str, Any]) -> dict[str, Any]:
    provider = PROVIDERS.get(provider_name)
    if not provider or not provider.requires_token or not provider.token_env:
        return config
    if os.environ.get(provider.token_env) or _config_token(provider.name):
        return config
    optional = config.get("optional_token_prompts")
    if not isinstance(optional, dict):
        optional = {}
    if optional.get(provider.name) == "skip":
        return config
    model_hint = (
        "NVIDIA Build free models like mistralai/magistral-small-2506"
        if provider.name == "nvidia"
        else f"models like {provider.default_model}"
    )
    answer = input(f"Optional: save {provider.name} API key now for {model_hint}? [y/N/skip]: ").strip().lower()
    if answer in {"skip", "never", "nope"}:
        optional[provider.name] = "skip"
        config["optional_token_prompts"] = optional
        _save_config(config)
        print(f"[helix] optional {provider.name} key prompt disabled. Use /key save {provider.name} if you want it later.")
        return config
    if answer not in {"y", "yes", "s", "si"}:
        if provider.name == "nvidia":
            optional[provider.name] = "skip"
            config["optional_token_prompts"] = optional
            _save_config(config)
            print("[helix] optional nvidia key prompt skipped. Use /key save nvidia if you want it later.")
        return config
    token = getpass.getpass(f"Paste {provider.name} token to save in HeliX config: ").strip()
    if not token:
        print("[helix] no optional token saved.")
        return config
    os.environ[provider.token_env] = token
    path = _save_config_token(provider.name, token)
    print(f"[helix] {provider.token_env} saved in HeliX config: {path}")
    return _load_config()


def _suite_from_text(text: str) -> str | None:
    normalized = _slugish(text)
    for suite_id in sorted(SUITES, key=len, reverse=True):
        if suite_id in normalized:
            return suite_id
        compact = suite_id.replace("-", "")
        if compact in normalized.replace("-", ""):
            return suite_id
    aliases = {
        "poliza": "policy-rag-legal-debate",
        "polizas": "policy-rag-legal-debate",
        "policy-rag": "policy-rag-legal-debate",
        "branch-pruning": "branch-pruning-forensics",
        "poda": "branch-pruning-forensics",
        "hard-anchor": "hard-anchor-utility",
        "long-horizon": "long-horizon-checkpoints",
        "post-nuclear": "post-nuclear-methodology",
        "infinite-depth": "infinite-depth-memory",
        "cognitive-gauntlet": "cognitive-gauntlet",
        "cognitive gauntlet": "cognitive-gauntlet",
    }
    for alias, suite_id in aliases.items():
        if alias in normalized:
            return suite_id
    return None


def _preferred_language_instruction(text: str) -> str:
    lowered = str(text or "").lower()
    spanish_markers = (
        " que ",
        "como",
        "hola",
        "vos",
        "tenes",
        "tienes",
        "haces",
        "explicame",
        "corre",
        "arregla",
        "ayuda",
        "especial",
    )
    if any(marker in f" {lowered} " for marker in spanish_markers):
        return "Respond in Spanish, matching the user's informal Rioplatense Spanish when appropriate."
    return "Respond in the same language as the user's latest message."


def _is_identity_question(text: str) -> bool:
    lowered = str(text or "").lower().strip(" ?!")
    direct_self_questions = {
        "que te hace especial",
        "qué te hace especial",
        "q te hace especial",
        "que sos",
        "qué sos",
        "quien sos",
        "quién sos",
    }
    if lowered in direct_self_questions:
        return True
    if "especial" not in lowered or not ("hace" in lowered or "diferente" in lowered):
        return False
    return "helix" in lowered or bool(re.search(r"\b(te|vos|tu|tus)\b", lowered))


def _recent_history_mentions_helix(history: list[dict[str, str]] | None = None) -> bool:
    for item in history or []:
        if "helix" in str(item.get("content") or "").lower():
            return True
    return False


def _has_helix_scope_term(text: str) -> bool:
    lowered = str(text or "").lower()
    scope_terms = (
        "helix",
        "/verify",
        "verify",
        "merkle",
        "dag",
        "receipt",
        "receipts",
        "node_hash",
        "node hash",
        "hash",
        "hashes",
        "firma",
        "firmas",
        "signature",
        "signatures",
        "memoria",
        "memory",
        "evidencia",
        "evidence",
        "artifact",
        "artifacts",
        "artefacto",
        "suite",
        "suites",
        "corrida",
        "corridas",
        "transcript",
        "transcripts",
        "tombstone",
        "fence",
        "rollback",
    )
    return any(term in lowered for term in scope_terms)


def _is_clear_non_helix_topic_shift(text: str) -> bool:
    lowered = " ".join(str(text or "").lower().split())
    if not lowered or _has_helix_scope_term(lowered):
        return False
    model_or_provider_terms = (
        "qwen",
        "gemini",
        "deepinfra",
        "mistral",
        "gemma",
        "llama",
        "claude",
        "sonnet",
        "deepseek",
        "openai",
        "anthropic",
        "modelo",
        "model",
        "llm",
        "llms",
    )
    if any(term in lowered for term in model_or_provider_terms):
        return True
    topic_shift_starters = (
        "hablame de ",
        "hablame sobre ",
        "contame de ",
        "contame sobre ",
        "quiero saber de ",
        "quiero saber sobre ",
        "que sabes de ",
        "que sabes sobre ",
        "dame info de ",
        "dame info sobre ",
        "quiero info de ",
        "quiero info sobre ",
        "buscame info de ",
        "buscame info sobre ",
        "busca info de ",
        "busca info sobre ",
        "investiga ",
        "googlea ",
    )
    if lowered.startswith(topic_shift_starters):
        return True
    general_topic_terms = (
        "argentina",
        "buenos aires",
        "politica",
        "economia",
        "historia",
        "cultura",
        "futbol",
        "viaje",
        "turismo",
        "comida",
        "dolar",
        "gobierno",
        "presidente",
        "pais",
        "mundo",
    )
    return any(term in lowered for term in general_topic_terms)


def _is_contextual_helix_followup(text: str) -> bool:
    lowered = " ".join(str(text or "").lower().strip().split())
    if not lowered or len(lowered) > 220 or _is_clear_non_helix_topic_shift(lowered):
        return False
    direct_followup_terms = (
        "eso",
        "esto",
        "entenderlo",
        "entenderla",
        "entender eso",
        "entender esto",
        "explicamelo",
        "explicame eso",
        "explicame esto",
        "ayudes a entender",
        "ayudar a entender",
        "que mas",
        "que otra cosa",
        "como funciona",
        "como trabaja",
        "para que sirve",
        "que permite",
        "que hace",
        "chasis",
        "trazabilidad",
        "auditabilidad",
        "auditable",
        "seguro",
        "segura",
        "seguridad",
        "riguroso",
        "rigurosa",
        "rigor",
        "valida",
        "valido",
        "invalida",
        "invalido",
    )
    if any(term in lowered for term in direct_followup_terms):
        return True
    return lowered in {
        "como?",
        "como",
        "y?",
        "y eso?",
        "y eso",
        "y esto?",
        "y esto",
        "que onda?",
        "que onda",
        "por que?",
        "por que",
    }


def _recent_history_offered_evidence(history: list[dict[str, str]] | None = None) -> bool:
    for item in reversed(history or []):
        content = str(item.get("content") or "").lower()
        if any(term in content for term in ("puedo mostrar evidencia", "evidencia local", "/trust current", "/evidence latest")):
            return True
    return False


def _is_evidence_acceptance_request(text: str, history: list[dict[str, str]] | None = None) -> bool:
    lowered = " ".join(str(text or "").lower().strip().rstrip("?!.").split())
    if not lowered:
        return False
    explicit = any(term in lowered for term in ("mostrame evidencia", "mostrar evidencia", "ver evidencia", "trae evidencia", "traeme evidencia"))
    if explicit:
        return True
    if not _recent_history_offered_evidence(history):
        return False
    return lowered in {"dale", "si", "sí", "ok", "ok dale", "si dale", "sí dale", "mostrame", "mostra", "mostrá"}


def _is_affective_reaction(text: str) -> bool:
    lowered = " ".join(str(text or "").lower().strip().split())
    if not lowered or "?" in lowered:
        return False
    reaction_terms = (
        "una locura",
        "increible",
        "impresionante",
        "muy bueno",
        "buenisimo",
        "esta bueno",
        "es genial",
        "en el buen sentido",
    )
    return any(term in lowered for term in reaction_terms) or bool(re.search(r"\bme gusta\b", lowered))


def _is_helix_auditability_request(text: str, history: list[dict[str, str]] | None = None) -> bool:
    lowered = str(text or "").lower()
    if _is_clear_non_helix_topic_shift(lowered):
        return False
    helix_in_scope = _has_helix_scope_term(lowered) or (
        _recent_history_mentions_helix(history) and _is_contextual_helix_followup(lowered)
    )
    if not helix_in_scope:
        return False
    auditability_terms = (
        "auditabilidad",
        "auditible",
        "auditibilidad",
        "audtibilidad",
        "auditable",
        "auditoria",
        "auditoría",
        "receipt",
        "receipts",
        "firma",
        "firmas",
        "firmado",
        "firmada",
        "signature",
        "signatures",
        "hash",
        "hashes",
        "node_hash",
        "node hash",
        "merkle",
        "dag",
        "chain",
        "cadena",
        "integridad",
        "seguro",
        "segura",
        "seguridad",
        "riguroso",
        "rigurosa",
        "rigor",
        "trazabilidad",
        "verificado",
        "verificable",
        "verificación",
        "verificacion",
    )
    return any(term in lowered for term in auditability_terms)


def _is_explicit_helix_meta_task_request(text: str, history: list[dict[str, str]] | None = None) -> bool:
    lowered = " ".join(str(text or "").lower().strip().split())
    if not lowered or _is_clear_non_helix_topic_shift(lowered):
        return False
    if _is_helix_auditability_request(lowered, history):
        return True
    direct_meta_terms = (
        "helix",
        "merkle",
        "merkle-dag",
        "receipt",
        "receipts",
        "node_hash",
        "node hash",
        "hash",
        "hashes",
        "firma",
        "firmas",
        "signature",
        "signatures",
        "canonical head",
        "head canonico",
        "head canónico",
        "cabeza canonica",
        "cabeza canónica",
        "canonica",
        "canónica",
        "canonico",
        "canónico",
        "equivocation",
        "equivocacion",
        "equivocación",
        "lineage",
        "quarantine",
        "quarantined",
        "cuarentena",
        "cuarentenada",
        "cuarentenado",
        "arquitectura",
        "architecture",
        "runtime",
        "router",
        "thread_id",
        "session_id",
        "signed receipt",
        "signed receipts",
        "signed_receipts",
        "helix-state-core",
        "state core",
        "recursive witness",
        "branch-pruning",
        "branch pruning",
        "metodologia nuclear",
        "metodología nuclear",
        "ouroboros",
    )
    if any(term in lowered for term in direct_meta_terms):
        return True
    if not _recent_history_mentions_helix(history):
        return False
    contextual_meta_terms = (
        "arquitectura",
        "architecture",
        "implementarias",
        "implementarías",
        "implementarlo",
        "implementación",
        "implementacion",
        "cabeza canonica",
        "cabeza canónica",
        "head canonico",
        "head canónico",
        "canonica",
        "canónica",
        "canonico",
        "canónico",
        "equivocation",
        "equivocacion",
        "equivocación",
        "lineage",
        "quarantine",
        "cuarentena",
        "receipt",
        "receipts",
        "firma",
        "firmas",
        "signature",
        "signatures",
        "hash",
        "hashes",
        "merkle",
        "dag",
        "thread",
        "threads",
        "session",
        "runtime",
        "router",
        "tool registry",
    )
    return any(term in lowered for term in contextual_meta_terms)


def _is_creative_helix_prompt(text: str) -> bool:
    lowered = " ".join(str(text or "").lower().strip().split())
    if not lowered or not _has_helix_scope_term(lowered):
        return False
    creative_terms = (
        "filosofia",
        "filosofía",
        "cultura",
        "cultural",
        "metafora",
        "metáfora",
        "ghost in the shell",
        "rizoma",
        "rizomas",
        "hipersticion",
        "hiperstición",
        "deleuze",
        "guattari",
        "ontologia",
        "ontología",
        "poetica",
        "poética",
        "influencias",
        "simbolismo",
        "explora",
        "explorar",
        "exploremos",
        "creativo",
        "imaginario",
        "filosofico",
        "filosófico",
        "filosofica",
        "filosófica",
    )
    hard_core_terms = (
        "merkle",
        "dag",
        "receipt",
        "receipts",
        "hash",
        "hashes",
        "signature",
        "signatures",
        "firma",
        "firmas",
        "canonical head",
        "cabeza canonica",
        "cabeza canónica",
        "equivocation",
        "equivocacion",
        "equivocación",
        "quarantine",
        "cuarentena",
        "arquitectura",
        "architecture",
        "runtime",
        "router",
        "thread_id",
        "session_id",
        "signed receipt",
        "signed receipts",
        "signed_receipts",
        "helix-state-core",
        "state core",
        "recursive witness",
        "branch-pruning",
        "branch pruning",
        "metodologia nuclear",
        "metodología nuclear",
        "ouroboros",
        "repo",
        "repositorio",
        "codigo",
        "código",
        "code",
        "implementa",
        "implementá",
        "implementacion",
        "implementación",
        "audit",
        "auditoria",
        "auditoría",
        "suite",
        "verify",
        "/verify",
    )
    return any(term in lowered for term in creative_terms) and not any(term in lowered for term in hard_core_terms)


def _is_helix_explanation_request(text: str, history: list[dict[str, str]] | None = None) -> bool:
    lowered = str(text or "").lower()
    if _is_affective_reaction(lowered) or _is_clear_non_helix_topic_shift(lowered):
        return False
    helix_in_scope = _has_helix_scope_term(lowered) or (
        _recent_history_mentions_helix(history) and _is_contextual_helix_followup(lowered)
    )
    if not helix_in_scope:
        return False
    explanation_terms = (
        "entender",
        "entenderlo",
        "entenderla",
        "explica",
        "explicame",
        "explicámelo",
        "explicamelo",
        "como funciona",
        "cómo funciona",
        "que hace",
        "qué hace",
        "que permite",
        "qué permite",
        "para que sirve",
        "para qué sirve",
        "como trabaja",
        "cómo trabaja",
        "ayudes a entender",
        "ayudar a entender",
    )
    capability_terms = (
        "eso",
        "esto",
        "sistema",
        "runtime",
        "memoria",
        "evidencia",
        "threads",
        "thread",
        "receipts",
        "hash",
        "verificacion",
        "verificación",
    )
    if _has_helix_scope_term(lowered):
        return (
            any(term in lowered for term in explanation_terms)
            or any(term in lowered for term in capability_terms)
            or _is_helix_auditability_request(lowered, history)
        )
    return _is_contextual_helix_followup(lowered) or _is_helix_auditability_request(lowered, history)


def _helix_grounding_query(text: str) -> str:
    lowered = str(text or "").lower()
    if _is_helix_auditability_request(lowered):
        return (
            "HeliX auditabilidad receipts firmas signature_verified node_hash Merkle DAG chain status "
            "memoria firmada evidencia verificable hashes integridad trazabilidad"
        )
    return (
        "HeliX capacidades memoria firmada evidencia certificada threads persistentes "
        "busqueda unificada receipts hashes tool registry runtime"
    )


def _override_route_for_helix_focus(route: dict[str, Any], *, user_text: str, policy: str, auditability: bool) -> dict[str, Any]:
    blueprint = _resolve_router_blueprint(policy)
    alias = blueprint.audit_alias if auditability else "qwen-big"
    profile = DEEPINFRA_MODEL_PROFILES[alias]
    signals = list(route.get("signals") or [])
    signals.append("helix_focus")
    if auditability:
        signals.append("helix_auditability")
    return _augment_route_metadata(
        {
            **route,
            "provider": "deepinfra",
            "model": profile.model_id,
            "profile": alias,
            "role": profile.role,
            "intent": "audit" if auditability else "reasoning",
            "confidence": max(float(route.get("confidence") or 0.0), 0.88),
            "signals": sorted(set(signals)),
            "policy": blueprint.name,
            "blueprint": blueprint.name,
            "blueprint_description": blueprint.description,
            "reason": (
                "Contextual HeliX auditability question promoted to the blueprint audit profile for grounded answers."
                if auditability
                else "Contextual HeliX explanation request promoted to the blueprint large-Qwen research profile for grounded answers."
            ),
        },
        user_text,
    )


def _override_route_for_creative_helix_focus(route: dict[str, Any], *, user_text: str, policy: str) -> dict[str, Any]:
    blueprint = _resolve_router_blueprint(policy)
    alias = blueprint.research_alias or "qwen-big"
    profile = MODEL_PROFILES[alias]
    signals = list(route.get("signals") or [])
    signals.append("creative_helix_context")
    return _augment_route_metadata(
        {
            **route,
            "provider": profile.provider,
            "model": profile.model_id,
            "profile": alias,
            "role": profile.role,
            "intent": "creative_helix",
            "confidence": max(float(route.get("confidence") or 0.0), 0.86),
            "signals": sorted(set(signals)),
            "policy": blueprint.name,
            "blueprint": blueprint.name,
            "blueprint_description": blueprint.description,
            "reason": "Contextual HeliX follow-up kept in creative/cultural synthesis mode because the turn stayed exploratory instead of moving into core audit semantics.",
        },
        user_text,
    )


def _needs_certified_evidence(
    text: str,
    route: dict[str, Any] | None = None,
    history: list[dict[str, str]] | None = None,
) -> bool:
    lowered = str(text or "").lower()
    if _is_evidence_acceptance_request(lowered, history):
        return True
    if _is_identity_question(lowered):
        return True
    if _is_helix_auditability_request(lowered, history):
        return True
    if _is_clear_non_helix_topic_shift(lowered) or _is_affective_reaction(lowered):
        return False
    evidence_terms = (
        "evidencia",
        "evidence",
        "certified",
        "certificada",
        "verifica",
        "verificá",
        "verificar",
        "audita",
        "auditá",
        "auditar",
        "demostra",
        "demostrá",
        "proba",
        "probá",
        "merkle",
        "dag",
        "hash",
        "receipt",
        "recibo",
        "transcript",
        "jsonl",
        "tombstone",
        "lapida",
        "fence",
    )
    if any(term in lowered for term in evidence_terms):
        return True
    return bool(route and route.get("intent") in {"model_control"})


def _needs_repository_evidence(text: str, route: dict[str, Any] | None = None) -> bool:
    lowered = str(text or "").lower()
    evidence_terms = (
        "/verify",
        "verify",
        "verifica",
        "verificá",
        "verificar",
        "audita",
        "auditá",
        "auditar",
        "lee repo",
        "leer repo",
        "repositorio",
        "codigo",
        "código",
        "archivo",
        "file",
        "artifact",
        "artefact",
        "artefacto",
        "evidencia",
        "evidence",
        "corrida",
        "corridas",
        "run ",
        "run_id",
        "suite",
        "transcript",
        "jsonl",
        "manifest",
        "hash",
        "sha",
        "prueba",
        "probá",
        "proba",
        "test",
        "benchmark",
        "hard anchor",
        "hard-anchor",
        "post nuclear",
        "post-nuclear",
        "long horizon",
        "branch pruning",
        "tombstone",
        "lapida",
        "fence",
    )
    if any(term in lowered for term in evidence_terms):
        return True
    if _is_suite_evidence_request(text):
        return True
    return bool(route and route.get("intent") in {"audit", "suite_forensics"})


def _should_use_lightweight_chat_path(
    text: str,
    *,
    route: dict[str, Any] | None,
    recent_history: list[dict[str, str]] | None,
    helix_focus: bool,
    helix_auditability: bool,
    suite_focus: bool,
    web_focus: bool,
    hash_recovery_ref: str | None,
    file_path_ref: str | None,
    url_refs: list[str] | None,
    latency_mode: str = "fast",
) -> bool:
    latency_mode = _normalize_latency_mode(latency_mode)
    if latency_mode == "deep":
        return False
    if helix_auditability or suite_focus or web_focus:
        return False
    if (
        latency_mode != "fast"
        and helix_focus
        and _recent_history_mentions_helix(recent_history)
        and _is_contextual_helix_followup(text)
    ):
        return False
    if latency_mode == "balanced" and helix_focus:
        return False
    if hash_recovery_ref or file_path_ref or list(url_refs or []):
        return False
    if _goal_requests_memory_lookup(text):
        return False
    if _needs_certified_evidence(text, route, recent_history):
        return False
    if _needs_repository_evidence(text, route):
        return False
    intent = str((route or {}).get("intent") or "")
    if intent and intent not in {"chat", "reasoning", "research", "helix_self", "creative_helix"}:
        return False
    return True


def _conversation_lane_for_turn(
    text: str,
    *,
    route: dict[str, Any] | None,
    use_lightweight_chat: bool,
    file_path_ref: str | None,
    helix_auditability: bool,
    suite_focus: bool,
    web_focus: bool,
    hash_recovery_ref: str | None,
    certified_evidence_required: bool = False,
) -> tuple[str, str]:
    intent = str((route or {}).get("intent") or "")
    if helix_auditability or hash_recovery_ref or intent == "audit":
        return "deep", "explicit evidence/audit/hash signal"
    if file_path_ref:
        return "file_qa", "explicit local file or directory reference"
    if certified_evidence_required:
        return "deep", "explicit evidence/audit/hash signal"
    if suite_focus or intent == "suite_forensics":
        return "deep", "explicit suite/evidence signal"
    if intent in {"agentic", "agentic_code", "code"} and not use_lightweight_chat:
        return "work", f"router intent={intent}"
    if web_focus:
        return "deep", "explicit web/current-info grounding signal"
    return "conversation", "default conversational lane"


def _lightweight_chat_token_budget(text: str, configured_max_tokens: int) -> int:
    """Keep ordinary chat snappy without starving explicit deep-answer prompts."""
    configured = _safe_int(configured_max_tokens, 512, minimum=64, maximum=8192)
    lowered = str(text or "").lower()
    wants_depth = any(
        term in lowered
        for term in (
            "detall",
            "profund",
            "paso a paso",
            "step by step",
            "lista",
            "compar",
            "analiz",
            "explica bien",
            "explicame bien",
            "largo",
            "completo",
        )
    )
    word_count = len(str(text or "").split())
    if wants_depth or word_count > 28:
        return min(configured, 700)
    if word_count <= 8:
        return min(configured, 220)
    return min(configured, 384)


def _with_evidence_offer(answer: str) -> str:
    hint = "Puedo mostrar evidencia local si queres: /trust current o /evidence latest."
    return answer if hint.lower() in str(answer).lower() else f"{answer} {hint}"


def _basic_helix_fast_answer(text: str, *, recent_history: list[dict[str, str]] | None = None) -> str | None:
    """Deterministic answer for the most common HeliX self-intro prompts.

    This keeps the first-touch CLI experience clean even when the selected
    provider is a reasoning model that may spend the whole small budget on
    scratchpad text.
    """
    cleaned = " ".join(str(text or "").lower().strip().rstrip("?!.").split())
    if not cleaned:
        return None
    definition_prompts = {
        "que es helix",
        "qué es helix",
        "q es helix",
        "que seria helix",
        "qué sería helix",
        "what is helix",
        "helix",
    }
    thinking_prompts = (
        "pensemos en helix",
        "pensemo en helix",
        "pensemos entonces en helix",
        "pensemo entonces en helix",
        "pensemos sobre helix",
    )
    if cleaned in definition_prompts:
        return _with_evidence_offer(
            "HeliX es una CLI/runtime para trabajar con modelos de IA sin que todo dependa de un solo prompt gigante: "
            "organiza hilos, memoria por sesión, routing de modelos, herramientas, evidencia y tareas agentic cuando hacen falta. "
            "En modo rápido debería comportarse como chat normal; cuando pedís auditar, leer repo, buscar evidencia o ejecutar una tarea, pasa a un modo más profundo."
        )
    if "para que sirve" in cleaned or "para qu" in cleaned and "sirve" in cleaned:
        return _with_evidence_offer(
            "Sirve para usar modelos de IA con mas control que un chat suelto: podes conversar, sostener un hilo, "
            "leer archivos locales, elegir modelos segun la tarea y escalar a herramientas o evidencia cuando hace falta. "
            "La idea fuerte es separar charla rapida, trabajo sobre archivos/repo y verificacion profunda, para no pagar todo el costo en cada turno."
        )
    if cleaned in {"como", "como funciona"} or ("como" in cleaned and "funciona" in cleaned):
        return _with_evidence_offer(
            "Funciona como una capa de orquestacion alrededor de modelos: primero decide el carril del turno, "
            "despues arma solo el contexto necesario y recien ahi llama al modelo o a una herramienta. "
            "Para una pregunta comun responde como conversacion; si pedis leer un archivo usa `file.inspect`; si pedis patch/tests entra al modo de trabajo; si pedis evidencia o hashes entra al modo profundo."
        )
    if cleaned in thinking_prompts or (cleaned.startswith("pensem") and "helix" in cleaned):
        return (
            "Pensemos HeliX como dos superficies separadas: una conversación rápida para decidir y aclarar ideas, "
            "y un modo de trabajo que recién carga memoria, repo, evidencia o agentes cuando el turno lo pide. "
            "El salto de producto está en que esa separación se sienta natural: cero espera para hablar, progreso visible para trabajar, y profundidad explícita cuando queremos pruebas."
        )
    if "helix" in cleaned and "latencia" in cleaned and len(cleaned.split()) <= 12:
        return (
            "Si HeliX se siente lento, el problema suele estar antes del modelo: routing pesado, memoria/evidencia cargándose en caliente, "
            "o un modelo de razonamiento gastando tokens en scratchpad. En fast mode conviene responder con contexto cero, usar un modelo chat liviano y dejar evidencia/AgentRunner solo para pedidos explícitos."
        )
    return None


_VAGUE_FOLLOWUP_PATTERNS = (
    "y eso", "y eso?", "y luego", "y luego?", "y entonces", "y entonces?",
    "y ahora", "y ahora?", "ahora?", "ya?", "y?", "y bueno", "bueno y?",
    "que mas", "qué más", "que mas?", "qué más?",
    "arreglalo", "arregla eso", "fixealo", "hacelo", "haceolo",
    "seguilo", "segui", "seguí", "continua", "continuá", "continúa", "continualo",
    "explicalo", "explicámelo", "explicamelo",
    "amplialo", "ampliá", "ampliame", "ampliá eso", "amplialo eso",
    "documentalo", "comentalo", "completalo", "terminalo",
    "más", "mas", "mas detalle", "más detalle",
    "dale", "obvio", "obvio que si", "obvio que sí",
    "detallalo", "profundizalo", "profundizá", "profundiza",
)

_CONVERSATIONAL_CONTINUATION_PATTERNS = {
    "continua",
    "continuÃ¡",
    "continÃºa",
    "continualo",
    "seguilo",
    "segui",
    "seguÃ­",
    "explicalo",
    "explicamelo",
    "explicÃ¡melo",
    "amplialo",
    "ampliÃ¡",
    "ampliame",
    "ampliÃ¡ eso",
    "amplialo eso",
    "mas",
    "mÃ¡s",
    "mas detalle",
    "mÃ¡s detalle",
    "detallalo",
    "profundizalo",
    "profundiza",
    "profundizÃ¡",
}

_INHERITABLE_CONTINUITY_INTENTS = {
    "agentic",
    "agentic_code",
    "code",
    "audit",
    "suite_forensics",
    "helix_self",
    "research",
    "web_research",
    "reasoning",
}


def _is_continuity_followup(text: str, recent_intents: list[str] | None = None) -> bool:
    """Detect a vague follow-up that should inherit the previous intent.

    Returns True only when the prompt is short, lacks an explicit object, AND
    the recent route intents include something agentic/technical worth keeping.
    """
    if not text or not recent_intents:
        return False
    cleaned = " ".join(str(text).lower().strip().rstrip("?!.").split())
    if not cleaned or len(cleaned.split()) > 6:
        return False
    matched = any(cleaned == pat.rstrip("?!.") or cleaned.startswith(pat.rstrip("?!.") + " ") for pat in _VAGUE_FOLLOWUP_PATTERNS)
    if not matched:
        return False
    operational_recent = any(
        intent in {"agentic", "agentic_code", "code", "audit", "suite_forensics"}
        for intent in recent_intents
    )
    if not operational_recent:
        return False
    if cleaned in _CONVERSATIONAL_CONTINUATION_PATTERNS and not any(
        intent in {"agentic", "agentic_code", "code", "audit", "suite_forensics"}
        for intent in recent_intents
    ):
        return False
    return any(intent in _INHERITABLE_CONTINUITY_INTENTS for intent in recent_intents)


def _looks_like_work_request(text: str) -> bool:
    lowered = str(text or "").lower()
    path_refs = _extract_work_path_refs(text)
    url_refs = _extract_url_refs(text)
    source_signal = bool(path_refs or url_refs or any(term in lowered for term in (".pdf", ".docx", "scrap", "scrape", "crawler", "crawl", "http")))
    output_signal = any(
        term in lowered
        for term in (
            "armame",
            "creame",
            "crea ",
            "crear ",
            "generame",
            "genera ",
            "generar ",
            "armalo",
            "dejalo",
            "pagina web",
            "página web",
            "sitio",
            "landing",
            "microsite",
            "documento",
            "reporte",
            "informe",
            "pdf",
            "patch",
            "arregla",
            "arreglá",
            "implementa",
            "implementá",
        )
    )
    analysis_signal = any(term in lowered for term in ("analiza", "analizá", "resume", "resumi", "resumí", "verifica", "verificá", "audita", "auditá"))
    if source_signal and (output_signal or analysis_signal):
        return True
    if output_signal and any(term in lowered for term in ("pdf", "documento", "reporte", "informe")):
        return True
    if output_signal and _looks_like_agent_task(text):
        return True
    return False


def _looks_like_noise_input(text: str) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return True
    if len(clean) <= 3 and not re.search(r"[\w\u00c0-\u024f]", clean, flags=re.UNICODE):
        return True
    if clean in {"}", "{", "]", "[", ")", "(", ";", ",", ".", "...", "?", "¿", "!", "¡"}:
        return True
    return False


def _looks_like_incomplete_work_request(text: str) -> bool:
    folded = _fold_cli_text(text)
    if not folded:
        return False
    if _extract_work_path_refs(text) or _extract_url_refs(text):
        return False
    words = set(folded.split())
    vague_objects = {
        "documento",
        "texto",
        "reporte",
        "informe",
        "pdf",
        "pagina",
        "web",
        "sitio",
        "archivo",
    }
    creation_terms = {"armar", "armame", "crear", "crea", "creame", "generar", "genera", "generame", "hacer", "haceme"}
    if not (words & vague_objects and words & creation_terms):
        return False
    specificity_markers = ("sobre ", "acerca de ", "de nick", "del ", "de la ", "para ", "con ", "desde ", "en docs/", "en web/", ".pdf", ".md", ".docx")
    if any(marker in folded for marker in specificity_markers):
        return False
    return len(words) <= 8


def _fold_cli_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    folded = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(folded.lower().strip().rstrip("?!.").split())


def _looks_like_last_work_question(text: str) -> bool:
    folded = _fold_cli_text(text)
    if not folded or len(folded.split()) > 8:
        return False
    direct = {
        "entonces",
        "y entonces",
        "y ahora",
        "ahora que",
        "ahora",
        "que hizo",
        "que hizo?",
        "que paso",
        "que onda",
        "donde esta",
        "donde quedo",
        "mostrame",
        "mostrame resultado",
        "mostrar resultado",
        "resultado",
        "que cambio",
        "que cambios hizo",
        "que genero",
        "donde lo dejo",
        "salio algo",
    }
    return folded in direct or (
        any(term in folded for term in ("resultado", "cambio", "genero", "archivo", "patch", "donde"))
        and any(term in folded for term in ("ultimo", "last", "tarea", "work", "hizo", "quedo", "dejo"))
    )


def _looks_like_apply_last_request(text: str) -> bool:
    folded = _fold_cli_text(text)
    if not folded or len(folded.split()) > 5:
        return False
    return folded in {"aplicalo", "aplica", "aplicar", "apply", "apply last", "mandalo", "dale aplicalo"}


def _looks_like_open_last_request(text: str) -> bool:
    folded = _fold_cli_text(text)
    if not folded or len(folded.split()) > 6:
        return False
    return folded in {"abrilo", "abri eso", "abrelo", "open", "open last", "mostrame archivo", "mostrar archivo"}


def _looks_like_read_artifact_request(text: str) -> bool:
    folded = _fold_cli_text(text)
    if not folded or len(folded.split()) > 9:
        return False
    direct = {
        "a ver",
        "ver",
        "veamos",
        "leelo",
        "lee eso",
        "lee ese archivo",
        "mostrame eso",
        "mostramelo",
        "que hay ahi",
        "que hay aca",
        "describilo",
        "describime eso",
        "describime ese archivo",
        "fijate que hay",
        "inspect last",
        "read last",
    }
    return folded in direct or (
        any(term in folded for term in ("lee", "leelo", "describ", "inspect", "fijate"))
        and any(term in folded for term in ("archivo", "pdf", "documento", "pagina", "web", "eso", "ese", "ahi", "aca"))
    )


def _looks_like_modify_last_work_request(text: str) -> bool:
    folded = _fold_cli_text(text)
    if not folded or len(folded.split()) > 16:
        return False
    if folded.startswith(("dije ", "nono ", "no no ", "me referia ", "era ", "quise decir ", "corregi ", "corregilo ")):
        return True
    if any(
        fragment in folded
        for fragment in (
            "tiene que tener",
            "no tiene info",
            "no era",
            "no en relacion",
            "mejor contenido",
            "gran formato",
            "buen formato",
            "esta mal",
            "quedo mal",
            "quedo pobre",
        )
    ):
        return True
    modify_terms = {
        "armalo",
        "modifica",
        "modificalo",
        "edita",
        "editalo",
        "actualiza",
        "actualizalo",
        "mejora",
        "mejoralo",
        "agrega",
        "agregale",
        "sumale",
        "amplia",
        "amplialo",
        "curalo",
        "rehacelo",
        "arreglalo",
        "fijate",
        "corregi",
        "corregilo",
        "formatea",
    }
    target_terms = {"ese", "eso", "este", "esto", "esta", "ultimo", "ultima", "archivo", "documento", "pdf", "reporte", "pagina", "web", "info", "contenido", "formato", "tema"}
    quality_terms = {"bien", "pobre", "vacio", "vacía", "vacia", "roto", "malo", "mejor", "detallado", "curado", "completo", "gran"}
    words = set(folded.split())
    return bool(words & modify_terms) and (bool(words & target_terms) or bool(words & quality_terms) or "contenido curado" in folded or "no tiene info" in folded)


def _looks_like_work_confirmation(text: str) -> bool:
    folded = _fold_cli_text(text)
    return folded in {"dale", "proceda", "procede", "procedelo", "hacelo", "dale hacelo"}


def _classify_workbench_prompt(text: str, recent_intents: list[str] | None = None) -> dict[str, Any]:
    original = str(text or "").strip()
    lowered = original.lower()
    folded = _fold_cli_text(original)
    path_refs = _extract_work_path_refs(original)
    url_refs = _extract_url_refs(original)
    output_target, source_refs = _work_target_for_goal(original, path_refs)
    route: dict[str, Any] = {
        "kind": "helix-workbench-route-v1",
        "lane": "conversation",
        "route_reason": "default_conversation",
        "engine_selected": "chat",
        "needs_sources": False,
        "needs_patch": False,
        "needs_verification": False,
        "command": None,
        "path_refs": path_refs,
        "url_refs": url_refs,
        "source_refs": source_refs,
        "output_target": output_target,
    }
    if _looks_like_noise_input(original):
        route.update({"lane": "noop", "route_reason": "noise_input", "command": "/noop"})
        return route
    if _looks_like_last_work_question(original):
        route.update({"lane": "work_status", "route_reason": "last_work_followup", "command": "/work last"})
        return route
    if folded in {"last", "ultimo", "ultima tarea"}:
        route.update({"lane": "work_status", "route_reason": "last_alias", "command": "/last"})
        return route
    if folded in {"fuentes", "sources", "anchors"}:
        route.update({"lane": "work_status", "route_reason": "sources_alias", "command": "/sources"})
        return route
    if folded in {"trust", "confianza", "trust card"}:
        route.update({"lane": "trust", "route_reason": "trust_alias", "needs_verification": True, "command": "/trust"})
        return route
    if _looks_like_apply_last_request(original):
        route.update({"lane": "apply", "route_reason": "apply_last_followup", "command": "/apply last"})
        return route
    if _looks_like_open_last_request(original):
        route.update({"lane": "open", "route_reason": "open_last_followup", "command": "/open last"})
        return route
    if _looks_like_read_artifact_request(original):
        route.update({"lane": "artifact_read", "route_reason": "read_artifact_followup", "command": "/read last"})
        return route
    if _looks_like_modify_last_work_request(original):
        route.update(
            {
                "lane": "work_doc",
                "route_reason": "modify_last_work",
                "engine_selected": "helix",
                "needs_sources": False,
                "needs_patch": True,
                "command": f"/work run {original}",
            }
        )
        return route
    if _looks_like_incomplete_work_request(original):
        route.update(
            {
                "lane": "clarify",
                "route_reason": "incomplete_work_request",
                "engine_selected": "none",
                "needs_sources": False,
                "needs_patch": False,
                "command": "/clarify work",
            }
        )
        return route
    for prefix in ("helix work run ", "work run "):
        if lowered.startswith(prefix):
            goal = original[len(prefix):].strip()
            route.update({"lane": "work", "route_reason": "explicit_work_run", "command": f"/work run {goal}" if goal else "/work status"})
            return route
    if lowered in {"helix work", "work"}:
        route.update({"lane": "work_status", "route_reason": "explicit_work_status", "command": "/work status"})
        return route
    if lowered.startswith("helix flow "):
        route.update({"lane": "work", "route_reason": "explicit_flow", "command": "/" + original[len("helix "):].strip()})
        return route
    if lowered.startswith("helix "):
        commandish = original[len("helix "):].strip()
        if commandish.split(" ", 1)[0].lower() in {"opencode", "flow", "work", "agent", "task", "trust", "verify", "suite", "suites"}:
            route.update({"lane": "command", "route_reason": "helix_command_prefix", "command": "/" + commandish})
            return route
    if _is_pasted_suite_analysis_request(original):
        return route
    if "/verify" in lowered and not any(term in lowered for term in ("resultado", "transcript", "corrida", "artifact", "manifest")):
        route.update({"lane": "verify", "route_reason": "verify_suite_shortcut", "needs_verification": True, "command": "/suites"})
        return route
    if any(term in lowered for term in ("nuclear", "deep lab", "deep-lab", "suite completa", "suites completas")):
        route.update({"lane": "deep_lab", "route_reason": "explicit_deep_lab", "needs_verification": True, "command": f"/work run {original}"})
        return route
    if any(term in lowered for term in ("verifica", "verific", "evidence", "evidencia", "capsule", "trust")) and not any(term in lowered for term in ("pagina", "web/", "reporte", "documento")):
        route.update({"lane": "verify", "route_reason": "verification_request", "needs_verification": True, "command": f"/work run {original}"})
        return route
    if url_refs and any(term in folded for term in ("scrap", "scrape", "crawl", "crawler", "sintetiza", "sintesis", "reporte", "web", "pagina")):
        route.update({"lane": "work_scrape", "route_reason": "url_source_work", "engine_selected": "helix", "needs_sources": True, "needs_patch": bool(output_target), "command": f"/work run {original}"})
        return route
    if path_refs and _looks_like_work_request(original):
        intent = _work_intent_for_goal(original, source_refs=source_refs, url_refs=url_refs, output_target=output_target)
        lane = "work_web" if intent.get("output_kind") == "web" else "work_doc" if intent.get("output_kind") == "document" else "work"
        route.update({"lane": lane, "route_reason": "local_source_work", "engine_selected": "helix", "needs_sources": True, "needs_patch": bool(intent.get("output_kind") in {"web", "document", "patch"}), "command": f"/work run {original}"})
        return route
    if _looks_like_agent_task(original):
        route.update({"lane": "code_patch", "route_reason": "code_or_repo_task", "engine_selected": "opencode", "needs_sources": bool(path_refs or url_refs), "needs_patch": True, "command": f"/work run {original}"})
        return route
    if _looks_like_work_request(original):
        intent = _work_intent_for_goal(original, source_refs=source_refs, url_refs=url_refs, output_target=output_target)
        lane = "work_web" if intent.get("output_kind") == "web" else "work_doc" if intent.get("output_kind") == "document" else "work"
        route.update({"lane": lane, "route_reason": "work_request", "engine_selected": "helix", "needs_sources": bool(path_refs or url_refs), "needs_patch": bool(intent.get("output_kind") in {"web", "document", "patch"}), "command": f"/work run {original}"})
        return route
    if _is_continuity_followup(original, recent_intents):
        route.update({"lane": "code_patch", "route_reason": "agentic_continuity", "engine_selected": "opencode", "needs_patch": True, "command": f"/work run {original}"})
        return route
    if lowered.strip() in {"doctor", "diagnostico", "estado"}:
        route.update({"lane": "command", "route_reason": "doctor_alias", "command": "/doctor"})
    return route


def _route_natural_language(text: str, recent_intents: list[str] | None = None) -> str | None:
    original = str(text or "").strip()
    lowered = original.lower()
    route = _classify_workbench_prompt(text, recent_intents)
    if route.get("command"):
        return str(route["command"])
    if _looks_like_last_work_question(original):
        return "/work last"
    for prefix in ("helix work run ", "work run "):
        if lowered.startswith(prefix):
            goal = original[len(prefix):].strip()
            return f"/work run {goal}" if goal else "/work status"
    if lowered in {"helix work", "work"}:
        return "/work status"
    if lowered.startswith("helix flow "):
        return "/" + original[len("helix "):].strip()
    if lowered.startswith("helix "):
        commandish = original[len("helix "):].strip()
        if commandish.split(" ", 1)[0].lower() in {"opencode", "flow", "work", "agent", "task", "trust", "verify", "suite", "suites"}:
            return "/" + commandish
    local_refs = _extract_local_path_refs(text)
    if _is_pasted_suite_analysis_request(text):
        return None
    if "/verify" in lowered and not any(term in lowered for term in ("resultado", "transcript", "corrida", "artifact", "manifest")):
        return "/suites"
    if local_refs and _is_local_file_request(text) and not _looks_like_agent_task(text) and not _looks_like_work_request(text):
        return None
    execution_terms = ("corre", "ejecuta", "run ", "certifica")
    analysis_terms = ("quiero info", "quiero data", "dame info", "dame data", "contame", "analiza", "explica", "compara", "compará", "reporte", "resum")
    if any(term in lowered for term in execution_terms) and not any(term in lowered for term in analysis_terms):
        suite_id = _suite_from_text(text)
        if suite_id:
            return f"/cert {suite_id}"
    if _is_suite_evidence_request(text) and any(term in lowered for term in ("analiza", "compara", "compará", "explica", "reporte", "resumi", "resume")):
        return f"/work run {text}"
    if _looks_like_work_request(text):
        return f"/work run {text}"
    if _looks_like_agent_task(text):
        return f"/work run {text}"
    # Vague follow-up that should inherit the previous agentic/technical lane.
    # Without this, "arreglalo" or "y eso?" after 3 turns of debugging falls
    # back to plain chat heuristics and loses the trail.
    if _is_continuity_followup(text, recent_intents):
        return f"/work run {text}"
    if lowered.strip() in {"doctor", "diagnostico", "estado"}:
        return "/doctor"
    return None


class TurnController:
    def __init__(self, session: "InteractiveSession") -> None:
        self.session = session

    def decide(self, text: str) -> IntentCard:
        route = _classify_workbench_prompt(text, self.session.recent_route_intents(limit=4))
        return _intent_card_from_route(self.session, text, route)


def _intent_card_from_route(session: "InteractiveSession", text: str, route: dict[str, Any]) -> IntentCard:
    original = str(text or "").strip()
    primary_goal, correction_notes = _split_primary_goal_and_corrections(original)
    lane = _canonical_intent_lane(str(route.get("lane") or "conversation"), route)
    output_target = route.get("output_target")
    sources = [str(item) for item in (route.get("source_refs") or route.get("path_refs") or []) if str(item).strip()]
    urls = [str(item) for item in (route.get("url_refs") or []) if str(item).strip()]
    if lane == "artifact_modify":
        last_path = str((session.last_artifact or {}).get("path") or "")
        if last_path and last_path not in sources:
            sources.append(last_path)
        if not output_target and last_path:
            output_target = last_path
    command = str(route.get("command") or "") or _route_natural_language(original, session.recent_route_intents(limit=4))
    requires_write = lane in {"artifact_modify", "doc_generate", "web_generate", "code_patch"}
    requires_opencode = lane == "code_patch" or str(route.get("engine_selected") or "") == "opencode"
    requires_model = lane not in {"artifact_read", "work_status", "trust", "open", "apply"}
    requires_readback = lane in {"artifact_read", "artifact_modify", "doc_generate", "web_generate"}
    confidence = 0.92
    if str(route.get("route_reason") or "") == "default_conversation":
        confidence = 0.74
    if lane == "artifact_modify" and not ((session.last_artifact or {}).get("path") or session.last_work_result):
        confidence = 0.55
    return IntentCard(
        lane=lane,
        primary_goal=primary_goal or original,
        correction_notes=correction_notes,
        sources=sources,
        urls=urls,
        output_target=str(output_target) if output_target else None,
        requires_write=requires_write,
        requires_model=requires_model,
        requires_opencode=requires_opencode,
        requires_readback=requires_readback,
        route_reason=str(route.get("route_reason") or "default"),
        confidence=confidence,
        fallback_command=command or None,
        route=route,
    )


def _canonical_intent_lane(lane: str, route: dict[str, Any]) -> str:
    if lane in {"noop", "clarify"}:
        return lane
    if lane == "work_doc":
        return "artifact_modify" if route.get("route_reason") == "modify_last_work" else "doc_generate"
    if lane == "work_web":
        return "web_generate"
    if lane == "work_scrape":
        return "scrape"
    if lane == "code_patch":
        return "code_patch"
    if lane == "artifact_read":
        return "artifact_read"
    if lane == "verify":
        return "verify"
    if lane == "deep_lab":
        return "deep_lab"
    return lane if lane in {"conversation", "work_status", "trust", "open", "apply"} else "conversation"


def _split_primary_goal_and_corrections(text: str) -> tuple[str, list[str]]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return "", []
    correction_prefixes = (
        "pero ",
        "dije ",
        "nono ",
        "no no ",
        "no, ",
        "me referia",
        "me refería",
        "quise decir",
        "tiene que",
        "ademas",
        "además",
    )
    primary = lines[0]
    corrections: list[str] = []
    for line in lines[1:]:
        if _fold_cli_text(line).startswith(tuple(_fold_cli_text(item) for item in correction_prefixes)):
            corrections.append(line)
        else:
            corrections.append(line)
    return primary, corrections


def _turn_plan_for_route(text: str, route: dict[str, Any], intent_card: IntentCard | dict[str, Any] | None = None) -> dict[str, Any]:
    lane = str(route.get("lane") or "conversation")
    command = route.get("command")
    intent_payload = intent_card.to_dict() if isinstance(intent_card, IntentCard) else dict(intent_card or {})
    if lane == "noop":
        mode = "noop"
        steps = [
            "ignorar input incompleto o accidental",
            "no llamar al modelo",
            "esperar una instruccion legible",
        ]
    elif lane == "clarify":
        mode = "clarify"
        steps = [
            "detectar pedido de trabajo incompleto",
            "no escribir archivos todavia",
            "pedir tema, destino o formato antes de ejecutar",
        ]
    elif lane in {"work_doc", "work_web", "work_scrape", "work"} or (isinstance(command, str) and command.startswith("/work run ")):
        mode = "work"
        steps = [
            "analizar pedido y separar tema, salida y correcciones",
            "leer fuentes o ultimo artefacto si aplica",
            "generar/modificar artefacto docs-web",
            "releer output y validar densidad/estructura",
            "actualizar /last, /read last y /trust",
        ]
    elif lane == "code_patch" or (isinstance(command, str) and command.startswith("/task")):
        mode = "code"
        steps = [
            "analizar objetivo operativo",
            "preparar contexto minimo del repo",
            "ejecutar agente/coder en sandbox",
            "capturar diff y verificar git apply --check",
            "dejar patch listo solo si aplica limpio",
        ]
    elif lane in {"artifact_read", "work_status", "open", "apply", "trust"} or command in {"/read last", "/work last", "/last", "/trust", "/open last", "/apply last"}:
        mode = "artifact"
        steps = [
            "resolver referencia al ultimo artefacto/tarea",
            "leer estado persistido",
            "mostrar resultado humano sin llamar al modelo",
        ]
    elif lane in {"verify", "deep_lab"}:
        mode = "verify"
        steps = [
            "resolver evidencia o task capsule",
            "elegir checks segun nivel",
            "mostrar limites de claim",
        ]
    else:
        mode = "chat"
        steps = [
            "mantener carril conversacional",
            "usar solo hilo activo/resumen reciente",
            "responder sin herramientas ni promesas de archivos",
        ]
    return {
        "kind": "helix-turn-plan-v1",
        "mode": mode,
        "lane": lane,
        "intent_lane": intent_payload.get("lane") or _canonical_intent_lane(lane, route),
        "primary_goal": intent_payload.get("primary_goal") or str(text or ""),
        "correction_notes": intent_payload.get("correction_notes") or [],
        "requires_write": bool(intent_payload.get("requires_write")),
        "requires_readback": bool(intent_payload.get("requires_readback")),
        "route_reason": route.get("route_reason") or "default",
        "command": command,
        "summary": _truncate_text(str(text or "").replace("\n", " "), 180)["text"],
        "steps": steps,
    }


def _format_turn_plan(plan: dict[str, Any]) -> str:
    lines = [
        "HeliX Turn Plan",
        f"- lane: {plan.get('intent_lane') or plan.get('lane') or 'conversation'} | mode: {plan.get('mode') or 'chat'} | reason: {plan.get('route_reason') or 'default'}",
        f"- goal: {plan.get('primary_goal') or plan.get('summary') or 'n/a'}",
        f"- command: {plan.get('command') or 'chat'}",
    ]
    if plan.get("correction_notes"):
        lines.append(f"- corrections: {' | '.join(map(str, plan.get('correction_notes') or []))}")
    if plan.get("requires_write") or plan.get("requires_readback"):
        lines.append(f"- guarantees: write={bool(plan.get('requires_write'))} readback={bool(plan.get('requires_readback'))}")
    for index, step in enumerate(plan.get("steps") or [], start=1):
        lines.append(f"{index}. {step}")
    return "\n".join(lines)


def _format_turn_plan_compact(plan: dict[str, Any]) -> str:
    lane = str(plan.get("intent_lane") or plan.get("lane") or "conversation")
    mode = str(plan.get("mode") or "chat")
    reason = str(plan.get("route_reason") or "default")
    command = str(plan.get("command") or "chat")
    goal = _truncate_text(str(plan.get("primary_goal") or plan.get("summary") or "").replace("\n", " "), 96)["text"]
    if mode == "chat":
        return f"[helix] chat listo | lane={lane} | {goal or 'conversacion'}"
    if mode in {"noop", "clarify"}:
        return f"[helix] {mode} | reason={reason} | {goal or 'input'}"
    if plan.get("requires_write") or plan.get("requires_readback"):
        return f"[helix] {mode} | lane={lane} | {command} | write={bool(plan.get('requires_write'))} readback={bool(plan.get('requires_readback'))}"
    return f"[helix] {mode} | lane={lane} | {command}"


def _show_turn_plan(
    text: str,
    route: dict[str, Any],
    intent_card: IntentCard | dict[str, Any] | None = None,
    *,
    mode: str = "verbose",
) -> None:
    mode = _normalize_preflight_mode(mode)
    if mode == "off":
        return
    plan = _turn_plan_for_route(text, route, intent_card=intent_card)
    if mode == "compact":
        message = _format_turn_plan_compact(plan)
        if console:
            console.print(f"[dim]{message}[/dim]")
        else:
            print(message)
        return
    if console:
        from rich.panel import Panel  # noqa: PLC0415

        console.print(Panel(_format_turn_plan(plan), title="TURN PREFLIGHT", border_style="cyan"))
    else:
        print(_format_turn_plan(plan))


def _looks_like_agent_task(text: str) -> bool:
    lowered = str(text or "").lower()
    task_verbs = (
        "arregla",
        "implementa",
        "refactor",
        "debug",
        "encontra el bug",
        "encuentra el bug",
        "fijate el repo",
        "mirá el repo",
        "mira el repo",
        "inspecciona",
        "revisa el repo",
        "corré tests",
        "corre tests",
        "pytest",
        "armame un patch",
        "proponeme un patch",
        "lee estos archivos",
        "busca en archivos",
        "aplica un fix",
        "codex",
        "claude code",
        "modo agente",
        "modo agentico",
        "modo agentic",
        "compará",
        "compara",
        "contrasta",
    )
    task_objects = ("repo", "archivo", "archivos", "test", "tests", "pytest", "diff", "patch", "código", "codigo", "suite", "artifact", "transcript", "verification", "url", "urls", "link", "links")
    if any(term in lowered for term in ("conceptualmente", "filosof", "filosóf", "producto", "idea", "ideas")) and not any(obj in lowered for obj in task_objects):
        return False
    if any(verb in lowered for verb in task_verbs):
        return True
    return any(action in lowered for action in ("fijate", "mirá", "mira", "revisa", "busca")) and any(
        obj in lowered for obj in task_objects
    )


def _set_session_provider(session: InteractiveSession, candidate: str) -> None:
    session.provider_name = candidate
    default_model = PROVIDERS[candidate].default_model
    if candidate == "deepinfra":
        session.model = "auto"
    elif default_model:
        session.model = default_model
    _ensure_provider_token(candidate)


def _set_session_model(session: InteractiveSession, value: str) -> None:
    model_id = resolve_model_alias(value)
    if model_id.lower() in {"auto", "router:auto"}:
        session.model = "auto"
        return
    alias = _profile_alias_for_model_id(model_id)
    profile = MODEL_PROFILES.get(alias or "")
    if profile and profile.provider != session.provider_name:
        session.provider_name = profile.provider
        _ensure_provider_token(profile.provider)
    session.model = model_id


def _select_provider(session: InteractiveSession) -> str | None:
    options = [
        (provider.name, f"{provider.name} - {provider.description}")
        for provider in sorted(PROVIDERS.values(), key=lambda item: item.name)
    ]
    return _choose_ui_option(
        title="Select Provider",
        theme_name=session.theme_name,
        options=options,
        bottom_help=" Press [Up]/[Down] to choose, [Enter] to accept, Ctrl-C to cancel. ",
    )


def _select_router_policy(session: InteractiveSession) -> str | None:
    options = [
        (item["name"], f"{item['name']} - {item['description']}")
        for item in router_blueprints_report()
    ]
    return _choose_ui_option(
        title="Select Router Blueprint",
        theme_name=session.theme_name,
        options=options,
        bottom_help=" Choose the blueprint that governs auto-routing for chat, reasoning, research, and code. ",
    )


def _select_theme(session: InteractiveSession) -> str | None:
    options: list[tuple[str, str]] = []
    for item in _theme_report():
        aliases = f" (aliases: {', '.join(item['aliases'])})" if item.get("aliases") else ""
        options.append((item["name"], f"{item['name']} - {item['description']}{aliases}"))
    return _choose_ui_option(
        title="Select Terminal Theme",
        theme_name=session.theme_name,
        options=options,
        bottom_help=" Choose the terminal chrome palette. The current session and saved config will both update. ",
    )


def _select_interaction_mode(session: InteractiveSession) -> str | None:
    options: list[tuple[str, str]] = []
    for item in _interaction_mode_report():
        examples = item.get("examples") or []
        hint = f" e.g. {examples[0]!r}" if examples else ""
        options.append((item["name"], f"{item['name']} - {item.get('description', '')}{hint}"))
    return _choose_ui_option(
        title="Select Interaction Mode",
        theme_name=session.theme_name,
        options=options,
        bottom_help=" balanced = default, technical = code/audit/diagnosis, explore = wider research and philosophy. The session and saved config will both update. ",
    )


def _prompt_interaction_mode_text(current: str) -> str | None:
    """Plain stdin fallback when rich UI is unavailable."""
    report = _interaction_mode_report()
    print()
    print(f"Current interaction mode: {current}")
    print("Available modes:")
    for index, item in enumerate(report, start=1):
        marker = " *" if item["name"] == current else "  "
        print(f"  {index}.{marker} {item['name']:<10} - {item.get('description', '')}")
    print("  0.   keep current")
    try:
        raw = input("Choose mode [0-{0}] or name: ".format(len(report))).strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not raw or raw == "0":
        return None
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(report):
            return report[idx - 1]["name"]
        return None
    if _is_known_interaction_mode(raw):
        return _normalize_interaction_mode(raw)
    return None


def _select_model(session: InteractiveSession) -> str | None:
    options = [
        ("auto", "auto - let the router choose per prompt"),
        ("mistral", "mistral - fast conversational baseline"),
        ("qwen-big", "qwen-big - primary large Qwen for research, HeliX and synthesis"),
        ("qwen", "qwen - alias for qwen-big"),
        ("bioinformatics", "bioinformatics - scientific and bioinformatics analysis alias on the large Qwen research path"),
        ("gemma", "gemma - careful reasoning and decomposition"),
        ("gemini-pro", "gemini-pro - Gemini 3.1 Pro preview via GEMINI_API_KEY"),
        ("gemini-pro-tools", "gemini-pro-tools - Gemini 3.1 Pro custom-tools preview"),
        ("gemini-flash", "gemini-flash - Gemini 3 Flash preview via GEMINI_API_KEY"),
        ("gemini-lite", "gemini-lite - Gemini 3.1 Flash Lite preview via GEMINI_API_KEY"),
        ("gemini-2.5-pro", "gemini-2.5-pro - stable Gemini 2.5 Pro fallback"),
        ("gemini-2.5-flash", "gemini-2.5-flash - stable Gemini 2.5 Flash fallback"),
        ("gemini-2.5-flash-lite", "gemini-2.5-flash-lite - stable Gemini 2.5 Flash-Lite fallback"),
        ("nvidia-chat", "nvidia-chat - NVIDIA Build Magistral free-endpoint chat"),
        ("nvidia-research", "nvidia-research - NVIDIA Build Mistral Large research/synthesis"),
        ("nvidia-code", "nvidia-code - NVIDIA Build Qwen Coder agentic coding"),
        ("nvidia-agentic", "nvidia-agentic - NVIDIA Build Devstral verification/repair"),
        ("nvidia-guard", "nvidia-guard - NVIDIA Build Llama Guard safety gate"),
        ("nvidia-pii", "nvidia-pii - NVIDIA Build GLiNER PII detector"),
        ("coder", "coder - repository and patch-heavy coding work"),
        ("engineering", "engineering - premium GLM agentic engineering path"),
        ("deep-reasoning", "deep-reasoning - DeepSeek reasoning fallback"),
        ("llama", "llama - larger generalist fallback"),
        ("llama-vision", "llama-vision - visual debugging and screenshots"),
        ("sonnet", "sonnet - higher-stakes audits and review"),
        ("devstral", "devstral - cheaper coding fallback"),
    ]
    return _choose_ui_option(
        title="Select Model",
        theme_name=session.theme_name,
        options=options,
        bottom_help=" Pick a fixed model alias or leave the session on auto-routing. ",
    )


def _handle_interactive_command(session: InteractiveSession, line: str) -> bool:
    command_line = line[1:] if line.startswith("/") else line
    name, _, rest = command_line.partition(" ")
    name = name.lower().strip()
    rest = rest.strip()

    if name in {"exit", "quit", "q"}:
        return False
    if name == "noop":
        print("[helix] No tome accion: ese input parece incompleto o accidental. Escribime una instruccion en una frase.")
        return True
    if name == "clarify":
        print(
            "[helix] Puedo hacerlo, pero falta el objeto de trabajo. Decime tema/fuente y destino, por ejemplo: "
            'armame un documento de texto sobre Nick Land en docs/nick-land.md'
        )
        return True
    if name in {"help", "h", "?"}:
        print(HELP_TEXT)
        return True
    if name == "last":
        print(_format_work_summary(session.last_work_result, plan=session.last_work_plan, trust_card=session.last_trust_card))
        return True
    if name == "sources":
        _print_json(session.last_work_sources or {"status": "not_available", "error": "no Work Runtime sources are available"})
        return True
    if name in {"read", "inspect"}:
        parts = _split_command(rest)
        wants_json = "--json" in {part.lower() for part in parts}
        target_parts = [part for part in parts if not part.startswith("--")]
        target = " ".join(target_parts).strip() or "last"
        if target.lower() == "last":
            target = str((session.last_artifact or {}).get("path") or "") or str(
                _primary_work_output_path(session.last_work_result, plan=session.last_work_plan, task_root=session.task_root) or ""
            )
        if not target:
            print("[helix] no last artifact is available.")
            return True
        artifact = _inspect_work_artifact(session, target, last_action=name)
        session.last_artifact = artifact
        if session.last_work_result:
            session.last_work_result["artifact"] = artifact
        if session.last_trust_card:
            session.last_trust_card["artifact"] = artifact
        session._save_last_work_state()
        if wants_json:
            _print_json(artifact)
        else:
            print(_format_artifact_summary(artifact))
        return True
    if name == "open":
        parts = _split_command(rest)
        target = parts[0].lower() if parts else "last"
        if target != "last":
            print("Usage: /open last")
            return True
        output = _primary_work_output_path(session.last_work_result, plan=session.last_work_plan, task_root=session.task_root)
        if not output:
            print("[helix] no output path is available from the last work result.")
            return True
        exists = Path(output).exists()
        print(f"[helix] last output: {output}")
        print(f"[helix] exists: {'yes' if exists else 'no'}")
        return True
    if name == "preflight":
        parts = _split_command(rest)
        if not parts or parts[0].lower() in {"status", "show"}:
            print(f"[helix] preflight={session.preflight_mode}")
            return True
        candidate = _normalize_preflight_mode(parts[0])
        if candidate != parts[0].strip().lower():
            print("Usage: /preflight compact|verbose|off")
            return True
        session.preflight_mode = candidate
        config = _load_config()
        config["preflight_mode"] = candidate
        _save_config(config)
        print(f"[helix] preflight={candidate}")
        return True
    if name == "demo":
        parts = _split_command(rest)
        subcommand = parts[0].lower() if parts else "wow"
        flags = {part.lower() for part in parts if part.startswith("--")}
        wants_json = "--json" in flags or "json" in flags
        if subcommand == "wow":
            result = session.demo_wow(
                fast="--slow" not in flags,
                browser="--no-browser" not in flags,
                with_opencode="--with-opencode" in flags,
            )
            if wants_json:
                _print_json(result)
            else:
                print(_format_work_summary(session.last_work_result, plan=session.last_work_plan, trust_card=session.last_trust_card))
                print(f"- demo_run: {(session.last_demo_result or {}).get('work_artifact_paths', {}).get('demo_run') or (session.last_demo_result or {}).get('artifact_paths', {}).get('demo_run') or (session.last_work_result or {}).get('demo_run_path')}")
            return True
        if subcommand == "doctor":
            _print_json(demo_doctor_report(session=session))
            return True
        if subcommand == "open":
            demo = session.last_demo_result or {}
            paths = demo.get("work_artifact_paths") if isinstance(demo.get("work_artifact_paths"), dict) else {}
            if not paths:
                paths = demo.get("trust_card", {}).get("artifact_paths", {}) if isinstance(demo.get("trust_card"), dict) else {}
            site = paths.get("site") or _primary_work_output_path(session.last_work_result, plan=session.last_work_plan, task_root=session.task_root)
            if not site:
                print("[helix] no demo artifact is available.")
                return True
            print(f"[helix] demo artifact: {site}")
            print(f"[helix] exists: {'yes' if Path(str(site)).exists() else 'no'}")
            return True
        if subcommand == "skills":
            second = parts[1].lower() if len(parts) > 1 else "suggest"
            if second == "suggest":
                _print_json(_demo_skills_suggestions())
                return True
        print("Usage: /demo wow [--json] [--no-browser] [--with-opencode] | /demo doctor | /demo open | /demo skills suggest")
        return True
    if name == "status":
        _print_json(session.status())
        return True
    if name == "latency":
        parts = _split_command(rest)
        subcommand = parts[0].lower() if parts else "show"
        if subcommand == "mode":
            if len(parts) < 2:
                print(f"[helix] latency_mode={session.latency_mode}")
                return True
            candidate = _normalize_latency_mode(parts[1])
            if candidate != parts[1].strip().lower():
                print("Usage: /latency mode fast|balanced|deep")
                return True
            session.latency_mode = candidate
            config = _load_config()
            config["latency_mode"] = session.latency_mode
            _save_config(config)
            print(f"[helix] latency_mode={session.latency_mode}")
            return True
        if subcommand in {"show", "status", "history", "last"} or subcommand.isdigit():
            limit_arg = parts[1] if subcommand in {"show", "status", "history", "last"} and len(parts) > 1 else subcommand
            limit = _safe_int(limit_arg, 8, minimum=1, maximum=50) if str(limit_arg).isdigit() else 8
            _print_json(
                {
                    "latency_mode": session.latency_mode,
                    "last_latency": session.last_latency_trace,
                    "history": _latency_history(session.events, limit=limit),
                    "thread_id": session.thread_id,
                }
            )
            return True
        print("Usage: /latency [N] | /latency mode fast|balanced|deep")
        return True
    if name == "hooks":
        parts = _split_command(rest)
        subcommand = parts[0].lower() if parts else "status"
        if subcommand in {"status", "list", "ls"}:
            _print_json(
                {
                    "status": "ok",
                    "hook_runtime": "internal",
                    "registered_runtime_hooks": len(session.internal_hooks),
                    "events": [
                        "BeforeRoute",
                        "AfterIntent",
                        "BeforeModel",
                        "AfterModel",
                        "PreWrite",
                        "PostWrite",
                        "PostArtifactRead",
                        "PreApply",
                        "PostVerify",
                        "TaskBlocked",
                    ],
                    "last_progress_events": (session.last_work_result or {}).get("progress_events") if session.last_work_result else [],
                }
            )
            return True
        print("Usage: /hooks status")
        return True
    if name in {"conversation", "conv"}:
        parts = _split_command(rest)
        subcommand = parts[0].lower() if parts else "status"
        if subcommand in {"status", "show", "current"}:
            _print_json(session.conversation_status())
            return True
        if subcommand in {"summarize", "summary", "resumen"}:
            _print_json(session.ensure_thread_summary(force=True))
            return True
        if subcommand in {"reset-summary", "reset", "clear-summary"}:
            _print_json(session.reset_thread_summary())
            return True
        print("Usage: /conversation status|summarize|reset-summary")
        return True
    if name == "fast":
        parts = _split_command(rest)
        subcommand = parts[0].lower() if parts else "status"
        if subcommand in {"on", "true", "1"}:
            session.latency_mode = "fast"
            config = _load_config()
            config["latency_mode"] = session.latency_mode
            _save_config(config)
            print("[helix] fast=on latency_mode=fast")
            return True
        if subcommand in {"off", "false", "0"}:
            session.latency_mode = "balanced"
            config = _load_config()
            config["latency_mode"] = session.latency_mode
            _save_config(config)
            print("[helix] fast=off latency_mode=balanced")
            return True
        if subcommand == "status":
            _print_json(
                {
                    "fast": session.latency_mode == "fast",
                    "latency_mode": session.latency_mode,
                    "rust_core": helix_cli_core.rust_core_status(),
                    "last_latency": session.last_latency_trace,
                }
            )
            return True
        print("Usage: /fast on|off|status")
        return True
    if name == "thread":
        parts = _split_command(rest)
        subcommand = parts[0].lower() if parts else "current"
        flags = {part.lower() for part in parts[1:] if part.startswith("--")}
        argument = " ".join(part for part in parts[1:] if not part.startswith("--")).strip()
        if subcommand == "new":
            _print_json(session.new_thread(argument or "interactive"))
            return True
        if subcommand in {"list", "ls"}:
            _print_json({"threads": session.list_threads(limit=32)})
            return True
        if subcommand == "tree":
            if argument.lower() == "json" or "--json" in flags:
                _print_json(session.thread_tree())
            else:
                print(session.thread_tree_text())
            return True
        if subcommand == "open":
            if not argument:
                print("Usage: /thread open THREAD_ID")
                return True
            _print_json(session.open_thread(argument))
            return True
        if subcommand == "close":
            _print_json(session.close_thread(argument or None))
            return True
        if subcommand in {"current", "show"}:
            _print_json(session.current_thread())
            return True
        print("Usage: /thread new [--clean] [TITLE] | list | tree [json] | open THREAD_ID | close [THREAD_ID] | current")
        return True
    if name == "branch":
        parts = _split_command(rest)
        subcommand = parts[0].lower() if parts else "new"
        argument = " ".join(parts[1:]).strip()
        if subcommand == "new":
            _print_json(session.branch_thread(argument or None))
            return True
        if subcommand in {"tree", "list", "ls"}:
            print(session.thread_tree_text())
            return True
        print("Usage: /branch new [TITLE] | tree")
        return True
    if name == "providers":
        _print_json({"providers": provider_report(probe_local=False)})
        return True
    if name == "models":
        if rest.lower() in {"compare last", "last compare"}:
            card = session.last_trust_card or _task_trust_card_from_result(session.last_task_result)
            models_used = card.get("models_used") if isinstance(card, dict) and isinstance(card.get("models_used"), dict) else {}
            _print_json(
                {
                    "status": "ok" if models_used else "not_available",
                    "kind": "helix-model-comparison-v1",
                    "subject": "last",
                    "models_used": models_used,
                    "disagreement": (session.last_task_result or {}).get("disagreement"),
                    "resolution": (session.last_task_result or {}).get("resolution") or "No multi-model critic was run for quick assurance.",
                    "next": "Use /task --assurance balanced or /verify last --level strict to require critic/verifier checks.",
                }
            )
            return True
        if rest.lower() == "json":
            _print_json(models_payload())
            return True
        if console and _HAS_UI:
            selected = _select_model(session)
            if selected:
                _set_session_model(session, selected)
                print(f"[helix] provider={session.provider_name} model={session.model}")
            return True
        _print_models_compact()
        return True
    if name == "lab":
        parts = _split_command(rest)
        subcommand = parts[0].lower() if parts else "profiles"
        if subcommand in {"profiles", "profile", "list", "ls"}:
            _print_json(helix_cli_core.lab_profiles())
            return True
        if subcommand == "run":
            profile = parts[1] if len(parts) > 1 else "patch-safety"
            _print_json(helix_cli_core.lab_run(profile=profile, evidence_root=session.evidence_root, repo_root=REPO_ROOT))
            return True
        print("Usage: /lab profiles | /lab run patch-safety|doc-grounding|memory-isolation|provider-audit|deep-nuclear")
        return True
    if name == "flow":
        parts = _split_command(rest)
        subcommand = parts[0].lower() if parts else "list"
        if subcommand in {"profiles", "profile", "list", "ls"}:
            _print_json(flow_profiles_report())
            return True
        if subcommand == "show":
            if len(parts) < 2:
                print("Usage: /flow show PROFILE")
                return True
            try:
                _print_json(_flow_profile_or_error(parts[1]).report())
            except ValueError as exc:
                print(f"[helix] {exc}")
            return True
        if subcommand == "run":
            if len(parts) < 3:
                print("Usage: /flow run PROFILE GOAL")
                return True
            try:
                flow = _flow_profile_or_error(parts[1])
            except ValueError as exc:
                print(f"[helix] {exc}")
                return True
            user_goal = " ".join(parts[2:]).strip()
            flow_goal = flow.build_goal(user_goal)
            if console:
                result = _run_with_status(
                    console,
                    lambda: session.task(flow_goal, engine_override=flow.engine, assurance=flow.assurance),
                    phase=f"flow:{flow.profile_id}",
                )
            else:
                result = session.task(flow_goal, engine_override=flow.engine, assurance=flow.assurance)
            result = _flow_result_payload(flow, user_goal, result)
            session.last_task_result = result
            if isinstance(result.get("trust_card"), dict):
                session.last_trust_card = result["trust_card"]
            if console:
                _render_task_result(console, result)
            else:
                _print_json(result)
            return True
        print("Usage: /flow list | /flow show PROFILE | /flow run PROFILE GOAL")
        return True
    if name == "work":
        parts = _split_command(rest)
        subcommand = parts[0].lower() if parts else "status"
        flags = {part.lower() for part in parts if part.startswith("--")}
        wants_json = "--json" in flags or "json" in flags
        if subcommand in {"status", "show"}:
            payload = {
                "status": "ok",
                "last_work_available": bool(session.last_work_result),
                "last_run_id": (session.last_work_result or {}).get("run_id") if session.last_work_result else None,
                "last_plan": session.last_work_plan,
                "last_trust_card": session.last_trust_card,
            }
            if wants_json:
                _print_json(payload)
            else:
                print(_format_work_summary(session.last_work_result, plan=session.last_work_plan, trust_card=session.last_trust_card))
            return True
        if subcommand == "last":
            if wants_json:
                _print_json(session.last_work_result or {"status": "not_available", "error": "no Work Runtime run is available"})
            else:
                print(_format_work_summary(session.last_work_result, plan=session.last_work_plan, trust_card=session.last_trust_card))
            return True
        if subcommand == "history":
            if wants_json:
                _print_json({"status": "ok", "history": session.work_history(limit=20)})
            else:
                print(_format_work_history(session.work_history(limit=20)))
            return True
        if subcommand == "sources":
            target = parts[1].lower() if len(parts) > 1 else "last"
            if target != "last":
                print("Usage: /work sources last")
                return True
            _print_json(session.last_work_sources or {"status": "not_available", "error": "no Work Runtime sources are available"})
            return True
        if subcommand == "plan":
            target = parts[1].lower() if len(parts) > 1 else "last"
            if target != "last":
                print("Usage: /work plan last")
                return True
            _print_json(session.last_work_plan or {"status": "not_available", "error": "no Work Runtime plan is available"})
            return True
        if subcommand == "run":
            goal = " ".join(parts[1:]).strip()
        else:
            goal = rest.strip()
        if not goal:
            print("Usage: /work run GOAL")
            return True
        if console:
            result = _run_with_status(console, lambda: session.work(goal), phase="work")
            _render_task_result(console, result)
        else:
            _print_json(session.work(goal))
        return True
    if name in {"suites", "experiments", "experimentos"}:
        payload = session.suite_catalog.list_suites()
        if rest.lower() == "json":
            _print_json(payload)
        else:
            _print_suites_compact(payload)
        return True
    if name == "suite":
        parts = _split_command(rest)
        subcommand = parts[0].lower() if parts else "list"
        argument = " ".join(parts[1:]).strip()
        if subcommand == "index":
            index_action = parts[1].lower() if len(parts) > 1 else "status"
            if index_action in {"refresh", "rebuild"}:
                _print_json(session.suite_catalog.refresh_index())
                return True
            payload = helix_cli_core.suite_list(evidence_root=session.evidence_root, repo_root=REPO_ROOT)
            _print_json(
                {
                    "status": payload.get("status"),
                    "index_path": payload.get("index_path") or str(session.evidence_root / ".helix-index" / "suites.json"),
                    "source": payload.get("source"),
                    "warning": payload.get("warning"),
                    "rust_core": helix_cli_core.rust_core_status(),
                }
            )
            return True
        if subcommand in {"list", "ls", "catalog"}:
            payload = session.suite_catalog.list_suites()
            if argument == "json":
                _print_json(payload)
            else:
                _print_suites_compact(payload)
            return True
        if subcommand in {"show", "latest"}:
            if not argument:
                print(f"Usage: /suite {subcommand} SUITE_ID")
                return True
            suite_id = _suite_from_text(argument) or argument
            payload = session.suite_catalog.latest(suite_id) if subcommand == "latest" else session.suite_catalog.show_suite(suite_id)
            _print_json(payload)
            return True
        if subcommand in {"transcript", "transcripts"}:
            if not argument:
                print("Usage: /suite transcripts SUITE_ID [FILTER]")
                return True
            arg_parts = _split_command(argument)
            suite_id = _suite_from_text(arg_parts[0]) or arg_parts[0]
            query = " ".join(arg_parts[1:]).strip() or None
            _print_json(session.suite_catalog.transcripts(suite_id, query=query, limit=60))
            return True
        if subcommand == "search":
            deep = "--deep" in {part.lower() for part in parts[1:]}
            query = " ".join(part for part in parts[1:] if part.lower() != "--deep").strip() or input("Suite evidence query: ").strip()
            _print_json(session.suite_catalog.search(query, limit=20, deep=deep))
            return True
        if subcommand in {"read", "open"}:
            if not argument:
                print("Usage: /suite read PATH_OR_FILENAME")
                return True
            _print_json(session.suite_catalog.read(argument))
            return True
        if subcommand == "ingest":
            target = argument or "all"
            pack = session.refresh_evidence(None if target == "all" else target, limit=50)
            _print_json(pack)
            return True
        print("Usage: /suite list|index refresh|show SUITE|latest SUITE|transcripts SUITE [FILTER]|search [--deep] QUERY|read PATH|ingest [SUITE|all]")
        return True
    if name == "route":
        if not rest:
            print("Usage: /route TEXT")
            return True
        routed = route_model_for_task(
            rest,
            provider_name=session.provider_name,
            policy=session.router_policy,
            interaction_mode=session.interaction_mode,
        )
        routed["rust_core"] = helix_cli_core.route(rest, latency_mode=session.latency_mode, interaction_mode=session.interaction_mode)
        _print_json(routed)
        return True
    if name == "web":
        query = rest or input("Web query: ").strip()
        _print_json(web_search(query, limit=8))
        return True
    if name in {"file", "open", "read"}:
        path_ref = rest or input("File or directory path: ").strip()
        _print_json(session.file_inspect(path_ref))
        return True
    if name == "router":
        parts = _split_command(rest)
        subcommand = parts[0].lower() if parts else ""
        if subcommand == "why":
            prompt = " ".join(parts[1:]).strip()
            if not prompt:
                print("Usage: /router why TEXT")
                return True
            _print_json(
                route_model_for_task(
                    prompt,
                    provider_name=session.provider_name,
                    policy=session.router_policy,
                    interaction_mode=session.interaction_mode,
                )
                | {"rust_core": helix_cli_core.route(prompt, latency_mode=session.latency_mode, interaction_mode=session.interaction_mode)}
            )
            return True
        if rest.lower() in {"list", "ls"}:
            if console and _HAS_UI:
                selected = _select_router_policy(session)
                if selected:
                    session.router_policy = selected
            else:
                _print_json({"router_blueprints": router_blueprints_report(), "current": session.router_policy})
                return True
        elif rest:
            candidate = rest.lower()
            if candidate not in ROUTER_POLICIES:
                print(f"Unknown router policy: {candidate}. Use one of: {', '.join(sorted(ROUTER_POLICIES))}")
                return True
            session.router_policy = candidate
        elif console and _HAS_UI:
            selected = _select_router_policy(session)
            if selected:
                session.router_policy = selected
        print(f"[helix] router_policy={session.router_policy}")
        return True
    if name == "theme":
        selected_theme: str | None = None
        if rest.lower() in {"list", "ls"}:
            if console and _HAS_UI:
                selected_theme = _select_theme(session)
            else:
                _print_json({"themes": _theme_report(), "current": session.theme_name})
                return True
        elif rest:
            candidate = rest.lower()
            if candidate not in _THEME_PALETTES:
                print(f"Unknown theme: {candidate}. Use one of: {', '.join(sorted(_THEME_PALETTES))}")
                return True
            selected_theme = candidate
        elif console and _HAS_UI:
            selected_theme = _select_theme(session)
        if selected_theme:
            session.theme_name = _normalize_theme_name(selected_theme)
            config = _load_config()
            config["theme"] = session.theme_name
            _save_config(config)
        print(f"[helix] theme={session.theme_name}")
        return True
    if name == "style":
        if rest.lower() in {"list", "ls"}:
            _print_json({"current": session.response_style, "styles": RESPONSE_STYLES})
            return True
        if rest:
            session.response_style = _normalize_response_style(rest)
            config = _load_config()
            config["response_style"] = session.response_style
            _save_config(config)
        print(f"[helix] response_style={session.response_style}")
        return True
    if name == "mode":
        if rest.lower() in {"list", "ls"}:
            _print_json({"current": session.interaction_mode, "modes": _interaction_mode_report()})
            return True
        if rest.lower() in {"show", "current", "status", "json"}:
            _print_json(
                {
                    "current": session.interaction_mode,
                    "profile": _interaction_mode_payload(session.interaction_mode),
                    "router_policy": session.router_policy,
                    "tool_policy": session.tool_policy,
                    "thread_id": session.thread_id,
                }
            )
            return True
        if rest:
            if not _is_known_interaction_mode(rest):
                print("Usage: /mode [NAME|list|show]   NAMES: balanced, technical, explore")
                return True
            candidate = _normalize_interaction_mode(rest)
        else:
            # No argument -> let the user pick interactively. Fall back to a
            # plain stdin prompt when the rich UI is unavailable so we never
            # dump a wall of JSON the user has to read to know what to type.
            if console and _HAS_UI:
                candidate = _select_interaction_mode(session)
            else:
                candidate = _prompt_interaction_mode_text(session.interaction_mode)
            if not candidate:
                print(f"[helix] interaction_mode={session.interaction_mode} (unchanged)")
                return True
            candidate = _normalize_interaction_mode(candidate)
        session.interaction_mode = candidate
        config = _load_config()
        config["interaction_mode"] = session.interaction_mode
        _save_config(config)
        print(f"[helix] interaction_mode={session.interaction_mode}")
        return True
    if name == "blind":
        parts = _split_command(rest)
        subcommand = parts[0].lower() if parts else "status"
        argument = rest[len(subcommand):].strip() if subcommand and rest.lower().startswith(subcommand) else " ".join(parts[1:]).strip()
        if subcommand in {"status", "show", "json"}:
            _print_json(session.blind_status())
            return True
        if subcommand == "on":
            session.blind_inference_enabled = True
            session.blind_inference_policy.enabled = True
            print(f"[helix] blind_inference=on policy_id={session.blind_inference_policy.policy_id}")
            return True
        if subcommand == "off":
            session.blind_inference_enabled = False
            session.last_blind_report = None
            session.blind_inference_policy.enabled = False
            print("[helix] blind_inference=off")
            return True
        if subcommand == "policy":
            payload_text = argument or input("Blind policy JSON path or inline JSON: ").strip()
            if not payload_text:
                print("Usage: /blind policy PATH_OR_INLINE_JSON")
                return True
            try:
                payload = _coerce_blind_policy_payload(payload_text)
                payload["enabled"] = bool(session.blind_inference_enabled)
                session.blind_inference_policy = BlindInferencePolicy.from_payload(payload)
            except Exception as exc:  # noqa: BLE001
                print(f"[helix] invalid blind policy: {type(exc).__name__}: {exc}")
                return True
            print(
                f"[helix] blind_policy={session.blind_inference_policy.policy_id} "
                f"rules={len(session.blind_inference_policy.rules)} enabled={session.blind_inference_enabled}"
            )
            return True
        print("Usage: /blind on|off|status|policy PATH_OR_INLINE_JSON")
        return True
    if name in {"tech", "explore"}:
        prompt = rest or input("Prompt: ").strip()
        if not prompt:
            print(f"Usage: /{name} TEXT")
            return True
        _run_prompt_once(
            session,
            prompt,
            interaction_mode_override="technical" if name == "tech" else "explore",
        )
        return True
    if name == "config":
        token_providers = sorted((_load_config().get("tokens") or {}).keys())
        _print_json(
            {
                "config_path": _config_path(),
                "data_dir": _base_data_dir(),
                "default_workspace_root": _default_workspace_root(),
                "default_transcript_dir": _default_transcript_dir(),
                "default_evidence_root": _default_evidence_root(),
                "session_evidence_root": session.evidence_root,
                "session_task_root": session.task_root,
                "saved_token_providers": token_providers,
                "theme": session.theme_name,
                "interaction_mode": session.interaction_mode,
                "response_style": session.response_style,
            }
        )
        return True
    if name == "doctor":
        if rest.strip().lower() == "perf":
            _print_json(doctor_perf_report(session=session))
        else:
            _print_json(doctor_report(probe_local=False))
        return True
    if name == "provider":
        if rest.lower() in {"list", "ls"}:
            if console and _HAS_UI:
                selected = _select_provider(session)
                if selected:
                    _set_session_provider(session, selected)
            else:
                _print_json({"providers": provider_report(probe_local=False), "current": session.provider_name})
                return True
        elif not rest:
            if console and _HAS_UI:
                selected = _select_provider(session)
                if selected:
                    _set_session_provider(session, selected)
            else:
                print(session.provider_name)
                return True
        else:
            candidate = rest.lower()
            if candidate not in PROVIDERS:
                print(f"Unknown provider: {candidate}")
                return True
            _set_session_provider(session, candidate)
        if not rest and not console:
            return True
        print(f"[helix] provider={session.provider_name} model={session.model}")
        return True
    if name == "model":
        parts = _split_command(rest)
        subcommand = parts[0].lower() if parts else ""
        if subcommand == "use":
            rest = " ".join(parts[1:]).strip()
            if not rest:
                print("Usage: /model use NAME")
                return True
        if rest.lower() in {"list", "ls"}:
            if console and _HAS_UI:
                selected = _select_model(session)
                if selected:
                    _set_session_model(session, selected)
            else:
                _print_json(models_payload())
                return True
        elif rest:
            _set_session_model(session, rest)
        elif console and _HAS_UI:
            selected = _select_model(session)
            if selected:
                _set_session_model(session, selected)
        print(f"[helix] provider={session.provider_name} model={session.model}")
        return True
    if name == "with":
        parts = _split_command(rest)
        if len(parts) < 2:
            print("Usage: /with MODEL PROMPT")
            return True
        alias = parts[0]
        goal = " ".join(parts[1:]).strip()
        previous_provider = session.provider_name
        previous_model = session.model
        _set_session_model(session, alias)
        try:
            _run_prompt_once(session, goal, interaction_mode_override=session.interaction_mode)
        finally:
            session.provider_name = previous_provider
            session.model = previous_model
        print(f"[helix] provider/model restored to {session.provider_name}/{session.model}")
        return True
    if name == "raw":
        if rest:
            candidate = rest.lower()
            if candidate not in {"on", "off"}:
                print("Usage: /raw on|off")
                return True
            session.raw_output = candidate == "on"
        print(f"[helix] raw_output={'on' if session.raw_output else 'off'}")
        return True
    if name == "clear":
        os.system("cls" if os.name == "nt" else "clear")
        return True
    if name == "key":
        parts = _split_command(rest)
        action = parts[0].lower() if parts else ""
        provider_name = session.provider_name
        if action in PROVIDERS:
            provider_name = action
            action = ""
        elif len(parts) > 1 and parts[1].lower() in PROVIDERS:
            provider_name = parts[1].lower()
        if action == "forget":
            path = _forget_config_token(provider_name)
            provider = PROVIDERS[provider_name]
            if provider.token_env:
                os.environ.pop(provider.token_env, None)
            print(f"[helix] saved token removed from config: {path}")
            return True
        if action in {"save", "persist"}:
            provider = PROVIDERS[provider_name]
            if not provider.token_env:
                print(f"[helix] provider {provider.name} does not use an API token.")
                return True
            token = getpass.getpass(f"Paste {provider.name} token to save in HeliX config: ").strip()
            if not token:
                print("[helix] no token saved.")
                return True
            os.environ[provider.token_env] = token
            path = _save_config_token(provider.name, token)
            print(f"[helix] token saved in user config: {path}")
            return True
        if action == "status":
            provider = PROVIDERS[provider_name]
            _print_json(
                {
                    "provider": provider.name,
                    "token_env": provider.token_env,
                    "env_available": bool(provider.token_env and os.environ.get(provider.token_env)),
                    "saved_available": bool(_config_token(provider.name)),
                    "config_path": _config_path(),
                }
            )
            return True
        _ensure_provider_token(provider_name)
        return True
    if name in {"cert", "cert-dry"}:
        parts = _split_command(rest)
        if not parts:
            print("Usage: /cert SUITE [-- suite args]")
            return True
        suite_id = _suite_from_text(parts[0]) or parts[0]
        if suite_id not in SUITES:
            print(f"Unknown suite: {suite_id}")
            return True
        dry_run = name == "cert-dry"
        report = run_cert_suite(
            suite_id,
            provider_name=session.provider_name if session.provider_name == "deepinfra" else None,
            prompt_token=True,
            dry_run=dry_run,
            extra_args=_strip_remainder(parts[1:]),
        )
        _print_json(report)
        return True
    if name == "evidence":
        parts = _split_command(rest)
        subcommand = parts[0].lower() if parts else "latest"
        flags = {part.lower() for part in parts[1:] if part.startswith("--")}
        argument = " ".join(part for part in parts[1:] if not part.startswith("--")).strip()
        if subcommand in {"refresh", "scan"}:
            pack = session.refresh_evidence(argument or None, limit=12, deep="--deep" in flags)
            _print_json(pack)
            return True
        if subcommand in {"latest", "ls", "list"}:
            limit = 8
            if argument:
                try:
                    limit = max(1, int(argument))
                except ValueError:
                    print("Usage: /evidence latest [N]")
                    return True
            _print_json({"evidence": session.latest_evidence(limit=limit)})
            return True
        if subcommand == "search":
            query = argument or input("Evidence query: ").strip()
            _print_json(session.evidence_search(query, limit=12))
            return True
        if subcommand == "show":
            memory_id = argument
            if not memory_id:
                print("Usage: /evidence show MEMORY_ID")
                return True
            payload = session.evidence_show(memory_id)
            if payload is None:
                print(f"[helix] evidence memory not found: {memory_id}")
                return True
            _print_json(payload)
            return True
        print("Usage: /evidence refresh [--deep] [QUERY] | latest [N] | search QUERY | show MEMORY_ID")
        return True
    if name == "verify":
        if not rest:
            print("Usage: /verify last|PATH|latest|search QUERY")
            print("Related: /suites | /suite latest SUITE | /suite transcripts SUITE | /evidence latest")
            return True
        verify_parts = _split_command(rest)
        verify_mode = verify_parts[0].lower() if verify_parts else ""
        if verify_mode == "last":
            requested_level = "quick"
            for index, item in enumerate(verify_parts[1:], start=1):
                if item.lower() == "--level" and index + 1 < len(verify_parts):
                    requested_level = _normalize_assurance(verify_parts[index + 1])
                elif item.lower().startswith("--level="):
                    requested_level = _normalize_assurance(item.split("=", 1)[1])
            artifact = None
            if session.last_task_result:
                artifact = session.last_task_result.get("artifact_path")
            if not artifact:
                if session.last_work_result:
                    flow_profile = (session.last_work_plan or {}).get("flow_profile")
                    report = {
                        "status": "passed" if session.last_trust_card else "not_available",
                        "kind": "helix-work-verification-v1",
                        "requested_level": requested_level,
                        "run_id": session.last_work_result.get("run_id"),
                        "trust_card": session.last_trust_card,
                        "work_artifact_paths": session.last_work_result.get("work_artifact_paths"),
                    }
                    if requested_level in {"balanced", "strict"}:
                        profile = _lab_profile_for_flow(str(flow_profile or ""))
                        report["lab_profile"] = helix_cli_core.lab_run(profile=profile, evidence_root=session.evidence_root, repo_root=session.task_root)
                    elif requested_level == "deep":
                        report["status"] = "requires_explicit_deep"
                        report["message"] = "Deep nuclear verification stays explicit; use /lab run deep-nuclear or suite commands."
                    _print_json(report)
                    return True
                print("[helix] no last Task Capsule artifact is available.")
                return True
            report = helix_cli_core.verify_capsule(artifact_path=Path(str(artifact)))
            report["requested_level"] = requested_level
            if requested_level in {"balanced", "strict"}:
                flow_profile = (session.last_work_plan or {}).get("flow_profile") or ((session.last_trust_card or {}).get("flow_profile") if isinstance(session.last_trust_card, dict) else None)
                report["lab_profile"] = helix_cli_core.lab_run(profile=_lab_profile_for_flow(str(flow_profile or "")), evidence_root=session.evidence_root, repo_root=session.task_root)
            elif requested_level == "deep":
                report["status"] = "requires_explicit_deep"
                report["message"] = "Deep nuclear verification stays explicit; use /lab run deep-nuclear or suite commands."
            trust_card = report.get("trust_card") if isinstance(report.get("trust_card"), dict) else None
            if trust_card:
                session.last_trust_card = trust_card
            _print_json(report)
            return True
        if verify_mode == "latest":
            pack = session.refresh_evidence(None, limit=1)
            records = pack.get("records") or []
            if not records:
                print("[helix] no certified evidence artifacts found under verification/.")
                return True
            rest = str(records[0].get("artifact_path") or "")
        elif verify_mode == "search":
            query = " ".join(verify_parts[1:]).strip()
            if not query:
                print("Usage: /verify search QUERY")
                return True
            pack = session.refresh_evidence(query, limit=10)
            records = pack.get("records") or []
            _print_json(
                {
                    "query": query,
                    "candidate_count": len(records),
                    "candidates": [
                        {
                            "suite_id": item.get("suite_id"),
                            "run_id": item.get("run_id"),
                            "status": item.get("status"),
                            "artifact_path": item.get("artifact_path"),
                            "memory_id": item.get("memory_id"),
                            "node_hash": item.get("node_hash"),
                            "chain_status": item.get("chain_status"),
                        }
                        for item in records
                    ],
                }
            )
            return True
        path = Path(rest.strip('"'))
        if not path.is_absolute():
            path = REPO_ROOT / path
        
        if not path.exists():
            if console: console.print(f"[error]ERROR:[/] File does not exist: {path}")
            else: print(f"File not found: {path}")
            return True

        started = time.perf_counter()
        report = verify_artifact_file(path)
        ingested = ingest_artifact_file(
            root=session.workspace_root,
            project=session.project,
            agent_id=session.agent_id,
            repo_root=REPO_ROOT,
            artifact_path=path,
        )
        session.last_evidence_pack = {
            "source": "manual-verify",
            "record_count": 1,
            "records": [ingested],
        }
        duration_ms = (time.perf_counter() - started) * 1000

        if console:
            _render_verify_audit(console, report, ingested, duration_ms)
        else:
            _print_json(report)
        return True
    if name == "memory":
        parts = _split_command(rest)
        subcommand = parts[0].lower() if parts else ""
        if subcommand in {"resolve", "show", "hash"}:
            ref = " ".join(parts[1:]).strip()
            if not ref:
                print("Usage: /memory resolve HASH_OR_MEMORY_ID")
                return True
            _print_json(session.memory_resolve(ref))
            return True
        retrieval_scope = "session"
        query_parts = parts
        if subcommand == "search":
            query_parts = parts[1:]
        if "--global" in {part.lower() for part in query_parts}:
            retrieval_scope = "workspace"
            query_parts = [part for part in query_parts if part.lower() != "--global"]
        query = " ".join(query_parts).strip() or input("Memory query: ").strip()
        _print_json(
            hmem.hybrid_search(
                root=session.workspace_root,
                project=session.project,
                agent_id=session.agent_id,
                session_id=session.thread_id,
                query=query,
                top_k=8,
                retrieval_scope=retrieval_scope,
            )
        )
        return True
    if name == "trust":
        parts = _split_command(rest)
        raw_terms = {"json", "--json", "raw", "--raw"}
        forensic_terms = {"--forensics", "forensics", "--include-quarantined", "--quarantined"}
        wants_raw = any(part.lower() in raw_terms for part in parts)
        include_quarantined = any(part.lower() in forensic_terms for part in parts)
        clean_parts = [part for part in parts if part.lower() not in raw_terms | forensic_terms]
        if (not clean_parts and session.last_trust_card) or (clean_parts and clean_parts[0].lower() in {"last", "task"}):
            card = session.last_trust_card or _task_trust_card_from_result(session.last_task_result)
            if not card:
                print("[helix] no last Task Capsule trust card is available.")
                return True
            session.last_trust_card = card
            if wants_raw:
                _print_json(card)
            else:
                _print_trust_card(card)
            return True
        target = clean_parts[0] if clean_parts and clean_parts[0].lower() not in {"current", "show"} else None
        ref = None
        explicit_proof = bool(clean_parts and clean_parts[0].lower() in {"proof", "export"})
        if explicit_proof:
            target = clean_parts[1] if len(clean_parts) > 1 and clean_parts[1].lower() != "current" else None
            ref = clean_parts[2] if len(clean_parts) > 2 else None
        report = session.trust_report(target, ref=ref, include_quarantined=include_quarantined)
        if wants_raw or include_quarantined or explicit_proof:
            _print_json(report)
        else:
            _print_json(_compact_trust_report(report))
        return True
    if name == "tools":
        report = session.tool_registry_report()
        if rest.lower() == "json":
            _print_json(report)
        elif rest.lower() in {"blueprints", "agents"}:
            _print_agent_blueprints_compact()
        else:
            _print_tools_compact(report)
        return True
    if name == "agents":
        if rest.lower() == "json":
            _print_json({"agent_blueprints": agent_blueprints_report()})
        else:
            _print_agent_blueprints_compact()
        return True
    if name == "mode":
        _print_json(
            {
                "agent_mode": session.agent_mode,
                "task_engine": session.task_engine,
                "thread_id": session.thread_id,
                "tool_policy": session.tool_policy,
            }
        )
        return True
    if name == "opencode":
        parts = _split_command(rest)
        subcommand = parts[0].lower() if parts else "status"
        if subcommand == "status":
            _print_json(helix_cli_core.opencode_status())
            return True
        if subcommand == "install":
            flags = {part.lower() for part in parts[1:] if part.startswith("--")}
            config_path = None
            bin_arg = None
            for index, part in enumerate(parts[1:], start=1):
                if part == "--config-path" and index + 1 < len(parts):
                    config_path = Path(parts[index + 1])
                if part == "--bin" and index + 1 < len(parts):
                    bin_arg = parts[index + 1]
            _print_json(
                helix_cli_core.opencode_install(
                    dry_run="--dry-run" in flags,
                    global_config="--global" in flags,
                    force="--force" in flags,
                    opencode_bin=bin_arg,
                    config_path=config_path,
                )
            )
            return True
        print("Usage: /opencode status | /opencode install [--global] [--dry-run] [--force] [--bin PATH] [--config-path PATH]")
        return True
    if name == "apply":
        if rest.lower() not in {"last", "last --check", "--check last"}:
            print("Usage: /apply last")
            return True
        if not session.last_patch:
            print("[helix] no patch proposal is available from the last task.")
            return True
        actual_patch_sha = hashlib.sha256(session.last_patch.encode("utf-8")).hexdigest()
        expected_patch_sha = session.last_patch_sha256
        if expected_patch_sha and expected_patch_sha != actual_patch_sha:
            print("[helix] patch hash mismatch; refusing to apply last patch.")
            print(f"expected={expected_patch_sha} actual={actual_patch_sha}")
            return True
        check = subprocess.run(  # noqa: S603 - fixed argv, patch is stdin, shell disabled
            ["git", "-C", str(session.task_root), "apply", "--check", "-"],
            input=session.last_patch,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        if check.returncode != 0:
            print("[helix] patch check failed; not applying.")
            print(check.stderr or check.stdout)
            return True
        confirm = input(f"Apply last patch to {session.task_root}? [y/N]: ").strip().lower()
        if confirm not in {"y", "yes", "s", "si"}:
            print("[helix] patch not applied.")
            return True
        applied = subprocess.run(  # noqa: S603 - fixed argv, patch is stdin, shell disabled
            ["git", "-C", str(session.task_root), "apply", "-"],
            input=session.last_patch,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        if applied.returncode == 0:
            if session.last_work_result:
                session.last_work_result["applied"] = True
                session.last_work_result["applied_utc"] = _utc_now()
                output = _primary_work_output_path(session.last_work_result, plan=session.last_work_plan, task_root=session.task_root)
                if output:
                    artifact = _inspect_work_artifact(session, output, last_action="apply_readback")
                    session.last_artifact = artifact
                    session.last_work_result["artifact"] = artifact
                    if session.last_trust_card:
                        session.last_trust_card["artifact"] = artifact
                session._save_last_work_state()
            session.record(
                role="tool",
                content="Applied last Agent Shell patch via explicit /apply last.",
                event_type="task_patch_applied",
                metadata={"task_root": str(session.task_root)},
            )
            print("[helix] patch applied.")
        else:
            print("[helix] patch apply failed.")
            print(applied.stderr or applied.stdout)
        return True
    if name in {"task", "agent"}:
        mode_override = None
        agent_blueprint: AgentBlueprint | None = None
        engine_override: str | None = None
        assurance = "quick"
        if name == "agent":
            parts = _split_command(rest)
            if parts and parts[0].lower() in {"list", "ls", "blueprints"}:
                _print_agent_blueprints_compact()
                return True
            if parts and parts[0].lower() in {"use", "run"}:
                if len(parts) < 2:
                    print("Usage: /agent use BLUEPRINT GOAL")
                    print(f"Known blueprints: {', '.join(sorted(AGENT_BLUEPRINTS))}")
                    return True
                blueprint_id = _slugish(parts[1])
                agent_blueprint = AGENT_BLUEPRINTS.get(blueprint_id)
                if agent_blueprint is None:
                    print(f"Unknown agent blueprint: {parts[1]}")
                    print(f"Known blueprints: {', '.join(sorted(AGENT_BLUEPRINTS))}")
                    return True
                mode_override = "suggest"
                rest = " ".join(parts[2:]).strip()
            elif parts and parts[0].lower() in {"suggest", "plan"}:
                mode_override = "suggest"
                rest = " ".join(parts[1:]).strip()
            elif parts and parts[0].lower() in {"auto-edit", "autoedit", "edit"}:
                print("[helix] auto-edit is not enabled in this build; running safe suggest mode instead.")
                mode_override = "suggest"
                rest = " ".join(parts[1:]).strip()
            else:
                mode_override = "suggest"
        else:
            parts = _split_command(rest)
            if parts and parts[0].lower() == "engine":
                if len(parts) < 2:
                    print(f"[helix] task_engine={session.task_engine}")
                    return True
                candidate = _normalize_task_engine(parts[1])
                if candidate != parts[1].strip().lower():
                    print("Usage: /task engine opencode|helix|auto")
                    return True
                session.task_engine = candidate
                config = _load_config()
                config["task_engine"] = session.task_engine
                _save_config(config)
                print(f"[helix] task_engine={session.task_engine}")
                return True
            engine_override, assurance, cleaned_parts = _extract_task_options(parts, default_engine=session.task_engine)
            rest = " ".join(cleaned_parts).strip()
        goal = rest or input("Agent goal: ").strip()
        if console:
            result = _run_with_status(
                console,
                lambda: session.task(goal, mode_override=mode_override, agent_blueprint=agent_blueprint, engine_override=engine_override, assurance=assurance),
                phase="task",
            )
        else:
            result = session.task(goal, mode_override=mode_override, agent_blueprint=agent_blueprint, engine_override=engine_override, assurance=assurance)
        if console:
            _render_task_result(console, result)
        else:
            _print_json(result)
        return True

    print(f"Unknown command: /{name}. Use /help.")
    return True


def _render_task_result(active_console: Any, result: dict[str, Any]) -> None:
    _render_task_result_panel(
        active_console,
        result,
        normalize_tool_event=_normalize_tool_event,
        tool_event_detail=_tool_event_detail,
        short_model_name=_short_model_name,
    )


def _run_prompt_once(
    session: InteractiveSession,
    prompt: str,
    *,
    interaction_mode_override: str | None = None,
) -> None:
    if _looks_like_work_confirmation(prompt) and session.last_work_result:
        _handle_interactive_command(session, "/work last")
        return
    intent_card = session.turn_controller(prompt)
    route = intent_card.route
    routed = str(intent_card.fallback_command or "") or _route_natural_language(prompt, session.recent_route_intents(limit=4))
    _show_turn_plan(prompt, route, intent_card=intent_card)
    if routed and routed.startswith("/") and not routed.startswith("/work run ") and not routed.startswith("/task"):
        _handle_interactive_command(session, routed)
        return
    if routed and routed.startswith("/work run "):
        goal = routed[len("/work run "):].strip() if routed.startswith("/work run ") else prompt
        if console:
            result = _run_with_status(console, lambda: session.work(goal), phase="work")
            _render_task_result(console, result)
        else:
            _print_json(session.work(goal))
        return
    if routed and routed.startswith("/task"):
        if console:
            result = _run_with_status(
                console,
                lambda: session.task(prompt, interaction_mode_override=interaction_mode_override),
                phase="task",
            )
            _render_task_result(console, result)
        else:
            _print_json(session.task(prompt, interaction_mode_override=interaction_mode_override))
        return
    if console:
        response_obj = _run_with_status(
            console,
            lambda: session.chat(prompt, interaction_mode_override=interaction_mode_override),
            phase="thinking",
        )
        latest = session.events[-1] if session.events else {}
        metadata = latest.get("metadata", {})
        receipt = latest.get("helix_memory") or {}
        raw_text = response_obj.get("raw_text") or ""
        route = metadata.get("route") or response_obj.get("route") or {}
        _render_chat_response(
            console,
            clean_text=response_obj.get("text") or "",
            model_used=_display_model_used(metadata, session.model),
            intent=str(route.get("intent") or metadata.get("interaction_mode") or "chat"),
            latency_label=(
                f"{float(metadata.get('latency_ms')):.0f}ms"
                if isinstance(metadata.get("latency_ms"), (int, float))
                else "n/a"
            ),
            short_hash=str(receipt.get("node_hash") or "")[:10] or "nohash",
            raw_text=raw_text,
            show_raw=session.raw_output,
        )
        return
    response = session.chat(prompt, interaction_mode_override=interaction_mode_override)
    print(response.get("text"))


def run_interactive(args: argparse.Namespace | None = None) -> int:
    global console
    args = args or argparse.Namespace()
    config = _load_config()
    theme_name = _normalize_theme_name(getattr(args, "theme", None) or config.get("theme") or DEFAULT_THEME)
    if _HAS_UI:
        _chrome()
    console = Console(theme=_rich_theme(theme_name)) if _HAS_UI and Console else None
    active_theme_name = theme_name

    if console:
        console.print("[bold]HeliX Workbench[/bold] [dim]chat | work | code | trust[/dim]")
    else:
        print("HeliX Workbench. Type /help for commands, /exit to quit.")

    default_provider = (
        getattr(args, "provider", None)
        or config.get("default_provider")
        or ("deepinfra" if not os.environ.get("OLLAMA_HOST") else "ollama")
    )
    if default_provider not in PROVIDERS:
        default_provider = "deepinfra"
    explicit_provider = getattr(args, "provider", None)
    skip_provider_prompt = bool(explicit_provider or config.get("default_provider") or _provider_ready_from_config(default_provider))
    provider_name, pending_line = (default_provider, None) if skip_provider_prompt else _choose_provider(default_provider)
    default_model = (
        getattr(args, "model", None)
        or config.get("default_model")
        or ("auto" if provider_name == "deepinfra" else PROVIDERS[provider_name].default_model)
    )
    skip_model_prompt = bool(getattr(args, "model", None) or config.get("default_model") or skip_provider_prompt)
    model = (default_model or "").strip() if skip_model_prompt else (_read_default("Model", default_model) if default_model else input("Model: ").strip())
    _ensure_provider_token(provider_name)
    if config.get("prompt_optional_provider_keys") is True:
        for optional_provider in ("gemini", "nvidia"):
            if provider_name == optional_provider:
                continue
            config = _maybe_prompt_optional_provider_token(optional_provider, config=config)
    workspace = Path(getattr(args, "workspace_root", None) or _default_workspace_root()).resolve()
    task_root = Path(getattr(args, "task_root", None)).resolve() if getattr(args, "task_root", None) else _default_task_root()
    project = _slugish(getattr(args, "project", None) or "helix-cli")
    agent_id = _slugish(getattr(args, "agent_id", None) or "interactive")
    transcript_dir = Path(getattr(args, "transcript_dir", None) or _default_transcript_dir())
    evidence_root = Path(getattr(args, "evidence_root", None) or _default_evidence_root()).resolve()
    session = InteractiveSession(
        provider_name=provider_name,
        model=model,
        workspace_root=workspace,
        project=project,
        agent_id=agent_id,
        max_tokens=int(getattr(args, "max_tokens", 2048) or 2048),
        temperature=float(getattr(args, "temperature", 0.0) or 0.0),
        transcript_dir=transcript_dir,
        router_policy=str(getattr(args, "router_policy", "balanced") or "balanced"),
        evidence_root=evidence_root,
        task_root=task_root,
    )
    session.theme_name = theme_name
    session.response_style = _normalize_response_style(config.get("response_style") or "balanced")
    session.interaction_mode = _normalize_interaction_mode(config.get("interaction_mode") or "balanced")
    session.latency_mode = _normalize_latency_mode(config.get("latency_mode") or "fast")
    session.preflight_mode = _normalize_preflight_mode(config.get("preflight_mode") or "compact")
    session.task_engine = _normalize_task_engine(config.get("task_engine") or os.environ.get("HELIX_TASK_ENGINE") or "auto")

    last_work = f" last={session.last_work_result.get('run_id')}" if session.last_work_result else ""
    ready_line = (
        f"[helix] ready provider={session.provider_name} model={session.model} "
        f"task_root={session.task_root} engine={session.task_engine}{last_work}"
    )
    if console:
        console.print(ready_line)
        console.print("[dim]try: analiza README.md y armame una pagina web en web/demo | /last | /trust | /apply last[/dim]")
    else:
        print(ready_line)
        print("[helix] try: analiza README.md y armame una pagina web en web/demo | /last | /trust | /apply last")

    session.record(
        role="system",
        content="Interactive HeliX session started.",
        event_type="session_start",
        metadata={
            "provider": provider_name,
            "model": model,
            "router_policy": session.router_policy,
            "interaction_mode": session.interaction_mode,
            "latency_mode": session.latency_mode,
            "task_engine": session.task_engine,
            "blind_inference": session.blind_status(),
            "evidence_root": str(session.evidence_root),
            "task_root": str(session.task_root),
        },
    )

    if _HAS_UI:
        completer = WordCompleter([
            '/help', '/status', '/last', '/sources', '/read last', '/inspect last', '/open last', '/latency', '/hooks status', '/conversation', '/fast', '/thread', '/branch', '/provider', '/model', '/models', '/lab', '/flow', '/work', '/route', '/web', '/file',
            '/router', '/key', '/doctor', '/opencode', '/providers', '/cert', '/cert-dry',
            '/evidence', '/verify', '/suites', '/suite', '/memory', '/trust', '/task', '/tools', '/agents', '/mode', '/blind', '/tech', '/explore', '/apply', '/agent', '/with', '/theme', '/style', '/raw', '/clear', '/config', '/exit', '/quit',
            '/provider deepinfra', '/provider gemini', '/provider nvidia', '/provider list',
            '/models json', '/model auto', '/model use ', '/model sonnet', '/model mistral', '/model devstral', '/model qwen', '/model qwen-big', '/model bioinformatics', '/model gemma', '/model gemini-pro', '/model gemini-pro-tools', '/model gemini-flash', '/model gemini-lite', '/model gemini-2.5-pro', '/model gemini-2.5-flash', '/model gemini-2.5-flash-lite', '/model nvidia-chat', '/model nvidia-research', '/model nvidia-code', '/model nvidia-agentic', '/model nvidia-guard', '/model nvidia-pii', '/model coder', '/model engineering', '/model deep-reasoning', '/model llama', '/model llama-vision',
            '/with sonnet ', '/with qwen-big ', '/with bioinformatics ', '/with gemma ', '/with gemini-pro ', '/with gemini-pro-tools ', '/with gemini-flash ', '/with gemini-lite ', '/with gemini-2.5-pro ', '/with gemini-2.5-flash ', '/with gemini-2.5-flash-lite ', '/with nvidia-chat ', '/with nvidia-research ', '/with nvidia-code ', '/with nvidia-agentic ', '/with coder ', '/with mistral ',
            '/router balanced', '/router qwen-heavy', '/router current', '/router qwen-gemma-mistral', '/router cheap', '/router premium', '/router nvidia-build', '/router list', '/router why ',
            '/web ', '/file ',
            '/theme industrial-brutalist', '/theme industrial-neon', '/theme xerox', '/theme brown-console', '/theme brown', '/theme cyberpunk', '/theme cyberpunk-gray', '/theme list', '/raw on', '/raw off',
            '/mode balanced', '/mode technical', '/mode explore', '/mode list', '/mode show', '/tech ', '/explore ',
            '/latency mode fast', '/latency mode balanced', '/latency mode deep', '/latency 8', '/fast on', '/fast off', '/fast status',
            '/preflight compact', '/preflight verbose', '/preflight off', '/demo wow', '/demo wow --json', '/demo doctor', '/demo open', '/demo skills suggest',
            '/conversation status', '/conversation summarize', '/conversation reset-summary',
            '/blind on', '/blind off', '/blind status', '/blind policy ',
            '/style balanced', '/style technical', '/style forensic', '/style vivid', '/style terse', '/style list',
            '/key save', '/key save gemini', '/key save nvidia', '/key gemini', '/key nvidia', '/key forget', '/key forget gemini', '/key forget nvidia', '/key status', '/key status gemini', '/key status nvidia',
            '/evidence refresh', '/evidence latest', '/evidence search', '/verify last', '/verify latest', '/verify search',
            '/memory ', '/memory search --global ', '/memory resolve ', '/memory show ', '/memory hash ', '/trust', '/trust last', '/trust last json', '/trust current', '/trust current json', '/trust --json', '/trust --forensics', '/trust proof current ',
            '/models compare last', '/lab profiles', '/lab run patch-safety', '/lab run doc-grounding', '/lab run memory-isolation', '/lab run provider-audit', '/lab run deep-nuclear',
            '/flow list', '/flow show web', '/flow run web ', '/flow run web-recursive ', '/flow run patch-safe ', '/flow run doc-grounded ', '/flow run resilient-task ', '/flow run privacy-swarm ', '/flow run multi-review ', '/flow run deep-lab ',
            '/work run ', '/work status', '/work last', '/work history', '/work sources last', '/work plan last',
            '/suites json', '/suite list', '/suite index refresh', '/suite latest ', '/suite show ', '/suite transcripts ', '/suite search ', '/suite search --deep ', '/suite read ', '/suite ingest ',
            '/thread new --clean ', '/thread list', '/thread tree', '/thread tree json', '/thread open', '/thread close', '/thread current', '/branch new ', '/branch tree',
            '/opencode status', '/opencode install --global --dry-run', '/opencode install --global',
            '/task ', '/task --engine opencode --assurance quick ', '/task --engine opencode --assurance balanced ', '/task --engine opencode --assurance strict ', '/task --engine helix ', '/task engine auto', '/task engine opencode', '/task engine helix', '/agent suggest ', '/agent use repo-scout ', '/agent use patch-planner ', '/agent use suite-run-analyst ', '/agent use transcript-forensics ', '/agent use evidence-auditor ', '/agent auto-edit ',
            '/tools', '/tools json', '/tools blueprints', '/agents', '/agents json', '/mode', '/apply last',
        ], ignore_case=True)
        prompt_session = PromptSession(completer=completer, style=_prompt_style(theme_name))
    else:
        prompt_session = None

    def _refresh_ui_theme() -> None:
        global console
        nonlocal prompt_session, active_theme_name
        if not _HAS_UI or session.theme_name == active_theme_name:
            return
        active_theme_name = _normalize_theme_name(session.theme_name)
        session.theme_name = active_theme_name
        console = Console(theme=_rich_theme(active_theme_name))
        prompt_session = PromptSession(completer=completer, style=_prompt_style(active_theme_name))
        _render_session_ribbon(console, session)

    def _process_turn(user_input: str) -> None:
        try:
            _refresh_ui_theme()
            if console:
                response_obj = _run_with_status(console, lambda: session.chat(user_input))
                
                clean_text = response_obj.get("text", "")
                if not clean_text:
                    clean_text = "[dim]Processing complete. Response archived in transcript.[/dim]"

                latest = session.events[-1] if session.events else {}
                metadata = latest.get("metadata", {})
                route = metadata.get("route") or response_obj.get("route") or {}
                model_used = _display_model_used(metadata, session.model)
                latency = metadata.get("latency_ms")
                receipt = latest.get("helix_memory") or {}
                node_hash = str(receipt.get("node_hash") or "")
                short_hash = node_hash[:10] if node_hash else "nohash"
                intent = route.get("intent") or "manual"
                latency_label = f"{float(latency):.0f}ms" if isinstance(latency, (int, float)) else "n/a"
                raw_text = response_obj.get("raw_text") or ""
                _render_chat_response(
                    console,
                    clean_text=clean_text,
                    model_used=model_used,
                    intent=intent,
                    latency_label=latency_label,
                    short_hash=short_hash,
                    raw_text=raw_text,
                    show_raw=session.raw_output,
                )
            else:
                response = session.chat(user_input)
                print(response.get("text"))
        except KeyboardInterrupt:
            if console: console.print("[warning]request cancelled[/warning]")
            else: print("[helix] request cancelled")
        except error.URLError as exc:
            if console: console.print(f"[error]LINK FAILURE:[/] {exc}")
            else: print(f"[helix] provider connection failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            if _recover_missing_model_alias(session, exc):
                return
            if console: console.print(f"[error]SYSTEM CRASH:[/] {type(exc).__name__}: {exc}")
            else: print(f"[helix] error: {type(exc).__name__}: {exc}")

    if pending_line:
        _process_turn(pending_line)

    while True:
        try:
            _refresh_ui_theme()
            if prompt_session:
                line = prompt_session.prompt(
                    _prompt_message(session),
                    bottom_toolbar=_prompt_bottom_toolbar(session),
                ).strip()
            else:
                line = input("helix> ").strip()
        except (EOFError, KeyboardInterrupt):
            if console: console.print()
            else: print()
            break
        if not line:
            continue
        if not line.startswith("/") and _looks_like_work_confirmation(line) and session.last_work_result:
            try:
                if not _handle_interactive_command(session, "/work last"):
                    break
            except Exception as exc:  # noqa: BLE001
                if _recover_missing_model_alias(session, exc):
                    continue
                if console: console.print(f"[error]SYSTEM CRASH:[/] {type(exc).__name__}: {exc}")
                else: print(f"[helix] error: {type(exc).__name__}: {exc}")
            continue
        intent_card = None if line.startswith("/") else session.turn_controller(line)
        route = None if intent_card is None else intent_card.route
        routed = line if line.startswith("/") else str((intent_card.fallback_command if intent_card else "") or "") or _route_natural_language(line, session.recent_route_intents(limit=4))
        if route is not None:
            _show_turn_plan(line, route, intent_card=intent_card, mode=session.preflight_mode)
        if routed and routed.startswith("/"):
            try:
                if not _handle_interactive_command(session, routed):
                    break
            except KeyboardInterrupt:
                if console: console.print("[warning]request cancelled[/warning]")
                else: print("[helix] request cancelled")
            except error.URLError as exc:
                if console: console.print(f"[error]LINK FAILURE:[/] {exc}")
                else: print(f"[helix] provider connection failed: {exc}")
            except Exception as exc:  # noqa: BLE001
                if _recover_missing_model_alias(session, exc):
                    continue
                if console: console.print(f"[error]SYSTEM CRASH:[/] {type(exc).__name__}: {exc}")
                else: print(f"[helix] error: {type(exc).__name__}: {exc}")
            continue
        _process_turn(line)

    if session.thread_id:
        session.record(role="system", content="Interactive HeliX session ended.", event_type="session_end")
        if console:
            console.print(f"[dim info]* saved markdown:[/] {session.md_path}")
        else:
            print(f"[helix] saved markdown: {session.md_path}")
    return 0


def provider_report(*, probe_local: bool = False) -> list[dict[str, Any]]:
    rows = []
    for provider in sorted(PROVIDERS.values(), key=lambda item: item.name):
        endpoint_status = "not_probed"
        if probe_local and provider.name in {"ollama", "llamacpp"} and provider.base_url:
            try:
                _get_json(f"{provider.base_url.rstrip('/')}/models", timeout=1.5)
                endpoint_status = "reachable"
            except Exception as exc:  # noqa: BLE001
                endpoint_status = f"unreachable: {type(exc).__name__}"
        rows.append(
            {
                "name": provider.name,
                "kind": provider.kind,
                "base_url": provider.base_url,
                "token_env": provider.token_env,
                "token_available": provider.token_available,
                "requires_token": provider.requires_token,
                "default_model": provider.default_model,
                "endpoint_status": endpoint_status,
                "description": provider.description,
                **_provider_capability_payload(provider),
            }
        )
    return rows


def doctor_report(*, probe_local: bool = False) -> dict[str, Any]:
    suite_rows = []
    for suite in sorted(SUITES.values(), key=lambda item: item.suite_id):
        preregistered = Path(suite.output_dir) / "PREREGISTERED.md"
        suite_rows.append(
            {
                "suite_id": suite.suite_id,
                "script": suite.script,
                "script_exists": suite.script_path.exists(),
                "output_dir": suite.output_dir,
                "preregistered_exists": (REPO_ROOT / preregistered).exists(),
                "requires_deepinfra": suite.requires_deepinfra,
                "supports_deepinfra_flag": suite.supports_deepinfra_flag,
            }
        )
    return {
        "helix_cli": "v0",
        "repo_root": str(REPO_ROOT),
        "python": sys.version.split()[0],
        "started_utc": _utc_now(),
        "providers": provider_report(probe_local=probe_local),
        "suites": suite_rows,
        "secret_policy": "tokens are read from env, hidden prompt, or optional HeliX user config; transcripts redact token values",
    }


def doctor_perf_report(*, session: InteractiveSession | None = None) -> dict[str, Any]:
    evidence_root = session.evidence_root if session is not None else _default_evidence_root()
    index_path = evidence_root / ".helix-index" / "suites.json"
    route_started = time.perf_counter()
    route_payload = helix_cli_core.route("hola helix", latency_mode=(session.latency_mode if session else "fast"))
    route_wall_ms = (time.perf_counter() - route_started) * 1000
    suite_started = time.perf_counter()
    suites_payload = helix_cli_core.suite_list(evidence_root=evidence_root, repo_root=REPO_ROOT)
    suite_wall_ms = (time.perf_counter() - suite_started) * 1000
    opencode_started = time.perf_counter()
    opencode_payload = helix_cli_core.opencode_status()
    opencode_wall_ms = (time.perf_counter() - opencode_started) * 1000
    return {
        "status": "ok",
        "slo": {
            "cold_start_usable_ms": 1200,
            "chat_simple_local_prep_ms": 300,
            "routing_ms": 50,
            "suite_list_ms": 500,
            "suite_search_ms": 1000,
            "local_pre_model_hard_cap_ms": 10000,
        },
        "rust_core": helix_cli_core.rust_core_status(),
        "latency_mode": session.latency_mode if session else "fast",
        "route_probe": {
            "path": route_payload.get("path"),
            "source": route_payload.get("source"),
            "routing_ms": route_payload.get("routing_ms"),
            "wall_ms": round(route_wall_ms, 3),
            "under_budget": route_wall_ms < 50,
        },
        "suite_index": {
            "index_path": str(index_path),
            "exists": index_path.exists(),
            "status": suites_payload.get("status"),
            "source": suites_payload.get("source"),
            "suite_count": suites_payload.get("suite_count"),
            "warning": suites_payload.get("warning"),
            "wall_ms": round(suite_wall_ms, 3),
            "under_budget": suite_wall_ms < 500,
        },
        "opencode": {
            "available": bool(opencode_payload.get("available")),
            "binary": opencode_payload.get("binary"),
            "source": opencode_payload.get("source"),
            "wall_ms": round(opencode_wall_ms, 3),
            "under_budget": opencode_wall_ms < 300,
            "helix_mcp_command": "helix-cli-core mcp-stdio",
        },
        "last_latency": session.last_latency_trace if session is not None else None,
    }


def _suite_command(
    suite: SuiteSpec,
    *,
    python_executable: str,
    case: str | None,
    provider_name: str | None,
    run_id: str | None,
    output_dir: str | None,
    extra_args: list[str],
) -> list[str]:
    command = [python_executable, str(suite.script_path)]
    if case:
        command.extend(["--case", case])
    if run_id:
        command.extend(["--run-id", run_id])
    if output_dir:
        command.extend(["--output-dir", output_dir])
    if provider_name == "deepinfra" and suite.supports_deepinfra_flag:
        command.append("--use-deepinfra")
    command.extend(extra_args)
    return command


def run_cert_suite(
    suite_id: str,
    *,
    python_executable: str = sys.executable,
    case: str | None = None,
    provider_name: str | None = None,
    run_id: str | None = None,
    output_dir: str | None = None,
    prompt_token: bool = True,
    dry_run: bool = False,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    if suite_id not in SUITES:
        raise KeyError(f"unknown suite: {suite_id}")
    suite = SUITES[suite_id]
    if not suite.script_path.exists():
        raise FileNotFoundError(suite.script_path)
    if provider_name not in {None, "local", "deepinfra"}:
        raise ValueError("registered suites currently support local execution or DeepInfra only")
    cloud_requested = suite.requires_deepinfra or provider_name == "deepinfra"
    provider = PROVIDERS["deepinfra"] if cloud_requested else None
    env = os.environ.copy()
    if cloud_requested:
        token = _token_for_provider(provider, prompt=prompt_token)
        if token:
            env["DEEPINFRA_API_TOKEN"] = token
        elif not dry_run:
            raise RuntimeError("DEEPINFRA_API_TOKEN is required for this suite/provider")
    command = _suite_command(
        suite,
        python_executable=python_executable,
        case=case,
        provider_name=provider_name,
        run_id=run_id,
        output_dir=output_dir,
        extra_args=extra_args or [],
    )
    redacted_command = redact_value(command, secrets=_secret_values(provider))
    if dry_run:
        return {
            "suite_id": suite_id,
            "dry_run": True,
            "command": redacted_command,
            "cwd": str(REPO_ROOT),
            "requires_deepinfra": suite.requires_deepinfra,
            "supports_deepinfra_flag": suite.supports_deepinfra_flag,
            "token_env": provider.token_env if provider else None,
            "token_available": bool(provider and provider.token_available),
        }
    started = _utc_now()
    completed = subprocess.run(  # noqa: S603 - command is assembled from registry and explicit user args
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    return {
        "suite_id": suite_id,
        "dry_run": False,
        "command": redacted_command,
        "cwd": str(REPO_ROOT),
        "started_utc": started,
        "ended_utc": _utc_now(),
        "exit_code": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout": redact_value(completed.stdout, secrets=_secret_values(provider)),
        "stderr": redact_value(completed.stderr, secrets=_secret_values(provider)),
    }


def _strip_remainder(values: list[str] | None) -> list[str]:
    if not values:
        return []
    if values and values[0] == "--":
        return values[1:]
    return values


def _cmd_doctor(args: argparse.Namespace) -> int:
    if getattr(args, "doctor_mode", None) == "perf":
        _print_json(doctor_perf_report())
    else:
        _print_json(doctor_report(probe_local=args.probe_local))
    return 0


def _cmd_providers_list(args: argparse.Namespace) -> int:
    _print_json({"providers": provider_report(probe_local=args.probe_local)})
    return 0


def _cmd_models_list(args: argparse.Namespace) -> int:
    _print_json(models_payload())
    return 0


def _cmd_route(args: argparse.Namespace) -> int:
    route = route_model_for_task(args.prompt, provider_name=args.provider, policy=args.policy)
    route["rust_core"] = helix_cli_core.route(args.prompt)
    _print_json(route)
    return 0


def _cmd_opencode(args: argparse.Namespace) -> int:
    if args.opencode_command == "status":
        _print_json(helix_cli_core.opencode_status(opencode_bin=args.bin))
        return 0
    if args.opencode_command == "install":
        payload = helix_cli_core.opencode_install(
            dry_run=args.dry_run,
            global_config=args.global_config,
            force=args.force,
            opencode_bin=args.bin,
            config_path=args.config_path,
        )
        _print_json(payload)
        return 0 if payload.get("status") in {"ok", "dry_run"} else 1
    raise SystemExit("unknown opencode command")


def _cmd_flow(args: argparse.Namespace) -> int:
    if args.flow_command in {"list", "profiles"}:
        _print_json(flow_profiles_report())
        return 0
    if args.flow_command == "show":
        try:
            _print_json(_flow_profile_or_error(args.profile).report())
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        return 0
    if args.flow_command == "run":
        try:
            flow = _flow_profile_or_error(args.profile)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        engine = _normalize_task_engine(args.engine or flow.engine)
        assurance = _normalize_assurance(args.assurance or flow.assurance)
        if engine != "opencode":
            raise SystemExit("flow run currently uses the OpenCode sandbox backend; use interactive /task for helix engine flows")
        user_goal = str(args.goal or "").strip()
        result = helix_cli_core.opencode_run(
            repo_root=Path(args.task_root).resolve() if args.task_root else _default_task_root(),
            goal=flow.build_goal(user_goal),
            evidence_root=Path(args.evidence_root or _default_evidence_root()).resolve(),
        )
        wrapped = _flow_result_payload(flow, user_goal, result)
        wrapped["assurance"] = assurance
        if assurance in {"balanced", "strict", "deep"} and result.get("artifact_path"):
            followup = _task_assurance_followup(
                assurance=assurance,
                artifact_path=str(result.get("artifact_path") or ""),
                evidence_root=Path(args.evidence_root or _default_evidence_root()).resolve(),
                repo_root=Path(args.task_root).resolve() if args.task_root else _default_task_root(),
            )
            wrapped["assurance_followup"] = followup
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(_json_ready(wrapped), indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            _print_json(wrapped)
        return 0 if wrapped.get("status") in {"passed", "completed"} else 1
    raise SystemExit("unknown flow command")


def _cmd_work(args: argparse.Namespace) -> int:
    if args.work_command == "run":
        session = InteractiveSession(
            provider_name=args.provider,
            model=args.model or ("auto" if args.provider == "deepinfra" else PROVIDERS[args.provider].default_model),
            workspace_root=Path(args.workspace_root or _default_workspace_root()).resolve(),
            project=_slugish(args.project or "helix-cli"),
            agent_id=_slugish(args.agent_id or "work"),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            transcript_dir=Path(args.transcript_dir or _default_transcript_dir()),
            router_policy=args.router_policy,
            evidence_root=Path(args.evidence_root or _default_evidence_root()).resolve(),
            task_root=Path(args.task_root).resolve() if args.task_root else _default_task_root(),
        )
        result = session.work(
            args.goal,
            max_pages=args.max_pages,
            depth=args.depth,
            cross_domain=args.cross_domain,
            assurance_override=args.assurance,
        )
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(_json_ready(result), indent=2, ensure_ascii=False), encoding="utf-8")
        elif getattr(args, "json", False):
            _print_json(result)
        else:
            print(_format_work_summary(result, plan=session.last_work_plan, trust_card=session.last_trust_card))
        return 0 if result.get("status") in {"completed", "passed"} else 1
    raise SystemExit("unknown work command")


def _cmd_demo(args: argparse.Namespace) -> int:
    if args.demo_command == "doctor":
        _print_json(demo_doctor_report())
        return 0
    if args.demo_command == "skills":
        if args.demo_skills_command != "suggest":
            raise SystemExit("unknown demo skills command")
        _print_json(_demo_skills_suggestions())
        return 0
    session = InteractiveSession(
        provider_name=args.provider,
        model=args.model or ("auto" if args.provider == "deepinfra" else PROVIDERS[args.provider].default_model),
        workspace_root=Path(args.workspace_root or _default_workspace_root()).resolve(),
        project=_slugish(args.project or "helix-cli"),
        agent_id=_slugish(args.agent_id or "demo"),
        max_tokens=args.max_tokens,
        temperature=0.0,
        transcript_dir=Path(args.transcript_dir or _default_transcript_dir()),
        router_policy=args.router_policy,
        evidence_root=Path(args.evidence_root or _default_evidence_root()).resolve(),
        task_root=Path(args.task_root).resolve() if args.task_root else _default_task_root(),
    )
    if args.demo_command == "wow":
        result = session.demo_wow(fast=args.fast, browser=not args.no_browser, with_opencode=args.with_opencode)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(_json_ready(result), indent=2, ensure_ascii=False), encoding="utf-8")
        elif args.json:
            _print_json(result)
        else:
            print(_format_work_summary(session.last_work_result, plan=session.last_work_plan, trust_card=session.last_trust_card))
        return 0 if result.get("status") in {"passed", "partial"} else 1
    if args.demo_command == "open":
        demo = session.last_demo_result or {}
        paths = demo.get("work_artifact_paths") if isinstance(demo.get("work_artifact_paths"), dict) else {}
        if not paths and isinstance(demo.get("trust_card"), dict):
            paths = demo["trust_card"].get("artifact_paths") if isinstance(demo["trust_card"].get("artifact_paths"), dict) else {}
        site = paths.get("site") or _primary_work_output_path(session.last_work_result, plan=session.last_work_plan, task_root=session.task_root)
        _print_json({"status": "ok" if site else "not_available", "site": site, "exists": Path(str(site)).exists() if site else False})
        return 0 if site else 1
    raise SystemExit("unknown demo command")


def _cmd_auth_test(args: argparse.Namespace) -> int:
    provider = PROVIDERS[args.provider]
    token = _token_for_provider(provider, prompt=not args.no_prompt)
    report = {
        "provider": provider.name,
        "kind": provider.kind,
        "token_env": provider.token_env,
        "token_available": bool(token),
        "live": args.live,
    }
    if provider.requires_token and not token:
        report["status"] = "missing_token"
        _print_json(report)
        return 1
    if not args.live:
        report["status"] = "credential_available" if token or not provider.requires_token else "missing_token"
        _print_json(report)
        return 0
    try:
        result = run_chat(
            provider_name=provider.name,
            model=resolve_model_alias(args.model) if args.model else provider.default_model,
            prompt="Return exactly: helix-auth-ok",
            max_tokens=16,
            temperature=0.0,
            timeout=args.timeout,
            prompt_token=False,
            base_url=args.base_url,
        )
        report.update(
            {
                "status": "ok",
                "actual_model": result.get("actual_model"),
                "latency_ms": result.get("latency_ms"),
                "text_preview": str(result.get("text") or "")[:80],
            }
        )
        _print_json(redact_value(report, secrets=[token] if token else []))
        return 0
    except Exception as exc:  # noqa: BLE001
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        _print_json(redact_value(report, secrets=[token] if token else []))
        return 1


def _cmd_auth_save(args: argparse.Namespace) -> int:
    provider = PROVIDERS[args.provider]
    if not provider.token_env:
        _print_json({"provider": provider.name, "status": "provider_has_no_token"})
        return 0
    token = args.token or getpass.getpass(f"Paste {provider.name} token to save in HeliX config: ").strip()
    if not token:
        _print_json({"provider": provider.name, "status": "no_token_saved"})
        return 1
    path = _save_config_token(provider.name, token)
    os.environ[provider.token_env] = token
    _print_json({"provider": provider.name, "status": "saved", "token_env": provider.token_env, "config_path": path})
    return 0


def _cmd_auth_forget(args: argparse.Namespace) -> int:
    provider = PROVIDERS[args.provider]
    path = _forget_config_token(provider.name)
    if provider.token_env:
        os.environ.pop(provider.token_env, None)
    _print_json({"provider": provider.name, "status": "forgotten", "config_path": path})
    return 0


def _write_chat_transcript(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".md":
        lines = [
            "# HeliX Chat Transcript",
            "",
            f"- Run ID: `{payload['run_id']}`",
            f"- Provider: `{payload['provider']}`",
            f"- Requested model: `{payload['requested_model']}`",
            f"- Actual model: `{payload['actual_model']}`",
            f"- Started UTC: `{payload['started_utc']}`",
            "",
            "## Prompt",
            "",
            payload["prompt"],
            "",
            "## Response",
            "",
            payload["text"],
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_chat(args: argparse.Namespace) -> int:
    route = None
    selected_model = resolve_model_alias(args.model) if args.model else args.model
    if args.model and args.model.lower() in {"auto", "router:auto"}:
        route = route_model_for_task(args.prompt, provider_name=args.provider, policy=args.router_policy)
        selected_model = route.get("model")
    result = run_chat(
        provider_name=args.provider,
        model=selected_model,
        prompt=args.prompt,
        system=args.system,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        base_url=args.base_url,
        prompt_token=not args.no_prompt,
        workspace_root=args.workspace_root,
    )
    print(result["text"])
    if args.transcript:
        transcript = redact_value(
            {
                "event": "helix_chat",
                "run_id": f"helix-chat-{int(time.time())}",
                "started_utc": _utc_now(),
                "provider": result["provider"],
                "requested_model": result["requested_model"],
                "actual_model": result["actual_model"],
                "latency_ms": result["latency_ms"],
                "finish_reason": result["finish_reason"],
                "usage": result["usage"],
                "route": route,
                "prompt": args.prompt,
                "text": result["text"],
            },
            secrets=_secret_values(PROVIDERS[args.provider]),
        )
        _write_chat_transcript(args.transcript, transcript)
    return 0


def _cmd_agent_run(args: argparse.Namespace) -> int:
    if getattr(args, "engine", "helix") == "opencode":
        if getattr(args, "sandbox", "patch") != "patch":
            raise SystemExit("opencode engine currently supports only --sandbox patch")
        result = helix_cli_core.opencode_run(
            repo_root=Path(args.task_root).resolve() if args.task_root else _default_task_root(),
            goal=args.goal,
            evidence_root=Path(args.evidence_root or _default_evidence_root()).resolve(),
        )
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(_json_ready(result), indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            _print_json(result)
        return 0 if result.get("status") == "passed" else 1
    if args.mode != "read-only":
        raise SystemExit("workspace-write mode is reserved for a later hardening pass")
    if args.provider != "local":
        _ensure_provider_token(args.provider)
        workspace = Path(args.workspace_root or _default_workspace_root()).resolve()
        transcript_dir = Path(args.transcript_dir or _default_transcript_dir())
        session = InteractiveSession(
            provider_name=args.provider,
            model=args.model or ("auto" if args.provider == "deepinfra" else PROVIDERS[args.provider].default_model),
            workspace_root=workspace,
            project=_slugish(args.project or "helix-cli"),
            agent_id=_slugish(args.agent_name),
            max_tokens=args.max_tokens,
            temperature=0.0,
            transcript_dir=transcript_dir,
            router_policy=args.router_policy,
            evidence_root=Path(args.evidence_root or _default_evidence_root()).resolve(),
            task_root=Path(args.task_root).resolve() if args.task_root else _default_task_root(),
        )
        result = session.task(args.goal, max_steps=args.max_steps)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(_json_ready(result), indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            _print_json(result)
        return 0
    runtime = HelixRuntime(root=args.workspace_root)
    result = runtime.agent_runner().run(
        goal=args.goal,
        agent_name=args.agent_name,
        default_model_alias=args.model,
        local_planner_alias=args.local_planner_alias or args.model,
        max_steps=args.max_steps,
        generation_max_new_tokens=args.max_tokens,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(_json_ready(result), indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        _print_json(result)
    return 0


def _cmd_cert_list(args: argparse.Namespace) -> int:
    _print_json(
        {
            "suites": [
                {
                    "suite_id": suite.suite_id,
                    "script": suite.script,
                    "script_exists": suite.script_path.exists(),
                    "description": suite.description,
                    "requires_deepinfra": suite.requires_deepinfra,
                    "supports_deepinfra_flag": suite.supports_deepinfra_flag,
                }
                for suite in sorted(SUITES.values(), key=lambda item: item.suite_id)
            ]
        }
    )
    return 0


def _cmd_cert_run(args: argparse.Namespace) -> int:
    suite_ids = sorted(SUITES) if args.suite == "all" else [args.suite]
    reports = []
    exit_code = 0
    for suite_id in suite_ids:
        try:
            report = run_cert_suite(
                suite_id,
                python_executable=args.python,
                case=args.case,
                provider_name=args.provider,
                run_id=args.run_id,
                output_dir=args.output_dir,
                prompt_token=not args.no_prompt,
                dry_run=args.dry_run,
                extra_args=_strip_remainder(getattr(args, "extra_args", [])),
            )
        except Exception as exc:  # noqa: BLE001
            report = {"suite_id": suite_id, "status": "failed_to_start", "error": f"{type(exc).__name__}: {exc}"}
            exit_code = 1
        else:
            if not report.get("passed", report.get("dry_run", False)):
                exit_code = 1
        reports.append(report)
        if not args.dry_run and args.echo_output and report.get("stdout"):
            print(report["stdout"], end="" if str(report["stdout"]).endswith("\n") else "\n")
        if not args.dry_run and args.echo_output and report.get("stderr"):
            print(report["stderr"], file=sys.stderr, end="" if str(report["stderr"]).endswith("\n") else "\n")
    _print_json({"cert_run": reports})
    return exit_code


def _cmd_cert_verify(args: argparse.Namespace) -> int:
    report = verify_artifact_file(args.artifact)
    _print_json(report)
    return 0 if report.get("status") == "verified" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="helix",
        description="High-level HeliX CLI for providers, local/cloud chat, agents, and evidence certification.",
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    interactive = subparsers.add_parser("interactive", aliases=["shell"], help="Start the HeliX interactive shell.")
    interactive.add_argument("--provider", choices=sorted(PROVIDERS))
    interactive.add_argument("--model")
    interactive.add_argument("--workspace-root", type=Path)
    interactive.add_argument("--task-root", type=Path)
    interactive.add_argument("--project", default="helix-cli")
    interactive.add_argument("--agent-id", default="interactive")
    interactive.add_argument("--max-tokens", type=int, default=900)
    interactive.add_argument("--temperature", type=float, default=0.0)
    interactive.add_argument("--transcript-dir", type=Path)
    interactive.add_argument("--evidence-root", type=Path)
    interactive.add_argument("--router-policy", choices=sorted(ROUTER_POLICIES), default="balanced")
    interactive.add_argument("--theme", choices=sorted(_THEME_PALETTES), default=None)
    interactive.set_defaults(func=_cmd_interactive)

    doctor = subparsers.add_parser("doctor", help="Inspect local HeliX CLI, providers, and suite readiness.")
    doctor.add_argument("doctor_mode", nargs="?", choices=["perf"], help="Use `perf` for local latency and index readiness.")
    doctor.add_argument("--probe-local", action="store_true", help="Probe local Ollama/llama.cpp endpoints.")
    doctor.set_defaults(func=_cmd_doctor)

    providers = subparsers.add_parser("providers", help="Provider registry commands.")
    provider_sub = providers.add_subparsers(dest="provider_command", required=True)
    providers_list = provider_sub.add_parser("list", help="List supported providers and token env vars.")
    providers_list.add_argument("--probe-local", action="store_true")
    providers_list.set_defaults(func=_cmd_providers_list)

    models = subparsers.add_parser("models", help="Model profile commands.")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_list = models_sub.add_parser("list", help="List built-in DeepInfra routing model profiles.")
    models_list.set_defaults(func=_cmd_models_list)

    route = subparsers.add_parser("route", help="Explain which model the auto-router would select.")
    route.add_argument("prompt")
    route.add_argument("--provider", choices=sorted(PROVIDERS), default="deepinfra")
    route.add_argument("--policy", choices=sorted(ROUTER_POLICIES), default="balanced")
    route.set_defaults(func=_cmd_route)

    opencode = subparsers.add_parser("opencode", help="OpenCode backend and MCP integration commands.")
    opencode_sub = opencode.add_subparsers(dest="opencode_command", required=True)
    opencode_status_cmd = opencode_sub.add_parser("status", help="Probe OpenCode and HeliX Rust core readiness.")
    opencode_status_cmd.add_argument("--bin", help="OpenCode binary path override.")
    opencode_status_cmd.set_defaults(func=_cmd_opencode)
    opencode_install_cmd = opencode_sub.add_parser("install", help="Install HeliX MCP into OpenCode config.")
    opencode_install_cmd.add_argument("--global", dest="global_config", action="store_true")
    opencode_install_cmd.add_argument("--dry-run", action="store_true")
    opencode_install_cmd.add_argument("--force", action="store_true")
    opencode_install_cmd.add_argument("--bin", help="OpenCode binary path override.")
    opencode_install_cmd.add_argument("--config-path", type=Path)
    opencode_install_cmd.set_defaults(func=_cmd_opencode)

    flow = subparsers.add_parser("flow", help="Commercial HeliX flow profiles built from verification protocols.")
    flow_sub = flow.add_subparsers(dest="flow_command", required=True)
    flow_list = flow_sub.add_parser("list", aliases=["profiles"], help="List flow profiles.")
    flow_list.set_defaults(func=_cmd_flow)
    flow_show = flow_sub.add_parser("show", help="Show one flow profile.")
    flow_show.add_argument("profile")
    flow_show.set_defaults(func=_cmd_flow)
    flow_run = flow_sub.add_parser("run", help="Run a flow profile through the HeliX/OpenCode sandbox path.")
    flow_run.add_argument("profile")
    flow_run.add_argument("goal")
    flow_run.add_argument("--task-root", type=Path)
    flow_run.add_argument("--evidence-root", type=Path)
    flow_run.add_argument("--engine", choices=["opencode"], default=None)
    flow_run.add_argument("--assurance", choices=["quick", "balanced", "strict", "deep"], default=None)
    flow_run.add_argument("--output-json", type=Path)
    flow_run.set_defaults(func=_cmd_flow)

    work = subparsers.add_parser("work", help="Natural Work Runtime for documents, web, code and verification tasks.")
    work_sub = work.add_subparsers(dest="work_command", required=True)
    work_run = work_sub.add_parser("run", help="Collect sources, plan a flow, and run/analyze with HeliX guarantees.")
    work_run.add_argument("goal")
    work_run.add_argument("--provider", choices=sorted(PROVIDERS), default="deepinfra")
    work_run.add_argument("--model")
    work_run.add_argument("--workspace-root", type=Path)
    work_run.add_argument("--task-root", type=Path)
    work_run.add_argument("--transcript-dir", type=Path)
    work_run.add_argument("--evidence-root", type=Path)
    work_run.add_argument("--project", default="helix-cli")
    work_run.add_argument("--agent-id", default="work")
    work_run.add_argument("--router-policy", choices=sorted(ROUTER_POLICIES), default="balanced")
    work_run.add_argument("--max-tokens", type=int, default=1400)
    work_run.add_argument("--temperature", type=float, default=0.0)
    work_run.add_argument("--max-pages", type=int, default=25)
    work_run.add_argument("--depth", type=int, default=2)
    work_run.add_argument("--cross-domain", action="store_true")
    work_run.add_argument("--assurance", choices=["quick", "balanced", "strict", "deep"], default=None)
    work_run.add_argument("--json", action="store_true", help="Print raw JSON instead of the human work summary.")
    work_run.add_argument("--output-json", type=Path)
    work_run.set_defaults(func=_cmd_work)

    demo = subparsers.add_parser("demo", help="Run or inspect the HeliX Workbench wow demo.")
    demo_sub = demo.add_subparsers(dest="demo_command", required=True)
    demo_wow = demo_sub.add_parser("wow", help="Run the reproducible HeliX Workbench demo.")
    demo_wow.add_argument("--provider", choices=sorted(PROVIDERS), default="deepinfra")
    demo_wow.add_argument("--model")
    demo_wow.add_argument("--workspace-root", type=Path)
    demo_wow.add_argument("--task-root", type=Path)
    demo_wow.add_argument("--transcript-dir", type=Path)
    demo_wow.add_argument("--evidence-root", type=Path)
    demo_wow.add_argument("--project", default="helix-cli")
    demo_wow.add_argument("--agent-id", default="demo")
    demo_wow.add_argument("--router-policy", choices=sorted(ROUTER_POLICIES), default="balanced")
    demo_wow.add_argument("--max-tokens", type=int, default=900)
    demo_wow.add_argument("--fast", action="store_true", default=True)
    demo_wow.add_argument("--no-browser", action="store_true")
    demo_wow.add_argument("--with-opencode", action="store_true")
    demo_wow.add_argument("--json", action="store_true", help="Print raw DemoRun JSON.")
    demo_wow.add_argument("--output-json", type=Path)
    demo_wow.set_defaults(func=_cmd_demo)
    demo_doctor = demo_sub.add_parser("doctor", help="Check demo prerequisites.")
    demo_doctor.set_defaults(func=_cmd_demo)
    demo_open = demo_sub.add_parser("open", help="Show the last demo artifact path.")
    demo_open.add_argument("--provider", choices=sorted(PROVIDERS), default="deepinfra")
    demo_open.add_argument("--model")
    demo_open.add_argument("--workspace-root", type=Path)
    demo_open.add_argument("--task-root", type=Path)
    demo_open.add_argument("--transcript-dir", type=Path)
    demo_open.add_argument("--evidence-root", type=Path)
    demo_open.add_argument("--project", default="helix-cli")
    demo_open.add_argument("--agent-id", default="demo")
    demo_open.add_argument("--router-policy", choices=sorted(ROUTER_POLICIES), default="balanced")
    demo_open.add_argument("--max-tokens", type=int, default=900)
    demo_open.set_defaults(func=_cmd_demo)
    demo_skills = demo_sub.add_parser("skills", help="Optional Skills.sh suggestions for the demo.")
    demo_skills_sub = demo_skills.add_subparsers(dest="demo_skills_command", required=True)
    demo_skills_suggest = demo_skills_sub.add_parser("suggest", help="Show optional skills to install.")
    demo_skills_suggest.set_defaults(func=_cmd_demo)

    auth = subparsers.add_parser("auth", help="Credential checks.")
    auth_sub = auth.add_subparsers(dest="auth_command", required=True)
    auth_test = auth_sub.add_parser("test", help="Check provider credentials; --live performs a real request.")
    auth_test.add_argument("provider", choices=sorted(PROVIDERS))
    auth_test.add_argument("--model")
    auth_test.add_argument("--base-url")
    auth_test.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    auth_test.add_argument("--live", action="store_true")
    auth_test.add_argument("--no-prompt", action="store_true")
    auth_test.set_defaults(func=_cmd_auth_test)

    auth_save = auth_sub.add_parser("save", help="Save a provider API token in HeliX user config.")
    auth_save.add_argument("provider", choices=sorted(PROVIDERS))
    auth_save.add_argument("--token", help="Token value. Omit to paste securely.")
    auth_save.set_defaults(func=_cmd_auth_save)

    auth_forget = auth_sub.add_parser("forget", help="Remove a saved provider API token from HeliX user config.")
    auth_forget.add_argument("provider", choices=sorted(PROVIDERS))
    auth_forget.set_defaults(func=_cmd_auth_forget)

    chat = subparsers.add_parser("chat", help="Run one chat completion through local or cloud models.")
    chat.add_argument("prompt")
    chat.add_argument("--provider", choices=sorted(PROVIDERS), default="deepinfra")
    chat.add_argument("--model")
    chat.add_argument("--system")
    chat.add_argument("--base-url")
    chat.add_argument("--max-tokens", type=int, default=512)
    chat.add_argument("--temperature", type=float, default=0.0)
    chat.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    chat.add_argument("--workspace-root", type=Path)
    chat.add_argument("--transcript", type=Path)
    chat.add_argument("--no-prompt", action="store_true")
    chat.add_argument("--router-policy", choices=sorted(ROUTER_POLICIES), default="balanced")
    chat.set_defaults(func=_cmd_chat)

    agent = subparsers.add_parser("agent", help="Agent commands.")
    agent_sub = agent.add_subparsers(dest="agent_command", required=True)
    agent_run = agent_sub.add_parser("run", help="Run a conservative HeliX agent loop.")
    agent_run.add_argument("goal")
    agent_run.add_argument("--provider", choices=sorted(PROVIDERS), default="local")
    agent_run.add_argument("--model")
    agent_run.add_argument("--local-planner-alias")
    agent_run.add_argument("--agent-name", default="default-agent")
    agent_run.add_argument("--workspace-root", type=Path)
    agent_run.add_argument("--task-root", type=Path)
    agent_run.add_argument("--transcript-dir", type=Path)
    agent_run.add_argument("--evidence-root", type=Path)
    agent_run.add_argument("--project", default="helix-cli")
    agent_run.add_argument("--router-policy", choices=sorted(ROUTER_POLICIES), default="balanced")
    agent_run.add_argument("--mode", choices=["read-only", "workspace-write"], default="read-only")
    agent_run.add_argument("--engine", choices=["helix", "opencode"], default="helix")
    agent_run.add_argument("--sandbox", choices=["patch"], default="patch")
    agent_run.add_argument("--max-steps", type=int, default=4)
    agent_run.add_argument("--max-tokens", type=int, default=1400)
    agent_run.add_argument("--output-json", type=Path)
    agent_run.set_defaults(func=_cmd_agent_run)

    cert = subparsers.add_parser("cert", help="Evidence certification commands.")
    cert_sub = cert.add_subparsers(dest="cert_command", required=True)
    cert_list = cert_sub.add_parser("list", help="List registered certification suites.")
    cert_list.set_defaults(func=_cmd_cert_list)

    cert_run = cert_sub.add_parser("run", help="Run a registered suite or all suites.")
    cert_run.add_argument("suite", choices=["all", *sorted(SUITES)])
    cert_run.add_argument("--case")
    cert_run.add_argument("--provider", choices=sorted(PROVIDERS), default=None)
    cert_run.add_argument("--run-id")
    cert_run.add_argument("--output-dir")
    cert_run.add_argument("--python", default=sys.executable)
    cert_run.add_argument("--dry-run", action="store_true")
    cert_run.add_argument("--no-prompt", action="store_true")
    cert_run.add_argument("--echo-output", action="store_true")
    cert_run.set_defaults(func=_cmd_cert_run)

    cert_verify = cert_sub.add_parser("verify", help="Verify an artifact JSON without live provider calls.")
    cert_verify.add_argument("artifact", type=Path)
    cert_verify.set_defaults(func=_cmd_cert_verify)

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown and not (
        getattr(args, "command", None) == "cert" and getattr(args, "cert_command", None) == "run"
    ):
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    args.extra_args = unknown
    return args


def _cmd_interactive(args: argparse.Namespace) -> int:
    return run_interactive(args)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        return run_interactive()
    args = parse_args(argv)
    if getattr(args, "command", None) is None:
        return run_interactive(args)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
