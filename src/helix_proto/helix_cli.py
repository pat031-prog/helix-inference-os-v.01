from __future__ import annotations

import argparse
import fnmatch
import getpass
import hashlib
import html
import json
import os
import re
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request
from urllib import parse as urlparse

from helix_proto.helix_cli_chrome import Console
from helix_proto.helix_cli_chrome import PromptSession
from helix_proto.helix_cli_chrome import WordCompleter
from helix_proto.helix_cli_chrome import DEFAULT_THEME
from helix_proto.helix_cli_chrome import HAS_UI as _HAS_UI
from helix_proto.helix_cli_chrome import THEME_PALETTES as _THEME_PALETTES
from helix_proto.helix_cli_chrome import choose_option as _choose_ui_option
from helix_proto.helix_cli_chrome import normalize_theme_name as _normalize_theme_name
from helix_proto.helix_cli_chrome import panel_width as _chrome_panel_width
from helix_proto.helix_cli_chrome import play_boot_handshake as _play_boot_handshake
from helix_proto.helix_cli_chrome import prompt_bottom_toolbar as _prompt_bottom_toolbar
from helix_proto.helix_cli_chrome import prompt_message as _prompt_message
from helix_proto.helix_cli_chrome import prompt_style as _chrome_prompt_style
from helix_proto.helix_cli_chrome import prompt_toolbar_markup as _prompt_toolbar_markup
from helix_proto.helix_cli_chrome import render_boot_banner as _render_boot_banner
from helix_proto.helix_cli_chrome import render_chat_response as _render_chat_response
from helix_proto.helix_cli_chrome import render_session_ribbon as _render_session_ribbon
from helix_proto.helix_cli_chrome import render_task_result as _render_task_result_panel
from helix_proto.helix_cli_chrome import render_verify_audit as _render_verify_audit
from helix_proto.helix_cli_chrome import rich_theme as _chrome_rich_theme
from helix_proto.helix_cli_chrome import run_with_status as _chrome_run_with_status
from helix_proto.helix_cli_chrome import theme_palette as _chrome_theme_palette
from helix_proto.helix_cli_chrome import theme_report as _theme_report


def _theme_palette(theme_name: str | None) -> dict[str, str]:
    return _chrome_theme_palette(theme_name)


def _rich_theme(theme_name: str | None):
    return _chrome_rich_theme(theme_name)


def _prompt_style(theme_name: str | None):
    return _chrome_prompt_style(theme_name)

_VISIBLE_OUTPUT_RE = re.compile(r"(?is)<helix_output>(.*?)</helix_output>")
_INTERNAL_BLOCK_RE = re.compile(
    r"(?is)<(think|thinking|tool_call|scratchpad|analysis|reasoning|plan|draft)\b[^>]*>.*?</\1>"
)
_UNCLOSED_INTERNAL_RE = re.compile(
    r"(?is)<(think|thinking|tool_call|scratchpad|analysis|reasoning|plan|draft)\b[^>]*>.*$"
)
_FINAL_MARKER_RE = re.compile(
    r"(?im)^\s*(final answer|final output|actual output|refined answer|respuesta final|output|response)\s*:\s*"
)
_URL_REF_RE = re.compile(r"(?i)\bhttps?://[^\s<>\"]+")
_PROVIDER_COOLDOWNS: dict[str, dict[str, Any]] = {}
_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 75.0


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
    if re.match(r"^\s*(plan|draft|reasoning|analysis|mental|checks?)\s*:", lowered):
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
    return _chrome_panel_width(active_console, context="chat")


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


def _run_with_status(active_console: Any, func: Any, *, phase: str = "thinking") -> Any:
    return _chrome_run_with_status(active_console, func, phase=phase, phase_messages=_THINKING_MESSAGE_PHASES)


def _clean_assistant_text(text: str) -> str:
    """Return only the user-visible answer from a noisy model response."""
    if not text:
        return ""

    raw = str(text).strip()
    tagged = _VISIBLE_OUTPUT_RE.findall(raw)
    if tagged:
        raw = tagged[-1]

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

from helix_proto.agent import PlannerDecision
from helix_proto.api import HelixRuntime
from helix_proto.artifact_replay import verify_artifact_file
from helix_proto.evidence_ingest import ingest_artifact_file, list_ingested_evidence, refresh_evidence
from helix_proto import hmem
from helix_proto.helix_cli_agent_shell import extract_patch as _extract_patch
from helix_proto.helix_cli_agent_shell import normalize_tool_event as _normalize_tool_event
from helix_proto.helix_cli_agent_shell import parse_agent_tool_calls as _parse_agent_tool_calls
from helix_proto.helix_cli_agent_shell import tool_event_detail as _tool_event_detail
from helix_proto.provider_audit import OPENAI_COMPATIBLE_PROVIDERS
from helix_proto.tools import ToolRegistry, ToolSpec


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
        providers[provider.name] = ProviderSpec(
            name=provider.name,
            kind="openai-compatible",
            base_url=provider.base_url,
            token_env=provider.token_env,
            default_model="Qwen/Qwen3.6-35B-A3B",
            requires_token=True,
            description="OpenAI-compatible cloud provider",
            native_capabilities=("chat_completions", "structured_output"),
            native_constraints=("Capabilities are curated in-repo; HeliX does not do live model discovery at runtime.",),
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


MODEL_PROFILES: dict[str, ModelProfile] = {
    **DEEPINFRA_MODEL_PROFILES,
    **GEMINI_MODEL_PROFILES,
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
        for name in ("deepinfra", "openai", "anthropic")
        if name in PROVIDERS and name != provider_name and _provider_ready(name)
    ]
    if alternates:
        suffix += f" Hay fallback disponible: {', '.join(alternates[:3])}."
    else:
        suffix += " No veo otro provider listo con credenciales locales, así que prefiero no inventar una respuesta sin modelo."
    return (
        f"{provider_label} devolvió rate limit/cuota temporal (`HTTP 429`). "
        "El turno quedó registrado, pero la respuesta del modelo no se generó."
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
        "structured_output": False,
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


def route_model_for_task(
    text: str,
    *,
    provider_name: str = "deepinfra",
    policy: str = "balanced",
    interaction_mode: str = "balanced",
) -> dict[str, Any]:
    """Select a model for one turn using transparent heuristics.

    This is intentionally deterministic. The router should be auditable before
    it becomes another model call.
    """

    lowered = str(text or "").lower()
    active_mode = _normalize_interaction_mode(interaction_mode)
    url_refs = _extract_url_refs(text)
    path_refs = _extract_local_path_refs(text)
    blueprint = _resolve_router_blueprint(policy)
    policy = blueprint.name
    signals: list[str] = []
    explicit_alias = _explicit_model_control_alias(lowered)

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
    _print_table(
        _compact_model_rows(),
        [
            ("alias", "alias", 18),
            ("provider", "provider", 10),
            ("role", "role", 18),
            ("model", "model", 34),
            ("cost", "$/M in/out", 12),
            ("use", "use", 58),
        ],
    )
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
) -> dict[str, Any]:
    url = f"{(base_url or provider.base_url or '').rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    started = time.perf_counter()

    def _payload_for(call_messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "model": model,
            "messages": call_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

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
        return {
            "provider": provider.name,
            "requested_model": selected_model,
            "actual_model": selected_model,
            "text": str(result.get("completion_text") or result.get("generated_text") or ""),
            "finish_reason": None,
            "usage": None,
            "latency_ms": latency_ms,
            "raw": result,
        }

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
        return _anthropic_chat(
            provider,
            model=selected_model,
            messages=messages,
            token=str(token),
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
    if provider.kind == "gemini":
        return _gemini_chat(
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
    return _openai_compatible_chat(
        provider,
        model=selected_model,
        messages=messages,
        token=token,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        base_url=base_url,
    )


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

    def search(self, query: str, *, limit: int = 12) -> dict[str, Any]:
        query_l = str(query or "").lower().strip()
        if not query_l:
            return {"status": "error", "error": "query is required", "results": []}
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
    if any(str(item.get("tool") or "") == "helix.trust" for item in observations):
        instructions.append(
            "For helix.trust observations, separate local signed-checkpoint verification from semantic truth or global transparency; report canonical head, equivocation/quarantine and legacy warnings directly."
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
        self.last_runner_trace: dict[str, Any] | None = None
        self.last_model_turns: list[dict[str, Any]] = []
        self._state_path = self.workspace_root / "session-os" / "helix-cli-state.json"
        self.suite_catalog = SuiteEvidenceCatalog(evidence_root=self.evidence_root)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        active_thread_id = self._load_active_thread_id()
        if active_thread_id:
            self._activate_thread(active_thread_id, lifecycle_event="thread_resume")
        else:
            self.new_thread("interactive")

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

    def _load_active_thread_id(self) -> str | None:
        if not self._state_path.exists():
            return None
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        value = payload.get("active_thread_id")
        return _slugish(str(value)) if value else None

    def _save_active_thread_id(self, thread_id: str | None) -> None:
        self._state_path.write_text(
            json.dumps({"active_thread_id": thread_id}, indent=2, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )

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

    def _activate_thread(self, thread_id: str, *, lifecycle_event: str) -> dict[str, Any]:
        normalized_thread_id = _slugish(thread_id)
        self.thread_id = normalized_thread_id
        self.run_id = normalized_thread_id
        self.jsonl_path, self.md_path = self._thread_paths(normalized_thread_id)
        self.events = self._load_events(self.jsonl_path)
        self._save_active_thread_id(normalized_thread_id)
        payload = self.record(
            role="system",
            content=json.dumps({"thread_id": normalized_thread_id, "event": lifecycle_event}, ensure_ascii=False),
            event_type=lifecycle_event,
            metadata={"thread_id": normalized_thread_id},
            promote=True,
            session_id=normalized_thread_id,
        )
        try:
            self.refresh_evidence(None, limit=4)
        except Exception:
            pass
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
        return self._activate_thread(_run_id(title or "helix-thread"), lifecycle_event="thread_open")

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
        return threads

    def current_thread(self) -> dict[str, Any]:
        thread_id = self.ensure_active_thread("interactive")
        return {
            "thread_id": thread_id,
            "jsonl_path": str(self.jsonl_path),
            "md_path": str(self.md_path),
            "event_count": len(self.events),
            "interaction_mode": self.interaction_mode,
            "tool_policy": self.tool_policy,
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
        jsonl_path, _md_path = self._thread_paths(active_session_id)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        if self.thread_id == active_session_id:
            self.events.append(event)
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

    def memory_context(self, query: str) -> dict[str, Any]:
        active_thread_id = self.ensure_active_thread(query[:32] if query else "interactive")
        try:
            self.refresh_evidence(query or None, limit=6)
        except Exception:
            pass
        return hmem.build_context(
            root=self.workspace_root,
            project=self.project,
            agent_id=self.agent_id,
            session_id=active_thread_id,
            query=query,
            budget_tokens=900,
            mode="search",
            limit=6,
            retrieval_scope="workspace",
        )

    def refresh_evidence(self, query: str | None = None, *, limit: int = 8) -> dict[str, Any]:
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
            if mode in {"chat", "task"} and (helix_focus or helix_auditability) and architecture_context_pack and not observations:
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
        active_interaction_mode = _normalize_interaction_mode(interaction_mode_override or self.interaction_mode)
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
            route = route_model_for_task(
                f"{user_text}\nMode: chat",
                provider_name=self.provider_name,
                policy=self.router_policy,
                interaction_mode=active_interaction_mode,
            )
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
            route = _manual_route_for_model(
                selected_model,
                provider_name=self.provider_name,
                policy=self.router_policy,
                user_text=user_text,
                interaction_mode=active_interaction_mode,
            )
            selected_provider_name = str(route.get("provider") or self.provider_name)
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

        repository_evidence_pack = None
        if _needs_repository_evidence(user_text, route):
            repository_evidence_pack = self.refresh_evidence(user_text, limit=8)

        context = self.memory_context(user_text)
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
                "path_refs": file_path_refs,
                "url_refs": url_refs,
                "native_tool_plan": native_tool_plan,
                "capability_requirements": capability_requirements,
                "interaction_mode": active_interaction_mode,
                "mode_policy": mode_policy,
                "grounding_plan": grounding_plan,
                "tone_contract": tone_contract,
                "architecture_context_enabled": bool(architecture_context_pack),
            },
        )
        excluded_memory_ids = [str((user_event.get("helix_memory") or {}).get("memory_id") or "")]
        excluded_memory_ids = [item for item in excluded_memory_ids if item]
        identity_evidence = None
        if _needs_certified_evidence(user_text, route, recent_history):
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
            fallback_models=fallback_models,
            timeout=None,
        )
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
            retrieval_scope="workspace",
            memory_exclude_ids=excluded_memory_ids,
            max_steps=4,
        )
        self.last_runner_trace = trace
        self.last_model_turns = model_turns
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
                "path_refs": file_path_refs,
                "url_refs": url_refs,
                "native_tool_plan": native_tool_plan,
                "capability_requirements": capability_requirements,
                "interaction_mode": active_interaction_mode,
                "mode_policy": mode_policy,
                "grounding_plan": grounding_plan,
                "tone_contract": tone_contract,
                "architecture_context_enabled": bool(architecture_context_pack),
                "raw_model_text": raw_text,
                "visible_output_cleaned": text != raw_text,
                "reasoning_internal": "",
                "trace_path": trace.get("trace_path"),
                "thread_id": self.thread_id,
            },
        )
        return {"text": text, "raw_text": raw_text, "reasoning": "", "route": route, "trace": trace, "interaction_mode": active_interaction_mode}

    def task(
        self,
        goal: str,
        *,
        max_steps: int = 5,
        mode_override: str | None = None,
        interaction_mode_override: str | None = None,
        agent_blueprint: AgentBlueprint | None = None,
    ) -> dict[str, Any]:
        active_interaction_mode = _normalize_interaction_mode(interaction_mode_override or self.interaction_mode)
        route = route_model_for_task(
            f"{goal}\nTask mode: inspect repo, use tools, propose patch if needed.",
            provider_name=self.provider_name,
            policy=self.router_policy,
            interaction_mode=active_interaction_mode,
        )
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
        repository_evidence_pack = self.refresh_evidence(goal, limit=8)
        context = self.memory_context(goal)
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
            fallback_models=fallback_models,
            timeout=AGENT_TASK_TIMEOUT_SECONDS,
        )
        try:
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
                retrieval_scope="workspace",
                memory_exclude_ids=excluded_memory_ids,
                max_steps=max(1, max_steps),
            )
        except Exception as exc:  # noqa: BLE001
            error_text = f"{type(exc).__name__}: {exc}"
            final_text = _friendly_provider_failure_text(error_text) or f"Task failed: {error_text}"
            self.last_patch = None
            task_result = {
                "status": "error",
                "mode": active_agent_mode,
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
                "patch_available": False,
                "error": error_text,
            }
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
                    "url_refs": url_refs,
                    "native_tool_plan": native_tool_plan,
                    "capability_requirements": capability_requirements,
                    "interaction_mode": active_interaction_mode,
                    "mode_policy": mode_policy,
                    "grounding_plan": grounding_plan,
                    "tone_contract": tone_contract,
                    "architecture_context_enabled": bool(architecture_context_pack),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return task_result

        self.last_runner_trace = trace
        self.last_model_turns = model_turns
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
        task_result = {
            "status": status,
            "mode": active_agent_mode,
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
            "failover_used": model_turns[-1].get("failover_used") if model_turns else None,
            "failover_attempts": model_turns[-1].get("failover_attempts") if model_turns else [],
            "patch_available": bool(patch),
            "trace_path": trace.get("trace_path"),
        }
        if status == "error":
            task_result["error"] = planner_errors[-1] if planner_errors else final_text
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
                "url_refs": url_refs,
                "native_tool_plan": native_tool_plan,
                "capability_requirements": capability_requirements,
                "interaction_mode": active_interaction_mode,
                "mode_policy": mode_policy,
                "grounding_plan": grounding_plan,
                "tone_contract": tone_contract,
                "architecture_context_enabled": bool(architecture_context_pack),
                "tool_event_count": len(tool_events),
                "patch_available": bool(patch),
                "trace_path": trace.get("trace_path"),
            },
        )
        return task_result

    def status(self) -> dict[str, Any]:
        profile = _model_profile_for_id(self.model if self.model.lower() not in {"auto", "router:auto"} else None)
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
            "router_policy": self.router_policy,
            "interaction_mode": self.interaction_mode,
            "interaction_mode_profile": _interaction_mode_payload(self.interaction_mode),
            "theme": self.theme_name,
            "response_style": self.response_style,
            "agent_mode": self.agent_mode,
            "tool_policy": self.tool_policy,
            "last_patch_available": bool(self.last_patch),
            "config_path": str(_config_path()),
            "state_path": str(self._state_path),
        }


HELP_TEXT = """Commands:
  /help                         Show this help
  /status                       Show provider, model, workspace and transcript paths
  /provider NAME                Switch provider: deepinfra, gemini, openai, anthropic, ollama, llamacpp, local, ...
  /model NAME                   Switch model; aliases include auto, qwen-big, mistral, qwen, gemma, gemini-pro, gemini-pro-tools, gemini-flash, gemini-2.5-pro, coder, llama-vision, sonnet
  /model use NAME               Same as /model NAME; persists until /model auto
  /model list                   List model aliases and router blueprints
  /with MODEL PROMPT            Use one model for a single action, then restore the previous model
  /models                       Open/select or compact-list model profiles; /models json for full metadata
  /route TEXT                   Explain which model auto-routing would pick
  /web QUERY                    Search the public web directly and show raw result metadata
  /router NAME                  Change routing blueprint/policy: balanced, qwen-heavy, current, qwen-gemma-mistral, cheap, premium
  /router why TEXT              Explain intent scores, model choice and fallback chain
  /router list                  Open or print the routing blueprint selector
  /theme NAME                   Switch terminal theme: industrial-brutalist, industrial-neon, xerox, brown-console
  /theme list                   Open or print the theme selector/report
  /style NAME                   Response register only: balanced, technical, forensic, vivid, terse
  /mode [NAME]                  Interaction mode: balanced, technical, explore; /mode list shows profiles
  /tech TEXT                    One-shot technical turn without changing the sticky mode
  /explore TEXT                 One-shot exploratory turn without changing the sticky mode
  /raw on|off                   Toggle raw model output after the cleaned answer
  /clear                        Clear the terminal
  /key [PROVIDER]               Prompt for a provider API key for this process only
  /key save [PROVIDER]          Save provider API key in HeliX user config
  /key forget [PROVIDER]        Remove saved provider API key from HeliX user config
  /config                       Show HeliX config/data paths
  /doctor                       Run helix doctor
  /providers                    List providers
  /cert SUITE [-- args]         Run a certification suite
  /cert-dry SUITE [-- args]     Show the suite command without running it
  /evidence refresh [QUERY]     Verify and ingest artifacts from verification/ into HeliX memory
  /evidence latest [N]          Show latest certified evidence memories
  /evidence search QUERY        Search certified evidence memories
  /evidence show MEMORY_ID      Show one certified evidence memory, receipt and chain status
  /verify PATH|latest|search Q  Verify an artifact JSON or discover verified artifacts
  /suites                       Compact catalog of local verification suites and latest artifacts
  /suite latest SUITE           Show latest artifact, manifest and transcript paths for one suite
  /suite transcripts SUITE      List suite transcript exports; add a filter after the suite name
  /suite search QUERY           Search artifacts/transcripts under verification/nuclear-methodology
  /suite read PATH_OR_NAME      Read a bounded local artifact/transcript excerpt
  /file PATH                    Inspect a local file or directory path under allowed HeliX roots
  /memory QUERY                 Search unified HeliX memory for this workspace, prioritizing the active thread
  /memory resolve HASH_OR_ID    Resolve a memory_id or node_hash prefix to exact stored content
  /trust [current|THREAD_ID]     Verify local canonical head, checkpoint chain and lineage trust status
  /thread new [TITLE]           Create and switch to a new persistent thread
  /thread list                  List known workspace threads
  /thread open THREAD_ID        Reopen an existing thread
  /thread close [THREAD_ID]     Close a thread without deleting its memory
  /thread current               Show the active thread
  /task GOAL                    Run the unified HeliX runner in read-only agentic mode
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
    answer = input(f"Optional: save {provider.name} API key now for models like {provider.default_model}? [y/N/skip]: ").strip().lower()
    if answer in {"skip", "never", "nope"}:
        optional[provider.name] = "skip"
        config["optional_token_prompts"] = optional
        _save_config(config)
        print(f"[helix] optional {provider.name} key prompt disabled. Use /key save {provider.name} if you want it later.")
        return config
    if answer not in {"y", "yes", "s", "si"}:
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
    if _is_identity_question(lowered):
        return True
    if _is_helix_auditability_request(lowered, history):
        return True
    if _is_helix_explanation_request(lowered, history):
        return True
    if _is_clear_non_helix_topic_shift(lowered) or _is_affective_reaction(lowered):
        return False
    evidence_terms = (
        "helix",
        "merkle",
        "dag",
        "hash",
        "receipt",
        "memoria",
        "memory",
        "tombstone",
        "lapida",
        "fence",
        "modelo",
        "modelos",
        "mistral",
        "claude",
        "sonnet",
        "deepinfra",
    )
    if any(term in lowered for term in evidence_terms):
        return True
    return bool(route and route.get("intent") in {"model_control"})


def _needs_repository_evidence(text: str, route: dict[str, Any] | None = None) -> bool:
    lowered = str(text or "").lower()
    evidence_terms = (
        "/verify",
        "verify",
        "artifact",
        "artefact",
        "artefacto",
        "evidencia",
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
    return bool(route and route.get("intent") in {"research", "audit", "suite_forensics"})


def _route_natural_language(text: str) -> str | None:
    lowered = text.lower()
    local_refs = _extract_local_path_refs(text)
    if _is_pasted_suite_analysis_request(text):
        return None
    if "/verify" in lowered and not any(term in lowered for term in ("resultado", "transcript", "corrida", "artifact", "manifest")):
        return "/suites"
    if local_refs and _is_local_file_request(text) and not _looks_like_agent_task(text):
        return f"/read {local_refs[0]}"
    execution_terms = ("corre", "ejecuta", "run ", "certifica")
    analysis_terms = ("quiero info", "quiero data", "dame info", "dame data", "contame", "analiza", "explica", "compara", "compará", "reporte", "resum")
    if any(term in lowered for term in execution_terms) and not any(term in lowered for term in analysis_terms):
        suite_id = _suite_from_text(text)
        if suite_id:
            return f"/cert {suite_id}"
    if _is_suite_evidence_request(text) and any(term in lowered for term in ("analiza", "compara", "compará", "explica", "reporte", "resumi", "resume")):
        return f"/task {text}"
    if _looks_like_agent_task(text):
        return f"/task {text}"
    if lowered.strip() in {"doctor", "diagnostico", "estado"}:
        return "/doctor"
    return None


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


def _select_model(session: InteractiveSession) -> str | None:
    options = [
        ("auto", "auto - let the router choose per prompt"),
        ("mistral", "mistral - fast conversational baseline"),
        ("qwen-big", "qwen-big - primary large Qwen for research, HeliX and synthesis"),
        ("qwen", "qwen - alias for qwen-big"),
        ("gemma", "gemma - careful reasoning and decomposition"),
        ("gemini-pro", "gemini-pro - Gemini 3.1 Pro preview via GEMINI_API_KEY"),
        ("gemini-pro-tools", "gemini-pro-tools - Gemini 3.1 Pro custom-tools preview"),
        ("gemini-flash", "gemini-flash - Gemini 3 Flash preview via GEMINI_API_KEY"),
        ("gemini-lite", "gemini-lite - Gemini 3.1 Flash Lite preview via GEMINI_API_KEY"),
        ("gemini-2.5-pro", "gemini-2.5-pro - stable Gemini 2.5 Pro fallback"),
        ("gemini-2.5-flash", "gemini-2.5-flash - stable Gemini 2.5 Flash fallback"),
        ("gemini-2.5-flash-lite", "gemini-2.5-flash-lite - stable Gemini 2.5 Flash-Lite fallback"),
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
    if name in {"help", "h", "?"}:
        print(HELP_TEXT)
        return True
    if name == "status":
        _print_json(session.status())
        return True
    if name == "thread":
        parts = _split_command(rest)
        subcommand = parts[0].lower() if parts else "current"
        argument = " ".join(parts[1:]).strip()
        if subcommand == "new":
            _print_json(session.new_thread(argument or "interactive"))
            return True
        if subcommand in {"list", "ls"}:
            _print_json({"threads": session.list_threads(limit=32)})
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
        print("Usage: /thread new [TITLE] | list | open THREAD_ID | close [THREAD_ID] | current")
        return True
    if name == "providers":
        _print_json({"providers": provider_report(probe_local=False)})
        return True
    if name == "models":
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
            query = argument or input("Suite evidence query: ").strip()
            _print_json(session.suite_catalog.search(query, limit=20))
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
        print("Usage: /suite list|show SUITE|latest SUITE|transcripts SUITE [FILTER]|search QUERY|read PATH|ingest [SUITE|all]")
        return True
    if name == "route":
        if not rest:
            print("Usage: /route TEXT")
            return True
        _print_json(
            route_model_for_task(
                rest,
                provider_name=session.provider_name,
                policy=session.router_policy,
                interaction_mode=session.interaction_mode,
            )
        )
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
        if rest:
            if not _is_known_interaction_mode(rest):
                print("Usage: /mode [balanced|technical|explore|list]")
                return True
            candidate = _normalize_interaction_mode(rest)
            session.interaction_mode = candidate
            config = _load_config()
            config["interaction_mode"] = session.interaction_mode
            _save_config(config)
            print(f"[helix] interaction_mode={session.interaction_mode}")
            return True
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
        argument = " ".join(parts[1:]).strip()
        if subcommand in {"refresh", "scan"}:
            pack = session.refresh_evidence(argument or None, limit=12)
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
        print("Usage: /evidence refresh [QUERY] | latest [N] | search QUERY | show MEMORY_ID")
        return True
    if name == "verify":
        if not rest:
            print("Usage: /verify PATH|latest|search QUERY")
            print("Related: /suites | /suite latest SUITE | /suite transcripts SUITE | /evidence latest")
            return True
        verify_parts = _split_command(rest)
        verify_mode = verify_parts[0].lower() if verify_parts else ""
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
        query = rest or input("Memory query: ").strip()
        _print_json(
            hmem.hybrid_search(
                root=session.workspace_root,
                project=session.project,
                agent_id=session.agent_id,
                session_id=session.thread_id,
                query=query,
                top_k=8,
                retrieval_scope="workspace",
            )
        )
        return True
    if name == "trust":
        parts = _split_command(rest)
        include_quarantined = any(part.lower() in {"--forensics", "--include-quarantined", "--quarantined"} for part in parts)
        clean_parts = [part for part in parts if part.lower() not in {"--forensics", "--include-quarantined", "--quarantined"}]
        target = clean_parts[0] if clean_parts and clean_parts[0].lower() not in {"current", "show"} else None
        ref = None
        if clean_parts and clean_parts[0].lower() in {"proof", "export"}:
            target = clean_parts[1] if len(clean_parts) > 1 and clean_parts[1].lower() != "current" else None
            ref = clean_parts[2] if len(clean_parts) > 2 else None
        _print_json(session.trust_report(target, ref=ref, include_quarantined=include_quarantined))
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
                "thread_id": session.thread_id,
                "tool_policy": session.tool_policy,
            }
        )
        return True
    if name == "apply":
        if rest.lower() not in {"last", "last --check", "--check last"}:
            print("Usage: /apply last")
            return True
        if not session.last_patch:
            print("[helix] no patch proposal is available from the last task.")
            return True
        check = subprocess.run(  # noqa: S603 - fixed argv, patch is stdin, shell disabled
            ["git", "-C", str(session.task_root), "apply", "--check", "-"],
            input=session.last_patch,
            text=True,
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
            capture_output=True,
            check=False,
        )
        if applied.returncode == 0:
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
        goal = rest or input("Agent goal: ").strip()
        if console:
            result = _run_with_status(
                console,
                lambda: session.task(goal, mode_override=mode_override, agent_blueprint=agent_blueprint),
                phase="task",
            )
        else:
            result = session.task(goal, mode_override=mode_override, agent_blueprint=agent_blueprint)
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
    routed = _route_natural_language(prompt)
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
    console = Console(theme=_rich_theme(theme_name)) if _HAS_UI and Console else None
    active_theme_name = theme_name

    if console:
        _play_boot_handshake(console)
        _render_boot_banner(console)
    else:
        print("HeliX interactive. Type /help for commands, /exit to quit.")

    default_provider = (
        getattr(args, "provider", None)
        or config.get("default_provider")
        or ("deepinfra" if not os.environ.get("OLLAMA_HOST") else "ollama")
    )
    if default_provider not in PROVIDERS:
        default_provider = "deepinfra"
    provider_name, pending_line = _choose_provider(default_provider)
    default_model = (
        getattr(args, "model", None)
        or config.get("default_model")
        or ("auto" if provider_name == "deepinfra" else PROVIDERS[provider_name].default_model)
    )
    model = _read_default("Model", default_model) if default_model else input("Model: ").strip()
    _ensure_provider_token(provider_name)
    if provider_name != "gemini":
        config = _maybe_prompt_optional_provider_token("gemini", config=config)
    workspace = Path(getattr(args, "workspace_root", None) or _default_workspace_root()).resolve()
    task_root = Path(getattr(args, "task_root", None) or Path.cwd()).resolve()
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

    if console:
        _render_session_ribbon(console, session)
    else:
        print(f"[helix] thread={session.run_id}")
        print(f"[helix] transcript={session.jsonl_path}")
        print(f"[helix] evidence={session.evidence_root}")
        print(f"[helix] task_root={session.task_root}")
        print(f"[helix] interaction_mode={session.interaction_mode}")

    session.record(
        role="system",
        content="Interactive HeliX session started.",
        event_type="session_start",
        metadata={
            "provider": provider_name,
            "model": model,
            "router_policy": session.router_policy,
            "interaction_mode": session.interaction_mode,
            "evidence_root": str(session.evidence_root),
            "task_root": str(session.task_root),
        },
    )

    if _HAS_UI:
        completer = WordCompleter([
            '/help', '/status', '/thread', '/provider', '/model', '/models', '/route', '/web', '/file',
            '/router', '/key', '/doctor', '/providers', '/cert', '/cert-dry', 
            '/evidence', '/verify', '/suites', '/suite', '/memory', '/trust', '/task', '/tools', '/agents', '/mode', '/tech', '/explore', '/apply', '/agent', '/with', '/theme', '/style', '/raw', '/clear', '/config', '/exit', '/quit',
            '/provider deepinfra', '/provider gemini', '/provider list',
            '/models json', '/model auto', '/model use ', '/model sonnet', '/model mistral', '/model devstral', '/model qwen', '/model qwen-big', '/model gemma', '/model gemini-pro', '/model gemini-pro-tools', '/model gemini-flash', '/model gemini-lite', '/model gemini-2.5-pro', '/model gemini-2.5-flash', '/model gemini-2.5-flash-lite', '/model coder', '/model engineering', '/model deep-reasoning', '/model llama', '/model llama-vision',
            '/with sonnet ', '/with qwen-big ', '/with gemma ', '/with gemini-pro ', '/with gemini-pro-tools ', '/with gemini-flash ', '/with gemini-lite ', '/with gemini-2.5-pro ', '/with gemini-2.5-flash ', '/with gemini-2.5-flash-lite ', '/with coder ', '/with mistral ',
            '/router balanced', '/router qwen-heavy', '/router current', '/router qwen-gemma-mistral', '/router cheap', '/router premium', '/router list', '/router why ',
            '/web ', '/file ',
            '/theme industrial-brutalist', '/theme industrial-neon', '/theme xerox', '/theme brown-console', '/theme brown', '/theme cyberpunk', '/theme cyberpunk-gray', '/theme list', '/raw on', '/raw off',
            '/mode balanced', '/mode technical', '/mode explore', '/mode list', '/tech ', '/explore ',
            '/style balanced', '/style technical', '/style forensic', '/style vivid', '/style terse', '/style list',
            '/key save', '/key save gemini', '/key gemini', '/key forget', '/key forget gemini', '/key status', '/key status gemini',
            '/evidence refresh', '/evidence latest', '/evidence search', '/verify latest', '/verify search',
            '/memory resolve ', '/memory show ', '/memory hash ', '/trust', '/trust current', '/trust --forensics', '/trust proof current ',
            '/suites json', '/suite list', '/suite latest ', '/suite show ', '/suite transcripts ', '/suite search ', '/suite read ', '/suite ingest ',
            '/thread new', '/thread list', '/thread open', '/thread close', '/thread current',
            '/task ', '/agent suggest ', '/agent use repo-scout ', '/agent use patch-planner ', '/agent use suite-run-analyst ', '/agent use transcript-forensics ', '/agent use evidence-auditor ', '/agent auto-edit ',
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
        routed = line if line.startswith("/") else _route_natural_language(line)
        if routed and routed.startswith("/"):
            if not _handle_interactive_command(session, routed):
                break
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
    _print_json(doctor_report(probe_local=args.probe_local))
    return 0


def _cmd_providers_list(args: argparse.Namespace) -> int:
    _print_json({"providers": provider_report(probe_local=args.probe_local)})
    return 0


def _cmd_models_list(args: argparse.Namespace) -> int:
    _print_json(models_payload())
    return 0


def _cmd_route(args: argparse.Namespace) -> int:
    _print_json(route_model_for_task(args.prompt, provider_name=args.provider, policy=args.policy))
    return 0


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
            task_root=Path(args.task_root or Path.cwd()).resolve(),
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
