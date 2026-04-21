from __future__ import annotations

import argparse
import fnmatch
import getpass
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
            ),
            "anthropic": ProviderSpec(
                name="anthropic",
                kind="anthropic",
                base_url="https://api.anthropic.com/v1",
                token_env="ANTHROPIC_API_KEY",
                default_model="claude-4-sonnet",
                requires_token=True,
                description="Anthropic Messages API",
            ),
            "ollama": ProviderSpec(
                name="ollama",
                kind="openai-compatible",
                base_url="http://127.0.0.1:11434/v1",
                token_env=None,
                default_model="llama3.1",
                requires_token=False,
                description="Local Ollama OpenAI-compatible endpoint",
            ),
            "llamacpp": ProviderSpec(
                name="llamacpp",
                kind="openai-compatible",
                base_url="http://127.0.0.1:8080/v1",
                token_env=None,
                default_model="local-model",
                requires_token=False,
                description="Local llama.cpp server OpenAI-compatible endpoint",
            ),
            "local": ProviderSpec(
                name="local",
                kind="helix-local",
                base_url=None,
                token_env=None,
                default_model="",
                requires_token=False,
                description="Prepared local HeliX model alias via HelixRuntime",
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
    ),
    "code": ModelProfile(
        model_id="Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo",
        role="code",
        provider="deepinfra",
        input_per_million=0.22,
        output_per_million=1.00,
        notes="Agentic coding, repository-scale code understanding, function calling, and long context.",
    ),
    "qwen-122b": ModelProfile(
        model_id="Qwen/Qwen3.5-122B-A10B",
        role="qwen-general",
        provider="deepinfra",
        input_per_million=None,
        output_per_million=None,
        notes="Large Qwen generalist for deeper synthesis, broad retrieval, and non-trivial task planning.",
    ),
    "gemma": ModelProfile(
        model_id="google/gemma-4-31B",
        role="gemma-reasoning",
        provider="deepinfra",
        input_per_million=None,
        output_per_million=None,
        notes="Gemma reasoning/general profile for careful mid-weight analysis, decomposition, and precise answers.",
    ),
    "llama-vision": ModelProfile(
        model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        role="vision",
        provider="deepinfra",
        input_per_million=None,
        output_per_million=None,
        notes="Vision-capable Llama profile for screenshots, images, OCR-like descriptions, and visual debugging.",
    ),
    "llama-70b": ModelProfile(
        model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        role="llama-general",
        provider="deepinfra",
        input_per_million=None,
        output_per_million=None,
        notes="Large Llama generalist for broad instruction following, fallback synthesis, and high-context prose.",
    ),
    "reasoning": ModelProfile(
        model_id="google/gemma-4-31B",
        role="reasoning",
        provider="deepinfra",
        input_per_million=None,
        output_per_million=None,
        notes="Balanced reasoning profile backed by Gemma for decomposition, analysis, and deliberate medium-depth work.",
    ),
    "agentic": ModelProfile(
        model_id="Qwen/Qwen3.5-122B-A10B",
        role="agentic",
        provider="deepinfra",
        input_per_million=None,
        output_per_million=None,
        notes="Balanced agentic profile backed by Qwen 122B for long tasks, search-heavy work, and broad synthesis.",
    ),
    "research": ModelProfile(
        model_id="Qwen/Qwen3.5-122B-A10B",
        role="research",
        provider="deepinfra",
        input_per_million=None,
        output_per_million=None,
        notes="Research/search-oriented Qwen 122B profile for long context synthesis and careful uncertainty handling.",
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
    ),
    "deep-reasoning": ModelProfile(
        model_id="deepseek-ai/DeepSeek-V3.2",
        role="deep-reasoning",
        provider="deepinfra",
        input_per_million=0.26,
        output_per_million=0.38,
        notes="Reasoning and agentic tool-use model with efficient long-context behavior.",
    ),
    "sonnet": ModelProfile(
        model_id="anthropic/claude-4-sonnet",
        role="sonnet",
        provider="deepinfra",
        input_per_million=None,
        output_per_million=None,
        notes="Existing HeliX premium auditor model. Used for high-stakes audit/legal/claim-boundary turns.",
    ),
}


ROUTER_BLUEPRINTS: dict[str, RouterBlueprint] = {
    "balanced": RouterBlueprint(
        name="balanced",
        description="Preferred mixed blueprint: Mistral Small for chat, Gemma for reasoning, Qwen 122B for research/agentic, Qwen Coder for code, Sonnet for audits.",
        default_alias="chat",
        chat_alias="chat",
        reasoning_alias="reasoning",
        research_alias="research",
        code_alias="code",
        agentic_alias="agentic",
        audit_alias="sonnet",
        vision_alias="llama-vision",
    ),
    "qwen-gemma-mistral": RouterBlueprint(
        name="qwen-gemma-mistral",
        description="Explicit hybrid blueprint using Mistral Small + Gemma + Qwen families as the main stack.",
        default_alias="chat",
        chat_alias="chat",
        reasoning_alias="reasoning",
        research_alias="research",
        code_alias="code",
        agentic_alias="agentic",
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
        }
        for alias, profile in sorted(DEEPINFRA_MODEL_PROFILES.items())
    ]


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


def resolve_model_alias(value: str) -> str:
    candidate = str(value or "").strip()
    lowered = candidate.lower()
    if lowered in {"auto", "router:auto"}:
        return "auto"
    if lowered in DEEPINFRA_MODEL_PROFILES:
        return DEEPINFRA_MODEL_PROFILES[lowered].model_id
    aliases = {
        "claude": "sonnet",
        "claude-sonnet": "sonnet",
        "claude sonnet": "sonnet",
        "mistral-small": "mistral",
        "mistral small": "mistral",
        "mistral": "mistral",
        "devstral": "devstral",
        "qwen": "qwen-122b",
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
        return DEEPINFRA_MODEL_PROFILES[alias].model_id
    return candidate


def _resolve_router_blueprint(policy: str | None) -> RouterBlueprint:
    return ROUTER_BLUEPRINTS.get(str(policy or "").strip().lower(), ROUTER_BLUEPRINTS["balanced"])


def route_model_for_task(
    text: str,
    *,
    provider_name: str = "deepinfra",
    policy: str = "balanced",
) -> dict[str, Any]:
    """Select a model for one turn using transparent heuristics.

    This is intentionally deterministic. The router should be auditable before
    it becomes another model call.
    """

    lowered = str(text or "").lower()
    blueprint = _resolve_router_blueprint(policy)
    policy = blueprint.name
    signals: list[str] = []
    intent = "chat"

    model_control_terms = (
        "respondeme con",
        "responde con",
        "usa claude",
        "usar claude",
        "claude sonnet",
        "sonnet",
        "mistral",
        "devstral",
        "qwen",
        "gemma",
        "llama",
        "vision",
        "cambia a",
        "cambiar a",
        "modelo de",
        "que modelo",
        "modelos",
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
    )

    code_terms = (
        "code",
        "codigo",
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
    hard_agent_terms = (
        "claude code",
        "codex",
        "workspace",
        "multi-archivo",
        "multi archivo",
        "multi-file",
        "largo plazo",
        "long horizon",
        "arquitectura",
        "architecture",
        "terminal",
        "tools",
        "tool",
        "planifica",
        "orquesta",
        "suite",
    )
    reasoning_terms = (
        "razona",
        "reason",
        "matematica",
        "prueba",
        "proof",
        "hipotesis",
        "analiza",
        "desglosa",
        "compar",
        "tradeoff",
    )
    vision_terms = (
        "imagen",
        "imagenes",
        "imagenes",
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

    if any(term in lowered for term in audit_terms):
        signals.append("audit_or_high_stakes")
    if any(term in lowered for term in hard_agent_terms):
        signals.append("agentic_or_long_horizon")
    if any(term in lowered for term in code_terms):
        signals.append("code_or_repo")
    if any(term in lowered for term in reasoning_terms):
        signals.append("reasoning")
    if any(term in lowered for term in research_terms):
        signals.append("research")
    if any(term in lowered for term in vision_terms):
        signals.append("vision")
    if any(term in lowered for term in model_control_terms):
        signals.append("model_control")
    if len(text) > 1200:
        signals.append("long_prompt")

    if "model_control" in signals:
        intent = "model_control"
    elif "vision" in signals:
        intent = "vision"
    elif "research" in signals:
        intent = "research"
    elif "code_or_repo" in signals:
        intent = "code"
    elif "audit_or_high_stakes" in signals:
        intent = "audit"
    elif "reasoning" in signals or "agentic_or_long_horizon" in signals:
        intent = "reasoning"

    if provider_name != "deepinfra":
        return {
            "provider": provider_name,
            "model": None,
            "profile": "provider-default",
            "intent": intent,
            "confidence": 0.55,
            "signals": signals,
            "policy": policy,
            "blueprint": blueprint.name,
            "blueprint_description": blueprint.description,
            "reason": "non-DeepInfra providers currently keep their configured/default model",
        }

    if "model_control" in signals and ("sonnet" in lowered or "claude" in lowered):
        alias = "sonnet"
    elif "model_control" in signals and any(term in lowered for term in ("vision", "screenshot", "image", "imagen", "foto")):
        alias = "llama-vision"
    elif "model_control" in signals and ("llama" in lowered and any(term in lowered for term in ("70b", "3.3"))):
        alias = "llama-70b"
    elif "model_control" in signals and "llama" in lowered:
        alias = "llama-70b"
    elif "model_control" in signals and "gemma" in lowered:
        alias = "gemma"
    elif "model_control" in signals and ("qwen" in lowered and any(term in lowered for term in ("coder", "code"))):
        alias = "code"
    elif "model_control" in signals and "qwen" in lowered:
        alias = "qwen-122b"
    elif "model_control" in signals and "devstral" in lowered:
        alias = "devstral"
    elif "model_control" in signals and "mistral" in lowered:
        alias = "mistral"
    else:
        if "vision" in signals:
            alias = blueprint.vision_alias
        elif "audit_or_high_stakes" in signals and ("agentic_or_long_horizon" in signals or "reasoning" in signals):
            alias = blueprint.audit_alias
        elif "audit_or_high_stakes" in signals:
            alias = blueprint.audit_alias
        elif "code_or_repo" in signals:
            alias = blueprint.code_alias
        elif "research" in signals:
            alias = blueprint.research_alias
        elif "reasoning" in signals or "agentic_or_long_horizon" in signals:
            alias = blueprint.reasoning_alias if "reasoning" in signals else blueprint.agentic_alias
        else:
            alias = blueprint.chat_alias or blueprint.default_alias

    profile = DEEPINFRA_MODEL_PROFILES[alias]
    return {
        "provider": "deepinfra",
        "model": profile.model_id,
        "profile": alias,
        "role": profile.role,
        "intent": intent,
        "confidence": 0.85 if signals else 0.45,
        "signals": signals,
        "policy": policy,
        "blueprint": blueprint.name,
        "blueprint_description": blueprint.description,
        "reason": profile.notes,
        "input_per_million": profile.input_per_million,
        "output_per_million": profile.output_per_million,
    }


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
    
    try:
        response = _post_json(
            url,
            {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
            headers=headers,
            timeout=timeout,
        )
    except Exception as exc:
        raise RuntimeError(f"API Error ({provider.name}): {exc}") from exc

    if "error" in response:
        error_msg = response["error"].get("message", str(response["error"])) if isinstance(response["error"], dict) else str(response["error"])
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


def _safe_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


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


def _agent_observation_prompt(goal: str, observations: list[dict[str, Any]], *, max_chars: int = 18000) -> str:
    payload = {
        "goal": goal,
        "observations": observations[-12:],
        "instruction": (
            "Continue the task. Request another tool with <tool_call> JSON if more facts are needed. "
            "If enough evidence is available, return only <helix_output>final answer</helix_output>."
        ),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    return _truncate_text(text, max_chars)["text"]


@contextmanager
def _cli_receipt_signing(run_id: str, event_type: str, role: str):
    previous = {
        "HELIX_RECEIPT_SIGNING_MODE": os.environ.get("HELIX_RECEIPT_SIGNING_MODE"),
        "HELIX_RECEIPT_SIGNER_ID": os.environ.get("HELIX_RECEIPT_SIGNER_ID"),
        "HELIX_RECEIPT_SIGNING_SEED": os.environ.get("HELIX_RECEIPT_SIGNING_SEED"),
    }
    os.environ["HELIX_RECEIPT_SIGNING_MODE"] = "ephemeral_preregistered"
    os.environ["HELIX_RECEIPT_SIGNER_ID"] = "helix-cli"
    os.environ["HELIX_RECEIPT_SIGNING_SEED"] = f"helix-cli:{run_id}:{event_type}:{role}:{time.time_ns()}"
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
                "evidence.latest",
                "evidence.refresh",
                "evidence.show",
                "suite.list",
                "suite.dry_run",
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
            "tool_policy": self.tool_policy,
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
            }
        finally:
            catalog.close()

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
                "model_mode": self.model,
                "profiles": model_profiles_report(),
            },
            "tombstone_boundary": (
                "The CLI records signed memories and can call HeliX fence/tombstone primitives, "
                "but this interactive shell does not yet auto-tombstone normal chat turns."
            ),
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
                        ]
                    },
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
                {"name": "suite.list", "description": "List verification suites.", "safety": "auto", "kind": "suite"},
                {"name": "suite.dry_run", "description": "Inspect a suite command without executing it.", "safety": "auto", "kind": "suite"},
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
        }

    def _chat_system(
        self,
        *,
        user_text: str,
        memory_context: dict[str, Any],
        identity_evidence: dict[str, Any] | None,
        repository_evidence_pack: dict[str, Any] | None,
        tool_manifest: list[dict[str, Any]],
    ) -> str:
        return (
            "You are HeliX interactive, a practical coding and research shell running through the unified HeliX runtime. "
            "HeliX is the deterministic orchestration, memory, routing, and evidence layer around local or cloud models; "
            "do not claim that HeliX itself is the language model. "
            f"Runtime UTC now: {_utc_now()}. Thread ID: {self.thread_id}. "
            "You may either answer directly, or request exactly one tool by emitting "
            "<tool_call>{\"tool\":\"name\",\"arguments\":{...}}</tool_call>. "
            "If no tool is needed, return only the visible answer, optionally wrapped in <helix_output>...</helix_output>. "
            "Do not invent dates, run IDs, hashes, memory IDs, node hashes, or file paths. "
            "Do not reveal chain-of-thought, scratchpads, plans, hidden reasoning, or fake tool calls. "
            f"{_preferred_language_instruction(user_text)} "
            "Certified HeliX evidence pack:\n"
            f"{json.dumps(identity_evidence or {}, ensure_ascii=False, indent=2)}\n\n"
            "Certified repository evidence pack:\n"
            f"{json.dumps(repository_evidence_pack or self.last_evidence_pack or {}, ensure_ascii=False, indent=2)}\n\n"
            "Deep Memory:\n"
            f"{memory_context.get('context') or '(empty)'}\n\n"
            "Recent terminal turns:\n"
            f"{json.dumps(self.recent_history(limit=6, exclude_latest_user=True), ensure_ascii=False, indent=2)}\n\n"
            "Available tools:\n"
            f"{json.dumps(tool_manifest, ensure_ascii=False, indent=2)}"
        )

    def _task_system(
        self,
        *,
        tool_manifest: list[dict[str, Any]],
        memory_context: dict[str, Any],
        repository_evidence_pack: dict[str, Any] | None,
    ) -> str:
        return (
            "You are HeliX Agent Shell running through the unified HeliX runtime with persistent thread memory. "
            f"Thread ID: {self.thread_id}. Task root: {self.task_root}. "
            "Use at most one tool per turn. Request tools only with <tool_call> JSON. "
            "If enough evidence is available, answer directly or inside <helix_output>...</helix_output>. "
            "You may inspect repo files, git state, HeliX evidence, and suite metadata. "
            "Do not invent file paths, hashes, test results, or patch application claims. "
            "For explicit memory-review requests, a sentence like 'voy a buscar...' is not a final answer: "
            "either call helix.search / memory.search or provide the actual summary. "
            "If code changes are needed, you may propose a unified diff in the final answer, but never claim a patch was applied automatically.\n\n"
            "Current deep memory:\n"
            f"{memory_context.get('context') or '(empty)'}\n\n"
            "Certified repository evidence pack:\n"
            f"{json.dumps(repository_evidence_pack or self.last_evidence_pack or {}, ensure_ascii=False, indent=2)}\n\n"
            "Available tools:\n"
            f"{json.dumps(tool_manifest, ensure_ascii=False, indent=2)}"
        )

    def _planner_callback_factory(
        self,
        *,
        goal: str,
        mode: str,
        selected_model: str,
        tool_manifest: list[dict[str, Any]],
        memory_context: dict[str, Any],
        identity_evidence: dict[str, Any] | None,
        repository_evidence_pack: dict[str, Any] | None,
        timeout: float | None,
    ) -> tuple[Any, list[dict[str, Any]]]:
        model_turns: list[dict[str, Any]] = []

        def _callback(state: dict[str, Any]) -> PlannerDecision:
            observations = [
                {
                    "tool": item.get("tool_name"),
                    "arguments": item.get("arguments"),
                    "result": item.get("observation"),
                }
                for item in state.get("observations", [])
            ]
            history = self.recent_history(limit=6, exclude_latest_user=True)
            if observations:
                history.append(
                    {
                        "role": "user",
                        "content": "HeliX read-only tool results:\n"
                        + _truncate_text(json.dumps(observations[-8:], ensure_ascii=False, indent=2), 12000)["text"],
                    }
                )
            prompt = _agent_observation_prompt(goal, observations) if observations else goal
            system = (
                self._task_system(
                    tool_manifest=tool_manifest,
                    memory_context=memory_context,
                    repository_evidence_pack=repository_evidence_pack,
                )
                if mode == "task"
                else self._chat_system(
                    user_text=goal,
                    memory_context=memory_context,
                    identity_evidence=identity_evidence,
                    repository_evidence_pack=repository_evidence_pack,
                    tool_manifest=tool_manifest,
                )
            )
            result = run_chat(
                provider_name=self.provider_name,
                model=selected_model,
                prompt=prompt,
                system=system,
                history=history,
                max_tokens=max(self.max_tokens, 1400 if mode == "task" else self.max_tokens),
                temperature=self.temperature,
                workspace_root=self.workspace_root,
                prompt_token=False,
                timeout=timeout,
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
                    "tool_call_count": len(calls),
                    "raw_preview": raw_text[:2000],
                    "raw_text": raw_text,
                }
            )
            if calls:
                first = calls[0]
                return PlannerDecision(
                    kind="tool",
                    thought="provider planner requested a tool",
                    tool_name=str(first.get("tool") or ""),
                    arguments=first.get("arguments") if isinstance(first.get("arguments"), dict) else {},
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

    def chat(self, user_text: str) -> dict[str, Any]:
        route = None
        selected_model = self.model
        if self.model.lower() in {"auto", "router:auto"}:
            route = route_model_for_task(
                f"{user_text}\nMode: chat",
                provider_name=self.provider_name,
                policy=self.router_policy,
            )
            selected_model = route.get("model") or PROVIDERS[self.provider_name].default_model

        repository_evidence_pack = None
        if _needs_repository_evidence(user_text, route):
            repository_evidence_pack = self.refresh_evidence(user_text, limit=8)

        context = self.memory_context(user_text)
        memory_ids = list(context.get("memory_ids") or [])
        user_event = self.record(
            role="user",
            content=user_text,
            event_type="user_turn",
            metadata={"recall_memory_ids": memory_ids, "route": route, "thread_id": self.thread_id},
        )
        excluded_memory_ids = [str((user_event.get("helix_memory") or {}).get("memory_id") or "")]
        excluded_memory_ids = [item for item in excluded_memory_ids if item]
        identity_evidence = None
        if _needs_certified_evidence(user_text, route):
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
            selected_model=selected_model,
            tool_manifest=tool_manifest,
            memory_context=context,
            identity_evidence=identity_evidence,
            repository_evidence_pack=repository_evidence_pack,
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
        if trace.get("final_planner") == "none" and planner_errors:
            text = f"Task failed: {planner_errors[-1]}"
        raw_text = str(model_turns[-1].get("raw_text") if model_turns else text)
        self.record(
            role="assistant",
            content=text,
            event_type="assistant_turn",
            metadata={
                "actual_model": model_turns[-1].get("actual_model") if model_turns else None,
                "selected_model": selected_model,
                "route": route,
                "latency_ms": model_turns[-1].get("latency_ms") if model_turns else None,
                "finish_reason": model_turns[-1].get("finish_reason") if model_turns else None,
                "usage": model_turns[-1].get("usage") if model_turns else None,
                "recall_memory_ids": memory_ids,
                "raw_model_text": raw_text,
                "visible_output_cleaned": text != raw_text,
                "reasoning_internal": "",
                "trace_path": trace.get("trace_path"),
                "thread_id": self.thread_id,
            },
        )
        return {"text": text, "raw_text": raw_text, "reasoning": "", "route": route, "trace": trace}

    def task(self, goal: str, *, max_steps: int = 5) -> dict[str, Any]:
        route = route_model_for_task(
            f"{goal}\nTask mode: inspect repo, use tools, propose patch if needed.",
            provider_name=self.provider_name,
            policy=self.router_policy,
        )
        selected_model = self.model
        if self.model.lower() in {"auto", "router:auto"}:
            selected_model = route.get("model") or PROVIDERS[self.provider_name].default_model
        repository_evidence_pack = self.refresh_evidence(goal, limit=8)
        context = self.memory_context(goal)
        memory_ids = list(context.get("memory_ids") or [])
        task_start_event = self.record(
            role="user",
            content=goal,
            event_type="task_start",
            metadata={
                "mode": self.agent_mode,
                "task_root": str(self.task_root),
                "route": route,
                "thread_id": self.thread_id,
                "recall_memory_ids": memory_ids,
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
            tool_manifest=tool_manifest,
            memory_context=context,
            identity_evidence=None,
            repository_evidence_pack=repository_evidence_pack,
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
                tool_policy=self.tool_policy,
                retrieval_scope="workspace",
                memory_exclude_ids=excluded_memory_ids,
                max_steps=max(1, max_steps),
            )
        except Exception as exc:  # noqa: BLE001
            final_text = f"Task failed: {type(exc).__name__}: {exc}"
            self.last_patch = None
            task_result = {
                "status": "error",
                "mode": self.agent_mode,
                "goal": goal,
                "task_root": str(self.task_root),
                "selected_model": selected_model,
                "route": route,
                "final": final_text,
                "tool_events": [],
                "model_turns": model_turns,
                "patch_available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            self.last_task_result = task_result
            self.record(
                role="assistant",
                content=final_text,
                event_type="task_error",
                metadata={
                    "mode": self.agent_mode,
                    "selected_model": selected_model,
                    "route": route,
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
        status = "completed"
        if trace.get("final_planner") == "none" and planner_errors:
            status = "error"
            final_text = f"Task failed: {planner_errors[-1]}"
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
            "mode": self.agent_mode,
            "goal": goal,
            "task_root": str(self.task_root),
            "selected_model": selected_model,
            "route": route,
            "final": final_text,
            "tool_events": tool_events,
            "model_turns": model_turns,
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
                "mode": self.agent_mode,
                "selected_model": selected_model,
                "route": route,
                "tool_event_count": len(tool_events),
                "patch_available": bool(patch),
                "trace_path": trace.get("trace_path"),
            },
        )
        return task_result

    def status(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "provider": self.provider_name,
            "model": self.model,
            "project": self.project,
            "agent_id": self.agent_id,
            "workspace_root": str(self.workspace_root),
            "task_root": str(self.task_root),
            "jsonl_path": str(self.jsonl_path),
            "md_path": str(self.md_path),
            "evidence_root": str(self.evidence_root),
            "event_count": len(self.events),
            "router_policy": self.router_policy,
            "theme": self.theme_name,
            "agent_mode": self.agent_mode,
            "tool_policy": self.tool_policy,
            "last_patch_available": bool(self.last_patch),
            "config_path": str(_config_path()),
            "state_path": str(self._state_path),
        }


HELP_TEXT = """Commands:
  /help                         Show this help
  /status                       Show provider, model, workspace and transcript paths
  /provider NAME                Switch provider: deepinfra, openai, anthropic, ollama, llamacpp, local, ...
  /model NAME                   Switch model; aliases include auto, mistral, qwen, gemma, coder, llama, llama-vision, sonnet
  /model list                   List model aliases and router blueprints
  /models                       List built-in DeepInfra model profiles and router blueprints
  /route TEXT                   Explain which model auto-routing would pick
  /router NAME                  Change routing blueprint/policy: balanced, current, qwen-gemma-mistral, cheap, premium
  /router list                  Open or print the routing blueprint selector
  /theme NAME                   Switch terminal theme: industrial-brutalist, industrial-neon, xerox, brown-console
  /theme list                   Open or print the theme selector/report
  /raw on|off                   Toggle raw model output after the cleaned answer
  /clear                        Clear the terminal
  /key                          Prompt for the current provider API key for this process only
  /key save                     Save current provider API key in HeliX user config
  /key forget                   Remove saved current provider API key from HeliX user config
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
  /memory QUERY                 Search unified HeliX memory for this workspace, prioritizing the active thread
  /thread new [TITLE]           Create and switch to a new persistent thread
  /thread list                  List known workspace threads
  /thread open THREAD_ID        Reopen an existing thread
  /thread close [THREAD_ID]     Close a thread without deleting its memory
  /thread current               Show the active thread
  /task GOAL                    Run the unified HeliX runner in stronger agentic mode
  /tools                        List the unified tool registry exposed to the runner
  /mode                         Show tool policy and safety mode for the active thread
  /apply last                   Apply last proposed patch after explicit confirmation
  /agent GOAL                   Alias for /task in interactive mode
  /exit                         Leave the session

Natural language defaults to chat. Repo/debug/patch requests are routed to /task; certification suite requests are routed to /cert.
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
    return lowered in {"que te hace especial", "q te hace especial"} or (
        "especial" in lowered and ("hace" in lowered or "diferente" in lowered)
    )


def _needs_certified_evidence(text: str, route: dict[str, Any] | None = None) -> bool:
    lowered = str(text or "").lower()
    if _is_identity_question(lowered):
        return True
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
    return bool(route and route.get("intent") in {"research", "audit"})


def _route_natural_language(text: str) -> str | None:
    lowered = text.lower()
    if any(term in lowered for term in ("corre", "ejecuta", "run ", "certifica")):
        suite_id = _suite_from_text(text)
        if suite_id:
            return f"/cert {suite_id}"
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
    )
    task_objects = ("repo", "archivo", "archivos", "test", "tests", "pytest", "diff", "patch", "código", "codigo")
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
        ("qwen", "qwen - large generalist for synthesis and research"),
        ("gemma", "gemma - careful reasoning and decomposition"),
        ("coder", "coder - repository and patch-heavy coding work"),
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
        _print_json(
            {
                "deepinfra_model_profiles": model_profiles_report(),
                "router_blueprints": router_blueprints_report(),
            }
        )
        return True
    if name == "route":
        if not rest:
            print("Usage: /route TEXT")
            return True
        _print_json(route_model_for_task(rest, provider_name=session.provider_name, policy=session.router_policy))
        return True
    if name == "router":
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
        if rest.lower() in {"list", "ls"}:
            if console and _HAS_UI:
                selected = _select_model(session)
                if selected:
                    session.model = resolve_model_alias(selected)
            else:
                _print_json(
                    {
                        "deepinfra_model_profiles": model_profiles_report(),
                        "router_blueprints": router_blueprints_report(),
                    }
                )
                return True
        elif rest:
            session.model = resolve_model_alias(rest)
        elif console and _HAS_UI:
            selected = _select_model(session)
            if selected:
                session.model = resolve_model_alias(selected)
        print(f"[helix] model={session.model}")
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
        if rest.lower() == "forget":
            path = _forget_config_token(session.provider_name)
            provider = PROVIDERS[session.provider_name]
            if provider.token_env:
                os.environ.pop(provider.token_env, None)
            print(f"[helix] saved token removed from config: {path}")
            return True
        if rest.lower() in {"save", "persist"}:
            provider = PROVIDERS[session.provider_name]
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
        if rest.lower() == "status":
            provider = PROVIDERS[session.provider_name]
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
        _ensure_provider_token(session.provider_name)
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
    if name == "tools":
        _print_json(session.tool_registry_report())
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
        goal = rest or input("Agent goal: ").strip()
        if console:
            result = _run_with_status(console, lambda: session.task(goal), phase="task")
        else:
            result = session.task(goal)
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

    if console:
        _render_session_ribbon(console, session)
    else:
        print(f"[helix] thread={session.run_id}")
        print(f"[helix] transcript={session.jsonl_path}")
        print(f"[helix] evidence={session.evidence_root}")
        print(f"[helix] task_root={session.task_root}")

    session.record(
        role="system",
        content="Interactive HeliX session started.",
        event_type="session_start",
        metadata={
            "provider": provider_name,
            "model": model,
            "router_policy": session.router_policy,
            "evidence_root": str(session.evidence_root),
            "task_root": str(session.task_root),
        },
    )

    if _HAS_UI:
        completer = WordCompleter([
            '/help', '/status', '/thread', '/provider', '/model', '/models', '/route', 
            '/router', '/key', '/doctor', '/providers', '/cert', '/cert-dry', 
            '/evidence', '/verify', '/memory', '/task', '/tools', '/mode', '/apply', '/agent', '/theme', '/raw', '/clear', '/config', '/exit', '/quit',
            '/model auto', '/model sonnet', '/model mistral', '/model devstral', '/model qwen', '/model gemma', '/model llama', '/model llama-vision',
            '/router balanced', '/router current', '/router qwen-gemma-mistral', '/router cheap', '/router premium', '/router list',
            '/theme industrial-brutalist', '/theme industrial-neon', '/theme xerox', '/theme brown-console', '/theme brown', '/theme cyberpunk', '/theme cyberpunk-gray', '/theme list', '/raw on', '/raw off',
            '/key save', '/key forget', '/key status',
            '/evidence refresh', '/evidence latest', '/evidence search', '/verify latest', '/verify search',
            '/thread new', '/thread list', '/thread open', '/thread close', '/thread current',
            '/task ', '/tools', '/mode', '/apply last',
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
                model_used = _short_model_name(metadata.get("actual_model", session.model))
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
    _print_json(
        {
            "deepinfra_model_profiles": model_profiles_report(),
            "router_blueprints": router_blueprints_report(),
        }
    )
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
            model=args.model or provider.default_model,
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
    selected_model = args.model
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
