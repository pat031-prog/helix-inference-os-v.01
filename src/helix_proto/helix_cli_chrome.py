from __future__ import annotations

import html
import threading
import time
from typing import Any, Callable

try:
    from rich import box
    from rich.console import Console, Group
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.table import Table
    from rich.text import Text
    from rich.theme import Theme
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.filters import is_done
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.shortcuts import choice
    from prompt_toolkit.styles import Style

    HAS_UI = True
except ImportError:
    box = None
    Console = None
    Group = None
    Live = None
    Markdown = None
    Panel = None
    Spinner = None
    Table = None
    Text = None
    Theme = None
    PromptSession = None
    WordCompleter = None
    is_done = None
    HTML = None
    choice = None
    Style = None
    HAS_UI = False


DEFAULT_THEME = "industrial-brutalist"

_CANONICAL_THEMES: dict[str, dict[str, str]] = {
    "industrial-brutalist": {
        "description": "Cold gray terminal steel with desaturated orange accents and dense brutalist framing.",
        "fg": "#ddd6cd",
        "fg_dim": "#9e978f",
        "accent": "bold #cc7a45",
        "accent_soft": "#d99563",
        "warning": "bold #e2a458",
        "error": "bold #d46452",
        "success": "#a5b58e",
        "line": "#625b54",
        "line_soft": "#403a35",
        "panel_chat": "#4f4944",
        "panel_task": "#786556",
        "panel_verify": "#8b704f",
        "toolbar_fg": "#e1d8ce",
        "toolbar_bg": "#26221f",
        "prompt": "#cc7a45 bold",
        "separator": "#d99563 bold",
    },
    "industrial-neon": {
        "description": "Industrial chassis with colder neon edges, cyan telemetry, and restrained amber signal highlights.",
        "fg": "#dde5ea",
        "fg_dim": "#8997a3",
        "accent": "bold #ffb347",
        "accent_soft": "#57d5ff",
        "warning": "bold #ffd166",
        "error": "bold #ff657a",
        "success": "#76d39b",
        "line": "#55606a",
        "line_soft": "#384049",
        "panel_chat": "#49535c",
        "panel_task": "#51606a",
        "panel_verify": "#60717e",
        "toolbar_fg": "#e4edf2",
        "toolbar_bg": "#1f252b",
        "prompt": "#ffb347 bold",
        "separator": "#57d5ff bold",
    },
    "xerox": {
        "description": "Monochrome lab mode with stark grayscale contrast and quiet framing.",
        "fg": "#f0f0f0",
        "fg_dim": "#8f8f8f",
        "accent": "bold #ffffff",
        "accent_soft": "#d8d8d8",
        "warning": "bold #ffffff",
        "error": "bold #ffffff",
        "success": "#f2f2f2",
        "line": "#9a9a9a",
        "line_soft": "#666666",
        "panel_chat": "#8a8a8a",
        "panel_task": "#787878",
        "panel_verify": "#b3b3b3",
        "toolbar_fg": "#f2f2f2",
        "toolbar_bg": "#333333",
        "prompt": "#ffffff bold",
        "separator": "#bbbbbb bold",
    },
    "brown-console": {
        "description": "Retro amber console with smoky browns and soft phosphor warmth.",
        "fg": "#f3dfc1",
        "fg_dim": "#a88963",
        "accent": "bold #e7b96b",
        "accent_soft": "#d7a95f",
        "warning": "bold #e7b96b",
        "error": "bold #ff6b35",
        "success": "#d7a95f",
        "line": "#7a4f2a",
        "line_soft": "#5b3a20",
        "panel_chat": "#6f4726",
        "panel_task": "#815a31",
        "panel_verify": "#93683a",
        "toolbar_fg": "#f3dfc1",
        "toolbar_bg": "#2f1f14",
        "prompt": "#e7b96b bold",
        "separator": "#c58b42 bold",
    },
}

_THEME_ALIASES = {
    "cyberpunk": "industrial-neon",
    "cyberpunk-gray": "industrial-neon",
    "brown": "brown-console",
}


def _with_semantic_tokens(name: str, palette: dict[str, str]) -> dict[str, str]:
    fg = palette["fg"]
    fg_dim = palette["fg_dim"]
    accent = palette["accent"]
    accent_soft = palette["accent_soft"]
    toolbar_fg = palette["toolbar_fg"]
    toolbar_bg = palette["toolbar_bg"]
    return {
        **palette,
        "theme_name": name,
        "info": fg_dim,
        "system": accent,
        "model_tag": accent_soft,
        "thought": f"italic {fg_dim}",
        "creamy": fg,
        "panel": palette["line"],
        "panel_soft": palette["line_soft"],
        "muted": fg_dim,
        "hash": accent_soft,
        "latency": accent,
        "line": palette["line"],
        "line_soft": palette["line_soft"],
        "toolbar": f"{toolbar_fg} bg:{toolbar_bg} noreverse",
        "toolbar_title": accent_soft,
        "bottom-toolbar": f"{toolbar_fg} bg:{toolbar_bg} noreverse",
        "frame.border": palette["line"],
        "frame.label": accent_soft,
        "selected-option": f"bold {fg}",
    }


THEME_PALETTES: dict[str, dict[str, str]] = {}
for canonical_name, base_palette in _CANONICAL_THEMES.items():
    materialized = _with_semantic_tokens(canonical_name, base_palette)
    THEME_PALETTES[canonical_name] = materialized
for alias, canonical_name in _THEME_ALIASES.items():
    THEME_PALETTES[alias] = THEME_PALETTES[canonical_name]


def theme_names(*, include_aliases: bool = True) -> list[str]:
    values = THEME_PALETTES.keys() if include_aliases else _CANONICAL_THEMES.keys()
    return sorted(values)


def theme_report() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, base_palette in sorted(_CANONICAL_THEMES.items()):
        aliases = sorted(alias for alias, target in _THEME_ALIASES.items() if target == name)
        rows.append(
            {
                "name": name,
                "description": base_palette["description"],
                "aliases": aliases,
                "accent": base_palette["accent"].replace("bold ", ""),
                "accent_soft": base_palette["accent_soft"],
            }
        )
    return rows


def normalize_theme_name(theme_name: str | None) -> str:
    candidate = str(theme_name or DEFAULT_THEME).strip().lower()
    if candidate in THEME_PALETTES:
        return str(THEME_PALETTES[candidate]["theme_name"])
    return DEFAULT_THEME


def theme_palette(theme_name: str | None) -> dict[str, str]:
    return THEME_PALETTES.get(str(theme_name or DEFAULT_THEME).strip().lower(), THEME_PALETTES[DEFAULT_THEME])


def rich_theme(theme_name: str | None) -> Theme:
    palette = theme_palette(theme_name)
    if Theme is None:
        raise RuntimeError("Rich theme support is unavailable")
    excluded = {
        "theme_name",
        "description",
        "toolbar_bg",
        "toolbar_fg",
        "prompt",
        "separator",
        "toolbar",
        "toolbar_title",
        "bottom-toolbar",
        "frame.border",
        "frame.label",
        "selected-option",
    }
    return Theme({key: value for key, value in palette.items() if key not in excluded})


def prompt_style(theme_name: str | None) -> Style:
    palette = theme_palette(theme_name)
    if Style is None:
        raise RuntimeError("prompt_toolkit style support is unavailable")
    return Style.from_dict(
        {
            "prompt": palette["prompt"],
            "separator": palette["separator"],
            "bottom-toolbar": palette["bottom-toolbar"],
            "frame.border": palette["frame.border"],
            "frame.label": palette["frame.label"],
            "selected-option": palette["selected-option"],
        }
    )


def panel_width(active_console: Any, *, context: str = "chat") -> int:
    width = int(getattr(active_console, "width", 100) or 100)
    max_width = {
        "chat": 118,
        "task": 124,
        "verify": 116,
        "raw": 112,
        "status": 94,
        "session": 128,
    }.get(context, 118)
    min_width = {
        "status": 62,
        "session": 74,
    }.get(context, 68)
    return max(min_width, min(width - 6, max_width))


def _box_for(kind: str) -> Any:
    if box is None:
        return None
    return {
        "boot": box.HEAVY,
        "chat": box.ROUNDED,
        "task": box.SQUARE,
        "verify": box.DOUBLE,
        "ribbon": box.MINIMAL,
        "status": box.HEAVY,
        "raw": box.MINIMAL_HEAVY_HEAD,
        "timeline": box.MINIMAL_DOUBLE_HEAD,
    }.get(kind, box.ROUNDED)


def _status_message_panel(phase: str, message: str, elapsed_ms: float) -> Any:
    phase_title = str(phase or "thinking").upper().replace("_", " ")
    if Table is None or Panel is None or Spinner is None:
        return f"> {message} ({elapsed_ms:.0f}ms)"
    inline = Table.grid(expand=True)
    inline.add_column(width=2, no_wrap=True)
    inline.add_column(ratio=1)
    inline.add_row(
        Spinner("dots", style="system"),
        Text(message, style="system"),
    )
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_row(
        inline,
        Text(f"{elapsed_ms:.0f}ms", style="latency"),
    )
    grid.add_row(f"[muted]{phase_title}[/muted]", "[hash]live[/hash]")
    return Panel.fit(
        grid,
        title=f"[toolbar_title]{phase_title}[/toolbar_title]",
        border_style="line",
        box=_box_for("status"),
        padding=(0, 1),
    )


def run_with_status(
    active_console: Any,
    func: Callable[[], Any],
    *,
    phase: str = "thinking",
    phase_messages: dict[str, list[str]] | None = None,
) -> Any:
    if not (HAS_UI and active_console and Live is not None):
        return func()

    result: dict[str, Any] = {}
    messages_map = phase_messages or {}

    def worker() -> None:
        try:
            result["value"] = func()
        except BaseException as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    started = time.perf_counter()
    index = 0
    rotated_at = started
    try:
        with Live(console=active_console, refresh_per_second=12, transient=True) as live:
            while thread.is_alive():
                now = time.perf_counter()
                messages = messages_map.get(phase) or messages_map.get("thinking") or ["pensando..."]
                if now - rotated_at >= 1.1:
                    index += 1
                    rotated_at = now
                message = messages[index % len(messages)]
                live.update(_status_message_panel(phase, message, (now - started) * 1000))
                time.sleep(1 / 12)
    except KeyboardInterrupt:
        return {"text": "[request cancelled by user]", "raw_text": "", "reasoning": "", "route": None, "cancelled": True}
    thread.join(timeout=0)
    elapsed = time.perf_counter() - started
    if elapsed < 0.25:
        time.sleep(0.25 - elapsed)
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _toolbar_markup(session: Any) -> str:
    thread_id = str(getattr(session, "thread_id", None) or getattr(session, "run_id", "") or "thread")
    provider_name = str(getattr(session, "provider_name", "provider"))
    model_name = str(getattr(session, "model", "model"))
    router_policy = str(getattr(session, "router_policy", "balanced"))
    theme_name = str(getattr(session, "theme_name", DEFAULT_THEME))
    thread_short = thread_id[:18] if len(thread_id) > 18 else thread_id
    return (
        f" <b>thread</b> {html.escape(thread_short)}"
        f"  <b>provider</b> {html.escape(provider_name)}"
        f"  <b>model</b> {html.escape(model_name)}"
        f"  <b>router</b> {html.escape(router_policy)}"
        f"  <b>theme</b> {html.escape(theme_name)}"
        "  <b>/help</b> directives "
    )


def prompt_bottom_toolbar(session: Any) -> Any:
    if HTML is None:
        return None
    return HTML(_toolbar_markup(session))


def prompt_toolbar_markup(session: Any) -> str:
    return _toolbar_markup(session)


def prompt_message(session: Any) -> list[tuple[str, str]]:
    thread_id = str(getattr(session, "thread_id", None) or getattr(session, "run_id", "") or "")
    thread_short = thread_id[:12] if thread_id else "live"
    return [
        ("class:prompt", "HeliX"),
        ("class:separator", " ["),
        ("class:prompt", thread_short),
        ("class:separator", "] > "),
    ]


def choose_option(
    *,
    title: str,
    theme_name: str | None,
    options: list[tuple[str, str]],
    bottom_help: str,
) -> str | None:
    if not (HAS_UI and choice is not None and HTML is not None and is_done is not None):
        return None
    return choice(
        message=HTML(f"<b>{html.escape(title)}</b>"),
        options=options,
        style=prompt_style(theme_name),
        show_frame=~is_done,
        bottom_toolbar=HTML(bottom_help),
    )


def play_boot_handshake(active_console: Any) -> None:
    if not (HAS_UI and active_console and Live is not None):
        return
    frames = [
        "stabilizing runtime bus...",
        "syncing Merkle receipts...",
        "arming tool lattice...",
    ]
    started = time.perf_counter()
    try:
        with Live(console=active_console, refresh_per_second=12, transient=True) as live:
            index = 0
            while time.perf_counter() - started < 0.33:
                live.update(_status_message_panel("boot", frames[index % len(frames)], (time.perf_counter() - started) * 1000))
                index += 1
                time.sleep(1 / 12)
    except Exception:
        return


def render_boot_banner(active_console: Any) -> None:
    if not (HAS_UI and active_console and Panel is not None):
        return
    active_console.print(
        Panel.fit(
            "[bold white]HeliX Inference OS[/bold white]\n[muted]v5.4.1 Cryptographic Agent Shell[/muted]\n[system]Deterministic runtime link stabilized.[/system]",
            title="[toolbar_title]BOOTSTRAP[/toolbar_title]",
            border_style="line",
            box=_box_for("boot"),
            padding=(1, 2),
        )
    )
    active_console.print("[info]Use [success]/help[/success] for directives, [warning]/exit[/warning] to terminate.[/info]")


def render_session_ribbon(active_console: Any, session: Any) -> None:
    if not (HAS_UI and active_console and Table is not None and Panel is not None):
        return
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(
        f"[muted]thread[/muted] [hash]{getattr(session, 'run_id', '')}[/hash]",
        f"[muted]provider[/muted] [creamy]{getattr(session, 'provider_name', '')}[/creamy]",
    )
    grid.add_row(
        f"[muted]model[/muted] [creamy]{getattr(session, 'model', '')}[/creamy]",
        f"[muted]router[/muted] [creamy]{getattr(session, 'router_policy', '')}[/creamy]",
    )
    grid.add_row(
        f"[muted]transcript[/muted] [creamy]{getattr(session, 'jsonl_path', '')}[/creamy]",
        f"[muted]theme[/muted] [creamy]{getattr(session, 'theme_name', DEFAULT_THEME)}[/creamy]",
    )
    grid.add_row(
        f"[muted]evidence[/muted] [creamy]{getattr(session, 'evidence_root', '')}[/creamy]",
        f"[muted]task root[/muted] [creamy]{getattr(session, 'task_root', '')}[/creamy]",
    )
    active_console.print(
        Panel(
            grid,
            title="[toolbar_title]SESSION BUS[/toolbar_title]",
            border_style="line_soft",
            box=_box_for("ribbon"),
            padding=(0, 1),
            width=panel_width(active_console, context="session"),
        )
    )


def render_chat_response(
    active_console: Any,
    *,
    clean_text: str,
    model_used: str,
    intent: str,
    latency_label: str,
    short_hash: str,
    raw_text: str = "",
    show_raw: bool = False,
) -> None:
    if not (HAS_UI and active_console and Panel is not None and Table is not None and Markdown is not None):
        return
    meta = Table.grid(expand=True)
    meta.add_column(ratio=1)
    meta.add_column(justify="right", no_wrap=True)
    meta.add_row(
        f"[model_tag]{model_used}[/model_tag] [muted]|[/muted] [creamy]{intent}[/creamy]",
        f"[latency]{latency_label}[/latency] [muted]|[/muted] [hash]{short_hash}[/hash]",
    )
    body = Group(meta, "", Markdown(clean_text))
    active_console.print()
    active_console.print(
        Panel(
            body,
            title="[toolbar_title]LIVE TURN[/toolbar_title]",
            border_style="panel_chat",
            box=_box_for("chat"),
            padding=(1, 2),
            width=panel_width(active_console, context="chat"),
        )
    )
    if show_raw and raw_text and raw_text != clean_text:
        active_console.print(
            Panel(
                raw_text,
                title="[warning]RAW MODEL OUTPUT[/warning]",
                border_style="line_soft",
                box=_box_for("raw"),
                padding=(0, 1),
                width=panel_width(active_console, context="raw"),
            )
        )
    active_console.print()


def _tool_status_markup(status: str) -> str:
    lowered = str(status or "ok").lower()
    if lowered in {"ok", "completed", "verified", "done"}:
        return f"[success]{status}[/success]"
    if lowered in {"blocked", "error", "failed", "denied"}:
        return f"[error]{status}[/error]"
    if lowered in {"warning", "unavailable", "not_found"}:
        return f"[warning]{status}[/warning]"
    return f"[muted]{status}[/muted]"


def render_task_result(
    active_console: Any,
    result: dict[str, Any],
    *,
    normalize_tool_event: Callable[[Any], dict[str, Any]],
    tool_event_detail: Callable[[Any], str],
    short_model_name: Callable[[str | None], str],
) -> None:
    if not (HAS_UI and active_console and Panel is not None and Table is not None and Markdown is not None):
        return
    final = str(result.get("final") or "").strip() or "[dim]Task completed with no final text.[/dim]"
    model = short_model_name(str(result.get("selected_model") or "model"))
    route = result.get("route") if isinstance(result.get("route"), dict) else {}
    intent = str(route.get("intent") or "task")
    patch_label = "patch ready" if result.get("patch_available") else "no patch"
    tool_events = list(result.get("tool_events") or [])

    meta = Table.grid(expand=True)
    meta.add_column(ratio=1)
    meta.add_column(justify="right", no_wrap=True)
    meta.add_row(
        f"[model_tag]{model}[/model_tag] [muted]|[/muted] [creamy]{intent}[/creamy]",
        f"[warning]{patch_label}[/warning]",
    )

    renderables: list[Any] = [meta, "", Markdown(final)]
    if tool_events:
        timeline = Table(
            title="[toolbar_title]runtime tool timeline[/toolbar_title]",
            expand=True,
            show_lines=False,
            border_style="line_soft",
            box=_box_for("timeline"),
        )
        timeline.add_column("#", style="muted", justify="right", width=3)
        timeline.add_column("tool", style="model_tag", min_width=18)
        timeline.add_column("status", justify="center", no_wrap=True)
        timeline.add_column("detail", style="creamy", ratio=1)
        for index, event in enumerate(tool_events[-12:], start=1):
            normalized = normalize_tool_event(event)
            result_payload = normalized["result"] if isinstance(normalized.get("result"), dict) else {}
            detail = tool_event_detail(event)[:80] or "-"
            timeline.add_row(
                str(index),
                normalized["tool"],
                _tool_status_markup(str(result_payload.get("status") or "ok")),
                detail,
            )
        renderables.extend(["", timeline])
    if result.get("patch_available"):
        renderables.extend(["", "[warning]Patch proposal stored. Use [success]/apply last[/success] to apply after review.[/warning]"])

    active_console.print()
    active_console.print(
        Panel(
            Group(*renderables),
            title="[toolbar_title]AGENT SHELL[/toolbar_title]",
            border_style="panel_task",
            box=_box_for("task"),
            padding=(1, 2),
            width=panel_width(active_console, context="task"),
        )
    )
    active_console.print()


def render_verify_audit(active_console: Any, report: dict[str, Any], ingested: dict[str, Any], duration_ms: float) -> None:
    if not (HAS_UI and active_console and Panel is not None and Table is not None):
        return
    status_icon = "[success][ok][/success]" if report.get("status") == "verified" else "[error][fail][/error]"
    root_hash = str(report.get("artifact_file_sha256") or report.get("artifact_sha256") or "N/A")
    short_hash = f"{root_hash[:12]}..." if len(root_hash) > 12 else root_hash

    sig_status = "verified" if report.get("signature_verified_count", 0) > 0 else "N/A / unsigned"
    if report.get("signature_failed_count", 0) > 0:
        sig_status = f"{report['signature_failed_count']} failed signatures"
    fence_status = "detected and respected" if report.get("chain_verified_count", 0) > 0 else "not present in this artifact"
    verdict = (
        "[success]Causal integrity confirmed.[/success] The artifact is mathematically valid."
        if report.get("status") == "verified"
        else "[error]Integrity failure.[/error] The artifact is manipulated or invalid."
    )

    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(
        f"{status_icon} [muted]root hash[/muted] [hash]{short_hash}[/hash]",
        f"[muted]duration[/muted] [latency]{duration_ms:.3f} ms[/latency]",
    )
    grid.add_row(
        f"[muted]signatures[/muted] [creamy]{sig_status}[/creamy]",
        f"[muted]tombstone fence[/muted] [creamy]{fence_status}[/creamy]",
    )
    grid.add_row(
        f"[muted]HeliX memory[/muted] [creamy]{ingested.get('memory_id')}[/creamy]",
        f"[muted]DAG node[/muted] [hash]{str(ingested.get('node_hash') or '')[:12]}...[/hash]",
    )
    grid.add_row(
        f"[muted]chain[/muted] [creamy]{ingested.get('chain_status')}[/creamy]",
        "",
    )

    active_console.print()
    active_console.print(
        Panel(
            Group(grid, "", f"[bold]VERDICT:[/] {verdict}"),
            title="[toolbar_title]CRYPTOGRAPHIC AUDIT[/toolbar_title]",
            border_style="panel_verify",
            box=_box_for("verify"),
            padding=(1, 2),
            width=panel_width(active_console, context="verify"),
        )
    )
    active_console.print()
