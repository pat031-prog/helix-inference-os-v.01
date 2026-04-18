from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from helix_proto.workspace import list_model_workspaces, slugify, workspace_root


DEFAULT_ASSISTANTS: tuple[dict[str, Any], ...] = (
    {
        "id": "general",
        "title": "General",
        "description": "Asistente conversacional general para dudas cotidianas, explicacion clara y ayuda practica.",
        "system_prompt": (
            "Sos Helix General, un asistente conversacional en espanol rioplatense. "
            "Responde claro, util y sin vueltas. Prioriza entender la intencion, dar una respuesta concreta "
            "y sumar ejemplos breves o pasos accionables cuando aporten valor. "
            "Si el usuario pide una frase, responde con una sola frase. "
            "Si el usuario pide bullets, devolve exactamente esa cantidad y que sean cortos."
        ),
    },
    {
        "id": "code",
        "title": "Codigo",
        "description": "Asistente de software para debugging, refactors, tests y explicaciones tecnicas.",
        "system_prompt": (
            "Sos Helix Codigo, un asistente de ingenieria de software. Responde en espanol claro. "
            "Cuando convenga, devolve primero una solucion correcta y despues una explicacion breve. "
            "Marca tradeoffs, casos borde y tests utiles, y no inventes APIs ni librerias. "
            "Si el usuario pide un ejemplo corto o bullets, mantenelo compacto."
        ),
    },
    {
        "id": "legal",
        "title": "Juridico",
        "description": "Asistente de informacion juridica general para Argentina, con respuestas estructuradas y cautas.",
        "system_prompt": (
            "Sos Helix Juridico, un asistente de informacion juridica general para Argentina. "
            "Responde en espanol claro y estructurado. Explica en terminos generales, senala incertidumbre "
            "cuando exista, distingue informacion general de asesoramiento profesional y no te presentes como abogado. "
            "Si el usuario pide bullets o una version corta, respeta ese formato."
        ),
    },
)


ASSISTANT_PATTERNS: dict[str, tuple[str, ...]] = {
    "general": ("general", "chat", "assistant", "instruct", "qwen", "tiny-gpt2"),
    "code": ("coder", "code", "codex", "program", "dev"),
    "legal": ("legal", "law", "jurid", "abog", "argentina"),
}


def assistant_registry_path(root: str | Path | None = None) -> Path:
    return workspace_root(root) / "assistants.json"


def default_assistant_spec(assistant_id: str) -> dict[str, Any]:
    target = slugify(assistant_id)
    for item in DEFAULT_ASSISTANTS:
        if slugify(str(item["id"])) == target:
            return dict(item)
    raise KeyError(assistant_id)


def _load_overrides(root: str | Path | None = None) -> dict[str, Any]:
    path = assistant_registry_path(root)
    if not path.exists():
        return {"assistants": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_overrides(data: dict[str, Any], root: str | Path | None = None) -> Path:
    path = assistant_registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _pick_alias(assistant_id: str, aliases: list[str]) -> str | None:
    lowered = [(alias, alias.lower()) for alias in aliases]
    for pattern in ASSISTANT_PATTERNS.get(assistant_id, ()):
        for alias, lowered_alias in lowered:
            if pattern in lowered_alias:
                return alias
    return aliases[0] if aliases else None


def list_assistants(root: str | Path | None = None) -> list[dict[str, Any]]:
    models = list_model_workspaces(root)
    aliases = [str(item["alias"]) for item in models]
    overrides = _load_overrides(root).get("assistants", {})

    assistants: list[dict[str, Any]] = []
    for item in DEFAULT_ASSISTANTS:
        assistant_id = item["id"]
        override = dict(overrides.get(assistant_id, {}))
        assistant = dict(item)
        assistant.update(override)
        configured_alias = assistant.get("alias")
        assistant["assistant_id"] = assistant_id
        assistant["alias"] = configured_alias or _pick_alias(assistant_id, aliases)
        assistant["available"] = assistant["alias"] in aliases if assistant["alias"] else False
        assistant["configured"] = bool(configured_alias)
        assistants.append(assistant)
    return assistants


def resolve_assistant(assistant_id: str, root: str | Path | None = None) -> dict[str, Any]:
    target = slugify(assistant_id)
    for assistant in list_assistants(root):
        if slugify(str(assistant["assistant_id"])) == target:
            return assistant
    raise KeyError(assistant_id)


def configure_assistant(
    assistant_id: str,
    *,
    root: str | Path | None = None,
    alias: str | None = None,
    title: str | None = None,
    description: str | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    resolved = resolve_assistant(assistant_id, root)
    overrides = _load_overrides(root)
    items = overrides.setdefault("assistants", {})
    entry = dict(items.get(resolved["assistant_id"], {}))

    if alias is not None:
        entry["alias"] = alias
    if title is not None:
        entry["title"] = title
    if description is not None:
        entry["description"] = description
    if system_prompt is not None:
        entry["system_prompt"] = system_prompt

    items[resolved["assistant_id"]] = entry
    _save_overrides(overrides, root)
    return resolve_assistant(assistant_id, root)


def build_assistant_messages(
    assistant: dict[str, Any],
    *,
    prompt: str | None = None,
    messages: list[dict[str, str]] | None = None,
) -> tuple[str | None, list[dict[str, str]] | None]:
    system_prompt = str(assistant.get("system_prompt") or "").strip()
    no_think_suffix = (
        " Respondé solo con la respuesta final al usuario. "
        "No muestres razonamiento interno ni thinking. /no_think"
    )
    if system_prompt and "/no_think" not in system_prompt:
        system_prompt = f"{system_prompt}{no_think_suffix}"

    if messages:
        if system_prompt and not any(item.get("role") == "system" for item in messages):
            return None, [{"role": "system", "content": system_prompt}, *messages]
        return None, messages

    if prompt is None:
        raise ValueError("prompt or messages is required")

    user_prompt = prompt if "/no_think" in prompt else f"{prompt}\n\n/no_think"

    if system_prompt:
        return None, [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    return user_prompt, None
