from pathlib import Path

from helix_proto.assistants import (
    assistant_registry_path,
    build_assistant_messages,
    configure_assistant,
    list_assistants,
    resolve_assistant,
)
from helix_proto.workspace import model_workspace, save_model_info


def _save_model(root: Path, alias: str) -> None:
    model_dir = model_workspace(alias, root)
    save_model_info(
        model_dir,
        {
            "alias": alias,
            "alias_slug": alias.lower().replace(" ", "-"),
            "model_ref": alias,
            "model_dir": str(model_dir),
            "export_dir": str(model_dir / "export"),
        },
    )


def test_assistants_default_to_available_models(tmp_path: Path) -> None:
    _save_model(tmp_path, "tiny-gpt2")

    assistants = list_assistants(tmp_path)
    assistant_ids = {item["assistant_id"] for item in assistants}

    assert assistant_ids == {"general", "code", "legal"}
    assert all(item["available"] is True for item in assistants)
    assert all(item["alias"] == "tiny-gpt2" for item in assistants)


def test_configured_assistant_alias_is_persisted(tmp_path: Path) -> None:
    _save_model(tmp_path, "tiny-gpt2")
    _save_model(tmp_path, "coder-local")

    configured = configure_assistant("code", root=tmp_path, alias="coder-local")
    reloaded = resolve_assistant("code", tmp_path)

    assert configured["alias"] == "coder-local"
    assert reloaded["alias"] == "coder-local"
    assert assistant_registry_path(tmp_path).exists()


def test_build_assistant_messages_injects_system_prompt() -> None:
    assistant = {
        "assistant_id": "general",
        "system_prompt": "You are helpful.",
    }

    prompt, messages = build_assistant_messages(assistant, prompt="Hello")
    assert prompt is None
    assert messages == [
        {
            "role": "system",
            "content": "You are helpful. Respondé solo con la respuesta final al usuario. "
            "No muestres razonamiento interno ni thinking. /no_think",
        },
        {"role": "user", "content": "Hello\n\n/no_think"},
    ]

    prompt, messages = build_assistant_messages(
        assistant,
        messages=[{"role": "user", "content": "Hola"}],
    )
    assert prompt is None
    assert messages[0]["role"] == "system"
    assert "/no_think" in messages[0]["content"]
    assert messages[1]["content"] == "Hola"

    prompt, messages = build_assistant_messages({}, prompt="Che")
    assert messages is None
    assert prompt == "Che\n\n/no_think"
