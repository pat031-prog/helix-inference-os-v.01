from pathlib import Path

import pytest

from helix_proto.text import decode_tokens, encode_text, render_messages_prompt, save_toy_tokenizer
from helix_proto.tools import build_runtime_tool_registry


def test_render_messages_prompt_adds_assistant_prefix() -> None:
    prompt = render_messages_prompt(
        [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Say hi."},
        ]
    )

    assert prompt == "system: You are concise.\nuser: Say hi.\nassistant:"


def test_toy_tokenizer_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")

    save_toy_tokenizer(tmp_path / "toy-tokenizer", vocab_size=32)
    token_ids = encode_text(tmp_path / "toy-tokenizer", "tok3 tok7")

    assert token_ids == [6, 10]
    assert decode_tokens(tmp_path / "toy-tokenizer", token_ids) == "tok3 tok7"


def test_tool_registry_dispatches_to_runtime() -> None:
    class FakeRuntime:
        def list_models(self):  # noqa: ANN202
            return [{"alias": "tiny"}]

        def model_info(self, alias):  # noqa: ANN001,ANN202
            return {"alias": alias, "model_type": "gpt2"}

        def generate_text(self, **kwargs):  # noqa: ANN003,ANN202
            return {"mode": "generate", **kwargs}

        def resume_text(self, **kwargs):  # noqa: ANN003,ANN202
            return {"mode": "resume", **kwargs}

    registry = build_runtime_tool_registry(FakeRuntime())

    manifest = registry.manifest()
    assert {item["name"] for item in manifest} == {
        "gpt.generate_text",
        "gpt.resume_text",
        "workspace.list_models",
        "workspace.model_info",
    }

    result = registry.call("workspace.model_info", {"alias": "tiny"})
    assert result["result"]["alias"] == "tiny"

    generated = registry.call(
        "gpt.generate_text",
        {"alias": "tiny", "prompt": "hello", "max_new_tokens": 2},
    )
    assert generated["result"]["mode"] == "generate"

    with pytest.raises(ValueError, match="missing tool arguments"):
        registry.call("workspace.model_info", {})
