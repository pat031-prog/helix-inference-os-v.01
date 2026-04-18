import json
from pathlib import Path

from helix_proto import workspace as workspace_module
from helix_proto.workspace import (
    list_model_workspaces,
    load_model_info,
    model_session_dir,
    model_workspace,
    prepare_model_workspace,
    save_model_info,
    slugify,
)


def test_slugify_and_session_dir(tmp_path: Path) -> None:
    assert slugify("My Cool/Model") == "my-cool-model"
    session_dir = model_session_dir("My Cool/Model", "Run 01", root=tmp_path)
    assert session_dir.name == "run-01"
    assert session_dir.parent.name == "sessions"


def test_prepare_model_workspace_writes_metadata(tmp_path: Path, monkeypatch) -> None:
    def fake_export(model_ref, output_dir, **kwargs):  # noqa: ANN001
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        manifest = {
            "config": {"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]},
            "exported": [{"name": "transformer.wte.weight"}],
            "skipped": [],
        }
        (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    monkeypatch.setattr(workspace_module, "export_huggingface_model", fake_export)

    info = prepare_model_workspace(
        model_ref="acme/tiny-gpt",
        alias="Tiny GPT",
        root=tmp_path,
        block_rows=32,
    )

    model_dir = model_workspace("Tiny GPT", tmp_path)
    stored = load_model_info(model_dir)
    assert info["alias"] == "Tiny GPT"
    assert info["alias_slug"] == "tiny-gpt"
    assert info["model_type"] == "gpt2"
    assert info["exported_tensors"] == 1
    assert stored["manifest_path"].endswith("manifest.json")


def test_prepare_model_workspace_reuses_existing_export(tmp_path: Path, monkeypatch) -> None:
    model_dir = model_workspace("Reusable", tmp_path)
    export_dir = model_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    save_model_info(
        model_dir,
        {
            "alias": "Reusable",
            "alias_slug": "reusable",
            "model_ref": "cached/model",
            "model_dir": str(model_dir),
            "export_dir": str(export_dir),
        },
    )

    def should_not_run(*args, **kwargs):  # noqa: ANN002,ANN003
        raise AssertionError("export should not run when reuse is possible")

    monkeypatch.setattr(workspace_module, "export_huggingface_model", should_not_run)

    info = prepare_model_workspace(model_ref="cached/model", alias="Reusable", root=tmp_path)
    assert info["reused"] is True


def test_list_model_workspaces_includes_sessions(tmp_path: Path) -> None:
    model_dir = model_workspace("Session Model", tmp_path)
    save_model_info(
        model_dir,
        {
            "alias": "Session Model",
            "alias_slug": "session-model",
            "model_ref": "session/model",
            "model_dir": str(model_dir),
            "export_dir": str(model_dir / "export"),
        },
    )
    model_session_dir("Session Model", "beta", root=tmp_path).mkdir(parents=True, exist_ok=True)
    model_session_dir("Session Model", "alpha", root=tmp_path).mkdir(parents=True, exist_ok=True)

    models = list_model_workspaces(tmp_path)

    assert len(models) == 1
    assert models[0]["available_sessions"] == ["alpha", "beta"]


def test_prepare_model_workspace_registers_gguf_alias(tmp_path: Path) -> None:
    gguf_path = tmp_path / "qwen35-4b-q4_k_m.gguf"
    gguf_path.write_bytes(b"GGUF")

    info = prepare_model_workspace(
        model_ref=str(gguf_path),
        alias="Qwen3.5-4B Q4",
        root=tmp_path / "workspace",
    )

    assert info["alias"] == "Qwen3.5-4B Q4"
    assert info["source_format"] == "gguf"
    assert info["inference_backend"] == "llama-cpp-python"
    assert info["gguf_path"] == str(gguf_path.resolve())
    assert info["has_tokenizer"] is False
