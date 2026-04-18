from __future__ import annotations

from pathlib import Path

import pytest
import torch

from helix_proto.cdna import (
    compress_huggingface_model_cdnav3,
    load_cdna_text_generation_target,
    load_generation_target,
)
from helix_proto.text import save_toy_tokenizer
from helix_proto.workspace import model_workspace, save_model_info


def test_cdna_loader_runs_tiny_opt(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    OPTConfig = transformers.OPTConfig
    OPTForCausalLM = transformers.OPTForCausalLM

    torch.manual_seed(7)
    model_dir = tmp_path / "tiny-opt"
    cdna_dir = tmp_path / "tiny-opt-cdna"

    config = OPTConfig(
        vocab_size=64,
        hidden_size=32,
        word_embed_proj_dim=32,
        ffn_dim=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
        do_layer_norm_before=True,
    )
    dense_model = OPTForCausalLM(config)
    dense_model.save_pretrained(model_dir)
    save_toy_tokenizer(model_dir, vocab_size=64)

    manifest = compress_huggingface_model_cdnav3(str(model_dir), cdna_dir, local_files_only=True)
    assert manifest["compressed_tensors"] > 0

    compressed_target = load_cdna_text_generation_target(
        str(model_dir),
        cdna_dir,
        tokenizer_dir=model_dir,
        local_files_only=True,
    )

    input_ids = torch.tensor([[2, 5, 7, 9]], dtype=torch.long)
    with torch.inference_mode():
        dense_logits = dense_model(input_ids=input_ids).logits
        compressed_logits = compressed_target.model(input_ids=input_ids).logits

    assert dense_logits.shape == compressed_logits.shape
    assert torch.isfinite(compressed_logits).all()
    max_abs_err = float((dense_logits - compressed_logits).abs().max().item())
    assert max_abs_err < 0.75


def test_load_generation_target_supports_workspace_gguf(monkeypatch, tmp_path: Path) -> None:
    gguf_path = tmp_path / "qwen35-4b-q4.gguf"
    gguf_path.write_bytes(b"GGUF")

    class FakeLlama:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs

    monkeypatch.setattr("helix_proto.cdna._require_llama_cpp", lambda: FakeLlama)

    root = tmp_path / "workspace"
    model_dir = model_workspace("qwen35-4b-q4", root)
    save_model_info(
        model_dir,
        {
            "alias": "qwen35-4b-q4",
            "alias_slug": "qwen35-4b-q4",
            "model_ref": str(gguf_path),
            "model_dir": str(model_dir),
            "workspace_root": str(root),
            "source_format": "gguf",
            "inference_backend": "llama-cpp-python",
            "gguf_path": str(gguf_path),
            "n_ctx": 8192,
            "chat_format": "chatml",
        },
    )

    target = load_generation_target("qwen35-4b-q4", workspace_root=root)

    assert target.backend == "llama-cpp-python"
    assert target.load_mode == "llama_cpp_gguf"
    assert target.workspace_info is not None
    assert target.model.kwargs["model_path"] == str(gguf_path.resolve())
    assert target.model.kwargs["n_ctx"] == 8192
