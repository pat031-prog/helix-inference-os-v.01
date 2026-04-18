import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

import helix_kv.transformers_cache as transformers_cache
from helix_kv import (
    AdaptiveKVPolicy,
    CompressedKVCache,
    KVConfig,
    TransformersCompressedKVCache,
    build_adaptive_config,
    build_asymmetric_config,
    build_transformers_variant_set,
    run_adaptive_policy_benchmark,
    run_transformers_kv_benchmark,
)
from helix_kv.benchmark import build_gpu_transformers_variants
from helix_kv.torch_quant import Torch4BitKVArray, Torch4BitQuantizer, TorchRotation
from helix_kv.session import load_cache
from helix_proto.cli import _cmd_build_tiny_gpt2
from helix_proto.hf import _build_kv_rotation, _compute_lloyd_max_codebook, export_huggingface_model


def _build_export(tmp_path: Path) -> Path:
    model_dir = tmp_path / "tiny-gpt2"
    export_dir = tmp_path / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=64,
            hidden_size=32,
            max_position_embeddings=32,
            num_layers=2,
            num_heads=4,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=8, local_files_only=True)
    return export_dir


def test_kv_config_builders_map_to_engine_kwargs() -> None:
    adaptive = build_adaptive_config(calibration_tokens=32, hot_window=4, topk=8)
    adaptive_kwargs = adaptive.to_engine_kwargs()

    assert adaptive_kwargs["kv_cache_precision"] == "adaptive"
    assert adaptive_kwargs["kv_hot_window"] == 4
    assert adaptive_kwargs["kv_topk"] == 8
    assert adaptive_kwargs["kv_calibration_tokens"] == 32

    asymmetric = build_asymmetric_config(key_mode="turbo-4bit", value_mode="turbo-int8-hadamard", hot_window=2)
    asymmetric_kwargs = asymmetric.to_engine_kwargs()

    assert asymmetric_kwargs["kv_cache_precision"] == "turbo-int8"
    assert asymmetric_kwargs["kv_key_precision"] == "turbo-4bit"
    assert asymmetric_kwargs["kv_value_precision"] == "turbo-int8"
    assert asymmetric_kwargs["kv_hot_window"] == 2


def test_compressed_kv_cache_switch_mode_preserves_hot_window_and_resume_matches_stable_mode(tmp_path: Path) -> None:
    export_dir = _build_export(tmp_path)
    prompt_ids = [3, 5, 8, 13]

    baseline = CompressedKVCache(export_dir, KVConfig(mode="turbo-int8-hadamard", hot_window=2))
    baseline_run = baseline.generate(prompt_ids, max_new_tokens=4)

    switched = CompressedKVCache(export_dir, KVConfig(mode="turbo-4bit", hot_window=2))
    switched.generate(prompt_ids, max_new_tokens=2)
    before_hot = np.asarray(switched.engine.caches[0]["k"].hot, dtype=np.float32).copy()

    current_mode = switched.switch_mode("turbo-int8-hadamard", reason="test-upgrade")
    after_hot = np.asarray(switched.engine.caches[0]["k"].hot, dtype=np.float32).copy()

    assert current_mode == "turbo-int8-hadamard"
    assert np.allclose(before_hot, after_hot)

    session_dir = tmp_path / "switch-session"
    switched.save(session_dir)
    resumed = CompressedKVCache.load(session_dir)
    resumed_run = resumed.resume(session_dir, max_new_tokens=2)

    assert resumed_run["generated_ids"] == baseline_run["generated_ids"]
    assert resumed.current_mode == "turbo-int8-hadamard"


def test_load_cache_restores_policy_state_without_explicit_export_dir(tmp_path: Path) -> None:
    export_dir = _build_export(tmp_path)
    prompt_ids = [2, 3, 5, 8]

    cache = CompressedKVCache(export_dir, KVConfig(mode="turbo-int8-hadamard", hot_window=2))
    cache.set_policy(
        AdaptiveKVPolicy(),
        phase="plan",
        allowed_modes=("turbo-4bit", "turbo-int8-hadamard", "fp32"),
    )
    run = cache.generate(prompt_ids, max_new_tokens=3)

    session_dir = tmp_path / "policy-session"
    cache.save(session_dir)

    restored = load_cache(session_dir)

    assert restored.last_result is not None
    assert restored.last_result["generated_ids"] == run["generated_ids"]
    assert restored.engine._kv_policy is not None
    assert restored.engine._kv_policy_phase == "plan"
    assert restored.engine._kv_policy_allowed_modes == ("turbo-4bit", "turbo-int8-hadamard", "fp32")
    assert restored.current_mode == "turbo-int8-hadamard"


def test_adaptive_policy_benchmark_reports_switch_trace_and_session_sizes(tmp_path: Path) -> None:
    export_dir = _build_export(tmp_path)

    report = run_adaptive_policy_benchmark(
        export_dir,
        prompt_ids=[3, 5, 8, 13, 21, 34],
        max_new_tokens=4,
        hot_window=2,
        session_root=tmp_path / "policy-benchmark-sessions",
    )

    assert report["baseline_variant"] == "static-fp32"
    assert set(report["variants"]) == {
        "static-fp32",
        "static-turbo-int8-hadamard",
        "static-turbo-4bit",
        "adaptive-policy",
    }
    adaptive = report["variants"]["adaptive-policy"]
    assert adaptive["session_total_bytes"] > 0
    assert isinstance(adaptive["kv_mode_trace"], list)
    assert "switch_count" in adaptive
    assert "mode_histogram" in adaptive


def test_transformers_kv_benchmark_smoke_runs_on_local_tiny_gpt2(tmp_path: Path) -> None:
    model_dir = tmp_path / "tiny-gpt2-hf"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=64,
            hidden_size=32,
            max_position_embeddings=64,
            num_layers=2,
            num_heads=4,
        )
    )

    report = run_transformers_kv_benchmark(
        model_dir,
        prompt_ids=[3, 5, 8, 13],
        max_new_tokens=2,
        kv_calibration_tokens=4,
        local_files_only=True,
        device="cpu",
    )

    assert report["model_ref"] == str(model_dir)
    assert report["supports_selective_attention_acceleration"] is False
    assert report["variant_order"] == ["native-dense", "turbo-int8-hadamard", "turbo-int8k-4bitv", "adaptive-m9-h20"]
    assert report["rows"][0]["name"] == "native-dense"
    assert report["rows"][0]["prompt_perplexity_delta_pct_vs_native"] == 0.0
    assert report["rows"][0]["session_total_bytes"] > 0
    assert report["rows"][0]["gpu_peak_memory_mb"] is None
    assert report["rows"][0]["native_element_size_bytes"] > 0
    assert report["num_attention_heads"] == 4
    assert report["num_key_value_heads"] == 4
    assert report["gqa_group_size"] == 1.0
    assert report["warmup_prompt_length"] == 4
    assert report["input_adapter"] == "tokenizer-causal"
    assert report["processor_used"] is False
    assert report["chat_template_used"] is False
    assert report["is_hxq_compressed"] is False
    assert report["weight_compression_method"] == "none"
    assert report["variants"]["native-dense"]["generated_match_vs_baseline"] is True
    assert report["variants"]["turbo-int8-hadamard"]["kv_cache_ratio_vs_native"] > 1.0
    assert report["variants"]["turbo-int8-hadamard"]["session_total_bytes"] > 0
    assert report["variants"]["turbo-int8-hadamard"]["session_save_time_ms"] >= 0.0
    assert report["variants"]["turbo-int8-hadamard"]["session_load_time_ms"] >= 0.0
    assert report["variants"]["adaptive-m9-h20"]["layer_mode_counts"] is not None
    assert report["variants"]["adaptive-m9-h20"]["kv_norm_ratio_per_layer"] is not None
    assert report["variants"]["adaptive-m9-h20"]["protected_layer_indices"] == [0, 1]
    assert report["rows"][1]["total_inference_footprint_bytes"] is not None


def test_build_gpu_transformers_variants_uses_requested_adaptive_thresholds() -> None:
    variants = build_gpu_transformers_variants(
        kv_quant_seed=11,
        kv_hot_window=6,
        kv_calibration_tokens=96,
        kv_adaptive_medium_kurtosis=9.0,
        kv_adaptive_high_kurtosis=20.0,
    )

    assert [variant["name"] for variant in variants] == [
        "native-dense",
        "turbo-int8-hadamard",
        "turbo-int8k-4bitv",
        "adaptive-m9-h20",
    ]
    adaptive = variants[-1]
    assert adaptive["kv_hot_window"] == 6
    assert adaptive["kv_quant_seed"] == 11
    assert adaptive["kv_calibration_tokens"] == 96
    assert adaptive["kv_adaptive_medium_kurtosis"] == 9.0
    assert adaptive["kv_adaptive_high_kurtosis"] == 20.0


def test_build_transformers_variant_set_exposes_asymmetry_sweep() -> None:
    variants = build_transformers_variant_set(
        "asymmetry-sweep",
        kv_quant_seed=7,
        kv_hot_window=4,
        kv_calibration_tokens=32,
        kv_adaptive_medium_kurtosis=9.0,
        kv_adaptive_high_kurtosis=20.0,
    )

    assert [variant["name"] for variant in variants] == [
        "turbo-int8k-4bitv",
        "turbo-4bitk-int8v",
        "turbo-int8k-4bitv-perchannel",
        "turbo-4bit-perchannel",
        "adaptive-asymmetric-m9-h20",
    ]
    assert variants[2]["kv_key_scaling_strategy"] == "per-channel"
    assert variants[2]["kv_value_scaling_strategy"] == "per-token"
    assert variants[-1]["kv_cache_precision"] == "adaptive-asymmetric"


def test_build_transformers_variant_set_exposes_community_variants() -> None:
    variants = build_transformers_variant_set(
        "community",
        kv_quant_seed=7,
        kv_hot_window=4,
        kv_calibration_tokens=32,
        kv_adaptive_medium_kurtosis=9.0,
        kv_adaptive_high_kurtosis=20.0,
    )

    assert [variant["name"] for variant in variants] == [
        "native-dense",
        "turbo-int8-hadamard",
        "turbo-int8k-4bitv",
        "turbo-int8k-4bitv-online",
        "helix-optimal",
    ]
    assert variants[2]["kv_value_fourbit_max_iter"] == 0
    assert variants[3]["kv_value_fourbit_max_iter"] == 5
    assert variants[4]["kv_sparse_v_threshold"] == 1e-4


def test_transformers_compressed_kv_cache_exposes_mode_and_kurtosis_profile() -> None:
    class DummyTextConfig:
        num_hidden_layers = 3

    class DummyConfig:
        def get_text_config(self, decoder: bool = True) -> DummyTextConfig:  # noqa: ARG002
            return DummyTextConfig()

    cache = TransformersCompressedKVCache(
        DummyConfig(),
        kv_cache_precision="adaptive",
        kv_rotation_mode="hadamard",
        kv_hot_window=4,
    )

    assert cache.current_kv_mode == "adaptive"
    assert cache.kv_cache_bytes == 0
    assert cache.kv_kurtosis_profile is None
    assert cache.protected_layer_indices == [0, 1, 2]
    assert cache.layer_mode_counts == {"native-dense": 0, "turbo-int8": 0, "turbo-4bit": 0}
    assert cache.layer_kv_mode_counts is None


def test_transformers_compressed_layer_get_mask_sizes_accepts_scalar_cache_position() -> None:
    class DummyTextConfig:
        num_hidden_layers = 2

    class DummyConfig:
        def get_text_config(self, decoder: bool = True) -> DummyTextConfig:  # noqa: ARG002
            return DummyTextConfig()

    cache = TransformersCompressedKVCache(DummyConfig(), kv_cache_precision="native-dense")
    layer = cache._build_layer(0)
    layer.seq_length = 7

    assert layer.get_mask_sizes(3) == (8, 0)


def test_transformers_torch_backend_matches_numpy_backend_for_int8_generation(tmp_path: Path) -> None:
    model_dir = tmp_path / "tiny-gpt2-hf-parity"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=64,
            hidden_size=32,
            max_position_embeddings=64,
            num_layers=2,
            num_heads=4,
        )
    )

    numpy_report = run_transformers_kv_benchmark(
        model_dir,
        prompt_ids=[3, 5, 8, 13, 21, 34],
        max_new_tokens=4,
        kv_variants=[{"name": "turbo-int8-hadamard", "kv_cache_precision": "turbo-int8", "kv_rotation_mode": "hadamard"}],
        local_files_only=True,
        device="cpu",
        kv_backend="numpy",
    )
    torch_report = run_transformers_kv_benchmark(
        model_dir,
        prompt_ids=[3, 5, 8, 13, 21, 34],
        max_new_tokens=4,
        kv_variants=[{"name": "turbo-int8-hadamard", "kv_cache_precision": "turbo-int8", "kv_rotation_mode": "hadamard"}],
        local_files_only=True,
        device="cpu",
        kv_backend="torch",
    )

    assert numpy_report["variants"]["turbo-int8-hadamard"]["generated_ids"] == torch_report["variants"]["turbo-int8-hadamard"]["generated_ids"]


def _torch_rotation(head_dim: int = 8) -> TorchRotation:
    legacy = _build_kv_rotation(head_dim, 7, "hadamard")
    return TorchRotation.from_legacy(legacy, device=torch.device("cpu"))


def test_key_fourbit_per_channel_beats_per_token_on_fixed_channel_outliers() -> None:
    torch.manual_seed(7)
    rotation = TorchRotation(
        mode="qr",
        original_dim=8,
        rotated_dim=8,
        device=torch.device("cpu"),
        matrix=torch.eye(8, dtype=torch.float32),
    )
    values = torch.randn(4, 32, 8, dtype=torch.float32) * 0.05
    values[:, :, 0] *= 18.0
    centroids = torch.tensor(np.asarray(_compute_lloyd_max_codebook(rotation.rotated_dim, 4).centroids, dtype=np.float32))

    per_channel = Torch4BitQuantizer.from_calibration(
        values,
        rotation=rotation,
        initial_centroids=centroids,
        scaling_strategy="per-channel",
        max_iter=5,
    )
    per_token = Torch4BitQuantizer.from_calibration(
        values,
        rotation=rotation,
        initial_centroids=centroids,
        scaling_strategy="per-token",
        max_iter=5,
    )

    channel_err = torch.mean(torch.abs(Torch4BitKVArray.from_values(values, quantizer=per_channel).to_float(dtype=torch.float32) - values))
    token_err = torch.mean(torch.abs(Torch4BitKVArray.from_values(values, quantizer=per_token).to_float(dtype=torch.float32) - values))

    assert float(channel_err) < float(token_err)


def test_value_fourbit_per_token_beats_per_channel_on_token_scale_variation() -> None:
    torch.manual_seed(11)
    values = torch.randn(4, 32, 8, dtype=torch.float32)
    values = values * torch.linspace(0.1, 3.0, 32, dtype=torch.float32).view(1, 32, 1)
    rotation = _torch_rotation(8)
    centroids = torch.tensor(np.asarray(_compute_lloyd_max_codebook(rotation.rotated_dim, 4).centroids, dtype=np.float32))

    per_channel = Torch4BitQuantizer.from_calibration(
        values,
        rotation=rotation,
        initial_centroids=centroids,
        scaling_strategy="per-channel",
        max_iter=5,
    )
    per_token = Torch4BitQuantizer.from_calibration(
        values,
        rotation=rotation,
        initial_centroids=centroids,
        scaling_strategy="per-token",
        max_iter=5,
    )

    channel_err = torch.mean(torch.abs(Torch4BitKVArray.from_values(values, quantizer=per_channel).to_float(dtype=torch.float32) - values))
    token_err = torch.mean(torch.abs(Torch4BitKVArray.from_values(values, quantizer=per_token).to_float(dtype=torch.float32) - values))

    assert float(token_err) < float(channel_err)


def test_fourbit_warm_start_refinement_beats_fixed_codebook() -> None:
    torch.manual_seed(13)
    values = torch.randn(4, 24, 8, dtype=torch.float32) * 0.2
    values[:, :, 1] *= 9.0
    rotation = _torch_rotation(8)
    centroids = torch.tensor(np.asarray(_compute_lloyd_max_codebook(rotation.rotated_dim, 4).centroids, dtype=np.float32))

    fixed = Torch4BitQuantizer.from_calibration(
        values,
        rotation=rotation,
        initial_centroids=centroids,
        scaling_strategy="per-channel",
        max_iter=0,
    )
    refined = Torch4BitQuantizer.from_calibration(
        values,
        rotation=rotation,
        initial_centroids=centroids,
        scaling_strategy="per-channel",
        max_iter=5,
    )

    fixed_err = torch.mean(torch.abs(Torch4BitKVArray.from_values(values, quantizer=fixed).to_float(dtype=torch.float32) - values))
    refined_err = torch.mean(torch.abs(Torch4BitKVArray.from_values(values, quantizer=refined).to_float(dtype=torch.float32) - values))

    assert float(refined_err) <= float(fixed_err)


def test_load_text_adapter_uses_processor_for_gemma(monkeypatch) -> None:
    marker = object()

    def _fake_from_pretrained(model_ref: str, **kwargs) -> object:
        assert model_ref == "google/gemma-3-4b-it"
        assert kwargs["local_files_only"] is False
        assert kwargs["trust_remote_code"] is False
        return marker

    monkeypatch.setattr(transformers_cache.AutoProcessor, "from_pretrained", _fake_from_pretrained)

    adapter, input_adapter, processor_used, chat_template_used = transformers_cache._load_text_adapter(
        "google/gemma-3-4b-it",
        local_files_only=False,
        trust_remote_code=False,
    )

    assert adapter is marker
    assert input_adapter == "processor-text"
    assert processor_used is True
    assert chat_template_used is True


def test_resolve_prompt_inputs_processor_path_truncates_chat_template() -> None:
    class DummyProcessor:
        name_or_path = "dummy-gemma-processor"

        def apply_chat_template(self, *args, **kwargs) -> dict[str, torch.Tensor]:
            assert kwargs["tokenize"] is True
            assert kwargs["return_dict"] is True
            assert kwargs["return_tensors"] == "pt"
            return {
                "input_ids": torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long),
                "attention_mask": torch.ones((1, 5), dtype=torch.long),
            }

    prompt_inputs, resolved_prompt_ids = transformers_cache._resolve_prompt_inputs(
        "google/gemma-4-E4B-it",
        adapter=DummyProcessor(),
        input_adapter="processor-text",
        prompt_ids=None,
        prompt_text="hello gemma",
        prompt_length=3,
        local_files_only=False,
        trust_remote_code=False,
    )

    assert resolved_prompt_ids == [10, 11, 12]
    assert prompt_inputs["input_ids"].tolist() == [[10, 11, 12]]
    assert prompt_inputs["attention_mask"].tolist() == [[1, 1, 1]]


def test_hxq_report_metadata_marks_installed_first_runtime(monkeypatch) -> None:
    class DummyConfig:
        num_attention_heads = 8
        num_key_value_heads = 4

        def to_dict(self) -> dict[str, int]:
            return {"num_attention_heads": 8, "num_key_value_heads": 4}

    class DummyModel:
        def __init__(self) -> None:
            self.config = DummyConfig()
            self._param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def to(self, device: torch.device | str) -> "DummyModel":
            self._param = torch.nn.Parameter(self._param.to(device))
            return self

        def eval(self) -> "DummyModel":
            return self

        def parameters(self):
            yield self._param

        def buffers(self):
            return []

        def get_memory_footprint(self) -> int:
            return 4096

    def _fake_load_model(*args, **kwargs) -> DummyModel:
        return DummyModel()

    def _fake_load_text_adapter(*args, **kwargs) -> tuple[None, str, bool, bool]:
        return None, "tokenizer-causal", False, False

    def _fake_run_variant(*args, **kwargs) -> dict[str, object]:
        return {
            "generated_ids": [1, 2, 3],
            "prompt_perplexity": 1.0,
            "total_time_s": 1.0,
            "avg_step_ms": 10.0,
            "tokens_per_second": 2.0,
            "kv_cache_bytes": 128,
            "session_total_bytes": 256,
            "session_meta_bytes": 64,
            "session_npz_bytes": 192,
            "session_save_time_ms": 1.0,
            "session_load_time_ms": 1.0,
            "gpu_peak_memory_mb": None,
            "last_logits": np.asarray([0.1, 0.2], dtype=np.float32),
            "current_kv_mode": "native-dense",
            "kv_kurtosis_profile": None,
            "layer_mode_counts": None,
            "layer_kv_mode_counts": None,
            "kv_norm_ratio_per_layer": None,
            "protected_layer_indices": None,
            "sparse_v_skip_ratio": None,
            "model_device": "cpu",
            "cache_device": "cpu",
            "native_kv_dtype": "float32",
            "native_element_size_bytes": 4,
        }

    monkeypatch.setattr(transformers_cache, "_load_causal_model", _fake_load_model)
    monkeypatch.setattr(transformers_cache, "_load_text_adapter", _fake_load_text_adapter)
    monkeypatch.setattr(transformers_cache, "_run_transformers_variant", _fake_run_variant)
    monkeypatch.setattr(transformers_cache, "_huggingface_cache_size_bytes", lambda model_ref: 2048)
    monkeypatch.setattr(transformers_cache, "_model_vram_bytes", lambda model: 4096)
    monkeypatch.setattr(transformers_cache, "_weight_runtime_source_for_model", lambda model_ref: "pypi")
    monkeypatch.setattr(transformers_cache, "_ensure_hxq_hf_integration_registered", lambda: None)

    report = transformers_cache.run_transformers_kv_benchmark(
        "EchoLabs33/qwen2.5-3b-instruct-helix",
        prompt_ids=[1, 2],
        max_new_tokens=1,
        kv_variants=[{"name": "native-dense", "kv_cache_precision": "native-dense"}],
        local_files_only=True,
        device="cpu",
    )

    assert report["is_hxq_compressed"] is True
    assert report["hxq_model_ref"] == "EchoLabs33/qwen2.5-3b-instruct-helix"
    assert report["weight_compression_method"] == "hxq"
    assert report["weight_runtime_source"] == "pypi"
    assert report["model_size_bytes"] == 2048
    assert report["model_vram_bytes"] == 4096
    assert report["rows"][0]["total_inference_footprint_bytes"] == 4224


def test_hxq_benchmark_registers_hf_integration_before_loading(monkeypatch) -> None:
    events: list[str] = []

    class DummyConfig:
        num_attention_heads = 8
        num_key_value_heads = 4

        def to_dict(self) -> dict[str, int]:
            return {"num_attention_heads": 8, "num_key_value_heads": 4}

    class DummyModel:
        def __init__(self) -> None:
            self.config = DummyConfig()
            self._param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def to(self, device: torch.device | str) -> "DummyModel":
            self._param = torch.nn.Parameter(self._param.to(device))
            return self

        def eval(self) -> "DummyModel":
            return self

        def parameters(self):
            yield self._param

        def buffers(self):
            return []

        def get_memory_footprint(self) -> int:
            return 4096

    def _fake_register() -> None:
        events.append("register")

    def _fake_load_model(*args, **kwargs) -> DummyModel:
        events.append("model")
        return DummyModel()

    def _fake_load_text_adapter(*args, **kwargs) -> tuple[None, str, bool, bool]:
        events.append("adapter")
        return None, "tokenizer-causal", False, False

    def _fake_run_variant(*args, **kwargs) -> dict[str, object]:
        return {
            "generated_ids": [1, 2, 3],
            "prompt_perplexity": 1.0,
            "total_time_s": 1.0,
            "avg_step_ms": 10.0,
            "tokens_per_second": 2.0,
            "kv_cache_bytes": 128,
            "session_total_bytes": 256,
            "session_meta_bytes": 64,
            "session_npz_bytes": 192,
            "session_save_time_ms": 1.0,
            "session_load_time_ms": 1.0,
            "gpu_peak_memory_mb": None,
            "last_logits": np.asarray([0.1, 0.2], dtype=np.float32),
            "current_kv_mode": "native-dense",
            "kv_kurtosis_profile": None,
            "layer_mode_counts": None,
            "layer_kv_mode_counts": None,
            "kv_norm_ratio_per_layer": None,
            "protected_layer_indices": None,
            "sparse_v_skip_ratio": None,
            "model_device": "cpu",
            "cache_device": "cpu",
            "native_kv_dtype": "float32",
            "native_element_size_bytes": 4,
        }

    monkeypatch.setattr(transformers_cache, "_ensure_hxq_hf_integration_registered", _fake_register)
    monkeypatch.setattr(transformers_cache, "_load_causal_model", _fake_load_model)
    monkeypatch.setattr(transformers_cache, "_load_text_adapter", _fake_load_text_adapter)
    monkeypatch.setattr(transformers_cache, "_run_transformers_variant", _fake_run_variant)
    monkeypatch.setattr(transformers_cache, "_huggingface_cache_size_bytes", lambda model_ref: 2048)
    monkeypatch.setattr(transformers_cache, "_model_vram_bytes", lambda model: 4096)
    monkeypatch.setattr(transformers_cache, "_weight_runtime_source_for_model", lambda model_ref: "pypi")

    transformers_cache.run_transformers_kv_benchmark(
        "EchoLabs33/qwen2.5-3b-instruct-helix",
        prompt_ids=[1, 2],
        max_new_tokens=1,
        kv_variants=[{"name": "native-dense", "kv_cache_precision": "native-dense"}],
        local_files_only=True,
        device="cpu",
    )

    assert events[:3] == ["register", "model", "adapter"]


class _DummyZamba2Config:
    model_type = "zamba2"
    num_hidden_layers = 3
    layers_block_type = ["mamba", "hybrid", "mamba"]
    hidden_size = 32
    mamba_expand = 2
    mamba_d_state = 8
    mamba_d_conv = 4
    n_mamba_heads = 4
    mamba_headdim = 4
    mamba_ngroups = 1
    num_attention_heads = 4
    num_key_value_heads = 4

    def get_text_config(self, decoder: bool = True) -> "_DummyZamba2Config":
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "num_hidden_layers": self.num_hidden_layers,
            "layers_block_type": list(self.layers_block_type),
            "hidden_size": self.hidden_size,
            "mamba_expand": self.mamba_expand,
            "mamba_d_state": self.mamba_d_state,
            "mamba_d_conv": self.mamba_d_conv,
            "n_mamba_heads": self.n_mamba_heads,
            "mamba_headdim": self.mamba_headdim,
            "mamba_ngroups": self.mamba_ngroups,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
        }


def test_hybrid_cache_roundtrip_preserves_mamba_states_and_transformer_kv(tmp_path: Path) -> None:
    config = _DummyZamba2Config()
    cache = transformers_cache.TransformersHybridKVCache(
        config,
        batch_size=1,
        dtype=torch.float32,
        device="cpu",
        kv_cache_precision="native-dense",
    )
    key_states = torch.randn(1, 4, 3, 8, dtype=torch.float32)
    value_states = torch.randn(1, 4, 3, 8, dtype=torch.float32)
    cache.update(key_states, value_states, 1)
    cache.conv_states[0].copy_(torch.randn_like(cache.conv_states[0]))
    cache.ssm_states[2].copy_(torch.randn_like(cache.ssm_states[2]))

    session_dir = tmp_path / "zamba2-hybrid-session"
    transformers_cache._save_benchmark_cache(cache, model_config=config, path=session_dir)
    restored = transformers_cache._load_benchmark_cache(session_dir, model_config=config, device="cpu")

    assert isinstance(restored, transformers_cache.TransformersHybridKVCache)
    assert restored.transformer_layers == [1]
    assert restored.get_seq_length(layer_idx=1) == 3
    assert torch.allclose(restored.layers[1].keys, key_states)
    assert torch.allclose(restored.layers[1].values, value_states)
    assert torch.allclose(restored.conv_states[0], cache.conv_states[0])
    assert torch.allclose(restored.ssm_states[2], cache.ssm_states[2])


def test_q_mamba_probe_reports_state_compression_for_hybrid_cache() -> None:
    config = _DummyZamba2Config()
    cache = transformers_cache.TransformersHybridKVCache(
        config,
        batch_size=1,
        dtype=torch.float32,
        device="cpu",
        kv_cache_precision="turbo-int8",
        kv_rotation_mode="hadamard",
    )
    key_states = torch.randn(1, 4, 5, 8, dtype=torch.float32)
    value_states = torch.randn(1, 4, 5, 8, dtype=torch.float32)
    cache.update(key_states, value_states, 1)
    for layer_idx in range(config.num_hidden_layers):
        cache.conv_states[layer_idx].copy_(torch.randn_like(cache.conv_states[layer_idx]) * 0.1)
        cache.ssm_states[layer_idx].copy_(torch.randn_like(cache.ssm_states[layer_idx]) * 0.1)

    probe = transformers_cache._probe_mamba_state_compression(cache)

    assert probe is not None
    assert probe["method"] == "q-mamba-dsq-int4-probe"
    assert probe["original_bytes"] > 0
    assert probe["compressed_bytes"] > 0
    assert probe["compression_ratio"] > 1.0
    assert len(probe["conv_layers_profile"]) == config.num_hidden_layers
    assert len(probe["ssm_layers_profile"]) == config.num_hidden_layers


def test_hybrid_cache_q_mamba_runtime_roundtrip_preserves_shapes_and_reduces_bytes() -> None:
    config = _DummyZamba2Config()
    cache = transformers_cache.TransformersHybridKVCache(
        config,
        batch_size=1,
        dtype=torch.float32,
        device="cpu",
        kv_cache_precision="native-dense",
        mamba_state_precision="q-mamba-dsq-int4",
    )
    expected_conv = {idx: torch.randn_like(cache.conv_states[idx]) * 0.1 for idx in range(config.num_hidden_layers)}
    expected_ssm = {idx: torch.randn_like(cache.ssm_states[idx]) * 0.1 for idx in range(config.num_hidden_layers)}
    for idx in range(config.num_hidden_layers):
        cache.conv_states[idx].copy_(expected_conv[idx])
        cache.ssm_states[idx].copy_(expected_ssm[idx])

    native_bytes = cache.mamba_state_bytes
    cache.compress_mamba_state_runtime()

    assert cache.mamba_state_runtime_enabled is True
    assert cache.mamba_state_runtime_bytes < native_bytes

    cache.materialize_mamba_state_runtime()

    for idx in range(config.num_hidden_layers):
        assert cache.conv_states[idx].shape == expected_conv[idx].shape
        assert cache.ssm_states[idx].shape == expected_ssm[idx].shape
        assert torch.isfinite(cache.conv_states[idx]).all()
        assert torch.isfinite(cache.ssm_states[idx]).all()
        assert torch.allclose(cache.conv_states[idx], expected_conv[idx], atol=0.25)
        assert torch.allclose(cache.ssm_states[idx], expected_ssm[idx], atol=0.25)


def test_hybrid_cache_q_mamba_runtime_promotes_outlier_blocks_to_int8_or_dense() -> None:
    config = _DummyZamba2Config()
    cache = transformers_cache.TransformersHybridKVCache(
        config,
        batch_size=1,
        dtype=torch.float32,
        device="cpu",
        kv_cache_precision="native-dense",
        mamba_state_precision="q-mamba-dsq-int4",
        mamba_state_block_size=4,
        mamba_state_clip_threshold_pct=0.0,
        mamba_state_rel_rmse_threshold=0.01,
        mamba_state_auto_promote=True,
    )
    cache.conv_states[0].copy_(torch.randn_like(cache.conv_states[0]) * 0.5)
    cache.conv_states[0].view(-1)[0] = 10_000.0
    cache.ssm_states[0].copy_(torch.randn_like(cache.ssm_states[0]) * 0.01)

    cache.compress_mamba_state_runtime()

    counts = cache.mamba_state_fallback_counts
    assert counts["int8"] + counts["dense"] >= 1


def test_hybrid_cache_q_mamba_receipts_write_hash_chained_jsonl_gz(tmp_path: Path) -> None:
    config = _DummyZamba2Config()
    receipts_path = tmp_path / "receipts.jsonl.gz"
    cache = transformers_cache.TransformersHybridKVCache(
        config,
        batch_size=1,
        dtype=torch.float32,
        device="cpu",
        kv_cache_precision="native-dense",
        mamba_state_precision="q-mamba-dsq-int4",
        mamba_receipts_enabled=True,
        mamba_receipts_path=receipts_path,
        mamba_receipt_run_id="test-run",
    )
    for idx in range(config.num_hidden_layers):
        cache.conv_states[idx].copy_(torch.randn_like(cache.conv_states[idx]) * 0.05)
        cache.ssm_states[idx].copy_(torch.randn_like(cache.ssm_states[idx]) * 0.05)

    cache.compress_mamba_state_runtime()

    assert cache.mamba_receipt_count == config.num_hidden_layers * 2
    with gzip.open(receipts_path, "rt", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    assert len(lines) == config.num_hidden_layers * 2
    assert lines[0]["run_id"] == "test-run"
    assert lines[0]["prev_hash"] == "0" * 64
    assert lines[1]["prev_hash"] == lines[0]["receipt_hash"]
    assert "fallback_precision" in lines[0]
    assert "receipt_hash" in lines[0]
    assert "block_count" in lines[0]
    assert "int4_block_count" in lines[0]
    assert "int8_block_count" in lines[0]
    assert "dense_block_count" in lines[0]
    assert "promoted_block_count" in lines[0]
    assert "max_abs_value" in lines[0]
    assert "state_norm" in lines[0]
    assert lines[0]["block_count"] >= lines[0]["int4_block_count"]


def test_session_snapshot_receipt_is_deterministic(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "session.json").write_text('{"format":"stub"}', encoding="utf-8")
    np.savez_compressed(session_dir / "kv_cache.npz", sample=np.array([1, 2, 3], dtype=np.int32))

    first = transformers_cache._session_snapshot_receipt(session_dir)
    second = transformers_cache._session_snapshot_receipt(session_dir)

    assert first == second
    assert first["session_hash"]
    assert first["session_meta_hash"]
    assert first["session_npz_hash"]
    assert first["session_total_bytes"] > 0


def test_hybrid_cache_save_load_with_q_mamba_precision_restores_materialized_states(tmp_path: Path) -> None:
    config = _DummyZamba2Config()
    cache = transformers_cache.TransformersHybridKVCache(
        config,
        batch_size=1,
        dtype=torch.float32,
        device="cpu",
        kv_cache_precision="native-dense",
        mamba_state_precision="q-mamba-dsq-int4",
        mamba_state_block_size=32,
        mamba_state_clip_threshold_pct=1.5,
    )
    for idx in range(config.num_hidden_layers):
        cache.conv_states[idx].copy_(torch.randn_like(cache.conv_states[idx]) * 0.05)
        cache.ssm_states[idx].copy_(torch.randn_like(cache.ssm_states[idx]) * 0.05)

    cache.compress_mamba_state_runtime()
    session_dir = tmp_path / "zamba2-qmamba-session"
    transformers_cache._save_benchmark_cache(cache, model_config=config, path=session_dir)
    restored = transformers_cache._load_benchmark_cache(session_dir, model_config=config, device="cpu")

    assert isinstance(restored, transformers_cache.TransformersHybridKVCache)
    assert restored.mamba_state_precision == "q-mamba-dsq-int4"
    assert restored.mamba_state_block_size == 32
    assert restored.mamba_state_clip_threshold_pct == 1.5
    for idx in range(config.num_hidden_layers):
        assert isinstance(restored.conv_states[idx], torch.Tensor)
        assert isinstance(restored.ssm_states[idx], torch.Tensor)
        assert restored.conv_states[idx].shape == cache.conv_states[idx].shape
        assert restored.ssm_states[idx].shape == cache.ssm_states[idx].shape


def test_transformers_model_diagnostics_reports_nan_logits_and_canonical_hxq_ref(monkeypatch) -> None:
    events: list[str] = []

    class DummyConfig:
        def to_dict(self) -> dict[str, str]:
            return {"model_type": "zamba2"}

    class DummyOutputs:
        def __init__(self) -> None:
            self.logits = torch.full((1, 2, 8), float("nan"), dtype=torch.float32)

    class DummyModel:
        def __init__(self) -> None:
            self.config = DummyConfig()
            self._param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def to(self, device: torch.device | str) -> "DummyModel":
            self._param = torch.nn.Parameter(self._param.to(device))
            return self

        def eval(self) -> "DummyModel":
            return self

        def parameters(self):
            yield self._param

        def forward(self, input_ids: torch.Tensor, **kwargs) -> DummyOutputs:  # noqa: ARG002
            events.append("forward")
            return DummyOutputs()

        __call__ = forward

    def _fake_register() -> None:
        events.append("register")

    monkeypatch.setattr(transformers_cache, "_ensure_hxq_hf_integration_registered", _fake_register)
    monkeypatch.setattr(transformers_cache, "_load_causal_model", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(transformers_cache, "_load_text_adapter", lambda *args, **kwargs: (None, "tokenizer-causal", False, False))

    diagnostics = transformers_cache.run_transformers_model_diagnostics(
        "EchoLabs33/zamba2-1.2b-helix",
        prompt_ids=[1, 2],
        device="cpu",
        local_files_only=True,
    )

    assert diagnostics["effective_model_ref"] == "EchoLabs33/zamba2-1.2b-hxq"
    assert diagnostics["logits_finite"] is False
    assert diagnostics["nan_count"] == 16
    assert diagnostics["inf_count"] == 0
    assert events[:2] == ["register", "forward"]


def test_transformers_benchmark_reports_adaptive_asymmetric_mode_pairs(tmp_path: Path) -> None:
    model_dir = tmp_path / "tiny-gpt2-hf-asym"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=64,
            hidden_size=32,
            max_position_embeddings=64,
            num_layers=2,
            num_heads=4,
        )
    )

    report = run_transformers_kv_benchmark(
        model_dir,
        prompt_ids=[3, 5, 8, 13],
        max_new_tokens=2,
        kv_variants=build_transformers_variant_set(
            "asymmetry-sweep",
            kv_quant_seed=7,
            kv_hot_window=2,
            kv_calibration_tokens=4,
            kv_adaptive_medium_kurtosis=9.0,
            kv_adaptive_high_kurtosis=20.0,
        ),
        local_files_only=True,
        device="cpu",
    )

    asym = report["variants"]["adaptive-asymmetric-m9-h20"]
    assert asym["layer_kv_mode_counts"] is not None
    assert "selected_k_mode" in asym["kv_kurtosis_profile"][0]
    assert "selected_v_mode" in asym["kv_kurtosis_profile"][0]


def test_transformers_compressed_cache_save_load_roundtrip_preserves_resume_logits(tmp_path: Path) -> None:
    model_dir = tmp_path / "tiny-gpt2-hf-session"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=64,
            hidden_size=32,
            max_position_embeddings=64,
            num_layers=2,
            num_heads=4,
        )
    )
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype="auto")
    model.eval()
    input_ids = torch.tensor([[3, 5, 8, 13]], dtype=torch.long)
    cache = TransformersCompressedKVCache(
        model.config,
        kv_cache_precision="adaptive-asymmetric",
        kv_key_scaling_strategy="per-channel",
        kv_value_scaling_strategy="per-token",
        kv_rotation_mode="hadamard",
        kv_hot_window=2,
        kv_calibration_tokens=4,
        kv_adaptive_medium_kurtosis=9.0,
        kv_adaptive_high_kurtosis=20.0,
        kv_backend="torch",
        kv_async_compression=False,
    )

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, past_key_values=cache, use_cache=True, return_dict=True)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        baseline_logits = model(input_ids=next_token, past_key_values=cache, use_cache=True, return_dict=True).logits[:, -1, :]

    session_dir = tmp_path / "hf-transformers-session"
    cache.save(session_dir)
    restored = TransformersCompressedKVCache.load(session_dir, model_config=model.config, device="cpu")

    with torch.inference_mode():
        restored_logits = model(input_ids=next_token, past_key_values=restored, use_cache=True, return_dict=True).logits[:, -1, :]

    assert torch.equal(torch.argmax(baseline_logits, dim=-1), torch.argmax(restored_logits, dim=-1))
    assert restored.layer_kv_mode_counts is not None
