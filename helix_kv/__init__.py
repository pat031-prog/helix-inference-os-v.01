from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AdaptiveKVPolicy",
    "build_adaptive_config",
    "build_asymmetric_config",
    "CompressedKVCache",
    "TransformersCompressedKVCache",
    "KVConfig",
    "HadamardRotation",
    "DenseOrthogonalRotation",
    "TurboInt8KVArray",
    "Turbo4BitKVArray",
    "TurboQJLKVArray",
    "HotWindowKVArray",
    "run_adaptive_policy_benchmark",
    "run_kv_landscape",
    "run_transformers_kv_benchmark",
    "build_transformers_variant_set",
    "save_cache",
    "load_cache",
]


def __getattr__(name: str) -> Any:
    if name == "KVConfig":
        return import_module("helix_kv.config").KVConfig
    if name == "AdaptiveKVPolicy":
        return import_module("helix_kv.policy").AdaptiveKVPolicy
    if name == "build_adaptive_config":
        return import_module("helix_kv.adaptive").build_adaptive_config
    if name == "build_asymmetric_config":
        return import_module("helix_kv.asymmetric").build_asymmetric_config
    if name == "CompressedKVCache":
        return import_module("helix_kv.cache").CompressedKVCache
    if name == "TransformersCompressedKVCache":
        return import_module("helix_kv.transformers_cache").TransformersCompressedKVCache
    if name in {"HadamardRotation", "DenseOrthogonalRotation"}:
        module = import_module("helix_kv.rotation")
        return getattr(module, name)
    if name in {"TurboInt8KVArray", "Turbo4BitKVArray", "TurboQJLKVArray", "HotWindowKVArray"}:
        module = import_module("helix_kv.quantizer")
        return getattr(module, name)
    if name == "run_adaptive_policy_benchmark":
        return import_module("helix_kv.benchmark").run_adaptive_policy_benchmark
    if name == "run_kv_landscape":
        return import_module("helix_kv.benchmark").run_kv_landscape
    if name == "run_transformers_kv_benchmark":
        return import_module("helix_kv.benchmark").run_transformers_kv_benchmark
    if name == "build_transformers_variant_set":
        return import_module("helix_kv.benchmark").build_transformers_variant_set
    if name in {"save_cache", "load_cache"}:
        module = import_module("helix_kv.session")
        return getattr(module, name)
    raise AttributeError(name)
