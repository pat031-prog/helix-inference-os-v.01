from __future__ import annotations

from dataclasses import dataclass


_CANONICAL_MODES = {
    "fp32": "fp32",
    "turbo-int8": "turbo-int8-hadamard",
    "turbo-int8-hadamard": "turbo-int8-hadamard",
    "turbo-4bit": "turbo-4bit",
    "turbo-qjl": "turbo-qjl",
    "adaptive": "adaptive",
}

_ENGINE_MODES = {
    "fp32": ("fp32", "hadamard"),
    "turbo-int8-hadamard": ("turbo-int8", "hadamard"),
    "turbo-4bit": ("turbo-4bit", "hadamard"),
    "turbo-qjl": ("turbo-qjl", "hadamard"),
    "adaptive": ("adaptive", "hadamard"),
}


def canonical_mode_name(mode: str) -> str:
    try:
        return _CANONICAL_MODES[str(mode).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported KV mode: {mode}") from exc


def engine_mode_parts(mode: str) -> tuple[str, str]:
    canonical = canonical_mode_name(mode)
    return _ENGINE_MODES[canonical]


@dataclass(slots=True)
class KVConfig:
    mode: str = "fp32"
    rotation: str = "hadamard"
    key_mode: str | None = None
    value_mode: str | None = None
    hot_window: int = 0
    topk: int = 0
    index_refresh_interval: int = 8
    block_size: int = 0
    layer_share_stride: int = 0
    calibration_tokens: int = 128
    adaptive_high_kurtosis: float = 10.0
    adaptive_medium_kurtosis: float = 3.0

    def normalized_mode(self) -> str:
        return canonical_mode_name(self.mode)

    def to_engine_kwargs(self) -> dict[str, object]:
        kv_cache_precision, kv_rotation_mode = engine_mode_parts(self.mode)
        if self.key_mode is not None or self.value_mode is not None:
            key_precision = engine_mode_parts(self.key_mode or self.mode)[0] if self.key_mode else None
            value_precision = engine_mode_parts(self.value_mode or self.mode)[0] if self.value_mode else None
        else:
            key_precision = None
            value_precision = None
        return {
            "kv_cache_precision": kv_cache_precision,
            "kv_key_precision": key_precision,
            "kv_value_precision": value_precision,
            "kv_rotation_mode": kv_rotation_mode if kv_cache_precision != "fp32" else self.rotation,
            "kv_hot_window": int(self.hot_window),
            "kv_topk": int(self.topk),
            "kv_index_refresh_interval": int(self.index_refresh_interval),
            "kv_block_size": int(self.block_size),
            "kv_layer_share_stride": int(self.layer_share_stride),
            "kv_calibration_tokens": int(self.calibration_tokens),
            "kv_adaptive_high_kurtosis": float(self.adaptive_high_kurtosis),
            "kv_adaptive_medium_kurtosis": float(self.adaptive_medium_kurtosis),
        }

    @classmethod
    def from_engine_kwargs(cls, **kwargs: object) -> "KVConfig":
        mode = str(kwargs.get("kv_cache_precision", "fp32"))
        rotation = str(kwargs.get("kv_rotation_mode", "hadamard"))
        if mode == "turbo-int8" and rotation == "hadamard":
            mode = "turbo-int8-hadamard"
        return cls(
            mode=mode,
            rotation=rotation,
            key_mode=kwargs.get("kv_key_precision"),
            value_mode=kwargs.get("kv_value_precision"),
            hot_window=int(kwargs.get("kv_hot_window", 0)),
            topk=int(kwargs.get("kv_topk", 0)),
            index_refresh_interval=int(kwargs.get("kv_index_refresh_interval", 8)),
            block_size=int(kwargs.get("kv_block_size", 0)),
            layer_share_stride=int(kwargs.get("kv_layer_share_stride", 0)),
            calibration_tokens=int(kwargs.get("kv_calibration_tokens", 128)),
            adaptive_high_kurtosis=float(kwargs.get("kv_adaptive_high_kurtosis", 10.0)),
            adaptive_medium_kurtosis=float(kwargs.get("kv_adaptive_medium_kurtosis", 3.0)),
        )
