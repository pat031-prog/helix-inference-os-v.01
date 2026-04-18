from __future__ import annotations

from helix_kv.config import KVConfig


def build_asymmetric_config(
    *,
    key_mode: str = "turbo-int8-hadamard",
    value_mode: str = "turbo-4bit",
    hot_window: int = 0,
    topk: int = 0,
    index_refresh_interval: int = 8,
    block_size: int = 0,
    layer_share_stride: int = 0,
    calibration_tokens: int = 128,
) -> KVConfig:
    return KVConfig(
        mode="turbo-int8-hadamard",
        key_mode=key_mode,
        value_mode=value_mode,
        hot_window=hot_window,
        topk=topk,
        index_refresh_interval=index_refresh_interval,
        block_size=block_size,
        layer_share_stride=layer_share_stride,
        calibration_tokens=calibration_tokens,
    )
