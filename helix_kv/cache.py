from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from helix_kv.config import KVConfig
from helix_kv.policy import AdaptiveKVPolicy


class CompressedKVCache:
    def __init__(
        self,
        export_dir: str | Path,
        config: KVConfig | None = None,
        *,
        cache_mode: str = "session",
        max_tensor_bytes: int = 256 * 1024,
        kv_quant_seed: int = 7,
    ) -> None:
        from helix_proto.hf import GPT2StreamingEngine

        self.export_dir = Path(export_dir)
        self.config = config or KVConfig()
        self.kv_quant_seed = int(kv_quant_seed)
        self.engine = GPT2StreamingEngine(
            self.export_dir,
            cache_mode=cache_mode,
            max_tensor_bytes=max_tensor_bytes,
            kv_quant_seed=self.kv_quant_seed,
            **self.config.to_engine_kwargs(),
        )
        self.last_result: dict[str, Any] | None = None

    @property
    def current_mode(self) -> str:
        return str(getattr(self.engine, "current_kv_mode", self.config.normalized_mode()))

    @property
    def kv_cache_bytes(self) -> int:
        return int(
            sum(
                self.engine._kv_cache_bytes(layer["k"]) + self.engine._kv_cache_bytes(layer["v"])
                for layer in self.engine.caches
            )
        )

    def set_policy(
        self,
        policy: AdaptiveKVPolicy | None,
        *,
        phase: str | None = None,
        allowed_modes: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.engine.set_kv_policy(policy, phase=phase, allowed_modes=allowed_modes)

    def switch_mode(self, new_mode: str, *, reason: str | None = None) -> str:
        result = self.engine.switch_kv_mode(new_mode, reason=reason)
        self.config = KVConfig.from_engine_kwargs(
            kv_cache_precision=self.engine.kv_cache_precision,
            kv_key_precision=self.engine.kv_key_precision,
            kv_value_precision=self.engine.kv_value_precision,
            kv_rotation_mode=self.engine.kv_rotation_mode,
            kv_hot_window=self.engine.kv_hot_window,
            kv_topk=self.engine.kv_topk,
            kv_index_refresh_interval=self.engine.kv_index_refresh_interval,
            kv_block_size=self.engine.kv_block_size,
            kv_layer_share_stride=self.engine.kv_layer_share_stride,
            kv_calibration_tokens=self.engine.kv_calibration_tokens,
            kv_adaptive_high_kurtosis=self.engine.kv_adaptive_high_kurtosis,
            kv_adaptive_medium_kurtosis=self.engine.kv_adaptive_medium_kurtosis,
        )
        return str(result)

    def generate(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
        reset_sequence: bool = True,
    ) -> dict[str, Any]:
        self.last_result = self.engine.generate_advanced(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            reset_sequence=reset_sequence,
        )
        return dict(self.last_result)

    def resume(
        self,
        session_dir: str | Path,
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        self.last_result = self.engine.resume_advanced(
            session_dir,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
        return dict(self.last_result)

    def save(self, path: str | Path) -> Path:
        if self.last_result is None:
            raise ValueError("cannot save cache before running generation")
        generated_ids = self.last_result.get("generated_ids")
        if not isinstance(generated_ids, list):
            raise ValueError("last_result is missing generated_ids")
        return self.engine.save_session(
            path,
            generated_ids=generated_ids,
            last_logits=self.last_result.get("last_logits"),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        export_dir: str | Path | None = None,
        config: KVConfig | None = None,
        cache_mode: str = "session",
        max_tensor_bytes: int = 256 * 1024,
        kv_quant_seed: int = 7,
    ) -> "CompressedKVCache":
        session_dir = Path(path)
        meta: dict[str, Any] | None = None
        resolved_export_dir = export_dir
        meta_path = session_dir / "session.json"
        if not meta_path.exists():
            raise ValueError("session.json is missing")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if resolved_export_dir is None:
            resolved_export_dir = meta.get("export_dir")
            if not resolved_export_dir:
                raise ValueError("session.json is missing export_dir")
        resolved_config = config
        if resolved_config is None:
            resolved_config = KVConfig.from_engine_kwargs(
                kv_cache_precision=meta.get("kv_cache_precision", "fp32"),
                kv_key_precision=meta.get("kv_key_precision"),
                kv_value_precision=meta.get("kv_value_precision"),
                kv_rotation_mode=meta.get("kv_rotation_mode", "hadamard"),
                kv_hot_window=meta.get("kv_hot_window", 0),
                kv_topk=meta.get("kv_topk", 0),
                kv_index_refresh_interval=meta.get("kv_index_refresh_interval", 8),
                kv_block_size=meta.get("kv_block_size", 0),
                kv_layer_share_stride=meta.get("kv_layer_share_stride", 0),
                kv_calibration_tokens=meta.get("kv_calibration_tokens", 128),
                kv_adaptive_high_kurtosis=meta.get("kv_adaptive_high_kurtosis", 10.0),
                kv_adaptive_medium_kurtosis=meta.get("kv_adaptive_medium_kurtosis", 3.0),
            )
        resolved_seed = int(meta.get("kv_quant_seed", kv_quant_seed))
        cache = cls(
            resolved_export_dir,
            config=resolved_config,
            cache_mode=cache_mode,
            max_tensor_bytes=max_tensor_bytes,
            kv_quant_seed=resolved_seed,
        )
        cache.last_result = cache.engine.load_session(session_dir)
        return cache
