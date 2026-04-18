from __future__ import annotations

import gc
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CDNA_MANIFEST_FILE = "cdna_manifest.json"


def _cdna_safe_name(tensor_name: str) -> str:
    return tensor_name.replace("/", "_").replace(".", "_")


def _require_hf():
    try:
        import torch
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "CDNA features need optional dependencies: pip install -e '.[hf,substrate]'"
        ) from exc
    return torch, snapshot_download, safe_open, AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _require_llama_cpp():
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "GGUF inference needs optional dependencies: pip install -e '.[llama]'"
        ) from exc
    return Llama


def _require_substrate():
    try:
        from helix_substrate import CDNAv3Writer, get_policy, load_cdna_factors, swap_to_helix
    except ImportError as exc:
        raise RuntimeError(
            "CDNA substrate features need optional dependencies including torch."
        ) from exc
    return CDNAv3Writer, get_policy, load_cdna_factors, swap_to_helix


def _dir_size_bytes(path: str | Path) -> int:
    total = 0
    for item in Path(path).rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def resolve_hf_snapshot_dir(
    model_ref: str,
    *,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
) -> Path:
    del trust_remote_code
    path = Path(model_ref)
    if path.exists():
        return path.resolve()
    _, snapshot_download, _, _, _, _ = _require_hf()
    snapshot_path = snapshot_download(
        repo_id=model_ref,
        local_files_only=local_files_only,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.txt",
            "*.jinja",
            "*.model",
            "*.py",
        ],
    )
    return Path(snapshot_path).resolve()


def _load_config(
    model_ref: str,
    *,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
):
    _, _, _, AutoConfig, _, _ = _require_hf()
    return AutoConfig.from_pretrained(
        model_ref,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )


def _runtime_model_type(config: Any) -> str:
    return str(getattr(config, "model_type", "") or "")


def _text_config(config: Any) -> Any:
    text_cfg = getattr(config, "text_config", None)
    return text_cfg if text_cfg is not None else config


def _runtime_num_layers(config: Any) -> int | None:
    text_cfg = _text_config(config)
    value = getattr(text_cfg, "num_hidden_layers", None)
    return int(value) if isinstance(value, int) else None


def _checkpoint_key_to_runtime_param_name(config: Any, checkpoint_key: str) -> str | None:
    model_type = _runtime_model_type(config)
    if model_type == "qwen3_5":
        if checkpoint_key.startswith("model.language_model."):
            return "model." + checkpoint_key[len("model.language_model.") :]
        if checkpoint_key.startswith("model.visual.") or checkpoint_key.startswith("visual."):
            return None
        if checkpoint_key.startswith("mtp."):
            return None
    return checkpoint_key


def _runtime_param_name_to_checkpoint_key(config: Any, runtime_name: str) -> str | None:
    model_type = _runtime_model_type(config)
    if model_type == "qwen3_5":
        if runtime_name == "lm_head.weight":
            return None
        if runtime_name.startswith("model."):
            return "model.language_model." + runtime_name[len("model.") :]
    return runtime_name


def _should_compress_runtime_weight(runtime_param_name: str) -> bool:
    if not runtime_param_name.endswith(".weight"):
        return False
    if runtime_param_name == "model.embed_tokens.weight":
        return False
    if runtime_param_name == "lm_head.weight":
        return False
    return ".layers." in runtime_param_name


def _sample_fisher_kurtosis(array: np.ndarray, *, max_samples: int = 200_000) -> float:
    flat = np.asarray(array, dtype=np.float32).ravel()
    if flat.size == 0:
        return 0.0
    if flat.size > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(flat.size, size=max_samples, replace=False)
        flat = flat[indices]
    mean = float(np.mean(flat))
    centered = flat - mean
    variance = float(np.mean(centered * centered))
    if variance <= 1e-12:
        return 0.0
    fourth = float(np.mean(centered ** 4))
    return (fourth / (variance * variance)) - 3.0


class CheckpointReader:
    def __init__(self, model_dir: str | Path):
        _, _, safe_open, _, _, _ = _require_hf()
        self._safe_open = safe_open
        self.model_dir = Path(model_dir).resolve()
        self.weight_map = self._load_weight_map()
        self._handles: dict[str, Any] = {}

    def _load_weight_map(self) -> dict[str, str]:
        index_path = self.model_dir / "model.safetensors.index.json"
        single_path = self.model_dir / "model.safetensors"
        if index_path.exists():
            data = json.loads(index_path.read_text(encoding="utf-8"))
            return {str(key): str(value) for key, value in data.get("weight_map", {}).items()}
        if single_path.exists():
            return {}
        raise FileNotFoundError(f"no safetensors files found in {self.model_dir}")

    def _resolve_file(self, tensor_name: str) -> str:
        if self.weight_map:
            shard_file = self.weight_map.get(tensor_name)
            if shard_file is None:
                raise KeyError(f"tensor not found in weight map: {tensor_name}")
            return shard_file
        return "model.safetensors"

    def has_tensor(self, tensor_name: str) -> bool:
        if self.weight_map:
            return tensor_name in self.weight_map
        try:
            handle = self._handle_for("model.safetensors")
            return tensor_name in handle.keys()
        except FileNotFoundError:
            return False

    def _handle_for(self, file_name: str):
        handle = self._handles.get(file_name)
        if handle is not None:
            return handle
        path = self.model_dir / file_name
        handle = self._safe_open(str(path), framework="pt")
        self._handles[file_name] = handle
        return handle

    def get_tensor(self, tensor_name: str):
        file_name = self._resolve_file(tensor_name)
        handle = self._handle_for(file_name)
        return handle.get_tensor(tensor_name)

    def tensor_source(self, tensor_name: str) -> str:
        return str((self.model_dir / self._resolve_file(tensor_name)).resolve())

    def close(self) -> None:
        self._handles.clear()


def load_cdna_manifest(cdna_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(cdna_dir) / CDNA_MANIFEST_FILE).read_text(encoding="utf-8"))


def _load_existing_cdna_entry(
    cdna_dir: Path,
    checkpoint_key: str,
    runtime_param_name: str,
) -> dict[str, Any] | None:
    tensor_dir = cdna_dir / f"{_cdna_safe_name(checkpoint_key)}.cdnav3"
    meta_path = tensor_dir / "meta.json"
    stats_path = tensor_dir / "stats.json"
    if not meta_path.exists() or not stats_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return {
        "tensor_name": checkpoint_key,
        "runtime_param_name": runtime_param_name,
        "module_name": runtime_param_name[: -len(".weight")],
        "shape": list(meta.get("shape", [])),
        "kurtosis": None,
        "policy_storage_mode": meta.get("storage_mode"),
        "svd_residual_rank": int(meta.get("svd_residual_rank", 0)),
        "compressed_bytes": int(stats.get("compressed_bytes", 0)),
        "original_bytes": int(stats.get("original_bytes", 0)),
        "compression_ratio": float(stats.get("compression_ratio", 0.0)),
        "reused": True,
    }


def compress_huggingface_model_cdnav3(
    model_ref: str,
    output_dir: str | Path,
    *,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    CDNAv3Writer, get_policy, _, _ = _require_substrate()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / CDNA_MANIFEST_FILE
    if manifest_path.exists() and not force:
        return load_cdna_manifest(output_dir)

    config = _load_config(
        model_ref,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model_dir = resolve_hf_snapshot_dir(
        model_ref,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    reader = CheckpointReader(model_dir)
    writer = CDNAv3Writer(output_dir)
    compressed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    total_dense_bytes = 0

    try:
        tensor_names = sorted(reader.weight_map) if reader.weight_map else sorted(reader._handle_for("model.safetensors").keys())
        n_blocks = _runtime_num_layers(config)
        for checkpoint_key in tensor_names:
            if not checkpoint_key.endswith(".weight"):
                skipped.append({"tensor_name": checkpoint_key, "reason": "not-a-weight"})
                continue

            runtime_param_name = _checkpoint_key_to_runtime_param_name(config, checkpoint_key)
            if runtime_param_name is None:
                skipped.append({"tensor_name": checkpoint_key, "reason": "unsupported-runtime-path"})
                continue

            if not _should_compress_runtime_weight(runtime_param_name):
                skipped.append({"tensor_name": checkpoint_key, "runtime_param_name": runtime_param_name, "reason": "kept-dense"})
                continue

            existing = _load_existing_cdna_entry(output_dir, checkpoint_key, runtime_param_name)
            if existing is not None:
                compressed.append(existing)
                total_dense_bytes += int(existing["original_bytes"])
                continue

            tensor = reader.get_tensor(checkpoint_key)
            array = tensor.detach().cpu().float().numpy()
            if array.ndim != 2:
                skipped.append({"tensor_name": checkpoint_key, "runtime_param_name": runtime_param_name, "reason": "not-2d"})
                del tensor
                del array
                continue

            kurtosis = _sample_fisher_kurtosis(array)
            policy = get_policy(
                runtime_param_name,
                tuple(int(dim) for dim in array.shape),
                kurtosis=kurtosis,
                n_blocks=n_blocks,
            )
            stats = writer.write_tensor(
                array,
                checkpoint_key,
                policy=policy,
                source_artifact=reader.tensor_source(checkpoint_key),
            )
            compressed.append(
                {
                    "tensor_name": checkpoint_key,
                    "runtime_param_name": runtime_param_name,
                    "module_name": runtime_param_name[: -len(".weight")],
                    "shape": list(array.shape),
                    "kurtosis": round(float(kurtosis), 4),
                    "policy_storage_mode": policy.storage_mode,
                    "svd_residual_rank": int(policy.svd_residual_rank),
                    "compressed_bytes": int(stats.get("compressed_bytes", 0)),
                    "original_bytes": int(stats.get("original_bytes", int(array.nbytes))),
                    "compression_ratio": float(stats.get("compression_ratio", 0.0)),
                }
            )
            total_dense_bytes += int(array.nbytes)
            del tensor
            del array
            gc.collect()

    finally:
        reader.close()

    total_compressed_bytes = sum(int(item["compressed_bytes"]) for item in compressed)
    manifest = {
        "format": "helix-cdna-v3-manifest",
        "model_ref": model_ref,
        "model_dir": str(model_dir),
        "model_type": _runtime_model_type(config),
        "runtime_architecture": "Qwen3_5ForCausalLM" if _runtime_model_type(config) == "qwen3_5" else None,
        "compressed": compressed,
        "skipped": skipped,
        "compressed_tensors": len(compressed),
        "skipped_tensors": len(skipped),
        "dense_bytes_compressed_subset": total_dense_bytes,
        "compressed_bytes": total_compressed_bytes,
        "compression_ratio": round(total_dense_bytes / max(1, total_compressed_bytes), 4),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _set_state_tensor(model: Any, name: str, tensor: Any) -> None:
    torch, _, _, _, _, _ = _require_hf()
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    leaf = parts[-1]
    value = tensor.detach().cpu().contiguous()
    if torch.is_floating_point(value):
        value = value.float()
    if leaf in getattr(parent, "_parameters", {}):
        parent._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
        return
    if leaf in getattr(parent, "_buffers", {}):
        parent._buffers[leaf] = value
        return
    setattr(parent, leaf, value)


def _instantiate_meta_causal_model(config: Any, *, trust_remote_code: bool):
    torch, _, _, _, AutoModelForCausalLM, _ = _require_hf()
    model_type = _runtime_model_type(config)
    with torch.device("meta"):
        if model_type == "qwen3_5":
            from transformers import Qwen3_5ForCausalLM

            return Qwen3_5ForCausalLM(config.text_config)
        return AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)


def _count_meta_tensors(model: Any) -> int:
    count = 0
    for tensor in model.state_dict().values():
        if getattr(tensor, "is_meta", False):
            count += 1
    for _, module in model.named_modules():
        for _, buffer in module.named_buffers(recurse=False):
            if getattr(buffer, "is_meta", False):
                count += 1
    return count


def _materialize_nonpersistent_meta_buffers(model: Any) -> None:
    for _, module in model.named_modules():
        meta_buffer_names = [
            name
            for name, value in module.named_buffers(recurse=False)
            if getattr(value, "is_meta", False)
        ]
        if not meta_buffer_names:
            continue
        fresh = None
        config = getattr(module, "config", None)
        if config is not None:
            try:
                fresh = module.__class__(config, device="cpu")
            except TypeError:
                try:
                    fresh = module.__class__(config)
                except Exception:  # noqa: BLE001
                    fresh = None
            except Exception:  # noqa: BLE001
                fresh = None
        if fresh is None:
            continue
        for name, value in fresh.named_buffers(recurse=False):
            if name in meta_buffer_names:
                module._buffers[name] = value.detach().cpu().contiguous()


@dataclass(slots=True)
class LoadedGenerationTarget:
    requested_ref: str
    model_ref: str
    model_label: str
    model: Any
    tokenizer: Any
    load_mode: str
    storage_bytes: int | None
    workspace_info: dict[str, Any] | None = None
    backend: str = "transformers"


def _usage_completion_tokens(payload: dict[str, Any]) -> int | None:
    usage = payload.get("usage")
    if isinstance(usage, dict):
        value = usage.get("completion_tokens")
        if value is not None:
            try:
                return int(value)
            except Exception:  # noqa: BLE001
                return None
    return None


def load_cdna_text_generation_target(
    model_ref: str,
    cdna_dir: str | Path,
    *,
    tokenizer_dir: str | Path | None = None,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
) -> LoadedGenerationTarget:
    _, _, load_cdna_factors, swap_to_helix = _require_substrate()
    torch, _, _, AutoConfig, _, AutoTokenizer = _require_hf()
    config = AutoConfig.from_pretrained(
        model_ref,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model = _instantiate_meta_causal_model(config, trust_remote_code=trust_remote_code)
    helix_modules = load_cdna_factors(Path(cdna_dir), compute_dtype=torch.float32)
    model = swap_to_helix(model, helix_modules, compute_dtype=torch.float32)

    compressed_weight_names = {
        str(item["runtime_param_name"])
        for item in load_cdna_manifest(cdna_dir).get("compressed", [])
    }
    if bool(getattr(_text_config(config), "tie_word_embeddings", False)):
        compressed_weight_names.add("lm_head.weight")

    checkpoint_dir = resolve_hf_snapshot_dir(
        model_ref,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    reader = CheckpointReader(checkpoint_dir)
    try:
        for name in list(model.state_dict().keys()):
            if name in compressed_weight_names:
                continue
            checkpoint_key = _runtime_param_name_to_checkpoint_key(config, name)
            if checkpoint_key is None or not reader.has_tensor(checkpoint_key):
                continue
            tensor = reader.get_tensor(checkpoint_key)
            _set_state_tensor(model, name, tensor)
    finally:
        reader.close()

    if bool(getattr(_text_config(config), "tie_word_embeddings", False)):
        model.tie_weights()

    _materialize_nonpersistent_meta_buffers(model)

    meta_tensors = _count_meta_tensors(model)
    if meta_tensors:
        raise RuntimeError(f"compressed model still has {meta_tensors} meta tensors after loading")

    model.eval()
    tokenizer_source = tokenizer_dir if tokenizer_dir is not None else model_ref
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    return LoadedGenerationTarget(
        requested_ref=model_ref,
        model_ref=model_ref,
        model_label=f"{Path(cdna_dir).resolve().name} (cdnav3)",
        model=model,
        tokenizer=tokenizer,
        load_mode="cdnav3",
        storage_bytes=_dir_size_bytes(cdna_dir),
        backend="transformers",
    )


def load_llama_cpp_text_generation_target(
    gguf_path: str | Path,
    *,
    model_ref: str | None = None,
    chat_format: str | None = None,
    n_ctx: int = 4096,
    verbose: bool = False,
) -> LoadedGenerationTarget:
    Llama = _require_llama_cpp()
    gguf_file = Path(gguf_path).resolve()
    logical_cpus = max(1, int(os.cpu_count() or 1))
    default_threads = max(1, logical_cpus - 1)
    n_threads = int(os.environ.get("HELIX_LLAMA_THREADS", default_threads))
    n_threads_batch = int(os.environ.get("HELIX_LLAMA_THREADS_BATCH", max(1, logical_cpus)))
    n_batch = int(os.environ.get("HELIX_LLAMA_N_BATCH", min(int(n_ctx), 1024)))
    n_ubatch = int(os.environ.get("HELIX_LLAMA_N_UBATCH", min(int(n_ctx), 512)))
    llm_kwargs: dict[str, Any] = {
        "model_path": str(gguf_file),
        "n_ctx": int(n_ctx),
        "n_batch": n_batch,
        "n_ubatch": n_ubatch,
        "n_threads": n_threads,
        "n_threads_batch": n_threads_batch,
        "use_mmap": True,
        "use_mlock": False,
        "offload_kqv": True,
        "no_perf": True,
        "verbose": verbose,
    }
    if chat_format:
        llm_kwargs["chat_format"] = chat_format
    model = Llama(**llm_kwargs)
    return LoadedGenerationTarget(
        requested_ref=str(model_ref or gguf_file),
        model_ref=str(model_ref or gguf_file),
        model_label=gguf_file.stem,
        model=model,
        tokenizer=None,
        load_mode="llama_cpp_gguf",
        storage_bytes=int(gguf_file.stat().st_size),
        backend="llama-cpp-python",
    )


def build_generation_prompt(
    target: LoadedGenerationTarget,
    *,
    prompt: str | None = None,
    messages: list[dict[str, str]] | None = None,
    assistant_prefix: bool = True,
) -> str:
    from helix_proto.text import render_messages_prompt

    if messages is None:
        return prompt or ""
    tokenizer = target.tokenizer
    apply_chat = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat):
        try:
            return str(
                apply_chat(
                    messages,
                    tokenize=False,
                    add_generation_prompt=assistant_prefix,
                )
            )
        except Exception:  # noqa: BLE001
            pass
    return render_messages_prompt(messages, assistant_prefix=assistant_prefix)


def generate_text_with_target(
    target: LoadedGenerationTarget,
    *,
    prompt: str | None = None,
    messages: list[dict[str, str]] | None = None,
    max_new_tokens: int = 128,
    max_input_tokens: int = 2048,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> dict[str, Any]:
    if target.backend == "llama-cpp-python":
        chat_messages = messages
        if chat_messages is None and prompt is not None:
            chat_messages = [{"role": "user", "content": prompt}]
        if chat_messages:
            response = target.model.create_chat_completion(
                messages=chat_messages,
                max_tokens=max_new_tokens,
                temperature=temperature if do_sample else 0.0,
                top_k=top_k if do_sample else 0,
                top_p=top_p if do_sample else 1.0,
            )
            message = dict(response["choices"][0].get("message", {}) or {})
            text = str(message.get("content", "") or "").strip()
            generated_tokens = _usage_completion_tokens(response)
        else:
            prompt_text = prompt or ""
            response = target.model.create_completion(
                prompt=prompt_text,
                max_tokens=max_new_tokens,
                temperature=temperature if do_sample else 0.0,
                top_k=top_k if do_sample else 0,
                top_p=top_p if do_sample else 1.0,
            )
            text = str(response["choices"][0].get("text", "") or "").strip()
            generated_tokens = _usage_completion_tokens(response)
        if generated_tokens is None:
            generated_tokens = max(0, len(text.split()))
        return {
            "text": text,
            "generated_tokens": int(generated_tokens),
            "raw_response": response,
        }

    prompt_text = build_generation_prompt(
        target,
        prompt=prompt,
        messages=messages,
        assistant_prefix=True,
    )
    tokenizer = target.tokenizer
    model = target.model
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    generate_kwargs: dict[str, Any] = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded.get("attention_mask"),
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_k"] = top_k
        generate_kwargs["top_p"] = top_p
    output_ids = model.generate(
        **generate_kwargs,
    )
    generated_ids = output_ids[0, encoded["input_ids"].shape[1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "generated_tokens": int(generated_ids.shape[0]),
        "prompt_text": prompt_text,
        "generated_ids": generated_ids,
    }


def load_generation_target(
    model_ref: str,
    *,
    workspace_root: str | Path | None = None,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    torch_dtype: str = "auto",
) -> LoadedGenerationTarget:
    from helix_proto.workspace import resolve_gguf_path, resolve_model_info

    torch, _, _, _, AutoModelForCausalLM, AutoTokenizer = _require_hf()

    if workspace_root is not None:
        try:
            info = resolve_model_info(model_ref, workspace_root)
        except FileNotFoundError:
            info = None
        if info is not None and info.get("compression") == "cdnav3" and info.get("cdna_dir"):
            target = load_cdna_text_generation_target(
                str(info["model_ref"]),
                str(info["cdna_dir"]),
                tokenizer_dir=info.get("tokenizer_dir"),
                local_files_only=bool(info.get("local_files_only", local_files_only)),
                trust_remote_code=bool(info.get("trust_remote_code", trust_remote_code)),
            )
            target.requested_ref = model_ref
            target.model_label = f"{info.get('alias', model_ref)} (cdnav3)"
            target.workspace_info = info
            return target
        if info is not None and (
            info.get("source_format") == "gguf" or info.get("inference_backend") == "llama-cpp-python"
        ):
            target = load_llama_cpp_text_generation_target(
                resolve_gguf_path(model_ref, workspace_root),
                model_ref=str(info.get("model_ref", model_ref)),
                chat_format=info.get("chat_format"),
                n_ctx=int(info.get("n_ctx", 4096)),
            )
            target.requested_ref = model_ref
            target.model_label = str(info.get("alias", model_ref))
            target.workspace_info = info
            return target

    dtype_value: Any
    if torch_dtype == "float32":
        dtype_value = torch.float32
    elif torch_dtype == "float16":
        dtype_value = torch.float16
    elif torch_dtype == "bfloat16":
        dtype_value = torch.bfloat16
    else:
        dtype_value = "auto"

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype_value,
    )
    model.eval()
    return LoadedGenerationTarget(
        requested_ref=model_ref,
        model_ref=model_ref,
        model_label=model_ref,
        model=model,
        tokenizer=tokenizer,
        load_mode="dense",
        storage_bytes=None,
        backend="transformers",
    )
