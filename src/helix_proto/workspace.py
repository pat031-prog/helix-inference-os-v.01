from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from helix_proto.hf import export_huggingface_model
from helix_proto.text import tokenizer_path, try_prepare_tokenizer


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._").lower()
    return slug or "model"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def workspace_root(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root).resolve()
    return Path(__file__).resolve().parents[2] / "workspace"


def models_dir(root: str | Path | None = None) -> Path:
    path = workspace_root(root) / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_workspace(alias: str, root: str | Path | None = None) -> Path:
    return models_dir(root) / slugify(alias)


def tokenizer_dir(model_dir: str | Path) -> Path:
    return tokenizer_path(model_dir)


def sessions_dir(model_dir: str | Path) -> Path:
    path = Path(model_dir) / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_session_dir(
    alias: str,
    session_id: str,
    root: str | Path | None = None,
) -> Path:
    return sessions_dir(model_workspace(alias, root)) / slugify(session_id)


def model_info_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / "model_info.json"


def load_model_info(model_dir: str | Path) -> dict[str, Any]:
    return json.loads(model_info_path(model_dir).read_text(encoding="utf-8"))


def save_model_info(model_dir: str | Path, info: dict[str, Any]) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_info_path(model_dir)
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return path


def list_model_workspaces(root: str | Path | None = None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for child in sorted(models_dir(root).iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        info_file = model_info_path(child)
        if not info_file.exists():
            continue
        info = load_model_info(child)
        info["available_sessions"] = sorted(
            session.name for session in sessions_dir(child).iterdir() if session.is_dir()
        )
        entries.append(info)
    return entries


def resolve_model_info(alias: str, root: str | Path | None = None) -> dict[str, Any]:
    model_dir = model_workspace(alias, root)
    info_file = model_info_path(model_dir)
    if not info_file.exists():
        raise FileNotFoundError(f"model alias not found: {alias}")
    return load_model_info(model_dir)


def resolve_export_dir(alias: str, root: str | Path | None = None) -> Path:
    info = resolve_model_info(alias, root)
    export_dir_value = info.get("export_dir")
    if not export_dir_value:
        raise FileNotFoundError(f"export directory is not available for alias: {alias}")
    export_dir = Path(export_dir_value)
    if not export_dir.exists():
        raise FileNotFoundError(f"export directory is missing for alias: {alias}")
    return export_dir


def resolve_gguf_path(alias: str, root: str | Path | None = None) -> Path:
    info = resolve_model_info(alias, root)
    gguf_path_value = info.get("gguf_path")
    if not gguf_path_value:
        raise FileNotFoundError(f"GGUF file is not available for alias: {alias}")
    gguf_path = Path(gguf_path_value)
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF file is missing for alias: {alias}")
    return gguf_path


def resolve_tokenizer_dir(alias: str, root: str | Path | None = None) -> Path:
    info = resolve_model_info(alias, root)
    tokenizer_dir_value = info.get("tokenizer_dir")
    if not tokenizer_dir_value:
        raise FileNotFoundError(f"tokenizer is not available for alias: {alias}")
    path = Path(tokenizer_dir_value)
    if not path.exists():
        raise FileNotFoundError(f"tokenizer directory is missing for alias: {alias}")
    return path


def prepare_model_workspace(
    *,
    model_ref: str,
    alias: str | None = None,
    root: str | Path | None = None,
    block_rows: int = 256,
    compress: str | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    force: bool = False,
    chat_format: str | None = None,
    n_ctx: int = 4096,
) -> dict[str, Any]:
    alias = alias or model_ref
    model_dir = model_workspace(alias, root)
    export_dir = model_dir / "export"
    cdna_dir = model_dir / "cdna"
    info_path = model_info_path(model_dir)
    model_path = Path(model_ref)

    if model_path.suffix.lower() == ".gguf" and model_path.exists():
        if info_path.exists() and not force:
            info = load_model_info(model_dir)
            info["reused"] = True
            return info

        model_dir.mkdir(parents=True, exist_ok=True)
        info = {
            "alias": alias,
            "alias_slug": slugify(alias),
            "model_ref": str(model_path.resolve()),
            "workspace_root": str(workspace_root(root)),
            "model_dir": str(model_dir),
            "export_dir": None,
            "compression": None,
            "source_format": "gguf",
            "inference_backend": "llama-cpp-python",
            "gguf_path": str(model_path.resolve()),
            "gguf_filename": model_path.name,
            "gguf_size_bytes": int(model_path.stat().st_size),
            "tokenizer_dir": None,
            "sessions_dir": str(sessions_dir(model_dir)),
            "block_rows": block_rows,
            "local_files_only": True,
            "trust_remote_code": trust_remote_code,
            "chat_format": chat_format,
            "n_ctx": int(n_ctx),
            "manifest_path": None,
            "exported_tensors": 0,
            "skipped_tensors": 0,
            "model_type": None,
            "architectures": [],
            "prepared_at_utc": _utc_now_iso(),
            "reused": False,
            "has_tokenizer": False,
        }
        save_model_info(model_dir, info)
        return info

    if compress == "cdnav3":
        from helix_proto.cdna import compress_huggingface_model_cdnav3

        if cdna_dir.exists() and info_path.exists() and not force:
            info = load_model_info(model_dir)
            if not info.get("has_tokenizer") or not Path(info.get("tokenizer_dir", "")).exists():
                tokenizer_info = try_prepare_tokenizer(
                    model_ref,
                    tokenizer_dir(model_dir),
                    local_files_only=local_files_only,
                    trust_remote_code=trust_remote_code,
                )
                info.update(tokenizer_info)
                save_model_info(model_dir, info)
            info["reused"] = True
            return info

        model_dir.mkdir(parents=True, exist_ok=True)
        manifest = compress_huggingface_model_cdnav3(
            model_ref,
            cdna_dir,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            force=force,
        )
        tokenizer_info = try_prepare_tokenizer(
            model_ref,
            tokenizer_dir(model_dir),
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        info = {
            "alias": alias,
            "alias_slug": slugify(alias),
            "model_ref": model_ref,
            "workspace_root": str(workspace_root(root)),
            "model_dir": str(model_dir),
            "export_dir": None,
            "compression": "cdnav3",
            "cdna_dir": str(cdna_dir),
            "cdna_manifest_path": str(cdna_dir / "cdna_manifest.json"),
            "tokenizer_dir": str(tokenizer_dir(model_dir)),
            "sessions_dir": str(sessions_dir(model_dir)),
            "block_rows": block_rows,
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
            "manifest_path": None,
            "exported_tensors": 0,
            "skipped_tensors": 0,
            "compressed_tensors": int(manifest.get("compressed_tensors", 0)),
            "cdna_compression_ratio": float(manifest.get("compression_ratio", 0.0)),
            "model_type": None,
            "architectures": [],
            "prepared_at_utc": _utc_now_iso(),
            "reused": False,
        }
        info.update(tokenizer_info)
        save_model_info(model_dir, info)
        return info

    if export_dir.exists() and info_path.exists() and not force:
        info = load_model_info(model_dir)
        if not info.get("has_tokenizer") or not Path(info.get("tokenizer_dir", "")).exists():
            tokenizer_info = try_prepare_tokenizer(
                model_ref,
                tokenizer_dir(model_dir),
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
            info.update(tokenizer_info)
            save_model_info(model_dir, info)
        info["reused"] = True
        return info

    model_dir.mkdir(parents=True, exist_ok=True)
    manifest = export_huggingface_model(
        model_ref,
        export_dir,
        block_rows=block_rows,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    config = manifest.get("config", {})
    tokenizer_info = try_prepare_tokenizer(
        model_ref,
        tokenizer_dir(model_dir),
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    info = {
        "alias": alias,
        "alias_slug": slugify(alias),
        "model_ref": model_ref,
        "workspace_root": str(workspace_root(root)),
        "model_dir": str(model_dir),
        "export_dir": str(export_dir),
        "compression": None,
        "source_format": "transformers",
        "inference_backend": "transformers",
        "tokenizer_dir": str(tokenizer_dir(model_dir)),
        "sessions_dir": str(sessions_dir(model_dir)),
        "block_rows": block_rows,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
        "manifest_path": str(export_dir / "manifest.json"),
        "exported_tensors": len(manifest["exported"]),
        "skipped_tensors": len(manifest["skipped"]),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures", []),
        "prepared_at_utc": _utc_now_iso(),
        "reused": False,
    }
    info.update(tokenizer_info)
    save_model_info(model_dir, info)
    return info
