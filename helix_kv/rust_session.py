from __future__ import annotations

import hashlib
import json
import struct
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_CORE_MANIFEST = REPO_ROOT / "crates" / "helix-state-core" / "Cargo.toml"
GNU_TARGET = "x86_64-pc-windows-gnullvm"
HLX_MAGIC = b"HLXSTATE1\n"
FLAT_ARRAYS_META_KEY = "helix_flattened_arrays"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_pair(left: bytes, right: bytes) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(left)
    hasher.update(right)
    return hasher.digest()


def _merkle_root_hex(hex_hashes: list[str]) -> str:
    if not hex_hashes:
        return hashlib.sha256(b"").hexdigest()
    level = [bytes.fromhex(value) for value in hex_hashes]
    while len(level) > 1:
        next_level: list[bytes] = []
        for index in range(0, len(level), 2):
            left = level[index]
            right = level[index + 1] if index + 1 < len(level) else left
            next_level.append(_hash_pair(left, right))
        level = next_level
    return level[0].hex()


def _fnv1a_update(state: int, data: memoryview) -> int:
    value = int(state)
    for byte in data.cast("B"):
        value ^= int(byte)
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def _canonical_receipt_hash_payload(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_meta_hash": receipt.get("session_meta_hash"),
        "merkle_root": receipt.get("merkle_root"),
        "array_count": receipt.get("array_count"),
        "total_data_bytes": receipt.get("total_data_bytes"),
        "hlx_file_hash": receipt.get("hlx_file_hash"),
    }


def _session_hash(receipt: dict[str, Any]) -> str:
    payload = json.dumps(_canonical_receipt_hash_payload(receipt), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _state_core_module() -> Any | None:
    release_dir = _state_core_release_dir()
    if os.name == "nt":
        for dll_dir in (_mingw_bin(), Path(sys.executable).resolve().parent, release_dir):
            if dll_dir is None or not dll_dir.exists():
                continue
            try:
                os.add_dll_directory(str(dll_dir))
            except (AttributeError, OSError):
                pass
        pyd_path = _release_pyd()
        dll_path = release_dir / "_helix_state_core.dll"
        if dll_path.exists() and (
            not pyd_path.exists() or dll_path.stat().st_mtime > pyd_path.stat().st_mtime
        ):
            try:
                shutil.copyfile(dll_path, pyd_path)
            except OSError:
                pass
        if release_dir.exists() and str(release_dir) not in sys.path:
            sys.path.insert(0, str(release_dir))
    try:
        import _helix_state_core  # type: ignore[import-not-found]
    except Exception:
        return None
    return _helix_state_core


def _mingw_bin() -> Path | None:
    package_root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if not package_root.exists():
        return None
    candidates = sorted(package_root.glob("MartinStorsjo.LLVM-MinGW.*/*/bin"))
    return candidates[-1] if candidates else None


def _cargo_env() -> dict[str, str]:
    env = dict(os.environ)
    path_parts: list[str] = []
    mingw = _mingw_bin()
    if mingw is not None:
        path_parts.append(str(mingw))
    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.exists():
        path_parts.append(str(cargo_bin))
    path_parts.append(env.get("PATH", ""))
    env["PATH"] = os.pathsep.join(path_parts)
    return env


def _release_exe() -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return _state_core_release_dir() / f"helix-state-core{suffix}"


def _state_core_release_dir() -> Path:
    return REPO_ROOT / "crates" / "helix-state-core" / "target" / GNU_TARGET / "release"


def _release_pyd() -> Path:
    override = os.environ.get("HELIX_STATE_CORE_PYD")
    if override:
        return Path(override)
    suffix = ".pyd" if os.name == "nt" else ".so"
    return _state_core_release_dir() / f"_helix_state_core{suffix}"


def _debug_exe() -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return REPO_ROOT / "crates" / "helix-state-core" / "target" / GNU_TARGET / "debug" / f"helix-state-core{suffix}"


def _base_command() -> list[str]:
    override = os.environ.get("HELIX_STATE_CORE_CLI")
    if override:
        return [override]
    for candidate in (_release_exe(), _debug_exe()):
        if candidate.exists():
            return [str(candidate)]
    cargo = shutil.which("cargo", path=_cargo_env().get("PATH"))
    if cargo is None:
        raise RuntimeError("cargo not found and helix-state-core binary is unavailable")
    return [
        cargo,
        f"+stable-{GNU_TARGET}",
        "run",
        "--quiet",
        "--manifest-path",
        str(STATE_CORE_MANIFEST),
        "--target",
        GNU_TARGET,
        "--",
    ]


def _run_cli(args: list[str]) -> dict[str, Any]:
    command = _base_command() + args
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_cargo_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"helix-state-core failed: {completed.returncode}")
    return json.loads(completed.stdout)


def _array_to_staging(array: np.ndarray, *, name: str, staging_dir: Path) -> dict[str, Any]:
    if array.dtype == object:
        raise ValueError(f"object arrays are not supported by rust-hlx: {name}")
    contiguous = np.ascontiguousarray(array)
    arrays_dir = staging_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    relative = Path("arrays") / f"{name}.raw"
    contiguous.tofile(staging_dir / relative)
    return {
        "name": str(name),
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "path": str(relative).replace("\\", "/"),
    }


def write_hlx_session(path: str | Path, *, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="helix-hlx-staging-") as temp:
        staging_dir = Path(temp)
        staging_arrays = [
            _array_to_staging(array, name=name, staging_dir=staging_dir)
            for name, array in sorted(arrays.items())
        ]
        (staging_dir / "arrays.json").write_text(
            json.dumps({"arrays": staging_arrays}, indent=2),
            encoding="utf-8",
        )
        session_json = staging_dir / "session.json"
        session_json.write_text(json.dumps(_json_ready(meta), indent=2), encoding="utf-8")
        module = _state_core_module()
        if module is not None:
            receipt = json.loads(module.pack_hlx(str(staging_dir), str(session_json), str(destination)))
        else:
            receipt = _run_cli(
                [
                    "pack",
                    "--staging-dir",
                    str(staging_dir),
                    "--session-json",
                    str(session_json),
                    "--output-dir",
                    str(destination),
                ]
            )
    return receipt


def _array_to_buffer_spec(array: np.ndarray, *, name: str) -> tuple[dict[str, Any], bool]:
    if array.dtype == object:
        raise ValueError(f"object arrays are not supported by rust-hlx-buffered: {name}")
    copied = False
    source = array
    if not source.flags.c_contiguous:
        source = np.ascontiguousarray(source)
        copied = True
    view = memoryview(source).cast("B")
    return {
        "name": str(name),
        "dtype": str(source.dtype),
        "shape": list(source.shape),
        "byte_length": int(source.nbytes),
        "copied": bool(copied),
        "buffer": view,
        "_source": source,
    }, copied


def _write_hlx_buffered_arrays(
    path: str | Path,
    *,
    meta: dict[str, Any],
    arrays: dict[str, np.ndarray],
    requested_codec: str,
    receipt_extra: dict[str, Any] | None = None,
    audit_policy: str = "blocking",
) -> dict[str, Any]:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    module = _state_core_module()
    if module is None or not hasattr(module, "pack_hlx_buffers"):
        receipt = write_hlx_session(destination, meta=meta, arrays=arrays)
        receipt["session_codec_requested"] = str(requested_codec)
        receipt["buffered_fallback_reason"] = "pyo3_pack_hlx_buffers_unavailable"
        receipt["audit_policy_requested"] = str(audit_policy)
        receipt["audit_policy_effective"] = "blocking"
        if receipt_extra:
            receipt.update(_json_ready(receipt_extra))
        return receipt
    buffer_start = time.perf_counter()
    copied_array_count = 0
    specs: list[dict[str, Any]] = []
    keepalive: list[np.ndarray] = []
    for name, array in sorted(arrays.items()):
        spec, copied = _array_to_buffer_spec(array, name=name)
        copied_array_count += int(copied)
        keepalive.append(spec.pop("_source"))
        specs.append(spec)
    buffer_export_time_ms = (time.perf_counter() - buffer_start) * 1000.0
    meta_for_write = dict(meta)
    meta_for_write["session_codec"] = str(requested_codec)
    meta_for_write["kv_cache_file"] = "kv_cache.hlx"
    meta_for_write["audit_policy"] = str(audit_policy)
    meta_for_write["audit_status"] = "pending" if str(audit_policy) == "deferred" else "verified"
    meta_json = json.dumps(_json_ready(meta_for_write), separators=(",", ":"))
    start = time.perf_counter()
    if str(audit_policy) == "deferred":
        if not hasattr(module, "pack_hlx_buffers_pending"):
            receipt = json.loads(module.pack_hlx_buffers(meta_json, str(destination), specs))
            receipt["audit_policy_requested"] = "deferred"
            receipt["audit_policy_effective"] = "blocking"
            receipt["deferred_fallback_reason"] = "pyo3_pack_hlx_buffers_pending_unavailable"
        else:
            receipt = json.loads(module.pack_hlx_buffers_pending(meta_json, str(destination), specs))
            receipt["audit_policy_effective"] = "deferred"
    else:
        receipt = json.loads(module.pack_hlx_buffers(meta_json, str(destination), specs))
        receipt["audit_policy_effective"] = "blocking"
    session_json_path = destination / "session.json"
    try:
        written_meta = json.loads(session_json_path.read_text(encoding="utf-8"))
    except Exception:
        written_meta = {}
    if written_meta.get("session_codec") != str(requested_codec):
        written_meta["session_codec"] = str(requested_codec)
        written_meta["kv_cache_file"] = "kv_cache.hlx"
        session_json_path.write_text(json.dumps(_json_ready(written_meta), indent=2), encoding="utf-8")
        try:
            verified = verify_hlx_session(destination)
            inner = verified.get("receipt") if isinstance(verified.get("receipt"), dict) else verified
            receipt.update(inner)
        except Exception:
            receipt["session_codec_patch_verify_failed"] = True
    receipt["buffer_export_time_ms"] = float(buffer_export_time_ms)
    receipt["python_total_write_time_ms"] = (time.perf_counter() - start) * 1000.0 + buffer_export_time_ms
    receipt["copied_array_count"] = int(receipt.get("copied_array_count", copied_array_count))
    receipt["buffered_array_count"] = int(receipt.get("buffered_array_count", len(specs)))
    receipt["session_codec"] = str(requested_codec)
    receipt["buffer_spec_count"] = int(len(specs))
    receipt["audit_policy"] = str(audit_policy)
    receipt.setdefault("audit_status", "pending" if str(audit_policy) == "deferred" else "verified")
    receipt["time_to_pending_ms"] = receipt["python_total_write_time_ms"] if str(audit_policy) == "deferred" else None
    if receipt_extra:
        receipt.update(_json_ready(receipt_extra))
    (destination / "session-hlx-receipt.json").write_text(json.dumps(_json_ready(receipt), indent=2), encoding="utf-8")
    return receipt


def write_hlx_buffered_session(
    path: str | Path,
    *,
    meta: dict[str, Any],
    arrays: dict[str, np.ndarray],
    audit_policy: str = "blocking",
) -> dict[str, Any]:
    return _write_hlx_buffered_arrays(
        path,
        meta=meta,
        arrays=arrays,
        requested_codec="rust-hlx-buffered",
        audit_policy=audit_policy,
    )


def _flat_group_name(index: int) -> str:
    return f"__helix_flat_group_{int(index):02d}"


def _flatten_arrays_by_dtype(arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    start = time.perf_counter()
    grouped: dict[str, list[tuple[str, np.ndarray]]] = {}
    copied_input_count = 0
    original_total_bytes = 0
    for name, array in sorted(arrays.items()):
        source = np.asarray(array)
        if source.dtype == object:
            raise ValueError(f"object arrays are not supported by rust-hlx-buffered-flat: {name}")
        if not source.flags.c_contiguous:
            source = np.ascontiguousarray(source)
            copied_input_count += 1
        original_total_bytes += int(source.nbytes)
        grouped.setdefault(str(source.dtype), []).append((str(name), source))

    flat_arrays: dict[str, np.ndarray] = {}
    flat_groups: list[dict[str, Any]] = []
    for group_index, dtype_name in enumerate(sorted(grouped)):
        items = grouped[dtype_name]
        total_bytes = int(sum(int(array.nbytes) for _, array in items))
        slab = np.empty(total_bytes, dtype=np.uint8)
        offset = 0
        entries: list[dict[str, Any]] = []
        for name, array in items:
            byte_length = int(array.nbytes)
            if byte_length:
                slab[offset : offset + byte_length] = array.view(np.uint8).reshape(-1)
            entries.append(
                {
                    "name": name,
                    "dtype": str(array.dtype),
                    "shape": [int(dim) for dim in array.shape],
                    "group_offset": int(offset),
                    "byte_length": int(byte_length),
                }
            )
            offset += byte_length
        group_name = _flat_group_name(group_index)
        flat_arrays[group_name] = slab
        flat_groups.append(
            {
                "group_name": group_name,
                "source_dtype": dtype_name,
                "storage_dtype": "uint8",
                "byte_length": int(total_bytes),
                "arrays": entries,
            }
        )

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    flat_meta = {
        "format": "helix-flat-arrays-v1",
        "grouping": "by-dtype",
        "original_array_count": int(len(arrays)),
        "flat_group_count": int(len(flat_arrays)),
        "original_total_bytes": int(original_total_bytes),
        "flat_total_bytes": int(sum(int(array.nbytes) for array in flat_arrays.values())),
        "groups": flat_groups,
    }
    metrics = {
        "original_array_count": int(len(arrays)),
        "flat_group_count": int(len(flat_arrays)),
        "flattened_array_count": int(len(arrays)),
        "flatten_input_copied_array_count": int(copied_input_count),
        "flatten_copy_time_ms": float(elapsed_ms),
        "flatten_total_bytes": int(flat_meta["flat_total_bytes"]),
    }
    return flat_arrays, flat_meta, metrics


def _restore_flattened_arrays(arrays: dict[str, np.ndarray], flat_meta: dict[str, Any]) -> dict[str, np.ndarray]:
    restored: dict[str, np.ndarray] = {}
    for group in flat_meta.get("groups", []):
        if not isinstance(group, dict):
            continue
        group_name = str(group.get("group_name") or "")
        if group_name not in arrays:
            raise ValueError(f"flattened session is missing group array: {group_name}")
        slab = np.asarray(arrays[group_name])
        if slab.dtype != np.uint8:
            slab = slab.view(np.uint8).reshape(-1)
        else:
            slab = slab.reshape(-1)
        for entry in group.get("arrays", []):
            if not isinstance(entry, dict):
                continue
            name = str(entry["name"])
            dtype = np.dtype(str(entry["dtype"]))
            offset = int(entry["group_offset"])
            byte_length = int(entry["byte_length"])
            if offset % dtype.itemsize != 0:
                raise ValueError(f"flattened array offset is not aligned for {name}")
            if byte_length % dtype.itemsize != 0:
                raise ValueError(f"flattened array byte length is not aligned for {name}")
            if offset < 0 or byte_length < 0 or offset + byte_length > int(slab.nbytes):
                raise ValueError(f"flattened array range is out of bounds for {name}")
            count = byte_length // dtype.itemsize
            array = np.frombuffer(slab, dtype=dtype, count=count, offset=offset)
            restored[name] = array.reshape(tuple(int(dim) for dim in entry.get("shape", [])))
    return restored


def write_hlx_buffered_flat_session(
    path: str | Path,
    *,
    meta: dict[str, Any],
    arrays: dict[str, np.ndarray],
    audit_policy: str = "blocking",
) -> dict[str, Any]:
    flat_arrays, flat_meta, metrics = _flatten_arrays_by_dtype(arrays)
    meta_for_write = dict(meta)
    meta_for_write[FLAT_ARRAYS_META_KEY] = flat_meta
    receipt = _write_hlx_buffered_arrays(
        path,
        meta=meta_for_write,
        arrays=flat_arrays,
        requested_codec="rust-hlx-buffered-flat",
        receipt_extra=metrics,
        audit_policy=audit_policy,
    )
    receipt["buffered_flat_enabled"] = True
    receipt["buffer_spec_count"] = int(metrics["flat_group_count"])
    (Path(path) / "session-hlx-receipt.json").write_text(json.dumps(_json_ready(receipt), indent=2), encoding="utf-8")
    return receipt


def _read_hlx_payload(source: Path) -> tuple[bytearray, dict[str, Any], int]:
    hlx_path = source / "kv_cache.hlx"
    payload = bytearray(hlx_path.read_bytes())
    if bytes(payload[: len(HLX_MAGIC)]) != HLX_MAGIC:
        raise ValueError("invalid HeliX .hlx magic")
    offset = len(HLX_MAGIC)
    manifest_len = struct.unpack_from("<Q", payload, offset)[0]
    offset += 8
    manifest = json.loads(bytes(payload[offset : offset + manifest_len]).decode("utf-8"))
    data_start = offset + int(manifest_len)
    return payload, manifest, data_start


def _read_hlx_manifest_only(source: Path) -> tuple[dict[str, Any], int]:
    hlx_path = source / "kv_cache.hlx"
    with hlx_path.open("rb") as handle:
        magic = handle.read(len(HLX_MAGIC))
        if magic != HLX_MAGIC:
            raise ValueError("invalid HeliX .hlx magic")
        manifest_len = struct.unpack("<Q", handle.read(8))[0]
        manifest = json.loads(handle.read(int(manifest_len)).decode("utf-8"))
    return manifest, len(HLX_MAGIC) + 8 + int(manifest_len)


def _parse_hlx_arrays_direct(source: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    payload, manifest, data_start = _read_hlx_payload(source)
    arrays: dict[str, np.ndarray] = {}
    for entry in manifest.get("arrays", []):
        dtype = np.dtype(str(entry["dtype"]))
        byte_offset = data_start + int(entry["offset"])
        byte_length = int(entry["byte_length"])
        array = np.frombuffer(payload, dtype=dtype, count=byte_length // dtype.itemsize, offset=byte_offset)
        arrays[str(entry["name"])] = array.reshape(tuple(int(dim) for dim in entry.get("shape", [])))
    return arrays, manifest


def _parse_hlx_arrays_selected(source: Path, selected_names: set[str]) -> tuple[dict[str, np.ndarray], dict[str, Any], int]:
    manifest, data_start = _read_hlx_manifest_only(source)
    arrays: dict[str, np.ndarray] = {}
    bytes_read = 0
    with (source / "kv_cache.hlx").open("rb") as handle:
        for entry in manifest.get("arrays", []):
            name = str(entry["name"])
            if name not in selected_names:
                continue
            dtype = np.dtype(str(entry["dtype"]))
            byte_length = int(entry["byte_length"])
            handle.seek(data_start + int(entry["offset"]))
            data = handle.read(byte_length)
            if len(data) != byte_length:
                raise ValueError(f"short read for .hlx array: {name}")
            bytes_read += byte_length
            array = np.frombuffer(data, dtype=dtype, count=byte_length // dtype.itemsize)
            arrays[name] = array.reshape(tuple(int(dim) for dim in entry.get("shape", [])))
    return arrays, manifest, bytes_read


def _infer_layer_slice_meta(meta: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    existing = meta.get("helix_layer_slices")
    if isinstance(existing, dict):
        return existing
    layers: dict[str, dict[str, Any]] = {}
    pattern = re.compile(r"(?:^|_)layer[_-]?(\d+)(?:_|$)|^layer_(\d+)_")
    for entry in manifest.get("arrays", []):
        name = str(entry.get("name") or "")
        match = pattern.search(name)
        if not match:
            continue
        layer_index = int(next(item for item in match.groups() if item is not None))
        key = str(layer_index)
        item = layers.setdefault(
            key,
            {
                "layer_index": layer_index,
                "layer_name": f"layer_{layer_index}",
                "arrays": [],
            },
        )
        item["arrays"].append(
            {
                "name": name,
                "layer_index": layer_index,
                "layer_name": item["layer_name"],
                "cache_kind": "unknown",
            }
        )
    return {"format": "helix-layer-slices-v0", "layers": list(layers.values())}


def load_hlx_layer_slice(
    path: str | Path,
    layer_index: int,
    *,
    verify_policy: str = "receipt-only",
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    source = Path(path)
    meta = json.loads((source / "session.json").read_text(encoding="utf-8"))
    manifest, _ = _read_hlx_manifest_only(source)
    layer_meta = _infer_layer_slice_meta(meta, manifest)
    target_layer = int(layer_index)
    selected_entries: list[dict[str, Any]] = []
    for layer in layer_meta.get("layers", []):
        if int(layer.get("layer_index", -1)) != target_layer:
            continue
        for entry in layer.get("arrays", []):
            if isinstance(entry, dict) and entry.get("name"):
                selected_entries.append(dict(entry))
    selected_names = {str(entry["name"]) for entry in selected_entries}
    if not selected_names:
        return meta, {}, {
            "status": "miss",
            "layer_index": target_layer,
            "array_count": 0,
            "selected_array_names": [],
            "bytes_read": 0,
            "verify_policy": str(verify_policy),
            "manifest": manifest,
        }

    flat_meta = meta.get(FLAT_ARRAYS_META_KEY)
    read_mode = "direct-selected"
    if isinstance(flat_meta, dict):
        all_arrays, manifest = _parse_hlx_arrays_direct(source)
        restored = _restore_flattened_arrays(all_arrays, flat_meta)
        arrays = {name: restored[name] for name in selected_names if name in restored}
        bytes_read = int(sum(int(array.nbytes) for array in all_arrays.values()))
        read_mode = "flattened-full-slab"
    else:
        arrays, manifest, bytes_read = _parse_hlx_arrays_selected(source, selected_names)

    if str(verify_policy) == "full":
        receipt = verify_hlx_session(source)
    elif str(verify_policy) == "receipt-only":
        receipt = _read_receipt(source) or {"session_codec": meta.get("session_codec"), "kv_cache_file": "kv_cache.hlx"}
    else:
        raise ValueError(f"unsupported verify_policy: {verify_policy}")
    receipt["layer_slice"] = {
        "status": "hit",
        "layer_index": target_layer,
        "array_count": len(arrays),
        "selected_array_names": sorted(arrays),
        "declared_array_names": sorted(selected_names),
        "bytes_read": int(bytes_read),
        "read_mode": read_mode,
        "verify_policy": str(verify_policy),
        "entries": _json_ready(selected_entries),
    }
    receipt["manifest"] = manifest
    return meta, arrays, receipt


def _read_receipt(path: Path) -> dict[str, Any]:
    receipt_path = path / "session-hlx-receipt.json"
    if not receipt_path.exists():
        return {}
    try:
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    (path / "session-hlx-receipt.json").write_text(json.dumps(_json_ready(receipt), indent=2), encoding="utf-8")


def _is_deferred_receipt(receipt: dict[str, Any]) -> bool:
    return (
        str(receipt.get("audit_policy") or receipt.get("audit_policy_effective") or "") == "deferred"
        or str(receipt.get("format") or "") == "helix-session-pending-v0"
    )


def verify_deferred_session(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    start = time.perf_counter()
    previous = _read_receipt(source)
    try:
        meta = json.loads((source / "session.json").read_text(encoding="utf-8"))
        payload, manifest, data_start = _read_hlx_payload(source)
        all_chunk_hashes: list[str] = []
        fast_checksum = 14695981039346656037
        for entry in manifest.get("arrays", []):
            offset = data_start + int(entry["offset"])
            byte_length = int(entry["byte_length"])
            if offset < 0 or byte_length < 0 or offset + byte_length > len(payload):
                raise ValueError(f"array range is out of bounds: {entry.get('name')}")
            segment = memoryview(payload)[offset : offset + byte_length]
            fast_checksum = _fnv1a_update(fast_checksum, segment)
            for chunk_start in range(0, byte_length, 1024 * 1024):
                chunk = segment[chunk_start : chunk_start + 1024 * 1024]
                all_chunk_hashes.append(hashlib.sha256(chunk).hexdigest())
            if byte_length == 0:
                all_chunk_hashes.append(hashlib.sha256(b"").hexdigest())
        pending_checksum = previous.get("fast_payload_checksum")
        computed_checksum = f"{fast_checksum:016x}"
        if pending_checksum and str(pending_checksum) != computed_checksum:
            raise ValueError("fast payload checksum mismatch")
        merkle_root = _merkle_root_hex(all_chunk_hashes)
        session_total_bytes = int((source / "session.json").stat().st_size + (source / "kv_cache.hlx").stat().st_size)
        receipt: dict[str, Any] = {
            "format": "helix-session-receipt-v0",
            "session_codec": str(meta.get("session_codec") or previous.get("session_codec") or "rust-hlx"),
            "audit_policy": "deferred",
            "audit_status": "verified",
            "session_meta_hash": _sha256_file(source / "session.json"),
            "hlx_file_hash": _sha256_file(source / "kv_cache.hlx"),
            "merkle_root": merkle_root,
            "array_count": int(len(manifest.get("arrays", []))),
            "total_data_bytes": int(manifest.get("total_data_bytes") or 0),
            "session_total_bytes": session_total_bytes,
            "kv_cache_file": "kv_cache.hlx",
            "fast_payload_checksum": computed_checksum,
            "pending_fast_payload_checksum": pending_checksum,
            "deferred_audit_time_ms": (time.perf_counter() - start) * 1000.0,
        }
        receipt["session_hash"] = _session_hash(receipt)
        if previous.get("audit_status") == "verified" and previous.get("session_hash") and previous.get("session_hash") != receipt["session_hash"]:
            raise ValueError("verified session hash mismatch")
        receipt["ok"] = True
        _write_receipt(source, receipt)
        return receipt
    except Exception as exc:
        failed = dict(previous)
        failed.update(
            {
                "audit_policy": "deferred",
                "audit_status": "failed",
                "ok": False,
                "audit_error": str(exc),
                "deferred_audit_time_ms": (time.perf_counter() - start) * 1000.0,
            }
        )
        _write_receipt(source, failed)
        raise RuntimeError(str(exc)) from exc


def read_hlx_session(path: str | Path, *, verify_policy: str = "full") -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    source = Path(path)
    meta = json.loads((source / "session.json").read_text(encoding="utf-8"))
    arrays, manifest = _parse_hlx_arrays_direct(source)
    flat_meta = meta.get(FLAT_ARRAYS_META_KEY)
    if isinstance(flat_meta, dict):
        arrays = _restore_flattened_arrays(arrays, flat_meta)
    if str(verify_policy) == "full":
        receipt = verify_hlx_session(source)
    elif str(verify_policy) == "receipt-only":
        receipt_path = source / "session-hlx-receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {
            "session_codec": meta.get("session_codec"),
            "kv_cache_file": "kv_cache.hlx",
        }
    else:
        raise ValueError(f"unsupported verify_policy: {verify_policy}")
    receipt["manifest"] = manifest
    if isinstance(flat_meta, dict):
        receipt["flattened_arrays"] = {
            "format": flat_meta.get("format"),
            "original_array_count": flat_meta.get("original_array_count"),
            "flat_group_count": flat_meta.get("flat_group_count"),
        }
    return meta, arrays, receipt


def verify_hlx_session(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    receipt = _read_receipt(source)
    if _is_deferred_receipt(receipt):
        return verify_deferred_session(source)
    module = _state_core_module()
    if module is not None:
        return json.loads(module.verify_hlx(str(source)))
    return _run_cli(["verify", "--session-dir", str(source)])


def save_session_bundle(
    path: str | Path,
    *,
    meta: dict[str, Any],
    arrays: dict[str, np.ndarray],
    session_codec: str = "python-npz",
    audit_policy: str = "blocking",
) -> dict[str, Any]:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    codec = str(session_codec)
    if codec == "rust-hlx":
        start = time.perf_counter()
        receipt = write_hlx_session(destination, meta=meta, arrays=arrays)
        receipt["write_time_ms"] = (time.perf_counter() - start) * 1000.0
        return receipt
    if codec == "rust-hlx-buffered":
        start = time.perf_counter()
        receipt = write_hlx_buffered_session(destination, meta=meta, arrays=arrays, audit_policy=audit_policy)
        receipt["write_time_ms"] = (time.perf_counter() - start) * 1000.0
        return receipt
    if codec == "rust-hlx-buffered-flat":
        start = time.perf_counter()
        receipt = write_hlx_buffered_flat_session(destination, meta=meta, arrays=arrays, audit_policy=audit_policy)
        receipt["write_time_ms"] = (time.perf_counter() - start) * 1000.0
        return receipt
    if codec not in {"python-npz", "auto"}:
        raise ValueError(f"unsupported session codec: {session_codec}")
    meta = dict(meta)
    meta["session_codec"] = "python-npz"
    meta["kv_cache_file"] = "kv_cache.npz"
    (destination / "session.json").write_text(json.dumps(_json_ready(meta), indent=2), encoding="utf-8")
    with (destination / "kv_cache.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    return {
        "session_codec": "python-npz",
        "kv_cache_file": "kv_cache.npz",
        "session_total_bytes": int((destination / "session.json").stat().st_size + (destination / "kv_cache.npz").stat().st_size),
    }


def load_session_bundle(path: str | Path, *, verify_policy: str = "full") -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    source = Path(path)
    if (source / "kv_cache.hlx").exists():
        return read_hlx_session(source, verify_policy=verify_policy)
    meta = json.loads((source / "session.json").read_text(encoding="utf-8"))
    arrays = dict(np.load(source / "kv_cache.npz", allow_pickle=False))
    return meta, arrays, {
        "session_codec": "python-npz",
        "kv_cache_file": "kv_cache.npz",
        "session_total_bytes": int((source / "session.json").stat().st_size + (source / "kv_cache.npz").stat().st_size),
    }


def toolchain_report() -> dict[str, Any]:
    module = _state_core_module()
    report: dict[str, Any] = {
        "pyo3_module_available": module is not None,
        "cargo_manifest": str(STATE_CORE_MANIFEST),
        "release_exe": str(_release_exe()),
        "release_exe_exists": _release_exe().exists(),
        "pyo3_extension": str(_release_pyd()),
        "pyo3_extension_exists": _release_pyd().exists(),
        "debug_exe": str(_debug_exe()),
        "debug_exe_exists": _debug_exe().exists(),
        "gnu_target": GNU_TARGET,
    }
    try:
        command = _base_command() + ["verify", "--session-dir", str(REPO_ROOT / "__missing__")]
        report["cli_command_prefix"] = command[:-3]
        report["cli_available"] = True
    except Exception as exc:
        report["cli_available"] = False
        report["cli_error"] = str(exc)
    return report
