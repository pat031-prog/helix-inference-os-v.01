from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import tempfile
from pathlib import Path, PurePosixPath


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _run(command: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print("DRY_RUN:", " ".join(_quote(item) for item in command))
        return
    subprocess.run(command, check=True)


def _ssh_target(user: str, host: str) -> str:
    return f"{user}@{host}"


def _resolve_remote_workdir(user: str, remote_workdir: str) -> PurePosixPath:
    workdir = str(remote_workdir).strip()
    if workdir == "~":
        return PurePosixPath("/root" if user == "root" else f"/home/{user}")
    if workdir.startswith("~/"):
        base = "/root" if user == "root" else f"/home/{user}"
        return PurePosixPath(base) / workdir[2:]
    return PurePosixPath(workdir)


def _scp_command(
    identity_file: str | None,
    port: int,
    sources: list[str],
    target: str,
    *,
    recursive: bool = False,
) -> list[str]:
    command = ["scp", "-P", str(port)]
    if recursive:
        command.append("-r")
    if identity_file:
        command.extend(["-i", identity_file])
    command.extend(sources)
    command.append(target)
    return command


def _ssh_command(identity_file: str | None, port: int, target: str, remote_command: str) -> list[str]:
    command = ["ssh", "-p", str(port)]
    if identity_file:
        command.extend(["-i", identity_file])
    command.extend([target, remote_command])
    return command


def _environment_manifest(args: argparse.Namespace, *, runner_version_hash: str) -> dict[str, object]:
    return {
        "benchmark_runner_version_hash": runner_version_hash,
        "torch_index_url": str(args.torch_index_url),
        "packages": {
            "setuptools": "<82",
            "numpy": "latest",
            "safetensors": "latest",
            "huggingface_hub": "latest",
            "transformers": "latest",
            "torch": "latest",
            "helix_substrate": "latest",
            "mamba_scan_lite": "latest",
        },
    }


def _remote_setup_contents(args: argparse.Namespace, *, runner_version_hash: str) -> str:
    desired_manifest = json.dumps(_environment_manifest(args, runner_version_hash=runner_version_hash), indent=2)
    return f"""from __future__ import annotations
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

workdir = Path({str(args.remote_workdir)!r}).expanduser()
workdir.mkdir(parents=True, exist_ok=True)
manifest_path = workdir / ".helix_env_manifest.json"
venv_path = workdir / ".venv"
python_path = venv_path / "bin" / "python"
desired = {desired_manifest}

def _run(*cmd: str) -> None:
    subprocess.check_call(list(cmd))

reuse = False
force_setup = os.environ.get("HELIX_FORCE_SETUP") == "1"
if not force_setup and python_path.exists() and manifest_path.exists():
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        reuse = existing.get("desired") == desired
    except Exception:
        reuse = False

if not reuse:
    if not venv_path.exists():
        _run("python3", "-m", "venv", str(venv_path))
    _run(str(python_path), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools<82")
    _run(str(python_path), "-m", "pip", "install", "numpy", "safetensors", "huggingface_hub", "transformers")
    _run(str(python_path), "-m", "pip", "install", "--upgrade", "torch", "--index-url", {str(args.torch_index_url)!r})
    _run(str(python_path), "-m", "pip", "install", "helix-substrate", "mamba-scan-lite")

actual = json.loads(
    subprocess.check_output(
        [
            str(python_path),
            "-c",
            (
                "import importlib.metadata, json, sys; "
                "print(json.dumps({{"
                "'python': sys.version.split()[0], "
                "'torch': importlib.metadata.version('torch'), "
                "'transformers': importlib.metadata.version('transformers'), "
                "'numpy': importlib.metadata.version('numpy'), "
                "'safetensors': importlib.metadata.version('safetensors'), "
                "'huggingface_hub': importlib.metadata.version('huggingface_hub'), "
                "'helix_substrate': importlib.metadata.version('helix-substrate'), "
                "'mamba_scan_lite': importlib.metadata.version('mamba-scan-lite')"
                "}}))"
            ),
        ],
        text=True,
    )
)
manifest_path.write_text(json.dumps({{"desired": desired, "actual": actual}}, indent=2), encoding="utf-8")
print("SETUP_REUSED" if reuse else "SETUP_INSTALLED")
print(manifest_path)
"""


def _remote_runner_contents(args: argparse.Namespace) -> str:
    return f"""from __future__ import annotations
import json
import os
from pathlib import Path
import torch
import transformers
from helix_kv.benchmark import build_transformers_variant_set, published_benchmark_context, run_transformers_kv_benchmark

workdir = Path({str(args.remote_workdir)!r})
workdir = workdir.expanduser()
requested_model_ref = {str(args.model_ref)!r}
effective_model_ref = requested_model_ref
benchmark_error = None
device = "cuda" if torch.cuda.is_available() else "cpu"
report = None
preflight = {{
    "cuda_available": bool(torch.cuda.is_available()),
    "device": device,
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "transformers_version": transformers.__version__,
    "gpu_name": None,
    "gpu_total_memory_mb": None,
    "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")),
}}
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    preflight["gpu_name"] = props.name
    preflight["gpu_total_memory_mb"] = int(props.total_memory / (1024 * 1024))

try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable on remote host; refusing CPU fallback")
    variants = build_transformers_variant_set(
        {json.dumps(str(args.variant_set))},
        kv_quant_seed={int(args.kv_quant_seed)},
        kv_hot_window={int(args.kv_hot_window)},
        kv_calibration_tokens={int(args.kv_calibration_tokens)},
        kv_adaptive_medium_kurtosis={float(args.adaptive_medium_kurtosis)},
        kv_adaptive_high_kurtosis={float(args.adaptive_high_kurtosis)},
    )
    report = run_transformers_kv_benchmark(
        requested_model_ref,
        prompt_length={int(args.prompt_length)},
        max_new_tokens={int(args.max_new_tokens)},
        kv_variants=variants,
        kv_quant_seed={int(args.kv_quant_seed)},
        kv_hot_window={int(args.kv_hot_window)},
        kv_calibration_tokens={int(args.kv_calibration_tokens)},
        kv_adaptive_high_kurtosis={float(args.adaptive_high_kurtosis)},
        kv_adaptive_medium_kurtosis={float(args.adaptive_medium_kurtosis)},
        local_files_only=False,
        device="cuda",
        trust_remote_code={bool(args.trust_remote_code)},
    )
    if not str(report.get("device", "")).startswith("cuda"):
        raise RuntimeError(f"benchmark reported unexpected device: {{report.get('device')}}")
    for row in report.get("rows", []):
        if not str(row.get("model_device", "")).startswith("cuda"):
            raise RuntimeError(f"model did not stay on CUDA for variant {{row.get('name')}}: {{row.get('model_device')}}")
        if not str(row.get("cache_device", "")).startswith("cuda"):
            raise RuntimeError(f"cache path did not stay on CUDA for variant {{row.get('name')}}: {{row.get('cache_device')}}")
except Exception as exc:
    benchmark_error = repr(exc)

payload = {{
    "requested_model_ref": requested_model_ref,
    "effective_model_ref": effective_model_ref,
    "device": device,
    "preflight": preflight,
    "benchmark_error": benchmark_error,
    "benchmark_kind": "transformers-kv",
    "prompt_length": {int(args.prompt_length)},
    "max_new_tokens": {int(args.max_new_tokens)},
    "kv_hot_window": {int(args.kv_hot_window)},
    "kv_quant_seed": {int(args.kv_quant_seed)},
    "kv_calibration_tokens": {int(args.kv_calibration_tokens)},
    "adaptive_medium_kurtosis": {float(args.adaptive_medium_kurtosis)},
    "adaptive_high_kurtosis": {float(args.adaptive_high_kurtosis)},
    "variant_set": {json.dumps(str(args.variant_set))},
    "published_context": published_benchmark_context(),
    "rows": [] if report is None else report.get("rows", []),
    "report": report,
}}
output_path = workdir / "benchmark_report.json"
output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(output_path)
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Helix KV benchmark on a remote host over SSH/SCP.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--identity-file")
    parser.add_argument("--model-ref", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("verification") / "remote-benchmark.json")
    parser.add_argument("--remote-workdir", default="~/helix-kv-transformers")
    parser.add_argument("--torch-index-url", default="https://download.pytorch.org/whl/cu128")
    parser.add_argument("--kv-hot-window", type=int, default=4)
    parser.add_argument("--kv-quant-seed", type=int, default=7)
    parser.add_argument("--kv-calibration-tokens", type=int, default=128)
    parser.add_argument("--adaptive-high-kurtosis", type=float, default=20.0)
    parser.add_argument("--adaptive-medium-kurtosis", type=float, default=9.0)
    parser.add_argument("--variant-set", choices=["stable", "asymmetry-sweep", "community"], default="stable")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--force-setup", action="store_true")
    parser.add_argument("--hf-token-env")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    remote_target = _ssh_target(args.user, args.host)
    remote_workdir = _resolve_remote_workdir(args.user, args.remote_workdir)
    remote_python = remote_workdir / ".venv" / "bin" / "python"

    with tempfile.TemporaryDirectory(prefix="helix-remote-benchmark-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        runner_path = tmp_path / "remote_runner.py"
        setup_path = tmp_path / "setup_remote.py"
        runner_version_hash = hashlib.sha256(
            (
                (repo_root / "tools" / "remote_benchmark.py").read_text(encoding="utf-8")
                + (repo_root / "helix_kv" / "transformers_cache.py").read_text(encoding="utf-8")
                + (repo_root / "helix_kv" / "benchmark.py").read_text(encoding="utf-8")
            ).encode("utf-8")
        ).hexdigest()[:16]
        runner_path.write_text(_remote_runner_contents(args), encoding="utf-8")
        setup_path.write_text(_remote_setup_contents(args, runner_version_hash=runner_version_hash), encoding="utf-8")

        _run(
            _ssh_command(
                args.identity_file,
                args.port,
                remote_target,
                f"mkdir -p {_quote(str(remote_workdir))}",
            ),
            dry_run=args.dry_run,
        )

        sources = [
            str(repo_root / "src" / "helix_proto"),
            str(repo_root / "helix_kv"),
            str(repo_root / "helix-kv"),
            str(setup_path),
            str(runner_path),
        ]
        _run(
            _scp_command(
                args.identity_file,
                args.port,
                sources,
                f"{remote_target}:{remote_workdir.as_posix()}/",
                recursive=True,
            ),
            dry_run=args.dry_run,
        )

        remote_setup = (
            f"cd {_quote(str(remote_workdir))} && "
            f"HELIX_FORCE_SETUP={'1' if args.force_setup else '0'} "
            f"python3 setup_remote.py"
        )
        _run(_ssh_command(args.identity_file, args.port, remote_target, remote_setup), dry_run=args.dry_run)

        env_prefix = ""
        if args.hf_token_env:
            env_prefix = f"export HF_TOKEN=${args.hf_token_env} && export HUGGINGFACE_HUB_TOKEN=${args.hf_token_env} && "
        remote_execute = (
            f"cd {_quote(str(remote_workdir))} && "
            f"{env_prefix}"
            f"PYTHONPATH=. {_quote(str(remote_python))} remote_runner.py"
        )
        _run(_ssh_command(args.identity_file, args.port, remote_target, remote_execute), dry_run=args.dry_run)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        _run(
            _scp_command(
                args.identity_file,
                args.port,
                [f"{remote_target}:{remote_workdir.as_posix()}/benchmark_report.json"],
                str(args.output),
            ),
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
