import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "tools" / "remote_benchmark.py"
_SPEC = importlib.util.spec_from_file_location("remote_benchmark", _SCRIPT_PATH)
assert _SPEC and _SPEC.loader
remote_benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(remote_benchmark)


def test_scp_command_adds_recursive_flag_when_requested() -> None:
    command = remote_benchmark._scp_command(
        identity_file="id_rsa",
        port=2222,
        sources=["src", "helix_kv"],
        target="user@example.com:~/work/",
        recursive=True,
    )

    assert command[:4] == ["scp", "-P", "2222", "-r"]
    assert "-i" in command


def test_resolve_remote_workdir_expands_home_for_root_and_regular_user() -> None:
    assert remote_benchmark._resolve_remote_workdir("root", "~/helix-kv-remote").as_posix() == "/root/helix-kv-remote"
    assert remote_benchmark._resolve_remote_workdir("pat", "~/helix-kv-remote").as_posix() == "/home/pat/helix-kv-remote"


def test_remote_runner_contents_reference_model_and_transformers_benchmark() -> None:
    args = argparse.Namespace(
        remote_workdir="~/helix-kv-transformers",
        model_ref="Qwen/Qwen2.5-1.5B-Instruct",
        prompt_length=512,
        max_new_tokens=32,
        kv_hot_window=4,
        kv_quant_seed=7,
        kv_calibration_tokens=128,
        adaptive_high_kurtosis=20.0,
        adaptive_medium_kurtosis=9.0,
        variant_set="asymmetry-sweep",
        trust_remote_code=True,
    )

    contents = remote_benchmark._remote_runner_contents(args)

    assert "Qwen/Qwen2.5-1.5B-Instruct" in contents
    assert "run_transformers_kv_benchmark" in contents
    assert "build_transformers_variant_set" in contents
    assert '"benchmark_kind": "transformers-kv"' in contents
    assert '"published_context"' in contents
    assert "benchmark_error" in contents
    assert 'if not torch.cuda.is_available()' in contents
    assert "cache path did not stay on CUDA" in contents
    assert '"variant_set": "asymmetry-sweep"' in contents
    assert '"hf_token_present"' in contents
    assert "trust_remote_code=True" in contents


def test_remote_setup_contents_tracks_manifest_and_reuses_env() -> None:
    args = argparse.Namespace(
        torch_index_url="https://download.pytorch.org/whl/cu128",
        remote_workdir="~/helix-kv-transformers",
    )

    contents = remote_benchmark._remote_setup_contents(args, runner_version_hash="abc123")

    assert ".helix_env_manifest.json" in contents
    assert "HELIX_FORCE_SETUP" in contents
    assert "SETUP_REUSED" in contents
    assert "SETUP_INSTALLED" in contents
    assert "helix-substrate" in contents
    assert "mamba-scan-lite" in contents


def test_remote_benchmark_dry_run_prints_ssh_and_recursive_scp(tmp_path: Path) -> None:
    output_path = tmp_path / "remote-report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--host",
            "example.com",
            "--user",
            "pat",
            "--output",
            str(output_path),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = completed.stdout
    assert "DRY_RUN: ssh" in stdout
    assert "DRY_RUN: scp -P 22 -r" in stdout
    assert "remote_runner.py" in stdout
    assert "setup_remote.py" in stdout
    assert "helix_proto" in stdout
    assert "PYTHONPATH=." in stdout
    assert "PYTHONPATH=src:." not in stdout
    assert "rm -rf" not in stdout
