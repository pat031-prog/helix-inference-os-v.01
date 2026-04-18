from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

LEGAL_SYSTEM_PROMPT = (
    "Sos Helix Juridico, un asistente de informacion juridica general para Argentina. "
    "Responde en espanol claro, ordenado y practico. Explica en terminos generales, "
    "marca incertidumbre cuando exista y aclara que no reemplaza asesoramiento profesional."
)

OPS_SYSTEM_PROMPT = (
    "Sos Helix Ops, un asistente de diagnostico para servidores Linux. "
    "Responde en espanol claro y accionable. Prioriza primero entender el sintoma, "
    "despues listar hipotesis, comandos de verificacion y pasos seguros de mitigacion. "
    "No inventes salidas de comandos y explicita riesgo cuando una accion pueda afectar produccion."
)


def repo_root() -> Path:
    return REPO_ROOT


def load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:  # pragma: no cover - direct error surface
            raise ValueError(f"invalid json on line {index} in {source}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"line {index} in {source} is not a JSON object")
        rows.append(item)
    return rows


def validate_dataset_rows(rows: list[dict[str, Any]], *, source: str | Path | None = None) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"dataset is empty: {source or '<memory>'}")

    missing_instruction: list[int] = []
    missing_output: list[int] = []
    keys: set[str] = set()
    instruction_chars = 0
    output_chars = 0

    for index, row in enumerate(rows, start=1):
        keys.update(row.keys())
        instruction = str(row.get("instruction") or "").strip()
        output = str(row.get("response") or row.get("output") or "").strip()
        if not instruction:
            missing_instruction.append(index)
        if not output:
            missing_output.append(index)
        instruction_chars += len(instruction)
        output_chars += len(output)

    if missing_instruction:
        raise ValueError(f"missing instruction in rows {missing_instruction[:10]}")
    if missing_output:
        raise ValueError(f"missing response/output in rows {missing_output[:10]}")

    return {
        "rows": len(rows),
        "keys": sorted(keys),
        "avg_instruction_chars": round(instruction_chars / len(rows), 2),
        "avg_output_chars": round(output_chars / len(rows), 2),
    }


def dataset_report(path: str | Path) -> dict[str, Any]:
    rows = load_jsonl_rows(path)
    report = validate_dataset_rows(rows, source=path)
    report["path"] = str(Path(path).resolve())
    report["sample_instruction"] = str(rows[0].get("instruction") or "")[:160]
    return report


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def load_job_config(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    config = dict(raw)
    config["job_config_path"] = str(Path(path).resolve())
    config["dataset_path"] = str(_resolve_path(config["dataset_path"]))
    return config


def training_script_path(trainer: str) -> Path:
    if trainer == "unsloth":
        return repo_root() / "finetune" / "train_qlora_unsloth.py"
    if trainer == "trl":
        return repo_root() / "finetune" / "train_qlora_trl.py"
    raise ValueError(f"unsupported trainer: {trainer}")


def pip_install_command(trainer: str) -> str:
    if trainer == "unsloth":
        return "pip install -U unsloth transformers datasets trl peft accelerate bitsandbytes safetensors"
    if trainer == "trl":
        return "pip install -U transformers datasets peft trl accelerate bitsandbytes safetensors"
    raise ValueError(f"unsupported trainer: {trainer}")


def build_train_command(config: dict[str, Any]) -> list[str]:
    script = training_script_path(str(config["trainer"]))
    command = [
        "python",
        str(script),
        "--model-name",
        str(config["model_name"]),
        "--dataset-path",
        str(config["dataset_path"]),
        "--output-dir",
        str(config["output_dir"]),
        "--system-prompt",
        str(config["system_prompt"]),
        "--max-seq-length",
        str(config.get("max_seq_length", 1024)),
        "--per-device-train-batch-size",
        str(config.get("per_device_train_batch_size", 1)),
        "--gradient-accumulation-steps",
        str(config.get("gradient_accumulation_steps", 4)),
        "--learning-rate",
        str(config.get("learning_rate", 2e-4)),
        "--num-train-epochs",
        str(config.get("num_train_epochs", 2)),
        "--max-steps",
        str(config.get("max_steps", -1)),
        "--warmup-steps",
        str(config.get("warmup_steps", 5)),
        "--logging-steps",
        str(config.get("logging_steps", 5)),
        "--save-steps",
        str(config.get("save_steps", 50)),
        "--seed",
        str(config.get("seed", 3407)),
        "--lora-r",
        str(config.get("lora_r", 16)),
        "--lora-alpha",
        str(config.get("lora_alpha", 32)),
        "--lora-dropout",
        str(config.get("lora_dropout", 0.05)),
        "--target-modules",
        str(config.get("target_modules", "")),
    ]
    if config.get("trust_remote_code"):
        command.append("--trust-remote-code")
    return command


def build_merge_command(config: dict[str, Any]) -> list[str] | None:
    if str(config["trainer"]) != "trl":
        return None
    merged_dir = str(config.get("merged_output_dir") or "")
    if not merged_dir:
        return None
    return [
        "python",
        str(repo_root() / "finetune" / "merge_peft_adapter.py"),
        "--base-model",
        str(config["model_name"]),
        "--adapter-dir",
        str(Path(config["output_dir"]) / "adapter"),
        "--output-dir",
        merged_dir,
    ]


def build_import_command(config: dict[str, Any], *, local_model_dir: str) -> list[str]:
    return [
        "python",
        "-m",
        "helix_proto.cli",
        "prepare-model",
        str(local_model_dir),
        "--alias",
        str(config.get("assistant_alias", config["job_name"])),
        "--workspace-root",
        str(config.get("workspace_root", ".\\workspace")),
        "--local-files-only",
    ]


def build_eval_command(config: dict[str, Any], *, local_model_dir: str) -> list[str]:
    output_root = str(config.get("eval_output_dir", f".\\benchmark-output\\{config['job_name']}"))
    return [
        "python",
        "-m",
        "helix_proto.cli",
        "eval-finetuned-model",
        str(local_model_dir),
        "--baseline-report",
        str(config["baseline_report"]),
        "--baseline-model-ref",
        str(config["baseline_model_ref"]),
        "--output",
        output_root,
    ]


def format_shell_command(argv: list[str]) -> str:
    return subprocess.list2cmdline(argv)


def render_colab_plan(config: dict[str, Any]) -> str:
    lines = [
        f"# Job: {config['job_name']}",
        "",
        "## Install",
        f"`{pip_install_command(str(config['trainer']))}`",
        "",
        "## Train",
        f"`{format_shell_command(build_train_command(config))}`",
    ]
    merge_command = build_merge_command(config)
    if merge_command:
        lines.extend(["", "## Merge", f"`{format_shell_command(merge_command)}`"])
    return "\n".join(lines) + "\n"
