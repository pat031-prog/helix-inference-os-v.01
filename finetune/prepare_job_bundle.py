from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from job_utils import (
    build_eval_command,
    build_import_command,
    build_merge_command,
    build_train_command,
    dataset_report,
    format_shell_command,
    load_job_config,
    pip_install_command,
)


def _powershell_with_pythonpath(command: str) -> str:
    return "$env:PYTHONPATH='src'\n" + command + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a portable job bundle with commands for Colab and the return path to helix-proto."
    )
    parser.add_argument("--job-config", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--local-model-dir", default=r"C:\ruta\modelo-merged")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_job_config(args.job_config)
    report = dataset_report(config["dataset_path"])

    bundle_dir = args.output_dir or (Path("finetune") / "bundles" / str(config["job_name"]))
    bundle_dir = bundle_dir.resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    dataset_target = bundle_dir / Path(config["dataset_path"]).name
    shutil.copyfile(config["dataset_path"], dataset_target)

    bundle_config = dict(config)
    bundle_config["dataset_path"] = dataset_target.name
    (bundle_dir / "job.json").write_text(json.dumps(bundle_config, indent=2), encoding="utf-8")

    train_command = build_train_command(config)
    merge_command = build_merge_command(config)
    import_command = build_import_command(config, local_model_dir=args.local_model_dir)
    eval_command = build_eval_command(config, local_model_dir=args.local_model_dir)

    (bundle_dir / "train_colab.txt").write_text(
        "\n".join(
            [
                pip_install_command(str(config["trainer"])),
                format_shell_command(train_command),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if merge_command:
        (bundle_dir / "merge_colab.txt").write_text(
            format_shell_command(merge_command) + "\n",
            encoding="utf-8",
        )
    (bundle_dir / "import_to_helix.ps1").write_text(
        _powershell_with_pythonpath(format_shell_command(import_command)),
        encoding="utf-8",
    )
    (bundle_dir / "eval_in_helix.ps1").write_text(
        _powershell_with_pythonpath(format_shell_command(eval_command)),
        encoding="utf-8",
    )

    summary = {
        "job_name": config["job_name"],
        "bundle_dir": str(bundle_dir),
        "dataset": report,
        "train_command": format_shell_command(train_command),
        "merge_command": format_shell_command(merge_command) if merge_command else None,
        "import_command": format_shell_command(import_command),
        "eval_command": format_shell_command(eval_command),
    }
    (bundle_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
