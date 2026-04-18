from __future__ import annotations

import argparse
import json
import subprocess

from job_utils import (
    build_merge_command,
    build_train_command,
    dataset_report,
    format_shell_command,
    load_job_config,
    pip_install_command,
    render_colab_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or print one Colab fine-tuning job config.")
    parser.add_argument("--job-config", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--print-colab", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_job_config(args.job_config)
    report = dataset_report(config["dataset_path"])
    train_command = build_train_command(config)
    merge_command = build_merge_command(config)

    payload = {
        "job_name": config["job_name"],
        "trainer": config["trainer"],
        "model_name": config["model_name"],
        "dataset": report,
        "pip_install": pip_install_command(str(config["trainer"])),
        "train_command": format_shell_command(train_command),
        "merge_command": format_shell_command(merge_command) if merge_command else None,
    }

    if args.print_colab:
        print(render_colab_plan(config))
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    if not args.execute:
        return 0

    subprocess.run(train_command, check=True)
    if merge_command:
        subprocess.run(merge_command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
