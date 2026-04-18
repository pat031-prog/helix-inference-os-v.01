import json
from pathlib import Path

from finetune.job_utils import (
    build_merge_command,
    build_train_command,
    dataset_report,
    load_job_config,
)


def test_dataset_report_accepts_output_key(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"instruction": "Hola", "output": "Mundo"}),
                json.dumps({"instruction": "Foo", "output": "Bar"}),
            ]
        ),
        encoding="utf-8",
    )

    report = dataset_report(dataset)

    assert report["rows"] == 2
    assert report["keys"] == ["instruction", "output"]


def test_load_job_config_resolves_relative_dataset_path(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "datasets").mkdir(parents=True)
    (repo_root / "datasets" / "seed.jsonl").write_text(
        json.dumps({"instruction": "Hola", "output": "Mundo"}) + "\n",
        encoding="utf-8",
    )
    job_path = repo_root / "job.json"
    job_path.write_text(
        json.dumps(
            {
                "job_name": "demo",
                "trainer": "unsloth",
                "model_name": "demo/model",
                "dataset_path": "datasets/seed.jsonl",
                "output_dir": "/content/out/demo",
                "system_prompt": "test",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("finetune.job_utils.REPO_ROOT", repo_root)
    config = load_job_config(job_path)

    assert config["dataset_path"] == str((repo_root / "datasets" / "seed.jsonl").resolve())


def test_build_train_and_merge_commands() -> None:
    config = {
        "job_name": "legal-qwen15b-trl",
        "trainer": "trl",
        "model_name": "Qwen/Qwen2.5-1.5B",
        "dataset_path": "C:/tmp/legal.jsonl",
        "output_dir": "/content/out/legal",
        "merged_output_dir": "/content/out/legal-merged",
        "system_prompt": "legal",
        "max_seq_length": 1024,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "num_train_epochs": 4,
        "max_steps": -1,
        "warmup_steps": 5,
        "logging_steps": 5,
        "save_steps": 20,
        "seed": 3407,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": "q_proj,k_proj",
    }

    train_command = build_train_command(config)
    merge_command = build_merge_command(config)

    assert "--model-name" in train_command
    assert "Qwen/Qwen2.5-1.5B" in train_command
    assert merge_command is not None
    assert "--adapter-dir" in merge_command
