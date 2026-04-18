from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_SYSTEM_PROMPT = (
    "Sos un asistente de informacion juridica general para Argentina. "
    "Responde con cautela, lenguaje claro, estructura ordenada y aclarando "
    "que no reemplaza asesoramiento profesional."
)

DEFAULT_TARGET_MODULES = (
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,"
    "c_attn,c_proj,fc1,fc2,dense,dense_h_to_4h,dense_4h_to_h"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA SFT training scaffold for Colab using Unsloth."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default=DEFAULT_TARGET_MODULES)
    return parser.parse_args()


def render_example(example: dict[str, str], fallback_system_prompt: str) -> str:
    system_prompt = str(example.get("system") or fallback_system_prompt).strip()
    instruction = str(example.get("instruction") or "").strip()
    context = str(example.get("input") or example.get("context") or "").strip()
    response = str(example.get("response") or example.get("output") or "").strip()
    if not instruction or not response:
        raise ValueError("each dataset row must include instruction and response/output")

    parts = [f"### Sistema\n{system_prompt}", f"### Instruccion\n{instruction}"]
    if context:
        parts.append(f"### Contexto\n{context}")
    parts.append(f"### Respuesta\n{response}")
    return "\n\n".join(parts)


def build_dataset(dataset_path: Path, system_prompt: str):
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    return dataset.map(
        lambda row: {"text": render_example(row, system_prompt)},
        remove_columns=dataset.column_names,
    )


def main() -> int:
    args = parse_args()

    import torch
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    output_dir = args.output_dir.resolve()
    adapter_dir = output_dir / "adapter"
    merged_dir = output_dir / "merged_16bit"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            module.strip()
            for module in args.target_modules.split(",")
            if module.strip()
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    dataset = build_dataset(args.dataset_path, args.system_prompt)

    train_args = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        gradient_checkpointing=True,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        optim="adamw_8bit",
        seed=args.seed,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    trainer.model.save_pretrained_merged(
        str(merged_dir),
        tokenizer,
        save_method="merged_16bit",
    )

    metadata = {
        "trainer": "unsloth",
        "base_model": args.model_name,
        "dataset_path": str(args.dataset_path),
        "output_dir": str(output_dir),
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir),
        "system_prompt": args.system_prompt,
        "target_modules": [module.strip() for module in args.target_modules.split(",") if module.strip()],
    }
    (output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    print("Training finished. Adapter saved to:", adapter_dir)
    print("Merged 16-bit model saved to:", merged_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
