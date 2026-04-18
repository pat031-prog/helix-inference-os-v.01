from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a PEFT adapter into full weights and save a plain Transformers directory."
    )
    parser.add_argument("--base-model", required=True, help="Base model name or path.")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_source = args.adapter_dir if (args.adapter_dir / "tokenizer_config.json").exists() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    peft_model = PeftModel.from_pretrained(model, args.adapter_dir)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "base_model": args.base_model,
        "adapter_dir": str(args.adapter_dir.resolve()),
        "output_dir": str(output_dir),
    }
    (output_dir / "merge_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    print("Merged model saved. Next step in helix-proto:")
    print(
        "python -m helix_proto.cli prepare-model "
        f"\"{output_dir}\" --alias legal-ft --local-files-only"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
