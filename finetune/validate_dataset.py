from __future__ import annotations

import argparse
import json
from pathlib import Path

from job_utils import dataset_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a JSONL fine-tuning dataset.")
    parser.add_argument("dataset_path", type=Path)
    return parser.parse_args()


def main() -> int:
    report = dataset_report(parse_args().dataset_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
