from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (REPO_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.run_agent_run_transparency_gauntlet_v1 import verify_standalone_bundle  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a HeliX agent-run attestation bundle JSON.")
    parser.add_argument("bundle_json", help="Path to a helix-agent-run-verifier-bundle-v0 JSON file.")
    args = parser.parse_args(argv)

    path = Path(args.bundle_json)
    payload = json.loads(path.read_text(encoding="utf-8"))
    bundle = payload.get("standalone_bundle") if isinstance(payload, dict) and isinstance(payload.get("standalone_bundle"), dict) else payload
    result = verify_standalone_bundle(bundle)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
