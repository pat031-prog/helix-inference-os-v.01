from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from helix_proto.artifact_replay import diff_reports, replay_cassette, verify_artifact_file


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} root must be a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay or verify HeliX evidence bundles without exposing secrets.")
    parser.add_argument("--mode", choices=["verify-only", "cassette", "live", "diff"], default="verify-only")
    parser.add_argument("--artifact", type=Path, help="Artifact JSON for verify-only/live modes.")
    parser.add_argument("--cassette", type=Path, help="Sanitized replay cassette for cassette mode.")
    parser.add_argument("--left", type=Path, help="Left JSON report/artifact for diff mode.")
    parser.add_argument("--right", type=Path, help="Right JSON report/artifact for diff mode.")
    parser.add_argument("--perturb-seed", default=None)
    parser.add_argument("--allow-live", action="store_true", help="Acknowledge live replay may call providers; not implemented in this safety batch.")
    args = parser.parse_args()

    if args.mode == "verify-only":
        if not args.artifact:
            parser.error("--artifact is required for verify-only")
        report = verify_artifact_file(args.artifact)
    elif args.mode == "cassette":
        if not args.cassette:
            parser.error("--cassette is required for cassette mode")
        report = replay_cassette(_load_json(args.cassette), perturb_seed=args.perturb_seed)
    elif args.mode == "diff":
        if not args.left or not args.right:
            parser.error("--left and --right are required for diff mode")
        report = diff_reports(_load_json(args.left), _load_json(args.right))
    else:
        if not args.allow_live:
            report = {
                "artifact": "helix-live-replay-report",
                "replay_mode": "live",
                "status": "skipped_requires_allow_live",
                "claim_boundary": "Live replay can introduce provider drift and is disabled unless explicitly requested.",
            }
        else:
            report = {
                "artifact": "helix-live-replay-report",
                "replay_mode": "live",
                "status": "not_implemented_in_privacy_safe_batch",
                "claim_boundary": "Live replay will require an explicit safe cassette/provider adapter policy before use.",
            }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report.get("status") in {"failed", "drift_detected"} else 0


if __name__ == "__main__":
    sys.exit(main())
