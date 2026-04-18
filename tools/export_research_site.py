from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helix_proto.research_artifacts import research_artifact_manifest, wrapped_research_artifact  # noqa: E402


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_site_dist(output_root: Path, repo_root: Path | None = None) -> Path:
    root = (repo_root or REPO_ROOT).resolve()
    web_root = root / "web"
    verification_root = root / "verification"

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    _copy_file(web_root / "index.html", output_root / "index.html")
    _copy_file(web_root / "research.html", output_root / "research.html")
    _copy_file(web_root / "research.html", output_root / "research" / "index.html")
    _copy_file(web_root / "research.html", output_root / "frontier" / "index.html")
    _copy_file(web_root / "app.html", output_root / "app.html")
    _copy_file(web_root / "app.html", output_root / "app" / "index.html")
    meta_demo = web_root / "meta-demo.html"
    if meta_demo.exists():
        _copy_file(meta_demo, output_root / "meta-demo.html")
        _copy_file(meta_demo, output_root / "meta-demo" / "index.html")
    real_cached_demo = web_root / "meta-demo-real-cached.html"
    if real_cached_demo.exists():
        _copy_file(real_cached_demo, output_root / "meta-demo-real-cached.html")
        _copy_file(real_cached_demo, output_root / "meta-demo-real-cached" / "index.html")

    static_root = web_root / "static"
    for item in static_root.rglob("*"):
        if item.is_file():
            _copy_file(item, output_root / "static" / item.relative_to(static_root))

    manifest = {"artifacts": research_artifact_manifest(verification_root=verification_root)}
    research_data_root = output_root / "research-data"
    artifacts_root = research_data_root / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    (research_data_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for artifact_meta in manifest["artifacts"]:
        name = str(artifact_meta["name"])
        payload = wrapped_research_artifact(name, verification_root=verification_root)
        (artifacts_root / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return output_root


def main() -> None:
    output_root = REPO_ROOT / "site-dist"
    built_root = build_site_dist(output_root)
    print(f"Exported research site to {built_root}")


if __name__ == "__main__":
    main()
