from __future__ import annotations

import json
import os
import threading
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tools.export_research_site import build_site_dist


def test_build_site_dist_creates_expected_files(tmp_path: Path) -> None:
    output_root = tmp_path / "site-dist"

    build_site_dist(output_root)

    assert (output_root / "index.html").exists()
    assert (output_root / "research.html").exists()
    assert (output_root / "research" / "index.html").exists()
    assert (output_root / "frontier" / "index.html").exists()
    assert (output_root / "app.html").exists()
    assert (output_root / "app" / "index.html").exists()
    assert (output_root / "meta-demo.html").exists()
    assert (output_root / "meta-demo" / "index.html").exists()
    assert (output_root / "meta-demo-real-cached.html").exists()
    assert (output_root / "meta-demo-real-cached" / "index.html").exists()
    assert (output_root / "static" / "research.css").exists()
    assert (output_root / "static" / "research.js").exists()
    assert (output_root / "static" / "home.js").exists()
    assert (output_root / "research-data" / "manifest.json").exists()
    assert (output_root / "research-data" / "artifacts" / "hybrid-memory-frontier-summary.json").exists()


def test_build_site_dist_writes_wrapped_manifest_and_artifacts(tmp_path: Path) -> None:
    output_root = tmp_path / "site-dist"

    build_site_dist(output_root)

    manifest = json.loads((output_root / "research-data" / "manifest.json").read_text(encoding="utf-8"))
    artifact = json.loads(
        (output_root / "research-data" / "artifacts" / "hybrid-memory-frontier-summary.json").read_text(encoding="utf-8")
    )

    names = {item["name"] for item in manifest["artifacts"]}

    assert "hybrid-memory-frontier-summary.json" in names
    assert artifact["name"] == "hybrid-memory-frontier-summary.json"
    assert artifact["title"] == "Hybrid memory frontier summary"
    assert artifact["payload"]["benchmark_kind"] == "hybrid-memory-frontier-summary-v1"


def test_exported_site_serves_static_routes(tmp_path: Path) -> None:
    output_root = tmp_path / "site-dist"
    build_site_dist(output_root)

    previous_cwd = Path.cwd()
    os.chdir(output_root)
    server = ThreadingHTTPServer(("127.0.0.1", 0), SimpleHTTPRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        for path in ["/", "/research", "/frontier", "/app", "/meta-demo", "/meta-demo-real-cached", "/research-data/manifest.json"]:
            with urllib.request.urlopen(f"{base}{path}") as response:
                assert response.status == 200
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
        os.chdir(previous_cwd)
