from __future__ import annotations

import os
from pathlib import Path

from helix_proto.api import serve_api


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    workspace_root = os.environ.get("HELIX_WORKSPACE_ROOT")
    root = Path(workspace_root).resolve() if workspace_root else None
    serve_api(host=host, port=port, root=root)


if __name__ == "__main__":
    main()
