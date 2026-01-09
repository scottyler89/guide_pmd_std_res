from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure():
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root in sys.path:
        sys.path.remove(repo_root)
    sys.path.insert(0, repo_root)

