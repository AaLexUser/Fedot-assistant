from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run Streamlit against ``app.py`` in the application root (sibling of ``frontend/``)."""
    app_py = Path(__file__).resolve().parent.parent / "app.py"
    if not app_py.is_file():
        msg = f"Streamlit entry file not found at {app_py}"
        raise FileNotFoundError(msg)
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_py)]
    try:
        rc = subprocess.call(cmd)
    except KeyboardInterrupt:
        # Ctrl+C while waiting on the child: avoid a traceback from wait().
        rc = 130
    raise SystemExit(rc)
