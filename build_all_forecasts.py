"""
build_all_forecasts.py — redirects to the enriched v3 version.
All stock names and codes are now maintained in shared.py only.
This file kept for backward compatibility only.
"""
import subprocess, sys
from pathlib import Path

if __name__ == "__main__":
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "build_all_forecasts_v3.py")],
        cwd=str(Path(__file__).parent)
    )
    sys.exit(result.returncode)
