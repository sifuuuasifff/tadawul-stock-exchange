"""
UPDATE AND PUSH — One command to update the live portal.
Run this after any batch, data refresh, or forecast update.

Usage:
    python update_and_push.py
    python update_and_push.py "Added Batch 4 telecom stocks"
"""

import sys
import subprocess
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent

REMOTE_URL = "https://github.com/sifuuuasifff/tadawul-stock-exchange.git"


def run(cmd: str, cwd=BASE) -> bool:
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")
    if result.returncode != 0 and result.stderr.strip():
        print(f"  ERROR: {result.stderr.strip()}")
    return result.returncode == 0


def main():
    msg = sys.argv[1] if len(sys.argv) > 1 else f"Update {datetime.now().strftime('%Y-%m-%d %H:%M')} — data refresh"

    print(f"\nUPDATE AND PUSH — Tadawul Stock Memory Engine")
    print(f"Message : {msg}")
    print(f"Time    : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("─" * 50)

    # Step 1: Rebuild current forecasts (v3 = enriched fundamentals + valuation)
    print("\n[1] Rebuilding current forecasts for all stocks (enriched v3)...")
    run("python build_all_forecasts_v3.py")

    # Step 2: Stage all updated files
    print("\n[2] Staging updated files...")
    run("git add portal.py memory/ reports/ data/processed/ config/ phases/ build_all_forecasts.py batch_*.py")

    # Step 3: Check if there's anything to commit
    result = subprocess.run("git status --short", shell=True, cwd=BASE, capture_output=True, text=True)
    if not result.stdout.strip():
        print("\n  Nothing changed — portal is already up to date.")
        return

    changes = len(result.stdout.strip().split("\n"))
    print(f"  {changes} files changed")

    # Step 4: Commit
    print("\n[3] Committing...")
    run(f'git commit -m "{msg}"')

    # Step 5: Push
    print("\n[4] Pushing to GitHub → Streamlit Cloud will redeploy automatically...")
    run(f"git remote set-url origin \"{REMOTE_URL}\"")
    success = run("git push origin main")

    if success:
        print("\n" + "=" * 50)
        print("  DONE — Portal will update in ~2 minutes")
        print("  URL: https://tadawul-stock-exchange.streamlit.app/")
        print("=" * 50)
    else:
        print("\n  Push failed — check your internet connection and try again")


if __name__ == "__main__":
    main()
