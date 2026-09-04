"""collect: headless entry point for Windows Task Scheduler."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.autopilot_runtime import cli

if __name__ == "__main__":
    raise SystemExit(cli("collect"))

