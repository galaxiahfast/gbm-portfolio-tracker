"""Wake-marker action for Task Scheduler; never acquires or fabricates a cut.

The collector's existing one-minute task performs the actual guarded capture.
This task only wakes a sleeping Windows host shortly before a registered cut.
"""
from datetime import datetime, timezone
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parents[1] / "output" / "prospective_archive"


def main():
    folder = ARCHIVE / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "wake.log").open("a", encoding="utf-8") as log:
        log.write(f"{datetime.now(timezone.utc).isoformat()} scheduler_wake_marker\n")


if __name__ == "__main__":
    main()
