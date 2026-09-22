"""Run nested embargoed walk-forward validation on signed replay artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portfolio_tracker.analytics.nested_walk_forward import (  # noqa: E402
    WalkForwardConfig,
    run_nested_walk_forward_replay,
    write_nested_walk_forward_artifact,
)


def _sources(values):
    paths = [Path(value).resolve() for value in values] if values else sorted(
        (ROOT / "output" / "historical_replay").glob("*.json")
    )
    if not paths:
        raise FileNotFoundError("No hay artefactos de replay causal para validar.")
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("Uno o más artefactos de replay no existen.")
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Walk-forward anidado, embargo XNYS y holdout final sellado."
    )
    parser.add_argument("inputs", nargs="*")
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "nested_walk_forward"))
    args = parser.parse_args(argv)
    output = Path(args.output_dir).resolve()
    for source in _sources(args.inputs):
        replay = json.loads(source.read_text(encoding="utf-8"))
        artifact = run_nested_walk_forward_replay(replay, WalkForwardConfig())
        destination = output / f"{artifact['symbol']}_{artifact['validation_run_id'][:16]}.json"
        write_nested_walk_forward_artifact(artifact, destination)
        statuses = ", ".join(
            f"{row['horizon']}={row['status']}" for row in artifact["results"]
        )
        print(f"{artifact['symbol']}: {destination}")
        print(f"  {statuses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
