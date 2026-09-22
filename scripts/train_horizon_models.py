"""Train isolated regularized models from signed historical replay artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portfolio_tracker.analytics.horizon_models import (  # noqa: E402
    TrainingConfig,
    train_replay_horizon_models,
    write_horizon_model_artifact,
)


def _inputs(values: list[str]) -> list[Path]:
    if values:
        paths = [Path(value).resolve() for value in values]
    else:
        paths = sorted((ROOT / "output" / "historical_replay").glob("*.json"))
    if not paths:
        raise FileNotFoundError("No hay artefactos de replay histórico para entrenar.")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Artefactos inexistentes: {missing}")
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Entrena seis modelos L2 independientes por símbolo y horizonte."
    )
    parser.add_argument("inputs", nargs="*", help="JSON de replay; por defecto usa output/historical_replay/*.json")
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "horizon_models"))
    parser.add_argument("--minimum-samples", type=int, default=300)
    parser.add_argument("--minimum-class-samples", type=int, default=20)
    parser.add_argument("--minimum-holdout-samples", type=int, default=60)
    args = parser.parse_args(argv)
    config = TrainingConfig(
        minimum_samples=args.minimum_samples,
        minimum_class_samples=args.minimum_class_samples,
        minimum_holdout_samples=args.minimum_holdout_samples,
    )
    destination = Path(args.output_dir).resolve()
    for source in _inputs(args.inputs):
        replay = json.loads(source.read_text(encoding="utf-8"))
        artifact = train_replay_horizon_models(replay, config)
        target = destination / f"{artifact['symbol']}_{artifact['training_run_id'][:16]}.json"
        write_horizon_model_artifact(artifact, target)
        statuses = ", ".join(f"{row['horizon']}={row['status']}" for row in artifact["models"])
        print(f"{artifact['symbol']}: {target}")
        print(f"  {statuses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
