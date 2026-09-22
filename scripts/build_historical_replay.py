"""Build signed historical causal-replay artifacts without Streamlit/SQLite."""
from __future__ import annotations

import argparse
from collections import Counter
from io import StringIO
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio_tracker.analytics.cross_correlation import PEERS
from portfolio_tracker.analytics.historical_replay import (
    build_historical_replay,
    write_historical_replay,
)
from portfolio_tracker.analytics.replay import ReplayDataset
from portfolio_tracker.config import DATA_DIR
from portfolio_tracker.services.quant_market_data import normalize_symbol
from portfolio_tracker.services.zone_forward import digest


def _signed_frame(path: Path):
    if not path.is_file():
        raise ValueError(f"No existe el histórico firmado: {path}")
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("sha256") != digest(payload):
        raise ValueError(f"Firma SHA-256 inválida: {path}")
    frame = pd.read_json(StringIO(payload["frame"]), orient="table")
    return frame, pd.Timestamp(payload["fetched_at"])


def load_dataset(cache_dir: Path, symbol: str, as_of=None) -> ReplayDataset:
    root = cache_dir / symbol
    intraday, intraday_at = _signed_frame(root / "5m.json")
    daily, daily_at = _signed_frame(root / "daily.json")
    available_at = min(intraday_at, daily_at)
    cutoff = available_at if as_of is None else pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    available_at = available_at.tz_localize("UTC") if available_at.tzinfo is None else available_at.tz_convert("UTC")
    if cutoff > available_at:
        raise ValueError(f"{symbol}: --as-of supera la adquisición firmada.")
    return ReplayDataset(intraday, daily, cutoff)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Replay causal a las 11:00 NY. Produce JSON firmado; nunca escribe "
            "en tablas live, contabilidad u órdenes."
        )
    )
    parser.add_argument("--symbols", nargs="+", default=["SMCI", "NVDA"])
    parser.add_argument("--cache-dir", type=Path, default=DATA_DIR / "autopilot" / "market")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "historical_replay")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--as-of")
    parser.add_argument("--max-cuts", type=int)
    parser.add_argument("--stop-atr-multiple", type=float, default=2.25)
    args = parser.parse_args(argv)

    symbols = tuple(dict.fromkeys(normalize_symbol(value) for value in args.symbols))
    datasets = {
        symbol: load_dataset(args.cache_dir.resolve(), symbol, args.as_of)
        for symbol in symbols
    }
    outputs = []
    for symbol, dataset in datasets.items():
        peer = datasets.get(PEERS.get(symbol, ""))
        payload = build_historical_replay(
            symbol,
            dataset,
            peer_dataset=peer,
            parameters={
                "minimum_probability": 0.55,
                "stop_atr_multiple": args.stop_atr_multiple,
                "risk_per_trade_pct": 1.0,
            },
            start=args.start,
            end=args.end,
            max_cuts=args.max_cuts,
        )
        destination = args.output_dir.resolve() / f"{symbol}_{payload['replay_id'][:16]}.json"
        write_historical_replay(payload, destination)
        resolved = Counter(
            horizon["operational_result"]["outcome"]
            for cut in payload["observations"]
            for horizon in cut["horizons"]
            if horizon["operational_result"]["status"] == "RESOLVED"
        )
        outputs.append({
            "symbol": symbol,
            "artifact": str(destination),
            "candidate_cuts": payload["candidate_cuts"],
            "completed_cuts": len(payload["observations"]),
            "rejected_cuts": sum(payload["rejected"].values()),
            "operational_outcomes": dict(sorted(resolved.items())),
            "content_sha256": payload["content_sha256"],
        })
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
