"""Download ~20 years of daily bars and run the daily factor study.

Research only: no portfolio.db access, no orders, no change to the live gate.
Prices are cached locally (gitignored) with a SHA-256 manifest; the signed
report goes to output/daily_study/.

    python scripts/run_daily_factor_study.py            # download (cached per day) and run
    python scripts/run_daily_factor_study.py --offline  # reuse the newest cache
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio_tracker.analytics import daily_factor_study as study  # noqa: E402

OUT = ROOT / "output" / "daily_study"
DATA = OUT / "data"
BENCHMARK = "SPY"
CASH = "^IRX"
START = "2004-01-01"
# Fixed before looking at results: today's large, liquid US names across
# sectors. They are survivors (see report), so momentum is judged against the
# same universe equally weighted.
UNIVERSE = (
    "AAPL MSFT AMZN GOOGL META NVDA TSLA BRK-B JPM JNJ V PG UNH HD MA XOM CVX PFE "
    "KO PEP MRK ABBV AVGO COST WMT DIS CSCO INTC ORCL ADBE CRM NFLX AMD QCOM TXN "
    "IBM BA CAT GE MMM HON UPS LMT RTX DE MCD NKE SBUX LOW TGT BAC WFC C GS MS AXP "
    "BLK SCHW USB PNC T VZ CMCSA TMO ABT DHR BMY AMGN GILD LLY MDT CVS CI MO PM CL "
    "KMB GIS MDLZ NEE DUK SO D EXC COP SLB EOG OXY F GM MU AMAT LRCX KLAC ADI MCHP SMCI"
).split()
VOL_SYMBOLS = ("SPY", "SMCI", "NVDA")
# Provider artefacts confirmed by inspection (not by strategy results).
PRICE_BREAKS = (
    ("DHR", "2016-07-05", "Fortive spin-off adjusted upward by the provider; true holder return ~0"),
)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def download() -> Path:
    import yfinance as yf
    DATA.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    target = DATA / f"closes_{stamp}.csv.gz"
    if target.exists() and (target.with_suffix(".sha256")).exists():
        return target
    symbols = sorted({*UNIVERSE, BENCHMARK, CASH})
    raw = yf.download(symbols, start=START, auto_adjust=True, progress=False,
                      threads=True, group_by="column")
    closes = raw["Close"].dropna(how="all")
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None)
    missing = [s for s in symbols if s not in closes or closes[s].notna().sum() == 0]
    if BENCHMARK in missing:
        raise RuntimeError("No se descargó el benchmark; no se ejecuta el estudio.")
    closes.to_csv(target, compression="gzip")
    target.with_suffix(".sha256").write_text(_sha_file(target), encoding="utf-8")
    (DATA / f"closes_{stamp}.missing.json").write_text(json.dumps(missing), encoding="utf-8")
    return target


def newest_cache() -> Path:
    files = sorted(DATA.glob("closes_*.csv.gz"))
    if not files:
        raise FileNotFoundError("No hay caché de precios; ejecuta sin --offline.")
    return files[-1]


def load(path: Path) -> pd.DataFrame:
    expected = path.with_suffix(".sha256").read_text(encoding="utf-8").strip()
    if _sha_file(path) != expected:
        raise ValueError(f"Caché de precios alterada: {path.name}")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    source = newest_cache() if args.offline else download()
    closes = load(source)
    # ^IRX is an annualized 13-week T-bill yield in percent; a rate, so short
    # holiday gaps may be carried forward (never done for prices).
    cash = (closes.pop(CASH).ffill(limit=5) / 100.0 / study.TRADING_DAYS).fillna(0.0) \
        if CASH in closes else pd.Series(0.0, index=closes.index)
    universe = [s for s in UNIVERSE if s in closes and closes[s].notna().sum() > study.TRADING_DAYS * 2]
    prices, corrections = study.neutralize_price_breaks(closes[[*universe, BENCHMARK]], PRICE_BREAKS)
    config = study.StudyConfig()
    strategies = study.run_strategy_study(prices, BENCHMARK, cash, config)
    returns = prices.pct_change(fill_method=None)
    volatility = {}
    for symbol in VOL_SYMBOLS:
        if symbol in returns:
            volatility[symbol] = study.volatility_study(returns[symbol], holdout_start=config.holdout_start)
    sizing = {}
    for symbol, item in volatility.items():
        daily = (item.get("current") or {}).get("forecast_daily_vol")
        if daily:
            sizing[symbol] = {
                "forecast_daily_vol": daily,
                "forecast_annual_vol": daily * math.sqrt(study.TRADING_DAYS),
                "max_position_per_1000_at_1pct_99_risk": study.risk_sized_position(1000, daily),
            }
    report = {
        "version": "DAILY_FACTOR_STUDY_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "price_cache": source.name, "price_cache_sha256": _sha_file(source),
        "benchmark": BENCHMARK, "cash_proxy": CASH,
        "universe_requested": len(UNIVERSE), "universe_used": universe,
        "survivorship_bias": "Universe = current large caps; delisted names absent. "
                             "Momentum is judged against the same universe equally weighted.",
        "parameters_fixed_before_results": True,
        "price_corrections": corrections,
        "strategies": strategies, "volatility": volatility, "sizing_examples": sizing,
        "portfolio_db_accessed": False, "production_model_changed": False,
    }
    report = json.loads(json.dumps(report, default=float))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("report_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
    path.write_text(json.dumps({"payload": report, "sha256": _sha(report)}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(path)
    for period, block in strategies["periods"].items():
        print(f"\n== {period} {block['start']} → {block['end']}")
        for name, row in block["strategies"].items():
            print(f"{name:24s} CAGR {row['cagr']:+.2%}  vol {row['volatility']:.1%}  "
                  f"Sharpe {row['sharpe'] or 0:+.2f}  MaxDD {row['max_drawdown']:.1%}  "
                  f"turnover/yr {row['turnover_per_year']:.1f}  costos/yr {row['cost_drag_per_year']:.2%}")
        for name, row in block["comparisons"].items():
            print(f"  {name:45s} Δret {row['annual_return_difference']:+.2%} "
                  f"IC95 [{row['annual_return_difference_95'][0]:+.2%}, {row['annual_return_difference_95'][1]:+.2%}]  "
                  f"ΔSharpe {row['sharpe_difference']:+.2f} "
                  f"IC95 [{row['sharpe_difference_95'][0]:+.2f}, {row['sharpe_difference_95'][1]:+.2f}]")
    print("\n== Volatilidad (QLIKE, menor es mejor)")
    for symbol, item in volatility.items():
        print(symbol, json.dumps({k: item.get(k) for k in ("qlike_all", "qlike_holdout", "years_beating_naive", "years_evaluated")}))
    print("\n== Dimensionamiento", json.dumps(sizing, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
