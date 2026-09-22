"""Point-in-time replay evidence for one signed six-horizon execution.

The existing immutable V3 forecast/outcome rows remain the source of truth.
This module adds a common identity, a compact feature snapshot and verifiable
input archive references *before* those rows are signed. No ledger migration.
"""

from __future__ import annotations

from datetime import date, datetime
from dataclasses import asdict, is_dataclass
from decimal import Decimal
from enum import Enum
from functools import lru_cache
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .model_observations import canonical, utc_timestamp

FEATURE_SCHEMA_VERSION = 1
FEATURE_NAMES = (
    "last_price", "probability_up", "probability_down", "operation_probability",
    "atr_5m", "stochastic_k", "stochastic_d", "volume_ratio", "volume_confirmed",
    "bollinger_upper", "bollinger_middle", "bollinger_lower", "vwap",
    "price_vs_vwap_pct", "above_vwap", "adx", "range_market",
    "ema9", "ema21", "ema50", "ema200", "macd_5m", "macd_signal_5m",
    "macd_histogram_5m", "macd_daily", "macd_signal_daily",
    "macd_histogram_daily", "nearest_support", "structural_support",
    "structural_resistance", "weekly_support", "weekly_resistance",
    "market_regime", "macro_permission", "macro_trending", "weekly_trend",
    "monthly_trend", "daily_trend", "risk_veto", "fundamental_risk_veto",
    "fundamental_score", "fundamental_label", "event_risk_level",
    "activation_trigger_met", "activation_trigger", "position_state",
    "position_management", "execution_plan_label", "tactical_short",
    "exposure_factor", "signal", "daily_trend", "ichimoku_5m",
    "ichimoku_daily", "candle_pattern", "candle_detail", "fibonacci",
    "annual_fibonacci", "pivots", "chart_patterns", "chart_pattern_impact",
    "chart_pattern_veto", "score_breakdown", "risk_reasons", "risk_alert",
    "scenario", "fundamental_reasons", "fundamental_news_audit",
    "fundamental_as_of", "event_risk_window_until", "hierarchy_detail",
    "buy_levels", "sell_levels", "cross_asset_context",
)
CODE_FILES = (
    "portfolio_tracker/analytics/technical_probability.py",
    "portfolio_tracker/analytics/multi_timeframe.py",
    "portfolio_tracker/analytics/closed_bars.py",
    "portfolio_tracker/analytics/causal_core.py",
    "portfolio_tracker/analytics/technical_validity.py",
    "portfolio_tracker/analytics/decision_engines.py",
    "portfolio_tracker/analytics/chart_patterns.py",
    "portfolio_tracker/analytics/cross_correlation.py",
    "portfolio_tracker/analytics/fundamental_news.py",
    "portfolio_tracker/analytics/probability_calibration.py",
    "portfolio_tracker/analytics/operational_target.py",
    "portfolio_tracker/analytics/historical_replay.py",
    "portfolio_tracker/analytics/horizon_models.py",
    "portfolio_tracker/analytics/nested_walk_forward.py",
    "portfolio_tracker/analytics/operational_calibration.py",
    "portfolio_tracker/analytics/net_expectation.py",
    "portfolio_tracker/analytics/backtesting.py",
    "portfolio_tracker/services/scenario_calibration.py",
    "portfolio_tracker/services/model_observations.py",
    "portfolio_tracker/services/directional_collection.py",
    "portfolio_tracker/services/cross_asset.py",
    "portfolio_tracker/services/operational_state.py",
    "portfolio_tracker/services/operational_model_registry.py",
    "portfolio_tracker/services/decision_engine.py",
    "portfolio_tracker/services/quant_market_data.py",
    "portfolio_tracker/services/model_execution_record.py",
    "scripts/autopilot_runtime.py", "scripts/autopilot_market_cache.py",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        if isinstance(value, (datetime, pd.Timestamp)) and value.tzinfo is not None:
            return utc_timestamp(value).isoformat()
        # Daily indicator indices are exchange-session labels, not instants.
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, Decimal)):
        number = float(value)
        return number if math.isfinite(number) else None
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if value is pd.NA:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    raise ValueError(f"Feature no serializable: {type(value).__name__}")


@lru_cache(maxsize=1)
def code_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in CODE_FILES}


def execution_id(symbol: str, session_date: str, protocol: str) -> str:
    identity = {"symbol": symbol.strip().upper(), "session_date": session_date, "protocol": protocol}
    return hashlib.sha256(canonical(identity).encode("utf-8")).hexdigest()


def _indicator_tail(analysis) -> dict[str, Any]:
    tails = {}
    for name in ("intraday_indicators", "hourly_indicators", "four_hour_indicators",
                 "daily_indicators", "weekly_indicators", "monthly_indicators"):
        frame = getattr(analysis, name, None)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        tails[name] = {
            "at": _jsonable(pd.Timestamp(frame.index[-1])),
            "values": _jsonable(frame.iloc[-1].to_dict()),
        }
    return tails


def build_replay_snapshot(analysis, *, observed_at: datetime, protocol: str,
                          input_artifacts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    observed = utc_timestamp(observed_at)
    source = utc_timestamp(analysis.source_bar_closed_at)
    session_date = observed.tz_convert("America/New_York").date().isoformat()
    artifacts = _jsonable(dict(input_artifacts or {}))
    features = {name: _jsonable(getattr(analysis, name, None)) for name in FEATURE_NAMES}
    snapshot = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "run_id": execution_id(analysis.symbol, session_date, protocol),
        "symbol": analysis.symbol.strip().upper(),
        "session_date": session_date,
        "collection_protocol": protocol,
        "observed_at": observed.isoformat(),
        "source_bar_closed_at": source.isoformat(),
        "features": features,
        "indicator_tail": _indicator_tail(analysis),
        "fundamental_snapshot_sha256": str(getattr(analysis, "fundamental_snapshot_sha256", "") or ""),
        "input_artifacts": artifacts,
        "code_sha256": code_fingerprints(),
        # Peer OHLCV is not archived by the current cross-asset cache. The
        # derived peer context is frozen, but full bitwise replay is not claimed.
        "replay_level": "ARCHIVED_PRIMARY_INPUTS_WITH_PEER_FEATURE_SNAPSHOT"
        if set(artifacts) == {"5m", "1d"} else "FEATURE_SNAPSHOT_ONLY",
    }
    canonical(snapshot)  # reject malformed/non-finite nested values before any DB write
    return snapshot


def prediction_snapshot(horizon) -> dict[str, Any]:
    names = (
        "label", "engine_name", "probability_up", "probability_range",
        "probability_down", "bias", "bullish_target", "range_low",
        "range_high", "bearish_target", "atr_value", "local_support",
        "local_resistance", "probability_status", "calibration_samples",
        "brier_score",
    )
    return {name: _jsonable(getattr(horizon, name, None)) for name in names}
