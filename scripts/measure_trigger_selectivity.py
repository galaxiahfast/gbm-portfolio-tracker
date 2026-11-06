"""Describe asynchronous gates and clock-specific volume. No orders, SQL or fits.

This adds measurement artifacts only; production and the frozen replay stay
untouched. A shadow temporal window is NOT an authorization or a new replay.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_trigger_replay import OUTPUT, INPUT, TIMES, GATE_NAMES, digest_file
from scripts.build_historical_replay import load_dataset
from scripts.download_massive_history import read_signed, sha, write_signed
from portfolio_tracker.analytics.technical_probability import add_intraday_indicators
from portfolio_tracker.analytics.technical_validity import current_indicator_suffix
from portfolio_tracker.analytics.causal_core import INTRADAY_READY, trigger_momentum

VERSION = "TRIGGER_SELECTIVITY_MEASUREMENT_V1"
DEFAULT_OUTPUT = OUTPUT / "selectivity_v1"
CONFIRMATIONS = ("volume", "stoch_cross", "macd", "previous_bar_break")
LOOKBACK = 60  # Diagnostic censoring window only, NEVER an operational parameter.
SOURCES = {
    "gbm_short": {"url": "https://gbm.com/trading/", "checked_date": "2026-09-26",
                  "claim": "Trading USA: no posibilidad de abrir posiciones cortas propias; préstamo de acciones no es autorización SHORT."},
    "gbm_commission": {"url": "https://gbm.com/faqs/existe-alguna-comision-por-invertir-en-trading-usa",
                       "checked_date": "2026-09-26", "claim": "Tarifa pública actual: 0.25% por compra o venta. No prueba tarifa ni cargo observado en 2025."},
    "massive_quotes": {"url": "https://massive.com/pricing", "checked_date": "2026-09-26",
                       "claim": "Basic incluye agregados por minuto, no Quotes; Quotes aparece en Advanced USD199/mes. No se contrató."},
}


def confirmation_frame(bars, side):
    """The exact inequalities of trigger_momentum, on every closed bar."""
    prior = bars.shift(1)
    volume_mean = bars.Volume.shift(1).rolling(20, min_periods=20).mean()
    volume = (bars.Volume > volume_mean) & (volume_mean > 0)
    if side == "LONG":
        cross = (prior.StochRSI_K <= prior.StochRSI_D) & (bars.StochRSI_K > bars.StochRSI_D)
        macd = bars.MACD >= bars.MACD_signal
        breakout = bars.Close >= prior.High
    elif side == "SHORT":
        cross = (prior.StochRSI_K >= prior.StochRSI_D) & (bars.StochRSI_K < bars.StochRSI_D)
        macd = bars.MACD <= bars.MACD_signal
        breakout = bars.Close <= prior.Low
    else:
        raise ValueError("Dirección no reconocida.")
    result = pd.DataFrame({"volume": volume, "stoch_cross": cross,
                           "macd": macd, "previous_bar_break": breakout}, index=bars.index)
    result.iloc[:21] = False  # Shared core requires at least 22 valid suffix bars.
    return result


def lag_measurement(confirmations, cut, *, lookback=LOOKBACK):
    """Latest true bar per condition; no cross-session or future carryover.

    span is the distance between first/last of those confirmations. Age is
    ALSO reported: a compact sequence hours ago cannot qualify a fresh cut.
    A k-bar shadow requires all confirmations among [cut-k*5m, cut], inclusive.
    """
    cut = pd.Timestamp(cut).tz_convert("UTC")
    times = confirmations.index.tz_convert("UTC") + pd.Timedelta(minutes=5)
    dates = times.tz_convert("America/New_York").date
    same_day = dates == cut.tz_convert("America/New_York").date()
    lower = cut - pd.Timedelta(minutes=lookback * 5)
    valid = confirmations.loc[(times <= cut) & (times >= lower) & same_day]
    ends = valid.index.tz_convert("UTC") + pd.Timedelta(minutes=5)
    ages, timestamps, missing = {}, {}, []
    for key in CONFIRMATIONS:
        positive = ends[valid[key].to_numpy(dtype=bool)]
        if len(positive):
            age = int((cut - positive[-1]) / pd.Timedelta(minutes=5))
            ages[key], timestamps[key] = age, positive[-1].isoformat()
        else:
            ages[key], timestamps[key] = None, None
            missing.append(key)
    complete = not missing
    return {
        "last_confirmation_closed_at": timestamps, "ages_in_bars": ages,
        "missing_in_lookback": missing,
        "span_first_to_last_bars": max(ages.values()) - min(ages.values()) if complete else None,
        "oldest_confirmation_age_bars": max(ages.values()) if complete else None,
        "latest_confirmation_age_bars": min(ages.values()) if complete else None,
        "micro_shadow_by_tolerance": {str(k): complete and max(ages.values()) <= k for k in (0, 1, 2, 3)},
    }


def non_temporal_gates(record, side, bars):
    """Freeze macro/setup, veto, score, direction and bracket at the cut.

    Stoch overbought cap stays CURRENT; no stale safety state is imported.
    Rejection is reconstructed from its signed score component, then checked
    against the frozen worker flag where that extra flag was archived.
    """
    f, g = record["feature_snapshot"]["features"], record["gates"]
    contracts = [h["operational_contract"] for h in record["horizons"] if h["operational_contract"]]
    if not contracts:
        return {"valid_plan": False, "all_pass": False}
    c = contracts[0]
    rejected = bool(f["risk_veto"] or any(p["name"] == "Veto MACD intradía" for p in f["score_breakdown"]))
    if "signal_not_rejected" in g and bool(g["signal_not_rejected"]) != (not rejected):
        raise ValueError("Rechazo reconstruido no coincide con snapshot.")
    base = float((f.get("cross_asset_context") or {}).get("base_score", f["probability_up"]))
    score = base if side == "LONG" else 100-base
    current_cap = side != "LONG" or max(float(bars.StochRSI_K.iloc[-1]), float(bars.StochRSI_D.iloc[-1])) <= 80
    gates = {"permission": g["regime_long_only" if side == "LONG" else "regime_short_only"],
             "setup": g["setup_long" if side == "LONG" else "setup_short"],
             "risk_veto_inactive": not f["risk_veto"], "not_rejected": not rejected,
             "plan_same_direction": c["side"] == side, "valid_plan": True,
             "directional_score_at_least65": score >= 65, "current_stoch_cap": current_cap}
    gates["all_pass"] = all(gates.values())
    return gates


def volume_summary(values):
    a = np.asarray(values, dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("Ratios de volumen ausentes/no finitos.")
    return {"n": len(a), "p25": float(np.quantile(a, .25)), "median": float(np.median(a)),
            "p75": float(np.quantile(a, .75)), "p90": float(np.quantile(a, .9)),
            "threshold1_empirical_percentile": float(100 * np.mean(a <= 1)),
            "above1": int(np.sum(a > 1)), "at_or_below1": int(np.sum(a <= 1))}


def summarize_lags(rows):
    failed = [r for r in rows if r["failed_original_trigger"]]
    selected = [r for r in failed if r["selected_side_lag"]["span_first_to_last_bars"] is not None]
    spans = [r["selected_side_lag"]["span_first_to_last_bars"] for r in selected]
    return {
        "cuts": len(rows), "failed_trigger": len(failed), "span_observable": len(selected),
        "span_unavailable_missing_confirmation": len(failed) - len(selected),
        "span_p25_median_p75": np.quantile(spans, [.25, .5, .75]).tolist() if spans else [None]*3,
        "span_histogram": dict(sorted(Counter(str(v) for v in spans).items(), key=lambda kv: int(kv[0]))),
        "missing_counts": dict(Counter(k for r in failed for k in r["selected_side_lag"]["missing_in_lookback"])),
        "micro_shadow_all_cuts": {str(k): sum(r["selected_side_lag"]["micro_shadow_by_tolerance"][str(k)] for r in rows) for k in range(4)},
        "full_shadow_all_cuts": {str(k): sum(r["full_shadow_by_tolerance"][str(k)] for r in rows) for k in range(4)},
        "additional_failed_cuts_full_shadow": {str(k): sum(r["full_shadow_by_tolerance"][str(k)] for r in failed) for k in (1, 2, 3)},
        "full_shadow_sides": {str(k): dict(Counter(r["selected_side"] for r in rows if r["full_shadow_by_tolerance"][str(k)])) for k in range(4)},
    }


def verify_tail(record, bars):
    frozen = record["feature_snapshot"]["indicator_tail"]["intraday_indicators"]
    if pd.Timestamp(frozen["at"]) != bars.index[-1]:
        raise ValueError("Vela fuente reconstruida difiere del corte firmado.")
    for key, value in frozen["values"].items():
        if key in bars.columns and value is not None:
            if not np.isclose(float(value), float(bars[key].iloc[-1]), rtol=1e-11, atol=1e-9):
                raise ValueError(f"Indicador reconstruido difiere: {record['cut_at']} {key}")
    previous = record["gates"]["inputs"]
    for saved, col in (("previous_k", "StochRSI_K"), ("previous_d", "StochRSI_D"),
                       ("previous_high", "High"), ("previous_low", "Low")):
        if not np.isclose(previous[saved], float(bars[col].iloc[-2]), rtol=1e-11, atol=1e-9):
            raise ValueError("Indicador de vela anterior difiere del snapshot.")


def measure_symbol(symbol, protocol):
    completion = read_signed(OUTPUT / f"{symbol}_completion.json")
    dataset = load_dataset(INPUT, symbol)
    if dataset.fingerprint(symbol) != protocol["dataset_sha256"][symbol]:
        raise ValueError("El histórico cambió.")
    groups = defaultdict(list)
    for item in completion["records"]:
        path = OUTPUT / item["file"]
        if digest_file(path) != item["file_sha256"]:
            raise ValueError("Corte alterado.")
        record = read_signed(path)
        local_date = pd.Timestamp(record["cut_at"]).tz_convert("America/New_York").date()
        groups[local_date].append(record)
    measures = []
    for n, (day, records) in enumerate(sorted(groups.items()), 1):
        last_cut = max(pd.Timestamp(r["cut_at"]) for r in records)
        month_start = (last_cut.tz_convert("America/New_York") - pd.DateOffset(months=1)).normalize()
        raw = dataset.intraday.loc[(dataset.intraday.index >= month_start) & (dataset.intraday.index + pd.Timedelta(minutes=5) <= last_cut)]
        # All indicator operations are backward rolling/positive shift/EWM.
        # We verify every cut against its already signed live-parity snapshot.
        indicators = add_intraday_indicators(raw)
        for record in sorted(records, key=lambda r: r["cut_at"]):
            cut = pd.Timestamp(record["cut_at"])
            bars = current_indicator_suffix(indicators.loc[indicators.index + pd.Timedelta(minutes=5) <= cut],
                                             [*INTRADAY_READY, "RSI14"], "5m")
            verify_tail(record, bars)
            g = record["gates"]
            long, short = trigger_momentum(bars)
            if bool(long) != g["long_momentum_all"]:
                raise ValueError("Reconstrucción altera impulso LONG.")
            contracts = [h["operational_contract"] for h in record["horizons"] if h["operational_contract"]]
            side = contracts[0]["side"] if contracts else g["plan_side"]
            analyses = {}
            for direction in ("LONG", "SHORT"):
                lag = lag_measurement(confirmation_frame(bars, direction), cut)
                fixed = non_temporal_gates(record, direction, bars)
                full = {str(k): fixed["all_pass"] and lag["micro_shadow_by_tolerance"][str(k)] for k in range(4)}
                analyses[direction] = {"lag": lag, "non_temporal_gates": fixed, "full_shadow": full}
            current_native = bool((contracts[0] if contracts else {}).get("eligible_at_emission", False))
            if analyses[side]["full_shadow"]["0"] != current_native:
                raise ValueError(f"Control k=0 no reproduce elegibilidad: {symbol} {cut}")
            ratio = float(bars.Volume.iloc[-1] / bars.Volume.iloc[-21:-1].mean())
            if not np.isclose(ratio, record["feature_snapshot"]["features"]["volume_ratio"], rtol=1e-11, atol=1e-9):
                raise ValueError("Volumen reconstruido difiere.")
            measures.append({"symbol": symbol, "cut_at": record["cut_at"], "time_ny": record["time_ny"],
                             "cohorts": record["cohorts"], "selected_side": side,
                             "failed_original_trigger": not g["final_trigger_any_direction"],
                             "original_technical_eligible": current_native,
                             "selected_side_lag": analyses[side]["lag"],
                             "full_shadow_by_tolerance": analyses[side]["full_shadow"],
                             "long_diagnostic": analyses["LONG"], "short_diagnostic": analyses["SHORT"],
                             "volume_ratio": ratio})
        if n % 30 == 0 or n == len(groups):
            print(f"{symbol}: {n}/{len(groups)} sesiones; {len(measures)} cortes verificados", flush=True)
    clocks = {t: {"lag": summarize_lags([r for r in measures if r["time_ny"] == t]),
                   "volume": volume_summary([r["volume_ratio"] for r in measures if r["time_ny"] == t])} for t in TIMES}
    return {"cuts": measures, "by_clock": clocks, "lag_overall": summarize_lags(measures)}


def cost_components(entry, exit_price, terms):
    """Per share, assuming the frozen tariff; never call this a paid charge."""
    return {"commission_entry_assumed": entry * terms["commission_bps_per_side"] / 10000,
            "commission_exit_assumed": exit_price * terms["commission_bps_per_side"] / 10000,
            "commission_roundtrip_assumed": (entry+exit_price) * terms["commission_bps_per_side"] / 10000,
            "spread_roundtrip_assumed_NOT_observed": (entry+exit_price) * terms["spread_bps"] / 20000,
            "slippage_roundtrip_assumed_NOT_observed": (entry+exit_price) * terms["slippage_bps_per_side"] / 10000}


def fill_evidence(diagnosis):
    results = []
    for event in diagnosis["symbols"]["SMCI"]["events"]:
        utc_key = pd.Timestamp(event["cut_at"]).strftime("%Y%m%d_%H%M")
        r = read_signed(OUTPUT / "cuts/SMCI" / f"{utc_key}.json")
        per_horizon = []
        for h in r["horizons"]:
            c, x = h["operational_contract"], h["execution_result"]
            per_horizon.append({"horizon_minutes": h["horizon_minutes"], "fill_price_possible_OHLC": x.get("fill_price"),
                                "fill_at_possible_OHLC": x.get("fill_at"), "exit_price_simulated": x.get("exit_price"),
                                "exit_at_simulated": x.get("exit_at"), "frozen_cost_terms": c["execution_terms"],
                                "costs_per_share_assumed": cost_components(x["fill_price"], x["exit_price"], c["execution_terms"])})
        results.append({"event_id": event["event_id"], "cut_at": event["cut_at"], "side": event["side"],
                        "limit_price": event["entry_price"], "hypothetical_size_shares": 1,
                        "spread_real": None, "bid_real": None, "ask_real": None,
                        "fill_with_actual_spread": "NOT_OBSERVABLE_NO_QUOTES",
                        "real_broker_fill": "NO_ORDER_OR_EXECUTION_RECEIPT_FOR_REPLAY",
                        "gbm_trading_usa_short_executable_current_product": False if event["side"] == "SHORT" else None,
                        "classification": "SHORT_NOT_EXECUTABLE_TRADING_USA" if event["side"] == "SHORT" else "LONG_OHLC_POSSIBLE_BROKER_FILL_UNVERIFIED",
                        "actual_commission_paid": None, "actual_spread_paid": None,
                        "historical_broker_permission_at_cut": "NOT_ARCHIVED",
                        "resolutions": per_horizon})
    return results


def method_manifest(protocol, closed_report, diagnosis):
    return {"version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "DESCRIPTIVE_ONLY_NO_FILTER_OR_MODEL_CHANGES", "lookback_bars": LOOKBACK,
            "shadow_tolerances_bars": [0, 1, 2, 3], "same_session_only": True,
            "span_definition": "max_age_minus_min_age_of_last_true_observations_each_confirmation",
            "shadow_definition": "all_four_confirmations_present_in_last_k_plus_one_closed_bars_at_cut",
            "safety_and_macro_state": "CURRENT_AT_CUT_NEVER_CARRIED_BACK",
            "direction": "FROZEN_ORIGINAL_PLAN; other_side_diagnostic_only",
            "cannot_infer_minimum_span_outside_censored_lookback": True,
            "cannot_infer_live_authorization_or_profitability": True,
            "ledger_before_sha256": digest_file(ROOT / "data/portfolio.db"),
            "preregistration_sha256": sha(protocol), "closed_counts_report_sha256": sha(closed_report),
            "closed_diagnosis_sha256": sha(diagnosis), "measurement_script_sha256": digest_file(Path(__file__)),
            "engine_code_sha256": protocol["engine_code_sha256"], "sources": SOURCES}


def run(output=DEFAULT_OUTPUT):
    protocol = read_signed(OUTPUT / "preregistration.json")
    closed_report = read_signed(OUTPUT / "report.json")
    diagnosis = read_signed(OUTPUT / "diagnosis.json")
    for name, expected in protocol["engine_code_sha256"].items():
        if digest_file(ROOT / name) != expected:
            raise ValueError("Se modificó el motor congelado.")
    method_path = output / "measurement_method.json"
    if method_path.exists():
        method = read_signed(method_path)
        if (method["measurement_script_sha256"] != digest_file(Path(__file__))
                or method["closed_counts_report_sha256"] != sha(closed_report)
                or method["closed_diagnosis_sha256"] != sha(diagnosis)):
            raise ValueError("Método/código/base de evidencia cambiaron; no mezclar mediciones.")
    else:
        method = method_manifest(protocol, closed_report, diagnosis)
        write_signed(method_path, method)
    symbols = {}
    for symbol in protocol["symbols"]:
        cache = output / f"{symbol}_measurements.json"
        if cache.exists():
            measured = read_signed(cache)
            if measured["method_sha256"] != sha(method):
                raise ValueError("Mediciones pertenecen a otro método.")
        else:
            measured = {"method_sha256": sha(method), "symbol": symbol, **measure_symbol(symbol, protocol)}
            write_signed(cache, measured)
        symbols[symbol] = {k: v for k, v in measured.items() if k not in ("cuts", "method_sha256", "symbol")}
    report = {"version": VERSION, "method_sha256": sha(method), "created_at": datetime.now(timezone.utc).isoformat(),
              "closed_counts": closed_report, "closed_event_counts": {s: {k: d[k] for k in ("distinct_authorized_events", "nonoverlapping_authorized_events")}
                                                                   for s, d in diagnosis["symbols"].items()},
              "symbols": symbols, "fill_evidence": fill_evidence(diagnosis),
              "quotes_acquired": False, "real_spread_measurable": False,
              "ledger_after_sha256": digest_file(ROOT / "data/portfolio.db")}
    if report["ledger_after_sha256"] != method["ledger_before_sha256"]:
        raise ValueError("La base cambió durante la medición.")
    write_signed(output / "measurement_report.json", report)
    print(json.dumps({s: v["by_clock"] for s, v in symbols.items()}, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output)
