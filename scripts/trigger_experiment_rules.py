"""Pure, preregistered replay ablations. NEVER imported by production."""
from __future__ import annotations

from copy import deepcopy
import re

import numpy as np
import pandas as pd

from scripts.measure_trigger_selectivity import non_temporal_gates

VARIANTS = ("CURRENT", "VOLUME_P50_CLOCK", "STOCH_ONE_BAR", "LONG_MAJORITY", "BOTH_REDUCED", "SIMPLE_EMA_CROSS")


def rules_specification():
    return {
        "CURRENT": "Frozen trigger: unchanged.",
        "VOLUME_P50_CLOCK": "Only volume: current closed-bar volume strictly above median (P50) of 20 PREVIOUS sessions of that same 5m source clock. No search over percentiles.",
        "STOCH_ONE_BAR": "Only crossover freshness: crossover at current OR previous closed 5m bar in SAME session; current K>D LONG / K<D SHORT. Current LONG cap<=80 remains.",
        "LONG_MAJORITY": "Only LONG macro permission: unchanged daily trending/hysteresis AND at least 2 of 3 S/D/4h trend votes +1. SHORT permission unchanged. Original reduced exposure retained when original permission BOTH_REDUCED.",
        "BOTH_REDUCED": "Only macro permission: original LONG_ONLY/SHORT_ONLY OR BOTH_REDUCED, always same frozen plan side + setup. Exposure capped at 0.25 under BOTH. SHORT measured separately, not broker-executable.",
        "SIMPLE_EMA_CROSS": "Control, not one-gate ablation: EMA9 crosses EMA21 on current CLOSED 5m bar in plan direction. Removes volume/crossover/MACD/previous-bar breakout microconditions. Same ORIGINAL macro/setup, veto/rejection, valid plan, score>=65, current LONG Stoch<=80, clock, brackets and costs.",
    }


def clock_reference(intraday):
    """Baseline at each clock uses only 20 previous observations of that clock."""
    clocks = intraday.index.tz_convert("America/New_York").strftime("%H:%M")
    grouped = intraday.Volume.groupby(clocks)
    return grouped.transform(lambda x: x.shift(1).rolling(20, min_periods=20).quantile(.5))


def crossover(bars, side, back=0):
    if len(bars) < 2 + back:
        return False
    prev, curr = bars.iloc[-2-back], bars.iloc[-1-back]
    return bool(prev.StochRSI_K <= prev.StochRSI_D and curr.StochRSI_K > curr.StochRSI_D) if side == "LONG" else bool(
        prev.StochRSI_K >= prev.StochRSI_D and curr.StochRSI_K < curr.StochRSI_D)


def evaluate_variants(record, bars, same_clock_median):
    contracts = [h["operational_contract"] for h in record["horizons"] if h.get("operational_contract")]
    if not contracts:
        return {v: {"eligible": False, "cause": "NO_VALID_PLAN", "exposure": 0., "side": None} for v in VARIANTS}
    side = contracts[0]["side"]
    if len(bars) < 22:
        raise ValueError("Cut lacks shared trigger's minimum suffix history.")
    f, g = record["feature_snapshot"]["features"], record["gates"]
    fixed = non_temporal_gates(record, side, bars)
    fixed.pop("all_pass")
    latest, prior = bars.iloc[-1], bars.iloc[-2]
    micro = {"volume": bool(latest.Volume > bars.Volume.iloc[-21:-1].mean() > 0),
             "cross": crossover(bars, side),
             "macd": bool(latest.MACD >= latest.MACD_signal) if side == "LONG" else bool(latest.MACD <= latest.MACD_signal),
             "break": bool(latest.Close >= prior.High) if side == "LONG" else bool(latest.Close <= prior.Low)}
    vote = re.search(r"Votos S/D/4h=\[(-?\d+), (-?\d+), (-?\d+)\]", f["hierarchy_detail"])
    if not vote:
        raise ValueError("Missing macro vote provenance.")
    votes = tuple(map(int, vote.groups()))
    previous_same_session = bars.index[-2].tz_convert("America/New_York").date() == bars.index[-1].tz_convert("America/New_York").date()
    ema9, ema21 = bars.Close.ewm(span=9, adjust=False).mean(), bars.Close.ewm(span=21, adjust=False).mean()
    simple = (ema9.iloc[-2] <= ema21.iloc[-2] and ema9.iloc[-1] > ema21.iloc[-1]) if side == "LONG" else (
        ema9.iloc[-2] >= ema21.iloc[-2] and ema9.iloc[-1] < ema21.iloc[-1])
    result = {}
    for variant in VARIANTS:
        safety, timing = deepcopy(fixed), deepcopy(micro)
        exposure = float(f["exposure_factor"])
        if variant == "VOLUME_P50_CLOCK":
            timing["volume"] = bool(np.isfinite(same_clock_median) and same_clock_median > 0 and latest.Volume > same_clock_median)
        elif variant == "STOCH_ONE_BAR":
            current_sign = latest.StochRSI_K > latest.StochRSI_D if side == "LONG" else latest.StochRSI_K < latest.StochRSI_D
            timing["cross"] = bool(current_sign and (micro["cross"] or (previous_same_session and crossover(bars, side, 1))))
        elif variant == "LONG_MAJORITY" and side == "LONG":
            safety["permission"] = bool(f["macro_trending"] and sum(v == 1 for v in votes) >= 2)
        elif variant == "BOTH_REDUCED":
            safety["permission"] = bool(safety["permission"] or f["macro_permission"] == "BOTH_REDUCED")
            if f["macro_permission"] == "BOTH_REDUCED":
                exposure = min(exposure, .25)
        elif variant == "SIMPLE_EMA_CROSS":
            timing = {"ema_cross": bool(simple)}
        if not np.isfinite(exposure) or not 0 < exposure <= 1:
            safety["valid_exposure"] = False
        eligible = bool(all(safety.values()) and all(timing.values()))
        result[variant] = {"eligible": eligible, "side": side, "exposure": exposure,
                           "non_temporal": safety, "micro": timing,
                           "failed_conditions": [k for k, v in {**safety, **timing}.items() if not v],
                           "same_clock_p50_volume": float(same_clock_median) if np.isfinite(same_clock_median) else None,
                           "macro_votes": votes}
    native = bool(contracts[0]["eligible_at_emission"])
    if result["CURRENT"]["eligible"] != native:
        raise ValueError(f"Current control differs from frozen cut: {record['symbol']} {record['cut_at']}")
    for variant in ("VOLUME_P50_CLOCK", "STOCH_ONE_BAR", "LONG_MAJORITY", "BOTH_REDUCED"):
        permitted = {"VOLUME_P50_CLOCK": "volume", "STOCH_ONE_BAR": "cross", "LONG_MAJORITY": "permission", "BOTH_REDUCED": "permission"}[variant]
        before = {**result["CURRENT"]["non_temporal"], **result["CURRENT"]["micro"]}
        after = {**result[variant]["non_temporal"], **result[variant]["micro"]}
        if any(before[k] != after[k] for k in before if k != permitted):
            raise ValueError("A variant changed more than its single condition.")
    return result


def nonoverlapping_events(rows):
    """Earliest eligible event; embargo until MAX full timeout, not outcome."""
    last_end, kept = None, []
    for row in sorted(rows, key=lambda r: r["cut_at"]):
        if last_end is None or pd.Timestamp(row["cut_at"]) >= last_end:
            kept.append(row)
            last_end = max(pd.Timestamp(h["contract"]["timeout_at"]) for h in row["horizons"])
    return kept


def statistics(rows, horizon, side=None):
    if side:
        rows = [r for r in rows if r["side"] == side]
    entries = [h for r in rows for h in r["horizons"] if h["horizon_minutes"] == horizon]
    from collections import Counter
    outcomes = Counter(h["execution"].get("outcome") for h in entries)
    statuses = Counter(h["execution"]["status"] for h in entries)
    pnl = []
    for row in rows:
        h = next(h for h in row["horizons"] if h["horizon_minutes"] == horizon)
        e = h["execution"]
        if e.get("net_pnl_per_share") is not None:
            pnl.append(float(e["net_pnl_per_share"]) / float(e["fill_price"]) * 10000 * row["exposure"])
    return {"events": len(rows), "LONG": sum(r["side"] == "LONG" for r in rows), "SHORT": sum(r["side"] == "SHORT" for r in rows),
            "possible_fills": sum(h["execution"].get("fill_price") is not None for h in entries),
            **{k: outcomes[k] for k in ("TP_FIRST", "SL_FIRST", "TIMEOUT")},
            "ambiguous": statuses["AMBIGUOUS_NO_TRADE"], "pending": statuses["PENDING_ENTRY"] + statuses["FILLED_PENDING"],
            "no_fill": sum(v for k, v in statuses.items() if k.startswith("NO_FILL")), "incomplete": statuses["EVIDENCE_INCOMPLETE"],
            "resolved_pnl_observations": len(pnl), "net_bps_sum_exposure_scaled": sum(pnl) if pnl else None,
            "net_bps_median_exposure_scaled": float(np.median(pnl)) if pnl else None,
            "worst_net_bps_exposure_scaled": min(pnl) if pnl else None}
