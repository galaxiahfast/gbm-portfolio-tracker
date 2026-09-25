"""Read-only, cohort-level coverage of the scheduled XNYS forward collector.

Six horizons from one 11:00 cut are one observation opportunity, never six
independent trades.  Missing cuts are reported, never synthesized.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from portfolio_tracker.analytics.closed_bars import NY, _calendar
from portfolio_tracker.services.directional_collection import HORIZON_MINUTES
from portfolio_tracker.services.model_observations import utc_timestamp


def collection_funnel(repository, symbols=("SMCI", "NVDA"), *, now=None, sessions=10):
    """Inspect only signed records and the exchange calendar; never writes DB."""
    if not 1 <= int(sessions) <= 250:
        raise ValueError("sessions debe estar entre 1 y 250.")
    instant = utc_timestamp(now or datetime.now(timezone.utc))
    today = pd.Timestamp(instant.tz_convert(NY).date())
    schedule = _calendar(today.year - 1, today.year + 1).schedule
    due = [day for day in schedule.index if day <= today and
           instant >= utc_timestamp(schedule.loc[day, "open"]) + pd.Timedelta(minutes=95)]
    days = due[-int(sessions):]
    symbols = tuple(dict.fromkeys(str(value).strip().upper() for value in symbols))
    if not symbols or any(not value for value in symbols):
        raise ValueError("Se requiere al menos un símbolo válido.")

    with repository.database.connect() as connection:
        indexed = {}
        enrolled = {}
        for symbol in symbols:
            rows = connection.execute(
                """SELECT json_extract(parameters_json, '$.session_date') AS session,
                          json_extract(parameters_json, '$.replay.run_id') AS run_id
                   FROM live_model_observations
                   WHERE symbol=? AND json_valid(parameters_json)
                     AND json_extract(parameters_json, '$.replay.run_id') IS NOT NULL
                   GROUP BY session, run_id""",
                (symbol,),
            ).fetchall()
            indexed[symbol] = {}
            for row in rows:
                if row["session"] and row["run_id"]:
                    indexed[symbol].setdefault(row["session"], []).append(row["run_id"])
            enrolled[symbol] = min(indexed[symbol], default=None)

    results = []
    for symbol in symbols:
        for day in days:
            session = day.date().isoformat()
            if enrolled[symbol] is None or session < enrolled[symbol]:
                continue  # No retrospective obligations before enrollment.
            ids = indexed[symbol].get(session, ())
            records = [repository.live_model_execution_record(run_id) for run_id in ids]
            verified = next((record for record in records if record is not None), None)
            if verified is None:
                results.append({
                    "symbol": symbol, "session": session,
                    "status": "INVALID_SIGNED_CUT" if ids else "MISSING_CUT",
                    "reason": "Cohorte parcial o firma inválida" if ids else "Sin corte firmado a las 11 NY",
                    "directional_resolved_horizons": 0,
                    "operational_eligible_at_emission": False,
                    "possible_fill_status": "N/D",
                })
                continue
            predictions = verified["predictions"]
            first = next(row for row in predictions if row["horizon_minutes"] == HORIZON_MINUTES["1 Hora"])
            target = first.get("operational_target") or {}
            result = first.get("operational_result") or {}
            features = verified.get("features") or {}
            eligible = target.get("eligible_at_emission") is True
            execution = (result.get("evidence") or {}).get("execution_assessment") or {}
            if first.get("operational_status"):
                reason = first["operational_status"]
            elif features.get("risk_veto"):
                reason = "RISK_VETO"
            elif not features.get("activation_trigger_met"):
                reason = "TRIGGER_NOT_CONFIRMED"
            elif features.get("execution_plan_conditional"):
                reason = "CONDITIONAL_PLAN"
            elif not eligible:
                reason = "NOT_ELIGIBLE_AT_EMISSION"
            else:
                reason = "ELIGIBLE_AT_EMISSION"
            results.append({
                "symbol": symbol, "session": session, "status": "SIGNED_CUT",
                "reason": reason,
                "directional_resolved_horizons": sum(
                    row["resolution_status"] == "RESOLVED" for row in predictions
                ),
                "operational_eligible_at_emission": eligible,
                "possible_fill_status": execution.get("status") or "PENDING_OR_UNAVAILABLE",
            })
    return {
        "as_of": instant.isoformat(),
        "scope": "Cohortes 11:00 NY; no confundir 6 horizontes con 6 operaciones",
        "rows": results,
        "summary": {
            "expected_after_enrollment": len(results),
            "signed": sum(row["status"] == "SIGNED_CUT" for row in results),
            "missing": sum(row["status"] == "MISSING_CUT" for row in results),
            "invalid": sum(row["status"] == "INVALID_SIGNED_CUT" for row in results),
            "eligible_at_emission": sum(row["operational_eligible_at_emission"] for row in results),
            "simulated_resolved_fills": sum(
                row["possible_fill_status"] == "SIMULATED_RESOLVED" for row in results
            ),
        },
    }
