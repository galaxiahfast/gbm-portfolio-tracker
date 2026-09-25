"""One preregistered XNYS 11:00 cohort per symbol/session, independent of UI reruns."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
import json
import math
from zoneinfo import ZoneInfo

from .model_observations import is_regular_close, utc_timestamp
from .model_execution_record import build_replay_snapshot, prediction_snapshot
from portfolio_tracker.analytics.operational_target import make_operational_contract
from portfolio_tracker.analytics.temporal_contract import (
    ANCHOR_VERSION, is_actionable_emission, scheduled_cut,
)
from .scenario_calibration import make_scenario_contract

NY = ZoneInfo("America/New_York")
LEGACY_COLLECTION_PROTOCOL = "XNYS_1100_WINDOW_V1"
PREVIOUS_COLLECTION_PROTOCOL = "XNYS_1100_OPERATIONAL_TARGET_V2"
COLLECTION_PROTOCOL = "XNYS_1100_OPERATIONAL_TARGET_V3"
SUPPORTED_COLLECTION_PROTOCOLS = {
    LEGACY_COLLECTION_PROTOCOL, PREVIOUS_COLLECTION_PROTOCOL, COLLECTION_PROTOCOL,
}
NO_VALID_PLAN = "NO_VALID_FIRST_PASSAGE_PLAN"
HORIZON_MINUTES = {
    "1 Hora": 60,
    "6 Horas": 360,
    "1 Día": 1_440,
    "1 Semana": 10_080,
    "1 Mes": 43_200,
    "6 Meses": 259_200,
}


def valid_unavailable_plan(metadata, reference_price) -> bool:
    """A signed directional-only row must prove why no barrier was created."""
    if (metadata.get("operational_status") != NO_VALID_PLAN
            or metadata.get("primary_validation_target") != "DIRECTIONAL_ONLY"
            or metadata.get("operational_contract") is not None):
        return False
    levels = metadata.get("invalid_plan_levels")
    if not isinstance(levels, dict) or set(levels) != {
            "side", "entry_price", "stop_loss", "take_profit"}:
        return False
    try:
        entry, stop, target = (float(levels[key]) for key in (
            "entry_price", "stop_loss", "take_profit"))
        reference = float(reference_price)
    except (TypeError, ValueError, OverflowError):
        return False
    if (not all(math.isfinite(value) and value > 0 for value in (
            entry, stop, target, reference)) or abs(entry - reference) > 1e-9):
        return False
    side = levels["side"]
    if side == "LONG":
        return not stop < entry < target
    if side == "SHORT":
        return not target < entry < stop
    return False


def scenario_parameters(parameters, *, protocol=COLLECTION_PROTOCOL):
    """Keep the same immutable model identity in UI calibration and headless emission."""
    return {**parameters, "observation_protocol": protocol}


def cut_forecasts(
    analysis,
    parameters,
    observed_at: datetime,
    *,
    protocol=COLLECTION_PROTOCOL,
    input_artifacts=None,
):
    """Build one signed six-horizon contract from an exact 11 NY cut.

    ``protocol`` is part of model identity.  Historical replay therefore uses
    the same contract builder without masquerading as live forward evidence.
    This function performs no database writes.
    """
    observed = utc_timestamp(observed_at)
    local = observed.tz_convert(NY)
    source = utc_timestamp(analysis.source_bar_closed_at)
    if not (source <= observed < source + timedelta(minutes=5)) or not is_regular_close(source):
        raise ValueError("La observación requiere el último cierre de 5m, no una vela abierta o antigua.")
    if source.tz_convert(NY).date() != local.date():
        raise ValueError("La vela fuente debe pertenecer a la sesión actual.")
    if not is_actionable_emission(observed, source):
        raise ValueError("Corte no accionable: se requiere la vela 11:00 NY y su ventana causal de emisión.")
    horizons = {item.label: item for item in analysis.horizon_projections}
    if set(horizons) != set(HORIZON_MINUTES):
        raise ValueError("Se requieren los seis horizontes direccionales completos.")
    model_parameters = scenario_parameters(parameters, protocol=protocol)
    replay = build_replay_snapshot(
        analysis, observed_at=observed_at, protocol=protocol,
        input_artifacts=input_artifacts,
    )
    rows = []
    from portfolio_tracker.analytics.horizon_models import model_feature_contract
    from .model_execution_record import technical_horizon
    for label, minutes in HORIZON_MINUTES.items():
        horizon = horizons[label]
        model_prediction = prediction_snapshot(technical_horizon(analysis, label))
        contract = make_scenario_contract(analysis.symbol, horizon, minutes, model_parameters)
        try:
            operational_contract = make_operational_contract(
                analysis.symbol, horizon, minutes, analysis, observed_at, model_parameters,
            )
        except ValueError as exc:
            # A stale plan (e.g. price has already crossed TP1) cannot be
            # treated as a realizable entry. Preserve the real directional
            # forecast, but never invent replacement barriers or an outcome.
            if str(exc) != "TP/entrada/SL no encierran correctamente el precio de referencia.":
                raise
            operational_contract = None
        model_manifest = model_feature_contract(
            {"feature_snapshot": replay},
            {"prediction": prediction_snapshot(horizon),
             "model_prediction": model_prediction,
             "operational_contract": operational_contract},
        )
        rows.append(dict(
            horizon_minutes=minutes,
            raw_probability_up=Decimal(str(contract["probabilities"][0])),
            parameters_json=json.dumps({
                **model_parameters,
                "engine": horizon.engine_name,
                "feedback_version": 3,
                "scenario_contract": contract,
                "operational_contract": operational_contract,
                "primary_validation_target": (
                    operational_contract["version"] if operational_contract else "DIRECTIONAL_ONLY"
                ),
                "operational_status": (
                    "VALID_PLAN" if operational_contract else NO_VALID_PLAN
                ),
                "invalid_plan_levels": (
                    None if operational_contract else {
                        "side": str(analysis.execution_levels.direction).upper(),
                        "entry_price": float(analysis.last_price),
                        "stop_loss": float(analysis.execution_levels.stop_loss),
                        "take_profit": float(analysis.execution_levels.take_profit_1),
                    }
                ),
                "collection_protocol": protocol,
                "scheduled_cut_ny": scheduled_cut(observed).tz_convert(NY).strftime("%H:%M"),
                "anchor_version": ANCHOR_VERSION,
                "session_date": local.date().isoformat(),
                "replay": replay,
                "prediction_snapshot": prediction_snapshot(horizon),
                "model_prediction_snapshot": model_prediction,
                "model_feature_contract": model_manifest,
            }, sort_keys=True, allow_nan=False),
        ))
    return rows


def fixed_cut_forecasts(analysis, parameters, observed_at: datetime, *, input_artifacts=None):
    """Validate a real, fresh closed bar; never backdate a missed live cut."""
    return cut_forecasts(
        analysis,
        parameters,
        observed_at,
        protocol=COLLECTION_PROTOCOL,
        input_artifacts=input_artifacts,
    )


def record_fixed_directional(repository, analysis, parameters, observed_at: datetime,
                             *, input_artifacts=None) -> int:
    """Persist all six forecasts atomically; zero means the daily cut already exists."""
    rows = fixed_cut_forecasts(analysis, parameters, observed_at,
                               input_artifacts=input_artifacts)
    return repository.record_fixed_live_observations(
        symbol=analysis.symbol,
        observed_at=observed_at,
        source_bar_at=analysis.source_bar_closed_at,
        reference_price=Decimal(str(analysis.last_price)),
        forecasts=rows,
        session_date=utc_timestamp(observed_at).tz_convert(NY).date().isoformat(),
        protocol=COLLECTION_PROTOCOL,
    )
