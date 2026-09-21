"""One preregistered XNYS 11:00 cohort per symbol/session, independent of UI reruns."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from decimal import Decimal
import json
from zoneinfo import ZoneInfo

from .model_observations import is_regular_close, utc_timestamp
from .model_execution_record import build_replay_snapshot, prediction_snapshot
from portfolio_tracker.analytics.operational_target import make_operational_contract
from .scenario_calibration import make_scenario_contract

NY = ZoneInfo("America/New_York")
LEGACY_COLLECTION_PROTOCOL = "XNYS_1100_WINDOW_V1"
COLLECTION_PROTOCOL = "XNYS_1100_OPERATIONAL_TARGET_V2"
SUPPORTED_COLLECTION_PROTOCOLS = {LEGACY_COLLECTION_PROTOCOL, COLLECTION_PROTOCOL}
HORIZON_MINUTES = {
    "1 Hora": 60,
    "6 Horas": 360,
    "1 Día": 1_440,
    "1 Semana": 10_080,
    "1 Mes": 43_200,
    "6 Meses": 259_200,
}


def scenario_parameters(parameters):
    """Keep the same immutable model identity in UI calibration and headless emission."""
    return {**parameters, "observation_protocol": COLLECTION_PROTOCOL}


def fixed_cut_forecasts(analysis, parameters, observed_at: datetime, *, input_artifacts=None):
    """Validate a real, fresh closed bar; never backdate a missed fixed cut."""
    observed = utc_timestamp(observed_at)
    local = observed.tz_convert(NY)
    if not time(11) <= local.time() < time(11, 20):
        raise ValueError("La emisión direccional requiere la ventana 11:00–11:20 NY.")
    source = utc_timestamp(analysis.source_bar_closed_at)
    if not (source <= observed < source + timedelta(minutes=5)) or not is_regular_close(source):
        raise ValueError("La observación requiere el último cierre de 5m, no una vela abierta o antigua.")
    if source.tz_convert(NY).date() != local.date():
        raise ValueError("La vela fuente debe pertenecer a la sesión actual.")
    horizons = {item.label: item for item in analysis.horizon_projections}
    if set(horizons) != set(HORIZON_MINUTES):
        raise ValueError("Se requieren los seis horizontes direccionales completos.")
    model_parameters = scenario_parameters(parameters)
    replay = build_replay_snapshot(
        analysis, observed_at=observed_at, protocol=COLLECTION_PROTOCOL,
        input_artifacts=input_artifacts,
    )
    rows = []
    for label, minutes in HORIZON_MINUTES.items():
        horizon = horizons[label]
        contract = make_scenario_contract(analysis.symbol, horizon, minutes, model_parameters)
        operational_contract = make_operational_contract(
            analysis.symbol, horizon, minutes, analysis, observed_at, model_parameters,
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
                "primary_validation_target": operational_contract["version"],
                "collection_protocol": COLLECTION_PROTOCOL,
                "scheduled_cut_ny": "11:00",
                "session_date": local.date().isoformat(),
                "replay": replay,
                "prediction_snapshot": prediction_snapshot(horizon),
            }, sort_keys=True, allow_nan=False),
        ))
    return rows


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
