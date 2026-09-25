"""Causal three-class operational target: TP first, SL first, or timeout.

This module is deliberately pure: no SQLite, network, portfolio state or order
execution.  Bars are OPEN-labelled and only become usable at their close.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from functools import lru_cache
import hashlib
import math
from pathlib import Path

import pandas as pd

from .closed_bars import NY, _calendar
from .temporal_contract import ANCHOR_VERSION, first_evaluable_open, is_actionable_emission
from ..services.model_observations import (
    POLICY, PREVIOUS_POLICY, canonical, is_regular_close, maturity, utc_timestamp,
)
from ..services.scenario_calibration import engine_revision

LEGACY_TARGET_VERSION = "TP_FIRST_SL_FIRST_TIMEOUT_V1"
TARGET_VERSION = "TP_FIRST_SL_FIRST_TIMEOUT_V2"
SAME_BAR_POLICY = "SL_FIRST_CONSERVATIVE"
ENTRY_POLICY = "REFERENCE_CLOSED_PRICE_ASSUMED"
GAP_POLICY = "SL_AT_OPEN_TP_AT_TARGET_V1"
EVIDENCE_POLICY = "5M_PRIMARY_DAILY_CONSERVATIVE_FALLBACK_V1"
LABELER_SEMANTICS = "CAUSAL_FIRST_PASSAGE_XNYS_V1"


class IncompleteOperationalEvidence(ValueError):
    """The path is still unobservable; retrying with more bars is valid."""


@lru_cache(maxsize=1)
def labeler_revision():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class OperationalOutcome(StrEnum):
    TP_FIRST = "TP_FIRST"
    SL_FIRST = "SL_FIRST"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True, slots=True)
class OperationalResult:
    outcome: OperationalOutcome
    exit_price: float
    exit_at: str
    exit_source: str
    evidence_sha256: str = ""

    @property
    def outcome_bar_at(self):
        return self.exit_at


@dataclass(frozen=True, slots=True)
class OperationalScan:
    """Incremental first-passage scan safe to checkpoint between sessions."""
    result: OperationalResult | None
    scanned_through: str | None
    evidence_sha256: str
    evidence_count: int
    # Transient, already-validated rows processed in this scan. The repository
    # advances the separate possible-fill state from exactly this causal path.
    observed_bars: tuple = ()


@dataclass(frozen=True, slots=True)
class OperationalContract:
    """Minimal public contract used by causal unit tests and replay tools."""
    symbol: str
    direction: str
    observed_at: datetime
    source_bar_at: datetime
    expires_at: datetime
    reference_price: Decimal
    take_profit: Decimal
    stop_loss: Decimal

    def __post_init__(self):
        observed, source, expires = map(utc_timestamp, (
            self.observed_at, self.source_bar_at, self.expires_at,
        ))
        entry = _finite_price(self.reference_price, "Entrada")
        target = _finite_price(self.take_profit, "Take profit")
        stop = _finite_price(self.stop_loss, "Stop")
        direction = self.direction.upper()
        if source > observed or expires <= observed:
            raise ValueError("Tiempos operativos inválidos.")
        if not ((direction == "LONG" and stop < entry < target) or
                (direction == "SHORT" and target < entry < stop)):
            raise ValueError("Niveles incompatibles con la dirección.")


def _finite_price(value, name):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} inválido.")
    return number


def make_operational_contract(symbol, horizon, horizon_minutes, analysis, observed_at, parameters):
    """Freeze one hypothetical trade plan before any outcome is known."""
    from .execution_path import frozen_execution_terms
    observed = utc_timestamp(observed_at)
    entry_at = utc_timestamp(analysis.source_bar_closed_at)
    evaluation_start = first_evaluable_open(observed)
    timeout = maturity(observed, int(horizon_minutes))
    plan = analysis.execution_levels
    side = str(plan.direction).upper()
    entry = _finite_price(analysis.last_price, "Entrada")
    stop = _finite_price(plan.stop_loss, "Stop")
    target = _finite_price(plan.take_profit_1, "Take profit")
    if side == "LONG":
        valid = stop < entry < target
        directional_scores = [horizon.probability_up / 100, horizon.probability_down / 100,
                              horizon.probability_range / 100]
    elif side == "SHORT":
        valid = target < entry < stop
        directional_scores = [horizon.probability_down / 100, horizon.probability_up / 100,
                              horizon.probability_range / 100]
    else:
        raise ValueError("El plan operativo debe ser LONG o SHORT.")
    if not valid:
        raise ValueError("TP/entrada/SL no encierran correctamente el precio de referencia.")
    total = sum(float(value) for value in directional_scores)
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in directional_scores) or abs(total - 1) > 1e-6:
        raise ValueError("Vector direccional de origen inválido.")
    model = {
        "symbol": str(symbol).strip().upper(), "engine": horizon.engine_name,
        "horizon_minutes": int(horizon_minutes), "parameters": parameters,
        "horizon_policy": POLICY, "engine_revision": engine_revision(),
        "anchor_version": ANCHOR_VERSION,
        "target": TARGET_VERSION, "labeler_revision": labeler_revision(),
        "labeler_semantics": LABELER_SEMANTICS,
        "gap_policy": GAP_POLICY, "evidence_policy": EVIDENCE_POLICY,
    }
    eligible = bool(
        getattr(analysis, "activation_trigger_met", False)
        and not getattr(analysis, "execution_plan_conditional", True)
        and not getattr(analysis, "risk_veto", False)
        and not getattr(analysis, "signal_rejected", False)
    )
    contract = {
        "version": TARGET_VERSION,
        "model": model,
        "model_id": hashlib.sha256(canonical(model).encode()).hexdigest(),
        "classes": [item.value for item in OperationalOutcome],
        # UP/RANGE/DOWN does not identify which barrier was touched first.
        # Keep it as provenance only; never score it as an operational probability.
        "operational_probabilities": None,
        "probability_status": "N/D_HASTA_CALIBRACION_EMPIRICA_FIRST_PASSAGE",
        "directional_scores_source": [float(value) for value in directional_scores],
        "side": side, "entry_price": entry, "stop_loss": stop,
        "take_profit": target, "entry_at": entry_at.isoformat(),
        "observed_at": observed.isoformat(),
        "evaluation_starts_at": evaluation_start.isoformat(),
        "timeout_at": timeout.isoformat(),
        "entry_policy": ENTRY_POLICY, "same_bar_policy": SAME_BAR_POLICY,
        "execution_terms": frozen_execution_terms(observed),
        "eligible_at_emission": eligible and is_actionable_emission(observed, entry_at),
    }
    contract["contract_sha256"] = hashlib.sha256(canonical(contract).encode()).hexdigest()
    validate_operational_contract(contract)
    return contract


def validate_operational_contract(contract):
    if not isinstance(contract, dict) or contract.get("version") not in {
        LEGACY_TARGET_VERSION, TARGET_VERSION,
    }:
        raise ValueError("Contrato operativo desconocido.")
    unsigned = dict(contract)
    supplied = unsigned.pop("contract_sha256", None)
    if supplied != hashlib.sha256(canonical(unsigned).encode()).hexdigest():
        raise ValueError("Firma del contrato operativo inconsistente.")
    model = contract["model"]
    version = contract["version"]
    expected_policy = POLICY if version == TARGET_VERSION else PREVIOUS_POLICY
    if model.get("target") != version or model.get("horizon_policy") != expected_policy:
        raise ValueError("Modelo operativo incompatible.")
    if version == TARGET_VERSION and model.get("anchor_version") != ANCHOR_VERSION:
        raise ValueError("Anclaje operativo incompatible.")
    revision = model.get("labeler_revision")
    if (not isinstance(revision, str) or len(revision) != 64
            or any(char not in "0123456789abcdef" for char in revision)
            or model.get("labeler_semantics") != LABELER_SEMANTICS
            or model.get("gap_policy") != GAP_POLICY
            or model.get("evidence_policy") != EVIDENCE_POLICY):
        raise ValueError("Versión del etiquetador operativo incompatible.")
    if contract.get("model_id") != hashlib.sha256(canonical(model).encode()).hexdigest():
        raise ValueError("Identidad del modelo operativo inconsistente.")
    if contract.get("classes") != [item.value for item in OperationalOutcome]:
        raise ValueError("Clases operativas desordenadas.")
    if contract.get("operational_probabilities") is not None:
        raise ValueError("No hay probabilidades operativas calibradas para este contrato.")
    if contract.get("probability_status") != "N/D_HASTA_CALIBRACION_EMPIRICA_FIRST_PASSAGE":
        raise ValueError("Estado estadístico operativo inválido.")
    scores = tuple(float(value) for value in contract["directional_scores_source"])
    if len(scores) != 3 or any(not math.isfinite(value) or value < 0 for value in scores) or abs(sum(scores) - 1) > 1e-6:
        raise ValueError("Vector direccional de procedencia inválido.")
    side = contract["side"]
    entry = _finite_price(contract["entry_price"], "Entrada")
    stop = _finite_price(contract["stop_loss"], "Stop")
    target = _finite_price(contract["take_profit"], "Take profit")
    if not ((side == "LONG" and stop < entry < target) or
            (side == "SHORT" and target < entry < stop)):
        raise ValueError("Barreras operativas inválidas.")
    observed, entry_at, evaluation_start, timeout = map(utc_timestamp, (
        contract["observed_at"], contract["entry_at"],
        contract["evaluation_starts_at"], contract["timeout_at"],
    ))
    expected_start = (first_evaluable_open(observed) if version == TARGET_VERSION
                      else observed.ceil("5min"))
    expected_timeout = maturity(
        observed if version == TARGET_VERSION else expected_start,
        int(model["horizon_minutes"]), policy=expected_policy,
    )
    if (entry_at > observed or evaluation_start < observed
            or evaluation_start != expected_start or timeout != expected_timeout):
        raise ValueError("Tiempos del contrato operativo inconsistentes.")
    if version == TARGET_VERSION and contract.get("eligible_at_emission") and not is_actionable_emission(observed, entry_at):
        raise ValueError("Una emisión fuera de corte no puede ser operativa.")
    if contract.get("entry_policy") != ENTRY_POLICY or contract.get("same_bar_policy") != SAME_BAR_POLICY:
        raise ValueError("Política de ejecución operativa desconocida.")
    if "execution_terms" in contract:
        from .execution_path import validate_execution_terms
        validate_execution_terms(contract["execution_terms"], observed)
    return scores


def _validated_bars(frame, timeframe):
    if frame is None or frame.empty:
        return []
    required = ("Open", "High", "Low", "Close", "Volume")
    if not all(column in frame.columns for column in required):
        raise ValueError("La resolución operativa requiere OHLCV completo.")
    data = frame.loc[:, required].copy()
    index = pd.DatetimeIndex(data.index)
    if not index.is_unique:
        raise ValueError("Las velas deben tener timestamps únicos.")
    data = data.iloc[index.argsort()]
    index = pd.DatetimeIndex(data.index)
    records = []
    if timeframe == "5m":
        if index.tz is None:
            raise ValueError("Las velas de 5m requieren zona horaria explícita.")
        for label, values in zip(index, data.itertuples(index=False, name=None)):
            start = utc_timestamp(label)
            if (start.minute % 5 or start.second or start.microsecond):
                raise ValueError("Vela de 5m fuera de rejilla.")
            end = start + pd.Timedelta("5min")
            if is_regular_close(end):
                records.append((start, end, values, "5m"))
    elif timeframe == "1d":
        for label, values in zip(index, data.itertuples(index=False, name=None)):
            day = pd.Timestamp(pd.Timestamp(label).date())
            schedule = _calendar(day.year - 1, day.year + 1).schedule
            if day not in schedule.index:
                continue
            records.append((utc_timestamp(schedule.loc[day, "open"]),
                            utc_timestamp(schedule.loc[day, "close"]), values, "1d"))
    else:
        raise ValueError("Temporalidad de resolución desconocida.")
    checked = []
    for start, end, values, source in records:
        o, h, l, c, v = map(float, values)
        if not all(math.isfinite(x) for x in (o, h, l, c, v)) or min(o, h, l, c) <= 0 or v < 0:
            raise ValueError("OHLCV no finito o fuera de dominio.")
        if l > min(o, c) or h < max(o, c) or l > h:
            raise ValueError("OHLC inconsistente.")
        checked.append((start, end, o, h, l, c, v, source))
    return checked


def _contract_seed(contract):
    if isinstance(contract, dict):
        identity = contract.get("contract_sha256")
    else:
        identity = canonical({
            "symbol": contract.symbol,
            "direction": contract.direction,
            "observed_at": utc_timestamp(contract.observed_at).isoformat(),
            "source_bar_at": utc_timestamp(contract.source_bar_at).isoformat(),
            "expires_at": utc_timestamp(contract.expires_at).isoformat(),
            "reference_price": str(contract.reference_price),
            "take_profit": str(contract.take_profit),
            "stop_loss": str(contract.stop_loss),
        })
    return hashlib.sha256(canonical({
        "contract": identity,
        "evidence_policy": EVIDENCE_POLICY,
        "labeler_semantics": LABELER_SEMANTICS,
    }).encode()).hexdigest()


def _next_evidence_hash(previous, record):
    return hashlib.sha256((previous + canonical(record)).encode()).hexdigest()


def scan_operational_outcome(
    contract, bars_5m, as_of, daily_bars=None, *, resume_at=None,
    previous_evidence_sha256=None, previous_evidence_count=0,
):
    """Resolve causally; return None while the observable evidence is incomplete.

    5m evidence has priority for any session it covers. Daily evidence is used
    only for later complete sessions and applies the declared conservative
    same-bar policy. The observation session is never reconstructed from a
    daily candle because it contains prices from before entry.
    """
    if isinstance(contract, OperationalContract):
        observed = utc_timestamp(contract.observed_at)
        normalized = {
            "side": contract.direction.upper(),
            "entry_price": float(contract.reference_price),
            "stop_loss": float(contract.stop_loss),
            "take_profit": float(contract.take_profit),
            "entry_at": utc_timestamp(contract.source_bar_at).isoformat(),
            "observed_at": observed.isoformat(),
            "evaluation_starts_at": first_evaluable_open(observed).isoformat(),
            "timeout_at": utc_timestamp(contract.expires_at).isoformat(),
        }
    else:
        validate_operational_contract(contract)
        normalized = contract
    now = utc_timestamp(as_of)
    entry_at = utc_timestamp(normalized["entry_at"])
    evaluation_start = utc_timestamp(normalized["evaluation_starts_at"])
    timeout = utc_timestamp(normalized["timeout_at"])
    scan_start = utc_timestamp(resume_at) if resume_at is not None else evaluation_start
    if not evaluation_start <= scan_start <= timeout:
        raise ValueError("Cursor operativo fuera del contrato.")
    prior_count = int(previous_evidence_count)
    if prior_count < 0:
        raise ValueError("Conteo de evidencia operativo inválido.")
    if previous_evidence_sha256 is None:
        if prior_count:
            raise ValueError("Checkpoint operativo incompleto.")
        chain = _contract_seed(contract)
    else:
        chain = str(previous_evidence_sha256)
        if (prior_count <= 0 or len(chain) != 64
                or any(char not in "0123456789abcdef" for char in chain)):
            raise ValueError("Hash de checkpoint operativo inválido.")
    cutoff = min(now, timeout)
    intraday = _validated_bars(bars_5m, "5m")
    daily = _validated_bars(daily_bars, "1d")
    entry_day = entry_at.tz_convert(NY).date()
    intraday_by_day = {}
    for row in intraday:
        if row[0] >= scan_start and row[1] <= cutoff:
            intraday_by_day.setdefault(row[1].tz_convert(NY).date(), []).append(row)
    daily_by_day = {row[1].tz_convert(NY).date(): row for row in daily
                    if row[0] >= scan_start and row[1] <= cutoff}
    schedule = _calendar(scan_start.year - 1, cutoff.year + 1).schedule
    evidence = []
    for _, session in schedule.loc[
        str(scan_start.tz_convert(NY).date()):str(cutoff.tz_convert(NY).date())
    ].iterrows():
        session_open, session_close = utc_timestamp(session.open), utc_timestamp(session.close)
        start = max(scan_start, session_open)
        end = min(cutoff, session_close)
        if end < start + pd.Timedelta("5min"):
            continue
        expected = pd.date_range(start, end - pd.Timedelta("5min"), freq="5min", tz="UTC")
        day = session_close.tz_convert(NY).date()
        actual = intraday_by_day.get(day, [])
        actual_starts = pd.DatetimeIndex([row[0] for row in actual])
        if len(actual_starts) == len(expected) and actual_starts.equals(expected):
            evidence.extend(actual)
            continue
        # A complete future daily candle can conservatively replace a missing
        # intraday path. Never use the observation day's daily high/low because
        # it contains prices from before the forecast.
        if session_close <= cutoff and day != entry_day and day in daily_by_day:
            evidence.append(daily_by_day[day])
            continue
        raise IncompleteOperationalEvidence(
            "Cobertura XNYS incompleta; first-hit permanece pendiente."
        )
    evidence.sort(key=lambda row: row[1])
    side = normalized["side"]
    stop, target = float(normalized["stop_loss"]), float(normalized["take_profit"])
    count = prior_count
    processed = []
    scanned_through = scan_start.isoformat() if resume_at is not None else None
    for start, end, opening, high, low, close, volume, source in evidence:
        record = {
            "start": start.isoformat(), "end": end.isoformat(), "open": opening,
            "high": high, "low": low, "close": close, "volume": volume,
            "timeframe": source,
        }
        chain = _next_evidence_hash(chain, record)
        count += 1
        processed.append((start, end, opening, high, low, close, volume, source))
        scanned_through = end.isoformat()
        if side == "LONG":
            if opening <= stop:
                result = OperationalResult(OperationalOutcome.SL_FIRST, opening, end.isoformat(), f"{source}:gap-open", chain)
                return OperationalScan(result, scanned_through, chain, count, tuple(processed))
            if opening >= target:
                result = OperationalResult(OperationalOutcome.TP_FIRST, target, end.isoformat(), f"{source}:gap-target", chain)
                return OperationalScan(result, scanned_through, chain, count, tuple(processed))
            if low <= stop:  # also wins the deliberately conservative same-bar tie
                result = OperationalResult(OperationalOutcome.SL_FIRST, stop, end.isoformat(), f"{source}:barrier", chain)
                return OperationalScan(result, scanned_through, chain, count, tuple(processed))
            if high >= target:
                result = OperationalResult(OperationalOutcome.TP_FIRST, target, end.isoformat(), f"{source}:barrier", chain)
                return OperationalScan(result, scanned_through, chain, count, tuple(processed))
        else:
            if opening >= stop:
                result = OperationalResult(OperationalOutcome.SL_FIRST, opening, end.isoformat(), f"{source}:gap-open", chain)
                return OperationalScan(result, scanned_through, chain, count, tuple(processed))
            if opening <= target:
                result = OperationalResult(OperationalOutcome.TP_FIRST, target, end.isoformat(), f"{source}:gap-target", chain)
                return OperationalScan(result, scanned_through, chain, count, tuple(processed))
            if high >= stop:
                result = OperationalResult(OperationalOutcome.SL_FIRST, stop, end.isoformat(), f"{source}:barrier", chain)
                return OperationalScan(result, scanned_through, chain, count, tuple(processed))
            if low <= target:
                result = OperationalResult(OperationalOutcome.TP_FIRST, target, end.isoformat(), f"{source}:barrier", chain)
                return OperationalScan(result, scanned_through, chain, count, tuple(processed))
    if now < timeout:
        return OperationalScan(None, scanned_through, chain, count, tuple(processed))
    exact = next((row for row in evidence if row[1] == timeout), None)
    if exact is None:
        return OperationalScan(None, scanned_through, chain, count, tuple(processed))
    result = OperationalResult(
        OperationalOutcome.TIMEOUT, exact[5], timeout.isoformat(),
        f"{exact[7]}:timeout-close", chain,
    )
    return OperationalScan(result, scanned_through, chain, count, tuple(processed))


def resolve_operational_outcome(contract, bars_5m, as_of, daily_bars=None):
    """Compatibility wrapper for one-shot callers and pure unit tests."""
    return scan_operational_outcome(
        contract, bars_5m, as_of, daily_bars=daily_bars,
    ).result


def operational_checkpoint_digest(
    parent_observation_sha256, contract_sha256, scanned_through,
    evidence_sha256, evidence_count, evidence_json, updated_at,
):
    payload = {
        "observation_sha256": parent_observation_sha256,
        "contract_sha256": contract_sha256,
        "scanned_through": utc_timestamp(scanned_through).isoformat(),
        "evidence_sha256": evidence_sha256,
        "evidence_count": int(evidence_count),
        "evidence_json": evidence_json,
        "updated_at": utc_timestamp(updated_at).isoformat(),
    }
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


def operational_outcome_digest(parent_observation_sha256, contract_sha256, result, resolved_at,
                               evidence_json="{}"):
    payload = {
        "observation_sha256": parent_observation_sha256,
        "contract_sha256": contract_sha256,
        "outcome": result.outcome.value,
        "exit_price": str(result.exit_price),
        "exit_at": result.exit_at,
        "exit_source": result.exit_source,
        "evidence_sha256": result.evidence_sha256,
        "evidence_json": evidence_json,
        "resolved_at": utc_timestamp(resolved_at).isoformat(),
    }
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


def valid_operational_outcome(row, parent):
    """Verify child result against the immutable signed parent contract."""
    try:
        import json
        contract = json.loads(parent["parameters_json"])["operational_contract"]
        validate_operational_contract(contract)
        if (row["target_version"] != contract["version"]
                or row["contract_sha256"] != contract["contract_sha256"]):
            return False
        result_fields = ("outcome", "exit_price", "exit_at", "exit_source",
                         "evidence_sha256", "evidence_json",
                         "resolved_at", "outcome_sha256")
        checkpoint_fields = (
            "scanned_through", "scan_evidence_sha256", "scan_evidence_json",
            "checkpoint_updated_at", "checkpoint_sha256",
        )
        count = int(row["scan_evidence_count"] or 0)
        checkpoint_empty = count == 0 and all(row[name] is None for name in checkpoint_fields)
        checkpoint_complete = count > 0 and all(row[name] is not None for name in checkpoint_fields)
        if not (checkpoint_empty or checkpoint_complete):
            return False
        if checkpoint_complete:
            scanned = utc_timestamp(row["scanned_through"])
            evaluation_start = utc_timestamp(contract["evaluation_starts_at"])
            timeout = utc_timestamp(contract["timeout_at"])
            updated = utc_timestamp(row["checkpoint_updated_at"])
            scan_sha = row["scan_evidence_sha256"]
            scan_json = row["scan_evidence_json"]
            scan_payload = json.loads(scan_json) if isinstance(scan_json, str) else None
            if (not evaluation_start < scanned <= timeout or updated < scanned
                    or not isinstance(scan_sha, str) or len(scan_sha) != 64
                    or any(char not in "0123456789abcdef" for char in scan_sha)
                    or not isinstance(scan_json, str)
                    or canonical(scan_payload) != scan_json
                    or row["checkpoint_sha256"] != operational_checkpoint_digest(
                        parent["observation_sha256"], row["contract_sha256"], scanned,
                        scan_sha, count, scan_json, updated,
                    )):
                return False
            execution = scan_payload.get("execution_assessment") if isinstance(scan_payload, dict) else None
            if isinstance(execution, dict) and "scanned_through" in execution:
                from .execution_path import validate_execution_checkpoint
                validate_execution_checkpoint(execution, contract, scanned)
        if row["resolution_status"] == "PENDING":
            return all(row[name] is None for name in result_fields)
        if row["resolution_status"] != "RESOLVED":
            return False
        if not checkpoint_complete:
            return False
        outcome = OperationalOutcome(row["outcome"])
        price = _finite_price(row["exit_price"], "Salida")
        exit_at = utc_timestamp(row["exit_at"])
        resolved = utc_timestamp(row["resolved_at"])
        entry_at = utc_timestamp(contract["entry_at"])
        timeout = utc_timestamp(contract["timeout_at"])
        if not entry_at < exit_at <= timeout or resolved < exit_at or not row["exit_source"]:
            return False
        if outcome is OperationalOutcome.TIMEOUT and exit_at != timeout:
            return False
        evidence_sha = row["evidence_sha256"]
        if (not isinstance(evidence_sha, str) or len(evidence_sha) != 64
                or any(char not in "0123456789abcdef" for char in evidence_sha)):
            return False
        import json
        evidence_json = row["evidence_json"]
        if not isinstance(evidence_json, str) or canonical(json.loads(evidence_json)) != evidence_json:
            return False
        if (evidence_sha != row["scan_evidence_sha256"]
                or evidence_json != row["scan_evidence_json"]
                or exit_at != utc_timestamp(row["scanned_through"])):
            return False
        result = OperationalResult(outcome, price, exit_at.isoformat(), row["exit_source"], evidence_sha)
        return row["outcome_sha256"] == operational_outcome_digest(
            parent["observation_sha256"], row["contract_sha256"], result, resolved,
            evidence_json,
        )
    except (KeyError, TypeError, ValueError, OverflowError, json.JSONDecodeError):
        return False
