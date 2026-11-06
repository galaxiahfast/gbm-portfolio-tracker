"""Explicit research-only clock bridge. No mutation of any production module.

Production rejects every actionable emission outside 11:00. The diagnostic
experiment requests other times, but MUST NOT change that security boundary.
We therefore bind the identical execution function bytecode to a PRIVATE
namespace whose validator accepts only a signed preregistered research clock.
All prices, fills, gap policies, costs, indicators and risk gates are unchanged.
These artifacts cannot pass the production validator outside its live clock.
"""
from copy import deepcopy
import hashlib
from types import FunctionType

import pandas as pd

from portfolio_tracker.analytics import execution_path
from portfolio_tracker.analytics.operational_target import validate_operational_contract
from portfolio_tracker.services.model_observations import canonical

SCOPE = "RESEARCH_TECHNICAL_ONLY_NOT_LIVE_OR_BROKER_AUTHORIZATION"


def signed_hash(contract):
    return hashlib.sha256(canonical({k: v for k, v in contract.items()
                                     if k != "contract_sha256"}).encode()).hexdigest()


def research_contract(original, *, eligible, protocol, cut):
    """Only change the research clock eligibility, retaining the signed origin."""
    validate_operational_contract(original)
    result = deepcopy(original)
    result["research_clock"] = {
        "scope": SCOPE, "protocol_sha256": protocol["protocol_sha256"],
        "cut_at": pd.Timestamp(cut).isoformat(),
        "live_eligible_at_emission": original["eligible_at_emission"],
        "production_contract_sha256": original["contract_sha256"],
    }
    result["eligible_at_emission"] = bool(eligible)
    result["contract_sha256"] = signed_hash(result)
    validate_research_contract(result, protocol)
    return result


def validate_research_contract(contract, protocol):
    clock = contract.get("research_clock", {})
    if (clock.get("scope") != SCOPE or clock.get("protocol_sha256") != protocol["protocol_sha256"]
            or contract.get("contract_sha256") != signed_hash(contract)
            or clock.get("cut_at") != contract["observed_at"]
            or contract["entry_at"] != contract["observed_at"]):
        raise ValueError("Contrato de reloj de investigación inválido.")
    cut = pd.Timestamp(clock["cut_at"]).tz_convert("America/New_York")
    allowed = {c["time_ny"] for c in protocol["cohorts"]}
    if (cut.strftime("%H:%M") not in allowed or cut.second or cut.microsecond
            or not protocol["start"] <= cut.date().isoformat() <= protocol["end"]):
        raise ValueError("El reloj no pertenece al protocolo preregistrado.")
    origin = deepcopy(contract)
    origin.pop("research_clock")
    origin["eligible_at_emission"] = clock["live_eligible_at_emission"]
    origin["contract_sha256"] = signed_hash(origin)
    if origin["contract_sha256"] != clock["production_contract_sha256"]:
        raise ValueError("Se modificó algo más que la elegibilidad de reloj.")
    # Validates exact maturity, next-full-bar start, OHLC brackets, costs and
    # model identity against the original production functions, unmodified.
    return validate_operational_contract(origin)


def assess_research_execution(contract, dataset, protocol):
    """Original fill rules in an isolated namespace, never global monkeypatches."""
    validator = lambda value: validate_research_contract(value, protocol)
    namespace = dict(execution_path.advance_execution_checkpoint.__globals__)
    namespace["validate_operational_contract"] = validator
    original = execution_path.advance_execution_checkpoint
    advance = FunctionType(original.__code__, namespace, original.__name__,
                           original.__defaults__, original.__closure__)
    advance.__kwdefaults__ = original.__kwdefaults__
    validator(contract)
    if not contract["eligible_at_emission"]:
        return execution_path.ExecutionAssessment("NOT_AUTHORIZED").as_record()
    start = pd.Timestamp(contract["evaluation_starts_at"])
    end = min(pd.Timestamp(contract["timeout_at"]), dataset.as_of)
    bars = dataset.intraday.loc[(dataset.intraday.index >= start) & (dataset.intraday.index < end)]
    # The immutable, original contract has the same barriers/path clock. Its
    # scanner does not require eligibility, keeping hypotheses separate.
    origin = deepcopy(contract)
    clock = origin.pop("research_clock")
    origin["eligible_at_emission"] = clock["live_eligible_at_emission"]
    origin["contract_sha256"] = signed_hash(origin)
    from portfolio_tracker.analytics.operational_target import scan_operational_outcome
    try:
        scan = scan_operational_outcome(origin, bars, dataset.as_of)
    except execution_path.IncompleteOperationalEvidence as exc:
        return execution_path.ExecutionAssessment("EVIDENCE_INCOMPLETE", detail=str(exc)).as_record()
    if not scan.observed_bars:
        return execution_path.ExecutionAssessment("PENDING_ENTRY").as_record()
    record = advance(contract, scan.observed_bars, scan.scanned_through)
    return {key: record[key] for key in execution_path.ExecutionAssessment.__dataclass_fields__}
