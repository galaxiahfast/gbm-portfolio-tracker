"""Regularized models remain causal, horizon-specific and fail closed."""
from __future__ import annotations

from dataclasses import replace
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from portfolio_tracker.analytics.horizon_models import (
    FEATURE_NAMES,
    FeatureContractMismatch,
    MODEL_FEATURE_VERSION,
    HorizonSample,
    TrainingConfig,
    chronological_purged_split,
    feature_vector,
    predict_uncalibrated_scores,
    train_horizon_samples,
    horizon_samples,
    model_feature_contract,
    replay_population_report,
    train_replay_horizon_models,
    validate_horizon_model_artifact,
)
from portfolio_tracker.analytics.operational_target import TARGET_VERSION
from portfolio_tracker.analytics.nested_walk_forward import (
    run_nested_walk_forward_replay,
    validate_nested_walk_forward_artifact,
)
from portfolio_tracker.analytics.historical_replay import (
    HISTORICAL_REPLAY_CONTRACT,
    _sha as replay_sha,
)
from portfolio_tracker.services.directional_collection import HORIZON_MINUTES
from portfolio_tracker.services.model_execution_record import (
    build_replay_snapshot, prediction_snapshot, technical_horizon,
)
from portfolio_tracker.analytics.fundamental_news import apply_fundamental_filter
from portfolio_tracker.analytics import decision_engines
from portfolio_tracker.analytics.causal_core import CausalDecision, Permission, Regime, Setup, Trigger
from tests.test_fundamental_news import OBSERVED, _negative_snapshot
from tests.test_pdf_report import _analysis


def _samples(size=330):
    rows = []
    for index in range(size):
        observed = pd.Timestamp("2020-01-02", tz="UTC") + pd.DateOffset(days=index)
        category = index % 3
        values = np.zeros(len(FEATURE_NAMES), dtype=float)
        # A stable, learnable relationship present across every chronological block.
        values[0] = (-2.5, 2.5, 0.0)[category]
        values[1] = (0.2, 0.3, 0.8)[category]
        values[2] = np.sin(index / 7.0) * 0.05
        rows.append(HorizonSample(
            observed_at=observed.isoformat(),
            label_available_at=(observed + pd.DateOffset(hours=1)).isoformat(),
            features=tuple(values),
            outcome=("TP_FIRST", "SL_FIRST", "TIMEOUT")[category],
            cut_id=f"cut-{index}",
            eligible_at_emission=True,
            feature_version=MODEL_FEATURE_VERSION,
            available_features=tuple(FEATURE_NAMES),
        ))
    return tuple(rows)


def _config():
    return TrainingConfig(
        minimum_samples=300,
        minimum_class_samples=20,
        minimum_holdout_samples=60,
        maximum_iterations=500,
    )


def test_regularized_model_is_per_horizon_and_scores_are_explicitly_uncalibrated():
    rows = _samples()
    one_hour = train_horizon_samples("1 Hora", rows, _config())
    one_day = train_horizon_samples("1 Día", rows, _config())
    assert one_hour["horizon_minutes"] == 60
    assert one_day["horizon_minutes"] == 1_440
    assert one_hour["model_sha256"] != one_day["model_sha256"]
    assert one_hour["status"] == "TRAINED_OOS_UNCALIBRATED"
    assert one_hour["promotable"] is True
    assert one_hour["metrics"]["source"] == "HOLDOUT_ONLY"
    assert one_hour["metrics"]["multiclass_brier"] < one_hour["metrics"]["baseline_brier"]
    prediction = predict_uncalibrated_scores(one_hour, rows[-1].features)
    assert prediction["semantics"] == "REGULARIZED_SCORE_UNCALIBRATED"
    assert prediction["sum"] == pytest.approx(100.0)
    assert set(prediction["scores"]) == {"TP_FIRST", "SL_FIRST", "TIMEOUT"}


def test_l2_selection_does_not_look_at_holdout_labels():
    original = _samples()
    split_at = int(len(original) * 0.8)
    changed = original[:split_at] + tuple(
        replace(row, outcome="SL_FIRST" if row.outcome != "SL_FIRST" else "TP_FIRST")
        for row in original[split_at:]
    )
    before = train_horizon_samples("6 Horas", original, _config())
    after = train_horizon_samples("6 Horas", changed, _config())
    assert before["selection"] == after["selection"]
    assert before["model"] == after["model"]
    assert before["metrics"] != after["metrics"]


def test_chronological_split_purges_labels_unknown_at_next_boundary():
    rows = list(_samples(30))
    # Both belong to training chronologically, but their outcomes become known
    # only after calibration has begun.
    rows[0] = replace(rows[0], label_available_at=rows[20].observed_at)
    rows[1] = replace(rows[1], label_available_at=rows[29].observed_at)
    training, calibration, holdout, purged = chronological_purged_split(rows)
    assert (len(training), len(calibration), len(holdout)) == (16, 6, 6)
    assert purged == {"training": 2, "calibration": 0}
    assert max(pd.Timestamp(row.observed_at) for row in training) < min(
        pd.Timestamp(row.observed_at) for row in calibration
    )


def test_small_replay_cannot_create_or_promote_a_model():
    rejected = train_horizon_samples("1 Semana", _samples(60), _config())
    assert rejected["status"] == "INSUFFICIENT_DATA"
    assert rejected["promotable"] is False
    assert rejected["model"] is None
    assert rejected["score_semantics"] == "NO_SCORE_MODEL_NOT_TRAINED"
    with pytest.raises(ValueError, match="no está aprobado"):
        predict_uncalibrated_scores(rejected, rejected.get("features", ()))


def test_invalid_feature_shape_and_unknown_horizon_fail_closed():
    bad = (replace(_samples(1)[0], features=(1.0, 2.0)),) * 300
    with pytest.raises(ValueError, match="Longitud de features"):
        train_horizon_samples("1 Hora", bad, _config())
    with pytest.raises(ValueError, match="Horizonte desconocido"):
        train_horizon_samples("Todos", _samples(), _config())


def _minimal_signed_replay():
    observed = "2026-09-01T15:00:00+00:00"
    horizons = []
    for label, minutes in HORIZON_MINUTES.items():
        horizons.append({
            "label": label,
            "horizon_minutes": minutes,
            "prediction": {},
            "scenario_contract": {},
            "scenario_result": {"status": "RIGHT_CENSORED"},
            "operational_contract": {},
            "operational_result": {
                "status": "RESOLVED",
                "outcome": "TIMEOUT",
                "exit_at": "2026-09-01T16:00:00+00:00",
            },
        })
    cut = {
        "cut_id": "cut-1",
        "symbol": "SMCI",
        "observed_at": observed,
        "source_bar_closed_at": observed,
        "dataset_sha256": "dataset",
        "feature_snapshot": {"features": {}},
        "horizons": horizons,
        "limitations": ["HISTORICAL_REPLAY_NOT_LIVE_OOS"],
    }
    cut["cut_sha256"] = replay_sha(cut)
    deterministic = {
        "contract": HISTORICAL_REPLAY_CONTRACT,
        "symbol": "SMCI",
        "dataset_sha256": "dataset",
        "peer_dataset_sha256": None,
        "parameters": {"minimum_probability": 0.55, "stop_atr_multiple": 2.25, "risk_per_trade_pct": 1.0},
        "requested_start": None,
        "requested_end": None,
        "max_cuts": None,
        "candidate_cuts": 1,
        "observations": [cut],
        "rejected": {},
        "separation_policy": "REPLAY_NEVER_COUNTS_AS_LIVE_OOS",
    }
    replay = {
        **deterministic,
        "replay_id": replay_sha({
            "contract": HISTORICAL_REPLAY_CONTRACT,
            "symbol": "SMCI",
            "dataset_sha256": "dataset",
            "parameters": deterministic["parameters"],
            "requested_start": None,
            "requested_end": None,
            "max_cuts": None,
        }),
        "generated_at": "2026-09-02T00:00:00+00:00",
    }
    replay["content_sha256"] = replay_sha(deterministic)
    replay["artifact_sha256"] = replay_sha(replay)
    return replay


def test_six_model_artifact_is_signed_and_tamper_evident():
    artifact = train_replay_horizon_models(_minimal_signed_replay())
    assert validate_horizon_model_artifact(artifact)
    assert [row["horizon"] for row in artifact["models"]] == list(HORIZON_MINUTES)
    assert all(row["status"] == "EVIDENCIA_INSUFICIENTE" for row in artifact["models"])
    assert all(row["population"]["executable_entries"]["n"] == 0 for row in artifact["models"])
    assert all(row["population"]["barrier_hypotheses"]["resolved_n"] == 1 for row in artifact["models"])
    artifact["models"][0]["resolved_samples"] = 999
    with pytest.raises(ValueError, match="Firma de modelo"):
        validate_horizon_model_artifact(artifact)


def _replay_with_population(eligible_indices=(), symbol="SMCI"):
    replay = deepcopy(_minimal_signed_replay())
    replay["symbol"] = symbol
    replay["observations"][0]["feature_snapshot"]["model_feature_version"] = MODEL_FEATURE_VERSION
    replay["observations"][0]["feature_snapshot"]["features"]["market_regime"] = "NO_TRADE"
    cuts = []
    for index in range(20):
        cut = deepcopy(replay["observations"][0])
        cut["cut_id"] = f"cut-{index}"
        observed = pd.Timestamp("2026-08-03T15:00:00Z") + pd.Timedelta(days=index)
        cut["observed_at"] = observed.isoformat()
        cut["source_bar_closed_at"] = observed.isoformat()
        for horizon in cut["horizons"]:
            horizon["operational_contract"] = {
                "version": TARGET_VERSION,
                "eligible_at_emission": index in eligible_indices,
                "side": "LONG", "entry_price": 100.0, "stop_loss": 98.0,
            }
            horizon["operational_result"].update({
                "outcome": ("TP_FIRST" if index % 2 == 0 else "SL_FIRST"),
                "exit_price": (104.0 if index % 2 == 0 else 97.0),
                "exit_at": (observed + pd.Timedelta(hours=1)).isoformat(),
            })
            horizon["model_feature_contract"] = model_feature_contract(
                {"feature_snapshot": cut["feature_snapshot"]}, horizon,
            )
        cut["cut_sha256"] = replay_sha({key: value for key, value in cut.items() if key != "cut_sha256"})
        cuts.append(cut)
    replay["observations"] = cuts
    replay["candidate_cuts"] = len(cuts)
    replay["replay_id"] = replay_sha({
        "contract": replay["contract"], "symbol": symbol,
        "dataset_sha256": replay["dataset_sha256"], "parameters": replay["parameters"],
        "requested_start": None, "requested_end": None, "max_cuts": None,
    })
    deterministic = {key: replay[key] for key in (
        "contract", "symbol", "dataset_sha256", "peer_dataset_sha256", "parameters",
        "requested_start", "requested_end", "max_cuts", "candidate_cuts",
        "observations", "rejected", "separation_policy",
    )}
    replay["content_sha256"] = replay_sha(deterministic)
    replay["artifact_sha256"] = replay_sha({key: value for key, value in replay.items() if key != "artifact_sha256"})
    return replay


@pytest.mark.parametrize("symbol", ["SMCI", "NVDA"])
def test_hypothetical_barriers_without_authorized_entry_never_train(symbol):
    replay = _replay_with_population(symbol=symbol)
    report = replay_population_report(replay, "1 Hora")
    assert report["barrier_hypotheses"]["resolved_n"] == 20
    assert report["executable_entries"]["n"] == 0
    assert report["executable_entries"]["net_expectancy_per_share"] is None
    assert horizon_samples(replay, "1 Hora") == ()
    artifact = train_replay_horizon_models(replay)
    assert all(row["status"] == "EVIDENCIA_INSUFICIENTE" for row in artifact["models"])
    assert all(not row["promotable"] and row["model"] is None for row in artifact["models"])
    assert validate_horizon_model_artifact(artifact)
    walk_forward = run_nested_walk_forward_replay(replay)
    assert all(row["status"] == "EVIDENCIA_INSUFICIENTE" for row in walk_forward["results"])
    assert all(row["final_holdout"]["status"] == "SEALED_UNOPENED" for row in walk_forward["results"])
    assert validate_nested_walk_forward_artifact(walk_forward)


def test_only_eligible_entries_contribute_class_hits_and_net_expectancy():
    replay = _replay_with_population(eligible_indices=(0, 1))
    report = replay_population_report(replay, "1 Hora")
    assert report["barrier_hypotheses"]["resolved_n"] == 20
    assert report["executable_entries"]["n"] == 2
    assert report["executable_entries"]["class_hits"] == {
        "TP_FIRST": 1, "SL_FIRST": 1, "TIMEOUT": 0,
    }
    assert report["executable_entries"]["net_wins"] == 1
    assert report["executable_entries"]["net_expectancy_per_share"] == pytest.approx(-0.1015)
    assert len(horizon_samples(replay, "1 Hora")) == 2
    rejected = train_horizon_samples("1 Hora", (
        replace(row, eligible_at_emission=False) for row in _samples()
    ))
    assert rejected["resolved_samples"] == 0
    assert rejected["promotable"] is False


def test_undeclared_eligibility_fails_closed_before_training():
    rows = tuple(replace(row, eligible_at_emission=False) for row in _samples())
    assert train_horizon_samples("1 Hora", rows)["resolved_samples"] == 0


def test_replay_and_live_model_features_match_before_fundamental_filter():
    technical = _analysis()
    live = apply_fundamental_filter(technical, _negative_snapshot(), now=OBSERVED)
    assert live.fundamental_risk_veto is True
    replay_snapshot = build_replay_snapshot(
        technical, observed_at=OBSERVED, protocol="TEST_HISTORICAL_REPLAY",
    )
    live_snapshot = build_replay_snapshot(
        live, observed_at=OBSERVED, protocol="TEST_LIVE",
    )
    assert replay_snapshot["features"]["fundamental_score"] != live_snapshot["features"]["fundamental_score"]
    assert replay_snapshot["model_features"] == live_snapshot["model_features"]
    with pytest.raises(ValueError, match="vista técnica previa"):
        build_replay_snapshot(
            replace(live, model_technical_analysis=None),
            observed_at=OBSERVED, protocol="TEST_UNSAFE_LIVE",
        )
    assert "fundamental_score" not in FEATURE_NAMES
    assert "fundamental_risk_veto" not in FEATURE_NAMES
    contract = {
        "entry_price": technical.last_price,
        "stop_loss": technical.execution_levels.stop_loss,
        "take_profit": technical.execution_levels.take_profit_1,
    }
    horizon_replay = {
        "prediction": prediction_snapshot(technical_horizon(technical, "1 Hora")),
        "operational_contract": contract,
    }
    horizon_live = {
        "prediction": prediction_snapshot(next(
            item for item in live.horizon_projections if item.label == "1 Hora"
        )),
        "model_prediction": prediction_snapshot(technical_horizon(live, "1 Hora")),
        "operational_contract": contract,
    }
    replay_cut = {"feature_snapshot": replay_snapshot}
    live_cut = {"feature_snapshot": live_snapshot}
    assert model_feature_contract(replay_cut, horizon_replay) == model_feature_contract(live_cut, horizon_live)
    assert np.allclose(
        feature_vector(replay_cut, horizon_replay), feature_vector(live_cut, horizon_live),
        equal_nan=True,
    )


def test_training_rejects_different_feature_availability():
    rows = _samples()
    altered = (*rows[:-1], replace(rows[-1], available_features=rows[-1].available_features[:-1]))
    with pytest.raises(FeatureContractMismatch, match="cobertura histórica"):
        train_horizon_samples("1 Hora", altered)


def test_market_regime_is_not_overwritten_by_permission(monkeypatch):
    monkeypatch.setattr(decision_engines, "evaluate_causal_core", lambda **_: CausalDecision(
        Regime(Permission.LONG_ONLY, True, ""),
        Setup(False, False, 0.0, 0.0, ""),
        Trigger("NONE", False, ""),
    ))
    analysis = decision_engines.apply_hierarchy(_analysis())
    assert analysis.market_regime == "TREND"
    assert analysis.macro_permission == "LONG_ONLY"
    snapshot = build_replay_snapshot(analysis, observed_at=OBSERVED, protocol="TEST_REGIME")
    assert snapshot["model_features"]["market_regime"] == "TREND"
    snapshot["model_features"]["market_regime"] = "LONG_ONLY"
    with pytest.raises(FeatureContractMismatch, match="market_regime"):
        model_feature_contract(
            {"feature_snapshot": snapshot},
            {"prediction": prediction_snapshot(analysis.horizon_projections[0]),
             "operational_contract": {}},
        )


def test_missing_historical_fields_are_not_encoded_as_known_values():
    analysis = _analysis()
    snapshot = build_replay_snapshot(analysis, observed_at=OBSERVED, protocol="TEST_MISSING_FEATURE")
    snapshot["model_features"]["weekly_trend"] = None
    snapshot["model_features"]["risk_veto"] = None
    manifest = model_feature_contract(
        {"feature_snapshot": snapshot},
        {"prediction": prediction_snapshot(analysis.horizon_projections[0]),
         "operational_contract": {
             "entry_price": analysis.last_price,
             "stop_loss": analysis.execution_levels.stop_loss,
             "take_profit": analysis.execution_levels.take_profit_1,
         }},
    )
    assert "weekly_trend=UNKNOWN" not in manifest["available_features"]
    assert "risk_veto" not in manifest["available_features"]
    assert len(manifest["available_features"]) < len(FEATURE_NAMES)
