"""Isolated replay variants, no shared-core mutations or holdout reuse."""
from copy import deepcopy
import os

import numpy as np
import pandas as pd
import pytest

from scripts.trigger_experiment_rules import evaluate_variants, clock_reference, nonoverlapping_events, statistics
from scripts.seal_trigger_experiment_holdout import protect_key
from portfolio_tracker.analytics.holdout_registry import reserve_holdout_opening, HoldoutAlreadyOpenedError


def fixture(permission="LONG_ONLY", votes="1, 1, 1", volume=200., eligible=True):
    index=pd.date_range("2024-07-01 09:00",periods=25,freq="5min",tz="America/New_York")
    b=pd.DataFrame({"Close":100.,"High":100.,"Low":100.,"Volume":100.,"StochRSI_K":30.,"StochRSI_D":40.,"MACD":0.,"MACD_signal":0.},index=index)
    b.iloc[-1,b.columns.get_loc("Volume")]=volume
    b.iloc[-1,b.columns.get_loc("StochRSI_K")]=50
    f={"risk_veto":False,"score_breakdown":[],"probability_up":85.,"macro_trending":True,"macro_permission":permission,
       "exposure_factor":.25 if permission=="BOTH_REDUCED" else 1.,"hierarchy_detail":f"Votos S/D/4h=[{votes}]; Estructura 4h/1h=[1, 1]"}
    g={"regime_long_only":permission=="LONG_ONLY","regime_short_only":permission=="SHORT_ONLY","setup_long":True,"setup_short":False}
    r={"symbol":"SMCI","cut_at":"2024-07-01T15:05:00+00:00","feature_snapshot":{"features":f},"gates":g,
       "horizons":[{"operational_contract":{"side":"LONG","eligible_at_emission":eligible}}]}
    return r,b


def test_volume_percentile_changes_only_volume_gate():
    r,b=fixture(volume=100,eligible=False)
    before=deepcopy(r)
    x=evaluate_variants(r,b,80.)
    assert x["VOLUME_P50_CLOCK"]["eligible"]
    assert not x["CURRENT"]["eligible"]
    assert r==before
    assert x["VOLUME_P50_CLOCK"]["non_temporal"]==x["CURRENT"]["non_temporal"]


def test_stoch_tolerance_changes_only_crossover():
    r,b=fixture(eligible=False)
    b.iloc[-2,b.columns.get_loc("StochRSI_K")]=50
    b.iloc[-1,b.columns.get_loc("StochRSI_K")]=60
    x=evaluate_variants(r,b,100)
    assert x["STOCH_ONE_BAR"]["eligible"]
    assert not x["CURRENT"]["eligible"]
    assert not x["VOLUME_P50_CLOCK"]["eligible"]


def test_majority_two_positive_votes_retains_other_filters():
    r,b=fixture("BOTH_REDUCED","1, 1, -1",eligible=False)
    x=evaluate_variants(r,b,100)
    assert x["LONG_MAJORITY"]["eligible"]
    assert x["LONG_MAJORITY"]["exposure"]==.25
    r["feature_snapshot"]["features"]["macro_trending"]=False
    assert not evaluate_variants(r,b,100)["LONG_MAJORITY"]["eligible"]


def test_both_reduced_exposure_cap_is_effective():
    r,b=fixture("BOTH_REDUCED","1, 0, -1",eligible=False)
    r["feature_snapshot"]["features"]["exposure_factor"]=1.
    x=evaluate_variants(r,b,100)
    assert x["BOTH_REDUCED"]["eligible"]
    assert x["BOTH_REDUCED"]["exposure"]==.25
    assert not x["LONG_MAJORITY"]["eligible"]


def test_risk_veto_blocks_every_arm():
    r,b=fixture(eligible=False)
    r["feature_snapshot"]["features"]["risk_veto"]=True
    x=evaluate_variants(r,b,80)
    assert not any(v["eligible"] for v in x.values())


def test_clock_volume_has_no_current_or_future_contamination():
    idx=pd.date_range("2024-01-01 10:55",periods=22,freq="B",tz="America/New_York")
    f=pd.DataFrame({"Volume":np.arange(1,23,dtype=float)},index=idx)
    x=clock_reference(f)
    assert x.iloc[20]==10.5
    f.iloc[20,0]=100000
    assert clock_reference(f).iloc[20]==10.5
    assert np.isnan(x.iloc[19])


def test_embargo_uses_full_timeout_not_observed_exit():
    def row(cut, due):
        return {"cut_at":cut,"horizons":[{"contract":{"timeout_at":due}}]}
    a=row("2024-07-01T14:00:00Z","2024-07-02T20:00:00Z")
    b=row("2024-07-01T15:00:00Z","2024-07-02T20:00:00Z")
    c=row("2024-07-03T14:00:00Z","2024-07-05T20:00:00Z")
    assert nonoverlapping_events([a,b,c])==[a,c]


def test_net_costs_scaled_by_reduced_exposure():
    row={"cut_at":"2024-07-01T15:00:00Z","side":"LONG","exposure":.25,
         "horizons":[{"horizon_minutes":1440,"execution":{"status":"SIMULATED_RESOLVED","outcome":"TP_FIRST","fill_price":100.,"net_pnl_per_share":1.}}]}
    assert statistics([row],1440)["net_bps_sum_exposure_scaled"]==25.


def test_holdout_cannot_be_reserved_twice(tmp_path):
    args=dict(registry_path=tmp_path/"research.sqlite",dataset_sha256="a"*64,symbol="SMCI",horizon="EXP",
              protocol_sha256="b"*64,commitment_sha256="c"*64,members=(("one","2024-07-01T15:00:00Z"),))
    reserve_holdout_opening(**args)
    with pytest.raises(HoldoutAlreadyOpenedError):
        reserve_holdout_opening(**args)


@pytest.mark.skipif(os.name!="nt",reason="Windows current-user DPAPI")
def test_dpapi_protected_key_roundtrip():
    value=os.urandom(32)
    protected=protect_key(value)
    assert protected!=value
    assert protect_key(protected,decrypt=True)==value


def test_wrong_current_eligibility_fails_control():
    r,b=fixture(eligible=False)
    with pytest.raises(ValueError,match="Current control"):
        evaluate_variants(r,b,100)
