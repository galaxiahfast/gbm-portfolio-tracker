"""F05/F06 regression tests; isolated databases and synthetic market data."""
from collections import Counter
from dataclasses import asdict
from decimal import Decimal
import json
import numpy as np
import pandas as pd
import pytest

from portfolio_tracker.analytics import backtesting as bt
from portfolio_tracker.analytics import causal_core as core
from portfolio_tracker.analytics import decision_engines as adapter
from portfolio_tracker.analytics.replay import ReplayDataset, evaluate_replay_cut, REPLAY_CONTRACT
from portfolio_tracker.analytics.technical_probability import analyze_probability
from portfolio_tracker.db import Database
from portfolio_tracker.repository import PortfolioRepository
from tests.market_fixtures import intraday_index
from tests.test_pdf_report import _ohlcv


def candidate_frame():
    return pd.DataFrame(dict(Open=100.,High=103.,Low=99.,Close=102.,Volume=200.,
        Volume_MA20=100.,StochRSI_K=50.,StochRSI_D=50.,MACD=2.,MACD_signal=1.,
        ADX14=30.,ATR14=1.,EMA9=101.,EMA21=100.,EMA50=99.,
        WeeklyBias=2.,MonthlyBias=2.,PriorHigh20=101.,PriorLow20=98.,
        ATRPercent=.01,ATRPercentMedian=.01),index=pd.date_range('2025-01-01',periods=30))


def test_mirror_candidates_receive_identical_confluence_and_exposure():
    up = candidate_frame()
    down = up.copy()
    for column in ('Open','Close','EMA9','EMA21','EMA50'):
        down[column] = 200-up[column]
    for high,low in [('High','Low'),('PriorHigh20','PriorLow20')]:
        down[high],down[low] = 200-up[low],200-up[high]
    for column in ('MACD','MACD_signal','WeeklyBias','MonthlyBias'):
        down[column] = -up[column]
    config = bt.BacktestConfig(holding_sessions=1)
    longs,_ = bt._generate_candidates(up,config)
    shorts,_ = bt._generate_candidates(down,config)
    assert longs and len(longs)==len(shorts)
    for long,short in zip(longs,shorts):
        assert long.side=='LONG' and short.side=='SHORT'
        assert long.score==short.score==9.0
        assert long.strength_bucket==short.strength_bucket
        assert long.exposure_factor==short.exposure_factor


@pytest.mark.parametrize('macd,price,weekly,monthly', [(1.,2.,2.,2.),(-1.,2.,-2.,1.),(0.,0.,0.,0.)])
def test_direction_is_applied_once_including_neutral_votes(macd,price,weekly,monthly):
    long = core.directional_confluence('LONG',macd_delta=macd,price_delta=price,
        weekly_bias=weekly,monthly_bias=monthly,volume_ratio=1.3)
    short = core.directional_confluence('SHORT',macd_delta=-macd,price_delta=-price,
        weekly_bias=-weekly,monthly_bias=-monthly,volume_ratio=1.3)
    assert long == short


def test_live_adapter_reexports_exact_shared_engines():
    assert adapter.RegimeEngine is core.RegimeEngine
    assert adapter.SetupEngine is core.SetupEngine
    assert adapter.TriggerEngine is core.TriggerEngine
    assert adapter.evaluate_causal_core is core.evaluate_causal_core


@pytest.fixture
def dataset():
    index = intraday_index(700)
    intraday = _ohlcv(index,[100+i*.002+np.sin(i/5) for i in range(len(index))])
    daily_index = pd.bdate_range('2021-01-01',periods=1400)
    daily = _ohlcv(daily_index,[60+i*.025+np.sin(i/12) for i in range(len(daily_index))])
    return ReplayDataset(intraday,daily,index[-1]+pd.Timedelta(minutes=5))


def test_replay_and_live_equal_at_same_historical_cut_and_future_invariant(dataset):
    cut = dataset.intraday.index[400]+pd.Timedelta(minutes=5)
    replay = evaluate_replay_cut(dataset,cut,symbol='SMCI')
    live = analyze_probability('SMCI',dataset.intraday,dataset.daily,as_of_time=cut)
    fields = ('macro_permission','macro_trending','structural_support','structural_resistance',
              'activation_trigger_met','activation_trigger','risk_veto','operation_probability','execution_levels')
    assert all(getattr(replay,k)==getattr(live,k) for k in fields)
    future = dataset.intraday.copy()
    future.loc[future.index >= cut,['Open','High','Low','Close']] *= 1.8
    changed = evaluate_replay_cut(ReplayDataset(future,dataset.daily,dataset.as_of),cut,symbol='SMCI')
    assert all(getattr(changed,k)==getattr(replay,k) for k in fields)
    assert replay.intraday_indicators.index[-1]+pd.Timedelta(minutes=5) <= cut


def test_daily_bars_are_not_accepted_as_5m_replay(dataset):
    with pytest.raises(ValueError,match='5 minutos'):
        ReplayDataset(dataset.daily,dataset.daily,dataset.as_of)


def test_next_bar_execution_keeps_live_plan_and_real_session_horizon(dataset):
    frame = dataset.intraday.copy()
    frame.loc[:,['Open','Close']] = 100.
    frame.loc[:,'Low'],frame.loc[:,'High'] = 99.9,100.1
    candidate = bt.SetupCandidate(77,(frame.index[77]+pd.Timedelta(minutes=5)).isoformat(),
        'LONG',7.,7,1.,2.,30.,'BULLISH',1.,'shared trigger','ALCISTA',98.,104.)
    trade = bt._simulate_trade('SMCI','TEST',frame,candidate,
        bt.BacktestConfig(holding_sessions=1),10000,.6)
    assert pd.Timestamp(trade.entry_date)==frame.index[78]
    assert trade.stop_price==98 and trade.target_price==104
    assert pd.Timestamp(trade.exit_date)==frame.index[155]+pd.Timedelta(minutes=5)
    assert trade.costs_usd>0
    assert trade.net_pnl_usd==pytest.approx(trade.gross_pnl_usd-trade.costs_usd)


def test_replay_batch_uses_shared_candidates_and_preserves_contract(dataset,monkeypatch):
    calls=[]
    def prepared(data,config,**kwargs):
        calls.append(kwargs['symbol'])
        return data.intraday,[],Counter({'Veto del motor técnico compartido':1})
    monkeypatch.setattr(bt,'prepare_replay',prepared)
    batch = bt.run_backtest_batch({'SMCI':dataset},bt.BacktestConfig(holding_sessions=1),starting_capital_usd=10000)
    assert calls==['SMCI']
    assert batch.engine_version==bt.ENGINE_VERSION
    assert batch.replay_contract==REPLAY_CONTRACT
    assert batch.core_sha256==core.causal_revision()
    assert batch.aggregate_decision=='RECHAZADO'


@pytest.fixture
def repo(tmp_path):
    repository = PortfolioRepository(Database(tmp_path/'test.db'))
    repository.database.initialize()
    repository.ensure_initial_capital()
    return repository


def approved_payload(symbol='SMCI'):
    config = bt.BacktestConfig()
    metrics = bt.PerformanceMetrics(400,200,200,160,40,.8,.7,2.,5.,20.,1000.,-500.,30.,.1)
    return dict(engine_version=bt.ENGINE_VERSION,core_sha256=core.causal_revision(),
        replay_contract=REPLAY_CONTRACT,dataset_sha256='test-data',aggregate_decision='APROBADO ESTADÍSTICAMENTE',
        config=asdict(config),results=[dict(symbol=symbol,decision='APROBADO ESTADÍSTICAMENTE',
            replay_contract=REPLAY_CONTRACT,validation=asdict(metrics))])


def save(repo,payload,status='APPROVED',parameters=None):
    return repo.record_backtest_run(engine_version=payload['engine_version'],
        symbols_json=json.dumps([r['symbol'] for r in payload['results']]),
        parameters_json=json.dumps(payload['config'] if parameters is None else parameters),
        dataset_sha256=payload['dataset_sha256'],payload_json=json.dumps(payload),status=status)


def selected(repo,symbol='SMCI'):
    return repo.latest_backtest_parameters(symbol=symbol,engine_version=bt.ENGINE_VERSION)


def test_selector_requires_asset_version_and_skips_unrelated_rejected_runs(repo):
    payload = approved_payload()
    save(repo,payload)
    save(repo,approved_payload('TSLA'))
    rejected = approved_payload()
    rejected['config']['stop_atr_multiple']=2.5
    save(repo,rejected,status='REJECTED')
    assert selected(repo)==payload['config']
    assert selected(repo,'GME') is None
    assert repo.latest_backtest_parameters(symbol='SMCI',engine_version='old') is None
    with pytest.raises(TypeError):
        repo.latest_backtest_parameters()
    assert repo.cash_balance_usd()==Decimal('921.05')
    with repo.database.connect() as connection:
        assert connection.execute('SELECT COUNT(*) FROM trades').fetchone()[0]==0


@pytest.mark.parametrize('mutation', ['hash','sidecar','version','revision','contract','asset_rejected','weak','orphan'])
def test_unsafe_parameter_promotion_fails_closed(repo,mutation):
    payload = approved_payload()
    parameters = None
    if mutation=='sidecar':
        parameters = {**payload['config'],'stop_atr_multiple':2.5}
    elif mutation=='version': payload['engine_version']='legacy'
    elif mutation=='revision': payload['core_sha256']='previous-code'
    elif mutation=='contract': payload['replay_contract']='daily-research-only'
    elif mutation=='asset_rejected': payload['results'][0]['decision']='RECHAZADO'
    elif mutation=='weak': payload['results'][0]['validation']['setups']=2
    elif mutation=='orphan': payload['results']=[]
    run_id = save(repo,payload,parameters=parameters)
    if mutation=='hash':
        with repo.database.transaction() as connection:
            connection.execute("UPDATE backtest_runs SET payload_sha256='altered' WHERE id=?",(run_id,))
    assert selected(repo) is None
