"""Daily-bar study of documented, low-parameter effects. Research only.

Everything here is pure (no network, no database, no orders). Parameters are
the textbook defaults, fixed *before* looking at any result, so the study has
no in-sample tuning to overfit:

* MOMENTUM_12_1   rank by the return from t-252 to t-21 (skip last month),
                  hold the top quintile equally weighted, monthly rebalance.
* TREND_200       hold the benchmark while its close is above the 200-day
                  simple average at month end, otherwise cash.
* VOL_TARGET      scale benchmark exposure to a fixed annual volatility with
                  an EWMA forecast, never above 1x (no leverage), weekly.

Causality: target functions return *sparse* decisions (NaN on days with no
decision). A decision made with closes up to day t is executed at the close
of t + execution_lag (default 1: decide at the close, trade the next close)
and first earns the return of the following day. Between trades weights drift
with prices; nothing is silently rebalanced. Costs are charged on every unit
of turnover at ``cost_bps_per_side``. Cash earns the supplied daily rate.

Known bias: a universe of today's large caps contains survivors. Momentum is
therefore compared with the *same* universe equally weighted, which carries
the same bias, not only with the index.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True, slots=True)
class StudyConfig:
    cost_bps_per_side: float = 30.0  # GBM 0.25% commission + 5 bps slippage
    execution_lag_days: int = 1
    momentum_lookback: int = 252
    momentum_skip: int = 21
    momentum_quantile: float = 0.2
    momentum_min_names: int = 5
    trend_window: int = 200
    vol_target_annual: float = 0.15
    vol_ewma_lambda: float = 0.94
    vol_rebalance_days: int = 5
    holdout_start: str = "2019-01-01"
    bootstrap_draws: int = 1000
    bootstrap_block: int = 21
    seed: int = 20261111

    def __post_init__(self):
        if self.cost_bps_per_side < 0 or self.execution_lag_days < 0:
            raise ValueError("Costos y retraso de ejecución no pueden ser negativos.")
        if not 0 < self.momentum_quantile <= 1 or self.momentum_skip >= self.momentum_lookback:
            raise ValueError("Parámetros de momentum inválidos.")
        if not 0 < self.vol_ewma_lambda < 1 or self.vol_target_annual <= 0:
            raise ValueError("Parámetros de volatilidad inválidos.")


# --------------------------------------------------------------------------
# Data hygiene
# --------------------------------------------------------------------------

def clean_closes(closes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sorted unique dates, positive finite prices; never forward-fills.

    Gaps stay NaN; the corresponding return is NaN and a held position earns
    zero that day (reported), instead of inventing a price.
    """
    if not isinstance(closes.index, pd.DatetimeIndex):
        raise ValueError("Se requiere índice de fechas.")
    frame = closes.sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")].astype(float)
    present = frame.notna()
    invalid = ~np.isfinite(frame) | (frame <= 0)
    removed = int((invalid & present).sum().sum())
    frame = frame.mask(invalid)
    returns = frame.pct_change(fill_method=None)
    report = {
        "dates": int(len(frame)),
        "symbols": int(frame.shape[1]),
        "invalid_prices_removed": removed,
        "extreme_daily_moves_over_50pct": {
            str(symbol): int(count)
            for symbol, count in (returns.abs() > 0.5).sum().items() if count
        },
        "first_date": frame.index[0].date().isoformat() if len(frame) else None,
        "last_date": frame.index[-1].date().isoformat() if len(frame) else None,
    }
    return frame, report


def neutralize_price_breaks(closes: pd.DataFrame, breaks) -> tuple[pd.DataFrame, list[dict]]:
    """Remove documented adjustment artefacts (e.g. a spin-off adjusted the
    wrong way) by rescaling earlier prices so that day's return is zero.

    Only explicit (symbol, date, reason) entries are touched; nothing is
    detected or altered automatically.
    """
    frame = closes.copy()
    applied = []
    for symbol, day, reason in breaks:
        day = pd.Timestamp(day)
        if symbol not in frame or day not in frame.index:
            continue
        position = frame.index.get_loc(day)
        before = frame[symbol].iloc[:position].dropna()
        after = frame[symbol].iloc[position]
        if before.empty or not np.isfinite(after):
            continue
        factor = float(after / before.iloc[-1])
        frame.loc[frame.index[:position], symbol] *= factor
        applied.append({"symbol": symbol, "date": day.date().isoformat(),
                        "removed_return": factor - 1, "reason": reason})
    return frame, applied


def month_end_flags(index: pd.DatetimeIndex) -> pd.Series:
    """True on the last available trading day of each calendar month."""
    periods = index.to_period("M")
    flags = np.r_[periods[1:] != periods[:-1], True]
    return pd.Series(flags, index=index)


# --------------------------------------------------------------------------
# Target weights (information up to and including day t only)
# --------------------------------------------------------------------------

def momentum_targets(closes: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    """Top-quantile 12-1 momentum, equal weight, decided on month ends."""
    past = closes.shift(config.momentum_skip)
    origin = closes.shift(config.momentum_lookback)
    score = past / origin - 1
    # Eligible only with a price today and a full lookback window.
    score = score.where(closes.notna())
    rebalance = month_end_flags(closes.index)
    targets = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    for day in closes.index[rebalance.to_numpy()]:
        row = score.loc[day].dropna()
        if len(row) < config.momentum_min_names:
            targets.loc[day] = 0.0
            continue
        count = max(config.momentum_min_names, int(math.floor(len(row) * config.momentum_quantile)))
        chosen = row.sort_values(ascending=False, kind="mergesort").index[:count]
        weights = pd.Series(0.0, index=closes.columns)
        weights[chosen] = 1.0 / len(chosen)
        targets.loc[day] = weights
    return targets


def equal_weight_targets(closes: pd.DataFrame) -> pd.DataFrame:
    """Same-universe benchmark: all names with a price, rebalanced monthly."""
    rebalance = month_end_flags(closes.index)
    available = closes.notna().astype(float)
    weights = available.div(available.sum(axis=1).replace(0, np.nan), axis=0)
    targets = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    targets.loc[rebalance.to_numpy()] = weights.fillna(0.0).loc[rebalance.to_numpy()]
    return targets


def trend_targets(benchmark: pd.Series, config: StudyConfig) -> pd.Series:
    """1 while the month-end close is above its N-day SMA, else 0."""
    average = benchmark.rolling(config.trend_window, min_periods=config.trend_window).mean()
    signal = (benchmark > average).astype(float).where(average.notna())
    rebalance = month_end_flags(benchmark.index)
    return signal.fillna(0.0).where(rebalance, np.nan)


def ewma_variance(returns: pd.Series, lam: float) -> pd.Series:
    """One-step-ahead variance known at the close of day t (uses r_t)."""
    values = returns.fillna(0.0).to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    seed = values[:20]
    variance = float(np.mean(seed ** 2)) if len(seed) else np.nan
    for i, value in enumerate(values):
        variance = lam * variance + (1 - lam) * value * value
        out[i] = variance
    series = pd.Series(out, index=returns.index)
    series.iloc[:20] = np.nan  # burn-in; not a forecast yet
    return series


def vol_target_exposure(returns: pd.Series, config: StudyConfig) -> pd.Series:
    """min(1, target / forecast) recomputed every N days; no leverage."""
    forecast = np.sqrt(ewma_variance(returns, config.vol_ewma_lambda) * TRADING_DAYS)
    exposure = (config.vol_target_annual / forecast).clip(upper=1.0)
    keep = np.zeros(len(exposure), dtype=bool)
    keep[::config.vol_rebalance_days] = True
    return exposure.fillna(0.0).where(keep, np.nan)


# --------------------------------------------------------------------------
# Backtest engine
# --------------------------------------------------------------------------

def simulate(targets: pd.DataFrame, returns: pd.DataFrame, cash_daily: pd.Series,
             config: StudyConfig) -> pd.DataFrame:
    """Daily net returns of a long-only book trading only on decision days.

    A row of ``targets`` with any non-NaN value is a decision (missing names
    in that row mean weight 0). It executes ``execution_lag`` rows later at
    that close; the day's return accrues to the old weights first. Weights
    then drift with prices until the next execution.
    """
    returns = returns.reindex(columns=targets.columns)
    targets = targets.reindex(returns.index)
    decided = targets.notna().any(axis=1).to_numpy()
    goal = targets.fillna(0.0).to_numpy(dtype=float)
    if (goal < -1e-12).any() or (goal.sum(axis=1) > 1 + 1e-9).any():
        raise ValueError("Solo se permiten pesos largos sin apalancamiento.")
    asset = returns.to_numpy(dtype=float)
    missing = ~np.isfinite(asset)
    asset = np.where(missing, 0.0, asset)
    cash = cash_daily.reindex(returns.index).fillna(0.0).to_numpy(dtype=float)
    rate = config.cost_bps_per_side / 10_000.0
    lag = config.execution_lag_days
    weights = np.zeros(asset.shape[1])
    n = len(asset)
    net = np.zeros(n); gross = np.zeros(n); turnover = np.zeros(n)
    exposure = np.zeros(n); gaps = np.zeros(n, dtype=int)
    for i in range(n):
        cash_weight = max(0.0, 1.0 - weights.sum())
        day = float(weights @ asset[i] + cash_weight * cash[i])
        gaps[i] = int(((weights > 0) & missing[i]).sum())
        growth = 1.0 + day
        if growth > 0:
            weights = weights * (1.0 + asset[i]) / growth
        else:  # total loss: nothing left to hold
            weights = np.zeros_like(weights)
        traded = 0.0
        source = i - lag
        if source >= 0 and decided[source]:
            traded = float(np.abs(goal[source] - weights).sum())
            weights = goal[source].copy()
        gross[i] = day
        turnover[i] = traded
        net[i] = (1.0 + day) * (1.0 - traded * rate) - 1.0
        exposure[i] = weights.sum()
    return pd.DataFrame({
        "net": net, "gross": gross, "turnover": turnover,
        "costs": gross - net, "exposure": exposure, "gap_days": gaps,
    }, index=returns.index)


def performance(daily: pd.Series, cash_daily: pd.Series) -> dict:
    daily = daily.dropna()
    if len(daily) < 2:
        return {"days": int(len(daily))}
    wealth = (1 + daily).cumprod()
    years = len(daily) / TRADING_DAYS
    cagr = float(wealth.iloc[-1] ** (1 / years) - 1)
    volatility = float(daily.std(ddof=1) * math.sqrt(TRADING_DAYS))
    excess = daily - cash_daily.reindex(daily.index).fillna(0.0)
    sharpe = (float(excess.mean() / excess.std(ddof=1) * math.sqrt(TRADING_DAYS))
              if excess.std(ddof=1) > 0 else None)
    drawdown = wealth / wealth.cummax() - 1
    max_dd = float(drawdown.min())
    return {
        "days": int(len(daily)), "start": daily.index[0].date().isoformat(),
        "end": daily.index[-1].date().isoformat(), "cagr": cagr,
        "volatility": volatility, "sharpe": sharpe, "max_drawdown": max_dd,
        "calmar": (cagr / -max_dd) if max_dd < 0 else None,
        "final_wealth_per_1": float(wealth.iloc[-1]),
    }


def stationary_bootstrap_indices(length: int, block: int, rng) -> np.ndarray:
    """Politis–Romano indices: geometric blocks preserve volatility clustering."""
    indices = np.empty(length, dtype=np.int64)
    position = int(rng.integers(length))
    restart = rng.random(length) < 1.0 / block
    starts = rng.integers(length, size=length)
    for i in range(length):
        if i and restart[i]:
            position = int(starts[i])
        indices[i] = position
        position = (position + 1) % length
    return indices


def paired_bootstrap(strategy: pd.Series, benchmark: pd.Series, cash_daily: pd.Series,
                     config: StudyConfig) -> dict:
    """95% intervals for annual excess return and Sharpe difference."""
    frame = pd.concat({"s": strategy, "b": benchmark}, axis=1).dropna()
    if len(frame) < config.bootstrap_block * 4:
        return {"status": "INSUFFICIENT_DAYS"}
    cash = cash_daily.reindex(frame.index).fillna(0.0).to_numpy()
    s, b = frame["s"].to_numpy(), frame["b"].to_numpy()
    rng = np.random.default_rng(config.seed)

    def sharpe(values, rf):
        excess = values - rf
        sd = excess.std(ddof=1)
        return excess.mean() / sd * math.sqrt(TRADING_DAYS) if sd > 0 else 0.0

    mean_gap, sharpe_gap = [], []
    for _ in range(config.bootstrap_draws):
        idx = stationary_bootstrap_indices(len(frame), config.bootstrap_block, rng)
        mean_gap.append((s[idx].mean() - b[idx].mean()) * TRADING_DAYS)
        sharpe_gap.append(sharpe(s[idx], cash[idx]) - sharpe(b[idx], cash[idx]))
    q = lambda values: [float(v) for v in np.quantile(values, [0.025, 0.975])]
    return {
        "annual_return_difference": float((s.mean() - b.mean()) * TRADING_DAYS),
        "annual_return_difference_95": q(mean_gap),
        "sharpe_difference": float(sharpe(s, cash) - sharpe(b, cash)),
        "sharpe_difference_95": q(sharpe_gap),
        "method": f"stationary block bootstrap, mean block {config.bootstrap_block} days, "
                  f"{config.bootstrap_draws} draws",
    }


# --------------------------------------------------------------------------
# Volatility forecasting (walk-forward)
# --------------------------------------------------------------------------

def _garch_loglik(returns: np.ndarray, alphas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Gaussian GARCH(1,1) log-likelihood with variance targeting, vectorized
    over parameter candidates; returns -inf for non-stationary candidates."""
    sample_var = float(np.mean(returns ** 2))
    persistence = alphas + betas
    valid = persistence < 0.999
    total = np.full(alphas.shape, -np.inf)
    a, b = alphas[valid], betas[valid]
    omega = sample_var * (1 - a - b)
    variance = np.full(a.shape, sample_var)
    partial = np.zeros(a.shape)
    for value in returns:
        partial += -0.5 * (np.log(variance) + value * value / variance)
        variance = omega + a * value * value + b * variance
    total[valid] = partial
    return total


def fit_garch(returns: np.ndarray) -> dict:
    """Coarse-to-fine grid MLE (no SciPy). Deterministic and bounded."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 500:
        raise ValueError("GARCH requiere al menos 500 retornos.")
    a_grid, b_grid = np.meshgrid(np.linspace(0.01, 0.30, 15), np.linspace(0.60, 0.98, 20))
    alphas, betas = a_grid.ravel(), b_grid.ravel()
    ll = _garch_loglik(returns, alphas, betas)
    best = int(np.argmax(ll))
    a0, b0 = alphas[best], betas[best]
    a_fine, b_fine = np.meshgrid(
        np.clip(np.linspace(a0 - 0.02, a0 + 0.02, 9), 0.001, 0.5),
        np.clip(np.linspace(b0 - 0.02, b0 + 0.02, 9), 0.3, 0.998),
    )
    alphas, betas = a_fine.ravel(), b_fine.ravel()
    ll = _garch_loglik(returns, alphas, betas)
    best = int(np.argmax(ll))
    alpha, beta = float(alphas[best]), float(betas[best])
    if not np.isfinite(ll[best]):
        raise ValueError("GARCH sin solución estacionaria.")
    return {"alpha": alpha, "beta": beta, "long_run_variance": float(np.mean(returns ** 2)),
            "loglik": float(ll[best]), "observations": int(len(returns))}


def garch_forecasts(returns: pd.Series, horizon: int, refit_every: int = TRADING_DAYS,
                    minimum_history: int = 1000) -> pd.Series:
    """Walk-forward: parameters fitted only on returns up to each refit day.

    Value at t = forecast of the sum of variances for t+1..t+horizon.
    """
    values = returns.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    params, variance = None, None
    for i, value in enumerate(values):
        if not np.isfinite(value):
            value = 0.0
        if i + 1 >= minimum_history and (params is None or (i + 1 - minimum_history) % refit_every == 0):
            params = fit_garch(values[:i + 1])
            # Re-filter the conditional variance with the new parameters.
            alpha, beta, s2 = params["alpha"], params["beta"], params["long_run_variance"]
            omega = s2 * (1 - alpha - beta)
            variance = s2
            for past in values[:i + 1]:
                past = past if np.isfinite(past) else 0.0
                variance = omega + alpha * past * past + beta * variance
        elif params is not None:
            alpha, beta, s2 = params["alpha"], params["beta"], params["long_run_variance"]
            variance = s2 * (1 - alpha - beta) + alpha * value * value + beta * variance
        if params is not None:
            persistence = params["alpha"] + params["beta"]
            s2 = params["long_run_variance"]
            steps = persistence ** np.arange(horizon)
            out[i] = float(np.sum(s2 + steps * (variance - s2)))
    return pd.Series(out, index=returns.index)


def realized_forward_variance(returns: pd.Series, horizon: int) -> pd.Series:
    """Sum of squared returns over t+1..t+horizon (the target, never a feature)."""
    squared = returns.pow(2)
    return squared[::-1].rolling(horizon, min_periods=horizon).sum()[::-1].shift(-1)


def qlike(realized: pd.Series, forecast: pd.Series) -> float:
    frame = pd.concat({"r": realized, "f": forecast}, axis=1).dropna()
    frame = frame[(frame.r > 0) & (frame.f > 0)]
    ratio = frame.r / frame.f
    return float(np.mean(ratio - np.log(ratio) - 1))


def volatility_study(returns: pd.Series, horizon: int = 5, lam: float = 0.94,
                     minimum_history: int = 1000, holdout_start: str | None = None) -> dict:
    """Compare naive 20-day, EWMA and walk-forward GARCH forecasts by QLIKE."""
    returns = returns.dropna()
    target = realized_forward_variance(returns, horizon)
    naive = returns.pow(2).rolling(20, min_periods=20).mean() * horizon
    ewma = ewma_variance(returns, lam) * horizon
    garch = garch_forecasts(returns, horizon, minimum_history=minimum_history)
    common = pd.concat({"t": target, "n": naive, "e": ewma, "g": garch}, axis=1).dropna()

    def losses(frame):
        return {name: qlike(frame.t, frame[key]) for key, name in
                (("n", "NAIVE_20D"), ("e", "EWMA_094"), ("g", "GARCH_1_1_WF"))}

    result = {"horizon_days": horizon, "evaluated_days": int(len(common)),
              "qlike_all": losses(common)}
    if holdout_start is not None:
        late = common.loc[common.index >= pd.Timestamp(holdout_start)]
        result["qlike_holdout"] = losses(late) if len(late) else None
    yearly_wins = {"EWMA_094": 0, "GARCH_1_1_WF": 0}
    years = 0
    for _, frame in common.groupby(common.index.year):
        if len(frame) < 100:
            continue
        years += 1
        loss = losses(frame)
        for name in yearly_wins:
            yearly_wins[name] += int(loss[name] < loss["NAIVE_20D"])
    result["years_beating_naive"] = yearly_wins
    result["years_evaluated"] = years
    if len(returns) >= minimum_history:
        params = fit_garch(returns.to_numpy())
        last = garch.dropna()
        result["current"] = {
            "as_of": returns.index[-1].date().isoformat(),
            "garch": params,
            "forecast_daily_vol": float(math.sqrt(last.iloc[-1] / horizon)) if len(last) else None,
            "ewma_daily_vol": float(math.sqrt(ewma.iloc[-1] / horizon)),
            "realized_20d_daily_vol": float(math.sqrt(naive.iloc[-1] / horizon)),
        }
    return result


def risk_sized_position(capital: float, daily_vol: float, risk_fraction: float = 0.01,
                        z: float = 2.33) -> float:
    """Position value whose 1-day ~99% normal move equals risk_fraction of capital.

    Normal tails understate crashes; this is a ceiling, not a guarantee.
    """
    if capital <= 0 or not math.isfinite(daily_vol) or daily_vol <= 0:
        raise ValueError("Capital y volatilidad deben ser positivos.")
    return min(capital, capital * risk_fraction / (z * daily_vol))


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_strategy_study(closes: pd.DataFrame, benchmark: str, cash_daily: pd.Series,
                       config: StudyConfig | None = None) -> dict:
    config = config or StudyConfig()
    closes, hygiene = clean_closes(closes)
    if benchmark not in closes:
        raise ValueError("Falta el benchmark en los precios.")
    universe = closes.drop(columns=[benchmark])
    returns = closes.pct_change(fill_method=None)
    bench_returns = returns[benchmark]
    bench_frame = returns[[benchmark]]

    def single(weights: pd.Series) -> pd.DataFrame:
        return weights.to_frame(benchmark)

    hold = pd.Series(np.nan, index=closes.index)
    hold.iloc[0] = 1.0

    trend = trend_targets(closes[benchmark], config)
    momentum = momentum_targets(universe, config)
    books = {
        "BUY_HOLD_BENCHMARK": simulate(single(hold), bench_frame, cash_daily, config),
        "TREND_200": simulate(single(trend), bench_frame, cash_daily, config),
        "VOL_TARGET": simulate(single(vol_target_exposure(bench_returns, config)), bench_frame, cash_daily, config),
        "EQUAL_WEIGHT_UNIVERSE": simulate(equal_weight_targets(universe), returns[universe.columns], cash_daily, config),
        "MOMENTUM_12_1": simulate(momentum, returns[universe.columns], cash_daily, config),
        "MOMENTUM_12_1_TREND": simulate(momentum.mul(trend, axis=0), returns[universe.columns], cash_daily, config),
    }
    # Common evaluation window: every book has a full signal history.
    start = closes.index[max(config.momentum_lookback, config.trend_window) + 2 + config.execution_lag_days]
    split = pd.Timestamp(config.holdout_start)
    periods = {"FULL": (start, None), "DEVELOPMENT": (start, split), "HOLDOUT": (split, None)}
    comparisons = {
        "TREND_200": "BUY_HOLD_BENCHMARK", "VOL_TARGET": "BUY_HOLD_BENCHMARK",
        "MOMENTUM_12_1": "EQUAL_WEIGHT_UNIVERSE", "MOMENTUM_12_1_TREND": "EQUAL_WEIGHT_UNIVERSE",
        "EQUAL_WEIGHT_UNIVERSE": "BUY_HOLD_BENCHMARK",
    }
    result = {"hygiene": hygiene, "config": {k: getattr(config, k) for k in config.__slots__},
              "periods": {}}
    for period, (begin, end) in periods.items():
        window = lambda s: s.loc[(s.index >= begin) & ((s.index < end) if end is not None else True)]
        stats = {}
        for name, book in books.items():
            net = window(book["net"])
            stats[name] = {
                **performance(net, cash_daily),
                "turnover_per_year": float(window(book["turnover"]).mean() * TRADING_DAYS),
                "cost_drag_per_year": float(window(book["costs"]).mean() * TRADING_DAYS),
                "average_exposure": float(window(book["exposure"]).mean()),
                "position_gap_days": int((window(book["gap_days"]) > 0).sum()),
            }
        tests = {
            f"{name}_vs_{base}": paired_bootstrap(window(books[name]["net"]), window(books[base]["net"]),
                                                  cash_daily, config)
            for name, base in comparisons.items()
        }
        result["periods"][period] = {"start": begin.date().isoformat(),
                                     "end": (end.date().isoformat() if end is not None else hygiene["last_date"]),
                                     "strategies": stats, "comparisons": tests}
    return result
