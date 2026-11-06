"""Financial mathematics implemented from first principles (PRD Section 7).

The tests compare every function with NumPy/pandas/SciPy; see tests/test_finmath.py.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

PERIODS = 252


def simple_returns(prices: pd.Series | np.ndarray) -> pd.Series:
    s = pd.Series(prices).astype(float)
    return s.pct_change()


def log_returns(prices: pd.Series | np.ndarray) -> pd.Series:
    s = pd.Series(prices).astype(float)
    return np.log(s / s.shift(1))


def annualize_return(mean_daily: float, periods: int = PERIODS) -> float:
    return float(mean_daily) * periods


def annualize_volatility(std_daily: float, periods: int = PERIODS) -> float:
    return float(std_daily) * np.sqrt(periods)


def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Sample covariance from the definition (ddof=1), not np.cov's default shape."""
    r = returns.dropna()
    x = r.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / (len(x) - 1)
    return pd.DataFrame(cov, index=r.columns, columns=r.columns)


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    cov = covariance_matrix(returns)
    sd = np.sqrt(np.diag(cov.to_numpy()))
    corr = cov.to_numpy() / np.outer(sd, sd)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


def beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    df = pd.concat([asset_returns, market_returns], axis=1).dropna()
    a, m = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    a = a - a.mean()
    m = m - m.mean()
    return float((a @ m) / (m @ m))


def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    return float(np.asarray(weights) @ np.asarray(mu))


def portfolio_variance(weights: np.ndarray, cov: np.ndarray) -> float:
    w = np.asarray(weights)
    return float(w @ np.asarray(cov) @ w)


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(portfolio_variance(weights, cov)))


def sharpe_ratio(returns: pd.Series | np.ndarray, risk_free_rate: float = 0.0, periods: int = PERIODS) -> float:
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    excess = r - risk_free_rate / periods
    sd = excess.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods))


def sortino_ratio(returns: pd.Series | np.ndarray, risk_free_rate: float = 0.0, periods: int = PERIODS) -> float:
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    excess = r - risk_free_rate / periods
    downside = excess[excess < 0]
    dd = np.sqrt(np.mean(downside**2)) if len(downside) else 0.0
    if dd == 0:
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(periods))


def equity_curve(returns: pd.Series | np.ndarray) -> pd.Series:
    r = pd.Series(returns).fillna(0.0)
    return (1 + r).cumprod()


def max_drawdown(returns: pd.Series | np.ndarray) -> float:
    curve = equity_curve(returns)
    peak = curve.cummax()
    dd = curve / peak - 1
    return float(dd.min())


def calmar_ratio(returns: pd.Series | np.ndarray, periods: int = PERIODS) -> float:
    r = pd.Series(returns).dropna()
    if r.empty:
        return float("nan")
    ann = (1 + r).prod() ** (periods / len(r)) - 1
    mdd = abs(max_drawdown(r))
    return float(ann / mdd) if mdd else float("nan")


def historical_var(returns: pd.Series | np.ndarray, level: float = 0.95) -> float:
    """Positive loss at the (1-level) quantile."""
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    return float(-np.percentile(r, (1 - level) * 100))


def parametric_var_t(returns: pd.Series | np.ndarray, level: float = 0.95) -> float:
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    df, loc, scale = stats.t.fit(r)
    q = stats.t.ppf(1 - level, df, loc=loc, scale=scale)
    return float(-q)


def jarque_bera(returns: pd.Series | np.ndarray) -> Tuple[float, float]:
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    res = stats.jarque_bera(r)
    return float(res.statistic), float(res.pvalue)


def block_bootstrap_sharpe(
    returns: pd.Series | np.ndarray,
    block: int = 20,
    n_resamples: int = 5000,
    risk_free_rate: float = 0.0,
    periods: int = PERIODS,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Circular block bootstrap confidence interval for the Sharpe ratio."""
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    n = len(r)
    if n < block:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    sims = np.empty(n_resamples)
    for i in range(n_resamples):
        starts = rng.integers(0, n, size=n_blocks)
        sample = np.concatenate([np.take(r, np.arange(s, s + block), mode="wrap") for s in starts])[:n]
        sims[i] = sharpe_ratio(sample, risk_free_rate=risk_free_rate, periods=periods)
    lo, hi = np.percentile(sims, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def probabilistic_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    benchmark_sharpe: float = 0.0,
    periods: int = PERIODS,
) -> float:
    """Bailey & Lopez de Prado PSR: P(true SR > benchmark SR)."""
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    n = len(r)
    if n < 3:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)  # per-period
    g1 = stats.skew(r)
    g2 = stats.kurtosis(r, fisher=False)  # non-excess
    denom = np.sqrt(max(1e-12, 1 - g1 * sr + (g2 - 1) / 4 * sr**2))
    z = (sr - benchmark_sharpe / periods) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    trial_sharpes: np.ndarray,
    periods: int = PERIODS,
) -> float:
    """Deflated Sharpe ratio using the spread of Sharpe ratios across trials."""
    trials = np.asarray(trial_sharpes, dtype=float)
    trials = trials[np.isfinite(trials)]
    if len(trials) < 2:
        return float("nan")
    n_trials = len(trials)
    sr_std = trials.std(ddof=1)
    gamma = 0.5772156649015329
    e = np.e
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * e))
    benchmark = sr_std * ((1 - gamma) * z1 + gamma * z2)
    return probabilistic_sharpe_ratio(returns, benchmark_sharpe=benchmark, periods=periods)


def gd_linreg(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 1e-2,
    epochs: int = 2000,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """Manual gradient-descent linear regression (PRD Listing 7.1)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, size=d)
    b = 0.0
    for _ in range(epochs):
        yhat = X @ w + b
        err = yhat - y
        w -= lr * (2 / n) * (X.T @ err)
        b -= lr * (2 / n) * err.sum()
    return w, float(b)
