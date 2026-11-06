"""Portfolio constraints, one-day execution lag and the Monte Carlo check."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio import _optimise, efficient_frontier, simulate


def _panel(n=90, n_assets=5, seed=2):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    tickers = [f"T{i}" for i in range(n_assets)]
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, (n, n_assets)), axis=0)),
        index=dates,
        columns=tickers,
    )
    return dates, tickers, prices


def test_optimiser_respects_bounds(cfg):
    dates, tickers, prices = _panel()
    mu = np.log(prices).diff().mean().to_numpy() * 252
    w, method = _optimise(mu, prices, risk_free_rate=0.0, cap=0.3)
    assert abs(w.sum() - 1) < 1e-8
    assert (w >= -1e-12).all() and (w <= 0.3 + 1e-9).all()
    assert method in {"max_sharpe", "min_variance_fallback"}


def test_weights_are_applied_from_next_day(cfg):
    dates, tickers, prices = _panel(n=90)
    pred_dates = dates[-6:]
    rows = []
    for d in pred_dates:
        for t in tickers:
            rows.append({"date": d, "ticker": t, "y_pred": 0.01, "split": "test"})
    pred = pd.DataFrame(rows)
    macro = pd.DataFrame({"date": dates, "DGS3MO": 0.02})
    # A huge return on the first prediction date must not be earned by weights set that day.
    prices.loc[pred_dates[0]] *= 1.5
    rets, weights, methods = simulate(prices, macro, pred, tickers, cfg, split="test", lam=0.5, include_costs=False, rebalance_days=2)
    assert len(rets) > 0
    assert rets.index.min() > pred_dates[0]
    assert pred_dates[0] not in rets.index


def test_monte_carlo_does_not_beat_optimiser():
    dates, tickers, prices = _panel(n=300, n_assets=6)
    mu = np.log(prices).diff().mean().to_numpy() * 252 + 0.02
    frontier, cloud, chosen = efficient_frontier(mu, prices, cap=0.3, rf=0.0, points=8, mc_samples=2000, seed=1)
    if chosen["method"] == "max_sharpe":
        best_mc = float(cloud["sharpe"].max())
        assert best_mc <= chosen["tangency"]["sharpe"] + 1e-3
    assert len(frontier) > 0
