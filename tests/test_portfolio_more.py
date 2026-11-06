"""Additional portfolio unit tests (helpers, projections, baselines)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio import (
    _cap_weights,
    _predictions_for,
    _risk_free_at,
    blend_mu,
    buy_and_hold,
    equal_weight_backtest,
    historical_mu,
    model_mu,
    performance,
    select_lambda,
)


def _prices(n=90, k=5, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    tickers = [f"T{i}" for i in range(k)]
    return dates, tickers, pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, (n, k)), 0)), index=dates, columns=tickers)


def test_mu_helpers(cfg):
    dates, tickers, prices = _prices()
    r = np.log(prices).diff().dropna()
    mu = historical_mu(r)
    assert np.isfinite(mu).all()
    pred = pd.Series([0.01] * len(tickers), index=tickers)
    assert np.isfinite(model_mu(pred)).all()
    np.testing.assert_allclose(blend_mu(mu, model_mu(pred), 1.0), model_mu(pred))
    macro = pd.DataFrame({"date": dates, "DGS3MO": 0.02})
    assert np.isclose(_risk_free_at(macro, dates[-1], 0.04), 0.0002)
    assert _risk_free_at(pd.DataFrame({"date": [], "DGS3MO": []}), dates[-1], 0.04) == 0.04


def test_cap_projection():
    rng = np.random.default_rng(1)
    W = rng.dirichlet(np.ones(6), size=200)
    capped = _cap_weights(W, 0.3)
    np.testing.assert_allclose(capped.sum(axis=1), 1.0, atol=1e-8)
    assert (capped <= 0.3 + 1e-9).all()


def test_baselines_and_performance():
    dates, tickers, prices = _prices()
    eq = equal_weight_backtest(prices, dates, tickers)
    assert len(eq) == len(dates)
    broad = prices.copy()
    broad["^GSPC"] = prices.mean(axis=1)
    bh = buy_and_hold(broad, "^GSPC", dates)
    m = performance(eq, bh, 0.0)
    assert "sharpe" in m and "max_drawdown" in m and m["days"] == len(dates)


def test_predictions_for_requires_all_tickers():
    dates, tickers, _ = _prices()
    pred = pd.DataFrame({"date": [dates[0]] * len(tickers), "ticker": tickers, "y_pred": 0.01})
    assert _predictions_for(pred, dates[0], tickers) is not None
    assert _predictions_for(pred, dates[0], tickers + ["MISSING"]) is None


def test_select_lambda_returns_grid_value(cfg):
    dates, tickers, prices = _prices(n=100)
    pred = pd.DataFrame(
        [
            {"date": d, "ticker": t, "y_pred": 0.01, "split": "val"}
            for d in dates[-8:]
            for t in tickers
        ]
    )
    macro = pd.DataFrame({"date": dates, "DGS3MO": 0.02})
    lam, table = select_lambda(prices, prices.assign(**{"^GSPC": prices.mean(axis=1)}), macro, pred, tickers, cfg)
    assert lam in cfg["portfolio"]["mu_lambdas"]
    assert len(table) == len(cfg["portfolio"]["mu_lambdas"])
