"""PRD Section 7: from-scratch maths must agree with the libraries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression

from src import finmath as fm


@pytest.fixture
def returns():
    rng = np.random.default_rng(5)
    return pd.DataFrame(rng.normal(0.0004, 0.01, size=(400, 3)), columns=["a", "b", "c"])


def test_returns_and_annualisation():
    p = pd.Series([100.0, 101.0, 99.5, 103.0])
    simple = fm.simple_returns(p)
    log = fm.log_returns(p)
    assert np.isclose(simple.iloc[1], 0.01)
    assert np.isclose(log.iloc[3], np.log(103 / 99.5))
    assert np.isclose(fm.annualize_return(0.001), 0.252)
    assert np.isclose(fm.annualize_volatility(0.01), 0.01 * np.sqrt(252))


def test_covariance_matches_numpy(returns):
    cov = fm.covariance_matrix(returns)
    np.testing.assert_allclose(cov.to_numpy(), np.cov(returns.to_numpy().T, ddof=1), atol=1e-12)
    corr = fm.correlation_matrix(returns)
    np.testing.assert_allclose(corr.to_numpy(), returns.corr().to_numpy(), atol=1e-12)


def test_beta_matches_definition(returns):
    beta = fm.beta(returns["a"], returns["b"])
    expected = np.cov(returns["a"], returns["b"], ddof=1)[0, 1] / np.var(returns["b"], ddof=1)
    assert np.isclose(beta, expected, atol=1e-12)


def test_portfolio_quadratic_forms(returns):
    w = np.array([0.2, 0.3, 0.5])
    mu = returns.mean().to_numpy() * 252
    cov = fm.covariance_matrix(returns).to_numpy() * 252
    assert np.isclose(fm.portfolio_return(w, mu), w @ mu)
    assert np.isclose(fm.portfolio_variance(w, cov), w @ cov @ w)
    assert np.isclose(fm.portfolio_volatility(w, cov), np.sqrt(w @ cov @ w))


def test_risk_metrics(returns):
    r = returns["a"]
    sr = fm.sharpe_ratio(r, risk_free_rate=0.02)
    excess = r - 0.02 / 252
    assert np.isclose(sr, excess.mean() / excess.std(ddof=1) * np.sqrt(252))
    mdd = fm.max_drawdown(pd.Series([0.1, -0.2, 0.05]))
    assert np.isclose(mdd, (1.1 * 0.8) / 1.1 - 1)
    var = fm.historical_var(r, 0.95)
    assert np.isclose(var, -np.percentile(r, 5))
    tvar = fm.parametric_var_t(r, 0.95)
    df, loc, scale = stats.t.fit(r)
    assert np.isclose(tvar, -stats.t.ppf(0.05, df, loc=loc, scale=scale))


def test_jarque_bera(returns):
    stat, p = fm.jarque_bera(returns["a"])
    ref = stats.jarque_bera(returns["a"])
    assert np.isclose(stat, ref.statistic)
    assert np.isclose(p, ref.pvalue)


def test_block_bootstrap_is_deterministic(returns):
    a = fm.block_bootstrap_sharpe(returns["a"], block=20, n_resamples=200, seed=3)
    b = fm.block_bootstrap_sharpe(returns["a"], block=20, n_resamples=200, seed=3)
    assert a == b
    assert a[0] <= a[1]


def test_probabilistic_and_deflated_sharpe(returns):
    psr = fm.probabilistic_sharpe_ratio(returns["a"])
    assert 0 <= psr <= 1
    trials = np.array([0.1, 0.2, 0.15, 0.05, 0.25])
    dsr = fm.deflated_sharpe_ratio(returns["a"], trials)
    assert 0 <= dsr <= 1


def test_gradient_descent_matches_ols():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 4))
    X = (X - X.mean(0)) / X.std(0)
    true_w = np.array([1.5, -2.0, 0.5, 3.0])
    y = X @ true_w + 2.0 + rng.normal(0, 0.01, size=600)
    w_gd, b_gd = fm.gd_linreg(X, y, lr=0.05, epochs=20000)
    ols = LinearRegression().fit(X, y)
    np.testing.assert_allclose(w_gd, ols.coef_, atol=1e-4)
    assert np.isclose(b_gd, ols.intercept_, atol=1e-4)
