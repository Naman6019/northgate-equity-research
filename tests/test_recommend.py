"""Recommendation rules, z-scores and rebalancing bands (PRD Section 12)."""

from __future__ import annotations

import pandas as pd

from src.recommend import decide, rebalance, zscore


def test_decision_branches_and_no_trade_band():
    assert decide(0.51, 0.5, -0.5) == "BUY"
    assert decide(-0.51, 0.5, -0.5) == "SELL"
    assert decide(0.0, 0.5, -0.5) == "HOLD"
    assert decide(0.49, 0.5, -0.5) == "HOLD"


def test_rebalancing_band():
    current = {"A": 0.5, "B": 0.5}
    target = {"A": 0.62, "B": 0.38}
    actions = rebalance(current, target, band=0.05)
    assert actions["A"][0] == "INCREASE"
    assert actions["B"][0] == "REDUCE"
    small = rebalance({"A": 0.5, "B": 0.5}, {"A": 0.52, "B": 0.48}, band=0.05)
    assert small == {}


def test_zscore_is_cross_sectional():
    df = pd.DataFrame(
        {
            "date": ["2020-01-01"] * 3 + ["2020-01-02"] * 3,
            "x": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )
    z = zscore(df, "x")
    assert abs(z.iloc[:3].mean()) < 1e-12
    assert abs(z.iloc[3:].mean()) < 1e-12
    assert abs(z.iloc[0] + z.iloc[2]) < 1e-12
