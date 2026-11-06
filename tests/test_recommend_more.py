"""Additional recommendation tests: benchmark alignment, signal z-scores, hit rates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.recommend import _benchmark_forward, _hit_rates, add_signal_features, tune_weights


def _panel(n_dates=40, k=5):
    dates = pd.bdate_range("2021-01-01", periods=n_dates)
    rows = []
    rng = np.random.default_rng(2)
    for i, d in enumerate(dates):
        for j in range(k):
            rows.append(
                {
                    "date": d,
                    "ticker": f"T{j}",
                    "split": "val" if i < n_dates // 2 else "test",
                    "y_pred": rng.normal(0, 0.01),
                    "target_h5": rng.normal(0.001, 0.02),
                    "sentiment_mom3": rng.normal(0, 0.1),
                    "vol21": rng.uniform(0.005, 0.03),
                    "beta63": rng.uniform(0.5, 1.5),
                    "ret1": rng.normal(0, 0.01),
                    "benchmark_fwd": rng.normal(0.001, 0.01),
                    "sentiment_available": 1,
                }
            )
    df = pd.DataFrame(rows)
    df = add_signal_features(df)
    df["composite"] = 0.5 * df["forecast_z"] + 0.2 * df["sentiment_z"] - 0.3 * df["risk_z"]
    df["decision"] = np.where(df["composite"] >= 0.5, "BUY", np.where(df["composite"] <= -0.5, "SELL", "HOLD"))
    return df


def test_signals_are_cross_sectional():
    df = _panel()
    for col in ("forecast_z", "sentiment_z", "risk_z"):
        means = df.groupby("date")[col].mean()
        assert np.allclose(means, 0, atol=1e-10)


def test_benchmark_forward_is_five_day_return():
    dates = pd.bdate_range("2020-01-01", periods=10)
    prices = pd.DataFrame({"date": dates, "ticker": "^GSPC", "adj_close": np.arange(100.0, 110.0)})
    b = _benchmark_forward(prices, "^GSPC", 5)
    assert np.isclose(b["benchmark_fwd"].iloc[0], np.log(105 / 100))
    assert np.isnan(b["benchmark_fwd"].iloc[-1])


def test_hit_rates_and_weight_tuning(cfg):
    df = _panel()
    m = _hit_rates(df[df["split"] == "test"], cfg)
    assert 0 <= m["overall_hit_rate"] <= 1
    weights, grid = tune_weights(df[df["split"] == "val"], cfg)
    assert abs(weights["forecast"] + weights["sentiment"] + weights["risk"] - 1) < 1e-9
    assert len(grid) > 0
