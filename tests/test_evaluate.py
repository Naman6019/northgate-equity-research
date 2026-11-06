"""Evaluation harness sanity checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluate import (
    baseline_metrics,
    bootstrap_metric_ci,
    direction_accuracy,
    evaluate_predictions,
    price_metrics,
    return_metrics,
)


def test_price_and_return_metrics():
    yt = np.array([100.0, 101.0, 102.0])
    yp = np.array([100.0, 101.0, 102.0])
    m = price_metrics(yt, yp)
    assert m["rmse"] == 0 and m["mae"] == 0 and m["mape"] == 0 and m["r2"] == 1.0
    r = return_metrics(np.array([0.01, -0.02, 0.03]), np.array([0.01, -0.02, 0.03]))
    assert r["mae_return"] == 0 and r["ic"] == 1.0 and r["direction_accuracy"] == 1.0


def test_direction_accuracy():
    assert direction_accuracy(np.array([1.0, -1.0]), np.array([1.0, -1.0])) == 1.0
    assert direction_accuracy(np.array([1.0, -1.0]), np.array([-1.0, 1.0])) == 0.0


def test_baseline_has_direction_bars():
    yt = np.array([0.01, 0.02, -0.01, 0.03])
    pt = np.array([100.0, 101.0, 103.0, 102.0])
    pf = pt * np.exp(yt)
    b = baseline_metrics(yt, pt, pf)
    assert np.isclose(b["mae_return"], np.mean(np.abs(yt)))
    assert np.isclose(b["always_up_accuracy"], 0.75)
    assert b["coin_flip_accuracy"] == 0.5


def test_evaluate_predictions_price_space():
    yt = np.array([0.01, 0.02])
    pt = np.array([100.0, 100.0])
    pf = pt * np.exp(yt)
    m = evaluate_predictions(pt, pf, yt, yt)
    assert np.isclose(m["mape"], 0.0, atol=1e-9)
    assert np.isclose(m["r2"], 1.0)


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(0)
    dates = np.repeat(pd.date_range("2020-01-01", periods=60), 3)
    y = rng.normal(0, 0.01, len(dates))
    p = y + rng.normal(0, 0.005, len(dates))
    lo, hi = bootstrap_metric_ci(dates, y, p, metric="mae_return", n_resamples=100, block=5)
    assert lo <= np.mean(np.abs(y - p)) <= hi
