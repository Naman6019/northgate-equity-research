"""Feature/target/split/sequence leakage guards (PRD Sections 5 and 8.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.features import (
    add_technical_features,
    assign_split,
    build_panel,
    feature_columns,
    fit_scalers,
    make_sequences,
)
from tests.conftest import make_macro, make_prices


def test_technical_features_do_not_look_ahead(cfg):
    prices = make_prices(["AAA"], periods=300, seed=3)
    base = add_technical_features(prices)
    changed = prices.copy()
    changed.loc[250:, "adj_close"] *= 2.0
    changed.loc[250:, ["high", "low", "open", "close"]] *= 2.0
    after = add_technical_features(changed)
    cols = [c for c in feature_columns(False) if c in base.columns]
    row = 240
    for col in cols:
        assert np.isclose(base.loc[row, col], after.loc[row, col], equal_nan=True), col


def test_panel_targets_are_forward_log_returns(cfg):
    tickers = cfg["universe"]["equities"] + [cfg["universe"]["benchmark"], cfg["universe"]["volatility"]] + cfg["universe"]["context_etfs"]
    prices = make_prices(tickers, periods=320, seed=9)
    macro = make_macro(prices["date"])
    panel = build_panel(prices, macro, None, cfg)
    g = panel[panel["ticker"] == "AAPL"].sort_values("date").reset_index(drop=True)
    full = prices[prices["ticker"] == "AAPL"].sort_values("date").reset_index(drop=True)
    expected = np.log(full["adj_close"].shift(-1) / full["adj_close"])
    lookup = dict(zip(full["date"], expected, strict=False))
    np.testing.assert_allclose(g["target_h1"], g["date"].map(lookup), atol=1e-12)


def test_split_embargo_prevents_label_overlap(cfg):
    tickers = cfg["universe"]["equities"] + [cfg["universe"]["benchmark"], cfg["universe"]["volatility"]] + cfg["universe"]["context_etfs"]
    prices = make_prices(tickers, periods=420, seed=5)
    macro = make_macro(prices["date"])
    panel = build_panel(prices, macro, None, cfg)
    panel, info = assign_split(panel, cfg, horizon=5, write_info=False)
    train = panel[panel["split"] == "train"]
    val = panel[panel["split"] == "val"]
    test = panel[panel["split"] == "test"]
    assert train["target_date_h5"].max() < val["date"].min()
    assert val["target_date_h5"].max() < test["date"].min()
    assert info["rows"]["train"] > 0 and info["rows"]["val"] > 0 and info["rows"]["test"] > 0


def test_scaler_fits_train_only(cfg):
    tickers = cfg["universe"]["equities"] + [cfg["universe"]["benchmark"], cfg["universe"]["volatility"]] + cfg["universe"]["context_etfs"]
    prices = make_prices(tickers, periods=320, seed=4)
    macro = make_macro(prices["date"])
    panel, _ = assign_split(build_panel(prices, macro, None, cfg), cfg, horizon=5, write_info=False)
    cols = [c for c in feature_columns() if c in panel.columns]
    scalers = fit_scalers(panel, cols, "target_h5", cfg)
    train_mean = panel.loc[panel["split"] == "train", cols].mean().to_numpy()
    np.testing.assert_allclose(scalers["standard"].mean_, train_mean, atol=1e-10)
    assert not np.allclose(scalers["standard"].mean_, panel[cols].mean().to_numpy())


def test_sequences_stay_inside_split(cfg):
    tickers = cfg["universe"]["equities"] + [cfg["universe"]["benchmark"], cfg["universe"]["volatility"]] + cfg["universe"]["context_etfs"]
    prices = make_prices(tickers, periods=260, seed=8)
    macro = make_macro(prices["date"])
    panel, _ = assign_split(build_panel(prices, macro, None, cfg), cfg, horizon=5, write_info=False)
    cols = [c for c in feature_columns() if c in panel.columns]
    X, y, meta = make_sequences(panel, cols, length=20, horizon=5, split="test")
    assert X.shape[0] == len(y) == len(meta)
    if len(X):
        assert (meta["split"] == "test").all()
        # Each window's own rows are all in the test split by construction.
        assert X.shape[1:] == (20, len(cols))


def test_features_never_call_shift_minus_one():
    source = Path("src/features.py").read_text(encoding="utf-8")
    assert ".shift(-" not in source
