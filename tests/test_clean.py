"""Cleaning pipeline guards: no back-fill, limited forward-fill, OHLC
quarantine, outlier flags kept, and macro release lags (PRD Section 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.clean import apply_macro_lags, clean_prices


def _price_frame(ticker, dates, start_price=100.0):
    n = len(dates)
    close = start_price * np.exp(np.linspace(0, 0.15, n))
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": 1_000_000,
        }
    )


def test_ohlc_violation_is_quarantined_and_large_move_kept(cfg):
    dates = pd.bdate_range("2020-01-01", periods=12)
    a = _price_frame("AAA", dates)
    a.loc[7, ["low", "high"]] = [a.loc[7, "close"] * 1.2, a.loc[7, "close"] * 0.8]  # low > high
    a.loc[9, "adj_close"] = a.loc[8, "adj_close"] * 1.25  # +25% move
    a.loc[9, ["open", "high", "low", "close"]] = a.loc[9, "adj_close"]
    b = _price_frame("BBB", dates)
    b.loc[:3, "adj_close"] = np.nan  # leading missing: must not be back-filled

    clean, audit = clean_prices(pd.concat([a, b], ignore_index=True), dates, cfg)
    aa = clean[clean["ticker"] == "AAA"]
    assert dates[7] not in set(aa["date"]), "OHLC violation must be quarantined"
    assert bool(aa.loc[aa["date"] == dates[9], "is_large_move"].iloc[0])
    assert len(aa) == len(dates) - 1

    bb = clean[clean["ticker"] == "BBB"]
    assert bb["date"].min() >= dates[4], "leading NaNs must not be back-filled"
    assert audit["BBB"]["gaps_filled"] == 0


def test_forward_fill_stops_at_limit(cfg):
    dates = pd.bdate_range("2020-01-01", periods=12)
    a = _price_frame("AAA", dates)
    a = a.drop(index=[3, 4, 5, 6, 7]).reset_index(drop=True)  # five-session gap
    clean, _ = clean_prices(a, dates, cfg)
    assert len(clean) == len(dates) - 2, "ffill(limit=3) must leave two sessions missing"


def test_no_back_fill_on_leading_gap(cfg):
    dates = pd.bdate_range("2020-01-01", periods=6)
    a = _price_frame("AAA", dates)
    a.loc[:2, "adj_close"] = np.nan
    clean, _ = clean_prices(a, dates, cfg)
    assert len(clean) == 3
    assert clean["date"].min() == dates[3]


def test_macro_release_lags(cfg):
    cal = pd.bdate_range("2019-12-01", "2020-03-31")
    macro = pd.DataFrame(
        {
            "date": [pd.Timestamp("2019-12-01"), pd.Timestamp("2020-01-01")],
            "series": ["CPIAUCSL", "CPIAUCSL"],
            "value": [90.0, 100.0],
        }
    )
    wide, audit = apply_macro_lags(macro, cal, cfg)
    before = wide.loc[wide["date"] == pd.Timestamp("2020-02-14"), "CPIAUCSL"].iloc[0]
    at_lag = wide.loc[wide["date"] >= pd.Timestamp("2020-02-15"), "CPIAUCSL"].iloc[0]
    assert np.isclose(before, 90.0), "CPI for January must not be visible 44 days later"
    assert np.isclose(at_lag, 100.0), "CPI must appear at reference + 45 days"
    assert audit["CPIAUCSL"]["lag_days"] == 45


def test_macro_yield_uses_one_business_day(cfg):
    cal = pd.bdate_range("2020-01-01", "2020-01-15")
    macro = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-03")],
            "series": ["DGS10"],
            "value": [1.8],
        }
    )
    wide, _ = apply_macro_lags(macro, cal, cfg)
    assert np.isnan(wide.loc[wide["date"] == pd.Timestamp("2020-01-03"), "DGS10"].iloc[0])
    assert np.isclose(wide.loc[wide["date"] == pd.Timestamp("2020-01-06"), "DGS10"].iloc[0], 1.8)
