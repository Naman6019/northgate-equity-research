"""Shared fixtures for the Northgate test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def make_prices(tickers, periods=300, seed=7, start="2018-01-01"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=periods)
    frames = []
    for i, ticker in enumerate(tickers):
        r = rng.normal(0.0003 + i * 0.00002, 0.012, size=periods)
        close = 100 * np.exp(np.cumsum(r))
        high = close * (1 + rng.uniform(0.001, 0.02, periods))
        low = close * (1 - rng.uniform(0.001, 0.02, periods))
        open_ = close * (1 + rng.normal(0, 0.004, periods))
        low = np.minimum(low, np.minimum(open_, close) * 0.999)
        high = np.maximum(high, np.maximum(open_, close) * 1.001)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": close,
                    "volume": rng.integers(1_000_000, 9_000_000, periods),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def make_macro(dates, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    monthly = pd.date_range(dates.min(), dates.max(), freq="MS")
    for series, base in (("DGS10", 2.5), ("DGS3MO", 2.0), ("CPIAUCSL", 250.0), ("UNRATE", 5.0)):
        if series in ("DGS10", "DGS3MO"):
            idx = dates[::5]
            vals = base + rng.normal(0, 0.1, len(idx))
        else:
            idx = monthly
            vals = base * np.exp(np.cumsum(rng.normal(0, 0.002, len(idx))))
        rows.append(pd.DataFrame({"date": idx, "series": series, "value": vals}))
    long = pd.concat(rows, ignore_index=True)
    wide = long.pivot_table(index="date", columns="series", values="value").sort_index()
    return wide.reset_index()
