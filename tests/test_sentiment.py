"""Session alignment is the correctness crux of the sentiment module (PRD 10.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.clean import nyse_calendar
from src.sentiment import aggregate_daily, clean_text, session_dates


def test_clean_text():
    s = "ACME (NASDAQ: ACME) beats Q3 EPS estimates! https://example.com $ACME"
    out = clean_text(s)
    assert "http" not in out
    assert "acme" in out
    assert "third quarter" in out
    assert "earnings per share" in out


def test_session_alignment_boundaries():
    cal = nyse_calendar("2020-01-01", "2020-12-31")
    ts = pd.Series(
        pd.to_datetime(
            [
                "2020-01-10 20:59:00",  # Friday 15:59 ET
                "2020-01-10 21:00:00",  # Friday 16:00 ET -> Monday
                "2020-01-11 15:00:00",  # Saturday -> Monday
                "2020-07-03 15:00:00",  # July 4 observed holiday -> Monday 6th
                "2020-03-09 20:00:00",  # Monday 16:00 EDT -> Tuesday
            ],
            utc=True,
        )
    )
    got = session_dates(ts, cal).tolist()
    assert got[0] == pd.Timestamp("2020-01-10")
    assert got[1] == pd.Timestamp("2020-01-13")
    assert got[2] == pd.Timestamp("2020-01-13")
    assert got[3] == pd.Timestamp("2020-07-06")
    assert got[4] == pd.Timestamp("2020-03-10")


def test_date_only_counts_for_next_session():
    cal = nyse_calendar("2020-01-01", "2020-12-31")
    ts = pd.Series(pd.to_datetime(["2020-01-10 00:00:00"], utc=True))
    assert session_dates(ts, cal).iloc[0] == pd.Timestamp("2020-01-13")


def test_daily_aggregation_uses_session_and_article_mean():
    cal = nyse_calendar("2020-01-01", "2020-01-31")
    scored = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2020-01-10 21:30:00", "2020-01-11 15:00:00"], utc=True
            ),
            "ticker": ["AAPL", "AAPL"],
            "polarity": [1.0, -1.0],
            "vader": [0.5, -0.5],
        }
    )
    daily = aggregate_daily(scored, cal)
    row = daily[(daily["ticker"] == "AAPL") & (daily["date"] == pd.Timestamp("2020-01-13"))]
    assert len(row) == 1
    assert np.isclose(row["sentiment"].iloc[0], 0.0)
    assert row["articles"].iloc[0] == 2
