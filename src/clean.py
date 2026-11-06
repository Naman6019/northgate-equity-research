"""Layer 2 - deterministic cleaning, alignment and audit.

Order is fixed and each stage is logged:
  align to the NYSE calendar -> de-duplicate -> forward-fill short gaps (<=3,
  never backward) -> OHLC invariant quarantine -> return-outlier flags ->
  macro release lags -> tidy outputs + data-quality report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.utils import (
    get_logger,
    load_config,
    path_from_config,
)

LOG = get_logger("northgate.clean")

PRICE_VALUE_COLS = ["open", "high", "low", "close", "adj_close"]
OHLC_COLS = ["open", "high", "low", "close", "adj_close"]
MACRO_COLS = ["DGS10", "DGS3MO", "CPIAUCSL", "UNRATE"]


def nyse_calendar(start: str, end: str) -> pd.DatetimeIndex:
    """NYSE trading sessions between start and end (inclusive), tz-naive."""
    import pandas_market_calendars as mcal

    cal = mcal.get_calendar("NYSE")
    schedule = cal.schedule(start_date=start, end_date=end)
    idx = pd.DatetimeIndex(schedule.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def _clean_one_ticker(sub: pd.DataFrame, calendar: pd.DatetimeIndex, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean one ticker on the shared calendar and return an audit dict."""
    limit = int(cfg["cleaning"]["ffill_limit"])
    sub = sub.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    raw_rows = len(sub)
    sub = sub.reindex(calendar)
    reindexed_rows = len(sub)

    for col in PRICE_VALUE_COLS:
        if col not in sub:
            sub[col] = np.nan
    for col in PRICE_VALUE_COLS:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    volume = sub["volume"] if "volume" in sub else pd.Series(np.nan, index=sub.index)
    sub["volume"] = pd.to_numeric(volume, errors="coerce").fillna(0.0)

    before = sub[PRICE_VALUE_COLS].isna()
    sub[PRICE_VALUE_COLS] = sub[PRICE_VALUE_COLS].ffill(limit=limit)  # never bfill
    gaps_filled = int((before & sub[PRICE_VALUE_COLS].notna()).sum().sum())

    # Stage 5 - OHLC consistency. Violations are quarantined (set to NaN).
    bad = (
        (sub["low"] > sub[["open", "high", "close"]].min(axis=1))
        | (sub["high"] < sub[["open", "low", "close"]].max(axis=1))
        | (sub["low"] > sub["high"])
    )
    invariant_violations = int(bad.sum())
    sub.loc[bad, OHLC_COLS] = np.nan

    # Stage 4 - flag return outliers; keep every flagged row (signal, not error).
    r = np.log(sub["adj_close"]).diff()
    win = int(cfg["cleaning"]["outlier_z_window"])
    z = (r - r.rolling(win).mean()) / r.rolling(win).std()
    z_flag = z.abs() > float(cfg["cleaning"]["outlier_z"])
    q1, q3 = r.quantile(0.25), r.quantile(0.75)
    iqr = q3 - q1
    k = float(cfg["cleaning"]["iqr_multiplier"])
    iqr_flag = (r < q1 - k * iqr) | (r > q3 + k * iqr)
    sub["is_outlier"] = (z_flag | iqr_flag).fillna(False)
    sub["is_large_move"] = (r.abs() >= 0.20).fillna(False)
    sub["log_return"] = r

    # Rows that are structurally missing (leading history or quarantine) are dropped.
    clean = sub.dropna(subset=["adj_close"]).copy()
    clean["date"] = clean.index
    clean["gap_filled"] = before.sum(axis=1).reindex(clean.index).fillna(0).astype(int)
    clean = clean.reset_index(drop=True)

    audit = {
        "raw_rows": raw_rows,
        "calendar_rows": reindexed_rows,
        "clean_rows": int(len(clean)),
        "gaps_filled": gaps_filled,
        "invariant_violations": invariant_violations,
        "outliers_flagged": int(sub["is_outlier"].sum()),
        "large_moves_kept": int(sub["is_large_move"].sum()),
        "winsorised": 0,
        "first_date": str(clean["date"].min().date()) if len(clean) else None,
        "last_date": str(clean["date"].max().date()) if len(clean) else None,
    }
    return clean, audit


def clean_prices(prices: pd.DataFrame, calendar: pd.DatetimeIndex, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    frames = []
    audits: Dict[str, Any] = {}
    for ticker, sub in prices.groupby("ticker", sort=True):
        cleaned, audit = _clean_one_ticker(sub, calendar, cfg)
        cleaned["ticker"] = ticker
        frames.append(cleaned)
        audits[ticker] = audit
        LOG.info("cleaned %-5s raw=%d clean=%d gaps=%d bad=%d outliers=%d",
                 ticker, audit["raw_rows"], audit["clean_rows"], audit["gaps_filled"],
                 audit["invariant_violations"], audit["outliers_flagged"])
    out = pd.concat(frames, ignore_index=True)
    out = out[["date", "ticker"] + PRICE_VALUE_COLS + ["volume", "log_return", "is_outlier", "is_large_move", "gap_filled"]]
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out, audits


def apply_macro_lags(macro: pd.DataFrame, calendar: pd.DatetimeIndex, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Make each macro observation available only after its publication lag.

    A value with reference date D is usable from D + lag (calendar days for the
    monthly series, one business day for the daily yields). This prevents using
    CPI for March in the first week of March, which would leak the future.
    """
    lags = cfg["cleaning"]["macro_lags_days"]
    wide = (
        macro.pivot_table(index="date", columns="series", values="value", aggfunc="last")
        .sort_index()
        .reindex(columns=MACRO_COLS)
    )
    wide.index = pd.to_datetime(wide.index).normalize()
    out = pd.DataFrame({"date": calendar})
    audit: Dict[str, Any] = {}

    for series in MACRO_COLS:
        s = wide[series].dropna()
        if s.empty:
            out[series] = np.nan
            audit[series] = {"lag_days": lags.get(series, 0), "observations": 0}
            continue
        lag = int(lags.get(series, 0))
        if series in ("DGS10", "DGS3MO"):
            available = s.index + pd.offsets.BDay(lag)
        else:
            available = s.index + pd.Timedelta(days=lag)
        avail = pd.DataFrame({"date": pd.DatetimeIndex(available), "value": s.to_numpy()}).sort_values("date")
        avail = avail.drop_duplicates("date", keep="last")
        merged = pd.merge_asof(out[["date"]], avail, on="date", direction="backward")
        out[series] = merged["value"].to_numpy()
        audit[series] = {
            "lag_days": lag,
            "lag_type": "business_day" if series in ("DGS10", "DGS3MO") else "calendar_day",
            "observations": int(len(s)),
            "first_reference_date": str(s.index.min().date()),
            "last_reference_date": str(s.index.max().date()),
        }
    out = out.ffill()  # after availability alignment this only fills forward, never back
    return out, audit


def write_data_quality_report(
    price_audits: Dict[str, Any],
    macro_audit: Dict[str, Any],
    prices_clean: pd.DataFrame,
    macro_clean: pd.DataFrame,
    news: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Path:
    reports = path_from_config(cfg, "reports")
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "data_quality.md"

    total_raw = sum(a["raw_rows"] for a in price_audits.values())
    total_clean = sum(a["clean_rows"] for a in price_audits.values())
    total_gaps = sum(a["gaps_filled"] for a in price_audits.values())
    total_bad = sum(a["invariant_violations"] for a in price_audits.values())
    total_out = sum(a["outliers_flagged"] for a in price_audits.values())
    total_large = sum(a["large_moves_kept"] for a in price_audits.values())

    lines = [
        "# Northgate data-quality report",
        "",
        f"Generated for the research window **{cfg['dates']['start']} -> {cfg['dates']['end']}**.",
        "",
        "## Price panel",
        "",
        f"- Vendor rows in: **{total_raw:,}**; rows out on the NYSE calendar: **{total_clean:,}**.",
        f"- Short gaps forward-filled (limit {cfg['cleaning']['ffill_limit']}): **{total_gaps:,}**.",
        f"- OHLC invariant violations quarantined: **{total_bad:,}**.",
        f"- Return outliers flagged (z > {cfg['cleaning']['outlier_z']} or 1.5x IQR): **{total_out:,}**; all kept.",
        f"- Large moves kept (|return| >= 20%, likely earnings/corporate actions): **{total_large:,}**.",
        "- No value was winsorised; no price series was backward-filled.",
        "",
        "| Ticker | Rows in | Rows out | Gaps filled | Invariant violations | Outliers flagged | Large moves kept |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ticker, a in sorted(price_audits.items()):
        lines.append(
            f"| {ticker} | {a['raw_rows']:,} | {a['clean_rows']:,} | {a['gaps_filled']:,} | "
            f"{a['invariant_violations']:,} | {a['outliers_flagged']:,} | {a['large_moves_kept']:,} |"
        )

    lines += [
        "",
        "## Macro release lags",
        "",
        "A macro value is shifted by a conservative publication lag before it is",
        "merged onto the trading calendar, so the model never sees a figure before",
        "it was published.",
        "",
        "| Series | Lag | Observations | First reference | Last reference |",
        "|---|---:|---:|---|---|",
    ]
    for series, a in macro_audit.items():
        lines.append(
            f"| {series} | {a['lag_days']} {a.get('lag_type', 'day')}(s) | {a['observations']:,} | "
            f"{a.get('first_reference_date', '-')} | {a.get('last_reference_date', '-')} |"
        )

    lines += [
        "",
        "## News coverage (FNSPID)",
        "",
    ]
    if news is not None and len(news):
        news = news.copy()
        news["year"] = pd.to_datetime(news["timestamp"], utc=True).dt.year
        piv = news.pivot_table(index="ticker", columns="year", values="title", aggfunc="size", fill_value=0)
        years = sorted(piv.columns)
        lines.append("| Ticker | " + " | ".join(str(y) for y in years) + " |")
        lines.append("|---|" + "---:|" * len(years))
        for ticker, row in piv.iterrows():
            lines.append(f"| {ticker} | " + " | ".join(str(int(row[y])) for y in years) + " |")
        lines += [
            "",
            "Coverage is uneven: some tickers have no headlines before 2018 and four",
            "(HD, JNJ, JPM, PG) stop in 2020. The sentiment ablation is therefore run",
            "on the well-covered names and every row reports coverage explicitly, as",
            "the implementation plan's fallback requires.",
        ]
    else:
        lines.append("No news rows were available.")

    lines += [
        "",
        "## Outputs",
        "",
        f"- `{path_from_config(cfg, 'processed') / 'prices_clean.parquet'}`",
        f"- `{path_from_config(cfg, 'processed') / 'macro_clean.parquet'}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

    audit_json = {
        "prices": price_audits,
        "macro": macro_audit,
        "price_rows": int(len(prices_clean)),
        "macro_rows": int(len(macro_clean)),
        "news_rows": int(len(news)) if news is not None else 0,
    }
    (reports / "data_quality.json").write_text(json.dumps(audit_json, indent=2, default=str), encoding="utf-8")
    return path


def run_cleaning(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    raw = path_from_config(cfg, "raw")
    processed = path_from_config(cfg, "processed")
    processed.mkdir(parents=True, exist_ok=True)

    prices = pd.read_parquet(raw / "prices.parquet")
    macro = pd.read_parquet(raw / "macro.parquet")
    news_path = raw / "news.parquet"
    news = pd.read_parquet(news_path) if news_path.exists() else pd.DataFrame()

    calendar = nyse_calendar(cfg["dates"]["start"], cfg["dates"]["end"])
    LOG.info("NYSE calendar: %d sessions", len(calendar))
    prices_clean, price_audits = clean_prices(prices, calendar, cfg)
    macro_clean, macro_audit = apply_macro_lags(macro, calendar, cfg)

    prices_clean.to_parquet(processed / "prices_clean.parquet", index=False)
    macro_clean.to_parquet(processed / "macro_clean.parquet", index=False)
    report = write_data_quality_report(price_audits, macro_audit, prices_clean, macro_clean, news, cfg)
    LOG.info("wrote %s and processed panels", report)
    return {
        "prices": prices_clean,
        "macro": macro_clean,
        "audit": {"prices": price_audits, "macro": macro_audit},
    }


if __name__ == "__main__":
    run_cleaning()
