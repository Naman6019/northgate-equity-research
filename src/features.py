"""Layer 2 - leak-free feature engineering, targets, splits and sequences.

Every feature at date t uses rows with date <= t. Targets are forward log
returns built with explicit index arithmetic (never a negative shift), and the
shared chronological split leaves an h-day embargo at each boundary.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, path_from_config

LOG = get_logger("northgate.features")

TECH_FEATURES = [
    "ret1", "ret5", "ret10", "ret21",
    "lag_ret1", "lag_ret2", "lag_ret3", "lag_ret4", "lag_ret5",
    "sma10", "sma20", "sma50", "sma200",
    "p_sma10", "p_sma20", "p_sma50", "p_sma200",
    "ema10", "ema20", "ema50",
    "macd", "macd_signal", "macd_hist",
    "rsi14", "stoch_k", "stoch_d", "roc10",
    "vol21", "atr14", "bb_width",
    "volume_z63", "obv_slope", "vol_price_corr",
    "dow", "month", "is_month_start", "is_month_end",
    "mkt_ret1", "mkt_ret5", "mkt_ret21", "vix_level", "vix_change5",
    "term_spread", "cpi_yoy", "unrate", "sector_ret5", "beta63",
]

SENTIMENT_FEATURES = ["sentiment", "sentiment_mom3", "news_spike", "market_mood"]


def _rsi(p: pd.Series, window: int = 14) -> pd.Series:
    d = p.diff()
    up = d.clip(lower=0).rolling(window).mean()
    down = (-d.clip(upper=0)).rolling(window).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_technical_features(g: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker technical, volume and calendar features (all <= t)."""
    g = g.sort_values("date").copy()
    p = g["adj_close"]
    r = np.log(p).diff()
    g["ret1"] = r
    for h in (5, 10, 21):
        g[f"ret{h}"] = np.log(p / p.shift(h))
    for lag in range(1, 6):
        g[f"lag_ret{lag}"] = r.shift(lag)

    for w in (10, 20, 50, 200):
        sma = p.rolling(w).mean()
        g[f"sma{w}"] = sma
        g[f"p_sma{w}"] = p / sma
    for w in (10, 20, 50):
        g[f"ema{w}"] = p.ewm(span=w, adjust=False).mean()
    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    g["macd"] = ema12 - ema26
    g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
    g["macd_hist"] = g["macd"] - g["macd_signal"]

    g["rsi14"] = _rsi(p, 14)
    low14 = g["low"].rolling(14).min()
    high14 = g["high"].rolling(14).max()
    g["stoch_k"] = 100 * (g["close"] - low14) / (high14 - low14).replace(0, np.nan)
    g["stoch_d"] = g["stoch_k"].rolling(3).mean()
    g["roc10"] = p.pct_change(10) * 100

    g["vol21"] = r.rolling(21).std()
    prev_close = g["close"].shift(1)
    tr = pd.concat(
        [(g["high"] - g["low"]), (g["high"] - prev_close).abs(), (g["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    g["atr14"] = tr.rolling(14).mean()
    mid = p.rolling(20).mean()
    sd = p.rolling(20).std()
    g["bb_width"] = (4 * sd) / mid

    vol = g["volume"]
    g["volume_z63"] = (vol - vol.rolling(63).mean()) / vol.rolling(63).std().replace(0, np.nan)
    obv = (np.sign(r.fillna(0)) * vol).cumsum()
    g["obv_slope"] = obv.diff(5) / vol.rolling(21).mean().replace(0, np.nan)
    g["vol_price_corr"] = r.rolling(21).corr(vol.pct_change().replace([np.inf, -np.inf], np.nan))

    dates = pd.to_datetime(g["date"])
    g["dow"] = dates.dt.day_of_week
    g["month"] = dates.dt.month
    month_id = dates.dt.to_period("M")
    rank_start = dates.groupby(month_id).rank(method="first")
    size_month = dates.groupby(month_id).transform("size")
    g["is_month_start"] = (rank_start <= 2).astype(int)
    g["is_month_end"] = (rank_start > size_month - 2).astype(int)
    return g


def _market_context(prices: pd.DataFrame, macro: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """One row per date with market, macro, VIX and sector-ETF context."""
    p = prices.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index()
    u = cfg["universe"]
    out = pd.DataFrame(index=p.index)

    mkt = p[u["benchmark"]]
    out["mkt_ret1"] = np.log(mkt).diff()
    out["mkt_ret5"] = np.log(mkt / mkt.shift(5))
    out["mkt_ret21"] = np.log(mkt / mkt.shift(21))

    vix = p[u["volatility"]]
    out["vix_level"] = vix
    out["vix_change5"] = vix.diff(5)

    m = macro.set_index("date").sort_index().reindex(out.index).ffill()
    out["term_spread"] = m["DGS10"] - m["DGS3MO"]
    out["cpi_yoy"] = m["CPIAUCSL"].pct_change(12) * 100
    out["unrate"] = m["UNRATE"]

    for etf in u["context_etfs"]:
        if etf in p:
            out[f"etf_ret5_{etf}"] = np.log(p[etf] / p[etf].shift(5))
    return out


def _beta63(panel: pd.DataFrame, market: pd.Series) -> pd.Series:
    out = pd.Series(index=panel.index, dtype=float)
    for _ticker, idx in panel.groupby("ticker").groups.items():
        r = panel.loc[idx, "ret1"]
        m = market.reindex(panel.loc[idx, "date"]).to_numpy()
        rv = r.to_numpy()
        s = pd.Series(rv, index=panel.loc[idx, "date"])
        mv = pd.Series(m, index=s.index)
        cov = s.rolling(63).cov(mv)
        var = mv.rolling(63).var()
        out.loc[idx] = (cov / var.replace(0, np.nan)).to_numpy()
    return out


def build_panel(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    sentiment_daily: Optional[pd.DataFrame] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    cfg = cfg or load_config()
    u = cfg["universe"]
    macro = macro.sort_values("date")

    tech = pd.concat(
        [add_technical_features(g) for _, g in prices.groupby("ticker", sort=True)],
        ignore_index=True,
    )
    context = _market_context(prices, macro, cfg)

    panel = tech[tech["ticker"].isin(u["equities"])].copy()
    panel = panel.merge(context.reset_index(), on="date", how="left")
    panel["sector_ret5"] = np.nan
    etf_cols = [f"etf_ret5_{e}" for e in u["context_etfs"] if f"etf_ret5_{e}" in panel.columns]
    for ticker, etf in u["sector_map"].items():
        col = f"etf_ret5_{etf}"
        if col in panel.columns:
            mask = panel["ticker"] == ticker
            panel.loc[mask, "sector_ret5"] = panel.loc[mask, col]
    panel = panel.drop(columns=etf_cols)
    market = context["mkt_ret1"]
    panel["beta63"] = _beta63(panel, market)

    if sentiment_daily is not None and len(sentiment_daily):
        s = sentiment_daily.copy()
        s["date"] = pd.to_datetime(s["date"]).dt.normalize()
        cols = [c for c in SENTIMENT_FEATURES if c in s.columns]
        panel = panel.merge(s[["date", "ticker"] + cols], on=["date", "ticker"], how="left", suffixes=("", "_s"))
        panel["sentiment_available"] = panel[cols[0]].notna().astype(int) if cols else 0
    else:
        for col in SENTIMENT_FEATURES:
            panel[col] = 0.0
        panel["sentiment_available"] = 0

    for col in SENTIMENT_FEATURES:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0.0)

    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["ticker_id"] = panel["ticker"].map({t: i for i, t in enumerate(u["equities"])}).astype(int)

    # Targets: explicit forward index arithmetic, no negative shift.
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    for h in cfg["horizons"]:
        vals = np.full(len(panel), np.nan)
        dates = np.full(len(panel), np.datetime64("NaT"), dtype="datetime64[ns]")
        prices_fwd = np.full(len(panel), np.nan)
        for _, idx in panel.groupby("ticker").groups.items():
            idx = np.asarray(list(idx))
            p = panel.loc[idx, "adj_close"].to_numpy()
            d = panel.loc[idx, "date"].to_numpy()
            if len(p) > h:
                vals[idx[:-h]] = np.log(p[h:] / p[:-h])
                dates[idx[:-h]] = d[h:]
                prices_fwd[idx[:-h]] = p[h:]
        panel[f"target_h{h}"] = vals
        panel[f"target_price_h{h}"] = prices_fwd
        panel[f"target_date_h{h}"] = dates

    panel = panel.dropna(subset=TECH_FEATURES + ["target_h1"]).reset_index(drop=True)
    return panel


def assign_split(panel: pd.DataFrame, cfg: Dict[str, Any], horizon: int = 5, write_info: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Shared chronological 70/15/15 split with an h-day embargo at boundaries."""
    dates = np.sort(panel["date"].unique())
    n = len(dates)
    train_end = int(np.floor(cfg["split"]["train"] * n))
    val_end = int(np.floor((cfg["split"]["train"] + cfg["split"]["val"]) * n))
    train_stop = max(train_end - horizon, 0)
    val_stop = max(val_end - horizon, train_end)
    split = np.full(n, "gap", dtype=object)
    split[:train_stop] = "train"
    split[train_end:val_stop] = "val"
    split[val_end:] = "test"
    mapping = dict(zip(dates, split, strict=False))
    panel["split"] = panel["date"].map(mapping)
    panel = panel[panel["split"].isin(["train", "val", "test"])].reset_index(drop=True)

    def _range(name: str) -> List[str]:
        d = dates[split == name]
        return [str(pd.Timestamp(d.min()).date()), str(pd.Timestamp(d.max()).date())] if len(d) else []

    info = {
        "horizon_embargo": horizon,
        "sessions": int(n),
        "train": _range("train"),
        "val": _range("val"),
        "test": _range("test"),
        "gap_sessions": int((split == "gap").sum()),
        "rows": {k: int((panel["split"] == k).sum()) for k in ("train", "val", "test")},
    }
    cfg_path = path_from_config(cfg, "reports") / "split.json"
    if write_info:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return panel, info


def feature_columns(include_sentiment: bool = True) -> List[str]:
    cols = list(TECH_FEATURES)
    if include_sentiment:
        cols += SENTIMENT_FEATURES
    return cols


def fit_scalers(
    panel: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Fit feature scalers and the target scaler on the training split only."""
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    train = panel[panel["split"] == "train"]
    std = StandardScaler().fit(train[feature_cols].to_numpy(dtype=float))
    mm = MinMaxScaler().fit(train[feature_cols].to_numpy(dtype=float))
    ts = StandardScaler().fit(train[[target_col]].to_numpy(dtype=float))
    return {"standard": std, "minmax": mm, "target": ts, "feature_cols": list(feature_cols), "target_col": target_col}


def make_sequences(
    panel: pd.DataFrame,
    feature_cols: List[str],
    length: int,
    horizon: int,
    split: str,
    scaler=None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Windows that never cross a split boundary or a ticker boundary."""
    Xs, ys, meta = [], [], []
    scaler = scaler["minmax"] if isinstance(scaler, dict) else scaler
    for ticker, g in panel[panel["split"] == split].groupby("ticker", sort=True):
        g = g.sort_values("date")
        if len(g) < length:
            continue
        X = g[feature_cols].to_numpy(dtype=float)
        if scaler is not None:
            X = scaler.transform(X)
        y = g[f"target_h{horizon}"].to_numpy(dtype=float)
        for i in range(len(g) - length + 1):
            target_idx = i + length - 1
            if np.isnan(y[target_idx]):
                continue
            Xs.append(X[i : i + length])
            ys.append(y[target_idx])
            meta.append(
                {
                    "ticker": ticker,
                    "date": g["date"].iloc[target_idx],
                    "target_date": g[f"target_date_h{horizon}"].iloc[target_idx],
                    "split": split,
                }
            )
    if not Xs:
        return np.empty((0, length, len(feature_cols))), np.empty((0,)), pd.DataFrame(columns=["ticker", "date", "target_date", "split"])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), pd.DataFrame(meta)


def build_features(cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    processed = path_from_config(cfg, "processed")
    prices = pd.read_parquet(processed / "prices_clean.parquet")
    macro = pd.read_parquet(processed / "macro_clean.parquet")
    sent_path = processed / "sentiment_daily.parquet"
    sentiment = pd.read_parquet(sent_path) if sent_path.exists() else None
    panel = build_panel(prices, macro, sentiment, cfg)
    panel, info = assign_split(panel, cfg, horizon=max(cfg["horizons"]))
    panel.to_parquet(processed / "panel.parquet", index=False)
    LOG.info("panel: %d rows, %s", len(panel), info["rows"])
    return panel


if __name__ == "__main__":
    build_features()
