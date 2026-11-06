"""Layer 4 - financial-news sentiment with session alignment (PRD Section 10).

Pipeline: clean -> FinBERT + VADER (cached by sha1) -> map each headline to the
next tradable session -> aggregate to daily features and a market-mood index.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, path_from_config, save_json

LOG = get_logger("northgate.sentiment")

ABBREVIATIONS = {
    "q1": "first quarter",
    "q2": "second quarter",
    "q3": "third quarter",
    "q4": "fourth quarter",
    "eps": "earnings per share",
    "y/y": "year over year",
    "yoy": "year over year",
    "q/q": "quarter over quarter",
    "qoq": "quarter over quarter",
    "fy": "fiscal year",
    "ltm": "last twelve months",
    "guidance": "guidance",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
TICKER_RE = re.compile(r"\((?:(?:nasdaq|nyse|amex)\s*:\s*)?[A-Z]{1,6}\)|\$[A-Z]{1,6}\b")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lower-case, strip URLs/ticker tags and expand common finance shorthand."""
    if not isinstance(text, str):
        return ""
    s = text.lower()
    s = URL_RE.sub(" ", s)
    s = TICKER_RE.sub(" ", s)
    for short, long in ABBREVIATIONS.items():
        s = re.sub(rf"\b{re.escape(short)}\b", long, s)
    s = re.sub(r"[^a-z0-9%.\- ]+", " ", s)
    return WHITESPACE_RE.sub(" ", s).strip()


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def session_dates(timestamps: pd.Series, calendar: pd.DatetimeIndex, close_hour: int = 16, tz: str = "America/New_York") -> pd.Series:
    """Map UTC timestamps to the session they can inform.

    Rules: date-only stamps and stamps at/after the close count for the next
    session; weekends/holidays roll forward; DST is handled by tz conversion.
    """
    ts = pd.to_datetime(timestamps, errors="coerce", utc=True)
    raw = ts.dt.tz_localize(None)
    date_only = (raw.dt.hour == 0) & (raw.dt.minute == 0) & (raw.dt.second == 0)
    local = ts.dt.tz_convert(tz)
    naive = local.dt.tz_localize(None)
    after_close = naive.dt.hour >= close_hour
    candidate = naive.dt.normalize() + pd.to_timedelta((after_close).astype(int), unit="D")
    # A date-only stamp is the article's date: treat it as after the close of
    # that (UTC) date, so it can only inform the next session.
    candidate = candidate.mask(date_only, raw.dt.normalize() + pd.Timedelta(days=1))
    pos = calendar.searchsorted(candidate.to_numpy(), side="left")
    out = np.full(len(candidate), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = pos < len(calendar)
    out[valid] = calendar.to_numpy()[pos[valid]]
    return pd.Series(out, index=timestamps.index)


def _score_finbert(texts: List[str], model_name: str, batch_size: int = 64) -> pd.DataFrame:
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("sentiment-analysis", model=model_name, truncation=True, device=device)
    rows = []
    for start in range(0, len(texts), 512):
        batch = texts[start : start + 512]
        outputs = pipe(batch, batch_size=batch_size, truncation=True)
        rows.extend(outputs)
        if (start // 512) % 10 == 0:
            LOG.info("FinBERT scored %d/%d", min(start + 512, len(texts)), len(texts))
    labels = [o["label"].lower() for o in rows]
    scores = [float(o["score"]) for o in rows]
    return pd.DataFrame({"finbert_label": labels, "finbert_score": scores})


def score_news(news: pd.DataFrame, cfg: Dict[str, Any], force: bool = False) -> pd.DataFrame:
    out = news.copy()
    out["clean_text"] = out["text"].map(clean_text)
    out = out[out["clean_text"].str.len() > 0].copy()
    out["sha1"] = out["clean_text"].map(sha1)
    cache_dir = path_from_config(cfg, "raw") / "news_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "finbert_scores.parquet"
    model_name = cfg["news"]["finbert_model"]

    cache = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame(columns=["sha1", "finbert_label", "finbert_score"])
    unique = out.drop_duplicates("sha1")[["sha1", "clean_text"]]
    missing = unique[~unique["sha1"].isin(cache["sha1"])]
    if force:
        missing = unique
        cache = cache.iloc[0:0]
    LOG.info("sentiment: %d unique texts, %d missing from cache", len(unique), len(missing))

    if len(missing):
        scored = _score_finbert(missing["clean_text"].tolist(), model_name, int(cfg["news"]["batch_size"]))
        scored.insert(0, "sha1", missing["sha1"].to_numpy())
        cache = pd.concat([cache, scored], ignore_index=True).drop_duplicates("sha1", keep="last")
        cache.to_parquet(cache_path, index=False)
        LOG.info("cached FinBERT scores -> %s", cache_path)

    out = out.merge(cache, on="sha1", how="left")

    # VADER lexicon baseline (fast; no cache needed).
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    out["vader"] = out["clean_text"].map(lambda s: analyzer.polarity_scores(s)["compound"])

    sign = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    out["polarity"] = out["finbert_label"].map(sign).fillna(0.0) * out["finbert_score"].fillna(0.0)
    out["vader_polarity"] = out["vader"] * 0.0 + out["vader"]  # keep both for validation
    return out


def aggregate_daily(scored: pd.DataFrame, calendar: pd.DatetimeIndex, close_hour: int = 16, tz: str = "America/New_York") -> pd.DataFrame:
    out = scored.copy()
    out["session"] = session_dates(out["timestamp"], calendar, close_hour=close_hour, tz=tz)
    out = out.dropna(subset=["session"])
    daily = (
        out.groupby(["ticker", "session"])
        .agg(sentiment=("polarity", "mean"), vader_sentiment=("vader", "mean"), articles=("polarity", "size"))
        .reset_index()
        .rename(columns={"session": "date"})
    )

    tickers = sorted(out["ticker"].unique())
    full = pd.MultiIndex.from_product([tickers, calendar], names=["ticker", "date"]).to_frame(index=False)
    full = full.merge(daily, on=["ticker", "date"], how="left")
    full["articles"] = full["articles"].fillna(0).astype(int)
    full["sentiment"] = full["sentiment"].fillna(0.0)
    full["vader_sentiment"] = full["vader_sentiment"].fillna(0.0)

    full = full.sort_values(["ticker", "date"])
    full["sentiment_mom3"] = (
        full.groupby("ticker")["sentiment"].transform(lambda s: s.rolling(3).mean()).diff(3)
    )
    vol = full.groupby("ticker")["articles"].transform(lambda s: s.rolling(21).mean())
    vol_sd = full.groupby("ticker")["articles"].transform(lambda s: s.rolling(21).std())
    full["news_spike"] = ((full["articles"] - vol) / vol_sd.replace(0, np.nan) > 2).fillna(False).astype(int)
    full["market_mood"] = full.groupby("date")["sentiment"].transform("mean")
    full[["sentiment_mom3", "market_mood"]] = full[["sentiment_mom3", "market_mood"]].fillna(0.0)
    return full[["date", "ticker", "sentiment", "sentiment_mom3", "news_spike", "market_mood", "vader_sentiment", "articles"]]


def build_labelling_sheet(scored: pd.DataFrame, cfg: Dict[str, Any], n: int = 200) -> Path:
    """Write a stratified sample for human labelling; labels are never inferred from prices."""
    reports = path_from_config(cfg, "reports")
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "sentiment_labelling.csv"
    if path.exists() and pd.read_csv(path)["human_label"].notna().any():
        LOG.info("keeping labelled sheet %s; delete it to resample", path)
        return path
    s = scored.dropna(subset=["finbert_label"]).copy()
    s["year"] = pd.to_datetime(s["timestamp"], utc=True).dt.year
    groups = []
    for _, g in s.groupby(["ticker", "year"], sort=True):
        groups.append(g.sample(n=min(len(g), 4), random_state=cfg["project"]["seed"]))
    sample = pd.concat(groups).drop_duplicates("sha1")
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=cfg["project"]["seed"])
    sheet = sample[["ticker", "timestamp", "title"]].copy()
    sheet.insert(0, "id", range(1, len(sheet) + 1))
    sheet["human_label"] = ""
    sheet["notes"] = ""
    models = sample[["finbert_label", "finbert_score", "vader"]].copy()
    models.insert(0, "id", sheet["id"].to_numpy())
    sheet.to_csv(path, index=False)
    models.to_csv(reports / "sentiment_labelling_models.csv", index=False)
    LOG.info("wrote %d-row blind labelling sheet -> %s", len(sheet), path)
    return path


def evaluate_labels(cfg: Dict[str, Any]) -> Dict[str, Any]:
    path = path_from_config(cfg, "reports") / "sentiment_labelling.csv"
    if not path.exists():
        return {"status": "missing labelling sheet"}
    df = pd.read_csv(path)
    if "finbert_label" not in df.columns:
        models_path = path_from_config(cfg, "reports") / "sentiment_labelling_models.csv"
        if not models_path.exists():
            return {"status": "missing model predictions file"}
        df = df.merge(pd.read_csv(models_path), on="id", how="left")
    labelled = df[df["human_label"].astype(str).str.lower().isin(["positive", "negative", "neutral"])].copy()
    if labelled.empty:
        return {"status": "pending", "rows": int(len(df))}
    y = labelled["human_label"].str.lower()
    pred_f = labelled["finbert_label"].str.lower()
    pred_v = np.where(labelled["vader"] > 0.05, "positive", np.where(labelled["vader"] < -0.05, "negative", "neutral"))
    flagged = int(labelled["notes"].notna().sum()) if "notes" in labelled else 0
    out = {"status": "ok", "rows": int(len(labelled)), "flagged_unsure": flagged, "finbert": {}, "vader": {}}
    # Headlines are resampled independently; the same draws pair the two scorers.
    rng = np.random.default_rng(cfg["project"]["seed"])
    draws = rng.integers(0, len(labelled), size=(5000, len(labelled)))
    boot = {}
    for name, pred in (("finbert", pred_f), ("vader", pd.Series(pred_v, index=labelled.index))):
        acc = float((y == pred).mean())
        cm = pd.crosstab(y, pred, dropna=False)
        from sklearn.metrics import f1_score

        macro = float(f1_score(y, pred, average="macro", labels=["positive", "negative", "neutral"], zero_division=0))
        boot[name] = (y == pred).to_numpy(dtype=float)[draws].mean(axis=1)
        ci = [float(q) for q in np.quantile(boot[name], [0.025, 0.975])]
        out[name] = {"accuracy": acc, "accuracy_ci": ci, "macro_f1": macro, "confusion": cm.to_dict()}
    out["finbert_minus_vader_ci"] = [float(q) for q in np.quantile(boot["finbert"] - boot["vader"], [0.025, 0.975])]
    save_json(out, path_from_config(cfg, "reports") / "sentiment_validation.json")
    return out


def run_sentiment(cfg: Optional[Dict[str, Any]] = None, force: bool = False) -> Dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    raw = path_from_config(cfg, "raw")
    processed = path_from_config(cfg, "processed")
    news = pd.read_parquet(raw / "news.parquet")
    calendar = pd.DatetimeIndex(pd.read_parquet(processed / "macro_clean.parquet")["date"]).sort_values()

    scored = score_news(news, cfg, force=force)
    scored["session"] = session_dates(
        scored["timestamp"], calendar, close_hour=cfg["news"]["market_close_hour"], tz=cfg["news"]["timezone"]
    )
    daily = aggregate_daily(scored, calendar, cfg["news"]["market_close_hour"], cfg["news"]["timezone"])
    daily.to_parquet(processed / "sentiment_daily.parquet", index=False)
    scored.to_parquet(raw / "news_scored.parquet", index=False)
    sheet_path = path_from_config(cfg, "reports") / "sentiment_labelling.csv"
    if sheet_path.exists():
        existing = pd.read_csv(sheet_path)
        done = existing["human_label"].astype(str).str.lower().isin(["positive", "negative", "neutral"]).sum() if "human_label" in existing else 0
    else:
        done = 0
    if done == 0:
        build_labelling_sheet(scored, cfg, int(cfg["sentiment"]["validation_sample"]))
    else:
        LOG.info("keeping the existing %d-row labelled validation sheet", done)
    validate = evaluate_labels(cfg)
    LOG.info("sentiment daily rows=%d; validation=%s", len(daily), validate.get("status"))
    return {"scored": scored, "daily": daily, "validation": validate}


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score FNSPID headlines and build daily sentiment")
    parser.add_argument("--force", action="store_true", help="re-score every unique headline")
    args = parser.parse_args(argv)
    run_sentiment(load_config(), force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
