"""Layer 1 - deterministic ingestion of prices, macro series and news.

Public sources only:
  * Yahoo Finance via yfinance (OHLCV + adjusted close)
  * FRED CSV endpoint (DGS10, DGS3MO, CPIAUCSL, UNRATE)
  * FNSPID ticker-tagged headlines (Hugging Face)

Every download is saved as Parquet with a metadata sidecar so `--offline`
rebuilds reproduce the same input without touching the network.
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests

from src.utils import (
    ensure_dirs,
    get_logger,
    load_config,
    now_utc,
    path_from_config,
    save_json,
)

LOG = get_logger("northgate.ingest")

PRICE_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
FRED_SERIES = ["DGS10", "DGS3MO", "CPIAUCSL", "UNRATE"]


def _retry(fn, attempts: int = 3, base_sleep: float = 2.0):
    last: Optional[Exception] = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - network retries
            last = exc
            sleep_for = base_sleep * (2**i)
            LOG.warning("attempt %d/%d failed: %s; retrying in %.1fs", i + 1, attempts, exc, sleep_for)
            time.sleep(sleep_for)
    assert last is not None
    raise last


def download_prices(cfg: Dict[str, Any], force: bool = False) -> pd.DataFrame:
    """Download OHLCV for the universe and context tickers into long format."""
    raw = path_from_config(cfg, "raw")
    out = raw / "prices.parquet"
    meta_path = raw / "prices_meta.json"
    if out.exists() and not force:
        LOG.info("prices cache found (%s rows); use --force to refresh", len(pd.read_parquet(out)))
        return pd.read_parquet(out)

    import yfinance as yf

    u = cfg["universe"]
    tickers = (
        list(u["equities"])
        + [u["benchmark"], u["benchmark_total_return"], u["volatility"]]
        + list(u["context_etfs"])
    )
    start, end = cfg["dates"]["start"], cfg["dates"]["end"]
    end_inclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
    frames: List[pd.DataFrame] = []

    def _fetch() -> pd.DataFrame:
        return yf.download(
            tickers,
            start=start,
            end=end_inclusive,
            auto_adjust=False,
            actions=False,
            group_by="ticker",
            progress=False,
            threads=True,
        )

    wide = _retry(_fetch)
    if wide.empty:
        raise RuntimeError("yfinance returned no rows")

    for ticker in tickers:
        if isinstance(wide.columns, pd.MultiIndex):
            if ticker not in wide.columns.get_level_values(0):
                LOG.warning("ticker %s missing from download", ticker)
                continue
            sub = wide[ticker].copy()
        else:
            sub = wide.copy()
        sub = sub.reset_index()
        date_col = "Date" if "Date" in sub.columns else sub.columns[0]
        sub = sub.rename(columns={date_col: "date"})
        missing = [c for c in PRICE_COLS if c not in sub.columns]
        if missing:
            LOG.warning("ticker %s missing columns %s", ticker, missing)
            continue
        sub = sub[["date"] + PRICE_COLS].copy()
        sub["ticker"] = ticker
        sub = sub.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        frames.append(sub)

    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).dt.normalize()
    prices = prices.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    prices.to_parquet(out, index=False)
    save_json(
        {
            "downloaded_at": now_utc(),
            "source": "yfinance",
            "tickers": tickers,
            "start": start,
            "end": end,
            "rows": int(len(prices)),
            "file": str(out.relative_to(path_from_config(cfg, "raw").parent.parent)),
        },
        meta_path,
    )
    LOG.info("prices: %d rows -> %s", len(prices), out)
    return prices


def download_macro(cfg: Dict[str, Any], force: bool = False) -> pd.DataFrame:
    """Download FRED series through the public CSV endpoint (no API key)."""
    raw = path_from_config(cfg, "raw")
    out = raw / "macro.parquet"
    meta_path = raw / "macro_meta.json"
    if out.exists() and not force:
        LOG.info("macro cache found; use --force to refresh")
        return pd.read_parquet(out)

    start, end = cfg["dates"]["start"], cfg["dates"]["end"]
    frames = []
    for series in FRED_SERIES:
        url = (
            "https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id={series}&cosd={start}&coed={end}"
        )

        def _fetch(u: str = url) -> str:
            resp = requests.get(u, timeout=90)
            resp.raise_for_status()
            return resp.text

        text = _retry(_fetch)
        df = pd.read_csv(io.StringIO(text))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["series"] = series
        df = df.dropna(subset=["date"])
        frames.append(df)
        LOG.info("fred %s: %d rows", series, len(df))

    macro = pd.concat(frames, ignore_index=True).sort_values(["series", "date"])
    macro.to_parquet(out, index=False)
    save_json(
        {
            "downloaded_at": now_utc(),
            "source": "FRED fredgraph.csv",
            "series": FRED_SERIES,
            "start": start,
            "end": end,
            "rows": int(len(macro)),
        },
        meta_path,
    )
    LOG.info("macro: %d rows -> %s", len(macro), out)
    return macro


FNSPID_MIRROR_PARTS = [
    "https://huggingface.co/datasets/2084Collective/FNSPID_IMPROVED/resolve/main/"
    f"data/train-0000{i}-of-00006.parquet"
    for i in range(6)
]


def _download_file(url: str, dest: Path, force: bool = False) -> Path:
    """Download a large file to `dest` atomically, with retries."""
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def _get() -> Path:
        with requests.get(url, stream=True, timeout=900) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        fh.write(chunk)
        return tmp

    _retry(_get, attempts=3, base_sleep=5.0)
    tmp.replace(dest)
    return dest


def download_news(cfg: Dict[str, Any], force: bool = False) -> pd.DataFrame:
    """Download the FNSPID parquet mirror and keep the universe/window rows.

    The original `All_external.csv` stops in mid-2020; the parquet mirror
    (`2084Collective/FNSPID_IMPROVED`, six parts, ~1.25 GB) carries the same
    FNSPID headlines through 2023-12. Parts are cached locally so `--offline`
    rebuilds never touch the network.
    """
    raw = path_from_config(cfg, "raw")
    out = raw / "news.parquet"
    meta_path = raw / "news_meta.json"
    if out.exists() and not force:
        LOG.info("news cache found (%d rows); use --force to refresh", len(pd.read_parquet(out)))
        return pd.read_parquet(out)

    tickers = set(cfg["universe"]["equities"])
    start = pd.Timestamp(cfg["dates"]["start"], tz="UTC")
    end = pd.Timestamp(cfg["dates"]["end"], tz="UTC") + pd.Timedelta(days=1)
    cache = raw / "fnspid_improved"
    frames: List[pd.DataFrame] = []
    source_rows = 0

    for i, url in enumerate(FNSPID_MIRROR_PARTS):
        dest = cache / f"part{i}.parquet"
        _download_file(url, dest, force=force)
        df = pd.read_parquet(dest, columns=["date", "stock", "title"])
        n_part = len(df)
        source_rows += n_part
        df["stock"] = df["stock"].astype(str).str.strip().str.upper()
        df["timestamp"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df = df[
            df["stock"].isin(tickers)
            & df["timestamp"].notna()
            & (df["timestamp"] >= start)
            & (df["timestamp"] < end)
        ]
        if not df.empty:
            frames.append(df[["timestamp", "stock", "title"]])
        LOG.info("news part %d: %d rows, %d universe rows", i, n_part, len(df))

    if not frames:
        raise RuntimeError("no FNSPID rows matched the universe/window")

    news = pd.concat(frames, ignore_index=True)
    news["title"] = news["title"].fillna("").astype(str).str.strip()
    news = news[news["title"].str.len() > 0]
    news["text"] = news["title"]
    news = news.drop_duplicates(["ticker", "timestamp", "title"]) if "ticker" in news else news.drop_duplicates(
        ["stock", "timestamp", "title"]
    )
    news = news.rename(columns={"stock": "ticker"})
    news["url"] = ""
    news["publisher"] = ""
    news = news.sort_values(["timestamp", "ticker"]).reset_index(drop=True)
    news = news[["timestamp", "ticker", "title", "text", "url", "publisher"]]
    news.to_parquet(out, index=False)

    coverage = (
        news.assign(year=news["timestamp"].dt.year)
        .groupby(["ticker", "year"])
        .agg(articles=("title", "size"), active_days=("timestamp", lambda s: s.dt.date.nunique()))
        .reset_index()
    )
    coverage_path = path_from_config(cfg, "reports") / "news_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    save_json(
        {
            "downloaded_at": now_utc(),
            "source": "FNSPID (parquet mirror 2084Collective/FNSPID_IMPROVED)",
            "original_dataset": "Zihan1004/FNSPID",
            "universe": sorted(tickers),
            "window": [str(start), str(end)],
            "source_rows_scanned": int(source_rows),
            "rows_kept": int(len(news)),
            "coverage_file": str(coverage_path),
        },
        meta_path,
    )
    LOG.info("news: %d kept of %d scanned -> %s", len(news), source_rows, out)
    return news


def load_raw(cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    raw = path_from_config(cfg, "raw")
    out = {}
    for name in ("prices", "macro", "news"):
        path = raw / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run ingestion first (or --offline with cached files)")
        out[name] = pd.read_parquet(path)
    return out


def run_ingestion(cfg: Optional[Dict[str, Any]] = None, force: bool = False) -> Dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    ensure_dirs(cfg)
    return {
        "prices": download_prices(cfg, force=force),
        "macro": download_macro(cfg, force=force),
        "news": download_news(cfg, force=force),
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Northgate ingestion")
    parser.add_argument("--force", action="store_true", help="ignore cached Parquet and re-download")
    parser.add_argument("--offline", action="store_true", help="use cached raw files only")
    args = parser.parse_args(argv)
    cfg = load_config()
    ensure_dirs(cfg)
    if args.offline:
        data = load_raw(cfg)
        LOG.info("offline: %s", {k: len(v) for k, v in data.items()})
    else:
        data = run_ingestion(cfg, force=args.force)
    print({k: v.shape for k, v in data.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
