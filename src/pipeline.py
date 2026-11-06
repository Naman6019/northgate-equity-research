"""One-command orchestrator (PRD Section 15.3).

    python -m src.pipeline data  [--offline]
    python -m src.pipeline train
    python -m src.pipeline all   [--offline]
"""

from __future__ import annotations

import argparse
import time
from typing import Iterable, Optional

from src.utils import ensure_dirs, get_logger, load_config

LOG = get_logger("northgate.pipeline")


def stage_data(cfg, offline: bool = False) -> None:
    from src.clean import run_cleaning
    from src.features import build_features
    from src.ingest import load_raw, run_ingestion
    from src.sentiment import run_sentiment

    if offline:
        LOG.info("offline: loading cached raw files")
        load_raw(cfg)
    else:
        LOG.info("ingesting prices, macro and news")
        run_ingestion(cfg)
    LOG.info("cleaning + data-quality report")
    run_cleaning(cfg)
    LOG.info("engineering features")
    build_features(cfg)
    LOG.info("scoring news sentiment (cached FinBERT)")
    try:
        run_sentiment(cfg)
        build_features(cfg)
    except FileNotFoundError as exc:
        LOG.warning("sentiment skipped: %s", exc)


def stage_train(cfg) -> None:
    import pandas as pd

    from src.models_dl import build_comparison, pick_best_forecaster
    from src.models_dl import main as dl_main
    from src.models_ml import main as ml_main
    from src.portfolio import run_portfolio
    from src.recommend import run_recommendations
    from src.utils import path_from_config

    horizons = cfg["horizons"]
    LOG.info("training classical ML models")
    ml_main(["--horizon", *[str(h) for h in horizons], "--walk-forward"])
    LOG.info("training PyTorch sequence models")
    dl_main(["--horizon", *[str(h) for h in horizons]])
    panel = pd.read_parquet(path_from_config(cfg, "processed") / "panel.parquet")
    build_comparison(cfg, panel)
    for h in horizons:
        pick_best_forecaster(cfg, h)
    pick_best_forecaster(cfg, 5)
    LOG.info("portfolio optimisation + backtest")
    run_portfolio(cfg)
    LOG.info("recommendations + ranking diagnostic")
    run_recommendations(cfg)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Northgate pipeline")
    parser.add_argument("stage", choices=["data", "train", "all"])
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config()
    ensure_dirs(cfg)
    t0 = time.time()
    if args.stage in ("data", "all"):
        stage_data(cfg, offline=args.offline)
    if args.stage in ("train", "all"):
        stage_train(cfg)
    LOG.info("pipeline %s finished in %.1f min", args.stage, (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
