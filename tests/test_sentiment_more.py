"""Sentiment scoring cache, labelling sheet and validation path (FinBERT mocked)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.sentiment import build_labelling_sheet, evaluate_labels, score_news


def _cfg_with_paths(cfg):
    base = Path("tests/_tmp_sent")
    cfg = dict(cfg)
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["raw"] = "tests/_tmp_sent/raw"
    cfg["paths"]["reports"] = "tests/_tmp_sent/reports"
    cfg["paths"]["processed"] = "tests/_tmp_sent/processed"
    for sub in ("raw", "reports", "processed", "raw/news_cache"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return cfg, base


def test_score_news_cache_and_validation(monkeypatch, cfg):
    cfg2, base = _cfg_with_paths(cfg)
    try:
        def fake_finbert(texts, model_name, batch_size=64):
            labels = ["positive" if "beat" in t else "negative" for t in texts]
            return pd.DataFrame({"finbert_label": labels, "finbert_score": [0.9] * len(texts)})

        monkeypatch.setattr("src.sentiment._score_finbert", fake_finbert)
        news = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2020-01-10 21:30:00", "2020-01-10 15:00:00"], utc=True),
                "ticker": ["AAPL", "AAPL"],
                "text": ["Company beats estimates", "Company warns"],
                "title": ["Company beats estimates", "Company warns"],
                "url": ["", ""],
                "publisher": ["", ""],
            }
        )
        scored = score_news(news, cfg2)
        assert len(scored) == 2
        assert (scored["finbert_label"] == ["positive", "negative"]).all()
        assert (Path(base) / "raw" / "news_cache" / "finbert_scores.parquet").exists()
        # Second pass must hit the cache (no new rows requested).
        again = score_news(news, cfg2)
        assert len(again) == 2

        sheet_path = build_labelling_sheet(scored, cfg2, n=2)
        sheet = pd.read_csv(sheet_path)
        models = pd.read_csv(sheet_path.parent / "sentiment_labelling_models.csv")
        sheet = sheet.merge(models, on="id", how="left")
        assert "human_label" in sheet.columns
        sheet["human_label"] = sheet["finbert_label"]
        sheet.to_csv(sheet_path, index=False)
        # A pipeline rerun must not wipe finished labels.
        build_labelling_sheet(scored, cfg2, n=2)
        assert pd.read_csv(sheet_path)["human_label"].notna().all()
        result = evaluate_labels(cfg2)
        assert result["status"] == "ok"
        assert 0 <= result["finbert"]["accuracy"] <= 1
    finally:
        shutil.rmtree(base, ignore_errors=True)
