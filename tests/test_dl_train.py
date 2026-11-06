"""One tiny end-to-end neural training pass (covers loaders, loop and prediction)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import SENTIMENT_FEATURES, TECH_FEATURES
from src.models_dl import train_model


def _tiny_panel(cfg):
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2021-01-01", periods=36)
    rows = []
    for ticker in ("AAPL", "MSFT"):
        for i, d in enumerate(dates):
            row = {"date": d, "ticker": ticker, "adj_close": 100 + i, "split": "train" if i < 22 else ("val" if i < 29 else "test")}
            for col in TECH_FEATURES + SENTIMENT_FEATURES:
                row[col] = float(rng.normal())
            row["target_h5"] = float(rng.normal(0, 0.01))
            row["target_price_h5"] = 100 + i + 5
            row["target_date_h5"] = d + pd.Timedelta(days=5)
            rows.append(row)
    return pd.DataFrame(rows)


def test_train_model_one_epoch(cfg, monkeypatch):
    tmp = Path("tests/_tmp_dl")
    tmp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("src.models_dl.path_from_config", lambda cfg, key: tmp / key)
    try:
        cfg = dict(cfg)
        cfg["dl"] = dict(cfg["dl"])
        cfg["dl"]["max_epochs"] = 1
        cfg["dl"]["patience"] = 1
        panel = _tiny_panel(cfg)
        cols = TECH_FEATURES + SENTIMENT_FEATURES
        out = train_model("LSTM", panel, cols, length=4, horizon=5, hidden=8, dropout=0.1, cfg=cfg, seed=42)
        assert out["best_val_loss"] >= 0
        assert "val" in out["predictions"] and len(out["predictions"]["val"]) > 0
        assert "test" in out["predictions"] and len(out["predictions"]["test"]) > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
