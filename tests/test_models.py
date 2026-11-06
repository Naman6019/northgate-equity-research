"""Model plumbing: chronological folds, ticker dummies and network shapes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.features import SENTIMENT_FEATURES, TECH_FEATURES
from src.models_dl import SeqModel
from src.models_ml import add_ticker_dummies, date_splits, feature_list


def test_feature_lists():
    assert feature_list(True) == TECH_FEATURES + SENTIMENT_FEATURES
    assert "sentiment" in feature_list(True)
    assert "sentiment" not in feature_list(False)


def test_ticker_dummies(cfg):
    panel = pd.DataFrame({"ticker": cfg["universe"]["equities"] * 2})
    out = add_ticker_dummies(panel, cfg["universe"]["equities"])
    for t in cfg["universe"]["equities"]:
        assert f"t_{t}" in out.columns
    assert (out.filter(like="t_").sum(axis=1) == 1).all()


def test_date_splits_are_chronological_and_gapped():
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=100).to_numpy(), 2)
    for train, test in date_splits(dates, n_splits=3, gap=5):
        assert train.max() < test.min()
        # At least `gap` sessions separate the windows.
        unique = np.sort(np.unique(dates))
        assert np.searchsorted(unique, test.min()) - np.searchsorted(unique, train.max()) >= 5


def test_sequence_models_forward_shapes():
    for kind in ("LSTM", "GRU", "BiLSTM", "Transformer"):
        model = SeqModel(kind, n_features=7, n_tickers=3, hidden=16, dropout=0.1)
        x = torch.randn(4, 12, 7)
        tid = torch.tensor([0, 1, 2, 0])
        y = model(x, tid)
        assert y.shape == (4, 1)
