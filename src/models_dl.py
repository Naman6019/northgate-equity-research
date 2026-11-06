"""Layer 3 - four PyTorch sequence models: LSTM, GRU, BiLSTM, Transformer.

Same chronological split and features as the ML models, Huber loss, early
stopping on validation, seeded and deterministic, with loss curves saved.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import baseline_metrics, evaluate_predictions, log_experiment
from src.features import SENTIMENT_FEATURES, TECH_FEATURES, fit_scalers, make_sequences
from src.utils import get_logger, load_config, path_from_config, save_json, set_seed

LOG = get_logger("northgate.models_dl")


class SeqModel(nn.Module):
    def __init__(
        self,
        kind: str,
        n_features: int,
        n_tickers: int,
        hidden: int = 64,
        dropout: float = 0.2,
        emb_dim: int = 8,
    ):
        super().__init__()
        self.kind = kind
        self.emb = nn.Embedding(n_tickers, emb_dim)
        d_in = n_features + emb_dim
        self.drop = nn.Dropout(dropout)
        if kind == "LSTM":
            self.rnn1 = nn.LSTM(d_in, hidden, batch_first=True)
            self.rnn2 = nn.LSTM(hidden, max(hidden // 2, 8), batch_first=True)
            last = max(hidden // 2, 8)
            self.head = nn.Sequential(nn.Linear(last, 16), nn.ReLU(), nn.Linear(16, 1))
        elif kind == "GRU":
            self.rnn1 = nn.GRU(d_in, hidden, batch_first=True)
            self.rnn2 = nn.GRU(hidden, max(hidden // 2, 8), batch_first=True)
            last = max(hidden // 2, 8)
            self.head = nn.Sequential(nn.Linear(last, 16), nn.ReLU(), nn.Linear(16, 1))
        elif kind == "BiLSTM":
            self.rnn1 = nn.LSTM(d_in, hidden, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Linear(hidden * 2, 16), nn.ReLU(), nn.Linear(16, 1))
        elif kind == "Transformer":
            self.proj = nn.Linear(d_in, hidden)
            self.pos = _PositionalEncoding(hidden, dropout)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=4, dim_feedforward=64, dropout=dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Linear(hidden, 1)
        else:
            raise ValueError(kind)

    def forward(self, x: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
        e = self.emb(tid).unsqueeze(1).repeat(1, x.size(1), 1)
        z = torch.cat([x, e], dim=-1)
        if self.kind in ("LSTM", "GRU"):
            z, _ = self.rnn1(z)
            z = self.drop(z)
            z, _ = self.rnn2(z)
            z = self.drop(z[:, -1, :])
            return self.head(z)
        if self.kind == "BiLSTM":
            z, _ = self.rnn1(z)
            z = self.drop(z[:, -1, :])
            return self.head(z)
        z = self.pos(self.proj(z))
        z = self.encoder(z)
        return self.head(z.mean(dim=1))


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


def _loaders(
    panel: pd.DataFrame,
    cols: List[str],
    length: int,
    horizon: int,
    scalers: Dict[str, Any],
) -> Dict[str, Tuple[DataLoader, np.ndarray, np.ndarray, pd.DataFrame]]:
    out = {}
    for split in ("train", "val", "test"):
        X, y, meta = make_sequences(panel, cols, length, horizon, split, scaler=scalers["minmax"])
        if len(X) == 0:
            out[split] = (None, np.array([]), np.array([]), meta)
            continue
        ts = scalers["target"]
        y_scaled = ts.transform(y.reshape(-1, 1)).ravel().astype(np.float32)
        ticker_ids = meta["ticker"].map({t: i for i, t in enumerate(sorted(panel["ticker"].unique()))}).to_numpy()
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(ticker_ids.astype(np.int64)), torch.from_numpy(y_scaled))
        loader = DataLoader(ds, batch_size=64, shuffle=(split == "train"))
        out[split] = (loader, y, y_scaled, meta)
    return out


def _run_epoch(model, loader, optimizer, criterion, device, train: bool) -> float:
    model.train(train)
    total, n = 0.0, 0
    for xb, tb, yb in loader:
        xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
        if train:
            optimizer.zero_grad()
        pred = model(xb, tb).squeeze(-1)
        loss = criterion(pred, yb)
        if train:
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * len(yb)
        n += len(yb)
    return total / max(n, 1)


def train_model(
    kind: str,
    panel: pd.DataFrame,
    cols: List[str],
    length: int,
    horizon: int,
    hidden: int,
    dropout: float,
    cfg: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scalers = fit_scalers(panel, cols, f"target_h{horizon}", cfg)
    loaders = _loaders(panel, cols, length, horizon, scalers)
    train_loader, _, _, _ = loaders["train"]
    val_loader, _, _, _ = loaders["val"]
    if train_loader is None or val_loader is None:
        raise RuntimeError("empty train/val sequences")

    model = SeqModel(kind, len(cols), panel["ticker"].nunique(), hidden=hidden, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["dl"]["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    criterion = nn.HuberLoss()
    max_epochs = int(cfg["dl"]["max_epochs"])
    patience = int(cfg["dl"]["patience"])
    best = float("inf")
    best_state = None
    wait = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        tr = _run_epoch(model, train_loader, optimizer, criterion, device, True)
        va = _run_epoch(model, val_loader, optimizer, criterion, device, False)
        scheduler.step(va)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        if va < best - 1e-7:
            best = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if epoch % 5 == 0 or epoch == 1:
            LOG.info("%s h=%d L=%d hidden=%d epoch %d: train %.5f val %.5f", kind, horizon, length, hidden, epoch, tr, va)
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    metrics_dir = path_from_config(cfg, "models") / "loss_curves"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(history)
    hist.to_csv(metrics_dir / f"{kind.lower()}_h{horizon}.csv", index=False)

    predictions = {}
    for split in ("val", "test"):
        loader, y_raw, _, meta = loaders[split]
        if loader is None:
            continue
        preds = []
        with torch.no_grad():
            for xb, tb, _ in loader:
                preds.append(model(xb.to(device), tb.to(device)).squeeze(-1).cpu().numpy())
        y_scaled = np.concatenate(preds) if preds else np.array([])
        y_pred = scalers["target"].inverse_transform(y_scaled.reshape(-1, 1)).ravel()
        m = meta.copy()
        m["y_true"] = y_raw
        m["y_pred"] = y_pred
        m["model"] = kind
        m["horizon"] = horizon
        m["split"] = split
        predictions[split] = m
    return {"model": model, "scalers": scalers, "history": hist, "best_val_loss": best, "predictions": predictions}


def run_dl(cfg: Optional[Dict[str, Any]] = None, horizons: Optional[List[int]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    horizons = horizons or cfg["horizons"]
    seed = int(cfg["project"]["seed"])
    panel = pd.read_parquet(path_from_config(cfg, "processed") / "panel.parquet")
    cols = TECH_FEATURES + SENTIMENT_FEATURES
    models_dir = path_from_config(cfg, "models")
    reports = path_from_config(cfg, "reports")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for horizon in horizons:
        # Re-read prices for price-space predictions.
        prices = pd.read_parquet(path_from_config(cfg, "processed") / "prices_clean.parquet")
        px = prices.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
        for kind in ("LSTM", "GRU", "BiLSTM", "Transformer"):
            trials = [
                {"length": 60, "hidden": int(cfg["dl"]["hidden_size"]), "dropout": 0.2},
                {"length": 20, "hidden": int(cfg["dl"]["hidden_size"]), "dropout": 0.2},
                {"length": 60, "hidden": max(int(cfg["dl"]["hidden_size"] * 0.5), 16), "dropout": 0.1},
            ]
            best = None
            for trial in trials:
                LOG.info("trial %s h=%d %s", kind, horizon, trial)
                res = train_model(kind, panel, cols, trial["length"], horizon, trial["hidden"], trial["dropout"], cfg, seed)
                if best is None or res["best_val_loss"] < best["best_val_loss"]:
                    best = {**res, **trial}
            model_path = models_dir / f"{kind.lower()}_h{horizon}.pt"
            torch.save(
                {
                    "state_dict": best["model"].state_dict(),
                    "kind": kind,
                    "length": best["length"],
                    "hidden": best["hidden"],
                    "dropout": best["dropout"],
                    "features": cols,
                    "n_tickers": panel["ticker"].nunique(),
                    "ticker_order": sorted(panel["ticker"].unique().tolist()),
                },
                model_path,
            )
            for split in ("val", "test"):
                if split in best["predictions"]:
                    p = best["predictions"][split].copy()
                    p["price_t"] = [float(px.loc[d, t]) for d, t in zip(p["date"], p["ticker"], strict=False)]
                    p["price_future"] = [pt * np.exp(y) for pt, y in zip(p["price_t"], p["y_true"], strict=False)]
                    p.to_parquet(models_dir / f"dl_pred_{kind.lower()}_h{horizon}_{split}.parquet", index=False)
                    met = evaluate_predictions(
                        p["price_t"].to_numpy(), p["price_future"].to_numpy(), p["y_true"].to_numpy(), p["y_pred"].to_numpy()
                    )
                    row = {"model": kind, "horizon": horizon, "split": split, **met, "best_val_loss": best["best_val_loss"]}
                    all_rows.append(row)
                    log_experiment(
                        reports / "experiments.csv", kind, horizon,
                        {"length": best["length"], "hidden": best["hidden"], "dropout": best["dropout"], "trials": len(trials)},
                        met, seed, f"split={split}; pytorch",
                    )
                    save_json(
                        {
                            "model": kind,
                            "horizon": horizon,
                            "length": best["length"],
                            "hidden": best["hidden"],
                            "dropout": best["dropout"],
                            "features": cols,
                            "seed": seed,
                            "best_val_loss": best["best_val_loss"],
                            "data_hash": hashlib.sha256((path_from_config(cfg, "processed") / "panel.parquet").read_bytes()).hexdigest(),
                            "artifact": f"models/{kind.lower()}_h{horizon}.pt",
                            "loss_curve": f"models/loss_curves/{kind.lower()}_h{horizon}.csv",
                        },
                        models_dir / "model_cards" / f"{kind.lower()}_h{horizon}.json",
                    )
    table = pd.DataFrame(all_rows)
    table.to_csv(reports / "dl_metrics.csv", index=False)
    return {"table": table, "panel": panel}


def build_comparison(cfg: Dict[str, Any], panel: pd.DataFrame) -> pd.DataFrame:
    """8-model + random-walk table on the test window (PRD 9.5)."""
    reports = path_from_config(cfg, "reports")
    rows = []
    for h in cfg["horizons"]:
        ml_path = reports / f"ml_metrics_h{h}.csv"
        if ml_path.exists():
            ml = pd.read_csv(ml_path)
            rows.append(ml[ml["split"] == "test"])
        dl_path = reports / "dl_metrics.csv"
        if dl_path.exists():
            dl = pd.read_csv(dl_path)
            rows.append(dl[(dl["split"] == "test") & (dl["horizon"] == h)])
        df = panel[panel["split"] == "test"].dropna(subset=[f"target_h{h}", f"target_price_h{h}"])
        b = baseline_metrics(df[f"target_h{h}"].to_numpy(), df["adj_close"].to_numpy(), df[f"target_price_h{h}"].to_numpy())
        rows.append(pd.DataFrame([{**b, "horizon": h, "split": "test"}]))
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not table.empty:
        table.to_csv(reports / "model_comparison.csv", index=False)
    return table


def pick_best_forecaster(cfg: Dict[str, Any], horizon: int = 5) -> Dict[str, Any]:
    """Choose the lowest validation MAE across ML + DL and write a combined prediction file."""
    reports = path_from_config(cfg, "reports")
    models_dir = path_from_config(cfg, "models")
    candidates = []
    ml_path = reports / f"ml_metrics_h{horizon}.csv"
    if ml_path.exists():
        ml = pd.read_csv(ml_path)
        val = ml[ml["split"] == "val"].sort_values("mae_return")
        if len(val):
            candidates.append((val.iloc[0]["model"], float(val.iloc[0]["mae_return"]), "ml"))
    dl_path = reports / "dl_metrics.csv"
    if dl_path.exists():
        dl = pd.read_csv(dl_path)
        val = dl[(dl["split"] == "val") & (dl["horizon"] == horizon)].sort_values("mae_return")
        if len(val):
            candidates.append((val.iloc[0]["model"], float(val.iloc[0]["mae_return"]), "dl"))
    if not candidates:
        return {}
    name, mae, family = min(candidates, key=lambda x: x[1])
    slug = name.replace(" ", "_").lower()
    if family == "ml":
        parts = [pd.read_parquet(models_dir / f"pred_{slug}_h{horizon}_{s}.parquet") for s in ("val", "test")]
    else:
        parts = [pd.read_parquet(models_dir / f"dl_pred_{slug}_h{horizon}_{s}.parquet") for s in ("val", "test")]
    combined = pd.concat(parts, ignore_index=True)
    out = models_dir / f"best_forecast_h{horizon}.parquet"
    combined.to_parquet(out, index=False)
    info = {"model": name, "family": family, "val_mae": mae, "horizon": horizon, "predictions_file": out.name}
    save_json(info, models_dir / "best_forecaster.json")
    LOG.info("best forecaster h=%d: %s (val MAE %.6f)", horizon, name, mae)
    return info


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train PyTorch sequence models")
    parser.add_argument("--horizon", nargs="+", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = load_config()
    horizons = args.horizon or cfg["horizons"]
    out = run_dl(cfg, horizons)
    table = build_comparison(cfg, out["panel"])
    for h in horizons:
        pick_best_forecaster(cfg, h)
    # The portfolio and recommendations consume the 5-day forecaster; keep the
    # canonical file pinned to h=5 regardless of argument order.
    pick_best_forecaster(cfg, 5)
    print(table.to_string(index=False) if len(table) else "no comparison rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
