"""Evaluation harness: price-space and return-space metrics, IC, direction
baselines and block-bootstrap confidence intervals (PRD Section 8.3)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils import get_logger

LOG = get_logger("northgate.evaluate")


def price_metrics(y_true_price: np.ndarray, y_pred_price: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true_price, dtype=float)
    yp = np.asarray(y_pred_price, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) == 0:
        return {}
    mape = np.mean(np.abs((yt - yp) / np.where(yt == 0, np.nan, yt))) * 100
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "mape": float(mape),
        "r2": float(r2_score(yt, yp)),
        "n": int(len(yt)),
    }


def return_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) == 0:
        return {}
    ic = stats.spearmanr(yt, yp).statistic if len(yt) > 2 else np.nan
    return {
        "rmse_return": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae_return": float(mean_absolute_error(yt, yp)),
        "ic": float(ic),
        "direction_accuracy": direction_accuracy(yt, yp),
        "n": int(len(yt)),
    }


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp) & (yt != 0) & (yp != 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(yt[mask]) == np.sign(yp[mask])))


def baseline_metrics(y_true: np.ndarray, price_t: np.ndarray, price_future: np.ndarray) -> Dict[str, float]:
    """Random-walk baseline plus the two direction bars from the plan."""
    yt = np.asarray(y_true, dtype=float)
    mask = np.isfinite(yt)
    yt = yt[mask]
    pt = np.asarray(price_t, dtype=float)[mask]
    pf = np.asarray(price_future, dtype=float)[mask]
    return {
        "model": "Random walk",
        "mae_return": float(np.mean(np.abs(yt))),
        "rmse_return": float(np.sqrt(np.mean(yt**2))),
        "mae": float(np.mean(np.abs(pf - pt))),
        "rmse": float(np.sqrt(np.mean((pf - pt) ** 2))),
        "always_up_accuracy": float(np.mean(yt > 0)),
        "coin_flip_accuracy": 0.5,
        "n": int(len(yt)),
    }


def evaluate_predictions(
    price_t: np.ndarray,
    price_future: np.ndarray,
    y_true_return: np.ndarray,
    y_pred_return: np.ndarray,
) -> Dict[str, float]:
    """Convert predicted log returns to price space and report both metric sets."""
    pt = np.asarray(price_t, dtype=float)
    yp = np.asarray(y_pred_return, dtype=float)
    price_pred = pt * np.exp(yp)
    out = price_metrics(price_future, price_pred)
    out.update(return_metrics(y_true_return, y_pred_return))
    return out


def _date_block_bootstrap(
    dates: np.ndarray,
    n_resamples: int,
    block: int,
    seed: int,
    alpha: float = 0.05,
) -> Iterable[np.ndarray]:
    unique = np.sort(pd.unique(dates))
    n = len(unique)
    if n == 0:
        return
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    for _ in range(n_resamples):
        starts = rng.integers(0, n, size=n_blocks)
        chosen = np.concatenate([np.take(unique, np.arange(s, s + block), mode="wrap") for s in starts])[:n]
        yield chosen


def bootstrap_metric_ci(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "mae_return",
    block: int = 20,
    n_resamples: int = 5000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Block-bootstrap CI for a metric, resampling whole dates to keep cross-sections."""
    dates = np.asarray(dates)
    frame = pd.DataFrame({"date": dates, "y": y_true, "p": y_pred})
    frame = frame[np.isfinite(frame["y"]) & np.isfinite(frame["p"])]
    if frame.empty:
        return (float("nan"), float("nan"))
    unique_dates = np.sort(frame["date"].unique())
    index = {d: g for d, g in frame.groupby("date")}  # noqa: C416 - explicit pairs
    rng = np.random.default_rng(seed)
    n = len(unique_dates)
    n_blocks = int(np.ceil(n / block))
    sims = []
    for _ in range(n_resamples):
        starts = rng.integers(0, n, size=n_blocks)
        chosen = np.concatenate([np.take(unique_dates, np.arange(s, s + block), mode="wrap") for s in starts])[:n]
        sample = pd.concat([index[d] for d in chosen], ignore_index=True)
        if metric == "mae_return":
            value = np.mean(np.abs(sample["y"] - sample["p"]))
        elif metric == "ic":
            value = stats.spearmanr(sample["y"], sample["p"]).statistic
        elif metric == "direction":
            value = direction_accuracy(sample["y"].to_numpy(), sample["p"].to_numpy())
        else:
            raise ValueError(metric)
        sims.append(value)
    lo, hi = np.nanpercentile(sims, [100 * 0.025, 100 * 0.975])
    return float(lo), float(hi)


def log_experiment(
    path: str | Path,
    model: str,
    horizon: int,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    seed: int,
    note: str = "",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "model": model,
                "horizon": horizon,
                "params": json.dumps(params, sort_keys=True, default=str),
                "metrics": json.dumps(metrics, sort_keys=True, default=str),
                "seed": seed,
                "note": note,
            }
        ]
    )
    header = not path.exists()
    row.to_csv(path, mode="a", header=header, index=False)


def cross_sectional_ic(
    panel: pd.DataFrame,
    pred_col: str,
    target_col: str,
    date_col: str = "date",
) -> pd.DataFrame:
    """Daily cross-sectional Spearman IC (used by the ranking objective, N3)."""
    rows = []
    for date, g in panel.groupby(date_col):
        g = g[[pred_col, target_col]].dropna()
        if len(g) >= 3 and g[pred_col].nunique() > 1:
            ic = stats.spearmanr(g[pred_col], g[target_col]).statistic
            rows.append({"date": date, "ic": ic, "n": len(g)})
    return pd.DataFrame(rows)


def ic_summary(ic: pd.DataFrame) -> Dict[str, float]:
    if ic.empty:
        return {}
    x = ic["ic"].dropna().to_numpy()
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
    tstat = mean / (std / np.sqrt(len(x))) if std and len(x) > 1 else float("nan")
    return {
        "mean_ic": mean,
        "ic_std": std,
        "ic_tstat": float(tstat),
        "ic_ir": float(mean / std) if std else float("nan"),
        "days": int(len(x)),
    }
