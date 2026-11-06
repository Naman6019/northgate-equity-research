"""Layer 3 - four classical ML regressors with chronological validation.

Tuning uses expanding-window TimeSeriesSplit(gap=h) on the training dates only.
Model selection uses the validation snapshot; the test snapshot is scored once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.evaluate import evaluate_predictions, log_experiment
from src.features import SENTIMENT_FEATURES, TECH_FEATURES, build_features
from src.utils import get_logger, load_config, path_from_config, save_json, set_seed

LOG = get_logger("northgate.models_ml")


def data_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_ticker_dummies(panel: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    dummies = pd.get_dummies(panel["ticker"], prefix="t", dtype=float)
    for t in tickers:
        col = f"t_{t}"
        if col not in dummies:
            dummies[col] = 0.0
    return pd.concat([panel, dummies[[f"t_{t}" for t in tickers]]], axis=1)


def feature_list(include_sentiment: bool = True) -> List[str]:
    cols = list(TECH_FEATURES)
    if include_sentiment:
        cols += list(SENTIMENT_FEATURES)
    return cols


def date_splits(dates: np.ndarray, n_splits: int, gap: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    unique = np.sort(pd.unique(dates))
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    for tr, te in tscv.split(unique):
        yield unique[tr], unique[te]


def _subset(X: np.ndarray, y: np.ndarray, dates: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return X[mask], y[mask]


def _cv_mae(
    factory,
    params: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    n_splits: int,
    gap: int,
    scale: bool,
    max_train_rows: Optional[int] = None,
    seed: int = 42,
) -> float:
    scores = []
    rng = np.random.default_rng(seed)
    for train_dates, test_dates in date_splits(dates, n_splits, gap):
        tr = np.isin(dates, train_dates)
        te = np.isin(dates, test_dates)
        if tr.sum() < 50 or te.sum() < 10:
            continue
        idx = np.where(tr)[0]
        if max_train_rows and len(idx) > max_train_rows:
            idx = rng.choice(idx, size=max_train_rows, replace=False)
        Xtr, ytr = X[idx], y[idx]
        Xte, yte = X[te], y[te]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        model = factory(**params)
        model.fit(Xtr, ytr)
        scores.append(mean_absolute_error(yte, model.predict(Xte)))
    return float(np.mean(scores)) if scores else float("inf")


def _random_params(grid: Dict[str, List[Any]], n_trials: int, seed: int) -> List[Dict[str, Any]]:
    sampler = ParameterSampler(grid, n_iter=n_trials, random_state=seed)
    return list(sampler)


def tune_and_fit(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    split: np.ndarray,
    horizon: int,
    cfg: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """Tune each family on train CV, fit on train, and return fitted models + params."""
    gap = horizon
    n_splits = int(cfg["ml"]["cv_splits"])
    train = split == "train"
    Xtr, ytr, dtr = X[train], y[train], dates[train]
    results: Dict[str, Any] = {}

    # Ridge: exhaustive over a small alpha grid.
    best = (float("inf"), None)
    for alpha in cfg["ml"]["ridge_alphas"]:
        score = _cv_mae(lambda **p: Ridge(**p), {"alpha": alpha}, Xtr, ytr, dtr, n_splits, gap, True)
        if score < best[0]:
            best = (score, {"alpha": alpha})
    results["Ridge"] = {"factory": Ridge, "params": best[1], "cv_mae": best[0], "scale": True}

    # Random forest: random search.
    best = (float("inf"), None)
    for params in _random_params(cfg["ml"]["rf_grid"], min(int(cfg["ml"]["n_trials"]), 8), seed):
        p = {k: (None if v == "null" else v) for k, v in params.items()}
        p["random_state"] = seed
        p["n_jobs"] = -1
        score = _cv_mae(
            lambda **kw: RandomForestRegressor(**kw),
            p,
            Xtr,
            ytr,
            dtr,
            n_splits,
            gap,
            False,
        )
        if score < best[0]:
            best = (score, p)
    results["Random Forest"] = {"factory": RandomForestRegressor, "params": best[1], "cv_mae": best[0], "scale": False}

    # XGBoost: random search.
    best = (float("inf"), None)
    for params in _random_params(cfg["ml"]["xgb_grid"], min(int(cfg["ml"]["n_trials"]), 10), seed + 1):
        p = dict(params)
        p.update({"random_state": seed, "n_jobs": -1, "tree_method": "hist", "verbosity": 0})
        score = _cv_mae(lambda **kw: XGBRegressor(**kw), p, Xtr, ytr, dtr, n_splits, gap, False)
        if score < best[0]:
            best = (score, p)
    results["XGBoost"] = {"factory": XGBRegressor, "params": best[1], "cv_mae": best[0], "scale": False}

    # SVR: small search on a subsample (kernel SVR is cubic-ish in rows).
    best = (float("inf"), None)
    for params in _random_params(cfg["ml"]["svr_grid"], 4, seed + 2):
        p = dict(params)
        score = _cv_mae(
            lambda **kw: SVR(**kw, cache_size=1000),
            p,
            Xtr,
            ytr,
            dtr,
            min(n_splits, 3),
            gap,
            True,
            max_train_rows=6000,
        )
        if score < best[0]:
            best = (score, p)
    results["SVR"] = {"factory": SVR, "params": {**best[1], "cache_size": 1000}, "cv_mae": best[0], "scale": True}

    for name, spec in results.items():
        model = spec["factory"](**spec["params"])
        X_fit = StandardScaler().fit_transform(Xtr) if spec["scale"] else Xtr
        if spec["scale"]:
            spec["scaler"] = StandardScaler().fit(Xtr)
            X_fit = spec["scaler"].transform(Xtr)
        model.fit(X_fit, ytr)
        spec["model"] = model
        LOG.info("%s h=%d: CV MAE %.6f | params %s", name, horizon, spec["cv_mae"], spec["params"])
    return results


def predict_with(spec: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    if spec.get("scale"):
        X = spec["scaler"].transform(X)
    return spec["model"].predict(X)


def run_horizon(
    panel: pd.DataFrame,
    horizon: int,
    cfg: Dict[str, Any],
    include_sentiment: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    processed = path_from_config(cfg, "processed")
    models_dir = path_from_config(cfg, "models")
    reports = path_from_config(cfg, "reports")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    panel = add_ticker_dummies(panel, cfg["universe"]["equities"])
    cols = feature_list(include_sentiment) + [f"t_{t}" for t in cfg["universe"]["equities"]]
    target = f"target_h{horizon}"
    df = panel.dropna(subset=cols + [target]).reset_index(drop=True)
    X = df[cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    dates = df["date"].to_numpy()
    split = df["split"].to_numpy()

    spec_by_model = tune_and_fit(X, y, dates, split, horizon, cfg, seed)
    out_rows: List[Dict[str, Any]] = []
    predictions = {}

    for name, spec in spec_by_model.items():
        for split_name in ("val", "test"):
            mask = split == split_name
            pred = predict_with(spec, X[mask])
            pred_df = df.loc[mask, ["date", "ticker", "adj_close", f"target_price_h{horizon}", target]].copy()
            pred_df = pred_df.rename(columns={target: "y_true", f"target_price_h{horizon}": "price_future"})
            pred_df["price_t"] = df.loc[mask, "adj_close"].to_numpy()
            pred_df["y_pred"] = pred
            pred_df["model"] = name
            pred_df["horizon"] = horizon
            pred_df["split"] = split_name
            pred_df.to_parquet(models_dir / f"pred_{name.replace(' ', '_').lower()}_h{horizon}_{_short(split_name)}.parquet", index=False)
            predictions[(name, split_name)] = pred_df

            metrics = evaluate_predictions(
                pred_df["price_t"].to_numpy(),
                pred_df["price_future"].to_numpy(),
                pred_df["y_true"].to_numpy(),
                pred_df["y_pred"].to_numpy(),
            )
            row = {"model": name, "horizon": horizon, "split": split_name, **metrics, "cv_mae": spec["cv_mae"]}
            out_rows.append(row)
            log_experiment(
                reports / "experiments.csv",
                model=name,
                horizon=horizon,
                params=spec["params"],
                metrics=metrics,
                seed=seed,
                note=f"split={split_name}; sentiment={include_sentiment}",
            )

        # Persist the fitted model and its card.
        joblib.dump(spec["model"], models_dir / f"{_slug(name)}_h{horizon}.joblib")

        def _bounds(mask: np.ndarray) -> List[str]:
            s = pd.to_datetime(df.loc[mask, "date"])
            return [str(s.min().date()), str(s.max().date())] if len(s) else []

        card = {
            "model": name,
            "horizon": horizon,
            "params": spec["params"],
            "cv_mae": spec["cv_mae"],
            "scale": spec["scale"],
            "features": cols,
            "target": target,
            "train": _bounds(split == "train"),
            "val": _bounds(split == "val"),
            "test": _bounds(split == "test"),
            "seed": seed,
            "data_hash": data_hash(processed / "panel.parquet"),
            "artifact": f"models/{_slug(name)}_h{horizon}.joblib",
        }
        save_json(card, models_dir / "model_cards" / f"{_slug(name)}_h{horizon}.json")
        if spec.get("scale"):
            joblib.dump(spec["scaler"], models_dir / f"scaler_{_slug(name)}_h{horizon}.joblib")

    table = pd.DataFrame(out_rows).sort_values(["split", "mae_return"])
    table.to_csv(reports / f"ml_metrics_h{horizon}.csv", index=False)
    val = table[table["split"] == "val"].sort_values("mae_return")
    best = val.iloc[0].to_dict() if len(val) else {}
    save_json(
        {"horizon": horizon, "best_by_val_mae": best, "table": table.to_dict(orient="records")},
        reports / f"ml_best_h{horizon}.json",
    )
    return {"table": table, "predictions": predictions, "specs": spec_by_model, "best": best}


def walk_forward(
    panel: pd.DataFrame,
    horizon: int,
    model_name: str,
    cfg: Dict[str, Any],
    seed: int = 42,
    refit_every: int = 63,
) -> pd.DataFrame:
    """Expanding-window refit of the winning ML model over validation + test (N4)."""

    panel = add_ticker_dummies(panel, cfg["universe"]["equities"])
    cols = feature_list(True) + [f"t_{t}" for t in cfg["universe"]["equities"]]
    target = f"target_h{horizon}"
    df = panel.dropna(subset=cols + [target]).reset_index(drop=True)
    models_dir = path_from_config(cfg, "models")
    reports = path_from_config(cfg, "reports")
    spec_path = models_dir / "model_cards" / f"{_slug(model_name)}_h{horizon}.json"
    card = json.loads(spec_path.read_text(encoding="utf-8"))
    params = card["params"]
    scale = card["scale"]

    unique = np.sort(df["date"].unique())
    eval_dates = unique[df.groupby("date")["split"].first().reindex(unique).isin(["val", "test"]).to_numpy()]
    rows = []
    for start in range(0, len(eval_dates), refit_every):
        block = eval_dates[start : start + refit_every]
        idx = int(np.searchsorted(unique, block[0]))
        train_dates = unique[: max(0, idx - horizon)]
        if len(train_dates) < 300:
            continue
        tr = df["date"].isin(train_dates).to_numpy()
        te = df["date"].isin(block).to_numpy()
        if te.sum() == 0:
            continue
        Xtr, ytr = df.loc[tr, cols].to_numpy(float), df.loc[tr, target].to_numpy(float)
        Xte = df.loc[te, cols].to_numpy(float)
        if model_name == "Ridge":
            model = Ridge(**params)
        elif model_name == "Random Forest":
            model = RandomForestRegressor(**{**params, "random_state": seed, "n_jobs": -1})
        elif model_name == "XGBoost":
            model = XGBRegressor(**{**params, "random_state": seed, "n_jobs": -1, "tree_method": "hist", "verbosity": 0})
        else:
            model = SVR(**params)
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        block_df = df.loc[te, ["date", "ticker", "adj_close", f"target_price_h{horizon}", target]].copy()
        block_df = block_df.rename(columns={target: "y_true", f"target_price_h{horizon}": "price_future"})
        block_df["price_t"] = df.loc[te, "adj_close"].to_numpy()
        block_df["y_pred"] = pred
        block_df["model"] = model_name
        rows.append(block_df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(models_dir / f"walk_forward_{_slug(model_name)}_h{horizon}.parquet", index=False)
    m = evaluate_predictions(out["price_t"], out["price_future"], out["y_true"], out["y_pred"])
    m.update({"model": model_name, "horizon": horizon, "scheme": "walk_forward"})
    log_experiment(reports / "experiments.csv", model_name, horizon, {"refit_every": refit_every}, m, seed, "walk_forward")
    pd.DataFrame([m]).to_csv(reports / f"walk_forward_h{horizon}.csv", index=False)
    return out


def sentiment_ablation(cfg: Dict[str, Any], horizon: int = 5, n_resamples: int = 300) -> pd.DataFrame:
    """Best ML model with no sentiment, FinBERT features and VADER features.

    Reports the test-window change in MAE, direction and IC with date-block
    bootstrap intervals. This is the PRD Section 10.3 integration test.
    """
    from src.evaluate import evaluate_predictions

    seed = int(cfg["project"]["seed"])
    panel = add_ticker_dummies(build_features(cfg), cfg["universe"]["equities"])
    sent = pd.read_parquet(path_from_config(cfg, "processed") / "sentiment_daily.parquet")
    s = sent[["date", "ticker", "vader_sentiment"]].copy()
    s = s.sort_values(["ticker", "date"])
    s["vader_mom3"] = s.groupby("ticker")["vader_sentiment"].transform(lambda x: x.rolling(3).mean()).diff(3)
    panel = panel.merge(s, on=["date", "ticker"], how="left").fillna({"vader_sentiment": 0.0, "vader_mom3": 0.0})
    dummies = [f"t_{t}" for t in cfg["universe"]["equities"]]
    sets = {
        "none": feature_list(False) + dummies,
        "finbert": feature_list(True) + dummies,
        "vader": feature_list(False) + ["vader_sentiment", "vader_mom3", "news_spike"] + dummies,
    }
    card = json.loads((path_from_config(cfg, "reports") / f"ml_best_h{horizon}.json").read_text())
    best_model = card["best_by_val_mae"]["model"]
    model_card = json.loads((path_from_config(cfg, "models") / "model_cards" / f"{_slug(best_model)}_h{horizon}.json").read_text())
    params = model_card["params"]
    target = f"target_h{horizon}"
    df = panel.dropna(subset=[target]).reset_index(drop=True)
    split = df["split"].to_numpy()
    results = {}
    preds = {}
    for name, cols in sets.items():
        X = df[cols].to_numpy(float)
        y = df[target].to_numpy(float)
        if best_model == "XGBoost":
            model = XGBRegressor(**params)
        elif best_model == "Random Forest":
            model = RandomForestRegressor(**params)
        elif best_model == "SVR":
            sc = StandardScaler().fit(X[split == "train"])
            model = SVR(**params)
            X = sc.transform(X)
        else:
            sc = StandardScaler().fit(X[split == "train"])
            model = Ridge(**params)
            X = sc.transform(X)
        train = split == "train"
        model.fit(X[train], y[train])
        pred = model.predict(X)
        results[name] = {}
        for sp in ("val", "test"):
            mask = split == sp
            m = evaluate_predictions(df.loc[mask, "adj_close"], df.loc[mask, f"target_price_h{horizon}"], y[mask], pred[mask])
            results[name][sp] = m
            if sp == "test":
                preds[name] = (y[mask], pred[mask], df.loc[mask, "date"].to_numpy())
    rows = []
    for name in ("none", "finbert", "vader"):
        m = results[name]["test"]
        rows.append({"features": name, "split": "test", **m})
    base = results["none"]["test"]
    test_rows = df.loc[split == "test", ["date", "target_h5"]].copy()
    test_rows["pred_none"] = preds["none"][1]
    for name in ("finbert", "vader"):
        test_rows[f"pred_{name}"] = preds[name][1]
    dates = test_rows["date"].to_numpy()
    for name in ("finbert", "vader"):
        delta_mae = float(base["mae_return"] - results[name]["test"]["mae_return"])
        # Bootstrap paired differences by resampling whole dates.
        sims_mae, sims_dir, sims_ic = [], [], []
        rng = np.random.default_rng(seed)

        def _blocks(dates, block=20, rng=rng):
            unique = np.sort(pd.unique(dates))
            n = len(unique)
            n_blocks = int(np.ceil(n / block))
            for _ in range(n_resamples):
                starts = rng.integers(0, n, size=n_blocks)
                chosen = np.concatenate([np.take(unique, np.arange(s, s + block), mode="wrap") for s in starts])[:n]
                yield chosen

        def _metrics(sub, col):
            yt = sub["target_h5"].to_numpy(dtype=float)
            yp = sub[col].to_numpy(dtype=float)
            mae = float(np.mean(np.abs(yt - yp)))
            dirn = float(np.mean(np.sign(yt) == np.sign(yp)))
            ic = float(pd.Series(yt).corr(pd.Series(yp), method="spearman"))
            return mae, dirn, ic

        for chosen in _blocks(dates):
            sub = test_rows[test_rows["date"].isin(chosen)]
            if len(sub) < 10:
                continue
            m0 = _metrics(sub, "pred_none")
            m1 = _metrics(sub, f"pred_{name}")
            sims_mae.append(m0[0] - m1[0])
            sims_dir.append(m1[1] - m0[1])
            sims_ic.append(m1[2] - m0[2])
        rows.append(
            {
                "features": name,
                "split": "test",
                **results[name]["test"],
                "delta_mae_vs_none": delta_mae,
                "delta_direction_vs_none": results[name]["test"]["direction_accuracy"] - base["direction_accuracy"],
                "delta_ic_vs_none": results[name]["test"]["ic"] - base["ic"],
                "delta_mae_ci": np.nanpercentile(sims_mae, [2.5, 97.5]).tolist(),
                "delta_direction_ci": np.nanpercentile(sims_dir, [2.5, 97.5]).tolist(),
                "delta_ic_ci": np.nanpercentile(sims_ic, [2.5, 97.5]).tolist(),
            }
        )
    table = pd.DataFrame(rows).drop_duplicates("features", keep="last").reset_index(drop=True)
    table.to_csv(path_from_config(cfg, "reports") / "sentiment_ablation.csv", index=False)
    save_json({"best_model": best_model, "horizon": horizon, "table": rows}, path_from_config(cfg, "reports") / "sentiment_ablation.json")
    LOG.info("sentiment ablation:\n%s", table[["features", "mae_return", "direction_accuracy", "ic"]].to_string(index=False))
    return table


def _slug(name: str) -> str:
    return name.replace(" ", "_").lower()


def _short(split: str) -> str:
    return "val" if split == "val" else "test"


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the four classical ML regressors")
    parser.add_argument("--horizon", nargs="+", type=int, default=None)
    parser.add_argument("--no-sentiment", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config()
    seed = int(cfg["project"]["seed"])
    set_seed(seed)
    panel = build_features(cfg)
    horizons = args.horizon or cfg["horizons"]
    for h in horizons:
        run_horizon(panel, h, cfg, include_sentiment=not args.no_sentiment, seed=seed)
    if args.walk_forward:
        for h in horizons:
            info = json.loads((path_from_config(cfg, "reports") / f"ml_best_h{h}.json").read_text())
            name = info["best_by_val_mae"]["model"]
            if name in {"Ridge", "Random Forest", "XGBoost", "SVR"}:
                walk_forward(panel, h, name, cfg, seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
