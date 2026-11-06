"""Layer 6 - transparent BUY/HOLD/SELL signals and rebalancing guidance
(PRD Section 12), plus the cross-sectional ranking diagnostic (N3)."""

from __future__ import annotations

import json
from itertools import product
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.evaluate import cross_sectional_ic, ic_summary
from src.utils import get_logger, load_config, path_from_config, save_json

LOG = get_logger("northgate.recommend")


def zscore(df: pd.DataFrame, col: str) -> pd.Series:
    def _z(s: pd.Series) -> pd.Series:
        sd = s.std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / sd

    return df.groupby("date")[col].transform(_z)


def add_signal_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["forecast_z"] = zscore(df, "y_pred") if "y_pred" in df else 0.0
    df["sentiment_z"] = zscore(df, "sentiment_mom3")
    df["vol_z"] = zscore(df, "vol21")
    df["beta_z"] = zscore(df.assign(_b=(df["beta63"] - 1).abs()), "_b")
    # 63-session drawdown per ticker (negative), then cross-sectional z.
    def _dd(s: pd.Series) -> pd.Series:
        curve = (1 + s.fillna(0)).cumprod()
        return curve / curve.rolling(63, min_periods=10).max() - 1

    df["dd63"] = df.groupby("ticker")["ret1"].transform(_dd)
    df["dd_z"] = zscore(df, "dd63")
    df["risk_z"] = df[["vol_z", "beta_z", "dd_z"]].mean(axis=1)
    return df


def composite(df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    return (
        weights["forecast"] * df["forecast_z"]
        + weights["sentiment"] * df["sentiment_z"]
        - weights["risk"] * df["risk_z"]
    )


def decide(score: float, high: float, low: float) -> str:
    if score >= high:
        return "BUY"
    if score <= low:
        return "SELL"
    return "HOLD"


def rebalance(current: Dict[str, float], target: Dict[str, float], band: float = 0.05) -> Dict[str, Tuple[str, float]]:
    out = {}
    for ticker in target:
        drift = float(target[ticker] - current.get(ticker, 0.0))
        if abs(drift) > band:
            out[ticker] = ("INCREASE" if drift > 0 else "REDUCE", round(drift, 3))
    return out


def _benchmark_forward(prices: pd.DataFrame, benchmark: str, horizon: int) -> pd.DataFrame:
    p = prices[prices["ticker"] == benchmark].sort_values("date").reset_index(drop=True)
    a = p["adj_close"].to_numpy(dtype=float)
    fwd = np.full(len(a), np.nan)
    if len(a) > horizon:
        fwd[:-horizon] = np.log(a[horizon:] / a[:-horizon])
    return pd.DataFrame({"date": p["date"], "benchmark_fwd": fwd})


def _hit_rates(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, float]:
    d = df.dropna(subset=["target_h5", "benchmark_fwd", "decision"]).copy()
    buy = d[d["decision"] == "BUY"]
    sell = d[d["decision"] == "SELL"]
    buy_ok = float((buy["target_h5"] > buy["benchmark_fwd"]).mean()) if len(buy) else np.nan
    sell_ok = float((sell["target_h5"] < sell["benchmark_fwd"]).mean()) if len(sell) else np.nan
    calls = len(buy) + len(sell)
    hit = np.nansum([buy["target_h5"].gt(buy["benchmark_fwd"]).sum(), sell["target_h5"].lt(sell["benchmark_fwd"]).sum()])
    return {
        "buy_hit_rate": buy_ok,
        "sell_hit_rate": sell_ok,
        "overall_hit_rate": float(hit / calls) if calls else np.nan,
        "n_buy": int(len(buy)),
        "n_sell": int(len(sell)),
        "always_buy_hit_rate": float((d["target_h5"] > d["benchmark_fwd"]).mean()) if len(d) else np.nan,
    }


def tune_weights(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows = []
    for w1, w2 in product(np.arange(0.2, 0.9, 0.1), np.arange(0.0, 0.5, 0.1)):
        w3 = round(1 - w1 - w2, 10)
        if w3 < 0:
            continue
        weights = {"forecast": round(float(w1), 2), "sentiment": round(float(w2), 2), "risk": round(float(w3), 2)}
        scored = df.copy()
        scored["composite"] = composite(scored, weights)
        scored["decision"] = [decide(x, cfg["recommend"]["high"], cfg["recommend"]["low"]) for x in scored["composite"]]
        m = _hit_rates(scored, cfg)
        rows.append({**weights, **m})
    table = pd.DataFrame(rows).sort_values(["overall_hit_rate", "n_buy", "n_sell"], ascending=False)
    best = table.iloc[0] if len(table) else pd.Series({"forecast": 0.5, "sentiment": 0.2, "risk": 0.3})
    weights = {"forecast": float(best["forecast"]), "sentiment": float(best["sentiment"]), "risk": float(best["risk"])}
    return weights, table


def rank_ic_model(panel: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """XGBRanker (rank:pairwise) over dates; evaluate with daily rank-IC (N3)."""
    from xgboost import XGBRanker

    from src.features import SENTIMENT_FEATURES, TECH_FEATURES

    tickers = cfg["universe"]["equities"]
    df = panel.copy()
    dummies = pd.get_dummies(df["ticker"], prefix="t", dtype=float)
    df = pd.concat([df, dummies], axis=1)
    cols = TECH_FEATURES + SENTIMENT_FEATURES + [f"t_{t}" for t in tickers]
    df = df.dropna(subset=cols + ["target_h5"]).sort_values(["date", "ticker"]).reset_index(drop=True)
    train = df[df["split"] == "train"]
    groups = train.groupby("date").size().to_numpy()
    model = XGBRanker(
        objective="rank:pairwise",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=cfg["project"]["seed"],
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(train[cols].to_numpy(float), train["target_h5"].to_numpy(float), group=groups)

    preds = []
    for split in ("val", "test"):
        g = df[df["split"] == split].sort_values(["date", "ticker"]).reset_index(drop=True)
        if g.empty:
            continue
        X = g[cols].to_numpy(float)
        g = g[["date", "ticker", "target_h5", "split"]].copy()
        g["rank_score"] = model.predict(X)
        preds.append(g)
    pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    ic = cross_sectional_ic(pred[pred["split"] == "test"], "rank_score", "target_h5")
    summary = ic_summary(ic)
    spreads = []
    for _, g in pred[pred["split"] == "test"].groupby("date"):
        spreads.append(g.nlargest(3, "rank_score")["target_h5"].mean() - g.nsmallest(3, "rank_score")["target_h5"].mean())
    summary["top_minus_bottom_spread"] = float(np.mean(spreads)) if spreads else np.nan
    pred.to_parquet(path_from_config(cfg, "models") / "rank_predictions_h5.parquet", index=False)
    save_json(summary, path_from_config(cfg, "reports") / "rank_ic.json")
    LOG.info("rank model IC: %s", summary)
    return {"predictions": pred, "ic": ic, "summary": summary}


def run_recommendations(cfg: Optional[Dict[str, Any]] = None, with_rank: bool = True) -> Dict[str, Any]:
    cfg = cfg or load_config()
    processed = path_from_config(cfg, "processed")
    models_dir = path_from_config(cfg, "models")
    reports = path_from_config(cfg, "reports")
    tickers = cfg["universe"]["equities"]

    panel = pd.read_parquet(processed / "panel.parquet")
    prices = pd.read_parquet(processed / "prices_clean.parquet")
    best_path = models_dir / "best_forecaster.json"
    if best_path.exists():
        best = json.loads(best_path.read_text())
        pred = pd.read_parquet(models_dir / best["predictions_file"])
    else:
        info = json.loads((reports / "ml_best_h5.json").read_text())
        name = info["best_by_val_mae"]["model"].replace(" ", "_").lower()
        pred = pd.concat(
            [pd.read_parquet(models_dir / f"pred_{name}_h5_val.parquet"), pd.read_parquet(models_dir / f"pred_{name}_h5_test.parquet")],
            ignore_index=True,
        )
    pred = pred[["date", "ticker", "y_pred", "split"]].drop_duplicates(["date", "ticker"])
    df = panel.merge(pred, on=["date", "ticker"], how="inner", suffixes=("", "_pred"))
    df = df.rename(columns={"split_pred": "eval_split"}) if "split_pred" in df else df
    if "eval_split" not in df:
        df["eval_split"] = df["split"]
    bench = _benchmark_forward(prices, cfg["universe"]["benchmark"], 5)
    df = df.merge(bench, on="date", how="left")
    df = add_signal_features(df)

    val = df[df["eval_split"] == "val"]
    weights, grid = tune_weights(val, cfg)
    grid.to_csv(reports / "recommendation_weight_grid.csv", index=False)

    scored = df.copy()
    scored["composite"] = composite(scored, weights)
    scored["decision"] = [decide(x, cfg["recommend"]["high"], cfg["recommend"]["low"]) for x in scored["composite"]]
    scored["reason"] = scored.apply(
        lambda r: f"{r['decision']} - forecast {r['forecast_z']:+.2f}sd, sentiment {r['sentiment_z']:+.2f}sd, risk {r['risk_z']:+.2f}sd",
        axis=1,
    )
    metrics = {"weights": weights, "validation": _hit_rates(scored[scored["eval_split"] == "val"], cfg),
               "test": _hit_rates(scored[scored["eval_split"] == "test"], cfg)}
    pd.DataFrame([metrics["test"]]).to_csv(reports / "recommendation_backtest.csv", index=False)
    save_json(metrics, reports / "recommendation_metrics.json")

    latest_date = scored[scored["eval_split"] == "test"]["date"].max()
    latest = scored[scored["date"] == latest_date].copy()
    target = {}
    if (models_dir / "portfolio.json").exists():
        port = json.loads((models_dir / "portfolio.json").read_text())
        target = port.get("weights_last", {})
    current = {t: 1 / len(tickers) for t in tickers}
    actions = rebalance(current, target, cfg["recommend"]["rebalance_band"])
    latest["current_weight"] = latest["ticker"].map(current).fillna(0.0)
    latest["target_weight"] = latest["ticker"].map(target).fillna(0.0)
    latest["rebalance_action"] = latest["ticker"].map(lambda t: actions.get(t, ("HOLD", 0.0))[0])
    latest["rebalance_drift"] = latest["ticker"].map(lambda t: actions.get(t, ("HOLD", 0.0))[1])
    base_cols = ["date", "ticker", "forecast_z", "sentiment_z", "risk_z", "composite", "decision", "reason", "sentiment_available"]
    latest[base_cols + ["current_weight", "target_weight", "rebalance_action", "rebalance_drift"]].to_parquet(
        processed / "recommendations.parquet", index=False
    )
    latest[base_cols + ["current_weight", "target_weight", "rebalance_action", "rebalance_drift"]].to_csv(
        reports / "recommendations_latest.csv", index=False
    )
    scored[base_cols + ["target_h5", "benchmark_fwd", "eval_split"]].to_parquet(
        processed / "recommendation_history.parquet", index=False
    )

    out = {"weights": weights, "metrics": metrics, "latest": latest}
    if with_rank:
        out["rank"] = rank_ic_model(panel, cfg)
    LOG.info("recommendations: weights=%s test hit=%.3f", weights, metrics["test"].get("overall_hit_rate", np.nan))
    return out


if __name__ == "__main__":
    run_recommendations()
