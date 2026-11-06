"""Layer 5 - Modern Portfolio Theory, Monte Carlo cross-check and backtests.

Max-Sharpe weights use Ledoit-Wolf covariance, a long-only 30% cap and a
documented historical/model return blend. Backtests apply weights from t+1,
charge 10 bps per side, and report PSR/Deflated Sharpe (N4).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src import finmath as fm
from src.utils import get_logger, load_config, path_from_config, save_json

LOG = get_logger("northgate.portfolio")


def price_matrix(prices: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    wide = prices.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index()
    return wide.reindex(columns=tickers).ffill()


def return_matrix(prices_wide: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices_wide).diff()


def _optimise(
    mu: np.ndarray,
    prices_window: pd.DataFrame,
    risk_free_rate: float,
    cap: float,
    fallback: bool = True,
) -> Tuple[np.ndarray, str]:
    from pypfopt import EfficientFrontier, risk_models

    mu = np.nan_to_num(np.asarray(mu, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(mu).any() or np.allclose(mu, 0):
        return np.full(len(prices_window.columns), 1 / len(prices_window.columns)), "equal_weight_fallback"
    S = risk_models.CovarianceShrinkage(prices_window).ledoit_wolf()
    ef = EfficientFrontier(mu, S, weight_bounds=(0, cap))
    try:
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        method = "max_sharpe"
    except Exception as exc:  # noqa: BLE001 - solver can fail on infeasible rf/return combos
        LOG.warning("max_sharpe infeasible (rf=%.4f): %s; using min-variance", risk_free_rate, exc)
        if not fallback:
            raise
        ef = EfficientFrontier(mu, S, weight_bounds=(0, cap))
        try:
            ef.min_volatility()
            method = "min_variance_fallback"
        except Exception as exc2:  # noqa: BLE001
            LOG.warning("min-variance also failed: %s; using equal weight", exc2)
            return np.full(len(prices_window.columns), 1 / len(prices_window.columns)), "equal_weight_fallback"
    w = np.array([ef.clean_weights()[t] for t in prices_window.columns], dtype=float)
    w = np.clip(w, 0, cap)
    w = w / w.sum() if w.sum() else np.full(len(w), 1 / len(w))
    return w, method


def historical_mu(returns_window: pd.DataFrame, periods: int = 252) -> np.ndarray:
    return returns_window.mean().to_numpy() * periods


def model_mu(predictions: pd.Series, horizon: int = 5, periods: int = 252) -> np.ndarray:
    """Annualise a predicted h-day log return."""
    return np.expm1(predictions.to_numpy()) * (periods / horizon)


def blend_mu(mu_hist: np.ndarray, mu_model: np.ndarray, lam: float) -> np.ndarray:
    return lam * mu_model + (1 - lam) * mu_hist


def _risk_free_at(macro: pd.DataFrame, date: pd.Timestamp, default: float) -> float:
    if macro.empty or "date" not in macro or "DGS3MO" not in macro:
        return default
    m = macro.copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    row = m[m["date"] <= pd.Timestamp(date)].tail(1)
    if row.empty or pd.isna(row["DGS3MO"].iloc[0]):
        return default
    return float(row["DGS3MO"].iloc[0]) / 100.0


def _predictions_for(pred: pd.DataFrame, date: pd.Timestamp, tickers: List[str]) -> Optional[pd.Series]:
    g = pred[pred["date"] == date]
    if g.empty:
        return None
    s = g.drop_duplicates("ticker").set_index("ticker")["y_pred"].reindex(tickers)
    return s if s.notna().all() else None


def simulate(
    prices_wide: pd.DataFrame,
    macro: pd.DataFrame,
    predictions: pd.DataFrame,
    tickers: List[str],
    cfg: Dict[str, Any],
    split: str = "test",
    lam: float = 0.5,
    include_costs: bool = True,
    rebalance_days: int = 5,
    method: str = "max_sharpe",
) -> Tuple[pd.Series, pd.DataFrame, List[str]]:
    """Weekly rebalanced backtest; weights at t are applied from t+1."""
    p = cfg["portfolio"]
    lookback = int(p["covariance_lookback"])
    cap = float(p["per_asset_cap"])
    costs = float(p["cost_bps"]) / 10_000 if include_costs else 0.0
    rf_default = float(p["risk_free_default"])
    rets = return_matrix(prices_wide)
    split_dates = predictions[predictions["split"] == split]["date"].drop_duplicates().sort_values().tolist()
    if not split_dates:
        return pd.Series(dtype=float), pd.DataFrame(), []
    start, end = split_dates[0], split_dates[-1]
    dates = rets.index[(rets.index >= start) & (rets.index <= end)]
    rows: List[Dict[str, Any]] = []
    weights_prev = np.full(len(tickers), 1 / len(tickers))
    used_methods: List[str] = []
    holdings: Dict[pd.Timestamp, np.ndarray] = {}

    for i in range(0, len(dates), rebalance_days):
        t = dates[i]
        block = dates[i + 1 : i + 1 + rebalance_days]
        if len(block) == 0:
            break
        hist = prices_wide.loc[:t].tail(lookback)
        if len(hist) < 60:
            continue
        pred_t = _predictions_for(predictions, t, tickers)
        if pred_t is None:
            # Keep the previous weights if this date has no complete forecast.
            weights = weights_prev
        else:
            rw = rets.loc[:t].tail(lookback)
            rf = _risk_free_at(macro, t, rf_default)
            mu_h = historical_mu(rw)
            mu_m = model_mu(pred_t, horizon=5)
            mu = blend_mu(mu_h, mu_m, lam) if method == "max_sharpe" else mu_h
            if method == "min_variance":
                from pypfopt import EfficientFrontier, risk_models

                S = risk_models.CovarianceShrinkage(hist).ledoit_wolf()
                ef = EfficientFrontier(mu, S, weight_bounds=(0, cap))
                ef.min_volatility()
                weights = np.array([ef.clean_weights()[x] for x in tickers], dtype=float)
                used_methods.append("min_variance")
            else:
                weights, used = _optimise(mu, hist, rf, cap)
                used_methods.append(used)
        weights = np.clip(weights, 0, cap)
        weights = weights / weights.sum() if weights.sum() else np.full(len(weights), 1 / len(weights))

        # Turnover is charged on the first day after the decision.
        if i == 0:
            turnover = np.abs(weights).sum()
        else:
            turnover = np.abs(weights - weights_prev).sum()
        block_rets = rets.loc[block, tickers].fillna(0.0).to_numpy()
        port_rets = block_rets @ weights
        if costs:
            port_rets[0] -= costs * turnover
        for d, r in zip(block, port_rets, strict=False):
            holdings[d] = weights
            rows.append({"date": d, "return": float(r), "rebalance_date": t})
        weights_prev = weights

    if not rows:
        return pd.Series(dtype=float), pd.DataFrame(), used_methods
    out = pd.DataFrame(rows).set_index("date")
    weights_df = pd.DataFrame(
        {t: holdings[d] for d, t in zip(out.index, out["rebalance_date"], strict=False)},
        index=tickers,
    ).T
    return out["return"], weights_df, used_methods


def performance(returns: pd.Series, benchmark: pd.Series, rf: float = 0.0) -> Dict[str, float]:
    r = returns.dropna()
    b = benchmark.reindex(r.index).dropna()
    if len(r) == 0:
        return {}
    beta = fm.beta(r, b) if len(b) > 2 else np.nan
    return {
        "total_return": float((1 + r).prod() - 1),
        "annual_return": float((1 + r).prod() ** (252 / len(r)) - 1),
        "annual_volatility": float(r.std(ddof=1) * np.sqrt(252)),
        "sharpe": fm.sharpe_ratio(r, rf),
        "sortino": fm.sortino_ratio(r, rf),
        "max_drawdown": fm.max_drawdown(r),
        "calmar": fm.calmar_ratio(r),
        "beta": float(beta),
        "var95": fm.historical_var(r),
        "days": int(len(r)),
    }


def equal_weight_backtest(prices_wide: pd.DataFrame, dates: pd.DatetimeIndex, tickers: List[str], rebalance_days: int = 21) -> pd.Series:
    rets = return_matrix(prices_wide).loc[dates, tickers].fillna(0.0)
    w = np.full(len(tickers), 1 / len(tickers))
    out = pd.Series(index=dates, dtype=float)
    for i in range(0, len(dates), rebalance_days):
        block = dates[i : i + rebalance_days]
        out.loc[block] = rets.loc[block].to_numpy() @ w
    return out


def buy_and_hold(prices_wide: pd.DataFrame, ticker: str, dates: pd.DatetimeIndex) -> pd.Series:
    p = prices_wide[ticker].reindex(dates).ffill()
    return np.log(p).diff().fillna(0.0)


def _cap_weights(W: np.ndarray, cap: float) -> np.ndarray:
    """Project random simplex weights onto the long-only capped simplex."""
    W = np.clip(np.asarray(W, dtype=float), 0.0, cap)
    for _ in range(50):
        excess = 1.0 - W.sum(axis=1)
        free = W < cap - 1e-12
        denom = free.sum(axis=1)
        if np.all(np.abs(excess) < 1e-12):
            break
        add = np.where(free, excess[:, None] / np.maximum(denom, 1)[:, None], 0.0)
        W = np.clip(W + add, 0.0, cap)
    return W


def efficient_frontier(
    mu: np.ndarray,
    prices_window: pd.DataFrame,
    cap: float,
    rf: float,
    points: int = 40,
    mc_samples: int = 20000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    from pypfopt import EfficientFrontier, risk_models

    S = risk_models.CovarianceShrinkage(prices_window).ledoit_wolf()
    ef = EfficientFrontier(mu, S, weight_bounds=(0, cap))
    ef.min_volatility()
    w_min = np.array([ef.clean_weights()[t] for t in prices_window.columns], dtype=float)
    mu_min, sig_min = ef.portfolio_performance(risk_free_rate=rf)[:2]

    ef2 = EfficientFrontier(mu, S, weight_bounds=(0, cap))
    try:
        ef2.max_sharpe(risk_free_rate=rf)
        w_max = np.array([ef2.clean_weights()[t] for t in prices_window.columns], dtype=float)
        mu_max, sig_max, sr_max = ef2.portfolio_performance(risk_free_rate=rf)
        method = "max_sharpe"
    except ValueError:
        ef2 = EfficientFrontier(mu, S, weight_bounds=(0, cap))
        ef2.min_volatility()
        w_max = w_min.copy()
        mu_max, sig_max = mu_min, sig_min
        sr_max = (mu_max - rf) / sig_max if sig_max else np.nan
        method = "min_variance_fallback"

    rows = []
    lo, hi = float(np.min(mu)), float(np.max(mu))
    for target in np.linspace(lo, max(hi, lo + 1e-6), points):
        try:
            ef3 = EfficientFrontier(mu, S, weight_bounds=(0, cap))
            ef3.efficient_return(target)
            r, v = ef3.portfolio_performance(risk_free_rate=rf)[:2]
            rows.append({"target_return": target, "return": r, "volatility": v, "sharpe": (r - rf) / v if v else np.nan})
        except Exception:  # noqa: BLE001 - infeasible target
            continue
    frontier = pd.DataFrame(rows)

    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(len(mu)), size=mc_samples)
    W = _cap_weights(W, cap)
    mu_p = W @ mu
    vol_p = np.sqrt(np.einsum("ij,jk,ik->i", W, S.to_numpy(), W))
    sr_p = (mu_p - rf) / np.where(vol_p == 0, np.nan, vol_p)
    cloud = pd.DataFrame({"return": mu_p, "volatility": vol_p, "sharpe": sr_p})
    chosen = {"min_variance": {"weights": dict(zip(prices_window.columns, w_min, strict=False)), "return": float(mu_min), "volatility": float(sig_min)},
              "tangency": {"weights": dict(zip(prices_window.columns, w_max, strict=False)), "return": float(mu_max), "volatility": float(sig_max), "sharpe": float(sr_max)},
              "method": method}
    return frontier, cloud, chosen


def select_lambda(
    prices_wide: pd.DataFrame,
    bench_wide: pd.DataFrame,
    macro: pd.DataFrame,
    predictions: pd.DataFrame,
    tickers: List[str],
    cfg: Dict[str, Any],
) -> Tuple[float, pd.DataFrame]:
    rows = []
    for lam in cfg["portfolio"]["mu_lambdas"]:
        r, _, _ = simulate(prices_wide, macro, predictions, tickers, cfg, split="val", lam=lam)
        m = performance(r, buy_and_hold(bench_wide, cfg["universe"]["benchmark"], r.index), cfg["portfolio"]["risk_free_default"])
        rows.append({"lambda": lam, **m})
    table = pd.DataFrame(rows)
    best = float(table.sort_values("sharpe", ascending=False).iloc[0]["lambda"]) if len(table) else 0.5
    return best, table


def run_portfolio(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    processed = path_from_config(cfg, "processed")
    models_dir = path_from_config(cfg, "models")
    reports = path_from_config(cfg, "reports")
    tickers = cfg["universe"]["equities"]

    prices = pd.read_parquet(processed / "prices_clean.parquet")
    macro = pd.read_parquet(processed / "macro_clean.parquet")
    prices_wide = price_matrix(prices, tickers)
    bench_wide = price_matrix(prices, tickers + [cfg["universe"]["benchmark"], cfg["universe"]["benchmark_total_return"]])
    macro = macro.sort_values("date")

    best_path = models_dir / "best_forecaster.json"
    if best_path.exists():
        best = json.loads(best_path.read_text())
        pred = pd.read_parquet(models_dir / best["predictions_file"])
    else:
        cand = sorted(reports.glob("ml_best_h5.json"))
        if not cand:
            raise FileNotFoundError("run models_ml first (no h=5 predictions found)")
        info = json.loads(cand[0].read_text())
        name = info["best_by_val_mae"]["model"]
        pred = pd.read_parquet(models_dir / f"pred_{name.replace(' ', '_').lower()}_h5_test.parquet")
        val = pd.read_parquet(models_dir / f"pred_{name.replace(' ', '_').lower()}_h5_val.parquet")
        pred = pd.concat([val, pred], ignore_index=True)

    lam, lam_table = select_lambda(prices_wide, bench_wide, macro, pred, tickers, cfg)
    lam_table.to_csv(reports / "lambda_selection.csv", index=False)
    LOG.info("selected lambda=%.2f", lam)

    test = pred[pred["split"] == "test"]
    dates = pd.DatetimeIndex(sorted(test["date"].unique()))
    net, weights, methods = simulate(prices_wide, macro, pred, tickers, cfg, split="test", lam=lam, include_costs=True)
    gross, _, _ = simulate(prices_wide, macro, pred, tickers, cfg, split="test", lam=lam, include_costs=False)
    eq = equal_weight_backtest(prices_wide, dates, tickers, 21)
    minvar, _, _ = simulate(prices_wide, macro, pred, tickers, cfg, split="test", lam=lam, include_costs=True, method="min_variance")
    bench = buy_and_hold(bench_wide, cfg["universe"]["benchmark"], dates)
    spy = buy_and_hold(bench_wide, cfg["universe"]["benchmark_total_return"], dates)

    rf = cfg["portfolio"]["risk_free_default"]
    metrics = {
        "optimised_net": performance(net, bench, rf),
        "optimised_gross": performance(gross, bench, rf),
        "equal_weight": performance(eq, bench, rf),
        "min_variance": performance(minvar, bench, rf),
        "gspc_buy_hold": performance(bench, bench, rf),
        "spy_buy_hold": performance(spy, bench, rf),
    }
    metrics["optimised_net"]["sharpe_ci"] = fm.block_bootstrap_sharpe(net, block=cfg["portfolio"]["bootstrap_block"], n_resamples=cfg["portfolio"]["bootstrap_resamples"])
    metrics["optimised_net"]["psr"] = fm.probabilistic_sharpe_ratio(net)
    metrics["optimised_net"]["dsr"] = fm.deflated_sharpe_ratio(net, lam_table["sharpe"].to_numpy())

    last_date = dates.max()
    last_pred = _predictions_for(test, last_date, tickers)
    hist = prices_wide.loc[:last_date].tail(int(cfg["portfolio"]["covariance_lookback"]))
    rw = return_matrix(prices_wide).loc[:last_date].tail(int(cfg["portfolio"]["covariance_lookback"]))
    mu = blend_mu(historical_mu(rw), model_mu(last_pred), lam)
    frontier, cloud, chosen = efficient_frontier(mu, hist, cfg["portfolio"]["per_asset_cap"], rf,
                                                 cfg["portfolio"]["frontier_points"], cfg["portfolio"]["mc_samples"], cfg["project"]["seed"])
    frontier.to_parquet(reports / "efficient_frontier.parquet", index=False)
    cloud.sample(min(6000, len(cloud)), random_state=cfg["project"]["seed"]).to_parquet(reports / "monte_carlo_cloud.parquet", index=False)

    net.rename("return").to_frame().to_parquet(models_dir / "backtest_returns.parquet")
    weights.to_parquet(models_dir / "backtest_weights.parquet")
    comparison = pd.DataFrame(
        {
            "optimised_net": net,
            "optimised_gross": gross,
            "equal_weight": eq,
            "min_variance": minvar,
            "gspc": bench,
            "spy": spy,
        }
    ).fillna(0.0)
    comparison.to_parquet(models_dir / "backtest_comparison.parquet")
    pd.DataFrame(metrics).T.to_csv(reports / "portfolio_backtest.csv")
    save_json(
        {
            "lambda": lam,
            "min_variance": chosen["min_variance"],
            "tangency": chosen["tangency"],
            "method": chosen["method"],
            "metrics": metrics,
            "weights_last": chosen["tangency"]["weights"],
            "risk_free_rate": rf,
        },
        models_dir / "portfolio.json",
    )
    LOG.info("portfolio: net Sharpe %.3f vs equal-weight %.3f", metrics["optimised_net"].get("sharpe", np.nan), metrics["equal_weight"].get("sharpe", np.nan))
    return {"metrics": metrics, "lambda": lam, "weights": weights, "returns": net}


if __name__ == "__main__":
    run_portfolio()
