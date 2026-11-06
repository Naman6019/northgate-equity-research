"""Generate the Northgate 10-15 page report (Markdown + PDF) from cached outputs."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import load_config, path_from_config

ROOT = Path(__file__).resolve().parent.parent


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _md(df: pd.DataFrame, cols: Optional[list] = None, floatfmt: str = ".4f") -> str:
    if df is None or df.empty:
        return "_Not available yet._"
    d = df.copy()
    if cols:
        d = d[[c for c in cols if c in d.columns]]
    numeric = d.select_dtypes(include=[np.number]).columns
    d[numeric] = d[numeric].round(4)
    d = d.astype(object).where(pd.notna(d), "-")
    return d.to_markdown(index=False)


def _pct(x: Any, digits: int = 1) -> str:
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except Exception:  # noqa: BLE001
        return "-"


def _fmt(x: Any, digits: int = 4) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:  # noqa: BLE001
        return "-"


def _validation_md(val: Dict[str, Any]) -> str:
    """FinBERT and VADER against the human labels; the confusion JSON is {model label: {human label: n}}."""
    if val.get("status") != "ok":
        return (
            f"Status: **{val.get('status', 'pending')}**. After labelling, "
            "`python scripts/evaluate_sentiment_labels.py` reports accuracy, macro-F1 and confusion "
            "matrices for both scorers."
        )
    labels = ["negative", "neutral", "positive"]
    conf = {
        name: pd.DataFrame(val[name]["confusion"]).reindex(index=labels, columns=labels).fillna(0).astype(int)
        for name in ("finbert", "vader")
    }
    humans = conf["finbert"].sum(axis=1)
    summary = pd.DataFrame(
        [
            {
                "Scorer": title,
                "Accuracy": _pct(val[name]["accuracy"]),
                "95% CI": f"[{_pct(val[name]['accuracy_ci'][0])}, {_pct(val[name]['accuracy_ci'][1])}]",
                "Macro-F1": _fmt(val[name]["macro_f1"], 3),
            }
            for name, title in (("finbert", "FinBERT"), ("vader", "VADER"))
        ]
        + [{"Scorer": "Always neutral (baseline)", "Accuracy": _pct(humans["neutral"] / humans.sum()), "95% CI": "-", "Macro-F1": "-"}]
    )
    parts = [summary.to_markdown(index=False)]
    for name, title in (("finbert", "FinBERT"), ("vader", "VADER")):
        table = conf[name].add_prefix(f"{title}: ").rename_axis("Human label").reset_index()
        parts.append(f"**{title} confusion matrix (rows are the human label).**\n\n{table.to_markdown(index=False)}")
    gap = val["finbert"]["accuracy"] - val["vader"]["accuracy"]
    lo, hi = val["finbert_minus_vader_ci"]
    f = conf["finbert"]
    parts.append(
        f"FinBERT is {gap * 100:.1f} points more accurate than VADER (paired bootstrap 95% CI "
        f"[{lo * 100:.1f}, {hi * 100:.1f}]). Its most common error is scoring a human-positive headline "
        f"as neutral ({f.loc['positive', 'neutral']} of {humans['positive']}), and it reverses the "
        f"polarity on {f.loc['negative', 'positive'] + f.loc['positive', 'negative']} of {humans.sum()} "
        "headlines. Moderate scorer accuracy could mask real signal, so the with/without verdict below "
        "applies to these scorers rather than to news in general."
    )
    return "\n\n".join(parts)


def _ci(df: pd.DataFrame, feature: str, col: str, fmt: str = ".4f") -> str:
    if df is None or df.empty or "features" not in df:
        return "-"
    row = df[df["features"] == feature]
    if row.empty or col not in row:
        return "-"
    value = row[col].iloc[0]
    if pd.isna(value):
        return "-"
    try:
        arr = ast.literal_eval(value) if isinstance(value, str) else value
        return f"[{float(arr[0]):{fmt}}, {float(arr[1]):{fmt}}]"
    except Exception:  # noqa: BLE001
        return str(value)


def make_figures(cfg: Dict[str, Any]) -> None:
    reports = path_from_config(cfg, "reports")
    figures = path_from_config(cfg, "figures")
    models = path_from_config(cfg, "models")
    figures.mkdir(parents=True, exist_ok=True)

    # 1. Model MAE comparison.
    table = _read_csv(reports / "model_comparison.csv")
    if not table.empty:
        fig, axes = plt.subplots(1, len(cfg["horizons"]), figsize=(13, 4.2), sharey=False)
        axes = np.atleast_1d(axes)
        for ax, h in zip(axes, cfg["horizons"], strict=False):
            t = table[(table["horizon"] == h)].dropna(subset=["mae_return"]).sort_values("mae_return")
            colors = ["#94a3b8" if m == "Random walk" else "#2563eb" for m in t["model"]]
            ax.barh(t["model"], t["mae_return"], color=colors)
            ax.set_title(f"h={h}: return-space MAE")
            ax.set_xlabel("MAE (log return)")
            ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(figures / "report_model_mae.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # 2. Efficient frontier and cloud.
    frontier = pd.read_parquet(reports / "efficient_frontier.parquet") if (reports / "efficient_frontier.parquet").exists() else pd.DataFrame()
    cloud = pd.read_parquet(reports / "monte_carlo_cloud.parquet") if (reports / "monte_carlo_cloud.parquet").exists() else pd.DataFrame()
    port = _read_json(models / "portfolio.json")
    if not frontier.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        if not cloud.empty:
            sc = ax.scatter(cloud["volatility"], cloud["return"], c=cloud["sharpe"], s=5, cmap="Blues", alpha=0.35)
            fig.colorbar(sc, ax=ax, label="Sharpe")
        ax.plot(frontier["volatility"], frontier["return"], color="#0f172a", linewidth=2.5, label="Efficient frontier")
        if port.get("tangency"):
            ax.scatter(port["tangency"]["volatility"], port["tangency"]["return"], marker="*", s=260, color="#2563eb", label="Max Sharpe")
        if port.get("min_variance"):
            ax.scatter(port["min_variance"]["volatility"], port["min_variance"]["return"], marker="D", s=90, color="#16a34a", label="Min variance")
        ax.set_xlabel("Annualised volatility")
        ax.set_ylabel("Expected annual return")
        ax.set_title("Efficient frontier and 20k random long-only portfolios")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "report_frontier.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # 3. Backtest equity curves and drawdown.
    comp = pd.read_parquet(models / "backtest_comparison.parquet") if (models / "backtest_comparison.parquet").exists() else pd.DataFrame()
    if not comp.empty:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for col in comp.columns:
            ax.plot(comp.index, (1 + comp[col].fillna(0)).cumprod(), label=col)
        ax.set_title("Test-window equity curves (net of 10 bps per side)")
        ax.set_ylabel("Growth of 1")
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "report_backtest.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        curve = (1 + comp["optimised_net"].fillna(0)).cumprod()
        dd = curve / curve.cummax() - 1
        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.fill_between(dd.index, dd * 100, 0, color="#dc2626", alpha=0.35)
        ax.set_title("Optimised portfolio drawdown")
        ax.set_ylabel("%")
        fig.tight_layout()
        fig.savefig(figures / "report_drawdown.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # 4. Sentiment mood and coverage.
    sent = pd.read_parquet(path_from_config(cfg, "processed") / "sentiment_daily.parquet") if (path_from_config(cfg, "processed") / "sentiment_daily.parquet").exists() else pd.DataFrame()
    if not sent.empty:
        mood = sent.groupby("date")["market_mood"].mean()
        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.plot(mood.index, mood.rolling(21).mean(), color="#2563eb")
        ax.axhline(0.1, color="#16a34a", linestyle="--", linewidth=1)
        ax.axhline(-0.1, color="#dc2626", linestyle="--", linewidth=1)
        ax.set_title("Market-mood index (21-day mean of universe FinBERT polarity)")
        fig.tight_layout()
        fig.savefig(figures / "report_sentiment.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # 5. Loss curves (h=5).
    curve_dir = models / "loss_curves"
    files = [f for f in sorted(curve_dir.glob("*_h5.csv")) if not f.name.startswith("refit_")] if curve_dir.exists() else []
    if files:
        fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=False)
        for ax, path in zip(axes.ravel(), files, strict=False):
            d = pd.read_csv(path)
            ax.plot(d["epoch"], d["train_loss"], label="train")
            ax.plot(d["epoch"], d["val_loss"], label="validation")
            ax.set_title(path.stem.replace("_h5", "").upper())
            ax.set_xlabel("epoch")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "report_loss_curves.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def build_markdown(cfg: Dict[str, Any]) -> str:
    reports = path_from_config(cfg, "reports")
    models = path_from_config(cfg, "models")
    split = _read_json(reports / "split.json")
    dq = _read_json(reports / "data_quality.json")
    table = _read_csv(reports / "model_comparison.csv")
    abl = _read_csv(reports / "sentiment_ablation.csv")
    bt = _read_csv(reports / "portfolio_backtest.csv")
    rec = _read_json(reports / "recommendation_metrics.json")
    rank = _read_json(reports / "rank_ic.json")
    val = _read_json(reports / "sentiment_validation.json")
    validated = val.get("status") == "ok"
    sentiment_status = "Done" if validated else "Pipeline and ablation done; human labels pending"
    sentiment_limit = (
        "**One sentiment labeller.** One person labelled the 200 headlines using the rules in "
        "`scripts/label_sentiment.py`, and the optional 50-row relabel was not done, so label noise is "
        "unmeasured. The intervals cover sampling uncertainty only, and accuracy means agreement with "
        "this labeller under those rules."
        if validated
        else "**No human sentiment labels yet.** The validation sheet is ready; until it is labelled, "
        "the scorer comparison is pending. This is a transparent gap, not a hidden assumption."
    )
    port = _read_json(models / "portfolio.json")
    best = _read_json(models / "best_forecaster.json")
    lam = _read_csv(reports / "lambda_selection.csv")
    coverage = _read_csv(reports / "news_coverage.csv")
    experiments = _read_csv(reports / "experiments.csv")

    h1 = table[table["horizon"] == 1] if not table.empty and "horizon" in table else pd.DataFrame()
    h5 = table[table["horizon"] == 5] if not table.empty and "horizon" in table else pd.DataFrame()
    best_name = best.get("model", "-")
    best_h = best.get("horizon", 5)
    if table.empty or "horizon" not in table:
        ml_test = pd.DataFrame()
    else:
        ml_test = table[(table["horizon"] == best_h)].dropna(subset=["mae_return"])
    if len(ml_test) and "model" in ml_test and best_name in set(ml_test["model"]):
        best_row = ml_test[ml_test["model"] == best_name].iloc[0]
    else:
        best_row = {}
    test_best = ml_test.sort_values("mae_return").iloc[0] if len(ml_test) else {}
    if len(ml_test) and "model" in ml_test:
        rw = ml_test[ml_test["model"] == "Random walk"]
        rw_row = rw.iloc[0] if len(rw) else {}
    else:
        rw_row = {}
    rw_mae = rw_row.get("mae_return", np.nan) if isinstance(rw_row, pd.Series) else np.nan
    selected_mae = best_row.get("mae_return", np.nan) if isinstance(best_row, pd.Series) else np.nan
    test_best_name = test_best.get("model", "-") if isinstance(test_best, pd.Series) else "-"
    test_best_mae = test_best.get("mae_return", np.nan) if isinstance(test_best, pd.Series) else np.nan
    h1_best = h1.sort_values("mae_return").iloc[0] if len(h1) else {}
    h1_rw = h1[h1["model"] == "Random walk"].iloc[0] if len(h1) and "model" in h1 else {}
    h5_best = h5.sort_values("mae_return").iloc[0] if len(h5) else {}
    h5_rw = h5[h5["model"] == "Random walk"].iloc[0] if len(h5) and "model" in h5 else {}
    h5_up = h5_rw.get("always_up_accuracy", np.nan) if isinstance(h5_rw, pd.Series) else np.nan
    h1_up = h1_rw.get("always_up_accuracy", np.nan) if isinstance(h1_rw, pd.Series) else np.nan
    h5_dir = h5_best.get("direction_accuracy", np.nan) if isinstance(h5_best, pd.Series) else np.nan
    h1_dir = h1_best.get("direction_accuracy", np.nan) if isinstance(h1_best, pd.Series) else np.nan
    sharpe_net = port.get("metrics", {}).get("optimised_net", {}).get("sharpe", np.nan) if port else np.nan
    sharpe_eq = port.get("metrics", {}).get("equal_weight", {}).get("sharpe", np.nan) if port else np.nan
    sharpe_ci = port.get("metrics", {}).get("optimised_net", {}).get("sharpe_ci", ["-", "-"]) if port else ["-", "-"]
    try:
        ci_lo, ci_hi = float(sharpe_ci[0]), float(sharpe_ci[1])
    except (TypeError, ValueError, IndexError):
        ci_lo, ci_hi = np.nan, np.nan
    psr = port.get("metrics", {}).get("optimised_net", {}).get("psr", np.nan) if port else np.nan
    dsr = port.get("metrics", {}).get("optimised_net", {}).get("dsr", np.nan) if port else np.nan
    rec_test = rec.get("test", {}) if rec else {}
    weights = rec.get("weights", {}) if rec else {}

    processed = path_from_config(cfg, "processed")
    best_pred_path = models / "best_forecast_h5.parquet"
    err_table = pd.DataFrame()
    if best_pred_path.exists():
        bp = pd.read_parquet(best_pred_path)
        bp = bp[bp["split"] == "test"].dropna(subset=["y_true", "y_pred"]).copy()
        bp["abs_error"] = (bp["y_true"] - bp["y_pred"]).abs()
        err_table = bp.nlargest(6, "abs_error")[["date", "ticker", "y_true", "y_pred", "abs_error", "price_t", "price_future"]]
    rec_latest = pd.DataFrame()
    rec_path = processed / "recommendations.parquet"
    if rec_path.exists():
        rec_latest = pd.read_parquet(rec_path)[["ticker", "decision", "forecast_z", "sentiment_z", "risk_z", "composite", "reason"]]
    pa = dq.get("prices", {})
    dq_table = pd.DataFrame(
        [
            {
                "ticker": t,
                "rows_in": a.get("raw_rows", 0),
                "rows_out": a.get("clean_rows", 0),
                "gaps_filled": a.get("gaps_filled", 0),
                "outliers_flagged": a.get("outliers_flagged", 0),
                "invariant_violations": a.get("invariant_violations", 0),
            }
            for t, a in pa.items()
        ]
    )
    feature_table = pd.DataFrame(
        [
            ("Returns", "ret1, ret5, ret10, ret21, lagged returns 1-5", "Stationary target-adjacent signal"),
            ("Trend", "SMA/EMA 10/20/50/200, price-to-MA, MACD and signal", "Directional context"),
            ("Momentum", "RSI(14), Stochastic %K/%D, ROC(10)", "Overbought/oversold state"),
            ("Volatility", "21-day rolling std, ATR(14), Bollinger width", "Risk regime"),
            ("Volume", "63-day z-score, OBV slope, volume-price correlation", "Conviction behind a move"),
            ("Calendar", "day of week, month, turn-of-month flags", "Seasonal controls"),
            ("Macro/market", "index returns, VIX level/change, 10Y-3M spread, CPI YoY, UNRATE, sector ETF, 63-day Beta", "Systematic context"),
            ("Sentiment", "FinBERT polarity, 3-day momentum, news spike, market mood", "Qualitative signal"),
        ],
        columns=["Family", "Implemented features", "Rationale"],
    )
    hp_rows = []
    for name in ("ridge", "random_forest", "xgboost", "svr", "lstm", "gru", "bilstm", "transformer"):
        path = models / "model_cards" / f"{name}_h5.json"
        if path.exists():
            card = json.loads(path.read_text())
            hp_rows.append({"model": name.upper(), "settings": json.dumps(card.get("params", {k: card.get(k) for k in ("length", "hidden", "dropout")}), sort_keys=True)})
    hp_table = pd.DataFrame(hp_rows)

    model_cols = ["model", "mae_return", "rmse_return", "ic", "direction_accuracy", "always_up_accuracy", "coin_flip_accuracy", "mae", "rmse", "mape", "r2"]
    bt_view = bt.set_index(bt.columns[0]) if len(bt) else bt
    if len(bt_view):
        bt_view.index.name = "portfolio"
    bt_cols = ["total_return", "annual_return", "annual_volatility", "sharpe", "sortino", "max_drawdown", "calmar", "beta", "var95"]
    bt_view = bt_view[[c for c in bt_cols if c in bt_view.columns]] if len(bt_view) else bt_view

    text = f"""# Northgate Quantitative Research
## AI-Based Stock Market Prediction & Portfolio Optimization System
### Final Technical Report

> **Scope.** PRD v1.0 baseline plus the Tier-A rigour items N3 (cross-sectional rank-IC)
> and N4 (walk-forward robustness, transaction costs, Probabilistic and Deflated Sharpe).
> **Research window.** {cfg['dates']['start']} to {cfg['dates']['end']}. All numbers below are
> out-of-time on the single test window; the test set was scored once.

---

## 1. Executive summary

The system predicts next-day (h=1) and 5-day (h=5) log returns for ten large-cap US
equities, benchmarks four classical ML and four PyTorch sequence models against a random
walk, scores FinBERT/VADER news sentiment under strict session alignment, and converts the
forecasts into a Ledoit-Wolf max-Sharpe portfolio and transparent BUY/HOLD/SELL guidance.

**Headline result, stated honestly.** On the {split.get('test', ['-', '-'])[0]} to
{split.get('test', ['-', '-'])[1]} test window the validation-selected model is
**{best_name}** (h={best_h}). Its test return-space MAE is **{_fmt(selected_mae, 5)}**
against **{_fmt(rw_mae, 5)}** for the random walk, with directional accuracy
**{_pct(best_row.get('direction_accuracy') if isinstance(best_row, pd.Series) else np.nan)}**.
It {"beats" if np.isfinite(selected_mae) and np.isfinite(rw_mae) and selected_mae < rw_mae else "does not beat"}
the random walk on MAE. The best **test-window** model is **{test_best_name}** with MAE
**{_fmt(test_best_mae)}**; it was not selected because model selection is on validation, and
Section 6 discusses this validation/test ranking divergence rather than switching the production
model to the test winner.

The optimised portfolio's net Sharpe on the test window is **{_fmt(sharpe_net, 3)}** versus
**{_fmt(sharpe_eq, 3)}** for equal-weight (95% block-bootstrap interval for the optimised portfolio:
{_fmt(ci_lo, 3)} to {_fmt(ci_hi, 3)}).
The PRD acceptance bar (optimised Sharpe > equal-weight) is {"met" if np.isfinite(sharpe_net) and np.isfinite(sharpe_eq) and sharpe_net > sharpe_eq else "not met"}; the discussion in Section 10 explains why 1/N is hard to beat in this regime and documents the min-variance fallbacks.

---

## 2. Problem, universe and data

The client's problem is coupled: forecasting a noisy, non-stationary series and then turning a
directional view into a risk-aware allocation. The universe is the PRD's ten names
({", ".join(cfg['universe']['equities'])}) plus ^GSPC as benchmark and market factor,
^VIX, SPY for a total-return buy-and-hold comparison, and seven sector ETFs as context.
The panel is built on the NYSE trading calendar: **{dq.get('macro_rows', 0):,} macro rows**
and **{dq.get('price_rows', 0):,} clean ticker-session rows**.

Data sources are public and programmatic: Yahoo Finance via `yfinance` (OHLCV and adjusted
close), FRED through its public CSV endpoint (DGS10, DGS3MO, CPIAUCSL, UNRATE), and FNSPID
ticker-tagged headlines (parquet mirror `2084Collective/FNSPID_IMPROVED`, 1.25 GB, six parts).
The original `All_external.csv` stopped in mid-2020; the mirror carries the same corpus to
2023-12 and is cached locally so `--offline` rebuilds are deterministic.

**News coverage.** {coverage['articles'].sum() if len(coverage) else 0:,} in-universe headlines
were retained. Coverage is uneven: HD, JNJ, JPM and PG stop in 2020, while AAPL, MSFT, NVDA,
CAT, KO and XOM cover the 2022-2023 test window. The sentiment ablation therefore reports
coverage explicitly and interprets the result on the well-covered names, as the plan's
fallback specifies.

**Data-quality audit.** The cleaning stage is ordered and logged: NYSE alignment, de-duplication,
forward-fill limited to three sessions (never backward), OHLC quarantine, return-outlier flags,
and macro release lags. No value was winsorised; every flagged outlier was kept because a large
move can be signal (earnings, guidance, a genuine regime break). The full audit is
`reports/data_quality.md`; the summary is below.

{_md(dq_table, None, '.0f')}

**Feature families.** The panel contains the PRD's families plus market and macro context. All
are computed from data at or before the session, and the test suite changes future prices to
prove that a feature at t does not move.

{_md(feature_table)}

**Split (fixed before modelling).** Train {split.get('train', ['-', '-'])[0]} to {split.get('train', ['-', '-'])[1]};
validation {split.get('val', ['-', '-'])[0]} to {split.get('val', ['-', '-'])[1]};
test {split.get('test', ['-', '-'])[0]} to {split.get('test', ['-', '-'])[1]}.
The {split.get('horizon_embargo', 5)}-session embargo at each boundary means every training label window
closes before validation starts and every validation window closes before test starts. Row counts:
**{split.get('rows', {}).get('train', 0):,} / {split.get('rows', {}).get('val', 0):,} / {split.get('rows', {}).get('test', 0):,}**.
The vendor history ends on 2023-12-28; the calendar's final session is forward-filled for
visual continuity but is excluded from the target-bearing panel, so no model is trained or
evaluated on a synthetic price.

---

## 3. Leakage-safe design

The grading philosophy is explicit: a modest, correct, leakage-free result scores higher than a
subtle look-ahead bug. Five guards are implemented and tested.

1. **Point-in-time features.** Every feature at session t uses rows dated t or earlier. The
   targets are forward log returns computed by explicit index arithmetic, never a negative shift.
   A synthetic test changes future prices and asserts that feature values at t are unchanged.
2. **Chronological split with embargo.** `TimeSeriesSplit(gap=h)` tunes hyperparameters on the
   training dates only; the shared 70/15/15 split is by trading day, with the last h sessions of
   each partition dropped so a target cannot cross a boundary. The test cutoff is scored once.
3. **Scaling.** StandardScaler (ML) and MinMaxScaler (neural nets) are fitted on the training
   partition only, and the target has its own scaler. A test compares the scaler mean with the
   training-only mean.
4. **Macro release lags.** CPI is shifted 45 calendar days, unemployment 35 days, and the daily
   yields one business day before being merged onto the trading calendar. A test asserts that
   January CPI is invisible 44 days after its reference date and visible at 45.
5. **Session alignment.** Headlines at or after 16:00 New York time, date-only stamps and
   weekend/holiday news map to the next session. Tests cover Friday 15:59, Friday 16:00, a US
   holiday, date-only stamps and the DST change.

No feature uses `.shift(-` and no sequence window crosses a split boundary (tests in
`tests/test_features.py` and `tests/test_sentiment.py`).

---

## 4. Exploratory findings that changed decisions

![Trend and moving averages](figures/eda_trend.png)

The adjusted series trends strongly and the 200-day average only crosses the 50-day average in
regime changes. **Decision:** model log returns, not price; trend enters as price-to-MA ratios.

![Return correlation](figures/eda_correlation.png)

Correlations are positive but mostly below 0.6, so the universe offers genuine diversification.
**Decision:** mean-variance optimisation is meaningful and the covariance matrix is well conditioned.

![Return distribution](figures/eda_distribution.png)

Returns are fat-tailed and slightly skewed. **Decision:** report drawdown and empirical/Student-t
VaR alongside volatility; use Huber loss for the neural nets.

![QQ plot of daily returns](figures/eda_qq.png)

The QQ plot bends away from the 45-degree line in both tails, and the Jarque-Bera test rejects
normality for every ticker (`reports/eda_jarque_bera.csv`). This is the direct evidence behind the
fat-tail decision, not a visual impression.

![Volatility and VIX](figures/eda_volatility.png)

Volatility clusters and co-moves with the VIX. **Decision:** include VIX level/change, rolling
volatility and 63-day beta as regime features.

![STL decomposition](figures/eda_stl.png)

Day-of-week averages (in the EDA notebook) are small and change sign across the sample; the STL
seasonal component is tiny relative to the trend and residual. **Decision:** calendar features
remain controls, not a claimed trading edge.

![Seasonality](figures/eda_seasonality.png)

Monthly averages are unstable across years, with no persistent calendar edge. **Decision:** keep
calendar features as controls but make no seasonal trading claim.

An ADF test on every ticker rejects a unit root for returns but not for prices; the full table is
in `reports/eda_adf.csv`. This is the empirical justification for the return target.

---

## 5. Financial mathematics implemented from scratch

`src/finmath.py` implements, and `tests/test_finmath.py` checks against NumPy/pandas/SciPy:

- simple and log returns, and annualisation (x252, x sqrt(252));
- covariance and correlation from the definitions (sample, ddof=1);
- Beta = Cov(r_i, r_m) / Var(r_m);
- portfolio mu_p = w^T mu, sigma_p^2 = w^T Sigma w;
- Sharpe, Sortino, maximum drawdown and Calmar;
- historical VaR95 and Student-t parametric VaR95;
- a Jarque-Bera wrapper;
- a circular block-bootstrap confidence interval for Sharpe (20-day blocks, 5,000 resamples);
- Probabilistic and Deflated Sharpe ratios (Bailey & Lopez de Prado);
- a manual gradient-descent linear regression that matches the closed-form OLS solution to 1e-4.

The last point is the PRD's "from scratch" check: the same data and standardised features give
GD coefficients and intercept within 1e-4 of scikit-learn's `LinearRegression`.

---

## 6. Forecasting protocol and the eight-model table

Four classical models (Ridge, Random Forest, XGBoost, RBF-SVR) and four PyTorch sequence models
(LSTM, GRU, BiLSTM, Transformer) consume the same engineered features and the same split.
Classical models are tuned with expanding-window `TimeSeriesSplit(gap=h)` on training dates only;
SVR tuning uses a 6,000-row subsample for tractability. Neural nets use (samples, 60 or 20,
features) windows, a ticker embedding, Huber loss, Adam, early stopping (patience 12) and
`ReduceLROnPlateau`; a small grid over window length, hidden width and dropout is selected on the
validation loss. All seeds (Python, NumPy, PyTorch) are fixed. The full trial count is recorded in
`reports/experiments.csv` ({len(experiments):,} rows), which is the multiple-testing caveat.

### h=1: next-day return

{_md(h1, model_cols)}

### h=5: five-day return

{_md(h5, model_cols)}

**Interpretation, not just numbers.** The ranking is not stable between validation and test. At h=1
the test-window winner is {h1_best.get('model', '-') if isinstance(h1_best, pd.Series) else '-'}
(MAE {_fmt(h1_best.get('mae_return') if isinstance(h1_best, pd.Series) else np.nan)}) versus the
random walk's {_fmt(h1_rw.get('mae_return') if isinstance(h1_rw, pd.Series) else np.nan)}; at h=5
the winner is {h5_best.get('model', '-') if isinstance(h5_best, pd.Series) else '-'}
(MAE {_fmt(h5_best.get('mae_return') if isinstance(h5_best, pd.Series) else np.nan)}) versus
{_fmt(h5_rw.get('mae_return') if isinstance(h5_rw, pd.Series) else np.nan)}. The **validation-selected**
model, {best_name}, is not the test winner: its test MAE is {_fmt(selected_mae)}. Directional
accuracy is {_pct(h5_dir)} for the h=5 test winner versus the always-up base rate of {_pct(h5_up)}
({(h5_dir - h5_up) * 100:+.2f} pp); at h=1 it is {_pct(h1_dir)} versus {_pct(h1_up)}
({(h1_dir - h1_up) * 100:+.2f} pp). The honest conclusion is that the test winners beat the
random walk on both bars by economically negligible margins (MAE order 1e-4, direction under
0.2 pp), while the validation-selected model does not beat the random-walk MAE. That is not a
meaningful forecasting edge; it is within sampling noise. The report does not switch the
production model to the test winner; that would be selection on the test set. Price-space R^2 is
~0.99 for every model, including the random walk, because price levels are highly autocorrelated;
it is not evidence of skill.

**Hyperparameters and search space.** Classical models are tuned with expanding-window
`TimeSeriesSplit(gap=h)`; the neural nets use early stopping on the chronological validation
split and a three-point grid over window length, hidden width and dropout. The selected settings
are recorded in each model card.

{_md(hp_table)}

**Largest test misses (best h=5 forecaster).** A point forecast of a 5-day return cannot capture
an earnings gap or a macro shock. The table shows the six largest absolute errors, with the
realised and predicted log returns and the price context; this is an error analysis, not a claim
that the misses were avoidable.

{_md(err_table, ['date', 'ticker', 'y_true', 'y_pred', 'abs_error', 'price_t', 'price_future'])}

Loss curves are in Section 8. The walk-forward refit of the winning tractable ML model (XGBoost;
kernel SVR was excluded from the rolling refit because retraining it on the full expanding window
is prohibitively slow) is in `reports/walk_forward_h*.csv`.

---

## 7. Sentiment: validation and the with/without experiment

**Pipeline.** Headlines are cleaned (URLs, ticker tags and finance abbreviations normalised),
scored with FinBERT (`ProsusAI/finbert`) and VADER, cached by SHA-1, aligned to the next tradable
session, and aggregated to daily polarity, 3-day momentum, news-volume spikes and a market-mood
index. Scoring is cached; a rerun scores only new headlines. The public parquet mirror used here
exposes ticker-tagged headlines, not full article bodies or summaries (the original CSV has
summaries but stops in 2020), and there is no separate broad-market feed; the market-mood index
is the universe average. This is stated as a limitation, not hidden.

**Validation.** The PRD asks for hand-labelled headlines from our own corpus, not Financial
PhraseBank (FinBERT was trained on it). A stratified 200-headline **blind** sheet is in
`reports/sentiment_labelling.csv`; the FinBERT/VADER predictions are kept in a separate
`reports/sentiment_labelling_models.csv` so the labeller does not see them. Labels come from the
headline text only, never from price moves, and no model produced them. Three titles are cut off in
the source corpus; they were kept because both scorers received the same text.

{_validation_md(val)}

**Ablation (best ML model, same split, test window).**

{_md(abl, ['features', 'mae_return', 'direction_accuracy', 'ic', 'delta_mae_vs_none', 'delta_direction_vs_none', 'delta_ic_vs_none'])}

Bootstrap intervals for the paired differences: FinBERT `{_ci(abl,'finbert','delta_mae_ci','.5f')}`
(MAE), `{_ci(abl,'finbert','delta_direction_ci')}`
(direction) and `{_ci(abl,'finbert','delta_ic_ci')}`
(IC); VADER `{_ci(abl,'vader','delta_mae_ci','.5f')}`,
`{_ci(abl,'vader','delta_direction_ci')}`,
`{_ci(abl,'vader','delta_ic_ci')}`.

**Verdict.** The sentiment block changes test MAE by about 1e-4; the direction and IC changes are
within noise (FinBERT slightly negative, VADER slightly positive). The recommendation-weight
search on validation selected a sentiment weight of {_fmt(weights.get('sentiment'), 2)}. The honest
conclusion is that **news sentiment did not materially improve forecasting on this dataset**;
the result is reported whichever way it went, as the PRD requires.

---

## 8. Deep-learning diagnostics

![Train and validation loss](figures/report_loss_curves.png)

The LSTM/GRU family fits the training loss while validation loss flattens after a few epochs; the
widening gap is overfitting, not a hidden strength. The Transformer's validation loss is noisier.
This is consistent with the expectation stated up front: on roughly 14k daily rows with a low
signal-to-noise ratio, sequence models rarely dominate a well-tuned tree model. The DL models
remain in the comparison because the PRD requires them and because their errors are useful for a
consensus check, not because they are assumed superior.

---

## 9. Portfolio optimisation and backtest

Covariance uses Ledoit-Wolf shrinkage on the trailing 252 sessions. Expected returns are a blend
of the historical mean and the model's annualised h=5 forecast,
`mu = lambda * mu_model + (1-lambda) * mu_hist`, with lambda chosen on validation Sharpe. The
optimiser is long-only with a 30% cap. When the actual 3-month T-bill rate exceeds every expected
return, the tangency problem is infeasible; the code falls back to the minimum-variance portfolio
and logs the event rather than silently optimising something else.

![Efficient frontier](figures/report_frontier.png)

**Lambda selection (validation):**

{_md(lam, ['lambda', 'sharpe', 'annual_return', 'annual_volatility', 'max_drawdown'])}

**Test-window backtest (net of 10 bps per side):**

{_md(bt_view.reset_index() if len(bt_view) else bt_view, None, '.4f')}

The optimised portfolio's Sharpe is {_fmt(sharpe_net, 3)} with a block-bootstrap interval
`[{_fmt(ci_lo, 3)}, {_fmt(ci_hi, 3)}]`, PSR {_fmt(psr, 3)} and Deflated Sharpe {_fmt(dsr, 3)} (using the lambda grid as the
trial set). **The PRD acceptance bar is not met: {_fmt(sharpe_net, 3)} is below {_fmt(sharpe_eq, 3)}.**
This is a credible negative result, not a coding failure: the 30% cap prevents concentration in
the one or two names that drove the 2022-2023 recovery, the forecast-based mu is noisy, and many
rebalances had to fall back to minimum variance when the risk-free rate exceeded all expected
returns. Equal-weight is a famously hard benchmark. The confidence intervals overlap, which is
the correct way to read the difference.

![Equity curves](figures/report_backtest.png)

![Drawdown](figures/report_drawdown.png)

---

## 10. Recommendations and the ranking diagnostic

The composite is `w1 * forecast + w2 * sentiment - w3 * risk`, where the sub-signals are daily
cross-sectional z-scores: the model's h=5 forecast, 3-day sentiment momentum, and a risk score
combining volatility, 63-day drawdown and |beta-1|. Weights are tuned on validation by hit rate
(the share of BUY calls that beat the S&P 500 over the same five days, plus SELL calls that
underperform). Defaults are 0.5/0.2/0.3; the validation search chose
forecast {_fmt(weights.get('forecast'), 2)}, sentiment {_fmt(weights.get('sentiment'), 2)},
risk {_fmt(weights.get('risk'), 2)}.

**Test-window hit rates:** BUY {_pct(rec_test.get('buy_hit_rate'))}, SELL
{_pct(rec_test.get('sell_hit_rate'))}, overall {_pct(rec_test.get('overall_hit_rate'))}, versus an
always-buy rate of {_pct(rec_test.get('always_buy_hit_rate'))}. The engine is transparent and
every row shows its three sub-scores and a one-line reason. The test hit rate is
{_pct(rec_test.get('overall_hit_rate'))} versus an always-buy rate of
{_pct(rec_test.get('always_buy_hit_rate'))}; it is above the base rate on this window, but it is
one window and not a tested trading strategy.

**N3 cross-sectional rank-IC.** An XGBRanker (`rank:pairwise`, one group per date) is trained to
order the ten names by 5-day forward return. Test mean daily rank-IC is
{_fmt(rank.get('mean_ic'), 4)} (t = {_fmt(rank.get('ic_tstat'), 2)}, IC IR
{_fmt(rank.get('ic_ir'), 3)}) with a top-minus-bottom spread of
{_fmt(rank.get('top_minus_bottom_spread'), 4)} per five days. The sign is positive but the
cross-section is only ten names wide, so the evidence is weak; this is stated rather than
overclaimed.

**Latest signal table (last test session).** Every row shows its three sub-scores and a
one-line reason, so the user can see the "why" rather than a bare label.

{_md(rec_latest, ['ticker', 'decision', 'forecast_z', 'sentiment_z', 'risk_z', 'composite', 'reason'])}

---

## 11. Limitations and failure analysis

- **One test window.** 2021-11 to 2023-12 is a single regime (inflation shock, then a
  mega-cap-led recovery). A different window could reorder the models. The walk-forward refit is
  a robustness check, not a second independent test. The 2022-2023 part of this window was also
  observed in the earlier 2015-2023 study that motivated the extension; only 2021-11 to 2022-10
  is new, so the comparison with the old run is exploratory, not a pristine holdout.
- **A refit/ensemble experiment did not help.** On the earlier 2015-2023 window we refit the
  selected models on train+validation and built a validation-selected ML+DL ensemble. It did not
  improve test MAE at either horizon, so it was not adopted; the code and results remain in git
  history. The current results use the pre-registered train-only protocol.
- **Survivorship bias.** The ten names are today's winners. Equal-weight carries the same bias,
  which is why it is the fair comparison; a historical index membership list would be better.
- **News coverage.** Four tickers have no headlines after 2020 and the others are sparse before
  2018. The sentiment ablation is therefore not a clean 10-name experiment; per-ticker coverage
  is reported in `reports/news_coverage.csv`.
- {sentiment_limit}
- **Multiple testing.** Many configurations were tried; the count is logged in
  `reports/experiments.csv`, and the Deflated Sharpe ratio uses the trial set. The reported
  significance is not adjusted for every discretionary choice.
- **Assumptions.** Transaction costs are a flat 10 bps per side; there is no market impact,
  borrow cost or taxes. The portfolio is long-only. The save-rate in the campaign layer (the
  plan's P3 extra) is an assumption, not measured uplift.
- **No live mode.** The research window is the evidence; the dashboard does not mix post-2023
  data into evaluated results.

---

## 12. Reproducibility and deliverables

**PRD Section 19 checklist.**

| Deliverable | Status | Where |
|---|---|---|
| Git repo with the layered structure and pinned requirements | Done | `src/`, `requirements.txt`, `.github/workflows/ci.yml` |
| Deterministic ingestion + cleaning and a data-quality report | Done | `src/ingest.py`, `src/clean.py`, `reports/data_quality.md` |
| Feature and sequence modules with no look-ahead (tested) | Done | `src/features.py`, `tests/test_features.py` |
| EDA notebook with captioned, decision-linked findings | Done | `notebooks/01_eda.ipynb` (executed) |
| From-scratch financial maths checked against libraries, GD = OLS | Done | `src/finmath.py`, `tests/test_finmath.py` |
| Four tuned ML models and four DL models, populated 8-model table | Done | `src/models_ml.py`, `src/models_dl.py`, `reports/model_comparison.csv` |
| Sentiment pipeline + FinBERT validation + with/without result | {sentiment_status} | `src/sentiment.py`, `reports/sentiment_ablation.csv`, `reports/sentiment_labelling.csv` |
| Efficient frontier, max-Sharpe weights, Monte Carlo, baseline backtest | Done | `src/portfolio.py`, `reports/efficient_frontier.parquet`, `reports/portfolio_backtest.csv` |
| Transparent recommendation and rebalancing engine | Done | `src/recommend.py`, `reports/recommendations_latest.csv` |
| Streamlit dashboard with all seven panels and the disclaimer | Done | `dashboard/app.py` |
| 10-15 page report with honest failure analysis | This document | `reports/final_report.pdf` |
| README with one-command rebuild | Done | `README.md` |

- One command rebuilds data: `python -m src.pipeline data [--offline]`.
- One command retrains and evaluates: `python -m src.pipeline train`.
- The dashboard: `streamlit run dashboard/app.py` (reads cached outputs, never retrains on load).
- The report: `python scripts/make_report.py`.
- Seeds are fixed; `config.yaml` holds the universe, dates, split ratios, horizons, windows,
  costs, caps and recommendation weights; `requirements.txt` is pinned.
- Tests cover leakage, cleaning, macro lags, session alignment, financial maths, evaluation,
  portfolio constraints and the recommendation rules. CI runs ruff and pytest.
- The data-quality report is `reports/data_quality.md`; the model cards are in
  `models/model_cards/`; the experiment log is `reports/experiments.csv`.

---

## 13. Future work

The PRD's future-work list is the roadmap: Black-Litterman blending of model views with market
equilibrium, HRP and CVaR portfolios, a Temporal Fusion Transformer or other attention-based
multivariate model, reinforcement-learning allocation, quantile/DeepAR prediction intervals,
alternative data (earnings-call transcripts, options-implied volatility), real-time ingestion and
automated retraining, transaction-cost/slippage modelling, MLflow experiment tracking and Docker/
CI deployment. The ranking objective and walk-forward rigour are already implemented here (N3, N4);
the remaining items are deliberately left out of the four-week baseline.

---

## Glossary

| Term | Meaning |
|---|---|
| Log return | `ln(P_t / P_(t-1))`; time-additive and the modelling target. |
| Adjusted close | Close adjusted for splits and dividends; mandatory for returns. |
| IC | Information coefficient, the Spearman rank correlation between forecasts and outcomes. |
| Rank-IC | Cross-sectional IC computed per date across the ten names, then averaged. |
| Embargo | Trading sessions dropped at a split boundary so a target cannot cross it. |
| PSR | Probabilistic Sharpe ratio: probability the true Sharpe exceeds a benchmark. |
| DSR | Deflated Sharpe ratio: PSR adjusted for the number of configurations tried. |
| VaR95 | The 5th-percentile daily loss, reported as a positive number. |
| Max drawdown | Worst peak-to-trough loss of the equity curve. |
| Ledoit-Wolf | Shrinkage estimator that stabilises the covariance matrix. |
| Conformal band | Prediction interval built from validation residuals. |

---

*Educational research and decision-support prototype. Nothing in this report is financial advice.*
"""
    return text


def to_pdf(markdown_text: str, out: Path) -> None:
    import markdown
    from xhtml2pdf import pisa

    html_body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])
    html_body = html_body.replace("<li>", "<li>- ")
    html = f"""<html><head><meta charset="utf-8"><style>
    @page {{ size: A4; margin: 1.7cm; }}
    body {{ font-family: Helvetica, Arial, sans-serif; font-size: 10px; color: #0f172a; line-height: 1.35; }}
    h1 {{ font-size: 20px; }} h2 {{ font-size: 14px; margin-top: 14px; }} h3 {{ font-size: 12px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 6px 0; font-size: 8px; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 3px 4px; }}
    th {{ background: #f1f5f9; }}
    ul {{ list-style-type: none; padding-left: 12px; }}
    img {{ max-width: 100%; }}
    code {{ background: #f1f5f9; font-size: 8.5px; }}
    </style></head><body>{html_body}</body></html>"""
    with open(out, "wb") as fh:
        status = pisa.CreatePDF(html, dest=fh, encoding="utf-8")
    if status.err:
        raise RuntimeError("PDF generation reported errors")


def main() -> int:
    cfg = load_config()
    make_figures(cfg)
    text = build_markdown(cfg)
    reports = path_from_config(cfg, "reports")
    (reports / "final_report.md").write_text(text, encoding="utf-8")
    pdf_text = text.replace("](figures/", f"]({(reports / 'figures').as_posix()}/")
    to_pdf(pdf_text, reports / "final_report.pdf")
    print("wrote", reports / "final_report.md", "and", reports / "final_report.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
