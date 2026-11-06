"""Northgate - seven-panel Streamlit dashboard (PRD Section 13).

Reads cached model outputs only; it never retrains on page load. Every page
carries the educational-research disclaimer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

DISCLAIMER = (
    "Educational research, not financial advice. This system is a decision-support "
    "prototype; forecasts are uncertain and past performance does not guarantee future results."
)

st.set_page_config(page_title="Northgate Quant", page_icon="N", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem;}
    div[data-testid="stMetric"] {background: #0f172a11; border: 1px solid #0f172a22;
        padding: 10px 14px; border-radius: 10px;}
    .ng-disclaimer {position: fixed; bottom: 0; left: 0; right: 0; background: #0f172a;
        color: #f8fafc; text-align: center; padding: 6px; font-size: 12px; z-index: 999;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300)
def load_parquet(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=300)
def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=300)
def load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def money(x: float) -> str:
    return f"${x:,.2f}" if np.isfinite(x) else "-"


def _date_text(value) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return "-"


def sidebar() -> Dict[str, Any]:
    st.sidebar.title("Northgate Quant")
    st.sidebar.caption("Forecasting · Sentiment · MPT")
    page = st.sidebar.radio(
        "Panel",
        [
            "Overview",
            "Price & prediction",
            "Model comparison",
            "Portfolio analytics",
            "Risk dashboard",
            "Sentiment",
            "Recommendations",
        ],
    )
    panel = load_parquet(str(DATA / "panel.parquet"))
    tickers = sorted(panel["ticker"].unique()) if len(panel) else []
    ticker = st.sidebar.selectbox("Ticker", tickers) if tickers else None
    horizon = st.sidebar.selectbox("Forecast horizon", [1, 5], index=1)
    rf = st.sidebar.slider("Risk-free rate", 0.0, 0.10, 0.04, 0.005)
    st.sidebar.markdown("---")
    st.sidebar.caption("Research window 2009-01-01 to 2023-12-29. Live mode is disabled.")
    return {"page": page, "ticker": ticker, "horizon": horizon, "rf": rf}


def overview(ctx):
    st.title("Overview")
    panel = load_parquet(str(DATA / "panel.parquet"))
    prices = load_parquet(str(DATA / "prices_clean.parquet"))
    sent = load_parquet(str(DATA / "sentiment_daily.parquet"))
    port = load_json(str(MODELS / "portfolio.json"))
    best = load_json(str(MODELS / "best_forecaster.json"))
    if panel.empty or prices.empty:
        st.info("Run `python -m src.pipeline all --offline` first.")
        return
    last = panel["date"].max()
    prices = prices[prices["date"] <= last]
    rows = []
    rec = load_parquet(str(DATA / "recommendations.parquet"))
    decisions = rec.set_index("ticker")["decision"].to_dict() if len(rec) else {}
    for ticker, g in prices[prices["ticker"].isin(panel["ticker"].unique())].groupby("ticker"):
        g = g.sort_values("date")
        if len(g) < 22:
            continue
        rows.append(
            {
                "Ticker": ticker,
                "Last close": money(g["adj_close"].iloc[-1]),
                "1-day": f"{g['adj_close'].pct_change().iloc[-1] * 100:+.2f}%",
                "1-month": f"{g['adj_close'].pct_change(21).iloc[-1] * 100:+.2f}%",
                "Signal": decisions.get(ticker, "HOLD"),
            }
        )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickers", len(rows))
    c2.metric("Sessions", f"{panel['date'].nunique():,}")
    c3.metric("Best forecaster", best.get("model", "pending"))
    m = port.get("metrics", {}).get("optimised_net", {})
    eq = port.get("metrics", {}).get("equal_weight", {})
    c4.metric("Test Sharpe (net)", f"{m.get('sharpe', float('nan')):.2f}" if m else "-")
    st.caption(f"Equal-weight Sharpe: {eq.get('sharpe', float('nan')):.2f}" if eq else "Portfolio backtest pending.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Universe snapshot")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    with col2:
        st.subheader("Market mood")
        mood = float(sent[sent["date"] == sent["date"].max()]["market_mood"].mean()) if len(sent) else 0.0
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=mood,
                number={"suffix": " score", "valueformat": ".2f"},
                gauge={
                    "axis": {"range": [-0.3, 0.3]},
                    "bar": {"color": "#2563eb"},
                    "steps": [
                        {"range": [-0.3, -0.1], "color": "#fecaca"},
                        {"range": [-0.1, 0.1], "color": "#e2e8f0"},
                        {"range": [0.1, 0.3], "color": "#bbf7d0"},
                    ],
                    "threshold": {"line": {"color": "#0f172a", "width": 3}, "value": 0.1},
                },
                title={"text": "Bullish above +0.10"},
            )
        )
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"As of {_date_text(last)}; universe-average FinBERT polarity.")


def price_prediction(ctx):
    st.title("Price & prediction")
    best = load_json(str(MODELS / "best_forecaster.json"))
    hpath = MODELS / f"best_forecast_h{ctx['horizon']}.parquet"
    if hpath.exists():
        pred = load_parquet(str(hpath))
    else:
        pred = load_parquet(str(MODELS / (best.get("predictions_file", "")))) if best else pd.DataFrame()
    if pred.empty:
        st.info("Forecast artifacts are not built yet.")
        return
    ticker = ctx["ticker"]
    h = ctx["horizon"]
    p = pred[(pred["ticker"] == ticker) & (pred["horizon"] == h)]
    if p.empty:
        st.info(f"No {h}-day predictions for {ticker}.")
        return
    p = p.sort_values("date")
    hist = load_parquet(str(DATA / "prices_clean.parquet"))
    hist = hist[hist["ticker"] == ticker].sort_values("date").tail(260)
    q = 0.0
    v = p[p["split"] == "val"]
    if len(v):
        q = float(np.nanquantile(np.abs(v["y_true"] - v["y_pred"]), 0.9))
    test = p[p["split"] == "test"].tail(120)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["adj_close"], name="Adjusted close", line=dict(color="#0f172a")))
    if len(test):
        forecast_date = test["date"] + pd.offsets.BDay(h)
        pred_price = test["price_t"] * np.exp(test["y_pred"])
        upper = test["price_t"] * np.exp(test["y_pred"] + q)
        lower = test["price_t"] * np.exp(test["y_pred"] - q)
        fig.add_trace(go.Scatter(x=forecast_date, y=pred_price, name=f"Forecast h={h}", line=dict(color="#2563eb")))
        fig.add_trace(go.Scatter(x=forecast_date, y=upper, line=dict(width=0), showlegend=False))
        fig.add_trace(
            go.Scatter(x=forecast_date, y=lower, fill="tonexty", fillcolor="rgba(37,99,235,0.15)",
                       line=dict(width=0), name="90% split-conformal band")
        )
    fig.update_layout(height=460, xaxis_title="Session", yaxis_title="Price (USD)",
                      legend=dict(orientation="h", y=1.05), margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{best.get('model')} on the {h}-day horizon; band = 90th percentile of validation residuals (q={q:.4f}).")


def model_comparison(ctx):
    st.title("Model comparison")
    table = load_csv(str(REPORTS / "model_comparison.csv"))
    if table.empty:
        st.info("The eight-model table is not built yet.")
        return
    h = ctx["horizon"]
    t = table[table["horizon"] == h].copy()
    st.subheader(f"Eight models + random walk, h={h}, test window")
    show = [c for c in ["model", "mae_return", "rmse_return", "ic", "direction_accuracy", "always_up_accuracy", "coin_flip_accuracy", "mae", "rmse", "mape", "r2"] if c in t]
    st.dataframe(t[show].sort_values("mae_return"), width="stretch", hide_index=True)
    chart = t.dropna(subset=["mae_return"]).sort_values("mae_return")
    fig = px.bar(chart, x="mae_return", y="model", orientation="h", color="model",
                 title="Return-space MAE (lower is better)")
    fig.update_layout(height=380, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    cur = sorted((MODELS / "loss_curves").glob("*.csv")) if (MODELS / "loss_curves").exists() else []
    if cur:
        st.subheader("Train vs validation loss")
        cols = st.columns(min(4, len(cur)))
        for i, path in enumerate(cur):
            d = pd.read_csv(path)
            if "train_loss" not in d:
                continue
            fig = px.line(d, x="epoch", y=["train_loss", "val_loss"], title=path.stem.upper())
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10), legend_title="")
            cols[i % len(cols)].plotly_chart(fig, width="stretch")

    abl = load_csv(str(REPORTS / "sentiment_ablation.csv"))
    if len(abl):
        st.subheader("With vs without sentiment (test window)")
        st.dataframe(abl, width="stretch", hide_index=True)


def portfolio_panel(ctx):
    st.title("Portfolio analytics")
    port = load_json(str(MODELS / "portfolio.json"))
    frontier = load_parquet(str(REPORTS / "efficient_frontier.parquet"))
    cloud = load_parquet(str(REPORTS / "monte_carlo_cloud.parquet"))
    if port.get("tangency") is None:
        st.info("Run the portfolio stage first.")
        return
    st.subheader("Efficient frontier and Monte Carlo cloud")
    fig = go.Figure()
    if len(cloud):
        fig.add_trace(go.Scattergl(x=cloud["volatility"], y=cloud["return"], mode="markers",
                                   marker=dict(size=3, color=cloud["sharpe"], colorscale="Blues", opacity=0.35),
                                   name="20k random portfolios"))
    if len(frontier):
        fig.add_trace(go.Scatter(x=frontier["volatility"], y=frontier["return"], mode="lines+markers",
                                 line=dict(color="#0f172a", width=3), name="Efficient frontier"))
    tang = port["tangency"]
    mv = port["min_variance"]
    fig.add_trace(go.Scatter(x=[tang["volatility"]], y=[tang["return"]], mode="markers+text",
                             marker=dict(size=14, color="#2563eb", symbol="star"), text=["Max Sharpe"],
                             textposition="top center", name="Tangency"))
    fig.add_trace(go.Scatter(x=[mv["volatility"]], y=[mv["return"]], mode="markers+text",
                             marker=dict(size=12, color="#16a34a", symbol="diamond"), text=["Min variance"],
                             textposition="bottom center", name="Min variance"))
    fig.update_layout(height=500, xaxis_title="Annualised volatility", yaxis_title="Expected annual return",
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Target weights (max Sharpe)")
        w = pd.Series(tang["weights"]).sort_values(ascending=False)
        figw = px.pie(values=w.values, names=w.index, hole=0.45)
        figw.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(figw, width="stretch")
    with c2:
        st.subheader("Backtest metrics")
        mt = load_csv(str(REPORTS / "portfolio_backtest.csv"))
        if len(mt):
            mt = mt.set_index(mt.columns[0])
            mt.index.name = "portfolio"
            keep = [c for c in ["total_return", "annual_return", "annual_volatility", "sharpe", "sortino", "max_drawdown", "calmar", "beta", "var95"] if c in mt]
            st.dataframe(mt[keep], width="stretch")
        st.caption(f"Selected mu blend lambda = {port.get('lambda', '-')}; risk-free rate in the report = {port.get('risk_free_rate', '-')}.")

    st.subheader("Equity curves, test window")
    comp = load_parquet(str(MODELS / "backtest_comparison.parquet"))
    if len(comp):
        fig = go.Figure()
        for col in comp.columns:
            fig.add_trace(go.Scatter(x=comp.index, y=(1 + comp[col].fillna(0)).cumprod(), name=col))
        fig.update_layout(height=330, yaxis_title="Growth of 1", margin=dict(l=10, r=10, t=20, b=10),
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")

    with st.expander("Re-optimise with the selected risk-free rate and mu source"):
        from src.portfolio import (
            _optimise,
            blend_mu,
            historical_mu,
            model_mu,
            price_matrix,
            return_matrix,
        )
        from src.utils import load_config

        cfg = load_config()
        panel = load_parquet(str(DATA / "panel.parquet"))
        tickers = sorted(panel["ticker"].unique())
        prices = load_parquet(str(DATA / "prices_clean.parquet"))
        hist = price_matrix(prices, tickers).loc[: panel["date"].max()].tail(int(cfg["portfolio"]["covariance_lookback"]))
        rets = return_matrix(hist)
        pred_file = MODELS / "best_forecast_h5.parquet"
        pred = load_parquet(str(pred_file)) if pred_file.exists() else pd.DataFrame()
        mu_source = st.selectbox("Expected-return source", ["blend", "hist", "model"], index=0)
        mu_h = historical_mu(rets)
        if len(pred):
            last_pred = pred.sort_values("date").groupby("ticker").tail(1).set_index("ticker")["y_pred"].reindex(tickers)
            mu_m = model_mu(last_pred.fillna(0.0))
        else:
            mu_m = mu_h
        if mu_source == "hist":
            mu = mu_h
        elif mu_source == "model":
            mu = mu_m
        else:
            mu = blend_mu(mu_h, mu_m, float(port.get("lambda", 0.25)))
        w, method = _optimise(mu, hist, ctx["rf"], float(cfg["portfolio"]["per_asset_cap"]))
        from pypfopt import risk_models

        cov = risk_models.CovarianceShrinkage(hist).ledoit_wolf().to_numpy()
        mu_p = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        sharpe = (mu_p - ctx["rf"]) / vol if vol else np.nan
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected return", f"{mu_p * 100:.2f}%")
        c2.metric("Volatility", f"{vol * 100:.2f}%")
        c3.metric("Sharpe", f"{sharpe:.3f}")
        st.caption(f"Solver: {method}. Bounds 0 to {cfg['portfolio']['per_asset_cap']:.0%}; "
                   f"risk-free rate {ctx['rf']:.2%}.")
        st.dataframe(pd.Series(w, index=tickers, name="weight").sort_values(ascending=False), width="stretch")


def risk_panel(ctx):
    st.title("Risk dashboard")
    rets = load_parquet(str(MODELS / "backtest_returns.parquet"))
    prices = load_parquet(str(DATA / "prices_clean.parquet"))
    if rets.empty:
        st.info("Run the portfolio stage first.")
        return
    r = rets.iloc[:, 0]
    curve = (1 + r).cumprod()
    dd = curve / curve.cummax() - 1
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy", line=dict(color="#dc2626")))
        fig.update_layout(title="Drawdown (%)", height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        vol = r.rolling(21).std() * np.sqrt(252) * 100
        vix = prices[prices["ticker"] == "^VIX"].set_index("date")["adj_close"].reindex(r.index).ffill()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vol.index, y=vol, name="21d realised vol"))
        fig.add_trace(go.Scatter(x=vix.index, y=vix, name="VIX"))
        fig.update_layout(title="Volatility vs VIX", height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Return correlation (test window)")
    wide = prices.pivot_table(index="date", columns="ticker", values="log_return").loc[r.index]
    panel = load_parquet(str(DATA / "panel.parquet"))
    eq_cols = [c for c in sorted(panel["ticker"].unique()) if c in wide.columns]
    corr = wide[eq_cols].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")

    from src.finmath import historical_var, parametric_var_t

    c1, c2, c3 = st.columns(3)
    c1.metric("Historical VaR 95%", f"{historical_var(r) * 100:.2f}%")
    c2.metric("Student-t VaR 95%", f"{parametric_var_t(r) * 100:.2f}%")
    c3.metric("Max drawdown", f"{dd.min() * 100:.2f}%")


def sentiment_panel(ctx):
    st.title("Sentiment")
    daily = load_parquet(str(DATA / "sentiment_daily.parquet"))
    scored = load_parquet(str(ROOT / "data" / "raw" / "news_scored.parquet"))
    if daily.empty:
        st.info("Run the sentiment stage first.")
        return
    ticker = ctx["ticker"]
    d = daily[daily["ticker"] == ticker].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["date"], y=d["sentiment"].rolling(21).mean(), name="21d mean sentiment"))
    fig.add_trace(go.Bar(x=d["date"], y=d["articles"], name="Articles", yaxis="y2", marker_color="#94a3b8", opacity=0.4))
    fig.update_layout(
        height=380, yaxis=dict(title="Polarity"), yaxis2=dict(title="Articles", overlaying="y", side="right"),
        margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig, width="stretch")

    if len(scored):
        s = scored[scored["ticker"] == ticker].copy()
        s["date"] = pd.to_datetime(s["session"])
        day = st.select_slider("Session", options=sorted(s["date"].dt.date.unique())[-30:])
        top = s[s["date"].dt.date == day].nlargest(8, "finbert_score")[["title", "finbert_label", "finbert_score"]]
        st.subheader(f"Top headlines on {day}")
        st.dataframe(top, width="stretch", hide_index=True)

    mood = daily.groupby("date")["market_mood"].mean()
    fig = px.line(mood, title="Market mood over time")
    fig.add_hline(y=0.1, line_dash="dot")
    fig.add_hline(y=-0.1, line_dash="dot")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    st.plotly_chart(fig, width="stretch")


def recommendations_panel(ctx):
    st.title("Recommendations")
    rec = load_parquet(str(DATA / "recommendations.parquet"))
    if len(rec) == 0:
        st.info("Run the recommendation stage first.")
        return
    meta = load_json(str(REPORTS / "recommendation_metrics.json"))
    tuned = meta.get("weights", {"forecast": 0.5, "sentiment": 0.2, "risk": 0.3})
    w1 = st.slider("Forecast weight", 0.0, 1.0, float(tuned.get("forecast", 0.5)), 0.05)
    w2 = st.slider("Sentiment weight", 0.0, 1.0, float(tuned.get("sentiment", 0.2)), 0.05)
    w3 = st.slider("Risk penalty", 0.0, 1.0, float(tuned.get("risk", 0.3)), 0.05)
    total = w1 + w2 + w3
    if total == 0:
        st.warning("Weights sum to zero.")
        return
    a, b, c = w1 / total, w2 / total, w3 / total
    df = rec.copy()
    df["composite"] = a * df["forecast_z"] + b * df["sentiment_z"] - c * df["risk_z"]
    df["decision"] = np.where(df["composite"] >= 0.5, "BUY", np.where(df["composite"] <= -0.5, "SELL", "HOLD"))
    df["reason"] = df.apply(
        lambda r: f"{r['decision']} - forecast {r['forecast_z']:+.2f}sd, sentiment {r['sentiment_z']:+.2f}sd, risk {r['risk_z']:+.2f}sd",
        axis=1,
    )
    st.subheader("Signals and the why")
    st.dataframe(
        df[["ticker", "decision", "forecast_z", "sentiment_z", "risk_z", "composite", "reason"]].sort_values(
            "composite", ascending=False
        ),
        width="stretch",
        hide_index=True,
    )
    st.subheader("Rebalancing plan")
    st.dataframe(
        df[["ticker", "current_weight", "target_weight", "rebalance_drift", "rebalance_action"]],
        width="stretch",
        hide_index=True,
    )
    bt = load_csv(str(REPORTS / "recommendation_backtest.csv"))
    if len(bt):
        st.subheader("Backtested hit rates (test window)")
        st.dataframe(bt, width="stretch", hide_index=True)
    st.caption("SELL means exit/avoid (long-only). Hit rate compares each call with the S&P 500 over the same 5 days.")


def main():
    ctx = sidebar()
    page = ctx["page"]
    if page == "Overview":
        overview(ctx)
    elif page == "Price & prediction":
        price_prediction(ctx)
    elif page == "Model comparison":
        model_comparison(ctx)
    elif page == "Portfolio analytics":
        portfolio_panel(ctx)
    elif page == "Risk dashboard":
        risk_panel(ctx)
    elif page == "Sentiment":
        sentiment_panel(ctx)
    elif page == "Recommendations":
        recommendations_panel(ctx)
    st.markdown(f'<div class="ng-disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
