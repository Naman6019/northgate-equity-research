# Northgate Quantitative Research - Stock Prediction & Portfolio Optimisation

An end-to-end, leakage-safe research platform for ten large-cap US equities. It forecasts
next-day (h=1) and 5-day (h=5) log returns with four classical ML models and four PyTorch
sequence models, benchmarks every one against a random walk, scores financial-news sentiment
with FinBERT and VADER under strict trading-session alignment, and turns the forecasts into a
Ledoit-Wolf max-Sharpe portfolio with transparent BUY/HOLD/SELL and rebalancing guidance.

> **Educational research and decision-support prototype. Nothing here is financial advice.**
> The evaluation is out-of-time on a single held-out window; the test set is scored once.
> Negative results (a model that does not beat the random walk, or a portfolio that does not
> beat equal-weight) are reported honestly in `reports/final_report.pdf`.

> **About this project.** Built during an AI internship at Internmo from a client-style brief.
> Northgate Quantitative Research is a fictional client in that brief, and the brief itself isn't
> included here. The scored-headline file (`data/raw/news_scored.parquet`) isn't included either,
> because it holds the full article text: `python -m src.pipeline all` rebuilds it from FNSPID,
> and until then the dashboard's headline table stays empty.

---

## Results at a glance (out-of-time test, 2021-11-09 to 2023-12-28)

| Item | Result | How to read it |
|---|---|---|
| h=5 best test MAE | **GRU 0.03034** vs random walk 0.03039 | The GRU beats the random walk by 5e-5; the margin is within noise. |
| h=5 validation-selected | **Transformer** (val MAE 0.03024, test MAE 0.03044) | It does not beat the random walk on test MAE by 5e-5; selection stays on validation. |
| h=1 best test MAE | **LSTM 0.01317** vs random walk 0.01319 | The sequence models edge the random walk, again by a negligible margin. |
| Direction | h=5 winner 53.3% vs always-up 53.1% | Technically above, but under 0.2 pp and within noise; not a meaningful edge. |
| Portfolio | optimised net Sharpe **0.014** vs equal-weight **0.286** | The PRD acceptance bar is **not met**; CIs overlap and the discussion is in the report. |
| Sentiment | FinBERT ΔMAE +0.00012, direction/IC slightly negative; VADER tiny positive; tuned weight 0.2 | No material improvement; reported as a negative result. |
| Rank-IC (N3) | mean 0.023, t = 1.57, IC IR 0.068 | Positive but weak on a ten-name cross-section. |

---

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

# GPU (recommended; CUDA 12.4 wheels)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# CPU-only alternative: .../whl/cpu

pip install -r requirements.txt

# Rebuild everything (downloads prices, FRED macro and FNSPID headlines if not cached)
python -m src.pipeline all

# Rebuild from the cached raw Parquet/Sentiment cache (no network)
python -m src.pipeline all --offline

# Dashboard (reads cached outputs; never retrains on load)
streamlit run dashboard/app.py

# Report (Markdown + PDF)
python scripts/make_report.py
```

Individual stages:

```bash
python -m src.pipeline data [--offline]
python -m src.pipeline train
python -m src.ingest --offline
python -m src.sentiment
python -m src.models_ml --horizon 5 1 --walk-forward
python -m src.models_dl --horizon 5 1
python -m src.portfolio
python -m src.recommend
```

---

## Research window and data

- **Window:** 2009-01-01 to 2023-12-29 (NYSE calendar, 3,774 sessions).
- **Universe:** AAPL, MSFT, NVDA, JPM, XOM, JNJ, PG, KO, CAT, HD. ^GSPC is the benchmark and
  market factor; SPY gives a dividend-inclusive buy-and-hold comparison; ^VIX and seven sector
  ETFs are context features.
- **Prices:** Yahoo Finance via `yfinance` (OHLCV + adjusted close, which is mandatory for returns).
- **Macro:** FRED public CSV (DGS10, DGS3MO, CPIAUCSL, UNRATE). Release lags are applied before
  merging: CPI 45 calendar days, unemployment 35 days, daily yields one business day.
- **News:** FNSPID ticker-tagged headlines via the public parquet mirror
  (`2084Collective/FNSPID_IMPROVED`, 1.25 GB, six parts), filtered to the universe and window.
  The original `All_external.csv` stops in mid-2020, so the mirror is used and credited.
  Coverage is uneven: HD, JNJ, JPM and PG stop in 2020; AAPL, MSFT, NVDA, CAT, KO and XOM cover
  the test window. `reports/news_coverage.csv` reports it per ticker and year.

The split is fixed before modelling:

| Partition | Sessions | Rows | Purpose |
|---|---|---:|---|
| Train | 2009-10 to 2019-09 | 24,960 | tuning (TimeSeriesSplit, gap=h) and fitting |
| Validation | 2019-09 to 2021-11 | 5,310 | model selection, lambda, recommendation weights |
| Test (once) | 2021-11 to 2023-12 | 5,370 | final reporting |

A five-session embargo at each boundary keeps every training label before the validation start
and every validation label before the test start.

---

## Architecture (PRD Section 15.1)

```
L1 ingest.py      yfinance OHLCV+Adj Close, FRED macro, FNSPID headlines -> data/raw/*.parquet
L2 clean.py       NYSE align -> dedup -> ffill<=3 (never backfill) -> OHLC quarantine
   features.py    returns/trend/momentum/volatility/volume/calendar/macro/market features,
                  h=1 and h=5 targets, chronological split, leak-free sequences
L3 models_ml.py   Ridge, Random Forest, XGBoost, SVR (TimeSeriesSplit gap=h)
   models_dl.py   LSTM, GRU, BiLSTM, Transformer (PyTorch, Huber, early stopping)
L4 sentiment.py   clean -> FinBERT + VADER (sha1 cache) -> session alignment -> daily features
L5 portfolio.py   Ledoit-Wolf covariance, mu blend, max-Sharpe (<=30%), frontier, 20k Monte Carlo,
                  walk-forward backtest with 10 bps costs, PSR/Deflated Sharpe
L6 recommend.py   composite z-score -> BUY/HOLD/SELL, rebalancing table, hit-rate backtest,
                  XGBRanker cross-sectional rank-IC diagnostic
L7 dashboard/app.py  seven cached panels, Plotly, persistent disclaimer
```

---

## Repository layout

```
Northgate/
├── config.yaml                 universe, dates, split, horizons, windows, costs, caps, weights
├── requirements.txt            pinned direct dependencies
├── src/                        ingest, clean, features, finmath, evaluate, models_ml,
│                               models_dl, sentiment, portfolio, recommend, pipeline, utils
├── dashboard/app.py            seven-panel Streamlit app
├── notebooks/                  01_eda (executed, captioned), 02_sentiment_validation
├── models/                     joblib/.pt artifacts, model cards, predictions, portfolio.json
├── reports/                    final_report.md/.pdf, data_quality.md, experiments.csv, figures
├── data/raw/, data/processed/  raw Parquet + cleaned panels (git-ignored where large)
├── scripts/                    notebook builders and report generator
├── tests/                      leakage, maths, cleaning, API-free unit tests
└── .github/workflows/ci.yml    ruff + pytest on every push
```

---

## Evaluation and honest results

Every model is compared on the **same** held-out window with the **same** features. Metrics are
reported in return space (RMSE, MAE, information coefficient, directional accuracy) next to
price space (RMSE, MAE, MAPE, R²). Price-space R² is around 0.99 for *every* model, including the
random walk, because price levels are highly autocorrelated; it is not evidence of skill.

The report states which claims fail:

- **Forecasting:** the validation-selected h=5 Transformer does not beat the random walk on test
  MAE (0.03044 vs 0.03039); the GRU test winner does by 5e-5, but selecting it after seeing the
  test set would be test-set selection. Direction is 53.3% vs always-up 53.1%, i.e. at the base
  rate. Both facts are reported.
- **Sentiment:** FinBERT changes test MAE by about +1e-4 (economically negligible) with slightly
  negative direction/IC; VADER is tiny-positive. Validation chose a sentiment weight of 0.2, but
  the effect is within noise. Reported as a negative result.
- **Portfolio:** the optimised portfolio's test Sharpe is 0.014 vs 0.286 for equal-weight. The 30%
  cap, noisy forecast-based mu, and minimum-variance fallbacks when the risk-free rate exceeds
  expected returns all contribute. Confidence intervals are reported and overlap.
- **Recommendations:** the BUY/SELL hit rate is 54.0% vs 51.4% always-buy on the test window — above
  the base rate, but one window and not a tested strategy; the rank-IC is positive but weak.
- **Rank-IC (N3):** positive mean daily IC (0.023) with t = 1.57; not significant enough to
  claim a trading edge.

These outcomes are consistent with the PRD's grading philosophy: a correct, leakage-free result
with honest analysis is worth more than an impressive-looking result built on a subtle bug.

---

## Testing and quality

```bash
ruff check .
pytest -q
pytest -q --cov=src --cov-report=term-missing
```

Tests cover:

- from-scratch financial maths against NumPy/pandas/SciPy, including GD = OLS to 1e-4;
- no back-fill, limited forward-fill, OHLC quarantine, retained outlier flags;
- macro release lags (CPI at 45 days, yields at one business day);
- point-in-time features (changing future prices cannot change a feature at t), split embargo,
  train-only scalers, and no sequence crossing a split;
- session alignment (Friday 15:59 vs 16:00, weekends, US holidays, DST, date-only stamps);
- evaluation metrics, random-walk baselines and bootstrap intervals;
- portfolio bounds, one-day execution lag and the Monte Carlo cross-check;
- recommendation rules, the no-trade band and the rebalancing band;
- model plumbing and PyTorch output shapes.

CI runs `ruff` and the test suite on every push. Model training and FinBERT scoring are not run
in CI; they are reproducible from the cached raw data with the commands above.

---

## Limitations

- Survivorship bias: the universe is today's large caps. Equal-weight carries the same bias, which
  is why it is the fair comparison.
- One test window (2021-11 to 2023-12): regime-dependent conclusions. The 2022-2023 part was also seen in the earlier 2015-2023 study; only 2021-11 to 2022-10 is new, so the extension is exploratory for the shared subperiod.
- A refit-on-train+validation and validation-selected ensemble experiment was also run on the earlier window; it did not improve test MAE and was not adopted (git history).
- News coverage is uneven after 2020; the sentiment ablation is reported on the well-covered names.
- Sentiment validation rests on 200 headlines labelled by one person (FinBERT 67.0%, VADER 44.5%
  accuracy). The optional 50-row relabel was not done, so label noise is unmeasured.
- Flat 10 bps/side costs; no market impact, borrow cost or taxes; long-only.
- No live mode: post-2023 data is not mixed into the evaluated results.
- No claim of profitability.

---

## Attribution

- Prices: Yahoo Finance via `yfinance`.
- Macro: Federal Reserve Economic Data (FRED), public CSV endpoint.
- News: FNSPID (Dong, Fan & Peng, 2024), CC BY-NC 4.0; parquet mirror by 2084Collective.
- FinBERT: ProsusAI/finbert on Hugging Face.
- Code: MIT, see [LICENSE](LICENSE).
