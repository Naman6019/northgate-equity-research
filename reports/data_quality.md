# Northgate data-quality report

Generated for the research window **2009-01-01 -> 2023-12-29**.

## Price panel

- Vendor rows in: **75,480**; rows out on the NYSE calendar: **75,480**.
- Short gaps forward-filled (limit 3): **0**.
- OHLC invariant violations quarantined: **0**.
- Return outliers flagged (z > 5.0 or 1.5x IQR): **4,337**; all kept.
- Large moves kept (|return| >= 20%, likely earnings/corporate actions): **94**.
- No value was winsorised; no price series was backward-filled.

| Ticker | Rows in | Rows out | Gaps filled | Invariant violations | Outliers flagged | Large moves kept |
|---|---:|---:|---:|---:|---:|---:|
| AAPL | 3,774 | 3,774 | 0 | 0 | 183 | 0 |
| CAT | 3,774 | 3,774 | 0 | 0 | 230 | 0 |
| HD | 3,774 | 3,774 | 0 | 0 | 189 | 1 |
| JNJ | 3,774 | 3,774 | 0 | 0 | 189 | 0 |
| JPM | 3,774 | 3,774 | 0 | 0 | 259 | 4 |
| KO | 3,774 | 3,774 | 0 | 0 | 199 | 0 |
| MSFT | 3,774 | 3,774 | 0 | 0 | 196 | 0 |
| NVDA | 3,774 | 3,774 | 0 | 0 | 198 | 4 |
| PG | 3,774 | 3,774 | 0 | 0 | 201 | 0 |
| SPY | 3,774 | 3,774 | 0 | 0 | 281 | 0 |
| XLE | 3,774 | 3,774 | 0 | 0 | 219 | 1 |
| XLF | 3,774 | 3,774 | 0 | 0 | 241 | 0 |
| XLI | 3,774 | 3,774 | 0 | 0 | 221 | 0 |
| XLK | 3,774 | 3,774 | 0 | 0 | 255 | 0 |
| XLP | 3,774 | 3,774 | 0 | 0 | 174 | 0 |
| XLV | 3,774 | 3,774 | 0 | 0 | 169 | 0 |
| XLY | 3,774 | 3,774 | 0 | 0 | 240 | 0 |
| XOM | 3,774 | 3,774 | 0 | 0 | 224 | 0 |
| ^GSPC | 3,774 | 3,774 | 0 | 0 | 286 | 0 |
| ^VIX | 3,774 | 3,774 | 0 | 0 | 183 | 84 |

## Macro release lags

A macro value is shifted by a conservative publication lag before it is
merged onto the trading calendar, so the model never sees a figure before
it was published.

| Series | Lag | Observations | First reference | Last reference |
|---|---:|---:|---|---|
| DGS10 | 1 business_day(s) | 3,752 | 2009-01-02 | 2023-12-29 |
| DGS3MO | 1 business_day(s) | 3,752 | 2009-01-02 | 2023-12-29 |
| CPIAUCSL | 45 calendar_day(s) | 180 | 2009-01-01 | 2023-12-01 |
| UNRATE | 35 calendar_day(s) | 180 | 2009-01-01 | 2023-12-01 |

## News coverage (FNSPID)

| Ticker | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AAPL | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 472 | 0 | 3018 | 5651 |
| CAT | 2 | 58 | 183 | 550 | 506 | 494 | 606 | 425 | 638 | 1082 | 685 | 429 | 356 | 490 | 903 |
| HD | 61 | 271 | 244 | 264 | 245 | 263 | 217 | 197 | 227 | 775 | 1158 | 493 | 0 | 0 | 0 |
| JNJ | 0 | 0 | 147 | 289 | 273 | 331 | 287 | 316 | 300 | 314 | 432 | 231 | 0 | 0 | 0 |
| JPM | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 497 | 1699 | 655 | 0 | 0 | 0 |
| KO | 40 | 337 | 317 | 459 | 540 | 929 | 786 | 913 | 878 | 777 | 797 | 816 | 545 | 952 | 1278 |
| MSFT | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2606 | 5910 |
| NVDA | 0 | 0 | 224 | 179 | 144 | 118 | 180 | 336 | 614 | 526 | 507 | 309 | 702 | 2332 | 5537 |
| PG | 0 | 0 | 0 | 0 | 96 | 202 | 154 | 210 | 234 | 178 | 146 | 92 | 0 | 0 | 0 |
| XOM | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 101 | 542 | 1492 | 1173 | 1902 | 1941 |

Coverage is uneven: some tickers have no headlines before 2018 and four
(HD, JNJ, JPM, PG) stop in 2020. The sentiment ablation is therefore run
on the well-covered names and every row reports coverage explicitly, as
the implementation plan's fallback requires.

## Outputs

- `data/processed/prices_clean.parquet`
- `data/processed/macro_clean.parquet`
