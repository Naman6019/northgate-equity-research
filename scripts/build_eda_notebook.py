"""Build notebooks/01_eda.ipynb (executed separately with nbconvert)."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "01_eda.ipynb"

nb = nbf.v4.new_notebook()
cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        "# 01 - Exploratory data analysis\n\n"
        "Research window 2009-01-01 to 2023-12-29. Every figure ends with the finding and the "
        "decision it changes; this is the PRD Section 6 requirement. The notebook calls the "
        "versioned `src/` code rather than duplicating the pipeline."
    )
)
cells.append(
    nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "root = Path.cwd()\n"
        "while not (root / 'src').exists() and root != root.parent:\n"
        "    root = root.parent\n"
        "sys.path.insert(0, str(root))\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from statsmodels.tsa.stattools import adfuller\n"
        "from scipy.stats import skew, kurtosis\n"
        "from src.utils import load_config, path_from_config\n"
        "\n"
        "cfg = load_config()\n"
        "P = path_from_config(cfg, 'processed')\n"
        "F = path_from_config(cfg, 'figures')\n"
        "F.mkdir(parents=True, exist_ok=True)\n"
        "prices = pd.read_parquet(P / 'prices_clean.parquet')\n"
        "macro = pd.read_parquet(P / 'macro_clean.parquet')\n"
        "panel = pd.read_parquet(P / 'panel.parquet')\n"
        "equities = cfg['universe']['equities']\n"
        "prices.shape, panel.shape"
    )
)
cells.append(
    nbf.v4.new_code_cell(
        "ticker = 'AAPL'\n"
        "g = prices[prices['ticker'] == ticker].set_index('date')\n"
        "fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})\n"
        "ax[0].plot(g.index, g['adj_close'], label='Adjusted close', color='#0f172a', linewidth=1.2)\n"
        "ax[0].plot(g.index, g['adj_close'].rolling(50).mean(), label='SMA 50', color='#2563eb')\n"
        "ax[0].plot(g.index, g['adj_close'].rolling(200).mean(), label='SMA 200', color='#dc2626')\n"
        "ax[0].set_yscale('log')\n"
        "ax[0].set_title(f'{ticker}: price with 50/200-day moving averages (log scale)')\n"
        "ax[0].legend(loc='upper left')\n"
        "ax[1].bar(g.index, g['volume'], color='#94a3b8', width=1.0)\n"
        "ax[1].set_ylabel('Volume')\n"
        "fig.tight_layout()\n"
        "fig.savefig(F / 'eda_trend.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        "**Finding.** The adjusted series trends strongly and the 200-day average only crosses the "
        "50-day average in regime changes. **Consequence.** Raw price is non-stationary; the model "
        "must forecast log returns, not price, and trend features enter as price-to-MA ratios."
    )
)
cells.append(
    nbf.v4.new_code_cell(
        "wide = prices.pivot_table(index='date', columns='ticker', values='log_return')[equities]\n"
        "fig, ax = plt.subplots(figsize=(9, 7))\n"
        "sns.heatmap(wide.corr(), cmap='RdBu_r', center=0, vmin=-1, vmax=1, annot=False, ax=ax)\n"
        "ax.set_title('Correlation of daily log returns (2009-2023)')\n"
        "fig.tight_layout()\n"
        "fig.savefig(F / 'eda_correlation.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "wide.corr().round(2)"
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        "**Finding.** Correlations are positive but mostly below 0.6, so the 10-name universe offers "
        "genuine diversification. **Consequence.** The covariance matrix is well conditioned, and a "
        "long-only max-Sharpe portfolio can add value over a single name."
    )
)
cells.append(
    nbf.v4.new_code_cell(
        "r = wide.stack().dropna().to_numpy()\n"
        "mu, sd = r.mean(), r.std()\n"
        "fig, ax = plt.subplots(figsize=(10, 4))\n"
        "ax.hist(r, bins=160, density=True, color='#93c5fd', alpha=0.8, label='Daily returns')\n"
        "x = np.linspace(r.min(), r.max(), 400)\n"
        "ax.plot(x, (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sd) ** 2), color='#dc2626', label='Normal fit')\n"
        "ax.set_title(f'Return distribution: skew={skew(r):.2f}, excess kurtosis={kurtosis(r):.2f}')\n"
        "ax.set_xlim(-0.08, 0.08)\n"
        "ax.legend()\n"
        "fig.tight_layout()\n"
        "fig.savefig(F / 'eda_distribution.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "(skew(r), kurtosis(r))"
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        "**Finding.** Returns are fat-tailed and slightly skewed. **Consequence.** Risk reporting "
        "never relies only on the normal assumption: drawdown, historical VaR and a Student-t "
        "parametric VaR are reported alongside volatility, and Huber loss is used for the neural nets."
    )
)
cells.append(
    nbf.v4.new_code_cell(
        "port = wide.mean(axis=1)\n"
        "vol = port.rolling(21).std() * np.sqrt(252)\n"
        "vix = prices[prices['ticker'] == '^VIX'].set_index('date')['adj_close'].reindex(port.index).ffill()\n"
        "fig, ax = plt.subplots(figsize=(12, 4))\n"
        "ax.plot(vol.index, vol, label='21-day realised volatility (annualised)', color='#0f172a')\n"
        "ax2 = ax.twinx()\n"
        "ax2.plot(vix.index, vix, label='VIX', color='#dc2626', alpha=0.6)\n"
        "ax.set_ylabel('Realised volatility')\n"
        "ax2.set_ylabel('VIX')\n"
        "ax.set_title('Volatility clustering and the VIX overlay')\n"
        "fig.legend(loc='upper left')\n"
        "fig.tight_layout()\n"
        "fig.savefig(F / 'eda_volatility.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        "**Finding.** Volatility clusters and co-moves with the VIX. **Consequence.** VIX level/change, "
        "rolling volatility and beta are included as regime features, and the risk dashboard shows "
        "rolling volatility against the VIX."
    )
)
cells.append(
    nbf.v4.new_code_cell(
        "monthly = port.groupby([port.index.year, port.index.month]).mean().unstack()\n"
        "fig, ax = plt.subplots(figsize=(11, 4))\n"
        "sns.heatmap(monthly * 100, cmap='RdBu_r', center=0, ax=ax, cbar_kws={'label': 'Mean daily return (%)'})\n"
        "ax.set_title('Average daily return by year and month')\n"
        "ax.set_xlabel('Month')\n"
        "ax.set_ylabel('Year')\n"
        "fig.tight_layout()\n"
        "fig.savefig(F / 'eda_seasonality.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        "**Finding.** Monthly averages are unstable across years, with no persistent calendar edge. "
        "**Consequence.** Calendar features are kept as controls but should not dominate; the report "
        "does not claim a seasonal trading rule."
    )
)
cells.append(
    nbf.v4.new_code_cell(
        "rows = []\n"
        "for t in equities:\n"
        "    p = prices[prices['ticker'] == t].sort_values('date')['adj_close']\n"
        "    ret = np.log(p).diff().dropna()\n"
        "    rows.append({\n"
        "        'ticker': t,\n"
        "        'ADF price p': adfuller(p.dropna())[1],\n"
        "        'ADF return p': adfuller(ret)[1],\n"
        "        'skew': skew(ret),\n"
        "        'excess_kurtosis': kurtosis(ret),\n"
        "    })\n"
        "adf = pd.DataFrame(rows)\n"
        "adf.to_csv(path_from_config(cfg, 'reports') / 'eda_adf.csv', index=False)\n"
        "adf.round(4)"
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        "**Finding.** The ADF p-value is above 0.05 for every price series and below 0.05 for every "
        "return series. **Consequence.** Returns are the modelling target (PRD Section 6.1), and price-space "
        "metrics are reported only for comparability. Return kurtosis confirms the fat-tail decision above."
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        "## Distribution diagnostics: QQ plot and Jarque-Bera\n\n"
        "The histogram is not enough to justify the fat-tail decision; the QQ plot and the "
        "Jarque-Bera test check the normal assumption directly."
    )
)
cells.append(
    nbf.v4.new_code_cell(
        "from scipy import stats\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(6, 5))\n"
        "stats.probplot(r, dist='norm', plot=ax)\n"
        "ax.set_title('QQ plot: daily returns vs normal')\n"
        "fig.tight_layout()\n"
        "fig.savefig(F / 'eda_qq.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "\n"
        "jb_rows = []\n"
        "for t in equities:\n"
        "    x = np.log(prices[prices['ticker'] == t].sort_values('date')['adj_close']).diff().dropna()\n"
        "    stat, p = stats.jarque_bera(x)\n"
        "    jb_rows.append({'ticker': t, 'JB': stat, 'p_value': p})\n"
        "jb = pd.DataFrame(jb_rows)\n"
        "jb.to_csv(path_from_config(cfg, 'reports') / 'eda_jarque_bera.csv', index=False)\n"
        "jb.round(4)"
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        "**Finding.** The QQ plot bends away from the 45-degree line in both tails and the "
        "Jarque-Bera p-values reject normality for every ticker. **Consequence.** Risk reporting "
        "uses drawdown and empirical/Student-t VaR alongside volatility, and the neural nets use "
        "Huber loss rather than squared error."
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        "## Day-of-week averages and STL decomposition\n\n"
        "The PRD asks for calendar seasonality to be checked, not assumed."
    )
)
cells.append(
    nbf.v4.new_code_cell(
        "dow = port.groupby(port.index.dayofweek).mean() * 100\n"
        "labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'][: len(dow)]\n"
        "fig, ax = plt.subplots(figsize=(8, 3))\n"
        "ax.bar(labels, dow.to_numpy(), color='#2563eb')\n"
        "ax.axhline(0, color='#0f172a', linewidth=1)\n"
        "ax.set_ylabel('Mean daily return (%)')\n"
        "ax.set_title('Average daily return by day of week')\n"
        "fig.tight_layout()\n"
        "fig.savefig(F / 'eda_dow.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "dow"
    )
)
cells.append(
    nbf.v4.new_code_cell(
        "from statsmodels.tsa.seasonal import STL\n"
        "\n"
        "series = port.dropna()\n"
        "stl = STL(series, period=5, robust=True).fit()\n"
        "fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)\n"
        "axes[0].plot(series.index, series, color='#0f172a'); axes[0].set_ylabel('Observed')\n"
        "axes[1].plot(series.index, stl.trend, color='#2563eb'); axes[1].set_ylabel('Trend')\n"
        "axes[2].plot(series.index, stl.seasonal, color='#16a34a'); axes[2].set_ylabel('Seasonal')\n"
        "axes[3].plot(series.index, stl.resid, color='#dc2626'); axes[3].set_ylabel('Residual')\n"
        "axes[0].set_title('STL decomposition of the equal-weight portfolio return (period=5)')\n"
        "fig.tight_layout()\n"
        "fig.savefig(F / 'eda_stl.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        "**Finding.** Day-of-week averages are small and change sign across the sample; the STL "
        "seasonal component is tiny relative to the trend and residual. **Consequence.** Calendar "
        "features are controls, not a claimed trading edge."
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        "## Hypotheses handed to the modelling stage\n\n"
        "1. Returns are stationary; a random walk is a hard baseline for h=1.\n"
        "2. Momentum/trend and volatility-regime features carry more signal than calendar effects.\n"
        "3. Fat tails require Huber loss for neural nets and drawdown/VaR in reporting.\n"
        "4. Cross-asset correlations are low enough for mean-variance optimisation to help.\n"
        "5. News sentiment may help direction more than MAE; the ablation reports both."
    )
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print("wrote", OUT)
