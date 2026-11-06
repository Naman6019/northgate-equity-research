"""Build notebooks/02_sentiment_validation.ipynb (executed separately)."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "02_sentiment_validation.ipynb"

nb = nbf.v4.new_notebook()
cells = [
    nbf.v4.new_markdown_cell(
        "# 02 - Sentiment validation and session alignment\n\n"
        "The PRD asks for hand-labelled headlines from *our own* corpus. Financial PhraseBank is "
        "excluded because FinBERT was trained on it. Labels come from the text only, never from "
        "subsequent price moves. Fill the `human_label` column of "
        "`reports/sentiment_labelling.csv` with `positive`, `negative` or `neutral`; this notebook "
        "then reports accuracy, macro-F1 and confusion matrices for FinBERT and VADER."
    ),
    nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "root = Path.cwd()\n"
        "while not (root / 'src').exists() and root != root.parent:\n"
        "    root = root.parent\n"
        "sys.path.insert(0, str(root))\n"
        "\n"
        "import pandas as pd\n"
        "from src.utils import load_config, path_from_config\n"
        "from src.sentiment import evaluate_labels\n"
        "from src.clean import nyse_calendar\n"
        "from src.sentiment import session_dates\n"
        "\n"
        "cfg = load_config()\n"
        "sheet = pd.read_csv(path_from_config(cfg, 'reports') / 'sentiment_labelling.csv')\n"
        "models = pd.read_csv(path_from_config(cfg, 'reports') / 'sentiment_labelling_models.csv')\n"
        "sheet = sheet.merge(models, on='id', how='left')\n"
        "labelled = sheet['human_label'].astype(str).str.lower().isin(['positive', 'negative', 'neutral']).sum()\n"
        "print(f'{len(sheet)} headlines, {labelled} labelled')\n"
        "sheet.head(8)"
    ),
    nbf.v4.new_markdown_cell(
        "## Protocol\n\n"
        "- 200 headlines stratified by ticker and year.\n"
        "- Labelled from the text only; never from the next-day return (that would leak the target).\n"
        "- One person labelled every headline. The optional 50-headline re-label (Cohen's kappa) was not done.\n"
        "- VADER is a lexicon baseline; FinBERT is finance-tuned. Neither is assumed correct."
    ),
    nbf.v4.new_code_cell(
        "if labelled:\n"
        "    result = evaluate_labels(cfg)\n"
        "    display(result)\n"
        "else:\n"
        "    print('Labels are pending. Fill human_label in reports/sentiment_labelling.csv, then re-run.')\n"
        "    print('Until then the report states the validation status as pending, never as a result.')"
    ),
    nbf.v4.new_code_cell(
        "examples = sheet.sample(10, random_state=cfg['project']['seed'])[['ticker', 'timestamp', 'title', 'finbert_label', 'finbert_score', 'vader']]\n"
        "examples"
    ),
    nbf.v4.new_markdown_cell(
        "## Session-alignment checks\n\n"
        "A headline at 16:00 New York time or later, a date-only stamp and a weekend/holiday headline "
        "must all inform the next session, not the session that has already closed."
    ),
    nbf.v4.new_code_cell(
        "cal = nyse_calendar('2020-01-01', '2020-12-31')\n"
        "ts = pd.Series(pd.to_datetime([\n"
        "    '2020-01-10 20:59:00',  # Friday 15:59 ET\n"
        "    '2020-01-10 21:00:00',  # Friday 16:00 ET\n"
        "    '2020-01-11 15:00:00',  # Saturday\n"
        "    '2020-07-03 15:00:00',  # Independence Day observed\n"
        "    '2020-03-09 20:00:00',  # DST: Monday 16:00 EDT\n"
        "], utc=True))\n"
        "mapped = session_dates(ts, cal)\n"
        "pd.DataFrame({'utc': ts, 'session': mapped})"
    ),
    nbf.v4.new_markdown_cell(
        "The expected rows are Friday 10th, Monday 13th, Monday 13th, Monday 6th July and "
        "Tuesday 10th March. DST is handled by converting to America/New_York before applying the "
        "16:00 rule, not by shifting UTC by a fixed offset."
    ),
]
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print("wrote", OUT)
