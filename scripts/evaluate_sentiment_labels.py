"""Evaluate the hand-labelled sentiment sheet and print the FinBERT/VADER table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.sentiment import evaluate_labels
from src.utils import load_config, path_from_config

cfg = load_config()
reports = path_from_config(cfg, "reports")
sheet_path = reports / "sentiment_labelling.csv"
sheet = pd.read_csv(sheet_path)
labelled = sheet["human_label"].astype(str).str.lower().isin(["positive", "negative", "neutral"]).sum()
print(f"{len(sheet)} headlines, {labelled} labelled")
if labelled == 0:
    print("Fill the human_label column with positive, negative or neutral, then rerun.")
    raise SystemExit(0)
result = evaluate_labels(cfg)
print(json.dumps(result, indent=2))
print("\nNow regenerate the report: python scripts/make_report.py")
