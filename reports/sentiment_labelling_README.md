# Sentiment validation - labelling instructions

The PRD asks for hand-labelled headlines from our own corpus. Financial PhraseBank is excluded
because FinBERT was trained on it; labels must come from the text only, never from subsequent
price moves.

## What to do

1. Open `reports/sentiment_labelling.csv` in Excel or any CSV editor.
2. Fill the `human_label` column for each row with exactly one of:
   - `positive`
   - `negative`
   - `neutral`
3. Use `notes` only for ambiguous cases (optional).
4. Save as **CSV UTF-8**.
5. Run:

```bash
python scripts/evaluate_sentiment_labels.py
python scripts/make_report.py
```

The script prints accuracy, macro-F1 and the confusion matrices for FinBERT and VADER. The
report and `reports/sentiment_validation.json` are updated. Do **not** open
`reports/sentiment_labelling_models.csv` until you have finished: it holds the model predictions
and is kept separate so the labelling is blind.

## Protocol notes

- 200 headlines, stratified by ticker and year.
- If you want an intra-labeller agreement check, relabel 50 rows after a week and compute
  Cohen's kappa; this is optional.
- Do not label using the next-day return, price chart or model score. If a headline is genuinely
  ambiguous, mark it neutral and note why.
- A label you assign is a human label. If labels are ever produced by a model, the report marks
  them explicitly as an AI silver standard, not as human labels.
