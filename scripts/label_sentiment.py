"""Hand-label the blind sentiment sheet, one keypress per headline. Saves after every key.

Only the headline text is shown, which is all FinBERT and VADER see. This script never reads
sentiment_labelling_models.csv. Do not run evaluate_sentiment_labels.py until every row is done.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import msvcrt
except ImportError:
    msvcrt = None

from src.utils import load_config, path_from_config

LABELS = {"1": "negative", "2": "neutral", "3": "positive"}
RUBRIC = """
Label the headline's own wording. Do not look up prices, dates or what happened next.
Judge whoever the headline is about, even if that is not the tagged company or stock.

  1 negative  miss, downgrade, price target cut, fine, lawsuit, falling shares or markets, explicit avoid/sell
  2 neutral   facts with no direction, schedules, previews, questions, "to watch" lists, fund profiles,
              "movers" with no direction, mixed news where neither side clearly wins
  3 positive  beat, upgrade, price target raise, approval, rising shares or markets, explicit buy call

Unsure after 10 seconds? Press f to flag it, then 1, 2 or 3 to label it. Only a label moves you on.
Keys: 1 2 3 label | f flag unsure (press again to remove) | b back | q save and quit
"""


def read_key() -> str:
    if msvcrt and sys.stdin.isatty():
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):  # arrow and function keys arrive as two characters
            msvcrt.getwch()
            return ""
        return "q" if ch == "\x03" else ch.lower()
    line = sys.stdin.readline()
    return line.strip().lower()[:1] if line else "q"


def load_sheet(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def save_sheet(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            print(f"Cannot write {path.name}; close it in Excel or wait for OneDrive, then press any key.")
            read_key()


def next_open(rows: list[dict[str, str]], start: int = 0) -> int | None:
    """Next unlabelled row at or after start, wrapping to any earlier gap."""
    order = list(range(start, len(rows))) + list(range(start))
    return next((i for i in order if not rows[i]["human_label"]), None)


def label(path: Path) -> int:
    fields, rows = load_sheet(path)
    print(RUBRIC)
    i = next_open(rows)
    shown = None
    while i is not None:
        row = rows[i]
        if i != shown:
            tags = ", ".join(x for x in (row["human_label"], row["notes"]) if x)
            print(f"\n[{i + 1}/{len(rows)}]" + (f" (now: {tags})" if tags else ""))
            print(f"  {row['title']}")
            shown = i
        key = read_key()
        if key in LABELS:
            row["human_label"] = LABELS[key]
            save_sheet(path, fields, rows)
            print(f"  -> {row['human_label']}")
            i = next_open(rows, i + 1)
        elif key == "f":
            row["notes"] = "" if row["notes"] == "unsure" else "unsure"
            save_sheet(path, fields, rows)
            print("  flagged unsure; now press 1, 2 or 3 to label it" if row["notes"] else "  flag removed; press 1, 2 or 3 to label it")
        elif key == "b":
            if i == 0:
                print("  already at the first headline")
            i = max(i - 1, 0)
        elif key == "q":
            break
        else:
            print("  not a key this script uses; press 1, 2, 3, f, b or q")
    done = sum(1 for r in rows if r["human_label"])
    flagged = sum(1 for r in rows if r["notes"] == "unsure")
    print(f"\n{done}/{len(rows)} labelled, {flagged} flagged unsure. Saved to {path}")
    if done == len(rows):
        print("Done. Now run: python scripts/evaluate_sentiment_labels.py")
    else:
        print("Run this script again to resume.")
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else path_from_config(load_config(), "reports") / "sentiment_labelling.csv"
    return label(path)


if __name__ == "__main__":
    raise SystemExit(main())
