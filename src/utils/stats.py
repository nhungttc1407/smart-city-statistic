"""
Descriptive statistics for the English question / answer text.

Reads the raw engagement CSV and reports, for user_message_en (questions)
and agent_response_en (answers):
  - row / conversation counts, missing & empty text
  - turns per conversation
  - length distributions (characters and words)
  - thumbs up / down feedback rates
  - most frequent words (English stopwords removed)

Writes text_stats_summary.csv and prints a report.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
SRC = HERE / "Engdata(usage_binatanya_01-03-25_16-11-).csv"
OUT_CSV = HERE / "text_stats_summary.csv"

PCTILES = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
WORD_RE = re.compile(r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
# same throwaway tokens as topic_model.py
STOP = set("""
a an the and or but if of to in on at for with from by as is are was were be been
being this that these those it its he she they we you i my your our their them
not no do does did have has had can could will would should may might must
about into out up down over under again then once here there when where why how
all any both each few more most other some such only own same so than too very
href https http target blank www il aspx muni muninetanya pages html br li ul netanya
""".split())


def desc(series: pd.Series, name: str) -> dict:
    s = series.dropna()
    row = {"metric": name, "count": int(s.count()), "mean": round(s.mean(), 1),
           "std": round(s.std(), 1), "min": int(s.min()), "max": int(s.max())}
    for p in PCTILES:
        row[f"p{int(p * 100)}"] = round(s.quantile(p), 1)
    return row


def top_words(texts: pd.Series, n: int = 25) -> list[tuple[str, int]]:
    c: Counter[str] = Counter()
    for t in texts.dropna():
        c.update(w for w in WORD_RE.findall(t.lower()) if w not in STOP and len(w) > 2)
    return c.most_common(n)


def main() -> None:
    print(f"Reading {SRC.name} ...")
    df = pd.read_csv(SRC, encoding="utf-8", low_memory=False)
    print(f"  {len(df):,} rows (turns), {df['conversation_id'].nunique():,} conversations\n")

    q = df["user_message_en"]
    a = df["agent_response_en"]

    # --- coverage --------------------------------------------------------
    print("=== COVERAGE ===")
    for label, col in [("question (user_message_en)", q), ("answer (agent_response_en)", a)]:
        missing = col.isna().sum()
        empty = (col.fillna("").astype(str).str.strip() == "").sum()
        print(f"  {label:32s} missing={missing:>7,}  empty={empty:>7,}  "
              f"usable={len(col) - empty:>7,}")

    # --- turns per conversation -----------------------------------------
    turns = df.groupby("conversation_id")["conversation_turn"].count()
    print(f"\n=== TURNS PER CONVERSATION ===")
    print(f"  mean={turns.mean():.2f}  median={turns.median():.0f}  "
          f"max={turns.max()}  1-turn convos={(turns == 1).sum():,}")

    # --- length distributions -------------------------------------------
    qs = q.fillna("").astype(str)
    as_ = a.fillna("").astype(str)
    rows = [
        desc(qs.str.len(), "question_chars"),
        desc(qs.str.split().str.len(), "question_words"),
        desc(as_.str.len(), "answer_chars"),
        desc(as_.str.split().str.len(), "answer_words"),
    ]
    summary = pd.DataFrame(rows)
    print(f"\n=== LENGTH DISTRIBUTIONS ===")
    print(summary.to_string(index=False))

    # --- feedback --------------------------------------------------------
    if "thumbs_up" in df.columns:
        up = df["thumbs_up"].astype(str).str.upper().eq("TRUE").sum()
        down = df["thumbs_down"].astype(str).str.upper().eq("TRUE").sum()
        rated = up + down
        print(f"\n=== FEEDBACK ===")
        print(f"  thumbs_up={up:,}  thumbs_down={down:,}  "
              f"rated={rated:,} ({rated / len(df) * 100:.1f}% of turns)"
              + (f"  satisfaction={up / rated * 100:.1f}%" if rated else ""))

    # --- top words -------------------------------------------------------
    print(f"\n=== TOP WORDS — QUESTIONS ===")
    print("  " + ", ".join(f"{w}({n})" for w, n in top_words(q)))
    print(f"\n=== TOP WORDS — ANSWERS ===")
    print("  " + ", ".join(f"{w}({n})" for w, n in top_words(a)))

    summary.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV.name}")


if __name__ == "__main__":
    main()
