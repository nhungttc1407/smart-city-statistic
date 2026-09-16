"""
Sector classification of each conversation using a human-curated map from
user_topic_* / agent_topic_* to Resident / Business / Mixed sectors.

Reads topic_distribution.csv (from topic_model.py: per-conversation weights
over k=4 user topics and k=15 agent topics).

Three modes, mirroring classify.py:
  - question only : sector taken from the dominant user_topic
  - answer only   : sector taken from the dominant agent_topic
  - both          : sector assigned only when the question-side sector and
                    the answer-side sector are the same Resident/Business/
                    Mixed value; otherwise Uncategorized.

Outputs:
  - sector_labels_qa.csv  per-conversation labels + dominant topics
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
LDA_DIR = ROOT / "outputs" / "topic" / "lda"
LABELS_DIR = ROOT / "outputs" / "topic" / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)

DIST_CSV = LDA_DIR / "topic_distribution.csv"
OUT_LABELS = LABELS_DIR / "sector_labels_qa.csv"

USER_TOPIC_SECTOR = {
    0: "Resident",
    1: "Mixed",
    2: "Resident",
    3: "Resident",
}

AGENT_TOPIC_SECTOR = {
    0:  "Resident",
    1:  "Business",
    2:  "Resident",
    3:  "Mixed",
    4:  "Resident",
    5:  "Resident",
    6:  "Mixed",
    7:  "Resident",
    8:  "Business",
    9:  "Resident",
    10: "Business",
    11: "Resident",
    12: "Mixed",
    13: "Resident",
    14: "Resident",
}


def _topic_int(col: str) -> int:
    return int(col.rsplit("_", 1)[-1])


def main() -> None:
    df = pd.read_csv(DIST_CSV)
    user_cols = [c for c in df.columns if c.startswith("user_topic_")]
    agent_cols = [c for c in df.columns if c.startswith("agent_topic_")]
    n = len(df)

    df["dominant_user_topic"] = df[user_cols].idxmax(axis=1)
    df["dominant_agent_topic"] = df[agent_cols].idxmax(axis=1)

    df["question_sector"] = df["dominant_user_topic"].map(
        lambda c: USER_TOPIC_SECTOR[_topic_int(c)]
    )
    df["answer_sector"] = df["dominant_agent_topic"].map(
        lambda c: AGENT_TOPIC_SECTOR[_topic_int(c)]
    )

    df["sector_question_only"] = df["question_sector"]
    df["sector_answer_only"] = df["answer_sector"]
    df["sector_both"] = df["question_sector"].where(
        df["question_sector"] == df["answer_sector"], "Uncategorized"
    )

    out = df[[
        "conversation_id", "n_turns",
        "dominant_user_topic", "question_sector",
        "dominant_agent_topic", "answer_sector",
        "sector_question_only", "sector_answer_only", "sector_both",
    ]]
    out.to_csv(OUT_LABELS, index=False)
    print(f"wrote {OUT_LABELS.name}  ({n:,} rows)")

    for mode, col, order in [
        ("MODE 1  question only", "sector_question_only",
         ["Resident", "Business", "Mixed"]),
        ("MODE 2  answer only", "sector_answer_only",
         ["Resident", "Business", "Mixed"]),
        ("MODE 3  both must agree", "sector_both",
         ["Resident", "Business", "Mixed", "Uncategorized"]),
    ]:
        print(f"\n{mode}")
        vc = out[col].value_counts()
        for label in order:
            v = int(vc.get(label, 0))
            print(f"  {label:<14s} {v:>7,}  ({v / n * 100:5.1f}%)")

    agree = (df["question_sector"] == df["answer_sector"])
    print(f"\nagreement rate: {int(agree.sum()):,}/{n:,} = "
          f"{agree.sum() / n * 100:.1f}%")


if __name__ == "__main__":
    main()
