"""
Classify each conversation's question side and answer side into
Resident / Business topic categories, and report the result three ways:
  - question only : category from the question's dominant topic
  - answer only   : category from the answer's dominant topic
  - both          : category assigned only when the two sides agree

Input (from topic_model.py):
  - topic_distribution.csv   per-conversation topic weights

Topic -> class map (derived from topic_keywords.csv):
  - question (user) side : Business if user_topic_1, else Resident
  - answer  (agent) side : Neutral  if agent_topic_12,
                           Business if agent_topic_1/6/8/10,
                           else Resident

Outputs:
  - conversation_labels.csv    every conversation: both sides + all 3 categories
  - uncategorized_topics.csv   conversations the "both" mode could not categorize,
                               with the topic each question / answer belongs to
  - classification_report.md   the three-way breakdown + agreement cross-tab
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
LDA_DIR = ROOT / "outputs" / "topic" / "lda"
LABELS_DIR = ROOT / "outputs" / "topic" / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)

DIST_CSV = LDA_DIR / "topic_distribution.csv"
OUT_LABELS = LABELS_DIR / "conversation_labels.csv"
OUT_UNCAT = LABELS_DIR / "uncategorized_topics.csv"
OUT_REPORT = ROOT / "reports" / "classification_report.md"

# --- topic -> class map ----------------------------------------------------
BUSINESS_TOPICS_USER = {"user_topic_1"}
BUSINESS_TOPICS_AGENT = {"agent_topic_1", "agent_topic_6",
                         "agent_topic_8", "agent_topic_10"}
NEUTRAL_TOPICS = {"agent_topic_12"}


def breakdown(series: pd.Series, n: int, order: list[str]) -> str:
    """Markdown table of value counts in a fixed label order."""
    vc = series.value_counts()
    lines = ["| category | conversations | share |", "|---|---|---|"]
    for label in order:
        v = int(vc.get(label, 0))
        lines.append(f"| {label} | {v:,} | {v / n * 100:.1f}% |")
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(DIST_CSV)
    user_cols = [c for c in df.columns if c.startswith("user_topic_")]
    agent_cols = [c for c in df.columns if c.startswith("agent_topic_")]
    n = len(df)

    # dominant topic per side = the topic column carrying the most weight
    df["dominant_user_topic"] = df[user_cols].idxmax(axis=1)
    df["dominant_agent_topic"] = df[agent_cols].idxmax(axis=1)

    # --- per-side class ----------------------------------------------------
    # question side: only user_topic_1 is Business, the rest Resident
    df["question_class"] = "Resident"
    df.loc[df["dominant_user_topic"].isin(BUSINESS_TOPICS_USER),
           "question_class"] = "Business"

    # answer side: Neutral / Business / Resident
    df["answer_class"] = "Resident"
    df.loc[df["dominant_agent_topic"].isin(BUSINESS_TOPICS_AGENT),
           "answer_class"] = "Business"
    df.loc[df["dominant_agent_topic"].isin(NEUTRAL_TOPICS),
           "answer_class"] = "Neutral"

    # --- three classification modes ---------------------------------------
    # 1. question only  -> just the question class
    # 2. answer only    -> just the answer class
    # 3. both           -> a category only when the two sides agree
    df["cat_question_only"] = df["question_class"]
    df["cat_answer_only"] = df["answer_class"]
    agree = (df["question_class"] == df["answer_class"]) & \
            df["question_class"].isin(["Resident", "Business"])
    df["cat_both"] = df["question_class"].where(agree, "Uncategorized")

    labels = df[["conversation_id", "n_turns",
                 "dominant_user_topic", "question_class",
                 "dominant_agent_topic", "answer_class",
                 "cat_question_only", "cat_answer_only", "cat_both"]]
    labels.to_csv(OUT_LABELS, index=False)

    uncat = labels.loc[labels["cat_both"] == "Uncategorized",
                       ["conversation_id", "n_turns",
                        "dominant_user_topic", "question_class",
                        "dominant_agent_topic", "answer_class"]]
    uncat.to_csv(OUT_UNCAT, index=False)

    # --- agreement cross-tab ----------------------------------------------
    xtab = pd.crosstab(df["question_class"], df["answer_class"])
    agreed = int(agree.sum())

    # --- write markdown report --------------------------------------------
    md = [
        "# Conversation Classification Report",
        "",
        f"Source: `{DIST_CSV.name}`  |  **{n:,} conversations**  |  "
        "topic-only classification (dominant LDA topic per side).",
        "",
        "Topic to class map: question Business = `user_topic_1`; "
        "answer Business = `agent_topic_1/6/8/10`; answer Neutral = "
        "`agent_topic_12`; everything else Resident.",
        "",
        "## Mode 1 — Question only",
        "Category taken from the question's dominant topic. "
        "Every conversation is categorized (no Neutral on the question side).",
        "",
        breakdown(df["cat_question_only"], n, ["Resident", "Business"]),
        "",
        "## Mode 2 — Answer only",
        "Category taken from the answer's dominant topic.",
        "",
        breakdown(df["cat_answer_only"], n, ["Resident", "Business", "Neutral"]),
        "",
        "## Mode 3 — Both (question and answer must agree)",
        "A final category is assigned only when the question class and the "
        "answer class are the same Resident/Business value; otherwise the "
        "conversation is Uncategorized.",
        "",
        breakdown(df["cat_both"], n, ["Resident", "Business", "Uncategorized"]),
        "",
        "## Question × Answer agreement",
        "Rows = question class, columns = answer class.",
        "",
        "| question \\ answer | " + " | ".join(xtab.columns) + " |",
        "|" + "---|" * (len(xtab.columns) + 1),
    ]
    for q_cls, r in xtab.iterrows():
        md.append(f"| {q_cls} | " + " | ".join(f"{int(v):,}" for v in r) + " |")
    md += [
        "",
        f"Agreement rate (question and answer give the same Resident/Business "
        f"label): **{agreed:,} / {n:,} = {agreed / n * 100:.1f}%**. "
        f"The remaining {n - agreed:,} ({(n - agreed) / n * 100:.1f}%) disagree "
        "or have a Neutral answer, so they stay Uncategorized in Mode 3.",
        "",
    ]
    OUT_REPORT.write_text("\n".join(md), encoding="utf-8")

    # --- console summary --------------------------------------------------
    print(f"{n:,} conversations\n")
    for mode, col, order in [
        ("MODE 1  question only", "cat_question_only", ["Resident", "Business"]),
        ("MODE 2  answer only", "cat_answer_only", ["Resident", "Business", "Neutral"]),
        ("MODE 3  both", "cat_both", ["Resident", "Business", "Uncategorized"]),
    ]:
        print(mode)
        vc = labels[col].value_counts()
        for label in order:
            v = int(vc.get(label, 0))
            print(f"  {label:<14s} {v:>7,}  ({v / n * 100:5.1f}%)")
        print()
    print(f"agreement rate: {agreed:,}/{n:,} = {agreed / n * 100:.1f}%")
    print(f"\nwrote {OUT_LABELS.name}  ({n:,} rows)")
    print(f"wrote {OUT_UNCAT.name}  ({len(uncat):,} rows)")
    print(f"wrote {OUT_REPORT.name}")


if __name__ == "__main__":
    main()
