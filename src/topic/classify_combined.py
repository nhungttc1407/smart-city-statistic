"""
Classify each conversation as Resident, Business, or Mixed by aggregating
the 16 combined-topic weights from topic_distribution_combined.csv under a
human-curated topic -> category map.

For each conversation:
    resident_weight = sum(p[t] for t in RESIDENT_TOPICS)
    business_weight = sum(p[t] for t in BUSINESS_TOPICS)
    mixed_weight    = sum(p[t] for t in MIXED_TOPICS)
    label           = argmax(resident_weight, business_weight, mixed_weight)
    dominant_topic  = argmax(all 16 topic weights)
    dominant_label  = category of dominant_topic
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
LDA_DIR = ROOT / "outputs" / "topic" / "lda"
LABELS_DIR = ROOT / "outputs" / "topic" / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)

DIST_CSV = LDA_DIR / "topic_distribution_combined.csv"
OUT_CSV = LABELS_DIR / "conversation_labels_combined.csv"

RESIDENT_TOPICS = [0, 1, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]
BUSINESS_TOPICS = [7]
MIXED_TOPICS = [2, 8, 9]

TOPIC_NAMES = {
    0:  "Parking ticket / appeal",
    1:  "Street, shelter, waste",
    2:  "Building permit + business licensing",
    3:  "Property tax discount (senior/disabled)",
    4:  "Parking permit (resident)",
    5:  "Property tax payment",
    6:  "Parks & gardens",
    7:  "Tenders & procurement",
    8:  "Contact / department info",
    9:  "General municipal services",
    10: "Beach & events",
    11: "Hotline & resident services",
    12: "Senior citizens & welfare",
    13: "Events & community classes",
    14: "Property tax forms",
    15: "School & kindergarten registration",
}


def main() -> None:
    df = pd.read_csv(DIST_CSV)
    topic_cols = [f"combined_topic_{i}" for i in range(16)]
    P = df[topic_cols].to_numpy()

    res_w = P[:, RESIDENT_TOPICS].sum(axis=1)
    biz_w = P[:, BUSINESS_TOPICS].sum(axis=1)
    mix_w = P[:, MIXED_TOPICS].sum(axis=1)

    weights = pd.DataFrame({
        "Resident": res_w, "Business": biz_w, "Mixed": mix_w,
    })
    label = weights.idxmax(axis=1)

    dom_topic = P.argmax(axis=1)
    topic_to_cat = {t: "Resident" for t in RESIDENT_TOPICS}
    topic_to_cat.update({t: "Business" for t in BUSINESS_TOPICS})
    topic_to_cat.update({t: "Mixed" for t in MIXED_TOPICS})
    dom_label = pd.Series(dom_topic).map(topic_to_cat)

    out = pd.DataFrame({
        "conversation_id": df["conversation_id"],
        "n_turns": df["n_turns"],
        "resident_weight": res_w.round(4),
        "business_weight": biz_w.round(4),
        "mixed_weight": mix_w.round(4),
        "label": label,
        "dominant_topic": dom_topic,
        "dominant_topic_name": pd.Series(dom_topic).map(TOPIC_NAMES),
        "dominant_label": dom_label,
    })
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV.name}  ({len(out):,} conversations)")

    print("\n--- Label distribution (by weighted-category argmax) ---")
    counts = out["label"].value_counts()
    for k, v in counts.items():
        print(f"  {k:<10}  {v:>7,}  ({v / len(out) * 100:5.1f}%)")

    print("\n--- Dominant-topic breakdown ---")
    bd = out.groupby(["dominant_label", "dominant_topic", "dominant_topic_name"]) \
            .size().reset_index(name="conversations") \
            .sort_values(["dominant_label", "conversations"], ascending=[True, False])
    for _, r in bd.iterrows():
        print(f"  [{r['dominant_label']:<8}] t{r['dominant_topic']:>2}  "
              f"{r['dominant_topic_name']:<42}  {r['conversations']:>6,}")


if __name__ == "__main__":
    main()
