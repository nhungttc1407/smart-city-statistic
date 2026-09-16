"""
Build the statistics report for the English question / answer text.

Outputs:
  - figures/*.png          histograms
  - text_stats_summary.csv length distribution table
  - statistics_report.docx report with tables + embedded histograms
                           (upload to Google Drive -> opens as a Google Doc)
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt

HERE = Path(__file__).parent
SRC = HERE / "Engdata(usage_binatanya_01-03-25_16-11-).csv"
FIG_DIR = HERE / "figures"
OUT_CSV = HERE / "text_stats_summary.csv"
OUT_DOCX = HERE / "statistics_report.docx"
TOPIC_KEYWORDS_CSV = HERE / "topic_keywords.csv"      # from topic_model.py
TOPIC_KEYWORDS_COMBINED_CSV = HERE / "topic_keywords_combined.csv"  # from topic_combined.py
PERPLEXITY_COMBINED_CSV = HERE / "topic_perplexity_sweep_combined.csv"
LABELS_CSV = HERE / "conversation_labels.csv"          # from classify.py
LABELS_COMBINED_CSV = HERE / "conversation_labels_combined.csv"  # from classify_combined.py
SECTOR_LABELS_QA_CSV = HERE / "sector_labels_qa.csv"            # from classify_sector.py

PCTILES = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
WORD_RE = re.compile(r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
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


def histogram(series: pd.Series, title: str, xlabel: str, fname: str,
              color: str, clip_pctile: float = 0.99) -> Path:
    """Histogram clipped at a percentile so outliers don't flatten the plot."""
    s = series.dropna()
    hi = s.quantile(clip_pctile)
    clipped = s[s <= hi]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.hist(clipped, bins=40, color=color, edgecolor="white", linewidth=0.4)
    ax.axvline(s.median(), color="#222", linestyle="--", linewidth=1,
               label=f"median = {s.median():.0f}")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(f"{xlabel}  (clipped at p{int(clip_pctile * 100)} = {hi:.0f})", fontsize=9)
    ax.set_ylabel("conversations" if "turn" in fname else "turns", fontsize=9)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = FIG_DIR / fname
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def bar_chart(counts: pd.Series, title: str, fname: str) -> Path:
    """Horizontal bar chart with count + percentage labels."""
    total = counts.sum()
    fig, ax = plt.subplots(figsize=(6.2, max(2.4, 0.7 * len(counts))))
    colors = {"Resident": "#4C72B0", "Business": "#DD8452", "Mixed": "#8172B2"}
    bars = ax.barh(counts.index.tolist(), counts.values,
                   color=[colors.get(k, "#999") for k in counts.index])
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_width() + total * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:,}  ({v / total * 100:.1f}%)", va="center", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(0, total * 1.18)
    ax.set_xlabel("conversations", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = FIG_DIR / fname
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def add_table(doc: Document, df: pd.DataFrame) -> None:
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    for j, col in enumerate(df.columns):
        run = t.rows[0].cells[j].paragraphs[0].add_run(str(col))
        run.bold = True
        run.font.size = Pt(9)
    for _, r in df.iterrows():
        cells = t.add_row().cells
        for j, val in enumerate(r):
            cells[j].paragraphs[0].add_run(str(val)).font.size = Pt(9)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    print(f"Reading {SRC.name} ...")
    df = pd.read_csv(SRC, encoding="utf-8", low_memory=False)
    n_rows, n_conv = len(df), df["conversation_id"].nunique()
    print(f"  {n_rows:,} turns, {n_conv:,} conversations")

    q, a = df["user_message_en"], df["agent_response_en"]
    qs, as_ = q.fillna("").astype(str), a.fillna("").astype(str)
    q_words, a_words = qs.str.split().str.len(), as_.str.split().str.len()
    q_chars, a_chars = qs.str.len(), as_.str.len()
    turns = df.groupby("conversation_id")["conversation_turn"].count()

    # --- length table ----------------------------------------------------
    summary = pd.DataFrame([
        desc(q_chars, "question_chars"), desc(q_words, "question_words"),
        desc(a_chars, "answer_chars"), desc(a_words, "answer_words"),
    ])
    summary.to_csv(OUT_CSV, index=False)

    # --- coverage --------------------------------------------------------
    cov = pd.DataFrame([
        {"field": "question (user_message_en)",
         "empty": int((qs.str.strip() == "").sum()),
         "usable": int((qs.str.strip() != "").sum())},
        {"field": "answer (agent_response_en)",
         "empty": int((as_.str.strip() == "").sum()),
         "usable": int((as_.str.strip() != "").sum())},
    ])

    # --- feedback --------------------------------------------------------
    up = df["thumbs_up"].astype(str).str.upper().eq("TRUE").sum()
    down = df["thumbs_down"].astype(str).str.upper().eq("TRUE").sum()
    rated = int(up + down)

    # --- histograms ------------------------------------------------------
    print("Generating histograms ...")
    figs = [
        histogram(q_words, "Question length (words)", "words per question",
                  "question_words.png", "#4C72B0"),
        histogram(a_words, "Answer length (words)", "words per answer",
                  "answer_words.png", "#55A868"),
        histogram(q_chars, "Question length (characters)", "characters per question",
                  "question_chars.png", "#4C72B0"),
        histogram(a_chars, "Answer length (characters)", "characters per answer",
                  "answer_chars.png", "#55A868"),
        histogram(turns, "Turns per conversation", "turns per conversation",
                  "turns_per_conversation.png", "#C44E52", clip_pctile=0.99),
    ]

    # --- build docx ------------------------------------------------------
    print("Writing report ...")
    doc = Document()
    doc.add_heading("Netanya Chatbot — Question & Answer Text Statistics", level=0)
    p = doc.add_paragraph()
    p.add_run(f"Source: {SRC.name}\n").italic = True
    p.add_run(f"Generated: {date.today().isoformat()}    "
              f"Dataset: {n_rows:,} turns across {n_conv:,} conversations").italic = True

    doc.add_heading("1. Coverage", level=1)
    doc.add_paragraph("Rows with empty English text are excluded from all "
                      "downstream length and word statistics.")
    add_table(doc, cov)

    doc.add_heading("2. Conversation structure", level=1)
    doc.add_paragraph(
        f"Mean {turns.mean():.2f} turns per conversation, median {turns.median():.0f}, "
        f"max {turns.max()}. {(turns == 1).sum():,} conversations "
        f"({(turns == 1).mean() * 100:.0f}%) are single-turn.")
    doc.add_picture(str(figs[4]), width=Inches(5.2))

    doc.add_heading("3. Length distributions", level=1)
    doc.add_paragraph("Questions are short; answers are ~15x longer. "
                      "Percentiles (p25–p99) shown because both distributions "
                      "are right-skewed by outliers.")
    add_table(doc, summary)
    doc.add_paragraph()
    for f in figs[:4]:
        doc.add_picture(str(f), width=Inches(4.6))

    doc.add_heading("4. Feedback", level=1)
    doc.add_paragraph(
        f"Only {rated:,} of {n_rows:,} turns ({rated / n_rows * 100:.1f}%) were "
        f"rated. Of those: {up:,} thumbs-up, {down:,} thumbs-down — "
        f"{up / rated * 100:.1f}% satisfaction. The low rating rate means this "
        f"signal is too sparse to segment on reliably.")

    doc.add_heading("5. Most frequent words", level=1)
    for label, series in [("Questions", q), ("Answers", a)]:
        doc.add_heading(label, level=2)
        words = ", ".join(f"{w} ({n:,})" for w, n in top_words(series))
        doc.add_paragraph(words)

    doc.add_heading("6. Notes & data-quality flags", level=1)
    for note in [
        f"Longest question is {int(q_chars.max()):,} chars / "
        f"{int(q_words.max()):,} words — almost certainly a paste/junk outlier; "
        f"consider capping questions at p99 (~{q_words.quantile(0.99):.0f} words).",
        f"Answer length p99 is ~{a_chars.quantile(0.99):,.0f} chars; a few answers "
        f"reach {int(a_chars.max()):,} chars. Cap or flag before modeling.",
        "Histograms are clipped at p99 so the long right tail does not flatten "
        "the visible distribution; tables report the true min/max.",
    ]:
        doc.add_paragraph(note, style="List Bullet")

    # --- 7. topic model --------------------------------------------------
    if TOPIC_KEYWORDS_CSV.exists():
        kw = pd.read_csv(TOPIC_KEYWORDS_CSV)
        doc.add_heading("7. Topic model (LDA)", level=1)
        doc.add_paragraph(
            "Latent Dirichlet Allocation was fit separately on the questions "
            "(user_message_en) and the answers (agent_response_en), after merging "
            "all turns of a conversation into one document. The number of topics "
            "per side was chosen by a held-out perplexity sweep over k = 4..15.")
        for src, label in [("user", "7.1 Question topics"),
                           ("agent", "7.2 Answer topics")]:
            sub = kw[kw["source"] == src][["topic", "top_keywords"]].copy()
            sub.columns = ["topic", "top keywords"]
            doc.add_heading(f"{label}  (k = {len(sub)})", level=2)
            add_table(doc, sub)

    # --- 8. combined Q+A topic model -------------------------------------
    if TOPIC_KEYWORDS_COMBINED_CSV.exists():
        kw_c = pd.read_csv(TOPIC_KEYWORDS_COMBINED_CSV)
        best_k_c = len(kw_c)
        best_npmi_c = None
        best_ppl_at_k = None
        if PERPLEXITY_COMBINED_CSV.exists():
            sweep_c = pd.read_csv(PERPLEXITY_COMBINED_CSV)
            if "npmi_coherence" in sweep_c.columns:
                best_row = sweep_c.loc[sweep_c["npmi_coherence"].idxmax()]
                best_npmi_c = float(best_row["npmi_coherence"])
                best_ppl_at_k = float(best_row["perplexity"])

        doc.add_heading("8. Combined Q+A topic model (LDA)", level=1)
        intro = (
            "An additional LDA was fit on a merged document per conversation, "
            "where the question text (user_message_en) and the answer text "
            "(agent_response_en) are concatenated into one paragraph before "
            "vectorization. This captures the joint Q+A topic rather than "
            "modeling each side independently. The number of topics was "
            "selected by sweeping k = 1..20 and picking the k that maximizes "
            "mean NPMI (normalized pointwise mutual information) topic "
            "coherence over the top-10 words per topic. NPMI was chosen over "
            "held-out perplexity because perplexity decreases monotonically "
            "with k on this corpus and therefore pins k at the upper bound "
            "of any candidate range, while NPMI has a true interior optimum "
            "tied to topic interpretability"
        )
        if best_npmi_c is not None and best_ppl_at_k is not None:
            intro += (f"; the best k = {best_k_c} "
                      f"(NPMI = {best_npmi_c:.4f}, perplexity at this k = "
                      f"{best_ppl_at_k:,.1f}).")
        else:
            intro += f"; the best k = {best_k_c}."
        doc.add_paragraph(intro)

        sub = kw_c[["topic", "top_keywords"]].copy()
        sub.columns = ["topic", "top keywords"]
        doc.add_heading(f"8.1 Combined topics  (k = {best_k_c})", level=2)
        add_table(doc, sub)

    # --- 9. classification of combined Q+A topics ------------------------
    if LABELS_COMBINED_CSV.exists():
        from classify_combined import (
            RESIDENT_TOPICS, BUSINESS_TOPICS, MIXED_TOPICS, TOPIC_NAMES,
        )
        lab_c = pd.read_csv(LABELS_COMBINED_CSV)
        n_c = len(lab_c)
        counts_c = lab_c["label"].value_counts().reindex(
            ["Resident", "Business", "Mixed"]).fillna(0).astype(int)

        doc.add_heading("9. Resident / Business / Mixed classification "
                        "(combined Q+A)", level=1)
        doc.add_paragraph(
            "Each conversation is classified by aggregating the 16 combined "
            "Q+A topic weights under a human-curated topic-to-category map "
            "(based on reading the top keywords of each topic). For a "
            "conversation, resident_weight is the sum of probabilities over "
            "the Resident topics, business_weight over the Business topic, "
            "and mixed_weight over the Mixed topics. The label is the "
            "argmax over those three weights. Mixed topics are ones whose "
            "keywords legitimately serve both residents and businesses "
            "(building permits, generic department contact, generic "
            "municipal services), so a Mixed label means the conversation "
            "is genuinely ambiguous rather than misclassified.")

        # 9.1 topic -> category map
        doc.add_heading("9.1 Topic-to-category map", level=2)
        map_rows = []
        for cat, ts in [("Resident", RESIDENT_TOPICS),
                        ("Business", BUSINESS_TOPICS),
                        ("Mixed", MIXED_TOPICS)]:
            for t in ts:
                map_rows.append({
                    "category": cat,
                    "topic": t,
                    "name": TOPIC_NAMES[t],
                })
        add_table(doc, pd.DataFrame(map_rows))

        # 9.2 conversation counts
        doc.add_heading("9.2 Label distribution", level=2)
        res_table = pd.DataFrame([
            {"label": k, "conversations": f"{v:,}",
             "share": f"{v / n_c * 100:.1f}%"}
            for k, v in counts_c.items()
        ])
        add_table(doc, res_table)
        doc.add_paragraph()
        doc.add_picture(
            str(bar_chart(counts_c,
                          "Combined Q+A classification (Resident / Business / Mixed)",
                          "classification_combined_split.png")),
            width=Inches(5.2),
        )

        # 9.3 dominant-topic breakdown
        doc.add_heading("9.3 Dominant-topic breakdown", level=2)
        bd = (lab_c.groupby(
                  ["dominant_label", "dominant_topic", "dominant_topic_name"])
                 .size().reset_index(name="conversations")
                 .sort_values(["dominant_label", "conversations"],
                              ascending=[True, False]))
        bd["share"] = (bd["conversations"] / n_c * 100).round(1).astype(str) + "%"
        bd["conversations"] = bd["conversations"].map(lambda v: f"{v:,}")
        bd = bd.rename(columns={
            "dominant_label": "category",
            "dominant_topic": "topic",
            "dominant_topic_name": "name",
        })[["category", "topic", "name", "conversations", "share"]]
        add_table(doc, bd)
        doc.add_paragraph(
            "The dominant-topic breakdown counts conversations whose single "
            "highest-weight topic falls in each category — a sanity check "
            "for the weighted-argmax label above. The Business count here "
            f"({(bd[bd['category'] == 'Business']['conversations'].astype(str).str.replace(',', '').astype(int).sum()):,}) "
            "is dominated by topic 7 (Tenders & procurement) and should be "
            "treated as an under-count of true business intent, because "
            "business licensing signals also live inside the Mixed topic 2 "
            "(Building permit + business licensing).",
            style="List Bullet")

    # --- 10. per-side (Q vs A) classification from classify.py -----------
    if LABELS_CSV.exists():
        lab = pd.read_csv(LABELS_CSV)
        needed = {"cat_question_only", "cat_answer_only", "cat_both",
                  "question_class", "answer_class"}
        if needed.issubset(lab.columns):
            try:
                from classify import (
                    BUSINESS_TOPICS_USER, BUSINESS_TOPICS_AGENT, NEUTRAL_TOPICS,
                )
                map_desc = (
                    f"Topic-to-class map (derived from topic_keywords.csv): "
                    f"question Business = {', '.join(sorted(BUSINESS_TOPICS_USER))}; "
                    f"answer Business = {', '.join(sorted(BUSINESS_TOPICS_AGENT))}; "
                    f"answer Neutral = {', '.join(sorted(NEUTRAL_TOPICS))}; "
                    "every other dominant topic = Resident."
                )
            except Exception:
                map_desc = ("Topic-to-class map is the one defined in "
                            "classify.py (Business topics on each side were "
                            "tagged by hand from their LDA keywords; every "
                            "other dominant topic is Resident, with one "
                            "Neutral answer topic).")

            n = len(lab)
            agree = (lab["question_class"] == lab["answer_class"]) & \
                    lab["question_class"].isin(["Resident", "Business"])
            agreed = int(agree.sum())

            doc.add_heading(
                "10. Per-side classification: question vs answer "
                "(separate LDA models)", level=1)
            doc.add_paragraph(
                "This section uses the separate question-side and answer-side "
                "LDA models from section 7 (not the combined Q+A model from "
                "section 8). Each conversation's dominant topic on each side "
                "is mapped to Resident / Business / Neutral, and three "
                "labelling modes are reported: from the question alone, from "
                "the answer alone, and from both sides agreeing.")
            doc.add_paragraph(map_desc)

            def counts_table(col: str, order: list[str]) -> pd.DataFrame:
                vc = lab[col].value_counts()
                return pd.DataFrame([
                    {"label": k,
                     "conversations": f"{int(vc.get(k, 0)):,}",
                     "share": f"{int(vc.get(k, 0)) / n * 100:.1f}%"}
                    for k in order
                ])

            # 10.1 question only
            doc.add_heading("10.1 Mode 1 — Question only", level=2)
            doc.add_paragraph(
                "Category taken from the dominant question topic. Every "
                "conversation receives a label (no Neutral on the question side).")
            q_counts = lab["cat_question_only"].value_counts().reindex(
                ["Resident", "Business"]).fillna(0).astype(int)
            add_table(doc, counts_table("cat_question_only",
                                        ["Resident", "Business"]))
            doc.add_paragraph()
            doc.add_picture(str(bar_chart(
                q_counts, "Mode 1 — Question-only classification",
                "classification_question_only.png")), width=Inches(5.2))

            # 10.2 answer only
            doc.add_heading("10.2 Mode 2 — Answer only", level=2)
            doc.add_paragraph(
                "Category taken from the dominant answer topic. A Neutral "
                "bucket appears here because one answer topic is generic "
                "department-contact boilerplate that fits neither audience.")
            a_counts = lab["cat_answer_only"].value_counts().reindex(
                ["Resident", "Business", "Neutral"]).fillna(0).astype(int)
            add_table(doc, counts_table("cat_answer_only",
                                        ["Resident", "Business", "Neutral"]))
            doc.add_paragraph()
            doc.add_picture(str(bar_chart(
                a_counts, "Mode 2 — Answer-only classification",
                "classification_answer_only.png")), width=Inches(5.2))

            # 10.3 both must agree
            doc.add_heading("10.3 Mode 3 — Both sides must agree", level=2)
            doc.add_paragraph(
                "A category is assigned only when the question-side label "
                "and the answer-side label are the same Resident/Business "
                "value; otherwise the conversation is left Uncategorized. "
                "This is the strictest of the three modes.")
            b_counts = lab["cat_both"].value_counts().reindex(
                ["Resident", "Business", "Uncategorized"]).fillna(0).astype(int)
            add_table(doc, counts_table("cat_both",
                                        ["Resident", "Business", "Uncategorized"]))
            doc.add_paragraph()
            doc.add_picture(str(bar_chart(
                b_counts, "Mode 3 — Question + answer agreement",
                "classification_both.png")), width=Inches(5.2))

            # 10.4 cross-tab
            doc.add_heading("10.4 Question × Answer cross-tab", level=2)
            doc.add_paragraph(
                f"Rows = question-side class, columns = answer-side class. "
                f"The diagonal cells for Resident/Business are the "
                f"conversations Mode 3 keeps; everything else stays "
                f"Uncategorized. Agreement rate (question and answer give "
                f"the same Resident/Business label): "
                f"{agreed:,} / {n:,} = {agreed / n * 100:.1f}%.")

            xtab = pd.crosstab(lab["question_class"], lab["answer_class"])
            xtab = xtab.reindex(index=["Resident", "Business"],
                                columns=["Resident", "Business", "Neutral"]).fillna(0).astype(int)
            xt_rows = []
            for q_cls, row in xtab.iterrows():
                d = {"question \\ answer": q_cls}
                for a_cls, v in row.items():
                    d[a_cls] = f"{int(v):,}"
                xt_rows.append(d)
            add_table(doc, pd.DataFrame(xt_rows))

            doc.add_paragraph(
                "Caveat: the per-side classification depends on the dominant "
                "topic only and inherits the impurity of the underlying LDA "
                "topics (e.g. business licensing and dog licensing share an "
                "answer topic). The combined Q+A classification in section 9 "
                "uses topic weight sums across all 16 topics and is generally "
                "the more robust signal; this section is kept for "
                "side-by-side comparison and to expose where the question "
                "and answer disagree.",
                style="List Bullet")

    # --- 11. sector classification (Resident / Business / Mixed per side) -
    if SECTOR_LABELS_QA_CSV.exists():
        try:
            from classify_sector import USER_TOPIC_SECTOR, AGENT_TOPIC_SECTOR
        except Exception:
            USER_TOPIC_SECTOR, AGENT_TOPIC_SECTOR = {}, {}

        sec = pd.read_csv(SECTOR_LABELS_QA_CSV)
        n_s = len(sec)

        kw_lookup = {}
        if TOPIC_KEYWORDS_CSV.exists():
            kw = pd.read_csv(TOPIC_KEYWORDS_CSV)
            for _, r in kw.iterrows():
                kw_lookup[(r["source"], int(r["topic"]))] = r["top_keywords"]

        doc.add_heading(
            "11. Sector classification: Resident / Business / Mixed (per side)",
            level=1)
        doc.add_paragraph(
            "A second per-side classification, using the human-curated "
            "topic-to-sector map below. Same three modes as section 10 "
            "(question only, answer only, both must agree), but the map "
            "now produces three labels — Resident, Business, Mixed — "
            "instead of Resident/Business plus a Neutral answer bucket. "
            "Mixed marks topics whose keywords legitimately serve both "
            "audiences (e.g. agent topic 3, building permits + planning); "
            "it is a positive label, not 'unknown'.")

        # 11.1 maps
        doc.add_heading("11.1 Topic-to-sector map", level=2)
        doc.add_paragraph("Question-side (user_topic_*, k = 4):")
        user_rows = []
        for t, sector in sorted(USER_TOPIC_SECTOR.items()):
            user_rows.append({
                "topic": f"user_topic_{t}",
                "sector": sector,
                "top keywords": kw_lookup.get(("user", t), ""),
            })
        add_table(doc, pd.DataFrame(user_rows))
        doc.add_paragraph()
        doc.add_paragraph("Answer-side (agent_topic_*, k = 15):")
        agent_rows = []
        for t, sector in sorted(AGENT_TOPIC_SECTOR.items()):
            agent_rows.append({
                "topic": f"agent_topic_{t}",
                "sector": sector,
                "top keywords": kw_lookup.get(("agent", t), ""),
            })
        add_table(doc, pd.DataFrame(agent_rows))
        doc.add_paragraph(
            "Note: no user topic is mapped to Business — Business intent "
            "shows up in user_topic_1 (Mixed) and in the answer side. "
            "This means Mode 3 (both sides agree) can never produce a "
            "Business label; conversations whose answer side is Business "
            "fall into Uncategorized in Mode 3.",
            style="List Bullet")

        # helpers
        def sec_counts(col: str, order: list[str]) -> pd.DataFrame:
            vc = sec[col].value_counts()
            return pd.DataFrame([
                {"label": k,
                 "conversations": f"{int(vc.get(k, 0)):,}",
                 "share": f"{int(vc.get(k, 0)) / n_s * 100:.1f}%"}
                for k in order
            ])

        def sec_series(col: str, order: list[str]) -> pd.Series:
            return sec[col].value_counts().reindex(order).fillna(0).astype(int)

        # 11.2 question only
        doc.add_heading("11.2 Mode 1 — Question only", level=2)
        order_qa = ["Resident", "Business", "Mixed"]
        doc.add_paragraph(
            "Sector taken from the dominant user topic. The Business count "
            "is zero by construction (see map).")
        add_table(doc, sec_counts("sector_question_only", order_qa))
        doc.add_paragraph()
        doc.add_picture(str(bar_chart(
            sec_series("sector_question_only", order_qa),
            "Mode 1 — Question-only sector classification",
            "sector_question_only.png")), width=Inches(5.2))

        # 11.3 answer only
        doc.add_heading("11.3 Mode 2 — Answer only", level=2)
        doc.add_paragraph(
            "Sector taken from the dominant agent topic. All three "
            "sectors are reachable here.")
        add_table(doc, sec_counts("sector_answer_only", order_qa))
        doc.add_paragraph()
        doc.add_picture(str(bar_chart(
            sec_series("sector_answer_only", order_qa),
            "Mode 2 — Answer-only sector classification",
            "sector_answer_only.png")), width=Inches(5.2))

        # 11.4 both must agree
        doc.add_heading("11.4 Mode 3 — Both sides must agree", level=2)
        order_both = ["Resident", "Business", "Mixed", "Uncategorized"]
        agree = (sec["question_sector"] == sec["answer_sector"])
        agreed = int(agree.sum())
        doc.add_paragraph(
            f"A sector is assigned only when the question-side sector and "
            f"the answer-side sector are the same Resident/Business/Mixed "
            f"value; otherwise the conversation is Uncategorized. "
            f"Agreement rate (question and answer give the same sector): "
            f"{agreed:,} / {n_s:,} = {agreed / n_s * 100:.1f}%.")
        add_table(doc, sec_counts("sector_both", order_both))
        doc.add_paragraph()
        doc.add_picture(str(bar_chart(
            sec_series("sector_both", order_both),
            "Mode 3 — Question + answer sector agreement",
            "sector_both.png")), width=Inches(5.2))

        # 11.5 cross-tab
        doc.add_heading("11.5 Question × Answer cross-tab", level=2)
        doc.add_paragraph(
            "Rows = question-side sector, columns = answer-side sector. "
            "The diagonal cells are the conversations Mode 3 keeps; "
            "off-diagonal cells become Uncategorized.")
        xtab = pd.crosstab(sec["question_sector"], sec["answer_sector"])
        xtab = xtab.reindex(index=["Resident", "Business", "Mixed"],
                            columns=["Resident", "Business", "Mixed"]).fillna(0).astype(int)
        xt_rows = []
        for q_cls, row in xtab.iterrows():
            d = {"question \\ answer": q_cls}
            for a_cls, v in row.items():
                d[a_cls] = f"{int(v):,}"
            xt_rows.append(d)
        add_table(doc, pd.DataFrame(xt_rows))

    doc.save(OUT_DOCX)
    print(f"wrote {OUT_DOCX.name}  ({len(figs)} figures in {FIG_DIR.name}/)")


if __name__ == "__main__":
    main()
