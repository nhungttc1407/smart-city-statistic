"""
Sector-level analysis of the Netanya chatbot conversations.

  1. Assign a sector (Resident / Business / Mixed) to every conversation
     from the dominant LDA topic on each side.
  2. Measure conversation complexity and break it down by sector.
  3. Sector-based time analysis (volume + complexity over hour / weekday).
  4. Identify "complex inquiry patterns" and where they concentrate.

Inputs:
  - topic_distribution.csv     topic weights per conversation
  - merged_conversations.csv   user_doc / agent_doc text (lengths)
  - Engdata(...).csv           request_time + thumbs feedback

Outputs:
  - sector_labels.csv          per-conversation sector + complexity metrics
  - figures/sector_*.png       charts
  - sector_analysis_report.md  the written report
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
LDA_DIR = ROOT / "outputs" / "topic" / "lda"
LABELS_DIR = ROOT / "outputs" / "topic" / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)

DIST_CSV = LDA_DIR / "topic_distribution.csv"
MERGED_CSV = DATA_DIR / "merged_conversations.csv"
SRC = DATA_DIR / "Engdata(usage_binatanya_01-03-25_16-11-).csv"
FIG_DIR = ROOT / "reports" / "figures"
OUT_LABELS = LABELS_DIR / "sector_labels.csv"
OUT_REPORT = ROOT / "reports" / "sector_analysis_report.md"

# --- topic -> sector maps (provided) --------------------------------------
USER_TOPIC_SECTOR = {
    0: "Resident",   # street, shelter, waste, parking
    1: "Mixed",      # school + permit
    2: "Resident",   # beach, events, independence day
    3: "Resident",   # property tax, parking ticket
}
AGENT_TOPIC_SECTOR = {
    0:  "Resident",  # education, kindergarten
    1:  "Business",  # tender, procurement
    2:  "Resident",  # tax discount, senior, disabled
    3:  "Mixed",     # building permit + shelter
    4:  "Resident",  # tax forms
    5:  "Resident",  # community center
    6:  "Mixed",     # online payment
    7:  "Resident",  # parking permit resident
    8:  "Business",  # building exemption, committee
    9:  "Resident",  # waste removal
    10: "Business",  # business license, signage
    11: "Resident",  # beach, park, garden
    12: "Mixed",     # contact/hotline
    13: "Resident",  # parking ticket appeal
    14: "Resident",  # events, independence day
}

SECTORS = ["Resident", "Business", "Mixed"]
SECTOR_COLORS = {"Resident": "#4C72B0", "Business": "#DD8452", "Mixed": "#999999"}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def resolve(u: str, a: str) -> str:
    """Combine the question-side and answer-side sector into one label.
    Agree -> that sector. One side Mixed -> take the more specific other side.
    Resident vs Business conflict -> Mixed."""
    if u == a:
        return u
    if u == "Mixed":
        return a
    if a == "Mixed":
        return u
    return "Mixed"


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    # === load ============================================================
    print("Loading topic distribution + merged text ...")
    df = pd.read_csv(DIST_CSV)
    merged = pd.read_csv(MERGED_CSV)[["conversation_id", "user_doc", "agent_doc"]]
    df = df.merge(merged, on="conversation_id", how="left")

    user_cols = [c for c in df.columns if c.startswith("user_topic_")]
    agent_cols = [c for c in df.columns if c.startswith("agent_topic_")]
    n = len(df)

    print(f"Loading request_time + feedback from {SRC.name} ...")
    raw = pd.read_csv(SRC, low_memory=False,
                      usecols=["request_time", "conversation_id",
                               "conversation_turn", "thumbs_up", "thumbs_down"])
    raw["ts"] = pd.to_datetime(
        raw["request_time"].str.replace(" UTC", "", regex=False), errors="coerce")
    conv = raw.groupby("conversation_id").agg(
        ts=("ts", "min"),
        thumbs_down=("thumbs_down",
                     lambda s: int(s.astype(str).str.upper().eq("TRUE").sum())),
    )
    df = df.merge(conv, on="conversation_id", how="left")

    # === 1. sector assignment ===========================================
    df["dom_user_topic"] = df[user_cols].idxmax(axis=1).str.extract(r"(\d+)$").astype(int)
    df["dom_agent_topic"] = df[agent_cols].idxmax(axis=1).str.extract(r"(\d+)$").astype(int)
    df["user_sector"] = df["dom_user_topic"].map(USER_TOPIC_SECTOR)
    df["agent_sector"] = df["dom_agent_topic"].map(AGENT_TOPIC_SECTOR)
    df["sector"] = [resolve(u, a) for u, a in zip(df["user_sector"], df["agent_sector"])]

    # === 2. complexity metrics ==========================================
    df["q_words"] = df["user_doc"].fillna("").str.split().str.len()
    df["a_words"] = df["agent_doc"].fillna("").str.split().str.len()
    # topic entropy: how spread the answer is across the 15 agent topics
    p = df[agent_cols].to_numpy(dtype=float)
    p = p / p.sum(axis=1, keepdims=True)
    df["topic_entropy"] = (-(p * np.log(p + 1e-12)).sum(axis=1) / np.log(len(agent_cols)))

    # composite index = mean of z-scores (counts log-compressed first)
    comp_parts = {
        "n_turns": np.log1p(df["n_turns"]),
        "q_words": np.log1p(df["q_words"]),
        "a_words": np.log1p(df["a_words"]),
        "topic_entropy": df["topic_entropy"],
    }
    z = pd.DataFrame({k: (v - v.mean()) / v.std() for k, v in comp_parts.items()})
    df["complexity_index"] = z.mean(axis=1).round(3)

    # === 3. time fields =================================================
    df["hour"] = df["ts"].dt.hour
    df["weekday"] = df["ts"].dt.dayofweek          # 0 = Mon
    df["date"] = df["ts"].dt.date

    # === 4. complex inquiry patterns ====================================
    q90_turns = df["n_turns"].quantile(0.90)
    q90_aw = df["a_words"].quantile(0.90)
    q90_qw = df["q_words"].quantile(0.90)
    q90_ent = df["topic_entropy"].quantile(0.90)
    MULTI = max(4, int(q90_turns))

    df["pat_multiturn"] = df["n_turns"] >= MULTI
    df["pat_cross_sector"] = (
        ((df["user_sector"] == "Resident") & (df["agent_sector"] == "Business")) |
        ((df["user_sector"] == "Business") & (df["agent_sector"] == "Resident"))
    )
    df["pat_verbose_oneshot"] = (df["n_turns"] == 1) & (df["a_words"] >= q90_aw)
    df["pat_topic_scattered"] = df["topic_entropy"] >= q90_ent
    df["pat_heavy_question"] = df["q_words"] >= q90_qw
    pattern_cols = {
        "pat_multiturn": f"Multi-turn escalation (>= {MULTI} turns)",
        "pat_cross_sector": "Cross-sector mismatch (Resident<->Business)",
        "pat_verbose_oneshot": f"Verbose one-shot (1 turn, answer >= p90 = {q90_aw:.0f} words)",
        "pat_topic_scattered": f"Topic-scattered answer (entropy >= p90 = {q90_ent:.2f})",
        "pat_heavy_question": f"Heavy question (>= p90 = {q90_qw:.0f} words)",
    }

    # === persist per-conversation labels ================================
    keep = ["conversation_id", "ts", "n_turns", "user_sector", "agent_sector",
            "sector", "q_words", "a_words", "topic_entropy", "complexity_index",
            "thumbs_down"] + list(pattern_cols)
    df[keep].to_csv(OUT_LABELS, index=False)

    # === aggregates =====================================================
    sec_counts = df["sector"].value_counts().reindex(SECTORS).fillna(0).astype(int)

    cmp = df.groupby("sector").agg(
        conversations=("conversation_id", "count"),
        mean_turns=("n_turns", "mean"),
        mean_q_words=("q_words", "mean"),
        mean_a_words=("a_words", "mean"),
        mean_entropy=("topic_entropy", "mean"),
        complexity_index=("complexity_index", "mean"),
        thumbs_down_rate=("thumbs_down", lambda s: (s > 0).mean()),
    ).reindex(SECTORS).round(3)

    by_hour = df.pivot_table(index="hour", columns="sector",
                             values="conversation_id", aggfunc="count").fillna(0)
    by_hour = by_hour.reindex(columns=SECTORS).fillna(0)
    by_wd = df.pivot_table(index="weekday", columns="sector",
                           values="conversation_id", aggfunc="count").fillna(0)
    by_wd = by_wd.reindex(columns=SECTORS).fillna(0)
    complexity_by_hour = df.groupby("hour")["complexity_index"].mean()

    # === charts =========================================================
    print("Generating charts ...")

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(FIG_DIR / name, dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    ax.bar(sec_counts.index, sec_counts.values,
           color=[SECTOR_COLORS[s] for s in sec_counts.index])
    for i, v in enumerate(sec_counts.values):
        ax.text(i, v, f" {v:,}\n {v / n * 100:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title("Conversations by sector", fontweight="bold", fontsize=11)
    ax.set_ylim(0, sec_counts.max() * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "sector_volume.png")

    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    ci = cmp["complexity_index"]
    ax.bar(ci.index, ci.values, color=[SECTOR_COLORS[s] for s in ci.index])
    ax.axhline(0, color="#222", linewidth=0.8)
    for i, v in enumerate(ci.values):
        ax.text(i, v, f" {v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_title("Mean complexity index by sector", fontweight="bold", fontsize=11)
    ax.set_ylabel("composite z-score")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "sector_complexity.png")

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for s in SECTORS:
        ax.plot(by_hour.index, by_hour[s], marker="o", ms=3,
                color=SECTOR_COLORS[s], label=s)
    ax.set_title("Conversation volume by hour of day (UTC)", fontweight="bold", fontsize=11)
    ax.set_xlabel("hour"); ax.set_ylabel("conversations")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "sector_by_hour.png")

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(7)
    w = 0.26
    for i, s in enumerate(SECTORS):
        vals = by_wd[s].reindex(range(7)).fillna(0)
        ax.bar(x + (i - 1) * w, vals, w, color=SECTOR_COLORS[s], label=s)
    ax.set_title("Conversation volume by weekday", fontweight="bold", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(WEEKDAYS)
    ax.set_ylabel("conversations")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "sector_by_weekday.png")

    pat_sec = pd.DataFrame({
        name: df.loc[df[col], "sector"].value_counts().reindex(SECTORS).fillna(0)
        for col, name in pattern_cols.items()
    }).T
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    bottom = np.zeros(len(pat_sec))
    for s in SECTORS:
        ax.barh(range(len(pat_sec)), pat_sec[s], left=bottom,
                color=SECTOR_COLORS[s], label=s)
        bottom += pat_sec[s].to_numpy()
    ax.set_yticks(range(len(pat_sec)))
    ax.set_yticklabels([n.split(" (")[0] for n in pat_sec.index], fontsize=8)
    ax.set_title("Complex inquiry patterns by sector", fontweight="bold", fontsize=11)
    ax.set_xlabel("conversations")
    ax.legend(fontsize=8, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "pattern_by_sector.png")

    # === markdown report ================================================
    print("Writing report ...")
    dmin, dmax = df["date"].min(), df["date"].max()
    peak_hours = {s: int(by_hour[s].idxmax()) for s in SECTORS}

    cmp_tbl = cmp.reset_index().rename(columns={"index": "sector"})
    cmp_tbl["thumbs_down_rate"] = (cmp_tbl["thumbs_down_rate"] * 100).round(2).astype(str) + "%"

    pat_rows = []
    for col, name in pattern_cols.items():
        sub = df[df[col]]
        cnt = len(sub)
        sec_mix = sub["sector"].value_counts(normalize=True).reindex(SECTORS).fillna(0)
        # over-representation vs the baseline sector mix
        base = df["sector"].value_counts(normalize=True).reindex(SECTORS).fillna(0)
        lift = (sec_mix / base).round(2)
        top_sec = lift.idxmax()
        pat_rows.append({
            "pattern": name,
            "count": f"{cnt:,}",
            "share": f"{cnt / n * 100:.1f}%",
            "mean complexity": f"{sub['complexity_index'].mean():+.3f}",
            "thumbs-down rate": f"{(sub['thumbs_down'] > 0).mean() * 100:.2f}%",
            "sector mix (R/B/M)": " / ".join(f"{sec_mix[s] * 100:.0f}%" for s in SECTORS),
            "most over-represented": f"{top_sec} (x{lift[top_sec]})",
        })
    pat_tbl = pd.DataFrame(pat_rows)

    md = []
    md += [
        "# Sector Analysis Report",
        "",
        f"Source: `{DIST_CSV.name}` + `{SRC.name}`  |  **{n:,} conversations**  |  "
        f"date range {dmin} - {dmax}.",
        "",
        "## 1. Sector assignment",
        "",
        "Each conversation's dominant LDA topic on each side is mapped to a sector "
        "via the provided `USER_TOPIC_SECTOR` / `AGENT_TOPIC_SECTOR` maps. The two "
        "sides are then resolved into one `sector`: **agree -> that sector; one side "
        "Mixed -> take the more specific other side; Resident vs Business conflict "
        "-> Mixed**.",
        "",
        md_table(pd.DataFrame({
            "sector": SECTORS,
            "conversations": [f"{int(sec_counts[s]):,}" for s in SECTORS],
            "share": [f"{sec_counts[s] / n * 100:.1f}%" for s in SECTORS],
        })),
        "",
        "![sector volume](figures/sector_volume.png)",
        "",
        "## 2. Complexity by sector",
        "",
        "Complexity per conversation combines four signals: number of turns, "
        "question length, answer length (all log-compressed) and answer **topic "
        "entropy** (how spread the answer is across the 15 agent topics). The "
        "`complexity_index` is the mean of their z-scores - 0 is dataset average, "
        "positive is more complex.",
        "",
        md_table(cmp_tbl),
        "",
        "![sector complexity](figures/sector_complexity.png)",
        "",
        "## 3. Sector-based time analysis",
        "",
        f"Conversations span {dmin} to {dmax}. Peak hour (UTC): "
        + ", ".join(f"**{s}** {peak_hours[s]:02d}:00" for s in SECTORS) + ".",
        "",
        "Volume by hour of day (UTC):",
        "",
        md_table(by_hour.astype(int).reset_index().rename(columns={"hour": "hour"})),
        "",
        "![by hour](figures/sector_by_hour.png)",
        "",
        "Volume by weekday:",
        "",
        md_table(pd.DataFrame({
            "weekday": WEEKDAYS,
            **{s: by_wd[s].reindex(range(7)).fillna(0).astype(int).tolist() for s in SECTORS},
        })),
        "",
        "![by weekday](figures/sector_by_weekday.png)",
        "",
        "Mean complexity index by hour (peaks flag the hours that draw the "
        "hardest inquiries):",
        "",
        md_table(complexity_by_hour.round(3).reset_index().rename(
            columns={"complexity_index": "mean_complexity"})),
        "",
        "## 4. Complex inquiry patterns",
        "",
        "Five patterns flag conversations that are structurally hard to handle. "
        "Thresholds use the 90th percentile of each metric.",
        "",
        md_table(pat_tbl),
        "",
        "![patterns by sector](figures/pattern_by_sector.png)",
        "",
        "## 5. Key findings",
        "",
    ]
    # auto findings
    hardest = cmp["complexity_index"].idxmax()
    cross = int(df["pat_cross_sector"].sum())
    findings = [
        f"**{hardest}** is the most complex sector "
        f"(complexity index {cmp.loc[hardest, 'complexity_index']:+.3f}, "
        f"mean {cmp.loc[hardest, 'mean_turns']:.2f} turns, "
        f"{cmp.loc[hardest, 'mean_a_words']:.0f}-word answers).",
        f"Cross-sector mismatch hits {cross:,} conversations "
        f"({cross / n * 100:.1f}%) - the question and answer sectors genuinely "
        "disagree, the clearest signal of a misrouted or hard inquiry.",
        f"The busiest hour overall is "
        f"{int(by_hour.sum(axis=1).idxmax()):02d}:00 UTC; complexity peaks at "
        f"{int(complexity_by_hour.idxmax()):02d}:00 UTC.",
        "Each pattern's `most over-represented` column shows which sector is hit "
        "harder than its baseline share - use it to target routing or canned-"
        "answer improvements.",
    ]
    md += [f"- {f}" for f in findings]
    md.append("")
    OUT_REPORT.write_text("\n".join(md), encoding="utf-8")

    # === console summary ================================================
    print(f"\nsectors: " + ", ".join(f"{s}={int(sec_counts[s]):,}" for s in SECTORS))
    print(cmp.to_string())
    print(f"\nwrote {OUT_LABELS.name}  ({n:,} rows)")
    print(f"wrote {OUT_REPORT.name}")
    print(f"wrote 5 charts to {FIG_DIR.name}/")


if __name__ == "__main__":
    main()
