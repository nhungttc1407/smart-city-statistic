"""
Pipeline:
  1. Read the engagement-data CSV.
  2. Merge all turns per conversation_id (ordered by conversation_turn) into:
       - user_doc  = concat of user_message_en
       - agent_doc = concat of agent_response_en
     Write merged data to merged_conversations.csv.
  3. Sweep candidate topic counts and pick the best k per side
     (lowest held-out perplexity).
  4. Fit final LDA models, write topic_keywords.csv and topic_distribution.csv,
     and print topics + keywords for questions and answers separately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
LDA_DIR = ROOT / "outputs" / "topic" / "lda"
LDA_DIR.mkdir(parents=True, exist_ok=True)

SRC = DATA_DIR / "Engdata(usage_binatanya_01-03-25_16-11-).csv"
MERGED_CSV = DATA_DIR / "merged_conversations.csv"
TOPIC_DIST_CSV = LDA_DIR / "topic_distribution.csv"
TOPIC_KEYWORDS_CSV = LDA_DIR / "topic_keywords.csv"
PERPLEXITY_CSV = LDA_DIR / "topic_perplexity_sweep.csv"
SUMMARY_TXT = LDA_DIR / "topic_summary.txt"

CANDIDATE_KS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
N_TOP_WORDS = 15
MIN_DOC_LEN = 3

# Common HTML/URL artefacts from the agent responses that are not informative.
EXTRA_STOPWORDS = {
    "href", "https", "http", "target", "blank", "www", "il", "aspx",
    "muni", "muninetanya", "pages", "html", "br", "li", "ul",
    "netanya",  # appears in nearly every answer, drowns out signal
}


def merge_turns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["user_message_en"] = df["user_message_en"].fillna("").astype(str)
    df["agent_response_en"] = df["agent_response_en"].fillna("").astype(str)
    df["conversation_turn"] = pd.to_numeric(
        df["conversation_turn"], errors="coerce"
    ).fillna(0).astype(int)

    df = df.sort_values(["conversation_id", "conversation_turn"])
    merged = df.groupby("conversation_id", as_index=False).agg(
        n_turns=("conversation_turn", "count"),
        user_doc=("user_message_en", lambda s: " \n ".join(x for x in s if x.strip())),
        agent_doc=("agent_response_en", lambda s: " \n ".join(x for x in s if x.strip())),
    )
    return merged


def vectorize(docs: list[str], extra_stop: set[str]) -> tuple:
    stop = "english"
    vec = CountVectorizer(
        max_df=0.90,
        min_df=5,
        stop_words=stop,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out()

    # Drop extra stopwords from the matrix
    keep_mask = np.array([v not in extra_stop for v in vocab])
    X = X[:, keep_mask]
    vocab = vocab[keep_mask]
    return X, vocab


def sweep_topics(X, ks: list[int], label: str) -> tuple[int, pd.DataFrame]:
    """Train LDA at each k on 80% of docs, score perplexity on the other 20%.
    Lower perplexity is better."""
    train_idx, test_idx = train_test_split(
        np.arange(X.shape[0]), test_size=0.2, random_state=42
    )
    X_tr, X_te = X[train_idx], X[test_idx]

    rows = []
    for k in ks:
        lda = LatentDirichletAllocation(
            n_components=k,
            learning_method="online",
            batch_size=2048,
            random_state=42,
            max_iter=10,
            n_jobs=-1,
        )
        lda.fit(X_tr)
        ppl = lda.perplexity(X_te)
        print(f"  [{label}] k={k:>3}  perplexity={ppl:,.1f}")
        rows.append({"source": label, "k": k, "perplexity": ppl})

    sweep_df = pd.DataFrame(rows)
    best_k = int(sweep_df.loc[sweep_df["perplexity"].idxmin(), "k"])
    print(f"  [{label}] BEST k = {best_k}")
    return best_k, sweep_df


def fit_final(X, vocab, k: int, label: str) -> tuple[pd.DataFrame, np.ndarray]:
    lda = LatentDirichletAllocation(
        n_components=k,
        learning_method="batch",
        random_state=42,
        max_iter=30,
        n_jobs=-1,
    )
    doc_topic = lda.fit_transform(X)

    rows = []
    for t, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[::-1][:N_TOP_WORDS]
        rows.append({
            "source": label,
            "topic": t,
            "top_keywords": ", ".join(vocab[i] for i in top_idx),
        })
    return pd.DataFrame(rows), doc_topic


def print_topics(title: str, kw_df: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    for _, r in kw_df.iterrows():
        print(f"  topic {r['topic']:>2}: {r['top_keywords']}")


def main() -> None:
    print(f"Reading {SRC.name}")
    df = pd.read_csv(SRC, encoding="utf-8")
    print(f"  rows={len(df):,}  cols={len(df.columns)}")

    merged = merge_turns(df)
    print(f"Merged into {len(merged):,} conversation documents")
    merged.to_csv(MERGED_CSV, index=False)
    print(f"  wrote {MERGED_CSV.name}  (columns: conversation_id, n_turns, user_doc, agent_doc)")

    # Keep only docs with enough words to vectorize meaningfully.
    user_docs = merged["user_doc"].fillna("").tolist()
    agent_docs = merged["agent_doc"].fillna("").tolist()

    print("\nVectorizing …")
    Xu, vu = vectorize(user_docs, EXTRA_STOPWORDS)
    Xa, va = vectorize(agent_docs, EXTRA_STOPWORDS)
    print(f"  user  matrix: {Xu.shape}, vocab={len(vu)}")
    print(f"  agent matrix: {Xa.shape}, vocab={len(va)}")

    print("\nSweeping k for USER side …")
    best_ku, sweep_u = sweep_topics(Xu, CANDIDATE_KS, "user")

    print("\nSweeping k for AGENT side …")
    best_ka, sweep_a = sweep_topics(Xa, CANDIDATE_KS, "agent")

    pd.concat([sweep_u, sweep_a], ignore_index=True).to_csv(PERPLEXITY_CSV, index=False)
    print(f"\nwrote {PERPLEXITY_CSV.name}")

    print(f"\nFitting final USER LDA with k={best_ku} …")
    user_kw, user_dt = fit_final(Xu, vu, best_ku, "user")
    print(f"Fitting final AGENT LDA with k={best_ka} …")
    agent_kw, agent_dt = fit_final(Xa, va, best_ka, "agent")

    pd.concat([user_kw, agent_kw], ignore_index=True).to_csv(TOPIC_KEYWORDS_CSV, index=False)
    print(f"wrote {TOPIC_KEYWORDS_CSV.name}")

    dist = pd.concat(
        [
            merged[["conversation_id", "n_turns"]].reset_index(drop=True),
            pd.DataFrame(user_dt, columns=[f"user_topic_{i}" for i in range(best_ku)]),
            pd.DataFrame(agent_dt, columns=[f"agent_topic_{i}" for i in range(best_ka)]),
        ],
        axis=1,
    )
    dist.to_csv(TOPIC_DIST_CSV, index=False)
    print(f"wrote {TOPIC_DIST_CSV.name}")

    print_topics(f"QUESTION TOPICS  (user_message_en, k={best_ku})", user_kw)
    print_topics(f"ANSWER TOPICS    (agent_response_en, k={best_ka})", agent_kw)


if __name__ == "__main__":
    main()
