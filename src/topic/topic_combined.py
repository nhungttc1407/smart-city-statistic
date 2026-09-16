"""
Combined Q+A topic model.

Reads merged_conversations.csv (one row per conversation_id with user_doc and
agent_doc already concatenated across turns), joins user_doc + agent_doc into
a single paragraph per conversation, sweeps candidate k, fits the final LDA
at the best k, and writes:

  - topic_perplexity_sweep_combined.csv  (now also includes umass coherence)
  - topic_keywords_combined.csv
  - topic_distribution_combined.csv

Model-selection criterion
-------------------------
Best k = the k that MAXIMIZES mean NPMI (normalized pointwise mutual
information) topic coherence over the candidate range. For a topic's top N
words, NPMI(w_i, w_j) = log(P(w_i, w_j) / (P(w_i) P(w_j))) / -log(P(w_i, w_j)),
clipped to [-1, 1] (1 = always co-occur, 0 = independent, -1 = never
co-occur). The model score is the mean NPMI over all word pairs in all
topics. Unlike UMass coherence, NPMI does not reward degenerate small-k
models (two very common words are independent, so NPMI ≈ 0 rather than
high). Unlike held-out perplexity, NPMI has an interior optimum instead of
running away with k. Perplexity is still recorded for reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
LDA_DIR = ROOT / "outputs" / "topic" / "lda"
LDA_DIR.mkdir(parents=True, exist_ok=True)

MERGED_CSV = DATA_DIR / "merged_conversations.csv"
PERPLEXITY_CSV = LDA_DIR / "topic_perplexity_sweep_combined.csv"
TOPIC_KEYWORDS_CSV = LDA_DIR / "topic_keywords_combined.csv"
TOPIC_DIST_CSV = LDA_DIR / "topic_distribution_combined.csv"

CANDIDATE_KS = list(range(1, 21))
N_TOP_WORDS = 15

EXTRA_STOPWORDS = {
    "href", "https", "http", "target", "blank", "www", "il", "aspx",
    "muni", "muninetanya", "pages", "html", "br", "li", "ul",
    "netanya",
}


def vectorize(docs: list[str], extra_stop: set[str]) -> tuple:
    vec = CountVectorizer(
        max_df=0.90,
        min_df=5,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out()
    keep_mask = np.array([v not in extra_stop for v in vocab])
    X = X[:, keep_mask]
    vocab = vocab[keep_mask]
    return X, vocab


def npmi_coherence(components: np.ndarray, X_bin: sparse.csc_matrix,
                   top_n: int = 10) -> float:
    """Mean NPMI coherence across topics, averaged over word pairs.

    NPMI(w_i, w_j) = log(P(w_i, w_j) / (P(w_i) P(w_j))) / -log(P(w_i, w_j)),
    using document-level probabilities. NPMI ∈ [-1, 1] with 1 = perfect
    co-occurrence, 0 = independent, -1 = never co-occur. Pairs that never
    co-occur are pinned to -1. The model score is the mean NPMI across all
    top-word pairs across all topics; higher is better.
    """
    n_docs = X_bin.shape[0]
    df_w = np.asarray(X_bin.sum(axis=0)).ravel().astype(np.float64)
    p_w = df_w / n_docs
    idx_i, idx_j = np.tril_indices(top_n, k=-1)

    scores = []
    for comp in components:
        top = comp.argsort()[::-1][:top_n]
        sub = X_bin[:, top]
        co = (sub.T @ sub).toarray().astype(np.float64)
        co_pair = co[idx_i, idx_j]
        p_ij = co_pair / n_docs
        p_i, p_j = p_w[top[idx_i]], p_w[top[idx_j]]
        npmi = np.full(co_pair.shape, -1.0)
        mask = co_pair > 0
        npmi[mask] = np.log(p_ij[mask] / (p_i[mask] * p_j[mask])) / (-np.log(p_ij[mask]))
        scores.append(float(npmi.mean()))
    return float(np.mean(scores))


def sweep_topics(X, ks: list[int], label: str) -> tuple[int, pd.DataFrame]:
    """Sweep k and pick the value that maximizes mean NPMI topic coherence.

    Perplexity is also computed (on a held-out split) and recorded, but is no
    longer the selection criterion — on this corpus perplexity decreases
    monotonically with k and so always picks the boundary of the range.
    """
    train_idx, test_idx = train_test_split(
        np.arange(X.shape[0]), test_size=0.2, random_state=42
    )
    X_tr, X_te = X[train_idx], X[test_idx]
    X_bin = sparse.csc_matrix((X > 0).astype(np.int32))

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
        coh = npmi_coherence(lda.components_, X_bin, top_n=10)
        print(f"  [{label}] k={k:>3}  perplexity={ppl:>8,.1f}  npmi={coh:>7.4f}",
              flush=True)
        rows.append({"source": label, "k": k, "perplexity": ppl, "npmi_coherence": coh})

    sweep_df = pd.DataFrame(rows)
    best_k = int(sweep_df.loc[sweep_df["npmi_coherence"].idxmax(), "k"])
    print(f"  [{label}] BEST k = {best_k}  (by max NPMI coherence)", flush=True)
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


def main() -> None:
    print(f"Reading {MERGED_CSV.name}", flush=True)
    merged = pd.read_csv(MERGED_CSV)
    merged["user_doc"] = merged["user_doc"].fillna("").astype(str)
    merged["agent_doc"] = merged["agent_doc"].fillna("").astype(str)

    # One paragraph per conversation: question + answer text together.
    combined_docs = (merged["user_doc"] + " \n " + merged["agent_doc"]).tolist()
    print(f"  {len(combined_docs):,} combined Q+A documents", flush=True)

    print("\nVectorizing combined docs ...", flush=True)
    X, vocab = vectorize(combined_docs, EXTRA_STOPWORDS)
    print(f"  matrix: {X.shape}, vocab={len(vocab)}", flush=True)

    print("\nSweeping k for COMBINED side ...", flush=True)
    best_k, sweep_df = sweep_topics(X, CANDIDATE_KS, "combined")
    sweep_df.to_csv(PERPLEXITY_CSV, index=False)
    print(f"wrote {PERPLEXITY_CSV.name}", flush=True)

    print(f"\nFitting final COMBINED LDA with k={best_k} ...", flush=True)
    kw_df, doc_topic = fit_final(X, vocab, best_k, "combined")
    kw_df.to_csv(TOPIC_KEYWORDS_CSV, index=False)
    print(f"wrote {TOPIC_KEYWORDS_CSV.name}", flush=True)

    dist = pd.concat(
        [
            merged[["conversation_id", "n_turns"]].reset_index(drop=True),
            pd.DataFrame(doc_topic, columns=[f"combined_topic_{i}" for i in range(best_k)]),
        ],
        axis=1,
    )
    dist.to_csv(TOPIC_DIST_CSV, index=False)
    print(f"wrote {TOPIC_DIST_CSV.name}", flush=True)

    print(f"\n=== COMBINED TOPICS (Q + A merged, k={best_k}) ===", flush=True)
    for _, r in kw_df.iterrows():
        print(f"  topic {r['topic']:>2}: {r['top_keywords']}", flush=True)


if __name__ == "__main__":
    main()
