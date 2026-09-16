"""
NER over English Q&A columns using spaCy en_core_web_trf.

Input : Engdata(usage_binatanya_01-03-25_16-11-).csv
Output: entities.csv       (long form: one row per detected entity)
        entity_freq.csv    (label x text frequency)
        entity_by_conv.csv (wide per conversation aggregate)

Usage:
    python3 ner.py                       # process all rows
    python3 ner.py --limit 5000          # smoke test
    python3 ner.py --batch 64 --gpu      # tune throughput

Setup:
    pip install spacy spacy-transformers pandas beautifulsoup4 tqdm
    python -m spacy download en_core_web_trf
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import pandas as pd
import spacy
from bs4 import BeautifulSoup
from tqdm import tqdm

ROOT = Path(__file__).parent
INPUT_CSV = ROOT / "Engdata(usage_binatanya_01-03-25_16-11-).csv"
OUT_ENTITIES = ROOT / "entities.csv"
OUT_FREQ = ROOT / "entity_freq.csv"
OUT_BY_CONV = ROOT / "entity_by_conv.csv"

TEXT_COLS = ("user_message_en", "agent_response_en")
ID_COLS = ("conversation_id", "conversation_turn")

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def clean_text(raw: object) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return ""
    try:
        txt = BeautifulSoup(raw, "html.parser").get_text(" ")
    except Exception:
        txt = TAG_RE.sub(" ", raw)
    txt = html.unescape(txt)
    return WS_RE.sub(" ", txt).strip()


def load_frame(limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)
    keep = list(ID_COLS) + list(TEXT_COLS)
    df = df[keep].copy()
    if limit:
        df = df.head(limit)
    for c in TEXT_COLS:
        df[c] = df[c].map(clean_text)
    return df


def build_records(df: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    records: list[tuple[str, str, str, str]] = []
    for row in df.itertuples(index=False):
        cid = str(getattr(row, "conversation_id"))
        turn = str(getattr(row, "conversation_turn"))
        for text, source in ((row.user_message_en, "user"),
                             (row.agent_response_en, "agent")):
            if text:
                records.append((cid, turn, source, text))
    return records


def run_ner(records, nlp, batch: int) -> list[dict]:
    out: list[dict] = []
    texts = (r[3] for r in records)
    metas = [(r[0], r[1], r[2]) for r in records]
    pipe = nlp.pipe(texts, batch_size=batch)
    for meta, doc in tqdm(zip(metas, pipe), total=len(metas), desc="ner"):
        cid, turn, source = meta
        for ent in doc.ents:
            out.append({
                "conversation_id": cid,
                "conversation_turn": turn,
                "source": source,
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })
    return out


def summarise(ents: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    freq = (ents.groupby(["label", "text"])
                .size().reset_index(name="count")
                .sort_values(["label", "count"], ascending=[True, False]))
    by_conv = (ents.groupby(["conversation_id", "label"])
                   .size().unstack(fill_value=0).reset_index())
    return freq, by_conv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gpu", action="store_true", help="require GPU")
    ap.add_argument("--model", default="en_core_web_trf")
    args = ap.parse_args()

    if args.gpu:
        spacy.require_gpu()
    else:
        spacy.prefer_gpu()

    nlp = spacy.load(args.model, disable=["parser", "lemmatizer",
                                          "attribute_ruler", "tagger"])

    df = load_frame(args.limit)
    print(f"rows: {len(df):,}  cols: {list(df.columns)}")

    records = build_records(df)
    print(f"non-empty text units: {len(records):,}")

    rows = run_ner(records, nlp, args.batch)
    ents = pd.DataFrame(rows)
    ents.to_csv(OUT_ENTITIES, index=False)
    print(f"wrote {OUT_ENTITIES.name}  ({len(ents):,} entities)")

    if ents.empty:
        return
    freq, by_conv = summarise(ents)
    freq.to_csv(OUT_FREQ, index=False)
    by_conv.to_csv(OUT_BY_CONV, index=False)
    print(f"wrote {OUT_FREQ.name}  ({len(freq):,} label/text pairs)")
    print(f"wrote {OUT_BY_CONV.name}  ({len(by_conv):,} conversations)")


if __name__ == "__main__":
    main()
