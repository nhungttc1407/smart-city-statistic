"""
NER Pipeline MVP — GLM-5.3-Flash zero-shot extraction.

Implements the full pipeline from ner_pipeline_mvp.md:
  [1] Ingest & Clean
  [2] Batch & Prompt Build
  [3] GLM-5.3-Flash Inference (function calling, temp=0, reasoning=low)
  [4] Verbatim Verification
  [5] Aggregate
  [6] Cross-analysis (entity_by_sector, entity_by_topic)

Usage:
    python ner_glm_pipeline.py --sample 200          # dry run
    python ner_glm_pipeline.py --batch-size 25       # messages per API call
    python ner_glm_pipeline.py --resume               # continue from checkpoint
    python ner_glm_pipeline.py --concurrency 8        # parallel API calls
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from openai import AsyncOpenAI
from rapidfuzz import fuzz

import warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
NER_OUT = ROOT / "outputs" / "ner_glm"
NER_OUT.mkdir(parents=True, exist_ok=True)

INPUT_CSV = DATA_DIR / "Engdata(usage_binatanya_01-03-25_16-11-).csv"
OUT_RAW = NER_OUT / "entities_glm_raw.csv"
OUT_VERIFIED = NER_OUT / "entities_glm.csv"
OUT_UNVERIFIED = NER_OUT / "unverified_entities.csv"
OUT_FREQ = NER_OUT / "entity_freq_glm.csv"
OUT_BY_CONV = NER_OUT / "entity_by_conv_glm.csv"
OUT_COST_LOG = NER_OUT / "cost_log.csv"
OUT_CHECKPOINT = NER_OUT / "ner_glm_checkpoint.json"
OUT_LOG = NER_OUT / "ner_glm_full.log"

TEXT_COLS = ("user_message_en", "agent_response_en")
ID_COLS = ("conversation_id", "conversation_turn")

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------
TAXONOMY = [
    "LOCATION", "DEPARTMENT", "DATE_TIME", "MONEY_FEE", "PERMIT_DOC",
    "SERVICE_REQUEST", "CONTACT_INFO", "LAW_REGULATION", "PERSON",
]

# ---------------------------------------------------------------------------
# Function calling schema
# ---------------------------------------------------------------------------
EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_entities",
        "description": "Extract named entities per message from a batch",
        "parameters": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "msg_id": {"type": "string"},
                            "entities": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "text": {"type": "string", "description": "exact substring copied verbatim from input"},
                                        "label": {"type": "string", "enum": TAXONOMY},
                                    },
                                    "required": ["text", "label"],
                                },
                            },
                        },
                        "required": ["msg_id", "entities"],
                    },
                },
            },
            "required": ["results"],
        },
    },
}

SYSTEM_PROMPT = (
    "You are a precise information-extraction system. Extract entities strictly "
    "using the provided taxonomy. Copy `text` VERBATIM from the input — do not "
    "paraphrase, translate, or normalize. If no entity of a type exists, omit it. "
    "Do not invent entities."
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("ner_glm")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
logger.addHandler(_sh)
_fh = logging.FileHandler(OUT_LOG, mode="a", encoding="utf-8")
_fh.setFormatter(_fmt)
logger.addHandler(_fh)

# ---------------------------------------------------------------------------
# [1] Ingest & Clean
# ---------------------------------------------------------------------------
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

# PII patterns — mask before sending to external API
PII_PHONE = re.compile(r"(?<!\d)(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}(?!\d)")
PII_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PII_ID = re.compile(r"\b\d{6,12}\b")  # rough ID number pattern


def clean_text(raw: object) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return ""
    try:
        txt = BeautifulSoup(raw, "html.parser").get_text(" ")
    except Exception:
        txt = TAG_RE.sub(" ", raw)
    txt = html.unescape(txt)
    return WS_RE.sub(" ", txt).strip()



def load_and_clean(limit: int | None = None) -> pd.DataFrame:
    logger.info("Loading CSV …")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)
    keep = list(ID_COLS) + list(TEXT_COLS)
    df = df[keep].copy()
    if limit:
        df = df.head(limit)
    for c in TEXT_COLS:
        df[c] = df[c].map(clean_text)
    logger.info("Loaded %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# [2] Batch & Prompt Build
# ---------------------------------------------------------------------------
def build_records(df: pd.DataFrame) -> list[dict]:
    """Flatten df into list of {msg_id, conversation_id, conversation_turn, source, text}."""
    records = []
    idx = 0
    for row in df.itertuples(index=False):
        cid = str(getattr(row, "conversation_id"))
        turn = str(getattr(row, "conversation_turn"))
        for text, source in ((row.user_message_en, "user"),
                             (row.agent_response_en, "agent")):
            if text:
                records.append({
                    "msg_id": f"{idx:07d}",
                    "conversation_id": cid,
                    "conversation_turn": turn,
                    "source": source,
                    "text": text,
                })
                idx += 1
    return records


def build_batch_prompt(batch: list[dict], mask_pii: bool = False) -> str:
    lines = []
    for r in batch:
        text = mask_pii_text(r["text"]) if mask_pii else r["text"]
        lines.append(f"[{r['msg_id']}] {text}")
    return "Extract entities for each numbered message:\n\n" + "\n".join(lines)


def mask_pii_text(text: str) -> str:
    """Mask PII before sending to external API."""
    text = PII_PHONE.sub("[PHONE]", text)
    text = PII_EMAIL.sub("[EMAIL]", text)
    return text


# ---------------------------------------------------------------------------
# [3] GLM-5.3-Flash Inference
# ---------------------------------------------------------------------------
async def call_glm(
    client: AsyncOpenAI,
    batch: list[dict],
    model: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
    mask_pii: bool = False,
    reasoning_effort: str = "low",
) -> tuple[list[dict], dict | None, str | None]:
    """Call GLM with function calling. Returns (results, usage_dict, model_version)."""
    prompt = build_batch_prompt(batch, mask_pii=mask_pii)
    async with semaphore:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    temperature=0,
                    reasoning_effort=reasoning_effort,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    tools=[EXTRACT_TOOL],
                    tool_choice={"type": "function", "function": {"name": "extract_entities"}},
                )
                call = resp.choices[0].message.tool_calls[0]
                parsed = json.loads(call.function.arguments)
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                    "total_tokens": resp.usage.total_tokens if resp.usage else 0,
                }
                return parsed.get("results", []), usage, getattr(resp, "model", model)

            except Exception as e:
                logger.warning("batch %s..%s attempt %d failed: %s",
                               batch[0]["msg_id"], batch[-1]["msg_id"], attempt + 1, e)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    # Binary-split retry
                    if len(batch) > 1:
                        mid = len(batch) // 2
                        logger.info("binary-split: %d msgs → %d + %d",
                                    len(batch), mid, len(batch) - mid)
                        left, u1, m1 = await call_glm(client, batch[:mid], model, semaphore, max_retries, mask_pii, reasoning_effort)
                        right, u2, m2 = await call_glm(client, batch[mid:], model, semaphore, max_retries, mask_pii, reasoning_effort)
                        merged = {}
                        if u1:
                            merged["prompt_tokens"] = merged.get("prompt_tokens", 0) + u1.get("prompt_tokens", 0)
                            merged["completion_tokens"] = merged.get("completion_tokens", 0) + u1.get("completion_tokens", 0)
                            merged["total_tokens"] = merged.get("total_tokens", 0) + u1.get("total_tokens", 0)
                        if u2:
                            merged["prompt_tokens"] = merged.get("prompt_tokens", 0) + u2.get("prompt_tokens", 0)
                            merged["completion_tokens"] = merged.get("completion_tokens", 0) + u2.get("completion_tokens", 0)
                            merged["total_tokens"] = merged.get("total_tokens", 0) + u2.get("total_tokens", 0)
                        return left + right, merged or None, m1 or m2
                    return [], None, None
    return [], None, None


# ---------------------------------------------------------------------------
# [4] Verbatim Verification
# ---------------------------------------------------------------------------
def verify_verbatim(text_original: str, entity_text: str, threshold: int = 95) -> bool:
    if entity_text in text_original:
        return True
    return fuzz.partial_ratio(entity_text, text_original) >= threshold


# ---------------------------------------------------------------------------
# [5] Aggregate
# ---------------------------------------------------------------------------
def summarise(ents: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    freq = (ents.groupby(["label", "entity_text"])
                .size().reset_index(name="count")
                .sort_values(["label", "count"], ascending=[True, False]))
    by_conv = (ents.groupby(["conversation_id", "label"])
                   .size().unstack(fill_value=0).reset_index())
    return freq, by_conv


# ---------------------------------------------------------------------------
# [6] Cross-analysis
# ---------------------------------------------------------------------------
OUT_BY_SECTOR = NER_OUT / "entity_by_sector.csv"
OUT_BY_TOPIC = NER_OUT / "entity_by_topic.csv"
SECTOR_CSV = ROOT / "outputs" / "topic" / "labels" / "sector_labels.csv"
TOPIC_CSV = ROOT / "outputs" / "topic" / "labels" / "conversation_labels_combined.csv"


def cross_analysis(verified: pd.DataFrame) -> None:
    """Join verified entities with sector/topic conversation labels (spec §6 step 6)."""
    if SECTOR_CSV.exists():
        sec = pd.read_csv(SECTOR_CSV, low_memory=False)
        if "sector" in sec.columns:
            sec = sec[["conversation_id", "sector"]].drop_duplicates("conversation_id")
            merged = verified.merge(sec, on="conversation_id", how="left")
            merged["sector"] = merged["sector"].fillna("Unknown")
            by_sector = merged.groupby(["sector", "label"]).size().unstack(fill_value=0)
            by_sector.to_csv(OUT_BY_SECTOR)
            logger.info("Wrote %s (%d sectors)", OUT_BY_SECTOR.name, len(by_sector))
    else:
        logger.warning("sector_labels.csv not found — skipped %s", OUT_BY_SECTOR.name)

    if TOPIC_CSV.exists():
        top = pd.read_csv(TOPIC_CSV, low_memory=False)
        cols = [c for c in ("conversation_id", "dominant_topic", "dominant_topic_name")
                if c in top.columns]
        top = top[cols].drop_duplicates("conversation_id")
        merged = verified.merge(top, on="conversation_id", how="left")
        if "dominant_topic_name" in merged.columns:
            merged["dominant_topic_name"] = merged["dominant_topic_name"].fillna("Unknown")
            by_topic = merged.groupby(["dominant_topic_name", "label"]).size().unstack(fill_value=0)
            by_topic.to_csv(OUT_BY_TOPIC)
            logger.info("Wrote %s (%d topics)", OUT_BY_TOPIC.name, len(by_topic))
    else:
        logger.warning("conversation_labels_combined.csv not found — skipped %s", OUT_BY_TOPIC.name)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def load_checkpoint() -> set[str]:
    if OUT_CHECKPOINT.exists():
        data = json.loads(OUT_CHECKPOINT.read_text())
        return set(data.get("processed_ids", []))
    return set()


def save_checkpoint(processed: set[str]):
    OUT_CHECKPOINT.write_text(json.dumps({"processed_ids": sorted(processed)}, indent=0))


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------
def estimate_cost(total_input_tokens: int, total_output_tokens: int) -> dict:
    input_cost = total_input_tokens / 1_000_000 * 0.15
    output_cost = total_output_tokens / 1_000_000 * 0.50
    return {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
async def run_pipeline(args: argparse.Namespace):
    logger.info("Run %s | model=%s | provider(base_url)=%s | reasoning=low | temp=0",
                datetime.now(timezone.utc).isoformat(), args.model, args.base_url)
    df = load_and_clean(limit=args.sample)
    records = build_records(df)
    logger.info("Non-empty text units: %d", len(records))

    # [1] clean_messages.parquet (spec §6 step 1 output)
    try:
        df.to_parquet(ROOT / "clean_messages.parquet", index=False)
        logger.info("Wrote clean_messages.parquet (%d rows)", len(df))
    except Exception as e:
        logger.warning("Skipped clean_messages.parquet (%s) — pip install pyarrow to enable", type(e).__name__)

    # Load checkpoint
    processed = set()
    if args.resume:
        processed = load_checkpoint()
        if processed:
            before = len(records)
            records = [r for r in records if r["msg_id"] not in processed]
            logger.info("Resume: skipped %d already-processed, %d remaining",
                        before - len(records), len(records))

    if not records:
        logger.info("Nothing to process.")
        return

    # Chunk into batches
    batches = [records[i:i + args.batch_size]
               for i in range(0, len(records), args.batch_size)]
    logger.info("Batches: %d (batch_size=%d)", len(batches), args.batch_size)

    # Setup client
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    # Run batches with progress tracking
    all_results: list[dict] = []
    cost_rows: list[dict] = []
    total_input = 0
    total_output = 0
    t_start = time.time()

    # Process in chunks to avoid overwhelming — track progress
    processed_batch_count = 0
    model_versions: set[str] = set()
    for batch in batches:
        results, usage, resp_model = await call_glm(client, batch, args.model, semaphore, mask_pii=args.mask_pii)
        processed_batch_count += 1
        if resp_model:
            model_versions.add(resp_model)

        # Enrich results with conversation metadata
        batch_lookup = {r["msg_id"]: r for r in batch}
        for res in results:
            msg_id = res.get("msg_id", "")
            meta = batch_lookup.get(msg_id, {})
            for ent in res.get("entities", []):
                all_results.append({
                    "msg_id": msg_id,
                    "conversation_id": meta.get("conversation_id", ""),
                    "conversation_turn": meta.get("conversation_turn", ""),
                    "source": meta.get("source", ""),
                    "text_original": meta.get("text", ""),
                    "entity_text": ent["text"],
                    "label": ent["label"],
                })

        if usage:
            total_input += usage.get("prompt_tokens", 0)
            total_output += usage.get("completion_tokens", 0)
            cost_rows.append({
                "batch_idx": processed_batch_count,
                "msg_ids": f"{batch[0]['msg_id']}..{batch[-1]['msg_id']}",
                "n_messages": len(batch),
                "n_entities_raw": sum(len(r.get("entities", [])) for r in results),
                "model": resp_model or args.model,
                **usage,
            })

        # Checkpoint
        for r in batch:
            processed.add(r["msg_id"])
        save_checkpoint(processed)

        # Progress
        elapsed = time.time() - t_start
        rate = processed_batch_count / elapsed if elapsed > 0 else 0
        eta = (len(batches) - processed_batch_count) / rate if rate > 0 else 0
        logger.info(
            "batch %d/%d  entities=%d  rate=%.1f batch/s  ETA=%.0fs",
            processed_batch_count, len(batches),
            sum(len(r.get("entities", [])) for r in results),
            rate, eta,
        )

    # Save raw results
    df_raw = pd.DataFrame(all_results)
    df_raw.to_csv(OUT_RAW, index=False)
    logger.info("Wrote %s (%d raw entities)", OUT_RAW.name, len(df_raw))

    # Save cost log
    if cost_rows:
        df_cost = pd.DataFrame(cost_rows)
        df_cost.to_csv(OUT_COST_LOG, index=False)
        est = estimate_cost(total_input, total_output)
        logger.info(
            "Tokens — input: %s  output: %s  cost: $%s",
            f"{est['input_tokens']:,}", f"{est['output_tokens']:,}", est['total_cost_usd'],
        )

    # ---------------------------------------------------------------------------
    # [4] Verbatim Verification
    # ---------------------------------------------------------------------------
    if df_raw.empty:
        logger.info("No entities to verify.")
        return

    logger.info("Verifying verbatim match …")
    df_raw["verified"] = df_raw.apply(
        lambda row: verify_verbatim(row["text_original"], row["entity_text"]),
        axis=1,
    )
    verified = df_raw[df_raw["verified"]].drop(columns=["verified", "text_original"])
    unverified = df_raw[~df_raw["verified"]].copy()

    verified.to_csv(OUT_VERIFIED, index=False)
    logger.info("Wrote %s (%d verified entities)", OUT_VERIFIED.name, len(verified))

    if not unverified.empty:
        unverified.to_csv(OUT_UNVERIFIED, index=False)
        logger.info("Wrote %s (%d unverified — possible hallucinations)",
                     OUT_UNVERIFIED.name, len(unverified))
        hallucination_rate = len(unverified) / len(df_raw) * 100
        logger.info("Hallucination rate: %.1f%%", hallucination_rate)

    # ---------------------------------------------------------------------------
    # [5] Aggregate
    # ---------------------------------------------------------------------------
    if verified.empty:
        logger.info("No verified entities to aggregate.")
        return

    freq, by_conv = summarise(verified)
    freq.to_csv(OUT_FREQ, index=False)
    by_conv.to_csv(OUT_BY_CONV, index=False)
    logger.info("Wrote %s (%d label/text pairs)", OUT_FREQ.name, len(freq))
    logger.info("Wrote %s (%d conversations)", OUT_BY_CONV.name, len(by_conv))

    # ---------------------------------------------------------------------------
    # [6] Cross-analysis: entity_by_sector.csv, entity_by_topic.csv
    # ---------------------------------------------------------------------------
    cross_analysis(verified)

    # Final summary
    elapsed = time.time() - t_start
    logger.info("Pipeline complete in %.1fs", elapsed)
    if cost_rows:
        est = estimate_cost(total_input, total_output)
        logger.info("Estimated cost: $%.4f", est["total_cost_usd"])
    if model_versions:
        logger.info("Model version(s) returned by provider: %s", ", ".join(sorted(model_versions)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="NER Pipeline MVP — GLM-5.3-Flash")
    ap.add_argument("--sample", type=int, default=None,
                    help="Process only first N rows (for testing)")
    ap.add_argument("--batch-size", type=int, default=25,
                    help="Messages per API call (default: 25)")
    ap.add_argument("--concurrency", type=int, default=5,
                    help="Max concurrent API calls (default: 5)")
    ap.add_argument("--model", default="glm-5.3-flash",
                    help="Model name (default: glm-5.3-flash)")
    # ap.add_argument("--base-url", default="https://api.int2.net/v1",
    #                 help="OpenAI-compatible API base URL")
    # ap.add_argument("--api-key", ...)
    ap.add_argument("--resume", action="store_true",
                    help="Resume from checkpoint")
    ap.add_argument("--mask-pii", action="store_true",
                    help="Mask phone/email before sending to API (PII protection)")
    args = ap.parse_args()

    # Run
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
