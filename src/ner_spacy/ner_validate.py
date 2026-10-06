"""
Validation & agreement checks (ner_pipeline_mvp.md §6 step 7 + §9):

  [a] GLM-5.3-Flash reasoning=low  vs  reasoning=high  → agreement rate
      (high agreement ⇒ low is sufficient for the full corpus, saves cost)
  [b] GLM  vs  spaCy en_core_web_trf, label-mapped
      (LOCATION↔GPE/LOC/FAC, DATE_TIME↔DATE/TIME, PERSON↔PERSON, …)
  [c] Verbatim hallucination rate per GLM run

Output: agreement_report.md

Usage:
    python ner_validate.py --sample 200
    python ner_validate.py --sample 200 --skip-spacy
    python ner_validate.py --sample 200 --spacy-python .venv312/bin/python
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ner_glm_pipeline import (
    ROOT,
    TAXONOMY,
    AsyncOpenAI,
    asyncio,
    build_records,
    call_glm,
    load_and_clean,
    verify_verbatim,
)

OUT_REPORT = ROOT / "agreement_report.md"

SPACY_MODEL = "en_core_web_trf"
SPACY_TO_TAX = {
    "GPE": "LOCATION", "LOC": "LOCATION", "FAC": "LOCATION",
    "DATE": "DATE_TIME", "TIME": "DATE_TIME",
    "PERSON": "PERSON",
    "ORG": "DEPARTMENT",
    "LAW": "LAW_REGULATION",
    "MONEY": "MONEY_FEE",
}


# ---------------------------------------------------------------------------
# Entity-set comparison helpers
# ---------------------------------------------------------------------------
def norm_text(s: str) -> str:
    return " ".join(str(s).split()).lower()


def entity_key_set(entities: list[dict], label_map: dict | None = None) -> set:
    """(label, normalized_text) set; labels not in label_map are dropped."""
    out = set()
    for e in entities:
        lab = e["label"]
        if label_map is not None:
            lab = label_map.get(lab)
            if lab is None:
                continue
        out.add((lab, norm_text(e["text"])))
    return out


def prf(pred: set, gold: set) -> tuple[float, float, float]:
    """Micro precision/recall/F1 over entity sets. Both empty ⇒ 1.0 agreement."""
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def agreement(per_msg_a: dict[str, list], per_msg_b: dict[str, list],
              all_msg_ids: list[str], label_map_b: dict | None = None) -> dict:
    """Micro P/R/F1 + per-label exact-match counts. A=GLM(own labels), B=other method."""
    tp = fp = fn = 0
    f1s: list[float] = []
    label_stats = {lab: {"a": 0, "b": 0, "both": 0} for lab in TAXONOMY}
    n_both_empty = 0
    for mid in all_msg_ids:
        ka = entity_key_set(per_msg_a.get(mid, []))
        kb = entity_key_set(per_msg_b.get(mid, []), label_map_b)
        _, _, f = prf(ka, kb)
        f1s.append(f)
        tp += len(ka & kb)
        fp += len(ka - kb)
        fn += len(kb - ka)
        if not ka and not kb:
            n_both_empty += 1
        for lab, txt in ka:
            label_stats[lab]["a"] += 1
            if (lab, txt) in kb:
                label_stats[lab]["both"] += 1
        for lab, txt in kb:
            label_stats[lab]["b"] += 1
    micro_p = tp / (tp + fp) if (tp + fp) else 1.0
    micro_r = tp / (tp + fn) if (tp + fn) else 1.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    return {
        "micro_p": micro_p, "micro_r": micro_r, "micro_f": micro_f,
        "mean_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
        "labels": label_stats,
        "n_both_empty": n_both_empty,
        "n_msgs": len(all_msg_ids),
    }


# ---------------------------------------------------------------------------
# GLM run at a given reasoning effort
# ---------------------------------------------------------------------------
async def run_glm_variant(client: AsyncOpenAI, semaphore: asyncio.Semaphore,
                          batches: list[list[dict]], lookup: dict[str, dict],
                          model: str, effort: str, mask_pii: bool) -> dict:
    per_msg: dict[str, list] = {}
    usage_tot = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    t0 = time.time()
    versions: set[str] = set()
    n_halluc = 0
    n_total = 0
    n_failed_batches = 0
    for batch in batches:
        results, usage, mver = await call_glm(
            client, batch, model, semaphore, mask_pii=mask_pii, reasoning_effort=effort)
        if usage is None and not results:
            n_failed_batches += 1
        if mver:
            versions.add(mver)
        if usage:
            usage_tot["prompt_tokens"] += usage.get("prompt_tokens", 0)
            usage_tot["completion_tokens"] += usage.get("completion_tokens", 0)
            usage_tot["total_tokens"] += usage.get("total_tokens", 0)
        for res in results:
            mid = res.get("msg_id", "")
            orig = lookup.get(mid, {}).get("text", "")
            for ent in res.get("entities", []):
                n_total += 1
                if verify_verbatim(orig, ent["text"]):
                    per_msg.setdefault(mid, []).append({"label": ent["label"], "text": ent["text"]})
                else:
                    n_halluc += 1
    return {
        "per_msg": per_msg,
        "usage": usage_tot,
        "elapsed": time.time() - t0,
        "versions": sorted(versions),
        "n_halluc": n_halluc,
        "n_total": n_total,
        "n_failed_batches": n_failed_batches,
    }


# ---------------------------------------------------------------------------
# spaCy cross-check — runs in .venv312 (spaCy env) via subprocess
# ---------------------------------------------------------------------------
SPACY_WORKER = ROOT / "ner_spacy_worker.py"


def run_spacy(records: list[dict], spacy_python: str) -> dict[str, list] | None:
    """Dump records to temp JSON, run spaCy worker in its own env, read back."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
        in_path = f.name
    out_path = in_path + ".out"
    try:
        proc = subprocess.run(
            [spacy_python, str(SPACY_WORKER), in_path, out_path, SPACY_MODEL],
            capture_output=True, text=True, timeout=3600,
        )
        if proc.returncode != 0:
            print(proc.stderr[-2000:], file=sys.stderr)
            return None
        with open(out_path, encoding="utf-8") as f:
            return {k: v for k, v in json.load(f).items()}
    finally:
        Path(in_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def fmt_row(name: str, r: dict) -> str:
    return (f"| {name} | {r['micro_p']:.3f} | {r['micro_r']:.3f} | {r['micro_f']:.3f} "
            f"| {r['mean_f1']:.3f} | {r['tp']} | {r['fp']} | {r['fn']} |")


def per_label_table(a_name: str, b_name: str, labels: dict) -> str:
    lines = [f"| label | {a_name} count | {b_name} count | exact match |",
             "|---|---|---|---|"]
    for lab, s in labels.items():
        if s["a"] or s["b"]:
            lines.append(f"| {lab} | {s['a']} | {s['b']} | {s['both']} |")
    return "\n".join(lines)


def halluc_table(runs: dict[str, dict]) -> str:
    lines = ["| run | entities returned | unverified (hallucinated) | rate |",
             "|---|---|---|---|"]
    for name, r in runs.items():
        rate = r["n_halluc"] / r["n_total"] * 100 if r["n_total"] else 0.0
        lines.append(f"| {name} | {r['n_total']} | {r['n_halluc']} | {rate:.1f}% |")
    return "\n".join(lines)


def cost_table(runs: dict[str, dict]) -> str:
    lines = ["| run | prompt tokens | completion tokens | wall time (s) |",
             "|---|---|---|---|"]
    for name, r in runs.items():
        u = r["usage"]
        lines.append(f"| {name} | {u['prompt_tokens']:,} | {u['completion_tokens']:,} | {r['elapsed']:.0f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main_async(args):
    from ner_glm_pipeline import logger
    logger.info("=== Validation run (%s) — sample=%d ===",
                datetime.now(timezone.utc).isoformat(), args.sample)

    df = load_and_clean(limit=max(args.sample, 1))
    records = build_records(df)[:args.sample]
    lookup = {r["msg_id"]: r for r in records}
    msg_ids = [r["msg_id"] for r in records]
    batches = [records[i:i + args.batch_size]
               for i in range(0, len(records), args.batch_size)]
    logger.info("Validation: %d messages, %d batches", len(records), len(batches))

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    low = await run_glm_variant(client, semaphore, batches, lookup,
                                args.model, "low", args.mask_pii)
    logger.info("reasoning=low done: %.0fs, %d entities (%d halluc)",
                low["elapsed"], low["n_total"], low["n_halluc"])

    high = await run_glm_variant(client, semaphore, batches, lookup,
                                 args.model, "high", args.mask_pii)
    logger.info("reasoning=high done: %.0fs, %d entities (%d halluc)",
                high["elapsed"], high["n_total"], high["n_halluc"])

    ag_low_high = agreement(low["per_msg"], high["per_msg"], msg_ids)

    ag_low_spacy = ag_high_spacy = None
    spacy_per_msg = None
    if not args.skip_spacy:
        spacy_per_msg = run_spacy(records, args.spacy_python)
        if spacy_per_msg is not None:
            ag_low_spacy = agreement(low["per_msg"], spacy_per_msg, msg_ids, label_map_b=SPACY_TO_TAX)
            ag_high_spacy = agreement(high["per_msg"], spacy_per_msg, msg_ids, label_map_b=SPACY_TO_TAX)

    # ------------------------- write report -------------------------
    lines: list[str] = []
    lines.append("# NER Validation & Agreement Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Sample: {len(records)} messages (first {args.sample} of corpus)")
    lines.append(f"- LLM: {args.model} @ {args.base_url} "
                 f"(model versions: low={', '.join(low['versions']) or 'n/a'}, "
                 f"high={', '.join(high['versions']) or 'n/a'})")
    lines.append(f"- spaCy: {SPACY_MODEL} (label-mapped, see §2)" if spacy_per_msg
                 else f"- spaCy: SKIPPED")
    lines.append(f"- Decoding: temperature=0, function-calling schema, verbatim verification")
    lines.append("")
    lines.append("Metrics: entity match = (label, lowercase-normalized text) exact pair. "
                 "Messages with no entities from either side count as full agreement.")
    lines.append("")

    lines.append("## 1. GLM reasoning=low vs reasoning=high")
    lines.append("")
    lines.append("| comparison | micro P | micro R | micro F1 | mean per-msg F1 | TP | FP | FN |")
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append(fmt_row("low vs high", ag_low_high))
    lines.append("")
    lines.append(f"Messages where both found nothing: {ag_low_high['n_both_empty']}/{ag_low_high['n_msgs']}")
    lines.append("")
    lines.append("### Per-label overlap")
    lines.append("")
    lines.append(per_label_table("low", "high", ag_low_high["labels"]))
    lines.append("")
    lines.append("### Cost / latency (sample only)")
    lines.append("")
    lines.append(cost_table({"reasoning=low": low, "reasoning=high": high}))
    lines.append("")
    verdict = ("**low IS sufficient** for the full corpus (mean F1 ≥ 0.90)"
               if ag_low_high["mean_f1"] >= 0.90 else
               "**low is NOT clearly sufficient** — consider running high for the full corpus")
    lines.append(f"Verdict: {verdict} (spec §9).")
    lines.append("")

    if spacy_per_msg is not None:
        lines.append("## 2. GLM vs spaCy en_core_web_trf (label-mapped)")
        lines.append("")
        lines.append("spaCy → taxonomy label mapping:")
        lines.append("")
        lines.append("| spaCy label | taxonomy label |")
        lines.append("|---|---|")
        for k, v in sorted(SPACY_TO_TAX.items()):
            lines.append(f"| {k} | {v} |")
        lines.append("")
        lines.append("Unmapped spaCy labels (CARDINAL, ORDINAL, QUANTITY, PERCENT, NORP, "
                     "EVENT, PRODUCT, WORK_OF_ART, LANGUAGE) are excluded from comparison.")
        lines.append("")
        lines.append("| comparison | micro P | micro R | micro F1 | mean per-msg F1 | TP | FP | FN |")
        lines.append("|---|---|---|---|---|---|---|---|")
        lines.append(fmt_row("GLM low vs spaCy", ag_low_spacy))
        lines.append(fmt_row("GLM high vs spaCy", ag_high_spacy))
        lines.append("")
        lines.append("### Per-label overlap (GLM low vs spaCy)")
        lines.append("")
        lines.append(per_label_table("GLM low", "spaCy", ag_low_spacy["labels"]))
        lines.append("")
    else:
        lines.append("## 2. GLM vs spaCy — SKIPPED")
        lines.append("")
        lines.append("Run with a Python env that has `en_core_web_trf` installed, e.g.:")
        lines.append("    python ner_validate.py --sample 200 --spacy-python .venv312/bin/python")
        lines.append("")

    lines.append("## 3. Hallucination rate (verbatim verification, threshold 95)")
    lines.append("")
    lines.append(halluc_table({"reasoning=low": low, "reasoning=high": high}))
    lines.append("")
    if low["n_failed_batches"] or high["n_failed_batches"]:
        lines.append(f"Failed batches (excluded): low={low['n_failed_batches']}, high={high['n_failed_batches']}")
        lines.append("")
    lines.append("## 4. Manual spot-check (face validity)")
    lines.append("")
    lines.append("- [ ] TODO: manually review 50 sampled messages + extracted entities "
                 "(spec §9). Not used for training — face validity only.")
    lines.append("")
    lines.append("## 5. Limitations (for Methods)")
    lines.append("")
    lines.append("- Zero-shot prompt-based extraction, not deterministic across model snapshots "
                 "— log model version string (in cost_log.csv / above).")
    lines.append("- Taxonomy is hypothesis-driven (researcher-defined), not an industry standard.")
    lines.append("- spaCy cross-check uses a label mapping; overlap is a lower bound on agreement "
                 "since spaCy lacks municipal-specific types (PERMIT_DOC, SERVICE_REQUEST, …).")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", OUT_REPORT.name)

    # Console summary
    print(f"\nlow vs high  : micro F1={ag_low_high['micro_f']:.3f}  mean F1={ag_low_high['mean_f1']:.3f}")
    if ag_low_spacy:
        print(f"low vs spaCy : micro F1={ag_low_spacy['micro_f']:.3f}  mean F1={ag_low_spacy['mean_f1']:.3f}")
    print(f"hallucination: low={low['n_halluc']}/{low['n_total']}  high={high['n_halluc']}/{high['n_total']}")


def main():
    ap = argparse.ArgumentParser(description="NER validation — GLM vs GLM-high vs spaCy")
    ap.add_argument("--sample", type=int, default=200,
                    help="Number of messages to validate (default: 200)")
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--base-url", default="https://api.int2.net/v1")
    ap.add_argument("--api-key", default=os.environ.get("GLM_API_KEY", ""))
    ap.add_argument("--mask-pii", action="store_true")
    ap.add_argument("--skip-spacy", action="store_true",
                    help="Only run GLM low vs high")
    ap.add_argument("--spacy-python", default=str(ROOT / ".venv312" / "bin" / "python"),
                    help="Python interpreter that has spaCy + en_core_web_trf")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
