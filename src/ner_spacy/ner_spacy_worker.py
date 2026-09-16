"""spaCy NER worker — called by ner_validate.py in the spaCy env (.venv312).

Usage: python ner_spacy_worker.py in.json out.json [model]
Reads [{msg_id, text}, ...], writes {msg_id: [{label, text}, ...]}.
"""

import json
import sys

import spacy


def main() -> None:
    in_path, out_path = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "en_core_web_trf"
    nlp = spacy.load(model, disable=["parser", "lemmatizer",
                                     "attribute_ruler", "tagger"])
    with open(in_path, encoding="utf-8") as f:
        records = json.load(f)

    result: dict[str, list] = {}
    texts = [r["text"] for r in records]
    for rec, doc in zip(records, nlp.pipe(texts, batch_size=32)):
        result[rec["msg_id"]] = [{"label": e.label_, "text": e.text} for e in doc.ents]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
