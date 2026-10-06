"""Single-sentence NER via int2.net GLM."""
import json
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.int2.net/v1",
    api_key=os.environ["GLM_API_KEY"],
)

SYSTEM = (
    "You are a Named Entity Recognition engine. "
    "Extract entities from the user text. "
    'Return ONLY JSON: {"entities": [{"text": str, "label": str, "start": int, "end": int}]}. '
    "Labels: PERSON, ORG, GPE, LOC, DATE, TIME, MONEY, PRODUCT, EVENT, "
    "WORK_OF_ART, LAW, LANGUAGE, NORP, FAC, PERCENT, QUANTITY, ORDINAL, CARDINAL. "
    "start/end = character offsets in the input. No prose, no markdown, JSON only."
)


def extract(sentence: str) -> list[dict]:
    resp = client.chat.completions.create(
        model="glm-5.2",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": sentence},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").split("\n", 1)[1].rsplit("\n", 1)[0]
    data = json.loads(raw)
    return data.get("entities", []) if isinstance(data, dict) else data


if __name__ == "__main__":
    text = "Apple opened a new store in Tel Aviv on March 3, 2025 for $2 million."
    ents = extract(text)
    print(f"input : {text}")
    print(f"output: {json.dumps(ents, indent=2, ensure_ascii=False)}")
