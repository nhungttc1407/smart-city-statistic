# Netanya Chatbot Dataset Analysis

NLP analysis pipeline for Netanya municipality chatbot conversations — topic modeling, sector classification, and named entity recognition.

## Structure

```
├── src/
│   ├── topic/          # LDA topic modeling + classification
│   ├── ner_glm/        # GLM-5.3-Flash NER pipeline
│   ├── ner_spacy/      # spaCy NER pipeline (legacy)
│   └── utils/          # shared utilities
├── outputs/
│   ├── topic/
│   │   ├── lda/        # topic distributions, keywords, perplexity
│   │   └── labels/     # conversation + sector labels
│   ├── ner_glm/        # GLM entities, frequencies, cross-analysis
│   └── ner_spacy/      # spaCy entities
├── reports/            # markdown + PDF reports, figures
├── data/               # raw input CSV (gitignored)
└── requirements.txt
```

## Pipelines

### Topic Modeling
```bash
python src/topic/topic_model.py          # LDA (user + agent separate)
python src/topic/topic_combined.py       # LDA (Q+A combined, 16 topics)
python src/topic/classify_combined.py    # label conversations by topic
python src/topic/classify_sector.py      # label conversations by sector
python src/topic/sector_analysis.py      # sector complexity analysis
```

### NER — GLM-5.3-Flash
```bash
python src/ner_glm/ner_glm_pipeline.py --resume
python src/ner_glm/ner_glm_pipeline.py --batch-size 25 --concurrency 8
```

### NER — spaCy (legacy)
```bash
python src/ner_spacy/ner.py
```

## Requirements
```bash
pip install -r requirements.txt
```
