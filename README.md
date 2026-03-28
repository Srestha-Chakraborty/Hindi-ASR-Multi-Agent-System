# Hindi ASR Research Suite — Josh Talks

## 1. Overview
- Q1: Whisper-small Hindi preprocessing, baseline eval, finetune flow, error taxonomy, fix loop.
- Q2: Raw ASR cleanup with Hindi number normalization and English-word tagging.
- Q3: Large-scale Hindi spelling classification with rule + LLM fallback and confidence analysis.
- Q4: Lattice-based WER to compare strict reference WER vs alternative-aware scoring.

## 2. Prerequisites
- Python 3.10+
- Docker + Docker Compose
- ffmpeg
- Groq API key (free tier)

## 3. Setup
```bash
cp .env.example .env
# Add your GROQ_API_KEY in .env
pip install -r requirements.txt
```

## 4. Run Locally
```bash
streamlit run frontend/app.py
```

## 5. Run with Docker
```bash
docker-compose up --build
```

## 6. Free Groq API Key
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (no credit card required)
3. Create API key under API Keys
4. Put it in `.env` as `GROQ_API_KEY=...`

## 7. Architecture Diagram (ASCII)
```text
                 +--------------------+
                 |   Streamlit App    |
                 +---------+----------+
                           |
                           v
                +-----------------------+
                | LangGraph Supervisor  |
                | (selected q1..q4)     |
                +----+----+----+----+---+
                     |    |    |    |
                     v    v    v    v
                    Q1   Q2   Q3   Q4
                     |    |    |    |
                     +----+----+----+
                           |
                           v
                  outputs/final_report.json
```

## 8. Expected Outputs Per Question
- Q1: baseline vs finetuned WER, 25+ error samples, taxonomy JSON, fix before/after subset WER.
- Q2: raw/normalized/tagged transcripts, conversion examples, edge-case explanations.
- Q3: classified words table, confidence tiers, low-confidence review, unreliable categories.
- Q4: standard vs lattice WER comparison, lattice bins, model-wise improvement deltas.

## Notes
- All LLM calls use Groq via `ChatGroq`.
- CPU-only execution is supported.
- DEMO MODE is available in UI to skip heavy Q1 fine-tuning and still render complete outputs.
