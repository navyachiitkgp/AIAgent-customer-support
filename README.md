# VoiceIQ

Pharmacy support-call intelligence: audio/transcripts → redacted summaries →
SQLite analytics → coaching metrics → hybrid RAG (SQL + local embeddings).

### 60-second demo

![VoiceIQ demo](docs/voiceiq_demo.gif)

[Download MP4 (~60s)](docs/voiceiq_demo.mp4) · Analyze → Dashboard → Ask VoiceIQ → Batch

### Labeled eval (n=10, OpenRouter `gpt-4o-mini`)

| Metric | Score |
|--------|------:|
| Intent accuracy | **100%** |
| Sentiment accuracy | **80%** |
| Summary keyword coverage | **90%** |

Reproduce: `python scripts/run_eval.py` · snapshot in [`eval_data/published_eval_report.json`](eval_data/published_eval_report.json).

---

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-core.txt
cp .env.example .env   # add OPENROUTER_API_KEY for live LLM/RAG
bash scripts/demo.sh
./venv/bin/streamlit run app.py
```

Open http://localhost:8501 — **Analyze / Dashboard / Call detail / Ask / Batch**.

API (optional):

```bash
uvicorn api.main:app --reload
# GET  /health /calls /calls/{id}
# POST /transcribe /summarize /search
```

---

## What you get

| Area | Now |
|------|-----|
| Product entry | Unified `app.py` (not 4 scripts) |
| Config | `voiceiq/config.py` + clear missing-key errors |
| Storage | SQLite `data/voiceiq.db` |
| Privacy | PII redaction before LLM/RAG |
| Coaching | Talk ratio, interruptions proxy, resolved/escalated |
| RAG | SQL for counts/filters + vectors for free-text; citations |
| Batch | Drop files in `data/inbox/` → `python -m voiceiq.batch` |
| Eval | `eval_data/` + `scripts/run_eval.py` |
| Deps | `requirements-core.txt` vs `requirements-audio.txt` |
| Ops | Docker Compose + GitHub Actions CI |

---

## Full audio pipeline

```bash
pip install -r requirements-audio.txt
# brew install ffmpeg   # needed for mp3; wav works without it

# single file via UI, or:
python -c "from voiceiq.pipeline import ingest_path; print(ingest_path('sample_data/audio/Billing_Issue_sample.wav'))"
```

Diarization: set `USE_DIARIZATION=true` and `HUGGINGFACE_TOKEN` (pyannote). Without it, Whisper uses alternating speakers — documented limitation.

---

## Batch jobs

```bash
cp sample_data/transcripts/Billing_Issue_1.csv data/inbox/
python -m voiceiq.batch          # once
python -m voiceiq.batch --watch  # folder watcher
```

---

## Evaluation

Labeled set: 10 pharmacy-support calls across billing, delivery, refill, inquiry, and medication change.

```bash
python scripts/run_eval.py --offline   # heuristic smoke
python scripts/run_eval.py             # LLM vs labeled set (needs OPENROUTER_API_KEY)
python scripts/make_demo_video.py      # regenerate docs/voiceiq_demo.{gif,mp4}
```

---

## Architecture

```
Upload audio/CSV
   → Whisper (± pyannote)
   → PII redaction
   → OpenRouter summary / intent / sentiment
   → Coaching metrics
   → SQLite + HTML report + FAISS
   → Streamlit app / FastAPI / hybrid RAG
```

---

## Docker

```bash
docker compose up --build
```

Streamlit `:8501`, API `:8000`.
