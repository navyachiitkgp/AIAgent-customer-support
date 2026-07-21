# VoiceIQ: LLM-Driven Audio Analysis & Agentic RAG for Support Call Intelligence

Pharmacy customer-support calls → diarized transcripts → LLM summaries →
HTML/JSON reports → Streamlit dashboard + RAG chatbot.

Originally built for a Big Data Analytics coursework project (Carlson School /
University of Minnesota). This fork is set up to run end-to-end with sample data.

---

## What's in the repo

| Path | Role |
|------|------|
| `Audio_Analyzer_code/audio_to_transcript.py` | Whisper (+ optional pyannote) audio → CSV transcript |
| `Audio_Analyzer_code/batch_summarize.py` | Batch-run LLM summarization over sample CSVs |
| `individual_call_transform.py` | Summarizer class (OpenRouter): intent, sentiment, keywords, HTML/JSON |
| `class_call-individual_call_transform.py` | CLI for a single CSV |
| `streamlit_dashboard.py` | Agent / intent / sentiment dashboard |
| `smart_retrieval_bot/` | FAISS index + agentic RAG Streamlit bot |
| `sample_data/transcripts/` | Ready-to-use diarized CSV transcripts |
| `reports/json/` | Sample call summaries (dashboard works offline) |
| `reports/html/` | Sample HTML call reports |

---

## Setup

```bash
git clone https://github.com/navyachiitkgp/AIAgent-customer-support.git
cd AIAgent-customer-support

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Install ffmpeg (required for Whisper / pydub):

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install ffmpeg
```

Copy env template and add keys:

```bash
cp .env.example .env
```

| Key | Needed for |
|-----|------------|
| `OPENROUTER_API_KEY` | Call summarizer (HTML/JSON generation) |
| `OPENAI_API_KEY` | RAG embeddings + chatbot |
| `HUGGINGFACE_TOKEN` | Optional pyannote diarization |

Accept the pyannote model terms on Hugging Face if you use diarization.

---

## Run

### 1. Dashboard (works with bundled sample JSON — no API key)

```bash
streamlit run streamlit_dashboard.py
```

### 2. Summarize a transcript (needs OpenRouter)

```bash
python class_call-individual_call_transform.py \
  --csv sample_data/transcripts/Billing_Issue_1.csv \
  --call_id CALL-5159 \
  --representative Agent_D
```

Or batch:

```bash
python Audio_Analyzer_code/batch_summarize.py --limit 2
```

Outputs land in `reports/html/` and `reports/json/`.

### 3. Audio → transcript (needs Whisper; diarization optional)

```bash
# drop .wav/.mp3 into sample_data/audio/ first
python Audio_Analyzer_code/audio_to_transcript.py \
  --input_dir sample_data/audio \
  --output_dir sample_data/transcripts
```

### 4. RAG chatbot (needs OpenAI + existing FAISS index)

A prebuilt index ships under `smart_retrieval_bot/faiss_index/`. To rebuild:

```bash
cd smart_retrieval_bot
python build_vector_store.py
streamlit run query_bot.py
```

Extra RAG variants live in `smart_retrieval_bot/additional bots (llm and rag based)/`.

---

## Pipeline (short version)

```
audio (.wav/.mp3)
   → Audio_Analyzer_code (Whisper ± pyannote)
   → CSV transcript (Speaker, Text, Start, End)
   → OpenRouter summarizer
   → reports/json + reports/html
   → Streamlit dashboard
   → FAISS + agentic RAG bot
```

---

## Notes

- Sample reports and JSON are included so the dashboard runs immediately.
- Heavy models (Whisper, torch, pyannote) make the first install slow — that's expected.
- Don't commit your real `.env`. `.env.example` is the template only.
