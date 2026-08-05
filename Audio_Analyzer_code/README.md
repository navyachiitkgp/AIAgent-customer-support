# Audio analyzer

Turns support-call audio into a diarized CSV transcript (Whisper ± optional pyannote).

```bash
# from repo root
python Audio_Analyzer_code/audio_to_transcript.py --audio path/to/call.wav
python Audio_Analyzer_code/audio_to_transcript.py --input_dir sample_data/audio
```

Or use the product pipeline / UI:

```bash
python -c "from voiceiq.pipeline import ingest_path; print(ingest_path('sample_data/audio/Billing_Issue_sample.wav'))"
./venv/bin/streamlit run app.py
```
