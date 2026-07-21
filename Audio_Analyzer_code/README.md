# Audio analyzer

Turns support-call audio into a diarized CSV transcript.

```bash
# from repo root
python Audio_Analyzer_code/audio_to_transcript.py --audio path/to/call.wav
python Audio_Analyzer_code/audio_to_transcript.py --input_dir sample_data/audio
```

Then summarize:

```bash
python class_call-individual_call_transform.py \
  --csv sample_data/transcripts/<name>.csv \
  --call_id CALL-1001 \
  --representative Agent_A
```
