"""
Audio → diarized transcript pipeline.

Uses OpenAI Whisper for speech-to-text and (optionally) pyannote.audio
for speaker diarization. Output is a CSV with columns: Speaker, Text, Start, End
which feeds directly into individual_call_transform.py / the CLI summarizer.

If pyannote isn't available (or HUGGINGFACE_TOKEN is missing), we fall back to
a simple Whisper-only transcript labeled as alternating speakers — good enough
for demos when you don't want to pull the heavy diarization stack.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SUPPORTED_AUDIO = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}


def load_whisper(model_size: str = "base"):
    import whisper

    print(f"Loading Whisper model '{model_size}' ...")
    return whisper.load_model(model_size)


def load_audio_array(audio_path: str):
    """
    Load audio as float32 mono @ 16kHz for Whisper.
    Prefers ffmpeg (via whisper.load_audio). Falls back to stdlib wave for .wav
    so demos still work when ffmpeg isn't installed.
    """
    import numpy as np
    import whisper

    try:
        return whisper.load_audio(audio_path)
    except FileNotFoundError as exc:
        # Usually means ffmpeg binary is missing
        if Path(audio_path).suffix.lower() != ".wav":
            raise RuntimeError(
                "ffmpeg is required for non-WAV audio. Install it with: brew install ffmpeg"
            ) from exc

    import wave

    with wave.open(audio_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sampwidth}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    target_sr = 16000
    if sample_rate != target_sr:
        # lightweight linear resample (good enough for demo WAVs)
        duration = len(audio) / float(sample_rate)
        target_len = int(duration * target_sr)
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)

    return audio


def transcribe_with_whisper(model, audio_path: str):
    """Return whisper result dict (segments with start/end/text)."""
    audio = load_audio_array(audio_path)
    # fp16=False keeps things stable on CPU / Mac
    return model.transcribe(audio, fp16=False, verbose=False)


def diarize_with_pyannote(audio_path: str, hf_token: str):
    """
    Run pyannote speaker-diarization pipeline.
    Returns a list of (start, end, speaker_label).
    """
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    diarization = pipeline(audio_path)
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append((turn.start, turn.end, speaker))
    return turns


def assign_speakers(segments, diarization_turns=None):
    """
    Map each whisper segment to a speaker.
    If we have diarization turns, pick the speaker with max overlap.
    Otherwise alternate Customer / Representative by segment index.
    """
    rows = []

    if not diarization_turns:
        for i, seg in enumerate(segments):
            speaker = "Customer" if i % 2 == 0 else "Representative"
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "Speaker": speaker,
                    "Text": text,
                    "Start": round(float(seg.get("start", 0)), 2),
                    "End": round(float(seg.get("end", 0)), 2),
                }
            )
        return rows

    # Map SPEAKER_00 / SPEAKER_01 → Customer / Representative (first seen = Customer)
    label_map = {}

    def resolve_label(raw: str) -> str:
        if raw not in label_map:
            label_map[raw] = "Customer" if len(label_map) == 0 else (
                "Representative" if len(label_map) == 1 else raw
            )
        return label_map[raw]

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        s0, s1 = float(seg.get("start", 0)), float(seg.get("end", 0))
        best_speaker, best_overlap = "Customer", 0.0
        for d0, d1, spk in diarization_turns:
            overlap = max(0.0, min(s1, d1) - max(s0, d0))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = resolve_label(spk)
        rows.append(
            {
                "Speaker": best_speaker,
                "Text": text,
                "Start": round(s0, 2),
                "End": round(s1, 2),
            }
        )
    return rows


def process_audio(
    audio_path: str,
    output_csv: str,
    whisper_model: str = "base",
    use_diarization: bool = True,
):
    audio_path = str(audio_path)
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(audio_path)

    model = load_whisper(whisper_model)
    print(f"Transcribing {audio_path} ...")
    result = transcribe_with_whisper(model, audio_path)
    segments = result.get("segments") or []

    diarization_turns = None
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if use_diarization and hf_token:
        try:
            print("Running pyannote diarization ...")
            diarization_turns = diarize_with_pyannote(audio_path, hf_token)
        except Exception as exc:
            warnings.warn(f"Diarization failed ({exc}); falling back to alternating speakers.")
    elif use_diarization and not hf_token:
        print("No HUGGINGFACE_TOKEN set — skipping diarization, using alternating speakers.")

    rows = assign_speakers(segments, diarization_turns)
    if not rows:
        raise RuntimeError("No speech segments found in audio.")

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved transcript → {out}  ({len(rows)} turns)")
    return str(out)


def process_folder(input_dir: str, output_dir: str, whisper_model: str = "base"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in SUPPORTED_AUDIO]
    if not files:
        print(f"No audio files found in {input_dir}")
        return []

    written = []
    for audio in files:
        out_csv = output_dir / f"{audio.stem}.csv"
        written.append(
            process_audio(str(audio), str(out_csv), whisper_model=whisper_model)
        )
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe + diarize pharmacy support call audio into CSV."
    )
    parser.add_argument("--audio", help="Path to a single audio file.")
    parser.add_argument(
        "--input_dir",
        default="sample_data/audio",
        help="Folder of audio files (used when --audio is omitted).",
    )
    parser.add_argument(
        "--output",
        help="Output CSV path (single file mode).",
    )
    parser.add_argument(
        "--output_dir",
        default="sample_data/transcripts",
        help="Folder for CSV transcripts (batch mode).",
    )
    parser.add_argument(
        "--whisper_model",
        default="base",
        help="Whisper model size: tiny/base/small/medium/large",
    )
    parser.add_argument(
        "--no_diarization",
        action="store_true",
        help="Skip pyannote even if a HF token is present.",
    )
    args = parser.parse_args()

    if args.audio:
        out = args.output or str(
            Path(args.output_dir) / f"{Path(args.audio).stem}.csv"
        )
        process_audio(
            args.audio,
            out,
            whisper_model=args.whisper_model,
            use_diarization=not args.no_diarization,
        )
    else:
        process_folder(
            args.input_dir,
            args.output_dir,
            whisper_model=args.whisper_model,
        )


if __name__ == "__main__":
    main()
