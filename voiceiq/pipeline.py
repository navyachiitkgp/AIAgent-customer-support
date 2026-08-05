"""End-to-end call analysis pipeline: audio/CSV → DB + HTML + vector index."""

from __future__ import annotations

import csv
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voiceiq.config import get_settings
from voiceiq.db import init_db, replace_turns, upsert_call
from voiceiq.llm import OpenRouterClient
from voiceiq.metrics import compute_coaching_metrics, script_adherence_score
from voiceiq.pii import redact_text, redact_transcript_rows


def _new_call_id(prefix: str = "CALL") -> str:
    return f"{prefix}-{random.randint(1000, 9999)}-{datetime.utcnow().strftime('%H%M%S')}"


def load_transcript_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    if "speaker" in cols and cols["speaker"] != "Speaker":
        rename[cols["speaker"]] = "Speaker"
    if "text" in cols and cols["text"] != "Text":
        rename[cols["text"]] = "Text"
    if "start" in cols:
        rename[cols["start"]] = "Start"
    if "end" in cols:
        rename[cols["end"]] = "End"
    if rename:
        df = df.rename(columns=rename)
    if "Speaker" not in df.columns or "Text" not in df.columns:
        raise ValueError(f"CSV must have Speaker and Text columns. Found: {list(df.columns)}")
    return df


def audio_to_csv(audio_path: Path, output_csv: Path) -> Path:
    settings = get_settings()
    from Audio_Analyzer_code.audio_to_transcript import process_audio

    return Path(
        process_audio(
            str(audio_path),
            str(output_csv),
            whisper_model=settings.whisper_model,
            use_diarization=settings.use_diarization,
        )
    )


def write_html_report(call: Dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    intent = str(call.get("intent") or "General_Inquiry").replace("_", " ")
    keywords = call.get("keywords") or []
    if isinstance(keywords, str):
        try:
            keywords = json.loads(keywords)
        except json.JSONDecodeError:
            keywords = [keywords]
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>VoiceIQ Call Report – {intent}</title>
<style>
body {{ font-family: Georgia, serif; margin: 40px; color: #1a1a1a; background: #f7f4ef; }}
.box {{ background: #fff; border-left: 4px solid #0b6e4f; padding: 14px 18px; margin-bottom: 16px; }}
h1 {{ color: #0b6e4f; }}
</style></head><body>
<h1>VoiceIQ Call Report</h1>
<div class="box">
<b>Call ID:</b> {call.get('call_id')} |
<b>Date:</b> {(call.get('created_at') or '')[:10]} |
<b>Representative:</b> {call.get('representative_id')} |
<b>Customer:</b> {call.get('customer_id')}
</div>
<div class="box">
<b>Intent:</b> {intent}<br>
<b>Customer sentiment:</b> {call.get('sentiment_customer')}<br>
<b>Ending sentiment:</b> {call.get('sentiment_ending')}<br>
<b>Keywords:</b> {', '.join(keywords)}<br>
<b>Resolved:</b> {bool(call.get('resolved'))} |
<b>Escalated:</b> {bool(call.get('escalated'))}
</div>
<div class="box">
<h3>Summary</h3>
<p>{call.get('summary_redacted') or call.get('summary')}</p>
</div>
<div class="box">
<h3>Coaching signals</h3>
<ul>
<li>Talk ratio (customer/agent): {call.get('talk_ratio_customer')} / {call.get('talk_ratio_agent')}</li>
<li>Turns: {call.get('turn_count')} (customer {call.get('customer_turn_count')}, agent {call.get('agent_turn_count')})</li>
<li>Interruption proxy: {call.get('interruption_proxy')}</li>
<li>Duration (sec): {call.get('duration_sec')}</li>
</ul>
</div>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


def analyze_transcript(
    csv_path: Path,
    *,
    call_id: Optional[str] = None,
    representative: str = "Agent_Unknown",
    audio_path: Optional[Path] = None,
    skip_llm: bool = False,
) -> Dict[str, Any]:
    settings = get_settings()
    settings.ensure_dirs()
    init_db()

    df = load_transcript_csv(csv_path)
    rows = df.to_dict(orient="records")
    redacted_rows = redact_transcript_rows(rows)
    redacted_df = pd.DataFrame(redacted_rows)

    coaching = compute_coaching_metrics(df)
    call_id = call_id or _new_call_id()
    customer_id = f"CUST-{random.randint(10000, 99999)}"

    transcript_text = " ".join(str(r.get("Text", "")) for r in redacted_rows)

    if skip_llm:
        summary = transcript_text[:400]
        keywords = []
        intent = "General_Inquiry"
        sent_c = sent_r = ending = "neutral"
        model = "offline"
    else:
        client = OpenRouterClient()
        summary = client.summarize(transcript_text)
        keywords = client.keywords(summary)
        intent = client.intent(summary)
        sent_c = client.sentiment(summary, "customer")
        sent_r = client.sentiment(summary, "representative")
        ending = sent_c
        # prefer last customer turn sentiment if we ever add per-turn labels
        model = client.model

    script = script_adherence_score(summary, intent)
    summary_redacted = redact_text(summary)

    record = {
        "call_id": call_id,
        "customer_id": customer_id,
        "representative_id": representative,
        "intent": intent,
        "summary": summary,
        "summary_redacted": summary_redacted,
        "keywords": keywords,
        "sentiment_customer": sent_c,
        "sentiment_representative": sent_r,
        "sentiment_ending": ending,
        "model": model,
        "audio_path": str(audio_path) if audio_path else None,
        "transcript_path": str(csv_path),
        **coaching,
        **{k: v for k, v in script.items() if k != "script_checks"},
    }

    html_path = settings.reports_html_dir / f"{call_id}.html"
    write_html_report(record, html_path)
    record["html_path"] = str(html_path)

    upsert_call(record)
    turns = []
    for i, row in enumerate(redacted_rows):
        turns.append(
            {
                "turn_index": i,
                "speaker": row.get("Speaker"),
                "text": rows[i].get("Text"),
                "text_redacted": row.get("Text"),
                "start_sec": row.get("Start"),
                "end_sec": row.get("End"),
                "sentiment": None,
            }
        )
    replace_turns(call_id, turns)

    # best-effort vector index update
    try:
        from voiceiq.rag import index_call

        index_call(record)
    except Exception as exc:  # noqa: BLE001
        print(f"Vector index update skipped: {exc}")

    # also write legacy JSON for old dashboard compatibility
    legacy = {
        "call_id": call_id,
        "customer_id": customer_id,
        "representative_id": representative,
        "summary": summary_redacted,
        "keywords": keywords,
        "intent": intent,
        "sentiment_customer": sent_c,
        "sentiment_representative": sent_r,
        "sentiment_ending": ending,
        "model": model,
        "timestamp": datetime.utcnow().isoformat(),
    }
    json_dir = ROOT / "reports" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    (json_dir / f"{call_id}.json").write_text(
        json.dumps(legacy, indent=2), encoding="utf-8"
    )

    record["script_checks"] = script.get("script_checks")
    return record


def analyze_audio(
    audio_path: Path,
    *,
    call_id: Optional[str] = None,
    representative: str = "Agent_Unknown",
) -> Dict[str, Any]:
    settings = get_settings()
    settings.ensure_dirs()
    call_id = call_id or _new_call_id("AUD")
    out_csv = settings.sample_transcripts_dir / f"{call_id}.csv"
    audio_to_csv(Path(audio_path), out_csv)
    return analyze_transcript(
        out_csv,
        call_id=call_id,
        representative=representative,
        audio_path=Path(audio_path),
    )


def ingest_path(
    path: Path,
    *,
    representative: str = "Agent_Unknown",
    call_id: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return analyze_transcript(path, call_id=call_id, representative=representative)
    if suffix in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}:
        return analyze_audio(path, call_id=call_id, representative=representative)
    raise ValueError(f"Unsupported file type: {suffix}")
