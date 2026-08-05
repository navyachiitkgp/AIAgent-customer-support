"""Seed SQLite DB from existing sample JSON / transcripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voiceiq.config import get_settings
from voiceiq.db import init_db, list_calls, replace_turns, upsert_call
from voiceiq.metrics import compute_coaching_metrics
from voiceiq.pii import redact_text
from voiceiq.pipeline import analyze_transcript, load_transcript_csv
from voiceiq.rag import rebuild_index_from_db


def seed_from_json() -> int:
    settings = get_settings()
    init_db()
    json_dir = ROOT / "reports" / "json"
    n = 0
    for path in sorted(json_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        call_id = data.get("call_id") or path.stem
        summary = data.get("summary") or ""
        record = {
            "call_id": call_id,
            "customer_id": data.get("customer_id"),
            "representative_id": data.get("representative_id")
            or data.get("representative")
            or "Agent_Unknown",
            "intent": data.get("intent") or "General_Inquiry",
            "summary": summary,
            "summary_redacted": redact_text(summary),
            "keywords": data.get("keywords") or [],
            "sentiment_customer": data.get("sentiment_customer") or "neutral",
            "sentiment_representative": data.get("sentiment_representative")
            or "neutral",
            "sentiment_ending": data.get("sentiment_ending") or "neutral",
            "model": data.get("model") or "seed",
            "resolved": 1,
            "escalated": 0,
            "talk_ratio_customer": 0.5,
            "talk_ratio_agent": 0.5,
            "turn_count": 0,
            "customer_turn_count": 0,
            "agent_turn_count": 0,
            "interruption_proxy": 0,
            "duration_sec": 0,
            "created_at": data.get("timestamp"),
        }
        upsert_call(record)
        n += 1
    return n


def enrich_from_transcripts(limit: int = 0) -> int:
    """Optionally re-analyze local CSVs offline (no LLM) to attach turns/metrics."""
    settings = get_settings()
    files = sorted(settings.sample_transcripts_dir.glob("*.csv"))
    if limit:
        files = files[:limit]
    n = 0
    for path in files:
        try:
            df = load_transcript_csv(path)
            metrics = compute_coaching_metrics(df)
            call_id = f"SEED-{path.stem}"
            # if a nicer call already exists from JSON, just attach turns under SEED id
            summary = " ".join(df["Text"].astype(str).tolist())[:500]
            record = {
                "call_id": call_id,
                "customer_id": f"CUST-SEED-{n+1:03d}",
                "representative_id": "Agent_A",
                "intent": "General_Inquiry",
                "summary": summary,
                "summary_redacted": redact_text(summary),
                "keywords": [],
                "sentiment_customer": "neutral",
                "sentiment_representative": "neutral",
                "sentiment_ending": "neutral",
                "model": "seed-offline",
                "transcript_path": str(path),
                **metrics,
            }
            upsert_call(record)
            turns = []
            for i, row in df.iterrows():
                turns.append(
                    {
                        "turn_index": int(i) if not isinstance(i, int) else i,
                        "speaker": row.get("Speaker"),
                        "text": row.get("Text"),
                        "text_redacted": redact_text(str(row.get("Text"))),
                        "start_sec": row.get("Start"),
                        "end_sec": row.get("End"),
                    }
                )
            replace_turns(call_id, turns)
            n += 1
        except Exception as exc:  # noqa: BLE001
            print(f"skip {path.name}: {exc}")
    return n


def main():
    init_db()
    n_json = seed_from_json()
    print(f"Seeded {n_json} calls from reports/json")
    n_csv = enrich_from_transcripts()
    print(f"Added {n_csv} transcript-backed seed calls")
    indexed = rebuild_index_from_db()
    print(f"Indexed {indexed} summaries into {get_settings().faiss_index_dir}")
    print(f"DB now has {len(list_calls())} calls → {get_settings().db_path}")


if __name__ == "__main__":
    main()
