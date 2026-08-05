"""Agent coaching signals derived from diarized transcripts."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

ESCALATION_CUES = (
    "supervisor",
    "manager",
    "escalate",
    "speak to someone else",
    "corporate",
    "complaint",
    "attorney",
    "lawyer",
)

RESOLUTION_CUES = (
    "resolved",
    "all set",
    "taken care of",
    "refund",
    "reprocess",
    "confirmed",
    "you're welcome",
    "have a good",
    "anything else",
)


def _norm_speaker(value: str) -> str:
    v = (value or "").strip().lower()
    if "customer" in v or v in {"caller", "patient"}:
        return "customer"
    if "rep" in v or "agent" in v or "representative" in v:
        return "agent"
    return v or "unknown"


def compute_coaching_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute lightweight coaching metrics from Speaker/Text/[Start/End] turns.
    interruption_proxy: consecutive same-speaker overlaps aren't available without
    word timings, so we approximate as very short gaps between opposite speakers.
    """
    if df is None or df.empty:
        return {
            "talk_ratio_customer": 0.0,
            "talk_ratio_agent": 0.0,
            "turn_count": 0,
            "customer_turn_count": 0,
            "agent_turn_count": 0,
            "interruption_proxy": 0,
            "duration_sec": 0.0,
            "resolved": 1,
            "escalated": 0,
        }

    work = df.copy()
    text_col = "Text" if "Text" in work.columns else "text"
    speaker_col = "Speaker" if "Speaker" in work.columns else "speaker"
    work["_speaker"] = work[speaker_col].astype(str).map(_norm_speaker)
    work["_text"] = work[text_col].fillna("").astype(str)

    if "Start" in work.columns and "End" in work.columns:
        work["_dur"] = (
            pd.to_numeric(work["End"], errors="coerce").fillna(0)
            - pd.to_numeric(work["Start"], errors="coerce").fillna(0)
        ).clip(lower=0)
    else:
        # word-count proxy when timestamps missing
        work["_dur"] = work["_text"].str.split().str.len().fillna(0) * 0.4

    total = float(work["_dur"].sum()) or 1.0
    cust = work[work["_speaker"] == "customer"]
    agent = work[work["_speaker"] == "agent"]

    interruption_proxy = 0
    if "Start" in work.columns:
        ordered = work.sort_values("Start") if "Start" in work.columns else work
        prev_end = None
        prev_speaker = None
        for _, row in ordered.iterrows():
            start = float(row.get("Start") or 0)
            if (
                prev_end is not None
                and prev_speaker
                and row["_speaker"] != prev_speaker
                and start < prev_end + 0.15
            ):
                interruption_proxy += 1
            prev_end = float(row.get("End") or start)
            prev_speaker = row["_speaker"]

    full_text = " ".join(work["_text"].tolist()).lower()
    escalated = int(any(cue in full_text for cue in ESCALATION_CUES))
    resolved = int(any(cue in full_text for cue in RESOLUTION_CUES) and not escalated)
    if not escalated and not resolved:
        # default: assume handled unless clear escalation
        resolved = 1

    return {
        "talk_ratio_customer": round(float(cust["_dur"].sum()) / total, 3),
        "talk_ratio_agent": round(float(agent["_dur"].sum()) / total, 3),
        "turn_count": int(len(work)),
        "customer_turn_count": int(len(cust)),
        "agent_turn_count": int(len(agent)),
        "interruption_proxy": int(interruption_proxy),
        "duration_sec": round(float(work["_dur"].sum()), 2),
        "resolved": resolved,
        "escalated": escalated,
    }


def script_adherence_score(summary: str, intent: str) -> Dict[str, Any]:
    """Very rough checklist: verify / next step language present for common intents."""
    text = f"{summary or ''} {intent or ''}".lower()
    checks = {
        "verified_account": any(w in text for w in ("account", "verify", "confirmed", "pulled up")),
        "offered_next_step": any(
            w in text for w in ("upload", "fax", "pickup", "refund", "reprocess", "ship", "refill")
        ),
        "closed_politely": any(w in text for w in ("thank", "welcome", "help")),
    }
    score = round(sum(1 for v in checks.values() if v) / max(len(checks), 1), 2)
    return {"script_checks": checks, "script_adherence": score}
