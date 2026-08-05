"""Lightweight PII redaction for pharmacy support transcripts."""

from __future__ import annotations

import re
from typing import Dict, List, Pattern, Tuple

Rule = Tuple[Pattern[str], str]

RULES: List[Rule] = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_REDACTED]"),
    (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), "[PHONE_REDACTED]"),
    (
        re.compile(
            r"\b(?:member|policy|insurance)\s*(?:id|#|number)?\s*[:#]?\s*[A-Z0-9-]{6,}\b",
            re.IGNORECASE,
        ),
        "[INSURANCE_ID_REDACTED]",
    ),
    (
        re.compile(
            r"\b(?:dob|date of birth)\s*[:#]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            re.IGNORECASE,
        ),
        "[DOB_REDACTED]",
    ),
    (
        re.compile(
            r"\b(?:january|february|march|april|may|june|july|august|september|"
            r"october|november|december)\s+\d{1,2},?\s+\d{4}\b",
            re.IGNORECASE,
        ),
        "[DATE_REDACTED]",
    ),
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE), "[EMAIL_REDACTED]"),
    (re.compile(r"\b(?:rx|prescription)\s*#?\s*\d{5,}\b", re.IGNORECASE), "[RX_REDACTED]"),
    (re.compile(r"\bCUST-\d+\b"), "[CUSTOMER_ID_REDACTED]"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "[DATE_REDACTED]"),
    (
        re.compile(
            r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)?\s*"
            r"(?:James|Mary|Sam|Norton|Harris|Alex|Jordan|Taylor|Morgan|Casey)\b",
            re.IGNORECASE,
        ),
        "[NAME_REDACTED]",
    ),
]


def redact_text(text: str) -> str:
    if not text:
        return text
    out = text
    for pattern, repl in RULES:
        out = pattern.sub(repl, out)
    return out


def redact_transcript_rows(rows: list) -> list:
    cleaned = []
    for row in rows:
        item = dict(row)
        raw = str(item.get("Text") or item.get("text") or "")
        redacted = redact_text(raw)
        item["text_redacted"] = redacted
        if "Text" in item:
            item["Text"] = redacted
        if "text" in item:
            item["text"] = redacted
        cleaned.append(item)
    return cleaned


def redact_payload(payload: Dict) -> Dict:
    out = dict(payload)
    if out.get("summary"):
        out["summary_redacted"] = redact_text(str(out["summary"]))
    return out
