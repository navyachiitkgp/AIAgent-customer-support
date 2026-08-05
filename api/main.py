"""FastAPI layer for VoiceIQ."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voiceiq.config import PRODUCT_NAME, get_settings
from voiceiq.db import get_call, init_db, list_calls
from voiceiq.pipeline import ingest_path
from voiceiq.rag import ask

settings = get_settings()
settings.ensure_dirs()
init_db()

app = FastAPI(title=PRODUCT_NAME, version="2.0.0")


class AskBody(BaseModel):
    question: str


class SummarizeBody(BaseModel):
    csv_path: str
    representative: str = "Agent_Unknown"
    call_id: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "product": PRODUCT_NAME}


@app.get("/calls")
def calls(
    intent: Optional[str] = None,
    representative_id: Optional[str] = None,
    unresolved_only: bool = False,
):
    return list_calls(
        intent=intent,
        representative_id=representative_id,
        unresolved_only=unresolved_only,
    )


@app.get("/calls/{call_id}")
def call_detail(call_id: str):
    row = get_call(call_id)
    if not row:
        raise HTTPException(404, "Call not found")
    return row


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), representative: str = "Agent_Unknown"):
    suffix = Path(file.filename or "audio.wav").suffix
    raw = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)
    try:
        result = ingest_path(tmp_path, representative=representative)
        return result
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc


@app.post("/summarize")
def summarize(body: SummarizeBody):
    path = Path(body.csv_path)
    if not path.exists():
        raise HTTPException(404, f"CSV not found: {path}")
    try:
        return ingest_path(
            path, representative=body.representative, call_id=body.call_id
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc


@app.post("/search")
def search(body: AskBody):
    try:
        return ask(body.question)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc
