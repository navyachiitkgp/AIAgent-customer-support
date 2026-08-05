"""
VoiceIQ — pharmacy support call intelligence.

Central settings loaded from environment / .env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv()

PRODUCT_NAME = "VoiceIQ"
PRODUCT_TAGLINE = "Support call intelligence for pharmacy care teams"


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _bool(name: str, default: bool = False) -> bool:
    raw = _env(name, str(default)).lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    product_name: str = PRODUCT_NAME
    openrouter_api_key: str = ""
    huggingface_token: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    summary_model: str = "openai/gpt-4o-mini"
    rag_chat_model: str = "openai/gpt-4o-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    whisper_model: str = "base"
    use_diarization: bool = False
    db_path: Path = ROOT / "data" / "voiceiq.db"
    faiss_index_dir: Path = ROOT / "data" / "faiss_index"
    inbox_dir: Path = ROOT / "data" / "inbox"
    processed_dir: Path = ROOT / "data" / "processed"
    reports_html_dir: Path = ROOT / "reports" / "html"
    sample_transcripts_dir: Path = ROOT / "sample_data" / "transcripts"
    sample_audio_dir: Path = ROOT / "sample_data" / "audio"
    app_password: str = ""  # optional simple gate for demo

    def require_openrouter(self) -> str:
        key = self.openrouter_api_key
        if not key or "your-key-here" in key:
            raise ValueError(
                "OPENROUTER_API_KEY is missing. Copy .env.example to .env and add a real key."
            )
        return key

    def ensure_dirs(self) -> None:
        for path in (
            self.db_path.parent,
            self.faiss_index_dir,
            self.inbox_dir,
            self.processed_dir,
            self.reports_html_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        openrouter_api_key=_env("OPENROUTER_API_KEY"),
        huggingface_token=_env("HUGGINGFACE_TOKEN") or _env("HF_TOKEN"),
        summary_model=_env("SUMMARY_MODEL", "openai/gpt-4o-mini"),
        rag_chat_model=_env("RAG_CHAT_MODEL", "openai/gpt-4o-mini"),
        embedding_model=_env(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        whisper_model=_env("WHISPER_MODEL", "base"),
        use_diarization=_bool("USE_DIARIZATION", False),
        db_path=Path(_env("VOICEIQ_DB", str(ROOT / "data" / "voiceiq.db"))),
        faiss_index_dir=Path(
            _env("FAISS_INDEX_DIR", str(ROOT / "data" / "faiss_index"))
        ),
        app_password=_env("VOICEIQ_APP_PASSWORD"),
    )
