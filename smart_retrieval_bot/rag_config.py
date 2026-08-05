"""
Shared RAG config: local embeddings + OpenRouter chat.

- Embeddings: sentence-transformers (local, free)
- Chat: OpenRouter (same OPENROUTER_API_KEY as the summarizer)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
load_dotenv()

OPENROUTER_API_KEY = (os.getenv("OPENROUTER_API_KEY") or "").strip()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RAG_CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "openai/gpt-4o-mini").strip()
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
).strip()
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "faiss_index")
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")


def require_openrouter_key() -> str:
    if not OPENROUTER_API_KEY or "your-key-here" in OPENROUTER_API_KEY:
        raise SystemExit(
            "Set OPENROUTER_API_KEY in the project-root .env file "
            "(same key used by the call summarizer)."
        )
    return OPENROUTER_API_KEY


def get_embeddings():
    """Local HuggingFace embeddings — no OpenAI credits needed."""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_chat_llm(temperature: float = 0.0) -> ChatOpenAI:
    key = require_openrouter_key()
    return ChatOpenAI(
        model=RAG_CHAT_MODEL,
        temperature=temperature,
        api_key=key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/navyachiitkgp/AIAgent-customer-support",
            "X-Title": "VoiceIQ RAG",
        },
    )


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """Simple character splitter (avoids heavy langchain_text_splitters imports)."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def make_documents(text: str, metadata: dict | None = None) -> List[Document]:
    metadata = metadata or {}
    return [
        Document(page_content=chunk, metadata={**metadata, "chunk": i})
        for i, chunk in enumerate(split_text(text))
    ]
