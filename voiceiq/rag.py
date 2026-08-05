"""Hybrid RAG: structured SQL answers + vector search over redacted summaries."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from voiceiq.config import get_settings
from voiceiq.db import count_by_intent, get_call, list_calls


def _embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings

    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _chat_llm():
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    key = settings.require_openrouter()
    return ChatOpenAI(
        model=settings.rag_chat_model,
        temperature=0,
        api_key=key,
        base_url=settings.openrouter_base_url,
        default_headers={
            "HTTP-Referer": "https://github.com/navyachiitkgp/AIAgent-customer-support",
            "X-Title": "VoiceIQ",
        },
    )


def index_call(record: Dict[str, Any]) -> None:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    settings = get_settings()
    settings.ensure_dirs()
    text = record.get("summary_redacted") or record.get("summary") or ""
    if not text.strip():
        return
    doc = Document(
        page_content=text,
        metadata={
            "call_id": record.get("call_id"),
            "intent": record.get("intent"),
            "representative_id": record.get("representative_id"),
            "sentiment_ending": record.get("sentiment_ending"),
        },
    )
    embeddings = _embeddings()
    index_dir = settings.faiss_index_dir
    if (index_dir / "index.faiss").exists():
        db = FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True
        )
        db.add_documents([doc])
    else:
        db = FAISS.from_documents([doc], embeddings)
    db.save_local(str(index_dir))


def rebuild_index_from_db() -> int:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    settings = get_settings()
    settings.ensure_dirs()
    calls = list_calls()
    docs = []
    for c in calls:
        text = c.get("summary_redacted") or c.get("summary") or ""
        if not text.strip():
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "call_id": c.get("call_id"),
                    "intent": c.get("intent"),
                    "representative_id": c.get("representative_id"),
                    "sentiment_ending": c.get("sentiment_ending"),
                },
            )
        )
    if not docs:
        return 0
    db = FAISS.from_documents(docs, _embeddings())
    db.save_local(str(settings.faiss_index_dir))
    return len(docs)


def structured_answer(question: str) -> Optional[Dict[str, Any]]:
    """Answer count / filter style questions from SQLite when possible."""
    q = question.lower()
    calls = list_calls()
    if not calls:
        return {
            "answer": "No calls in the database yet. Analyze a transcript or run the seed script.",
            "sources": [],
            "mode": "sql",
        }

    if re.search(r"how many .*billing|billing.*count|number of billing", q):
        n = sum(1 for c in calls if (c.get("intent") or "").lower().startswith("billing"))
        return {
            "answer": f"There are {n} billing-related calls in the database.",
            "sources": [
                {"call_id": c["call_id"], "snippet": (c.get("summary_redacted") or "")[:180]}
                for c in calls
                if (c.get("intent") or "").lower().startswith("billing")
            ][:5],
            "mode": "sql",
        }

    if "how many" in q and "call" in q:
        by_intent = count_by_intent()
        lines = [f"- {k}: {v}" for k, v in by_intent.items()]
        return {
            "answer": f"Total calls: {len(calls)}\n" + "\n".join(lines),
            "sources": [{"call_id": c["call_id"], "snippet": c.get("intent")} for c in calls[:5]],
            "mode": "sql",
        }

    if "unresolved" in q or "escalat" in q:
        rows = [c for c in calls if not c.get("resolved") or c.get("escalated")]
        return {
            "answer": f"{len(rows)} calls look unresolved/escalated.",
            "sources": [
                {
                    "call_id": c["call_id"],
                    "snippet": (c.get("summary_redacted") or "")[:180],
                }
                for c in rows[:5]
            ],
            "mode": "sql",
        }

    m = re.search(r"agent[_\s-]?([a-d])\b", q)
    if m or "representative" in q:
        # leave detailed agent narrative to vector+LLM, but support simple counts
        for letter in "abcd":
            if f"agent_{letter}" in q or f"agent {letter}" in q:
                rows = [
                    c
                    for c in calls
                    if (c.get("representative_id") or "").lower() == f"agent_{letter}"
                ]
                return {
                    "answer": f"Agent_{letter.upper()} handled {len(rows)} calls.",
                    "sources": [
                        {
                            "call_id": c["call_id"],
                            "snippet": (c.get("summary_redacted") or "")[:180],
                        }
                        for c in rows[:5]
                    ],
                    "mode": "sql",
                }
    return None


def vector_search(question: str, k: int = 5) -> List[Dict[str, Any]]:
    from langchain_community.vectorstores import FAISS

    settings = get_settings()
    index_path = settings.faiss_index_dir
    if not (index_path / "index.faiss").exists():
        rebuild_index_from_db()
    if not (index_path / "index.faiss").exists():
        return []
    db = FAISS.load_local(
        str(index_path), _embeddings(), allow_dangerous_deserialization=True
    )
    docs = db.similarity_search(question, k=k)
    out = []
    for d in docs:
        out.append(
            {
                "call_id": d.metadata.get("call_id"),
                "snippet": d.page_content[:240],
                "metadata": d.metadata,
            }
        )
    return out


def ask(question: str) -> Dict[str, Any]:
    structured = structured_answer(question)
    if structured and structured.get("mode") == "sql" and (
        "how many" in question.lower()
        or "unresolved" in question.lower()
        or "escalat" in question.lower()
        or re.search(r"agent[_\s-]?[a-d]\b", question.lower())
    ):
        return structured

    sources = vector_search(question)
    if not sources and structured:
        return structured
    if not sources:
        return {
            "answer": "I don't have enough indexed call data to answer that yet.",
            "sources": [],
            "mode": "none",
        }

    context = "\n\n".join(
        f"[{s.get('call_id')}] {s.get('snippet')}" for s in sources
    )
    llm = _chat_llm()
    prompt = f"""You are VoiceIQ, an assistant for pharmacy support call analytics.
Answer ONLY from the sources. Cite call IDs like [CALL-1234].
If unsure, say you don't know.

Sources:
{context}

Question: {question}
Answer:"""
    answer = llm.invoke(prompt).content
    return {"answer": answer, "sources": sources, "mode": "vector+llm"}
