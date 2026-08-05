"""
Build a FAISS vector index from HTML call reports under ./reports/.

Uses local sentence-transformers embeddings (no OpenAI credits).
Run from the smart_retrieval_bot/ directory:

  cd smart_retrieval_bot
  python build_vector_store.py
"""

import os
import re

from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS

from rag_config import FAISS_INDEX_DIR, REPORTS_DIR, get_embeddings, make_documents


def extract_metadata(text: str):
    def match(pattern):
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else None

    return {
        "call_id": match(r"Call ID[:\-]?\s*(CALL-[\d\-]+)"),
        "date": match(r"Date[:\-]?\s*([0-9]{4}-[0-9]{2}-[0-9]{2})"),
        "rep_id": match(r"Representative(?: ID)?[:\-]?\s*(Agent_[A-Z]|REP-\d+)"),
        "customer_id": match(r"Customer(?: ID)?[:\-]?\s*(CUST-\d+)"),
        "intent": match(r"Intent[:\-]?\s*(.+?)(?:\n|$)"),
        "customer_sentiment": match(r"Customer Sentiment[:\-]?\s*(\w+)"),
        "rep_sentiment": match(r"Representative Sentiment[:\-]?\s*(\w+)"),
        "keywords": match(r"Keywords[:\-]?\s*(.+?)(?:\n|$)"),
    }


def main():
    all_documents = []
    for fname in sorted(os.listdir(REPORTS_DIR)):
        if not fname.endswith(".html"):
            continue
        path = os.path.join(REPORTS_DIR, fname)
        print(f"Loading {fname}")
        with open(path, encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            full_text = soup.get_text(separator=" ", strip=True)

        metadata = extract_metadata(full_text)
        docs = make_documents(full_text, {**metadata, "source": fname})
        all_documents.extend(docs)

    if not all_documents:
        raise SystemExit(f"No HTML reports found in {REPORTS_DIR}")

    print(
        f"Prepared {len(all_documents)} chunks. "
        "Embedding locally (sentence-transformers)..."
    )
    embeddings = get_embeddings()
    db = FAISS.from_documents(all_documents, embeddings)
    db.save_local(FAISS_INDEX_DIR)
    print(f"FAISS index saved to {FAISS_INDEX_DIR}")


if __name__ == "__main__":
    main()
