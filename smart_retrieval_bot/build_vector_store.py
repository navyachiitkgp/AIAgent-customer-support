"""
Build a FAISS vector index from HTML call reports under ./reports/.

Requires OPENAI_API_KEY in the environment (or .env).
Run from the smart_retrieval_bot/ directory:

  cd smart_retrieval_bot
  python build_vector_store.py
"""

import os
import re
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "faiss_index")

if not OPENAI_API_KEY:
    raise SystemExit("Set OPENAI_API_KEY in .env before building the vector store.")


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
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        for i, chunk in enumerate(splitter.split_text(full_text)):
            all_documents.append(
                Document(
                    page_content=chunk,
                    metadata={**metadata, "source": fname, "chunk": i},
                )
            )

    if not all_documents:
        raise SystemExit(f"No HTML reports found in {REPORTS_DIR}")

    print(f"Prepared {len(all_documents)} chunks. Embedding with OpenAI...")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.from_documents(all_documents, embeddings)
    db.save_local(FAISS_INDEX_DIR)
    print(f"FAISS index saved to {FAISS_INDEX_DIR}")


if __name__ == "__main__":
    main()
