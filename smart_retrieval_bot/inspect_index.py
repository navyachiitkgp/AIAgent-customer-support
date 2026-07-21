"""Quick sanity check that the FAISS index loads and returns chunks."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
load_dotenv()

FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "faiss_index")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if not OPENAI_API_KEY:
    raise SystemExit("Set OPENAI_API_KEY in .env")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db = FAISS.load_local(
    FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
)

docs = db.similarity_search("billing insurance claim", k=5)
for i, doc in enumerate(docs):
    print(f"\nChunk {i + 1}:")
    print("Text:", doc.page_content[:180], "...")
    print("Metadata:", doc.metadata)
