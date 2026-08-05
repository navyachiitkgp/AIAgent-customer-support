"""RAG chat with optional HTML/PDF upload into the FAISS index."""

import sys
import time
from pathlib import Path

import streamlit as st
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_config import (
    FAISS_INDEX_DIR,
    OPENROUTER_API_KEY,
    get_chat_llm,
    get_embeddings,
    make_documents,
)

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

st.set_page_config(page_title="Customer Care Analytics Bot", page_icon="📞")
st.title("Customer Care Analytics Bot")

if not OPENROUTER_API_KEY or "your-key-here" in OPENROUTER_API_KEY:
    st.error("Set OPENROUTER_API_KEY in .env")
    st.stop()

embeddings = get_embeddings()
vectorstore = FAISS.load_local(
    FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()
llm = get_chat_llm(temperature=0)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about call reports...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_messages.append({"role": "user", "content": prompt})

    docs = retriever.invoke(prompt)
    context = (
        "\n\n".join(doc.page_content for doc in docs[:3])
        if docs
        else "No relevant context found."
    )
    history = "\n".join(
        f"{m['role']}: {m['content']}" for m in st.session_state.chat_messages[-6:]
    )
    full_prompt = f"""
You are a helpful assistant answering questions based on customer care reports.

Chat history:
{history}

Relevant reports:
\"\"\"
{context}
\"\"\"

User: {prompt}
Assistant:"""

    raw_response = llm.invoke(full_prompt).content
    with st.chat_message("assistant"):
        box = st.empty()
        shown = ""
        for ch in raw_response:
            shown += ch
            box.markdown(shown + "▌")
            time.sleep(0.012)
        box.markdown(shown)

    st.session_state.chat_messages.append({"role": "assistant", "content": raw_response})

with st.expander("Upload report (HTML or PDF)"):
    uploaded = st.file_uploader(" ", label_visibility="collapsed", type=["html", "pdf"])

    def extract_text(file) -> str:
        if file.name.endswith(".html"):
            return BeautifulSoup(file.read(), "html.parser").get_text(
                separator=" ", strip=True
            )
        if file.name.endswith(".pdf"):
            if pdfplumber is None:
                st.error("Install pdfplumber to upload PDFs.")
                return ""
            with pdfplumber.open(file) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        return ""

    if uploaded:
        raw = extract_text(uploaded)
        if not raw.strip():
            st.error("No readable text found in the uploaded file.")
        else:
            documents = make_documents(raw, {"source": uploaded.name})
            vectorstore.add_documents(documents)
            st.success(f"Indexed {len(documents)} chunks from {uploaded.name}")
