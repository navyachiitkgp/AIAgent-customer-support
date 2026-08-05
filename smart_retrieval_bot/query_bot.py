"""
VoiceIQ RAG bot — ask natural-language questions over call reports.

Local embeddings + OpenRouter chat (no OpenAI billing required).

Run from smart_retrieval_bot/:
  ../venv/bin/streamlit run query_bot.py
"""

import os
import time

import streamlit as st
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS

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

st.set_page_config(page_title="VoiceIQ RAG Bot", page_icon="🤖", layout="centered")
st.markdown(
    "<h1 style='text-align: center;'>VoiceIQ RAG Bot</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Ask questions about the indexed call reports — no upload needed."
    "</p>",
    unsafe_allow_html=True,
)

if not OPENROUTER_API_KEY or "your-key-here" in OPENROUTER_API_KEY:
    st.error("Set `OPENROUTER_API_KEY` in your project-root `.env` file.")
    st.stop()

if not os.path.isdir(FAISS_INDEX_DIR):
    st.error(
        f"No FAISS index at `{FAISS_INDEX_DIR}`. "
        "Run `python build_vector_store.py` first."
    )
    st.stop()


@st.cache_resource
def load_stack():
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    llm = get_chat_llm(temperature=0)
    return vectorstore, llm


with st.spinner("Loading search index..."):
    vectorstore, llm = load_stack()

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

st.success("Index ready — try a sample question below, or type your own.")

sample_questions = [
    "What billing issues came up in the calls?",
    "Summarize delivery-related calls",
    "Which agents handled insurance or claim problems?",
]

cols = st.columns(len(sample_questions))
for col, q in zip(cols, sample_questions):
    if col.button(q, use_container_width=True):
        st.session_state["pending_prompt"] = q

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.session_state.pop("pending_prompt", None) or st.chat_input(
    "Ask about billing, delivery, agents, sentiment..."
)

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_messages.append({"role": "user", "content": prompt})

    docs = retriever.invoke(prompt)
    context = (
        "\n\n".join(doc.page_content for doc in docs)
        if docs
        else "No relevant context found."
    )
    history = "\n".join(
        f"{m['role']}: {m['content']}" for m in st.session_state.chat_messages[-6:]
    )
    full_prompt = f"""You are VoiceIQ, a helpful assistant for pharmacy customer-care call reports.
Answer using ONLY the relevant reports below. If unsure, say you don't know.
Cite call IDs when possible.

Chat history:
{history}

Relevant reports:
\"\"\"
{context}
\"\"\"

User: {prompt}
Assistant:"""

    final_output = llm.invoke(full_prompt).content

    with st.chat_message("assistant"):
        animated = st.empty()
        shown = ""
        for char in final_output:
            shown += char
            animated.markdown(shown + "▌")
            time.sleep(0.008)
        animated.markdown(shown)

    if docs:
        with st.expander("Sources used", expanded=False):
            for i, doc in enumerate(docs[:3]):
                meta = doc.metadata or {}
                st.markdown(
                    f"**{i + 1}. {meta.get('call_id', 'Unknown')}** "
                    f"({meta.get('source', 'n/a')})"
                )
                st.caption(doc.page_content[:220] + "...")

    st.session_state.chat_messages.append({"role": "assistant", "content": final_output})

with st.expander("Optional: add another report to the index", expanded=False):
    st.caption(
        "Reports are already loaded from `smart_retrieval_bot/reports/`. "
        "Use this only if you want to add a new HTML/PDF."
    )
    uploaded = st.file_uploader(
        "Extra report (optional)",
        type=["html", "pdf"],
        label_visibility="visible",
    )

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
            docs = make_documents(raw, {"source": uploaded.name})
            vectorstore.add_documents(docs)
            st.success(f"Indexed {len(docs)} chunks from **{uploaded.name}**")
