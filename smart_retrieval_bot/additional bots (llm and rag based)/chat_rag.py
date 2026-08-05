"""Conversational RAG with optional representative filter."""

import sys
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_config import FAISS_INDEX_DIR, OPENROUTER_API_KEY, get_chat_llm, get_embeddings

st.set_page_config(page_title="Conversational RAG Assistant", page_icon="🤖")
st.title("Conversational Call Report Assistant")

if not OPENROUTER_API_KEY or "your-key-here" in OPENROUTER_API_KEY:
    st.error("Set OPENROUTER_API_KEY in .env")
    st.stop()

embeddings = get_embeddings()
vectorstore = FAISS.load_local(
    FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()
llm = get_chat_llm(temperature=0)


def collect_rep_ids():
    docs = retriever.invoke("call")
    return sorted({d.metadata.get("rep_id") for d in docs if d.metadata.get("rep_id")})


def search_documents(query, rep_id=None):
    docs = retriever.invoke(query)
    if rep_id:
        docs = [doc for doc in docs if doc.metadata.get("rep_id") == rep_id]
    return (
        "\n\n".join(doc.page_content for doc in docs[:3])
        if docs
        else "No relevant context found."
    )


available_rep_ids = [r for r in collect_rep_ids() if r]
rep_id = st.selectbox("Filter by Representative", ["All"] + available_rep_ids)
rep_id_filter = None if rep_id == "All" else rep_id
user_input = st.text_input("Ask a question (supports follow-ups):")

if "chat" not in st.session_state:
    st.session_state.chat = []

if user_input:
    context = search_documents(user_input, rep_id_filter)
    chat_history = "\n".join(f"{role}: {msg}" for role, msg in st.session_state.chat[-6:])
    prompt = f"""
You are a helpful assistant analyzing customer care call reports.

Chat so far:
{chat_history}

Relevant reports:
\"\"\"
{context}
\"\"\"

Now answer this:
{user_input}
"""
    response = llm.invoke(prompt).content
    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Assistant", response))

for role, msg in st.session_state.chat:
    st.markdown(f"**{role}:** {msg}")
