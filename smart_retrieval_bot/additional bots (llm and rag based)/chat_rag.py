"""Conversational RAG with optional representative filter."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

HERE = Path(__file__).resolve().parent
BOT_ROOT = HERE.parent
load_dotenv(BOT_ROOT.parent / ".env")
load_dotenv()

FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", str(BOT_ROOT / "faiss_index"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

st.set_page_config(page_title="Conversational RAG Assistant", page_icon="🤖")
st.title("Conversational Call Report Assistant")

if not OPENAI_API_KEY:
    st.error("Set OPENAI_API_KEY in .env")
    st.stop()

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.load_local(
    FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)


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

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
if "chat" not in st.session_state:
    st.session_state.chat = []

if user_input:
    context = search_documents(user_input, rep_id_filter)
    memory = st.session_state.memory
    chat_history = "\n".join(
        f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}"
        for msg in memory.chat_memory.messages
    )
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
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(response)
    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Assistant", response))

for role, msg in st.session_state.chat:
    st.markdown(f"**{role}:** {msg}")
