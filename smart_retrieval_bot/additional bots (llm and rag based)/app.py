"""
Lightweight RAG chat over the FAISS call-report index.

Alternative to query_bot.py — simpler prompt-stuffing, no agent loop.
Run from the parent of faiss_index (usually smart_retrieval_bot/):

  streamlit run "additional bots (llm and rag based)/app.py"
"""

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# allow running from either this folder or smart_retrieval_bot/
HERE = Path(__file__).resolve().parent
BOT_ROOT = HERE.parent
load_dotenv(BOT_ROOT.parent / ".env")
load_dotenv()

FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", str(BOT_ROOT / "faiss_index"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

st.set_page_config(page_title="Customer Care Analytics Bot", page_icon="📞")
st.title("Customer Care Analytics Bot")

if not OPENAI_API_KEY:
    st.error("Set OPENAI_API_KEY in .env")
    st.stop()

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.load_local(
    FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question about call reports...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_messages.append({"role": "user", "content": user_input})

    docs = retriever.invoke(user_input)
    context = (
        "\n\n".join(doc.page_content for doc in docs[:3])
        if docs
        else "No relevant reports found."
    )
    history = "\n".join(
        f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
        for m in st.session_state.memory.chat_memory.messages
    )
    full_prompt = f"""
You are a helpful assistant answering questions based on customer care reports.

Chat history:
{history}

Relevant reports:
\"\"\"
{context}
\"\"\"

User: {user_input}
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

    st.session_state.memory.chat_memory.add_user_message(user_input)
    st.session_state.memory.chat_memory.add_ai_message(raw_response)
    st.session_state.chat_messages.append({"role": "assistant", "content": raw_response})
