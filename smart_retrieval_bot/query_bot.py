"""
VoiceIQ Agentic RAG bot — ask natural-language questions over call reports.

Run from smart_retrieval_bot/:
  streamlit run query_bot.py
"""

import os
import time
from pathlib import Path

import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, AgentType, Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

try:
    import pdfplumber
except ImportError:  # optional upload support
    pdfplumber = None

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
load_dotenv()

FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "faiss_index")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

st.set_page_config(page_title="VoiceIQ Agentic RAG Bot", page_icon="🤖")
st.markdown(
    "<h1 style='text-align: center;'>VoiceIQ Agentic RAG Bot</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Ask questions about customer care call reports"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

if not OPENAI_API_KEY:
    st.error("Set `OPENAI_API_KEY` in your `.env` file to use the RAG bot.")
    st.stop()

if not os.path.isdir(FAISS_INDEX_DIR):
    st.error(
        f"No FAISS index at `{FAISS_INDEX_DIR}`. "
        "Run `python build_vector_store.py` first."
    )
    st.stop()

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.load_local(
    FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
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


def search_documents(query: str) -> str:
    docs = retriever.invoke(query)
    return (
        "\n\n".join(doc.page_content for doc in docs)
        if docs
        else "No relevant context found."
    )


tool = Tool(
    name="SearchReports",
    func=search_documents,
    description="Search and analyze customer care call reports.",
)

base_agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=st.session_state.memory,
    verbose=True,
    handle_parsing_errors=True,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=base_agent.agent,
    tools=[tool],
    memory=st.session_state.memory,
    verbose=True,
    handle_parsing_errors=True,
)

prompt = st.chat_input("Ask VoiceIQ something about your reports...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_messages.append({"role": "user", "content": prompt})

    result = agent_executor.invoke({"input": prompt})
    final_output = result["output"]
    steps = result.get("intermediate_steps", [])

    with st.chat_message("assistant"):
        animated = st.empty()
        shown = ""
        for char in final_output:
            shown += char
            animated.markdown(shown + "▌")
            time.sleep(0.012)
        animated.markdown(shown)

    docs = retriever.invoke(prompt)
    if docs:
        with st.expander("Retrieved document metadata", expanded=False):
            for i, doc in enumerate(docs[:3]):
                st.markdown(f"**Chunk {i + 1}**")
                for key, value in doc.metadata.items():
                    st.markdown(f"- **{key}**: `{value}`")
                st.markdown("---")

    if steps:
        with st.expander("Agent reasoning", expanded=False):
            for i, step in enumerate(steps):
                st.markdown(f"**Step {i + 1}**")
                st.markdown(f"Thought: {step[0].log.strip()}")
                st.markdown(f"Observation: {step[1]}")
                st.markdown("---")

    st.session_state.memory.chat_memory.add_user_message(prompt)
    st.session_state.memory.chat_memory.add_ai_message(final_output)
    st.session_state.chat_messages.append({"role": "assistant", "content": final_output})

with st.expander("Upload report (HTML or PDF)", expanded=False):
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
                return "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        return ""

    if uploaded:
        raw = extract_text(uploaded)
        if not raw.strip():
            st.error("No readable text found in the uploaded file.")
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = [
                Document(page_content=chunk, metadata={"source": uploaded.name})
                for chunk in splitter.split_text(raw)
            ]
            vectorstore.add_documents(docs)
            st.success(f"Indexed {len(docs)} chunks from **{uploaded.name}**")
