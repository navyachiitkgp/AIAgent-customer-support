"""
VoiceIQ unified app — upload → analyze → dashboard → call detail → chat.

Run:
  ./venv/bin/streamlit run app.py
"""

from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from voiceiq.config import PRODUCT_NAME, PRODUCT_TAGLINE, get_settings
from voiceiq.db import get_call, init_db, list_calls
from voiceiq.pipeline import ingest_path
from voiceiq.rag import ask, rebuild_index_from_db

st.set_page_config(page_title=PRODUCT_NAME, layout="wide", page_icon="📞")
settings = get_settings()
settings.ensure_dirs()
init_db()

# Optional simple password gate for demos
if settings.app_password:
    if "authed" not in st.session_state:
        st.session_state.authed = False
    if not st.session_state.authed:
        st.title(PRODUCT_NAME)
        pwd = st.text_input("App password", type="password")
        if st.button("Enter") and pwd == settings.app_password:
            st.session_state.authed = True
            st.rerun()
        st.stop()

st.markdown(
    f"<h1 style='margin-bottom:0'>{PRODUCT_NAME}</h1>"
    f"<p style='color:#5b5b5b;margin-top:4px'>{PRODUCT_TAGLINE}</p>",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate",
    ["Analyze", "Dashboard", "Call detail", "Ask VoiceIQ", "Batch inbox"],
)

# ---------- Analyze ----------
if page == "Analyze":
    st.subheader("Analyze a call")
    st.caption("Upload audio (.wav/.mp3) or a diarized transcript CSV. PII is redacted before LLM/RAG.")
    up = st.file_uploader("Call file", type=["wav", "mp3", "m4a", "csv"])
    rep = st.text_input("Representative ID", value="Agent_A")
    call_id = st.text_input("Call ID (optional)")
    if st.button("Run analysis", type="primary") and up:
        suffix = Path(up.name).suffix.lower()
        tmp = settings.inbox_dir / f"upload_{datetime.utcnow().strftime('%H%M%S')}{suffix}"
        tmp.write_bytes(up.getvalue())
        with st.spinner("Transcribing / summarizing..."):
            try:
                result = ingest_path(
                    tmp,
                    representative=rep or "Agent_Unknown",
                    call_id=call_id or None,
                )
                st.success(f"Saved {result['call_id']}")
                st.json(
                    {
                        "call_id": result.get("call_id"),
                        "intent": result.get("intent"),
                        "sentiment_ending": result.get("sentiment_ending"),
                        "talk_ratio_customer": result.get("talk_ratio_customer"),
                        "talk_ratio_agent": result.get("talk_ratio_agent"),
                        "resolved": result.get("resolved"),
                        "escalated": result.get("escalated"),
                        "summary": result.get("summary_redacted"),
                    }
                )
                st.session_state["selected_call_id"] = result["call_id"]
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

# ---------- Dashboard ----------
elif page == "Dashboard":
    st.subheader("Operations dashboard")
    calls = list_calls()
    if not calls:
        st.warning("No calls in DB yet. Run `python scripts/seed_db.py` or analyze a file.")
        st.stop()

    df = pd.DataFrame(calls)
    agents = sorted(df["representative_id"].dropna().unique().tolist())
    intents = sorted(df["intent"].dropna().unique().tolist())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        selected_agents = st.multiselect("Agents", agents, default=agents)
    with c2:
        selected_intents = st.multiselect("Intents", intents, default=intents)
    with c3:
        unresolved_only = st.checkbox("Unresolved / escalated only")
    with c4:
        date_from = st.date_input("From", value=None)
        date_to = st.date_input("To", value=None)

    filtered = df[
        df["representative_id"].isin(selected_agents) & df["intent"].isin(selected_intents)
    ].copy()
    if unresolved_only:
        filtered = filtered[(filtered["resolved"] == 0) | (filtered["escalated"] == 1)]
    if date_from:
        filtered = filtered[pd.to_datetime(filtered["created_at"], errors="coerce").dt.date >= date_from]
    if date_to:
        filtered = filtered[pd.to_datetime(filtered["created_at"], errors="coerce").dt.date <= date_to]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Calls", len(filtered))
    m2.metric("Escalated", int(filtered["escalated"].fillna(0).sum()))
    m3.metric(
        "Avg customer talk ratio",
        round(float(filtered["talk_ratio_customer"].fillna(0).mean()), 2),
    )
    m4.metric(
        "Positive ending %",
        round(
            100
            * (filtered["sentiment_ending"].astype(str).str.lower() == "positive").mean(),
            1,
        ),
    )

    left, right = st.columns(2)
    with left:
        st.markdown("##### Intent mix")
        st.bar_chart(filtered["intent"].value_counts())
    with right:
        st.markdown("##### Ending sentiment")
        st.bar_chart(filtered["sentiment_ending"].astype(str).str.lower().value_counts())

    st.markdown("##### Agent coaching snapshot")
    coach = (
        filtered.groupby("representative_id")
        .agg(
            calls=("call_id", "count"),
            avg_customer_talk=("talk_ratio_customer", "mean"),
            escalations=("escalated", "sum"),
            interruptions=("interruption_proxy", "mean"),
        )
        .reset_index()
    )
    st.dataframe(coach, use_container_width=True)

    st.markdown("##### Calls")
    show = filtered[
        [
            c
            for c in [
                "call_id",
                "representative_id",
                "intent",
                "sentiment_ending",
                "resolved",
                "escalated",
                "talk_ratio_customer",
                "summary_redacted",
            ]
            if c in filtered.columns
        ]
    ]
    st.dataframe(show, use_container_width=True)

    export = filtered.copy()
    csv_bytes = export.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export filtered CSV (weekly report)",
        data=csv_bytes,
        file_name=f"voiceiq_report_{datetime.utcnow().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

# ---------- Call detail ----------
elif page == "Call detail":
    st.subheader("Call detail")
    calls = list_calls()
    if not calls:
        st.info("No calls yet.")
        st.stop()
    ids = [c["call_id"] for c in calls]
    default = st.session_state.get("selected_call_id", ids[0])
    idx = ids.index(default) if default in ids else 0
    selected = st.selectbox("Call", ids, index=idx)
    call = get_call(selected)
    if not call:
        st.error("Call not found")
        st.stop()

    a, b, c = st.columns(3)
    a.metric("Intent", call.get("intent"))
    b.metric("Ending sentiment", call.get("sentiment_ending"))
    c.metric("Resolved", "Yes" if call.get("resolved") else "No")

    st.markdown("#### Summary")
    st.write(call.get("summary_redacted") or call.get("summary"))

    st.markdown("#### Coaching")
    st.write(
        {
            "talk_ratio_customer": call.get("talk_ratio_customer"),
            "talk_ratio_agent": call.get("talk_ratio_agent"),
            "turns": call.get("turn_count"),
            "interruption_proxy": call.get("interruption_proxy"),
            "escalated": call.get("escalated"),
            "duration_sec": call.get("duration_sec"),
        }
    )

    if call.get("audio_path") and Path(call["audio_path"]).exists():
        st.markdown("#### Audio")
        st.audio(call["audio_path"])

    st.markdown("#### Transcript")
    turns = call.get("turns") or []
    if turns:
        st.dataframe(pd.DataFrame(turns), use_container_width=True)
    else:
        st.caption("No turn-level transcript stored for this seeded call.")

    if call.get("html_path") and Path(call["html_path"]).exists():
        st.download_button(
            "Download HTML report",
            data=Path(call["html_path"]).read_bytes(),
            file_name=f"{call['call_id']}.html",
            mime="text/html",
        )

# ---------- Ask ----------
elif page == "Ask VoiceIQ":
    st.subheader("Ask VoiceIQ")
    st.caption("Counts/filters use SQLite. Free-text questions use local embeddings + OpenRouter.")
    if st.button("Rebuild search index from DB"):
        n = rebuild_index_from_db()
        st.success(f"Indexed {n} call summaries")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, msg, sources in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(
                            f"- **{s.get('call_id')}**: {s.get('snippet')}"
                        )

    q = st.chat_input("e.g. How many billing calls? What delivery issues happened?")
    if q:
        with st.chat_message("user"):
            st.markdown(q)
        with st.spinner("Thinking..."):
            try:
                result = ask(q)
            except Exception as exc:  # noqa: BLE001
                result = {"answer": f"Error: {exc}", "sources": [], "mode": "error"}
        with st.chat_message("assistant"):
            st.markdown(result["answer"])
            st.caption(f"mode: {result.get('mode')}")
            if result.get("sources"):
                with st.expander("Sources", expanded=True):
                    for s in result["sources"]:
                        st.markdown(
                            f"- **{s.get('call_id')}**: {s.get('snippet')}"
                        )
        st.session_state.chat.append(("user", q, []))
        st.session_state.chat.append(
            ("assistant", result["answer"], result.get("sources") or [])
        )

# ---------- Batch ----------
else:
    st.subheader("Batch inbox")
    st.write(
        f"Drop `.wav` / `.mp3` / `.csv` files into `{settings.inbox_dir}` then process."
    )
    if st.button("Process inbox once"):
        from voiceiq.batch import process_inbox

        with st.spinner("Processing..."):
            process_inbox(once=True)
        st.success("Done — check Dashboard / Call detail")
    st.code("python -m voiceiq.batch --watch", language="bash")
