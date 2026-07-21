"""
WellSpring Pharmacy — Customer Call Insight Dashboard

Reads structured JSON call summaries from ./reports/json and charts
intent mix, sentiment by agent, and keyword trends.
"""

import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud

st.set_page_config(page_title="WellSpring Call Insights", layout="wide")
st.markdown(
    "<h1 style='font-size: 32px;'>WellSpring Pharmacy — Customer Call Insight Dashboard</h1>",
    unsafe_allow_html=True,
)
st.caption("Trends from structured call summaries (intent, sentiment, keywords).")


@st.cache_data
def load_json_data(folder: str = "./reports/json") -> pd.DataFrame:
    if not os.path.isdir(folder):
        return pd.DataFrame()

    rows = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(folder, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                rows.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return pd.DataFrame(rows)


df = load_json_data()

if df.empty:
    st.warning(
        "No call summaries found in `./reports/json`. "
        "Run the summarizer first, or keep the bundled sample reports."
    )
    st.stop()

# Normalize columns across older / newer JSON shapes
if "representative_id" not in df.columns and "representative" in df.columns:
    df["representative_id"] = df["representative"]
if "representative_id" not in df.columns:
    df["representative_id"] = "Agent_Unknown"
if "sentiment_ending" not in df.columns:
    df["sentiment_ending"] = df.get("sentiment_customer", "neutral")
if "keywords" not in df.columns:
    df["keywords"] = [[] for _ in range(len(df))]
if "intent" not in df.columns:
    df["intent"] = "General_Inquiry"

df["sentiment_ending"] = df["sentiment_ending"].fillna("neutral").astype(str).str.lower()
df["intent"] = df["intent"].fillna("General_Inquiry").astype(str)
df["representative_id"] = df["representative_id"].fillna("Agent_Unknown").astype(str)

if "model" in df.columns:
    df = df.drop(columns=["model"])

with st.expander("Filters", expanded=True):
    agents = sorted(df["representative_id"].unique().tolist())
    intents = sorted(df["intent"].unique().tolist())
    selected_agents = st.multiselect("Representatives", agents, default=agents)
    selected_intents = st.multiselect("Intents", intents, default=intents)

filtered_df = df[
    df["representative_id"].isin(selected_agents) & df["intent"].isin(selected_intents)
]

if filtered_df.empty:
    st.info("No calls match the current filters.")
    st.stop()

st.markdown(f"**{len(filtered_df)}** calls in view")

st.markdown("### Call Overview")
col1, col2 = st.columns(2)

with col1:
    st.markdown("##### Overall Intent Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(
        data=filtered_df,
        x="intent",
        order=filtered_df["intent"].value_counts().index,
        ax=ax1,
        palette="Set2",
    )
    ax1.set_xlabel("")
    ax1.set_ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig1)
    plt.close(fig1)

with col2:
    st.markdown("##### Sentiment Ending by Call Intent")
    fig2, ax2 = plt.subplots()
    sns.countplot(
        data=filtered_df,
        x="intent",
        hue="sentiment_ending",
        ax=ax2,
        palette="Set2",
    )
    ax2.set_xlabel("")
    ax2.set_ylabel("Number of Calls")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig2)
    plt.close(fig2)

st.markdown("---")
st.markdown("### Agent Performance")
col3, col4 = st.columns(2)

with col3:
    st.markdown("##### Stacked Sentiment Counts per Representative")
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    sentiment_counts = (
        filtered_df.groupby(["representative_id", "sentiment_ending"])
        .size()
        .unstack(fill_value=0)
    )
    # keep a stable column order when present
    ordered = [c for c in ["negative", "neutral", "positive"] if c in sentiment_counts.columns]
    extra = [c for c in sentiment_counts.columns if c not in ordered]
    sentiment_counts = sentiment_counts[ordered + extra]
    colors = {"negative": "#e74c3c", "neutral": "#66c2a5", "positive": "#fc8d62"}
    plot_colors = [colors.get(c, "#8da0cb") for c in sentiment_counts.columns]
    sentiment_counts.plot(kind="bar", stacked=True, ax=ax3, color=plot_colors)
    ax3.set_ylabel("Number of Calls")
    ax3.set_xlabel("")
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

with col4:
    st.markdown("##### Positive Call Rate per Agent")
    rates = (
        filtered_df.groupby("representative_id")["sentiment_ending"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    if "positive" not in rates.columns:
        rates["positive"] = 0.0
    fig6, ax6 = plt.subplots(figsize=(6, 5))
    rates["positive"].plot(kind="bar", color="#8da0cb", ax=ax6)
    ax6.set_ylabel("Proportion Ending Positive")
    ax6.set_xlabel("")
    ax6.set_ylim(0, 1)
    st.pyplot(fig6, use_container_width=True)
    plt.close(fig6)

st.markdown("---")
st.markdown("### Keyword Insights")
col5, col6 = st.columns(2)

all_keywords = []
for item in filtered_df["keywords"].tolist():
    if isinstance(item, list):
        all_keywords.extend([str(k).strip() for k in item if str(k).strip()])
    elif isinstance(item, str) and item.strip():
        all_keywords.extend([k.strip() for k in item.split(",") if k.strip()])

with col5:
    st.markdown("##### Word Cloud")
    if all_keywords:
        wordcloud = WordCloud(
            width=600, height=400, background_color="white"
        ).generate(" ".join(all_keywords))
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wordcloud, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)
        plt.close(fig_wc)
    else:
        st.info("No keywords available for the current filter.")

with col6:
    st.markdown("##### Top Keywords")
    if all_keywords:
        keyword_series = pd.Series(Counter(all_keywords)).sort_values(ascending=False).head(15)
        fig4, ax4 = plt.subplots(figsize=(6, 6))
        sns.barplot(x=keyword_series.values, y=keyword_series.index, color="#8da0cb", ax=ax4)
        ax4.set_xlabel("Frequency")
        st.pyplot(fig4)
        plt.close(fig4)
    else:
        st.info("No keywords to rank.")

with st.expander("Raw call table"):
    show_cols = [
        c
        for c in [
            "call_id",
            "representative_id",
            "intent",
            "sentiment_customer",
            "sentiment_ending",
            "summary",
        ]
        if c in filtered_df.columns
    ]
    st.dataframe(filtered_df[show_cols], use_container_width=True)
