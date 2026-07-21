"""
Call transcript summarizer — OpenRouter-backed LLM pipeline.

Takes a diarized CSV (Speaker, Text), produces a short summary plus
intent / sentiment / keywords, then writes JSON + HTML under ./reports/.
"""

import base64
import json
import os
import random
from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns


class OpenRouterTranscriptSummarizer:
    def __init__(self, api_key: str, model: str = "openchat/openchat-3.5-0106"):
        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/navyachiitkgp/AIAgent-customer-support",
        }
        self.model = model

    def _chat(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=90)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    def summarize_from_csv(
        self,
        csv_input,
        text_column="Text",
        max_tokens=512,
        call_id="call_001",
        representative="Agent_Unknown",
    ):
        if not isinstance(csv_input, str):
            csv_input.seek(0)

        df = pd.read_csv(csv_input)
        if text_column not in df.columns:
            raise ValueError(f"Expected column '{text_column}' in CSV. Found: {list(df.columns)}")

        text = " ".join(df[text_column].dropna().astype(str).tolist())[:3000]
        prompt = (
            "Summarize this customer support conversation into 2–3 meaningful sentences. "
            "Focus on what the customer wanted and how the representative responded.\n\n"
            f"Transcript:\n{text}"
        )

        summaries = {}
        try:
            summary_text = self._chat(prompt, max_tokens=max_tokens, temperature=0.2)
            keywords = self.extract_keywords_llm(summary_text)
            intent = self.extract_intent_llm(summary_text)
            sentiment_customer = self.extract_sentiment_llm(summary_text, speaker="customer")
            sentiment_representative = self.extract_sentiment_llm(
                summary_text, speaker="representative"
            )

            # Per-turn sentiment first — needed for ending sentiment + charts
            chart_base64_1, chart_base64_2 = self.generate_sentiment_charts(df)

            payload = {
                "summary": summary_text,
                "keywords": keywords,
                "intent": intent,
                "sentiment_customer": sentiment_customer,
                "sentiment_representative": sentiment_representative,
            }
            summaries[self.model] = payload

            self.save_summary_json(
                call_id=call_id,
                representative=representative,
                summary=summary_text,
                keywords=keywords,
                intent=intent,
                sentiment_customer=sentiment_customer,
                sentiment_representative=sentiment_representative,
                model=self.model,
                df=df,
            )

            safe_model = self.model.replace("/", "_")
            self.export_summary_html(
                payload,
                file_path=f"./reports/html/{call_id}_{safe_model}.html",
                title=call_id,
                representative=representative,
                chart_base64_list=[chart_base64_1, chart_base64_2],
            )
            print(f"Wrote reports for {call_id}")

        except Exception as e:
            summaries[self.model] = {
                "summary": f"Request failed: {str(e)}",
                "keywords": [],
                "intent": "N/A",
                "sentiment_customer": "N/A",
                "sentiment_representative": "N/A",
            }
            print(f"Summarization failed for {call_id}: {e}")

        return summaries

    def generate_sentiment_charts(self, df):
        sentiments = []
        for _, row in df.iterrows():
            content = str(row.get("Text", ""))
            prompt = (
                "Classify the sentiment of the following customer support utterance:\n\n"
                f"{content}\n\nChoose one of: positive, neutral, negative."
            )
            try:
                label = self._chat(prompt, max_tokens=10).lower()
                if "positive" in label:
                    label = "positive"
                elif "negative" in label:
                    label = "negative"
                else:
                    label = "neutral"
            except Exception:
                label = "neutral"
            sentiments.append(label)

        df = df.copy()
        df["Sentiment"] = sentiments
        sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
        df["SentimentScore"] = df["Sentiment"].map(sentiment_map)
        df["Exchange"] = range(1, len(df) + 1)
        self._last_sentiment_df = df

        palette = {
            "positive": "#2ecc71",
            "neutral": "#3498db",
            "negative": "#e74c3c",
        }

        fig1, ax1 = plt.subplots(figsize=(6, 5))
        speaker_col = "Speaker" if "Speaker" in df.columns else df.columns[0]
        sns.countplot(data=df, x=speaker_col, hue="Sentiment", ax=ax1, palette=palette)
        ax1.set_title("Overall Sentiment Distribution by Speaker", fontsize=14, weight="bold")
        ax1.set_ylabel("Number of Turns")
        ax1.set_xlabel("Speaker")
        chart_base64_1 = self.fig_to_base64(fig1)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        for speaker, color in zip(["Representative", "Customer"], ["orange", "blue"]):
            speaker_df = df[df[speaker_col].astype(str).str.lower() == speaker.lower()]
            if speaker_df.empty:
                continue
            ax2.plot(
                speaker_df["Exchange"],
                speaker_df["SentimentScore"],
                label=f"{speaker} Trend",
                color=color,
            )
            ax2.scatter(
                speaker_df["Exchange"],
                speaker_df["SentimentScore"],
                color=color,
                alpha=0.6,
            )
        ax2.axhline(0, color="gray", linestyle="--", label="Neutral Baseline")
        ax2.set_title("Sentiment Fluctuation Over Conversation by Speaker")
        ax2.set_xlabel("Conversation Exchange")
        ax2.set_ylabel("Sentiment Score (-1 to 1)")
        ax2.legend()
        chart_base64_2 = self.fig_to_base64(fig2)
        plt.close(fig2)

        return chart_base64_1, chart_base64_2

    def fig_to_base64(self, fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def export_summary_html(
        self,
        content: dict,
        file_path: str,
        title: str,
        representative: str = "Agent_Unknown",
        chart_base64_list=None,
    ):
        chart_base64_list = chart_base64_list or []
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        call_id = title if title.startswith("CALL-") else f"CALL-{random.randint(1000, 9999)}"
        customer_id = f"CUST-{random.randint(10000, 99999)}"
        today = datetime.now().date()
        intent = str(content.get("intent", "General_Inquiry")).replace("_", " ").title()

        html = f"""
        <html>
        <head>
            <title>Call Analysis Report – {intent}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metadata-box {{
                    background-color: #f4f4f4;
                    border-left: 5px solid #0077b6;
                    padding: 10px 20px;
                    margin-bottom: 20px;
                    font-size: 14px;
                }}
                .quick-insights {{
                    background-color: #fdfcdc;
                    border-left: 5px solid #ffb703;
                    padding: 15px 20px;
                    margin-bottom: 20px;
                    font-size: 15px;
                }}
                summary {{ font-weight: bold; margin-top: 15px; font-size: 16px; }}
                details {{
                    margin-bottom: 20px;
                    border: 1px solid #ccc;
                    padding: 10px;
                    border-radius: 4px;
                    background: #fafafa;
                }}
                button {{
                    background-color: #0077b6;
                    color: white;
                    padding: 10px 16px;
                    border: none;
                    border-radius: 4px;
                    font-size: 14px;
                    cursor: pointer;
                }}
                button:hover {{ background-color: #005f86; }}
            </style>
            <script>
                function printPage() {{ window.print(); }}
            </script>
        </head>
        <body>
            <h1>Call Analysis Report</h1>
            <div class="metadata-box">
                <b>Call ID:</b> {call_id} |
                <b>Date:</b> {today} |
                <b>Customer ID:</b> {customer_id} |
                <b>Representative:</b> {representative}
            </div>

            <div class="quick-insights">
                <b>Quick Insights:</b>
                <ul>
                    <li><b>Intent:</b> {intent}</li>
                    <li><b>Customer Sentiment:</b> {str(content.get('sentiment_customer', 'N/A')).title()}</li>
                    <li><b>Representative Sentiment:</b> {str(content.get('sentiment_representative', 'N/A')).title()}</li>
                    <li><b>Keywords:</b> {', '.join(content.get('keywords') or [])}</li>
                </ul>
            </div>

            <details open>
                <summary>Full Summary</summary>
                <p>{content.get('summary', '')}</p>
            </details>

            <details open>
                <summary>Charts</summary>
        """

        for chart_base64 in chart_base64_list:
            html += (
                f"<img src='data:image/png;base64,{chart_base64}' "
                "style='max-width:100%; margin-bottom:20px;'><br>"
            )

        html += """
            </details>
            <button onclick="printPage()">Print to PDF</button>
        </body>
        </html>
        """

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html)

    def extract_keywords_llm(self, summary_text):
        prompt = (
            "Extract 3–5 important keywords from this summary. "
            "Each keyword has to be a single word, not a phrase. "
            "Do not extract more than 5 keywords. "
            "Return only a comma-separated list, no explanation.\n\n"
            f"Summary:\n{summary_text}"
        )
        try:
            keyword_str = self._chat(prompt, max_tokens=100)
            return [kw.strip() for kw in keyword_str.split(",") if kw.strip()][:5]
        except Exception:
            return []

    def extract_intent_llm(self, summary_text):
        prompt = (
            "Classify the customer's intent from this summary.\n"
            "Choose one of the following intents: Refill_Request, Billing_Issue, "
            "Medication_Change, Delivery_Status, Side_Effect, Doctor_Contact, General_Inquiry.\n"
            f"Summary:\n{summary_text}\nIntent:"
        )
        try:
            intent = self._chat(prompt, max_tokens=50)
            # normalize spaces → underscores
            intent = intent.split()[0].replace(" ", "_")
            allowed = {
                "Refill_Request",
                "Billing_Issue",
                "Medication_Change",
                "Delivery_Status",
                "Side_Effect",
                "Doctor_Contact",
                "General_Inquiry",
            }
            for a in allowed:
                if a.lower() in intent.lower().replace(" ", "_"):
                    return a
            return "General_Inquiry"
        except Exception:
            return "General_Inquiry"

    def extract_sentiment_llm(self, summary_text, speaker):
        prompt = (
            f"Classify the sentiment of the {speaker} in this summary.\n"
            "Only respond with ONE of the following words:\n"
            "- positive\n- neutral\n- negative\n\n"
            "DO NOT explain. Just reply with one word only.\n"
            f"Summary:\n{summary_text}\nSentiment:"
        )
        try:
            label = self._chat(prompt, max_tokens=20).lower()
            for option in ("positive", "negative", "neutral"):
                if option in label:
                    return option
            return "neutral"
        except Exception:
            return "neutral"

    def save_summary_json(
        self,
        call_id,
        representative,
        summary,
        keywords,
        intent,
        sentiment_customer,
        sentiment_representative,
        model,
        df,
    ):
        customer_id = f"CUST-{random.randint(10000, 99999)}"

        sentiment_df = getattr(self, "_last_sentiment_df", None)
        ending_sentiment = "neutral"
        if sentiment_df is not None and "Speaker" in sentiment_df.columns:
            customer_rows = sentiment_df[
                sentiment_df["Speaker"].astype(str).str.lower() == "customer"
            ]
            if not customer_rows.empty and "Sentiment" in customer_rows.columns:
                ending_sentiment = customer_rows.iloc[-1]["Sentiment"]
        elif df is not None and "Speaker" in df.columns and "Sentiment" in df.columns:
            customer_rows = df[df["Speaker"].astype(str).str.lower() == "customer"]
            if not customer_rows.empty:
                ending_sentiment = customer_rows.iloc[-1]["Sentiment"]
        else:
            ending_sentiment = sentiment_customer or "neutral"

        # dashboard expects representative_id
        result = {
            "call_id": call_id,
            "customer_id": customer_id,
            "representative_id": representative,
            "summary": summary,
            "keywords": keywords,
            "intent": intent,
            "sentiment_customer": sentiment_customer,
            "sentiment_representative": sentiment_representative,
            "sentiment_ending": ending_sentiment,
            "model": model,
            "timestamp": datetime.now().isoformat(),
        }

        os.makedirs("./reports/json", exist_ok=True)
        path = f"./reports/json/{call_id}_{model.replace('/', '_')}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return path
