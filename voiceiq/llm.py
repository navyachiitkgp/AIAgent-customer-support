"""OpenRouter LLM helpers for summarization / intent / sentiment."""

from __future__ import annotations

from typing import List

import requests

from voiceiq.config import get_settings

ALLOWED_INTENTS = {
    "Refill_Request",
    "Billing_Issue",
    "Medication_Change",
    "Delivery_Status",
    "Side_Effect",
    "Doctor_Contact",
    "General_Inquiry",
}


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        settings = get_settings()
        self.api_key = api_key or settings.require_openrouter()
        self.model = model or settings.summary_model
        self.api_url = f"{settings.openrouter_base_url}/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/navyachiitkgp/AIAgent-customer-support",
            "X-Title": "VoiceIQ",
        }

    def chat(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = requests.post(
            self.api_url, headers=self.headers, json=payload, timeout=90
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def summarize(self, transcript: str) -> str:
        prompt = (
            "Summarize this pharmacy customer support conversation into 2–3 sentences. "
            "Focus on what the customer wanted and how the representative responded.\n\n"
            f"Transcript:\n{transcript[:3000]}"
        )
        return self.chat(prompt, max_tokens=400, temperature=0.2)

    def keywords(self, summary: str) -> List[str]:
        prompt = (
            "Extract 3–5 important single-word keywords. "
            "Return only a comma-separated list.\n\n"
            f"Summary:\n{summary}"
        )
        try:
            raw = self.chat(prompt, max_tokens=80)
            return [k.strip() for k in raw.split(",") if k.strip()][:5]
        except Exception:
            return []

    def intent(self, summary: str) -> str:
        prompt = (
            "Classify the customer's intent. Choose ONE of: "
            + ", ".join(sorted(ALLOWED_INTENTS))
            + f"\n\nSummary:\n{summary}\nIntent:"
        )
        try:
            raw = self.chat(prompt, max_tokens=40).replace(" ", "_")
            for a in ALLOWED_INTENTS:
                if a.lower() in raw.lower():
                    return a
        except Exception:
            pass
        return "General_Inquiry"

    def sentiment(self, summary: str, speaker: str) -> str:
        prompt = (
            f"Classify the sentiment of the {speaker}. "
            "Reply with exactly one word: positive, neutral, or negative.\n\n"
            f"Summary:\n{summary}"
        )
        try:
            label = self.chat(prompt, max_tokens=10).lower()
            for option in ("positive", "negative", "neutral"):
                if option in label:
                    return option
        except Exception:
            pass
        return "neutral"
