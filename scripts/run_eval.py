"""Run evaluation against labeled sample transcripts.

Usage:
  python scripts/run_eval.py            # uses LLM (needs OPENROUTER_API_KEY)
  python scripts/run_eval.py --offline  # heuristic-only smoke metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voiceiq.llm import ALLOWED_INTENTS, OpenRouterClient
from voiceiq.pipeline import load_transcript_csv
from voiceiq.pii import redact_text


def heuristic_intent(text: str) -> str:
    t = text.lower()
    rules = [
        ("Billing_Issue", ["bill", "charged", "refund", "insurance", "copay"]),
        ("Delivery_Status", ["delivery", "tracking", "ship", "usps", "parcel"]),
        ("Refill_Request", ["refill", "ran out"]),
        ("Medication_Change", ["switch", "changed", "dosage", "instead of"]),
        ("Doctor_Contact", ["doctor", "clinic", "fax request"]),
        ("Side_Effect", ["side effect", "dizzy", "rash", "nausea"]),
    ]
    for intent, keys in rules:
        if any(k in t for k in keys):
            return intent
    return "General_Inquiry"


def heuristic_sentiment(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["angry", "frustrated", "overcharged", "confused", "worried"]):
        return "negative"
    if any(w in t for w in ["thank", "great", "perfect", "appreciate"]):
        return "positive"
    return "neutral"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    labeled = json.loads((ROOT / "eval_data" / "labeled_calls.json").read_text())
    client = None if args.offline else OpenRouterClient()

    intent_ok = sent_ok = summary_ok = 0
    rows = []
    for item in labeled:
        path = ROOT / "sample_data" / "transcripts" / item["transcript_file"]
        df = load_transcript_csv(path)
        text = redact_text(" ".join(df["Text"].astype(str).tolist()))
        if args.offline:
            pred_intent = heuristic_intent(text)
            pred_sent = heuristic_sentiment(text)
            summary = text[:280]
        else:
            summary = client.summarize(text)
            pred_intent = client.intent(summary)
            pred_sent = client.sentiment(summary, "customer")

        i_hit = pred_intent == item["expected_intent"]
        s_hit = pred_sent == item["expected_sentiment_customer"]
        must = item.get("summary_must_include") or []
        sum_hit = all(m.lower() in summary.lower() for m in must)
        intent_ok += int(i_hit)
        sent_ok += int(s_hit)
        summary_ok += int(sum_hit)
        rows.append(
            {
                "call_id": item["call_id"],
                "intent_pred": pred_intent,
                "intent_ok": i_hit,
                "sentiment_pred": pred_sent,
                "sentiment_ok": s_hit,
                "summary_ok": sum_hit,
            }
        )

    n = max(len(labeled), 1)
    report = {
        "n": n,
        "intent_accuracy": round(intent_ok / n, 3),
        "sentiment_accuracy": round(sent_ok / n, 3),
        "summary_keyword_coverage": round(summary_ok / n, 3),
        "mode": "offline" if args.offline else "llm",
        "details": rows,
        "allowed_intents": sorted(ALLOWED_INTENTS),
    }
    out = ROOT / "eval_data" / "last_eval_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Stable artifact for README (committed)
    if not args.offline:
        published = ROOT / "eval_data" / "published_eval_report.json"
        published.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {published}")
    print(json.dumps({k: report[k] for k in report if k != "details"}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
