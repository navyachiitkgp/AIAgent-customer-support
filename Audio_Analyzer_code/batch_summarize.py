"""
Batch helper: turn every CSV under sample_data/transcripts/ into
JSON + HTML call reports via the OpenRouter summarizer.

Usage:
  python Audio_Analyzer_code/batch_summarize.py
  python Audio_Analyzer_code/batch_summarize.py --limit 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from individual_call_transform import OpenRouterTranscriptSummarizer  # noqa: E402

load_dotenv(ROOT / ".env")

# Map filename stems → (call_id, representative) for the sample set
CALL_META = {
    "Billing_Issue_1": ("CALL-5159", "Agent_D"),
    "Billing_Issue_2": ("CALL-5678", "Agent_A"),
    "Delivery_Status_1": ("CALL-6030", "Agent_B"),
    "Delivery_Status_2": ("CALL-7191", "Agent_A"),
    "Doctor_Contact_1": ("CALL-6208", "Agent_A"),
    "General_Inquiry_1": ("CALL-8606", "Agent_A"),
    "General_Inquiry_2": ("CALL-9462", "Agent_C"),
    "Medication_Change_1": ("CALL-2266", "Agent_B"),
    "Medication_Change_2": ("CALL-1815", "Agent_C"),
    "Refill_Request_1": ("CALL-7338", "Agent_C"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transcript_dir",
        default=str(ROOT / "sample_data" / "transcripts"),
    )
    parser.add_argument("--limit", type=int, default=0, help="Only process N files (0 = all).")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set OPENROUTER_API_KEY in .env before running batch summarization."
        )

    summarizer = OpenRouterTranscriptSummarizer(api_key=api_key)
    files = sorted(Path(args.transcript_dir).glob("*.csv"))
    if args.limit:
        files = files[: args.limit]

    if not files:
        raise SystemExit(f"No CSV transcripts in {args.transcript_dir}")

    os.chdir(ROOT)
    for csv_path in files:
        stem = csv_path.stem
        call_id, rep = CALL_META.get(stem, (stem, "Agent_Unknown"))
        print(f"\n=== {stem} → {call_id} ({rep}) ===")
        summarizer.summarize_from_csv(
            csv_input=str(csv_path),
            call_id=call_id,
            representative=rep,
        )


if __name__ == "__main__":
    main()
