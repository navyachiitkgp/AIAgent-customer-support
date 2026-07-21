"""
CLI wrapper around OpenRouterTranscriptSummarizer.

Example:
  python class_call-individual_call_transform.py \\
      --csv sample_data/transcripts/Billing_Issue_1.csv \\
      --call_id CALL-5159 \\
      --representative Agent_D
"""

import argparse
import os

from dotenv import load_dotenv

from individual_call_transform import OpenRouterTranscriptSummarizer

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY not found. Copy .env.example to .env and add your key."
    )

parser = argparse.ArgumentParser(description="Summarize a customer support call transcript.")
parser.add_argument("--csv", required=True, help="Path to the input CSV transcript.")
parser.add_argument("--call_id", default="call_001", help="Call ID (default: call_001).")
parser.add_argument(
    "--representative",
    default="Agent_Unknown",
    help="Representative / agent name.",
)
parser.add_argument(
    "--model",
    default="openchat/openchat-3.5-0106",
    help="OpenRouter model id.",
)
args = parser.parse_args()

if not os.path.isfile(args.csv):
    raise FileNotFoundError(f"CSV not found: {args.csv}")

summarizer = OpenRouterTranscriptSummarizer(api_key=api_key, model=args.model)

print(f"Summarizing call: {args.call_id} ...")
summarizer.summarize_from_csv(
    csv_input=args.csv,
    call_id=args.call_id,
    representative=args.representative,
)
print("Done. Check ./reports/html/ and ./reports/json/")
