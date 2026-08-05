#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv venv 2>/dev/null || true
# shellcheck disable=SC1091
source venv/bin/activate
pip install -q -r requirements-core.txt
python scripts/seed_db.py
echo ""
echo "VoiceIQ demo ready."
echo "  App:  ./venv/bin/streamlit run app.py"
echo "  API:  ./venv/bin/uvicorn api.main:app --reload"
echo "  Eval: python scripts/run_eval.py --offline"
