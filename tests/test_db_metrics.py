import pandas as pd

from voiceiq.db import get_call, init_db, list_calls, upsert_call
from voiceiq.metrics import compute_coaching_metrics


def test_coaching_metrics_talk_ratio():
    df = pd.DataFrame(
        {
            "Speaker": ["Customer", "Representative", "Customer"],
            "Text": ["hello there", "I can help you today", "thanks"],
            "Start": [0, 2, 5],
            "End": [2, 5, 6],
        }
    )
    m = compute_coaching_metrics(df)
    assert m["turn_count"] == 3
    assert m["customer_turn_count"] == 2
    assert m["agent_turn_count"] == 1
    assert 0 <= m["talk_ratio_customer"] <= 1
    assert abs(m["talk_ratio_customer"] + m["talk_ratio_agent"] - 1) < 1e-6


def test_db_roundtrip(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    upsert_call(
        {
            "call_id": "CALL-X",
            "representative_id": "Agent_A",
            "intent": "Billing_Issue",
            "summary": "Test",
            "summary_redacted": "Test",
            "keywords": ["bill"],
            "sentiment_ending": "neutral",
            "resolved": 1,
            "escalated": 0,
        },
        db_path=db,
    )
    rows = list_calls(db_path=db)
    assert len(rows) == 1
    assert rows[0]["call_id"] == "CALL-X"
    assert get_call("CALL-X", db_path=db)["intent"] == "Billing_Issue"
