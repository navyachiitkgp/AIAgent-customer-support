from voiceiq.rag import structured_answer


def test_structured_answer_empty_db(tmp_path, monkeypatch):
    import voiceiq.db as db
    import voiceiq.rag as rag

    monkeypatch.setattr(db, "list_calls", lambda **kwargs: [])
    monkeypatch.setattr(rag, "list_calls", lambda **kwargs: [])
    out = structured_answer("How many billing calls?")
    assert out is not None
    assert "No calls" in out["answer"]
