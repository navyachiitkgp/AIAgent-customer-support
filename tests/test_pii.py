from voiceiq.pii import redact_text


def test_redacts_phone_and_email():
    text = "Call me at 555-123-4567 or jane@example.com about policy id ABC123456"
    out = redact_text(text)
    assert "555-123-4567" not in out
    assert "jane@example.com" not in out
    assert "[PHONE_REDACTED]" in out
    assert "[EMAIL_REDACTED]" in out


def test_redacts_demo_names():
    assert "[NAME_REDACTED]" in redact_text("Thanks James Norton for calling")
