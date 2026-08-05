"""
Generate a ~60s VoiceIQ demo video (MP4) + GIF for the README.

Uses Pillow + imageio (bundles its own ffmpeg via imageio-ffmpeg).
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 1280, 720
BG = (247, 244, 239)
GREEN = (11, 110, 79)
DARK = (26, 26, 26)
MUTED = (91, 91, 91)
WHITE = (255, 255, 255)
CARD = (255, 255, 255)
ACCENT = (212, 163, 115)


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def new_frame() -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    # top bar
    draw.rectangle([0, 0, W, 72], fill=GREEN)
    draw.text((40, 18), "VoiceIQ", fill=WHITE, font=font(32, True))
    draw.text((180, 28), "Support call intelligence", fill=(220, 240, 230), font=font(18))
    return img


def card(draw, xy, wh, title: str, body_lines: list[str]):
    x, y = xy
    w, h = wh
    draw.rounded_rectangle([x, y, x + w, y + h], radius=16, fill=CARD, outline=(220, 220, 220))
    draw.rectangle([x, y, x + 8, y + h], fill=GREEN)
    draw.text((x + 24, y + 18), title, fill=GREEN, font=font(26, True))
    yy = y + 60
    for line in body_lines:
        draw.text((x + 24, yy), line, fill=DARK, font=font(20))
        yy += 32


def scene_title(subtitle: str, bullets: list[str]) -> Image.Image:
    img = new_frame()
    d = ImageDraw.Draw(img)
    d.text((80, 160), "VoiceIQ", fill=GREEN, font=font(64, True))
    d.text((80, 240), subtitle, fill=DARK, font=font(28))
    y = 320
    for b in bullets:
        d.ellipse([80, y + 8, 96, y + 24], fill=ACCENT)
        d.text((110, y), b, fill=MUTED, font=font(24))
        y += 48
    d.text((80, 640), "Pharmacy support calls → searchable insights", fill=MUTED, font=font(18))
    return img


def scene_analyze() -> Image.Image:
    img = new_frame()
    d = ImageDraw.Draw(img)
    d.text((80, 110), "1) Analyze a call", fill=DARK, font=font(36, True))
    card(
        d,
        (80, 180),
        (520, 400),
        "Upload audio / CSV",
        [
            "Billing_Issue_sample.wav",
            "",
            "Pipeline:",
            "• Whisper transcription",
            "• PII redaction",
            "• Intent / sentiment / keywords",
            "• Coaching metrics → SQLite",
        ],
    )
    card(
        d,
        (640, 180),
        (540, 400),
        "Result",
        [
            "Call ID: AUD-4821-193011",
            "Intent: Billing_Issue",
            "Sentiment: negative → resolved",
            "Talk ratio cust/agent: 0.48 / 0.52",
            "",
            "Summary (redacted):",
            "Customer disputed an inhaler",
            "bill; agent asked for updated",
            "insurance card to reprocess.",
        ],
    )
    return img


def scene_dashboard() -> Image.Image:
    img = new_frame()
    d = ImageDraw.Draw(img)
    d.text((80, 110), "2) Operations dashboard", fill=DARK, font=font(36, True))
    # metric boxes
    metrics = [("27", "Calls"), ("4", "Billing"), ("2", "Escalated"), ("81%", "Positive end")]
    x = 80
    for val, label in metrics:
        d.rounded_rectangle([x, 180, x + 260, 300], radius=14, fill=CARD, outline=(220, 220, 220))
        d.text((x + 30, 200), val, fill=GREEN, font=font(40, True))
        d.text((x + 30, 255), label, fill=MUTED, font=font(20))
        x += 290
    card(
        d,
        (80, 340),
        (1100, 280),
        "Manager view",
        [
            "Filters: agent · intent · unresolved · date range",
            "Charts: intent mix, ending sentiment, coaching snapshot",
            "Export: weekly CSV report for leadership",
        ],
    )
    return img


def scene_ask() -> Image.Image:
    img = new_frame()
    d = ImageDraw.Draw(img)
    d.text((80, 110), "3) Ask VoiceIQ (hybrid RAG)", fill=DARK, font=font(36, True))
    card(
        d,
        (80, 180),
        (1100, 160),
        "You",
        ["How many billing calls?", "", "mode: sql  →  exact count from SQLite"],
    )
    card(
        d,
        (80, 370),
        (1100, 240),
        "VoiceIQ",
        [
            "There are 4 billing-related calls in the database.",
            "",
            "Sources:",
            "• CALL-5159  Advair bill / insurance card update",
            "• CALL-5678  duplicate charge refund",
            "• CALL-5161  copay change / temporary credit",
        ],
    )
    return img


def scene_eval(metrics: dict) -> Image.Image:
    img = new_frame()
    d = ImageDraw.Draw(img)
    d.text((80, 110), "4) Measured quality", fill=DARK, font=font(36, True))
    n = metrics.get("n", 10)
    intent = int(round(100 * float(metrics.get("intent_accuracy", 0))))
    sent = int(round(100 * float(metrics.get("sentiment_accuracy", 0))))
    cov = int(round(100 * float(metrics.get("summary_keyword_coverage", 0))))
    mode = metrics.get("mode", "llm")
    card(
        d,
        (80, 180),
        (1100, 420),
        f"LLM eval on {n} labeled calls ({mode})",
        [
            f"Intent accuracy:                 {intent}%",
            f"Customer sentiment accuracy:     {sent}%",
            f"Summary keyword coverage:        {cov}%",
            "",
            "Eval script:  python scripts/run_eval.py",
            "Labels live in eval_data/labeled_calls.json",
            "",
            "Built for pharmacy support ops, QA coaching,",
            "and searchable call intelligence — not raw audio silos.",
        ],
    )
    return img


def scene_end() -> Image.Image:
    img = new_frame()
    d = ImageDraw.Draw(img)
    d.text((80, 220), "VoiceIQ", fill=GREEN, font=font(64, True))
    d.text((80, 300), "From call recording to decision-ready insight.", fill=DARK, font=font(28))
    d.text((80, 380), "streamlit run app.py", fill=MUTED, font=font(24))
    d.text((80, 430), "github.com/navyachiitkgp/AIAgent-customer-support", fill=MUTED, font=font(22))
    return img


def load_metrics() -> dict:
    path = ROOT / "eval_data" / "published_eval_report.json"
    if path.exists():
        import json

        return json.loads(path.read_text())
    return {
        "n": 10,
        "intent_accuracy": 0.8,
        "sentiment_accuracy": 0.8,
        "summary_keyword_coverage": 0.8,
        "mode": "pending",
    }


def build():
    import imageio.v2 as imageio

    metrics = load_metrics()
    # ~60 seconds at 2 fps = 120 frames; hold each scene ~10s = 20 frames
    scenes = [
        (scene_title(
            "Pharmacy support call intelligence",
            [
                "Transcribe & summarize calls",
                "Dashboard KPIs + agent coaching",
                "Ask questions with cited sources",
            ],
        ), 10),
        (scene_analyze(), 12),
        (scene_dashboard(), 12),
        (scene_ask(), 13),
        (scene_eval(metrics), 10),
        (scene_end(), 5),
    ]

    frames = []
    fps = 2
    for img, seconds in scenes:
        for _ in range(seconds * fps):
            frames.append(img.convert("RGB"))

    mp4_path = OUT_DIR / "voiceiq_demo.mp4"
    gif_path = OUT_DIR / "voiceiq_demo.gif"

    # MP4 via imageio-ffmpeg plugin
    writer = imageio.get_writer(
        mp4_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    )
    for fr in frames:
        writer.append_data(__import__("numpy").asarray(fr))
    writer.close()

    # Lighter GIF for README embed (downsample frames)
    gif_frames = frames[::2]
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(1000 / fps) * 2,
        loop=0,
        optimize=True,
    )

    print(f"Wrote {mp4_path} ({mp4_path.stat().st_size // 1024} KB)")
    print(f"Wrote {gif_path} ({gif_path.stat().st_size // 1024} KB)")
    print(f"Duration ~{sum(s for _, s in scenes)}s")


if __name__ == "__main__":
    build()
