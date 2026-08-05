"""Simple batch inbox processor (no Redis/Celery required)."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from voiceiq.config import get_settings
from voiceiq.db import enqueue_job, list_jobs, update_job
from voiceiq.pipeline import ingest_path

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".csv"}


def process_inbox(once: bool = True, sleep_sec: float = 5.0) -> None:
    settings = get_settings()
    settings.ensure_dirs()

    def run_once() -> int:
        count = 0
        for path in sorted(settings.inbox_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
                continue
            if path.name.startswith("."):
                continue
            job_id = enqueue_job(str(path))
            update_job(job_id, status="running")
            try:
                result = ingest_path(path)
                dest = settings.processed_dir / path.name
                shutil.move(str(path), str(dest))
                update_job(
                    job_id,
                    status="done",
                    call_id=result.get("call_id"),
                    error=None,
                )
                print(f"Processed {path.name} → {result.get('call_id')}")
                count += 1
            except Exception as exc:  # noqa: BLE001
                update_job(job_id, status="error", error=str(exc))
                print(f"Failed {path.name}: {exc}")
        return count

    if once:
        n = run_once()
        print(f"Batch complete ({n} files). Recent jobs:")
        for job in list_jobs(10):
            print(f"  #{job['id']} {job['status']} {job.get('source_path')}")
        return

    print(f"Watching {settings.inbox_dir} ... Ctrl+C to stop")
    while True:
        run_once()
        time.sleep(sleep_sec)


def main():
    parser = argparse.ArgumentParser(description="VoiceIQ inbox batch processor")
    parser.add_argument("--watch", action="store_true", help="Keep watching inbox/")
    parser.add_argument("--sleep", type=float, default=5.0)
    args = parser.parse_args()
    process_inbox(once=not args.watch, sleep_sec=args.sleep)


if __name__ == "__main__":
    main()
