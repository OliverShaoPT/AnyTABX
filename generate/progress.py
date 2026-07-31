"""Progress reporting for parallel record generation."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def render_bar(completed: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("?" * width) + "]"
    frac = min(1.0, completed / total)
    filled = int(round(frac * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


@dataclass
class ProgressTracker:
    """Aggregate per-record events into a console bar + on-disk snapshot."""

    total: int
    output_root: Path
    started_at: float = field(default_factory=time.time)
    completed: int = 0
    last_event: dict[str, Any] | None = None
    jsonl_path: Path | None = None
    snapshot_path: Path | None = None

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_root / "generation_progress.jsonl"
        self.snapshot_path = self.output_root / "generation_progress.json"
        # Fresh run marker.
        self.jsonl_path.write_text("", encoding="utf-8")
        self._write_snapshot(status="running")

    def handle(self, event: dict[str, Any]) -> None:
        kind = str(event.get("event", "record_done"))
        if kind == "record_done":
            self.completed += 1
            self.last_event = event
            self._append_jsonl(event)
            self._write_snapshot(status="running")
            self._print_line(final=False)
        elif kind == "worker_error":
            self.last_event = event
            self._append_jsonl(event)
            self._write_snapshot(status="running", error=event.get("error"))
            print(
                f"[parallel_records] worker_error worker={event.get('worker_id')} "
                f"{event.get('error')}",
                flush=True,
            )

    def finish(self, *, status: str = "done") -> None:
        self._write_snapshot(status=status)
        self._print_line(final=True)
        elapsed = time.time() - self.started_at
        rate = self.completed / elapsed if elapsed > 0 else 0.0
        print(
            f"[parallel_records] summary status={status} "
            f"completed={self.completed}/{self.total} "
            f"elapsed={format_duration(elapsed)} "
            f"rate={rate:.3f} rec/s "
            f"progress_file={self.snapshot_path}",
            flush=True,
        )

    def _append_jsonl(self, event: dict[str, Any]) -> None:
        assert self.jsonl_path is not None
        row = {
            **event,
            "completed": self.completed,
            "total": self.total,
            "elapsed_s": round(time.time() - self.started_at, 3),
        }
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    def _write_snapshot(self, *, status: str, error: Any = None) -> None:
        assert self.snapshot_path is not None
        elapsed = time.time() - self.started_at
        rate = self.completed / elapsed if elapsed > 0 and self.completed else 0.0
        remaining = max(0, self.total - self.completed)
        eta_s = (remaining / rate) if rate > 0 else None
        payload = {
            "status": status,
            "completed": self.completed,
            "total": self.total,
            "percent": round(100.0 * self.completed / self.total, 2) if self.total else 0.0,
            "elapsed_s": round(elapsed, 3),
            "rate_rec_per_s": round(rate, 4),
            "eta_s": None if eta_s is None else round(eta_s, 3),
            "eta": None if eta_s is None else format_duration(eta_s),
            "last": self.last_event,
            "error": error,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        temporary = self.snapshot_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(self.snapshot_path)

    def _print_line(self, *, final: bool) -> None:
        elapsed = time.time() - self.started_at
        rate = self.completed / elapsed if elapsed > 0 and self.completed else 0.0
        remaining = max(0, self.total - self.completed)
        eta = format_duration(remaining / rate) if rate > 0 else "?"
        pct = 100.0 * self.completed / self.total if self.total else 0.0
        last = ""
        if self.last_event is not None:
            pkg = self.last_event.get("package_name", "?")
            rid = self.last_event.get("record_id", "?")
            wid = self.last_event.get("worker_id", "?")
            last = f" last=w{wid}:{pkg}/record-{rid}"
        line = (
            f"\r[parallel_records] {render_bar(self.completed, self.total)} "
            f"{self.completed}/{self.total} ({pct:5.1f}%) "
            f"{rate:.3f} rec/s elapsed={format_duration(elapsed)} ETA={eta}{last}"
        )
        end = "\n" if final or self.completed >= self.total else ""
        sys.stdout.write(line + end)
        sys.stdout.flush()


class LocalQueue:
    """In-process stand-in for multiprocessing.Queue."""

    def __init__(self, on_put) -> None:
        self._on_put = on_put

    def put(self, item: Any) -> None:
        self._on_put(item)
