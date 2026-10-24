from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class UsageDecision:
    """Result of atomically reserving one daily generation call."""

    allowed: bool
    used: int
    limit: int
    remaining: int
    day: str
    whitelisted: bool = False
    total_used: int = 0
    total_limit: int = 0
    total_remaining: int = 0
    rejection: str = ""


class DailyUsageLimiter:
    """Persist all accepted requests and enforce personal and shared daily limits."""

    def __init__(
        self,
        path: Path,
        logger: object,
        *,
        today: Callable[[], date] | None = None,
        tasks_dir: Path | None = None,
    ) -> None:
        """Create a limiter with an injectable date source for deterministic tests."""
        self.path = Path(path)
        self._logger = logger
        self._today = today or (lambda: datetime.now().astimezone().date())
        self._lock = threading.RLock()
        self._tasks_dir = tasks_dir

    def acquire(
        self,
        sender_id: str,
        limit: int,
        *,
        whitelisted: bool = False,
        total_limit: int = 0,
    ) -> UsageDecision:
        """Reserve one call unless the personal or shared daily limit is exhausted."""
        normalized_limit = max(0, int(limit))
        normalized_total_limit = max(0, int(total_limit))
        sender = str(sender_id or "").strip() or "unknown"
        with self._lock:
            day = self._today().isoformat()
            state = self._read_state(day)
            counts = state["counts"]
            used = max(0, int(counts.get(sender, 0)))
            total_used = sum(counts.values())
            rejection = ""
            if normalized_total_limit and total_used >= normalized_total_limit:
                rejection = "total"
            elif not whitelisted and normalized_limit and used >= normalized_limit:
                rejection = "personal"
            if not rejection:
                used += 1
                total_used += 1
                counts[sender] = used
                self._write_state(state)
            return UsageDecision(
                allowed=not rejection,
                used=used,
                limit=normalized_limit,
                remaining=max(0, normalized_limit - used),
                day=day,
                whitelisted=whitelisted,
                total_used=total_used,
                total_limit=normalized_total_limit,
                total_remaining=max(0, normalized_total_limit - total_used),
                rejection=rejection,
            )

    def snapshot(self) -> dict[str, object]:
        """Return today's counts without consuming quota."""
        with self._lock:
            state = self._read_state(self._today().isoformat())
            return {
                **state,
                "counts": dict(state["counts"]),
                "total_used": sum(state["counts"].values()),
            }

    def historical_counts(self, day: str) -> dict[str, int]:
        """Recover accepted requests from retained tasks, excluding known retries."""
        if self._tasks_dir is None or not self._tasks_dir.exists():
            return {}
        tasks = {}
        retry_ids = set()
        for path in self._tasks_dir.glob("*.json"):
            try:
                task = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(task, dict) or task.get("action") != "generate":
                    continue
                task_id = str(task.get("task_id") or path.stem)
                tasks[task_id] = task
                verification = task.get("verification_summary") or {}
                attempts = verification.get("attempts", [])
                # The first attempt is the original request; later IDs are retries.
                retry_ids.update(
                    str(item["task_id"])
                    for item in attempts[1:]
                    if isinstance(item, dict) and item.get("task_id")
                )
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                self._logger.warning(
                    "[comfyui_agent] failed to read usage task: %s", exc
                )
        counts = {}
        for task_id, task in tasks.items():
            if task_id in retry_ids or not str(task.get("time") or "").startswith(
                day + "T"
            ):
                continue
            summary = task.get("prompt_summary")
            summary = summary if isinstance(summary, dict) else {}
            if summary.get("multi_person_prepared_retry") or (
                "【上次问题，请修正】" in str(task.get("original_prompt") or "")
            ):
                continue
            sender = str(task.get("sender_id") or "unknown").strip() or "unknown"
            counts[sender] = counts.get(sender, 0) + 1
        return counts

    def _read_state(self, day: str) -> dict[str, object]:
        """Load sanitized counts for ``day``; reset stale or corrupt state safely."""
        empty = {"version": 2, "date": day, "counts": {}}
        try:
            data = (
                json.loads(self.path.read_text(encoding="utf-8"))
                if self.path.exists()
                else {}
            )
            if not isinstance(data, dict):
                data = {}
            raw_counts = data.get("counts") if data.get("date") == day else {}
            if not isinstance(raw_counts, dict):
                raw_counts = {}
            counts: dict[str, int] = {}
            for sender, value in raw_counts.items():
                try:
                    counts[str(sender)] = max(0, int(value))
                except (TypeError, ValueError):
                    continue
            if data.get("version") != 2:
                for sender, count in self.historical_counts(day).items():
                    counts[sender] = max(counts.get(sender, 0), count)
            state = {"version": 2, "date": day, "counts": counts}
            if data.get("version") != 2:
                self._write_state(state)
            return state
        except Exception as exc:  # noqa: BLE001 - corrupt state must not block drawing.
            self._logger.warning("[comfyui_agent] failed to read daily usage: %s", exc)
            return empty

    def _write_state(self, state: dict[str, object]) -> None:
        """Best-effort atomically replace state without blocking generation on I/O."""
        try:
            payload = json.dumps(state, ensure_ascii=False, indent=2)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
            temp.write_text(payload, encoding="utf-8")
            os.replace(temp, self.path)
        except Exception as exc:  # noqa: BLE001 - generation remains available on I/O failure.
            self._logger.warning("[comfyui_agent] failed to write daily usage: %s", exc)
