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


class DailyUsageLimiter:
    """Persist per-sender daily generation counts in one atomic JSON file."""

    def __init__(
        self,
        path: Path,
        logger: object,
        *,
        today: Callable[[], date] | None = None,
    ) -> None:
        self.path = Path(path)
        self._logger = logger
        self._today = today or (lambda: datetime.now().astimezone().date())
        self._lock = threading.RLock()

    def acquire(
        self,
        sender_id: str,
        limit: int,
        *,
        whitelisted: bool = False,
    ) -> UsageDecision:
        """Reserve one call for ``sender_id`` unless its daily limit is exhausted."""
        day = self._today().isoformat()
        normalized_limit = max(0, int(limit))
        if whitelisted or normalized_limit == 0:
            return UsageDecision(
                allowed=True,
                used=0,
                limit=normalized_limit,
                remaining=normalized_limit,
                day=day,
                whitelisted=whitelisted,
            )

        sender = str(sender_id or "").strip() or "unknown"
        with self._lock:
            state = self._read_state(day)
            counts = state["counts"]
            used = max(0, int(counts.get(sender, 0)))
            if used >= normalized_limit:
                return UsageDecision(
                    allowed=False,
                    used=used,
                    limit=normalized_limit,
                    remaining=0,
                    day=day,
                )
            used += 1
            counts[sender] = used
            self._write_state(state)
            return UsageDecision(
                allowed=True,
                used=used,
                limit=normalized_limit,
                remaining=max(0, normalized_limit - used),
                day=day,
            )

    def _read_state(self, day: str) -> dict[str, object]:
        if not self.path.exists():
            return {"date": day, "counts": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or str(data.get("date") or "") != day:
                return {"date": day, "counts": {}}
            raw_counts = data.get("counts")
            if not isinstance(raw_counts, dict):
                raw_counts = {}
            counts: dict[str, int] = {}
            for sender, value in raw_counts.items():
                try:
                    counts[str(sender)] = max(0, int(value))
                except (TypeError, ValueError):
                    continue
            return {"date": day, "counts": counts}
        except Exception as exc:  # noqa: BLE001 - corrupt state must not block drawing.
            self._logger.warning("[comfyui_agent] failed to read daily usage: %s", exc)
            return {"date": day, "counts": {}}

    def _write_state(self, state: dict[str, object]) -> None:
        try:
            payload = json.dumps(state, ensure_ascii=False, indent=2)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
            temp.write_text(payload, encoding="utf-8")
            os.replace(temp, self.path)
        except Exception as exc:  # noqa: BLE001 - generation remains available on I/O failure.
            self._logger.warning("[comfyui_agent] failed to write daily usage: %s", exc)
