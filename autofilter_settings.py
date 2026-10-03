"""Persistent per-conversation image filtering choices."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class AutofilterSettings:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            data = self._read()
            sessions = data.get("sessions", {})
            return {"sessions": sessions if isinstance(sessions, dict) else {}}

    def observe(self, session: str) -> None:
        if not session:
            return
        with self._lock:
            data = self.snapshot()
            if session not in data["sessions"]:
                data["sessions"][session] = False
                self._write(data)

    def enabled(self, session: str) -> bool:
        return bool(self.snapshot()["sessions"].get(session, False)) if session else False

    def set_enabled(self, session: str, enabled: bool) -> None:
        if not session or len(session) > 512:
            raise ValueError("会话 ID 无效")
        if not isinstance(enabled, bool):
            raise ValueError("enabled 必须为布尔值")
        with self._lock:
            data = self.snapshot()
            data["sessions"][session] = enabled
            self._write(data)
