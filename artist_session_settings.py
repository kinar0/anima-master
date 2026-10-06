"""Persistent active artist preset choices keyed by AstrBot conversation."""
from __future__ import annotations

import json
import threading
from pathlib import Path


class ArtistSessionSettings:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()

    def _read(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        sessions = data.get("sessions") if isinstance(data, dict) else None
        return (
            {key: value for key, value in sessions.items()
             if isinstance(key, str) and isinstance(value, str)}
            if isinstance(sessions, dict) else {}
        )

    def _write(self, sessions: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"sessions": sessions}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def get(self, session: str) -> str | None:
        if not session:
            return None
        with self._lock:
            return self._read().get(session)

    def set(self, session: str, name: str) -> None:
        if not session or len(session) > 512:
            raise ValueError("会话 ID 无效")
        with self._lock:
            sessions = self._read()
            sessions[session] = name
            self._write(sessions)

    def remove_preset(self, name: str) -> None:
        with self._lock:
            sessions = self._read()
            updated = {
                key: ("" if value == name else value)
                for key, value in sessions.items()
            }
            if updated != sessions:
                self._write(updated)


def artist_session_key(event: object) -> str:
    """Use the platform-qualified chat origin, including group/private scope."""
    return str(getattr(event, "unified_msg_origin", "") or "").strip()
