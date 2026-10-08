"""Prevent duplicate deliveries of one platform message from starting new draws."""

from __future__ import annotations

import threading
import time
from typing import Any


class GenerationRequestGuard:
    def __init__(self, *, ttl_seconds: float = 600, max_entries: int = 4096):
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._seen: dict[tuple[str, str, str, str], float] = {}
        self._lock = threading.Lock()

    def claim(self, event: Any) -> bool:
        """Claim a real message ID; events without one remain unrestricted."""
        message = getattr(event, "message_obj", None)
        message_id = str(getattr(message, "message_id", "") or "").strip()
        if not message_id:
            return True
        platform = str(event.get_platform_id() or "")
        session = str(getattr(event, "unified_msg_origin", "") or "")
        sender = str(event.get_sender_id() or "")
        key = (platform, session, sender, message_id)
        now = time.monotonic()
        with self._lock:
            self._seen = {
                existing: expiry
                for existing, expiry in self._seen.items()
                if expiry > now
            }
            if key in self._seen:
                return False
            if len(self._seen) >= self._max_entries:
                oldest = min(self._seen, key=self._seen.get)
                del self._seen[oldest]
            self._seen[key] = now + self._ttl_seconds
            return True


# Shared by plugin instances importing this module in one interpreter.
generation_request_guard = GenerationRequestGuard()
