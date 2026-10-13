"""Shared parsing for editable alias-to-tag configuration rows."""
from __future__ import annotations

import re
from typing import Any


_PAIR_PATTERNS = tuple(re.compile(pattern) for pattern in (
    r"\s*(.+?)\s*(?:=>|=|＝)\s*(.+?)\s*",
    r"\s*(.+?)\s*→\s*(.+?)\s*",
    r"\s*(.+?)\s*[:：]\s*(.+?)\s*",
))


def parse_mapping_entry(value: Any) -> tuple[str, str] | None:
    """Prefer equals before legacy arrow/colon separators inside a name.

    A row such as ``作品：副标题=canonical_tag`` must retain its entire alias.
    Arrow and colon notation remain supported when no equals separator exists.
    """
    for pattern in _PAIR_PATTERNS:
        match = pattern.fullmatch(str(value or ""))
        if match:
            return match[1].strip(), match[2].strip()
    return None
