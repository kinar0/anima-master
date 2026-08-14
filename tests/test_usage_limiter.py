from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from usage_limiter import DailyUsageLimiter  # noqa: E402


class _Logger:
    def warning(self, *_args) -> None:
        pass


def test_daily_limit_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    today = lambda: date(2026, 8, 14)
    first = DailyUsageLimiter(path, _Logger(), today=today)

    assert first.acquire("10001", 2).allowed is True
    assert first.acquire("10001", 2).allowed is True

    restarted = DailyUsageLimiter(path, _Logger(), today=today)
    rejected = restarted.acquire("10001", 2)
    assert rejected.allowed is False
    assert rejected.used == 2
    assert json.loads(path.read_text(encoding="utf-8"))["counts"] == {"10001": 2}


def test_new_day_resets_counts_lazily(tmp_path: Path) -> None:
    current = [date(2026, 8, 14)]
    limiter = DailyUsageLimiter(tmp_path / "usage.json", _Logger(), today=lambda: current[0])
    assert limiter.acquire("10001", 1).allowed is True
    assert limiter.acquire("10001", 1).allowed is False

    current[0] = date(2026, 8, 15)
    assert limiter.acquire("10001", 1).allowed is True


def test_whitelist_and_zero_limit_do_not_write_usage(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    limiter = DailyUsageLimiter(path, _Logger(), today=lambda: date(2026, 8, 14))

    assert limiter.acquire("white", 1, whitelisted=True).allowed is True
    assert limiter.acquire("unlimited", 0).allowed is True
    assert not path.exists()


def test_concurrent_acquire_never_exceeds_limit(tmp_path: Path) -> None:
    limiter = DailyUsageLimiter(
        tmp_path / "usage.json", _Logger(), today=lambda: date(2026, 8, 14)
    )
    with ThreadPoolExecutor(max_workers=12) as executor:
        decisions = list(executor.map(lambda _index: limiter.acquire("10001", 5), range(30)))

    assert sum(decision.allowed for decision in decisions) == 5
    assert json.loads((tmp_path / "usage.json").read_text(encoding="utf-8"))["counts"]["10001"] == 5
