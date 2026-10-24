from __future__ import annotations

import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from datetime import date
from pathlib import Path
from types import SimpleNamespace

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from command_actions import CommandActionHandler
from command_router import help_text, parse_hard_route
from comfyui_runtime import ComfyUIRuntime
from main import ComfyUIAgentPlugin
from usage_limiter import DailyUsageLimiter


class _Logger:
    def warning(self, *_args):
        pass


def _limiter(tmp_path, **kwargs):
    return DailyUsageLimiter(
        tmp_path / "usage.json",
        _Logger(),
        today=lambda: date(2026, 10, 22),
        **kwargs,
    )


def test_shared_limit_counts_whitelist_and_unlimited_users(tmp_path):
    limiter = _limiter(tmp_path)
    assert limiter.acquire("white", 1, whitelisted=True, total_limit=3).allowed
    assert limiter.acquire("white", 1, whitelisted=True, total_limit=3).allowed
    assert limiter.acquire("unlimited", 0, total_limit=3).allowed
    for sender, white in [("white", True), ("new_user", False)]:
        result = limiter.acquire(sender, 0, whitelisted=white, total_limit=3)
        assert not result.allowed
        assert result.rejection == "total"
        assert result.total_used == 3
        assert result.total_remaining == 0
    assert limiter.snapshot()["counts"] == {"white": 2, "unlimited": 1}
    restarted = _limiter(tmp_path)
    assert not restarted.acquire("new_user", 0, total_limit=3).allowed
    assert restarted.acquire("new_user", 0, total_limit=4).allowed


def test_personal_rejection_does_not_consume_shared_quota(tmp_path):
    limiter = _limiter(tmp_path)
    assert limiter.acquire("one", 1, total_limit=3).allowed
    rejected = limiter.acquire("one", 1, total_limit=3)
    assert rejected.rejection == "personal"
    assert rejected.total_used == 1
    assert limiter.acquire("two", 1, total_limit=3).allowed
    assert limiter.snapshot()["total_used"] == 2


def test_completion_notice_includes_shared_remaining_for_whitelist():
    text = ComfyUIRuntime._remaining_usage_text(
        {
            "daily_usage": {
                "limit": 1,
                "remaining": 0,
                "whitelisted": True,
                "total_limit": 3,
                "total_remaining": 1,
            }
        }
    )
    assert text == "今日剩余次数：不限\n全体今日剩余次数：1"


def test_concurrent_different_users_share_one_atomic_limit(tmp_path):
    limiter = _limiter(tmp_path)
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(
            pool.map(lambda i: limiter.acquire(str(i), 0, total_limit=7), range(40))
        )
    assert sum(result.allowed for result in results) == 7
    assert limiter.snapshot()["total_used"] == 7
    assert max(result.total_used for result in results) == 7


def test_daily_shared_count_rolls_over_and_queries_do_not_charge(tmp_path):
    current = [date(2026, 10, 22)]
    limiter = DailyUsageLimiter(
        tmp_path / "usage.json", _Logger(), today=lambda: current[0]
    )
    assert limiter.acquire("white", 0, total_limit=1).allowed
    assert limiter.snapshot()["total_used"] == 1
    snapshot = limiter.snapshot()
    snapshot["counts"]["white"] = 999
    assert limiter.snapshot()["total_used"] == 1
    current[0] = date(2026, 10, 23)
    assert limiter.snapshot()["total_used"] == 0
    assert limiter.acquire("white", 0, total_limit=1).allowed


def test_upgrade_recovers_today_once_without_duplicate_retries(tmp_path):
    tasks = tmp_path / "tasks"
    tasks.mkdir()

    def write(task_id, **fields):
        task = {
            "task_id": task_id,
            "action": "generate",
            "sender_id": "one",
            "time": "2026-10-22T08:00:00",
            **fields,
        }
        (tasks / f"{task_id}.json").write_text(json.dumps(task), encoding="utf-8")

    write(
        "original",
        verification_summary={
            "attempts": [{"task_id": "original"}, {"task_id": "retry"}]
        },
    )
    write("retry")
    write("prepared", prompt_summary={"multi_person_prepared_retry": True})
    write("legacy_retry", original_prompt="scene\n【上次问题，请修正】fix")
    write("failed", sender_id="white", ok=False)
    write("yesterday", time="2026-10-21T08:00:00")
    write("edit", action="edit")
    (tasks / "broken.json").write_text("{", encoding="utf-8")
    (tmp_path / "usage.json").write_text(
        json.dumps({"date": "2026-10-22", "counts": {"one": 3, "bad": "invalid"}}),
        encoding="utf-8",
    )
    limiter = _limiter(tmp_path, tasks_dir=tasks)
    assert limiter.snapshot()["counts"] == {"one": 3, "white": 1}
    assert limiter.snapshot()["total_used"] == 4
    assert limiter.acquire("white", 0).total_used == 5
    # A later task file must not be re-imported after the one-time upgrade.
    write("later", sender_id="white")
    assert _limiter(tmp_path, tasks_dir=tasks).snapshot()["total_used"] == 5


def test_upgrade_recovers_when_old_ledger_is_from_previous_day(tmp_path):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "task.json").write_text(
        json.dumps(
            {
                "task_id": "task",
                "action": "generate",
                "sender_id": "white",
                "time": "2026-10-22T12:00:00",
                "ok": True,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "usage.json").write_text(
        json.dumps({"date": "2026-10-21", "counts": {"old": 100}}), encoding="utf-8"
    )
    assert _limiter(tmp_path, tasks_dir=tasks).snapshot()["counts"] == {"white": 1}


def _plugin(tmp_path):
    plugin = ComfyUIAgentPlugin.__new__(ComfyUIAgentPlugin)
    plugin.config = {
        "daily_generation_limit": 1,
        "daily_total_generation_limit": 2,
        "generation_whitelist_sender_ids": ["white"],
    }
    plugin._daily_usage = _limiter(tmp_path)
    plugin._generation_usage_summary = ContextVar("daily_usage_test", default={})
    return plugin


def _event(sender="white"):
    return SimpleNamespace(get_sender_id=lambda: sender, is_admin=lambda: False)


def test_plugin_shared_limit_rejection_happens_before_generation(tmp_path):
    plugin = _plugin(tmp_path)
    event = _event()
    assert plugin._reserve_generation_call(event) == ""
    assert plugin._reserve_generation_call(event) == ""
    assert plugin._generation_usage_summary.get()["total_used"] == 2
    messages = []
    event.plain_result = lambda text: text

    async def send(text):
        messages.append(text)

    event.send = send
    # No runtime/generation services are installed: rejection must return first.
    assert "总生图次数已达上限（2/2）" in asyncio.run(plugin._generate(event, "scene"))
    assert len(messages) == 1
    assert plugin._daily_usage.snapshot()["total_used"] == 2


def test_stats_routing_query_and_permissions_without_generation(tmp_path):
    plugin = _plugin(tmp_path)
    assert plugin._reserve_generation_call(_event()) == ""
    handler = CommandActionHandler.__new__(CommandActionHandler)
    handler._is_allowed = lambda event: True
    handler._generation_usage_text = plugin._generation_usage_text
    plugin._action_handler = handler
    for command in [
        "/anm 生图统计",
        "/anm 今日生图",
        "/anima stats",
        "/comfyui 生图次数",
    ]:
        route = parse_hard_route(command)
        assert route == ("generation_stats", "")
        result = asyncio.run(plugin._handle_action(_event(), *route))
        assert "今日总生图次数：1/2" in result
        assert "你的今日次数：1（不限额）" in result
        assert "2026-10-22" in result
    assert plugin._daily_usage.snapshot()["total_used"] == 1
    assert parse_hard_route("/anm 生图统计图表")[0] == "generate"
    assert "生图统计" in help_text()
    handler._is_allowed = lambda event: False
    assert "没有使用权限" in asyncio.run(
        plugin._handle_action(_event(), "generation_stats", "")
    )
