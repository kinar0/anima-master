from __future__ import annotations

import asyncio
from contextvars import ContextVar
from datetime import date
from pathlib import Path
import sys
import threading

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from main import ComfyUIAgentPlugin  # noqa: E402
from usage_limiter import DailyUsageLimiter  # noqa: E402


def _init_generation_queue(plugin: ComfyUIAgentPlugin) -> None:
    plugin._generation_queue_lock = threading.Lock()
    plugin._unfinished_generation_requests = 0


def test_generate_notifies_user_when_prompt_optimization_degrades() -> None:
    class _Event:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def plain_result(self, text: str) -> str:
            return text

        async def send(self, result: str) -> None:
            self.messages.append(result)

    class _Task:
        def record_delivery(self, task_id, delivery) -> None:
            self.recorded = (task_id, delivery)

    plugin = ComfyUIAgentPlugin.__new__(ComfyUIAgentPlugin)
    payload = {
        "ok": True,
        "task_id": "task",
        "prompt_degraded": True,
        "delivery": {"status": "sent"},
    }

    async def generate_payload(*_args, **_kwargs):
        return payload

    async def send_payload(_event, _payload):
        return "sent"

    plugin._generate_payload = generate_payload
    plugin._send_payload = send_payload
    plugin._generation_task = _Task()
    plugin._reserve_generation_call = lambda _event: ""
    plugin.config = {}
    _init_generation_queue(plugin)
    event = _Event()

    result = asyncio.run(plugin._generate(event, "画一个女孩"))

    assert result == "sent"
    assert len(event.messages) == 1
    assert "提示词优化服务不可用" in event.messages[0]


def test_generate_sends_daily_limit_rejection_without_starting_generation() -> None:
    class _Event:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def plain_result(self, text: str) -> str:
            return text

        async def send(self, result: str) -> None:
            self.messages.append(result)

    plugin = ComfyUIAgentPlugin.__new__(ComfyUIAgentPlugin)
    plugin._reserve_generation_call = lambda _event: "你今天的生图次数已用完（3/3）。请明天再试。"
    plugin.config = {"notify_drawing_and_at_sender": True}
    _init_generation_queue(plugin)

    async def should_not_generate(*_args, **_kwargs):
        raise AssertionError("quota rejection must happen before generation")

    plugin._generate_payload = should_not_generate
    event = _Event()

    result = asyncio.run(plugin._generate(event, "画一个女孩"))

    assert result == event.messages[0]
    assert "3/3" in result
    assert "正在绘画中" not in result


def test_generate_sends_progress_notice_after_quota_acceptance() -> None:
    class _Event:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def plain_result(self, text: str) -> str:
            return text

        async def send(self, result: str) -> None:
            self.messages.append(result)

    plugin = ComfyUIAgentPlugin.__new__(ComfyUIAgentPlugin)
    plugin.config = {"notify_drawing_and_at_sender": True}
    plugin._reserve_generation_call = lambda _event: ""
    plugin._generate_payload = lambda *_args, **_kwargs: asyncio.sleep(
        0, result={"ok": True, "task_id": "task", "delivery": {}}
    )
    plugin._send_payload = lambda *_args, **_kwargs: asyncio.sleep(0, result="sent")
    plugin._generation_task = type(
        "_Task", (), {"record_delivery": lambda self, *_args: None}
    )()
    _init_generation_queue(plugin)
    event = _Event()

    result = asyncio.run(plugin._generate(event, "画一个女孩"))

    assert result == "sent"
    assert event.messages == ["正在绘画中"]


def test_generate_reports_unfinished_requests_ahead_and_cleans_up() -> None:
    class _Event:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def plain_result(self, text: str) -> str:
            return text

        async def send(self, result: str) -> None:
            self.messages.append(result)

    async def run() -> None:
        plugin = ComfyUIAgentPlugin.__new__(ComfyUIAgentPlugin)
        plugin.config = {"notify_drawing_and_at_sender": True}
        plugin._reserve_generation_call = lambda _event: ""
        plugin._generation_task = type(
            "_Task", (), {"record_delivery": lambda self, *_args: None}
        )()
        _init_generation_queue(plugin)

        first_started = asyncio.Event()
        release_first = asyncio.Event()

        async def generate_payload(_event, prompt, **_kwargs):
            if prompt == "first":
                first_started.set()
                await release_first.wait()
            return {"ok": True, "task_id": prompt, "delivery": {}}

        plugin._generate_payload = generate_payload
        plugin._send_payload = lambda *_args, **_kwargs: asyncio.sleep(
            0, result="sent"
        )
        first_event = _Event()
        second_event = _Event()

        first = asyncio.create_task(plugin._generate(first_event, "first"))
        await first_started.wait()
        second = asyncio.create_task(plugin._generate(second_event, "second"))
        await asyncio.sleep(0)

        assert first_event.messages == ["正在绘画中"]
        assert second_event.messages == ["正在绘画中，前面还有1人"]
        assert plugin._unfinished_generation_requests == 2

        release_first.set()
        assert await asyncio.gather(first, second) == ["sent", "sent"]
        assert plugin._unfinished_generation_requests == 0

    asyncio.run(run())


def test_generation_queue_count_is_released_after_exception() -> None:
    class _Event:
        def plain_result(self, text: str) -> str:
            return text

        async def send(self, _result: str) -> None:
            return None

    async def run() -> None:
        plugin = ComfyUIAgentPlugin.__new__(ComfyUIAgentPlugin)
        plugin.config = {"notify_drawing_and_at_sender": False}
        plugin._reserve_generation_call = lambda _event: ""
        _init_generation_queue(plugin)

        async def fail(*_args, **_kwargs):
            raise RuntimeError("generation failed")

        plugin._generate_payload = fail

        try:
            await plugin._generate(_Event(), "fail")
        except RuntimeError as exc:
            assert str(exc) == "generation failed"
        else:
            raise AssertionError("generation exception should propagate")

        assert plugin._unfinished_generation_requests == 0

    asyncio.run(run())


def test_generation_queue_count_is_released_after_cancellation() -> None:
    class _Event:
        def plain_result(self, text: str) -> str:
            return text

        async def send(self, _result: str) -> None:
            return None

    async def run() -> None:
        plugin = ComfyUIAgentPlugin.__new__(ComfyUIAgentPlugin)
        plugin.config = {"notify_drawing_and_at_sender": False}
        plugin._reserve_generation_call = lambda _event: ""
        _init_generation_queue(plugin)
        started = asyncio.Event()

        async def wait_forever(*_args, **_kwargs):
            started.set()
            await asyncio.Event().wait()

        plugin._generate_payload = wait_forever
        task = asyncio.create_task(plugin._generate(_Event(), "cancelled"))
        await started.wait()
        assert plugin._unfinished_generation_requests == 1

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancelled generation should stay cancelled")

        assert plugin._unfinished_generation_requests == 0

    asyncio.run(run())


def test_blacklist_wins_and_whitelist_bypasses_legacy_allowlist(tmp_path: Path) -> None:
    class _Event:
        def __init__(self, sender_id: str) -> None:
            self.sender_id = sender_id

        def get_sender_id(self) -> str:
            return self.sender_id

        def is_admin(self) -> bool:
            return False

    plugin = ComfyUIAgentPlugin.__new__(ComfyUIAgentPlugin)
    plugin.config = {
        "allowed_sender_ids": ["legacy"],
        "generation_whitelist_sender_ids": ["white", "both"],
        "generation_blacklist_sender_ids": ["black", "both"],
    }

    assert plugin._is_allowed(_Event("legacy")) is True
    assert plugin._is_allowed(_Event("white")) is True
    assert plugin._is_allowed(_Event("ordinary")) is False
    assert plugin._is_allowed(_Event("black")) is False
    assert plugin._is_allowed(_Event("both")) is False


def test_generation_reservation_counts_normal_user_but_not_whitelist(
    tmp_path: Path,
) -> None:
    class _Event:
        def __init__(self, sender_id: str) -> None:
            self.sender_id = sender_id

        def get_sender_id(self) -> str:
            return self.sender_id

        def is_admin(self) -> bool:
            return False

    plugin = ComfyUIAgentPlugin.__new__(ComfyUIAgentPlugin)
    plugin.config = {
        "daily_generation_limit": 1,
        "generation_whitelist_sender_ids": ["white"],
        "generation_blacklist_sender_ids": [],
    }
    plugin._daily_usage = DailyUsageLimiter(
        tmp_path / "usage.json",
        type("_Logger", (), {"warning": lambda self, *_args: None})(),
        today=lambda: date(2026, 8, 14),
    )
    plugin._generation_usage_summary = ContextVar("test_generation_usage", default={})

    assert plugin._reserve_generation_call(_Event("ordinary")) == ""
    assert plugin._generation_usage_summary.get()["remaining"] == 0
    assert "1/1" in plugin._reserve_generation_call(_Event("ordinary"))
    assert plugin._reserve_generation_call(_Event("white")) == ""
    assert plugin._generation_usage_summary.get()["whitelisted"] is True
    assert plugin._reserve_generation_call(_Event("white")) == ""
