from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from generation_request_guard import GenerationRequestGuard  # noqa: E402
from main import ComfyUIAgentPlugin  # noqa: E402


class _Event:
    def __init__(self, message_id: str, session: str = "private:1"):
        self.message_obj = SimpleNamespace(message_id=message_id)
        self.unified_msg_origin = session
        self.messages: list[str] = []

    def get_platform_id(self) -> str:
        return "cq:bot"

    def get_sender_id(self) -> str:
        return "user"

    def plain_result(self, value: str) -> str:
        return value

    async def send(self, value: str) -> None:
        self.messages.append(value)


def test_message_identity_controls_deduplication() -> None:
    guard = GenerationRequestGuard()
    assert guard.claim(_Event("100"))
    assert not guard.claim(_Event("100"))
    assert guard.claim(_Event("101"))  # Same prompt may be sent again intentionally.
    assert guard.claim(_Event("100", session="private:2"))
    assert guard.claim(_Event(""))
    assert guard.claim(_Event(""))


def test_concurrent_duplicate_claims_have_one_winner() -> None:
    guard = GenerationRequestGuard()
    results: list[bool] = []
    lock = threading.Lock()

    def claim() -> None:
        result = guard.claim(_Event("100"))
        with lock:
            results.append(result)

    threads = [threading.Thread(target=claim) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count(True) == 1


def test_duplicate_dispatch_starts_one_generation_and_reserves_one_call() -> None:
    async def run() -> None:
        plugin = ComfyUIAgentPlugin.__new__(ComfyUIAgentPlugin)
        plugin.config = {}
        plugin._generation_request_guard = GenerationRequestGuard()
        plugin._generation_queue_lock = threading.Lock()
        plugin._unfinished_generation_requests = 0
        reserved: list[str] = []
        generated: list[str] = []
        started = asyncio.Event()
        release = asyncio.Event()

        def reserve(event: _Event) -> str:
            reserved.append(event.message_obj.message_id)
            return ""

        async def generate(_event: _Event, prompt: str, **_kwargs):
            generated.append(prompt)
            started.set()
            await release.wait()
            return {"ok": True, "task_id": prompt, "delivery": {}}

        plugin._reserve_generation_call = reserve
        plugin._generate_payload = generate
        plugin._send_payload = lambda *_args: asyncio.sleep(0, result="sent")
        plugin._generation_task = SimpleNamespace(record_delivery=lambda *_args: None)
        first = asyncio.create_task(plugin._generate(_Event("100"), "scene"))
        await started.wait()
        duplicate = await plugin._generate(_Event("100"), "scene")
        release.set()
        assert duplicate == ""
        assert await first == "sent"
        assert reserved == ["100"]
        assert generated == ["scene"]

        assert await plugin._generate(_Event("101"), "scene") == "sent"
        assert generated == ["scene", "scene"]

    asyncio.run(run())
