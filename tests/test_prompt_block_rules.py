from __future__ import annotations

import asyncio
from pathlib import Path
import sys

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from main import ComfyUIAgentPlugin  # noqa: E402
from prompt_block_rules import matches_blocked_combination  # noqa: E402


def test_combination_requires_all_terms_in_any_order() -> None:
    rules = ["黑丝 && 雨天", "Blue Hair && sword"]
    assert matches_blocked_combination("雨天里穿黑丝", rules)
    assert matches_blocked_combination("sword, BLUE HAIR", rules)
    assert not matches_blocked_combination("只有黑丝", rules)
    assert not matches_blocked_combination("sword only", rules)


def test_empty_or_malformed_rules_do_not_block() -> None:
    assert not matches_blocked_combination("任何内容", [])
    assert not matches_blocked_combination("任何内容", ["任何内容", "&& 任何内容"])
    assert matches_blocked_combination("原样 画风 # 禁词B, 禁词A", ["禁词A && 禁词B"])


def test_blocked_generation_returns_before_quota_and_submission() -> None:
    class Event:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def plain_result(self, text: str) -> str:
            return text

        async def send(self, result: str) -> None:
            self.messages.append(result)

    plugin = ComfyUIAgentPlugin.__new__(ComfyUIAgentPlugin)
    plugin.config = {"blocked_prompt_combinations": ["禁词A && 禁词B"]}
    plugin._reserve_generation_call = lambda _event: (_ for _ in ()).throw(
        AssertionError("quota must not be consumed")
    )
    plugin._generate_payload = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("generation must not start")
    )
    event = Event()

    result = asyncio.run(plugin._generate(event, "无优化 禁词B # 禁词A", multi_person=True))

    assert result == "请求包含禁止的词语组合，已拒绝生成。"
    assert event.messages == [result]
