from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from main import ComfyUIAgentPlugin  # noqa: E402
from prompt_block_rules import (  # noqa: E402
    blocked_prompt_message,
    find_blocked_prompt_terms,
    matches_blocked_combination,
    matches_blocked_prompt,
)


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


def test_any_a_term_with_any_b_term_blocks() -> None:
    config = {
        "blocked_prompt_rules": [
            {
                "__template_key": "group_pair",
                "group_a": ["  黑丝  ", "Blue Hair"],
                "group_b": ["雨天", "SWORD"],
            },
            {
                "__template_key": "group_pair",
                "group_a": ["盔甲", "披风"],
                "group_b": ["火山", "雪地"],
            },
        ],
    }
    assert matches_blocked_prompt("雨天里穿黑丝", config)
    assert matches_blocked_prompt("原样 blue hair # sword", config)
    assert matches_blocked_prompt("雪地上的披风", config)
    assert not matches_blocked_prompt("只有黑丝和蓝发", config)
    assert not matches_blocked_prompt("雨天握剑", config)
    assert not matches_blocked_prompt("黑丝走在雪地", config)


def test_empty_group_entries_cannot_match_every_request() -> None:
    config = {
        "blocked_prompt_rules": [
            {"group_a": ["", "   "], "group_b": ["雨天"]}
        ],
    }
    assert not matches_blocked_prompt("雨天", config)
    config["blocked_prompt_rules"][0]["group_a"] = ["黑丝"]
    config["blocked_prompt_rules"][0]["group_b"] = []
    assert not matches_blocked_prompt("黑丝", config)


def test_legacy_combinations_remain_active() -> None:
    assert matches_blocked_prompt(
        "甲和乙", {"blocked_prompt_combinations": ["甲 && 乙"]}
    )


def test_details_include_all_triggered_rules_but_no_unpaired_terms() -> None:
    config = {
        "blocked_prompt_rules": [
            {"group_a": ["黑丝", "Blue Hair", "未出现"], "group_b": ["雨天"]},
            {"group_a": ["披风"], "group_b": ["雪地"]},
            {"group_a": ["独立词"], "group_b": ["缺失词"]},
        ],
        "blocked_prompt_combinations": ["黑丝 && 披风", "独立词 && 未出现"],
    }
    assert find_blocked_prompt_terms(
        "雨天，黑丝，BLUE HAIR，雪地，披风，独立词", config
    ) == ["黑丝", "Blue Hair", "雨天", "披风", "雪地"]
    assert blocked_prompt_message("黑丝和雪地，独立词", config) == ""


def test_details_keep_config_spelling_and_deduplicate_compatible_rules() -> None:
    config = {
        "blocked_prompt_rules": [
            {"group_a": [" Ｂｌｕｅ Hair ", "blue hair"], "group_b": ["SWORD"]},
        ],
        "blocked_prompt_group_a": ["Blue Hair"],
        "blocked_prompt_group_b": ["Sword", "盾"],
        "blocked_prompt_combinations": ["blue hair && sword && 盾", "盾 && 缺失"],
    }
    assert find_blocked_prompt_terms("blue hair # sword, 盾", config) == [
        "Ｂｌｕｅ Hair", "SWORD", "盾"
    ]
    assert find_blocked_prompt_terms(
        "乙和甲", {"blocked_prompt_combinations": ["甲 && 乙"]}
    ) == ["甲", "乙"]
    assert find_blocked_prompt_terms(
        "乙和甲", {"blocked_prompt_group_a": ["甲"], "blocked_prompt_group_b": ["乙"]}
    ) == ["甲", "乙"]


def test_invalid_rules_do_not_appear_in_details() -> None:
    config = {
        "blocked_prompt_rules": [
            None,
            {"__template_key": "unknown", "group_a": ["甲"], "group_b": ["乙"]},
            {"group_a": [None, " "], "group_b": ["乙"]},
            {"group_a": "甲", "group_b": ["乙"]},
        ],
        "blocked_prompt_combinations": [None, "甲", "甲 &&", "&& 乙"],
    }
    assert blocked_prompt_message("甲和乙", config) == ""
    assert blocked_prompt_message("甲和乙", None) == ""


@pytest.mark.parametrize("multi_person", [False, True])
def test_blocked_generation_returns_before_quota_and_submission(multi_person: bool) -> None:
    class Event:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def plain_result(self, text: str) -> str:
            return text

        async def send(self, result: str) -> None:
            self.messages.append(result)

    plugin = ComfyUIAgentPlugin.__new__(ComfyUIAgentPlugin)
    plugin.config = {
        "blocked_prompt_rules": [
            {
                "__template_key": "group_pair",
                "group_a": ["禁词A", "其他词"],
                "group_b": ["禁词B", "第二词"],
            }
        ],
    }
    plugin._reserve_generation_call = lambda _event: (_ for _ in ()).throw(
        AssertionError("quota must not be consumed")
    )
    plugin._generate_payload = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("generation must not start")
    )
    event = Event()

    result = asyncio.run(
        plugin._generate(event, "无优化 禁词B # 禁词A", multi_person=multi_person)
    )

    assert result == (
        "请求包含禁止的词语组合，已拒绝生成。\n"
        "本次命中词语：「禁词A」、「禁词B」。\n"
        "请修改后再试。"
    )
    assert event.messages == [result]
