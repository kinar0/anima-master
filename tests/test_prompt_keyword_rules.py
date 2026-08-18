from __future__ import annotations

import sys
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from prompt_keyword_rules import match_keyword_prompt_rules  # noqa: E402
from prompt_templates import build_llm_prompt  # noqa: E402


def test_keyword_rule_matches_any_marker_case_insensitively() -> None:
    config = {
        "keyword_prompt_rules_enabled": True,
        "keyword_prompt_rules": [
            "黑丝|丝袜 => 必须添加 black pantyhose",
            "RAIN|下雨 => 必须添加 rain",
        ],
    }

    rules = match_keyword_prompt_rules("画一个丝袜少女 in rain", config)

    assert [(rule.marker, rule.instruction) for rule in rules] == [
        ("丝袜", "必须添加 black pantyhose"),
        ("RAIN", "必须添加 rain"),
    ]


def test_disabled_and_malformed_keyword_rules_are_ignored() -> None:
    assert match_keyword_prompt_rules(
        "黑丝",
        {
            "keyword_prompt_rules_enabled": False,
            "keyword_prompt_rules": ["黑丝 => add black pantyhose"],
        },
    ) == ()
    assert match_keyword_prompt_rules(
        "黑丝", {"keyword_prompt_rules": ["缺少分隔符", "=> 缺少触发词"]}
    ) == ()


def test_matched_rules_are_appended_to_custom_llm_template_forcefully() -> None:
    rules = match_keyword_prompt_rules(
        "画黑丝少女",
        {"keyword_prompt_rules": ["黑丝|丝袜 => 在 Tags 中加入 black pantyhose"]},
    )

    prompt = build_llm_prompt(
        "画黑丝少女",
        mode="img2img",
        prompt_builder_template="自定义模板：{theme}",
        keyword_prompt_rules=rules,
    )

    assert "自定义模板：画黑丝少女" in prompt
    assert "关键词强制规则（最高优先级，必须逐条执行）" in prompt
    assert "命中触发词“黑丝”" in prompt
    assert "在 Tags 中加入 black pantyhose" in prompt
    assert "硬性输出要求，不是建议" in prompt
