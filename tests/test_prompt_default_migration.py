from __future__ import annotations

import json
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from config_defaults import migrate_block_rule_groups, migrate_prompt_defaults  # noqa: E402
from danbooru_semantic import DEFAULT_SEMANTIC_PLAN_SYSTEM_PROMPT  # noqa: E402
from prompt_templates import DEFAULT_LLM_PROMPT_TEMPLATE, build_llm_prompt  # noqa: E402


def test_fixed_block_groups_migrate_to_repeatable_rule_once() -> None:
    config = {
        "blocked_prompt_group_a": ["甲", "乙"],
        "blocked_prompt_group_b": ["丙"],
        "blocked_prompt_rules": [
            {"__template_key": "group_pair", "group_a": ["丁"], "group_b": ["戊"]}
        ],
    }
    migrated = migrate_block_rule_groups(config)
    assert "blocked_prompt_group_a" not in migrated
    assert "blocked_prompt_group_b" not in migrated
    assert migrated["blocked_prompt_rules"] == [
        {"__template_key": "group_pair", "group_a": ["丁"], "group_b": ["戊"]},
        {"__template_key": "group_pair", "group_a": ["甲", "乙"], "group_b": ["丙"]},
    ]
    assert migrate_block_rule_groups(migrated) == migrated


def test_empty_builtin_prompt_fields_migrate_to_aesthetic_defaults() -> None:
    migrated = migrate_prompt_defaults(
        {
            "prompt_builder_template": "",
            "prompt_builder_max_content_tags": 80,
        }
    )

    assert migrated["prompt_builder_template"] == DEFAULT_LLM_PROMPT_TEMPLATE
    assert migrated["prompt_builder_max_content_tags"] == 65


def test_schema_builtin_prompt_migrates_to_current_default() -> None:
    schema = json.loads((PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))
    schema_default = schema["anima_master_prompting"]["items"][
        "prompt_builder_template"
    ]["default"]

    migrated = migrate_prompt_defaults({"prompt_builder_template": schema_default})

    assert migrated["prompt_builder_template"] == DEFAULT_LLM_PROMPT_TEMPLATE
    assert schema_default == DEFAULT_LLM_PROMPT_TEMPLATE


def test_stored_viewer_guidance_builtin_migrates_to_current_default() -> None:
    previous = DEFAULT_LLM_PROMPT_TEMPLATE.replace(
        "除此之外可以自由决定姿态、构图、镜头、光影、色彩、氛围和特效。",
        "除此之外可以自由决定姿态、构图、镜头、光影、色彩、氛围和特效。"
        "视线方向只根据用户明确要求或可靠参考证据描述；"
        "不要把看向观众当作默认姿态，人物互动时保留他们之间的视线关系。",
    )
    migrated = migrate_prompt_defaults({"prompt_builder_template": previous})

    assert migrated["prompt_builder_template"] == DEFAULT_LLM_PROMPT_TEMPLATE
    assert "看向观众" not in build_llm_prompt(
        "女孩在读书", prompt_builder_template=previous
    )


def test_builtin_writing_instructions_omit_viewer_gaze_guidance() -> None:
    for mode in ("txt2img", "img2img"):
        for sensual_mode in (False, True):
            prompt = build_llm_prompt(
                "女孩在读书", mode=mode, sensual_mode=sensual_mode
            )
            for phrase in ("looking at viewer", "looking_at_viewer", "看向观众"):
                assert phrase not in prompt

    # This is instruction cleanup, not a ban on explicitly requested content.
    requested = "1girl, looking at viewer"
    assert requested in build_llm_prompt(requested)


def test_stored_legacy_semantic_prompt_migrates_to_intent_only_default() -> None:
    migrated = migrate_prompt_defaults(
        {
            "danbooru_semantic_system_prompt": (
                "You extract semantic lookup anchors for a local Danbooru index. "
                "Return valid JSON only. Never claim that a candidate is verified."
            ),
            "prompt_builder_template": "我的自定义模板：{theme}",
        }
    )

    assert (
        migrated["danbooru_semantic_system_prompt"]
        == DEFAULT_SEMANTIC_PLAN_SYSTEM_PROMPT
    )


def test_custom_prompt_template_and_limit_are_preserved() -> None:
    migrated = migrate_prompt_defaults(
        {
            "prompt_builder_template": "我的自定义模板：{theme}",
            "prompt_builder_max_content_tags": 72,
        }
    )

    assert migrated["prompt_builder_template"] == "我的自定义模板：{theme}"
    assert migrated["prompt_builder_max_content_tags"] == 72


def test_duplicate_remote_lookup_switches_migrate_to_effective_state() -> None:
    cases = (
        (True, False, False),
        (False, True, False),
        (True, True, True),
    )
    for old_core, old_remote, expected in cases:
        migrated = migrate_prompt_defaults(
            {
                "prompt_builder_template": "custom template",
                "danbooru_core_tag_lookup_enabled": old_core,
                "danbooru_remote_lookup_enabled": old_remote,
            }
        )

        assert migrated["danbooru_core_tag_lookup_enabled"] is expected
        assert "danbooru_remote_lookup_enabled" not in migrated
