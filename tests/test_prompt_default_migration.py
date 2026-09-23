from __future__ import annotations

import json
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from config_defaults import migrate_prompt_defaults  # noqa: E402
from danbooru_semantic import DEFAULT_SEMANTIC_PLAN_SYSTEM_PROMPT  # noqa: E402
from prompt_templates import DEFAULT_LLM_PROMPT_TEMPLATE  # noqa: E402


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
