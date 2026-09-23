from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from danbooru_resolver import DanbooruResolver
from danbooru_semantic import SemanticAnchor, SemanticLookupResult
from prompt_pipeline import (
    PromptPipeline, StructuredPromptCharacter, _parse_structured_prompt,
    bind_confirmed_character_anchors, confirmed_semantic_character_tags,
    reconcile_confirmed_semantic_characters, validate_confirmed_character_roster,
)


CANONICAL = "togawa_sakiko_(master_of_melodia)"
ROSTER = ("chihaya_anon", "togawa_sakiko", CANONICAL)


def writer(names):
    return (
        "{Count: 3girls}\n{Characters: " + ", ".join(names) + "}\n"
        "{Copyright: bang_dream!}\n"
        "{Identity: " + "; ".join(f"{name} has blue eyes" for name in names) + "}\n"
        "{Details: " + "; ".join(f"{name} smiles" for name in names) + "}\n"
        "{Tags: standing, simple background}\n"
        "{Nltags: " + " stands beside ".join(names) + "}"
    )


def test_cached_owner_binds_without_lookup_but_donor_does_not():
    anchors = (
        SemanticAnchor("target", "target_character", "character", "粥祥", "", (CANONICAL,)),
        SemanticAnchor("donor", "outfit_source", "character", "衣服来源", "", ("donor",)),
        SemanticAnchor("unknown", "target_character", "character", "未知", "", ("guess",)),
    )
    result = bind_confirmed_character_anchors(
        SemanticLookupResult(confirmed_tags=(CANONICAL, "donor"), missing_descriptions=("粥祥", "未知")), anchors
    )
    assert result.anchor_tags == (("target", CANONICAL),)
    assert confirmed_semantic_character_tags(result) == (CANONICAL,)
    assert result.missing_descriptions == ("未知",)


@pytest.mark.parametrize("alias", ["zhouxiang", "zhou_xiang", "togawa_saiko"])
@pytest.mark.parametrize("first", [False, True])
def test_unique_remaining_owner_is_repaired_without_order_assumption(alias, first):
    names = (alias, *ROSTER[:2]) if first else (*ROSTER[:2], alias)
    parsed = validate_confirmed_character_roster(_parse_structured_prompt(writer(names)), ROSTER)
    assert not parsed.validation_errors
    assert {item.name for item in parsed.characters} == set(ROSTER)
    assert alias not in parsed.nltags
    assert CANONICAL in parsed.nltags
    assert all(alias not in item.identity_tags + item.detail_tags for item in parsed.characters)


def test_ambiguous_roster_requires_rewrite_and_anonymous_roster_remains_valid():
    parsed = validate_confirmed_character_roster(
        _parse_structured_prompt(writer(("unknown_a", "togawa_sakiko", "unknown_b"))), ROSTER
    )
    assert any("missing confirmed visible identities" in error for error in parsed.validation_errors)
    anonymous = _parse_structured_prompt(
        "{Count: 1girl}{Characters: }{Copyright: }{Identity: }{Details: }{Tags: standing}{Nltags: }"
    )
    assert not validate_confirmed_character_roster(anonymous, ()).validation_errors


def test_replacements_do_not_rewrite_base_inside_qualified_name():
    names = (CANONICAL, "young_togawa_sakiko", "chihaya_anon")
    characters = tuple(StructuredPromptCharacter(name, name + " smiles", "") for name in names)
    result, text = reconcile_confirmed_semantic_characters(characters, ", ".join(names), ROSTER)
    assert [item.name for item in result] == [CANONICAL, "togawa_sakiko", "chihaya_anon"]
    assert text.count("master_of_melodia") == 1
    assert "young_" not in text


@pytest.mark.parametrize("retry", [False, True])
def test_pipeline_cached_roster_repair_and_retry(tmp_path, retry):
    class Logger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    class Context:
        def __init__(self):
            plan = {"characters": [
                {"name": name, "aliases": [], "clothing": None, "clothing_source": None,
                 "clothing_changes": [], "appearance_changes": []}
                for name in ("千早爱音", "丰川祥子", "粥祥")
            ]}
            self.outputs = [json.dumps(plan, ensure_ascii=False), writer(
                ("unknown_a", "togawa_sakiko", "unknown_b") if retry
                else ("chihaya_anon", "togawa_sakiko", "zhouxiang")
            )]
            if retry:
                self.outputs.append(writer(ROSTER))
            self.calls = []

        async def get_current_chat_provider_id(self, _umo):
            return "provider"

        async def llm_generate(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(completion_text=self.outputs.pop(0))

    config = {"danbooru_semantic_lookup_enabled": False}
    accessors = {key: lambda name, default: config.get(name, default)
                 for key in ("get_bool", "get_int", "get_float", "get_str")}
    resolver = DanbooruResolver(config=config, logger=Logger(), cache={}, **accessors,
                               profile_cache_path=tmp_path / "profiles.json")
    resolver._profile_cache_data = {"version": 3, "profiles": {
        alias: {"kind": "character_outfit", "qualifier": "default", "aliases": [alias],
                "source_tags": [tag], "owner_tags": [tag], "outfit_tags": ["black_dress"],
                "evidence": {"source_category": 4}}
        for alias, tag in zip(("千早爱音", "丰川祥子", "粥祥"), ROSTER)
    }}
    context = Context()
    researcher = SimpleNamespace(plan=lambda prompt: SimpleNamespace(
        use_web_search=False, use_deep_thinking=False, search_reason="", thinking_reason=""
    ))
    pipeline = PromptPipeline(context=context, config=config, logger=Logger(),
                              danbooru_resolver=resolver, researcher=researcher,
                              **accessors, shorten=lambda value, limit=600: value[:limit])
    result = asyncio.run(pipeline.build(SimpleNamespace(unified_msg_origin="session"),
                                       "千早爱音、丰川祥子和粥祥站在一起"))
    assert result.final_prompt
    assert "zhouxiang" not in result.final_prompt
    assert r"togawa sakiko \(master of melodia\)" in result.final_prompt
    assert all(item["status"] == "semantic_confirmed" for item in result.summary["character_resolution_statuses"])
    assert f"粥祥 => {CANONICAL}" in context.calls[1]["prompt"]
    assert result.summary["prompt_llm_accepted_attempt"] == ("retry" if retry else "initial")
