from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import danbooru_resolver as module
from danbooru_resolver import DanbooruResolver
from danbooru_semantic import (
    SemanticAnchor, SemanticCharacterPlan, SemanticLookupResult, SemanticWardrobe,
    bind_parenthesized_character_aliases, parse_semantic_plan,
    parse_semantic_character_aliases, parse_semantic_character_plans,
)
from danbooru_tags import TagRecord, VariantOutfitProfile
from prompt_pipeline import build_character_effective_outfits

SOURCE = "togawa_sakiko_(master_of_melodia)"

@pytest.fixture
def resolver(tmp_path):
    return DanbooruResolver(
        logger=logging.getLogger("test"), cache={}, profile_cache_path=tmp_path / "profiles.json",
        get_bool=lambda k, d: d, get_int=lambda k, d: d,
        get_float=lambda k, d: d, get_str=lambda k, d: d,
    )

def save_rows(resolver, rows):
    return resolver.save_wardrobe(dict(baseRevision=resolver.wardrobe_snapshot()["revision"],
                                    outfits=rows, outfitSets=[], terms=[]))

def row(owner, variant, tags, source=None):
    return dict(key=owner + " wardrobe", aliases=[variant], ownerTags=[owner],
                sourceTags=[source or owner], tags=tags, qualifier=variant)

@pytest.mark.parametrize("name", ["togawa sakiko (master of melodia)", SOURCE])
def test_complete_source_never_replaced_by_base_cache(resolver, name):
    resolver.remember_outfit_summary("togawa sakiko", ("togawa_sakiko",), ("school_uniform",))
    anchor = SemanticAnchor("s", "outfit_source", "character", name, name, ())
    assert resolver.prefer_cached_character_anchors((anchor,))[0].candidates == (SOURCE,)
    assert resolver.cached_outfit_profiles_for_prompt(name) is None


def test_split_latin_name_becomes_separate_costume_source(resolver):
    resolver.remember_outfit_summary("togawa sakiko", ("togawa_sakiko",), ("school_uniform",))
    prompt = "togawa sakiko (master of melodia)站立"
    raw = json.dumps(dict(characters=[dict(name="togawa sakiko", aliases=["master of melodia"],
                                        clothing=None, clothing_source=None)]))
    anchors = bind_parenthesized_character_aliases(parse_semantic_plan(raw, prompt), prompt,
                                                  parse_semantic_character_aliases(raw, prompt))
    anchors = resolver.prefer_cached_character_anchors(anchors)
    plans = parse_semantic_character_plans(raw, prompt, anchors)
    anchors = tuple(replace(a, source_category=0) for a in anchors)
    anchors, plans = resolver.bind_character_wardrobes(anchors, plans)
    target = next(a for a in anchors if a.role == "target_character")
    source = next(a for a in anchors if a.role == "outfit_source")
    assert target.candidates == ("togawa_sakiko",)
    assert source.candidates == (SOURCE,)
    assert plans[0].wardrobe == SemanticWardrobe("outfit_source", source.anchor_id)


def test_custom_names_round_trip_and_select_per_owner(resolver):
    saved = save_rows(resolver, [row("alice", "生日礼服", ["red_dress"]),
                                 row("alice", "月夜巡演", ["black_dress"]),
                                 row("bob", "生日礼服", ["white_suit"])])
    assert {r["qualifier"] for r in saved["outfits"]} == {"生日礼服", "月夜巡演"}
    assert len(saved["outfits"]) == 3
    anchors = tuple(SemanticAnchor(n, "target_character", "character", n, n, (n,)) for n in ("alice", "bob"))
    anchors += tuple(SemanticAnchor("s" + n, "clothing", "clothing", "生日礼服", "生日礼服", ()) for n in ("alice", "bob"))
    plans = tuple(SemanticCharacterPlan(n, SemanticWardrobe("creative_fallback", "s" + n)) for n in ("alice", "bob"))
    anchors, plans = resolver.bind_character_wardrobes(anchors, plans)
    result = resolver.cached_bound_wardrobes(anchors)
    effective = build_character_effective_outfits(anchors, plans, result, user_prompt="alice和bob分别穿生日礼服")
    assert effective[0].effective.effective_tags == ("red_dress",)
    assert effective[1].effective.effective_tags == ("white_suit",)
    assert all(p.wardrobe.kind == "outfit_source" for p in plans)
    assert all(r["ownerTags"] == r["sourceTags"] for r in saved["outfits"])


def test_donor_custom_wardrobe_uses_donor_owner(resolver):
    save_rows(resolver, [row("alice", "生日礼服", ["red_dress"]),
                         row("bob", "生日礼服", ["white_suit"])])
    anchors = (SemanticAnchor("t", "target_character", "character", "alice", "alice", ("alice",)),
               SemanticAnchor("s", "outfit_source", "character", "bob", "bob的生日礼服", ("bob",)))
    plans = (SemanticCharacterPlan("t", SemanticWardrobe("outfit_source", "s")),)
    anchors, plans = resolver.bind_character_wardrobes(anchors, plans)
    assert resolver.cached_bound_wardrobes(anchors).anchor_outfit_profiles[0][2] == ("white_suit",)


def test_custom_wardrobe_does_not_activate_from_owner_alone(resolver):
    save_rows(resolver, [dict(key="alice", aliases=["alice"], sourceTags=["alice"], tags=["shirt"], qualifier="default"),
                         dict(row("alice", "生日礼服", ["red_dress"]), aliases=["alice", "生日礼服"])])
    anchors = (SemanticAnchor("t", "target_character", "character", "bob", "bob", ("bob",)),
               SemanticAnchor("s", "outfit_source", "character", "alice", "alice的衣服", ("alice",)))
    anchors, plans = resolver.bind_character_wardrobes(anchors, (SemanticCharacterPlan("t", SemanticWardrobe("outfit_source", "s")),))
    assert resolver.cached_bound_wardrobes(anchors).anchor_outfit_profiles[0][2] == ("shirt",)


def test_ambiguous_custom_name_is_not_arbitrarily_selected(resolver):
    rows = [row("alice", "月夜巡演", ["red_dress"]), row("alice", "生日礼服", ["black_dress"])]
    for r in rows:
        r["aliases"] = ["礼服"]
    save_rows(resolver, rows)
    anchors = (SemanticAnchor("t", "target_character", "character", "alice", "alice", ("alice",)),
               SemanticAnchor("s", "clothing", "clothing", "礼服", "礼服", ()))
    plans = (SemanticCharacterPlan("t", SemanticWardrobe("creative_fallback", "s")),)
    _, updated = resolver.bind_character_wardrobes(anchors, plans)
    assert updated == plans


@pytest.mark.parametrize("category,exact,deprecated", [(0, True, False), (4, True, False), (0, False, False), (0, True, True)])
def test_online_source_is_exact_and_category_scoped(resolver, monkeypatch, category, exact, deprecated):
    monkeypatch.setattr(resolver, "_local_cli_path", lambda: Path("dummy"))
    monkeypatch.setattr(module, "lookup_semantic_anchors", lambda anchors, **kwargs: SemanticLookupResult(anchors=anchors, status="resolved"))
    monkeypatch.setattr(module, "_fetch_tag_records", lambda *a, **k: [TagRecord(
        name=SOURCE if exact else "togawa_sakiko", category=category, post_count=600, deprecated=deprecated)])
    calls = []
    def sample(tag, **kwargs):
        calls.append((tag, kwargs["source_kind"]))
        return VariantOutfitProfile(tags=("black_dress",), focused_posts=8)
    monkeypatch.setattr(module, "fetch_variant_outfit_profile", sample)
    resolver.remember_outfit_summary("togawa sakiko", ("togawa_sakiko",), ("school_uniform",))
    anchor = SemanticAnchor("s", "outfit_source", "character", "togawa sakiko (master of melodia)", "", (SOURCE,))
    result = asyncio.run(resolver.resolve_semantic_anchors((anchor,)))
    if not exact or deprecated:
        assert not calls
        assert not result.anchor_outfit_profiles
    else:
        assert calls == [(SOURCE, "named_outfit" if category == 0 else "character")]
        assert result.confirmed_tags == ()
        assert result.anchor_outfit_profiles[0][2] == ("black_dress",)
        learned = next(r for r in resolver.wardrobe_snapshot()["outfits"] if r["sourceTags"] == [SOURCE])
        if category == 0:
            assert learned["qualifier"] == "master of melodia"
            assert learned["ownerTags"] == ["togawa_sakiko"]
            assert learned["appearanceTags"] == []


def test_insufficient_learning_does_not_create_empty_profile(resolver, monkeypatch):
    monkeypatch.setattr(resolver, "_local_cli_path", lambda: Path("dummy"))
    monkeypatch.setattr(module, "lookup_semantic_anchors", lambda anchors, **kwargs: SemanticLookupResult(anchors=anchors, status="resolved"))
    monkeypatch.setattr(module, "_fetch_tag_records", lambda *a, **k: [TagRecord(name=SOURCE, category=0, post_count=10)])
    monkeypatch.setattr(module, "fetch_variant_outfit_profile", lambda *a, **k: VariantOutfitProfile())
    a = SemanticAnchor("s", "outfit_source", "character", "togawa sakiko (master of melodia)", "", (SOURCE,))
    result = asyncio.run(resolver.resolve_semantic_anchors((a,)))
    assert result.outfit_source_tags == (SOURCE,)
    assert resolver.wardrobe_snapshot()["outfits"] == []


@pytest.mark.parametrize("category", [0, 4, None])
def test_qualified_target_category_controls_identity_split(resolver, monkeypatch, category):
    resolver.remember_outfit_summary("togawa sakiko", ("togawa_sakiko",), ("school_uniform",))
    monkeypatch.setattr(module, "_fetch_tag_records", lambda *a, **k: [] if category is None else
                        [TagRecord(name=SOURCE, category=category, post_count=10)])
    anchor = SemanticAnchor("t", "target_character", "character", "丰川祥子", "", ("togawa_sakiko",),
                            literal_name="togawa sakiko (master of melodia)")
    anchors = resolver.prefer_cached_character_anchors((anchor,))
    assert anchors[0].candidates == (SOURCE,)
    anchors = asyncio.run(resolver.classify_literal_targets(anchors))
    anchors, plans = resolver.bind_character_wardrobes(anchors, (SemanticCharacterPlan("t", SemanticWardrobe("none")),))
    if category == 0:
        assert plans[0].wardrobe.kind == "outfit_source"
        assert anchors[0].source_text == "丰川祥子"
        assert anchors[0].candidates == ("togawa_sakiko",)
    else:
        assert plans[0].wardrobe.kind == "none"
        assert anchors[0].candidates == (SOURCE,)


@pytest.mark.parametrize("online", [False, True])
def test_custom_wardrobe_reaches_writer_without_default_fallback(resolver, monkeypatch, online):
    from types import SimpleNamespace
    from prompt_pipeline import PromptPipeline
    save_rows(resolver, [dict(key="alice", aliases=["alice"], sourceTags=["alice"],
                             tags=["school_uniform"], appearanceTags=["blue_hair"], qualifier="default"),
                         row("alice", "月夜巡演", ["black_dress"])])
    monkeypatch.setattr(resolver, "semantic_lookup_available", lambda: online)
    async def resolve(anchors):
        assert all(not a.profile_key for a in anchors)
        return SemanticLookupResult(anchors=anchors, confirmed_tags=("alice",),
                                    anchor_tags=(("target_1", "alice"),), status="resolved")
    monkeypatch.setattr(resolver, "resolve_semantic_anchors", resolve)
    calls = []
    outputs = [json.dumps(dict(characters=[dict(name="alice", aliases=[], clothing="月夜巡演", clothing_source=None)])),
               "{Count: 1girl}\n{Characters: alice}\n{Copyright: }\n{Identity: alice has blue hair}\n"
               "{Details: alice wears a black dress}\n{Tags: standing, white background}\n{Nltags: }"]
    class Context:
        async def get_current_chat_provider_id(self, _):
            return "test"
        async def llm_generate(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(completion_text=outputs.pop(0))
    researcher = SimpleNamespace(plan=lambda _: SimpleNamespace(use_web_search=False, use_deep_thinking=False,
                                                                search_reason="", thinking_reason=""))
    pipeline = PromptPipeline(context=Context(), config={}, logger=logging.getLogger("test"),
                              danbooru_resolver=resolver, researcher=researcher,
                              get_bool=lambda k,d: d, get_int=lambda k,d: d,
                              get_float=lambda k,d: d, get_str=lambda k,d: d, shorten=lambda x,limit=600:x[:limit])
    result = asyncio.run(pipeline.build(SimpleNamespace(unified_msg_origin="test"), "alice穿月夜巡演"))
    assert len(calls) == 2
    assert result.summary["semantic_character_outfits"][0]["effective_tags"] == ["black_dress"]
    assert "black_dress" in calls[1]["prompt"]
    assert "school_uniform" not in calls[1]["prompt"]


def test_shared_clothing_phrase_keeps_each_owners_row(resolver):
    save_rows(resolver, [row("alice", "生日演出服", ["red_dress"]), row("bob", "生日演出服", ["white_suit"])])
    prompt = "alice和bob都穿生日演出服"
    raw = json.dumps(dict(characters=[dict(name=name, aliases=[], clothing="生日演出服", clothing_source=None)
                                     for name in ["alice", "bob"]]))
    anchors = parse_semantic_plan(raw, prompt)
    anchors = tuple(replace(a, candidates=(a.source_text,)) if a.role == "target_character" else a for a in anchors)
    plans = parse_semantic_character_plans(raw, prompt, anchors)
    assert plans[0].clothing_anchor_id == plans[1].clothing_anchor_id
    anchors, plans = resolver.bind_character_wardrobes(anchors, plans)
    assert plans[0].wardrobe.anchor_id != plans[1].wardrobe.anchor_id
    result = resolver.cached_bound_wardrobes(anchors)
    effective = build_character_effective_outfits(anchors, plans, result, user_prompt=prompt)
    assert effective[0].effective.effective_tags == ("red_dress",)
    assert effective[1].effective.effective_tags == ("white_suit",)


def test_empty_custom_components_remain_an_explicit_empty_record(resolver):
    save_rows(resolver, [row("alice", "生日礼服", [])])
    anchors = (SemanticAnchor("t", "target_character", "character", "alice", "", ("alice",)),
               SemanticAnchor("s", "clothing", "clothing", "生日礼服", "生日礼服", ()))
    anchors, plans = resolver.bind_character_wardrobes(anchors, (SemanticCharacterPlan("t", SemanticWardrobe("creative_fallback"), clothing_anchor_id="s"),))
    result = resolver.cached_bound_wardrobes(anchors)
    assert result.anchor_outfit_profiles[0][2] == ()
    assert plans[0].wardrobe.kind == "outfit_source"
