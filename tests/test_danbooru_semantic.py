from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

import danbooru_semantic as semantic_module  # noqa: E402
import danbooru_resolver as resolver_module  # noqa: E402
from danbooru_semantic import (  # noqa: E402
    SemanticAnchor,
    SemanticLookupResult,
    SemanticWardrobe,
    build_semantic_plan_prompt,
    build_semantic_plan_repair_prompt,
    extract_parenthesized_character_aliases,
    extract_parenthesized_copyright_aliases,
    lookup_semantic_anchors,
    merge_semantic_results,
    prefer_configured_character_anchors,
    parse_semantic_plan,
    parse_semantic_appearance_changes,
    parse_semantic_outfit_directives,
    parse_semantic_character_plans,
    semantic_plan_validation_issues,
)
from danbooru_resolver import DanbooruResolver  # noqa: E402
from danbooru_tags import VariantOutfitProfile  # noqa: E402


def test_semantic_prompt_prioritizes_multi_character_wardrobe_binding() -> None:
    prompt = build_semantic_plan_prompt(
        "爱音cosplay黑魔导，祥子cosplay黑魔导女孩"
    )

    assert '"characters"' in prompt
    assert '"lookups"' not in prompt
    assert '"appearance_changes"' in prompt
    assert "爱音cosplay黑魔导，祥子cosplay黑魔导女孩" in prompt
    assert "never translate, guess tags" in prompt
    assert "A cosplay C" in prompt
    assert "clothing_source" in prompt


def test_semantic_appearance_changes_are_source_grounded_advisory_hints() -> None:
    prompt = "千早爱音变成异色瞳，丰川祥子保持原样"
    raw = json.dumps(
        {
            "characters": [
                {
                    "name": "千早爱音",
                    "clothing": None,
                    "clothing_source": None,
                    "clothing_changes": [],
                    "appearance_changes": [
                        {"dimension": "eye_traits", "source_text": "异色瞳"}
                    ],
                }
            ]
        },
        ensure_ascii=False,
    )

    changes = parse_semantic_appearance_changes(raw, prompt)

    assert len(changes) == 1
    assert changes[0].character_name == "千早爱音"
    assert changes[0].source_text == "异色瞳"
    assert changes[0].dimensions == ("eye_traits",)


def test_intent_only_plan_allows_unknown_tags_and_builds_local_match_anchors() -> None:
    prompt = "千早爱音穿着羽丘夏季校服但没穿短裙，头发变成蓝色"
    raw = json.dumps(
        {
            "characters": [
                {
                    "name": "千早爱音",
                    "wardrobe": {
                        "kind": "named_outfit",
                        "source": "羽丘夏季校服",
                    },
                    "changes": [
                        {
                            "operation": "remove",
                            "slots": ["lower_body.skirt"],
                            "source_text": "没穿短裙",
                        }
                    ],
                    "appearance_changes": [
                        {"dimension": "hair_color", "source_text": "头发变成蓝色"}
                    ],
                }
            ]
        },
        ensure_ascii=False,
    )

    anchors = parse_semantic_plan(raw, prompt)
    plans = parse_semantic_character_plans(raw, prompt, anchors)

    assert semantic_plan_validation_issues(raw, prompt) == ()
    assert [(item.role, item.source_text, item.candidates) for item in anchors] == [
        ("outfit", "羽丘夏季校服", ()),
        ("target_character", "千早爱音", ()),
    ]
    assert plans[0].wardrobe.kind == "named_outfit"
    assert plans[0].wardrobe.anchor_id == anchors[0].anchor_id
    assert plans[0].directives[0].operation == "remove"


def test_intent_only_unknown_wardrobe_does_not_require_or_invent_tags() -> None:
    prompt = "千早爱音和丰川祥子站在一起"
    raw = json.dumps(
        {
            "characters": [
                {
                    "name": "千早爱音",
                    "wardrobe": {"kind": "none", "source": ""},
                    "changes": [],
                    "appearance_changes": [],
                },
                {
                    "name": "丰川祥子",
                    "wardrobe": {"kind": "none", "source": ""},
                    "changes": [],
                    "appearance_changes": [],
                },
            ]
        },
        ensure_ascii=False,
    )

    assert semantic_plan_validation_issues(raw, prompt) == ()
    anchors = parse_semantic_plan(raw, prompt)
    assert [anchor.candidates for anchor in anchors] == [(), ()]
    assert [plan.wardrobe.kind for plan in parse_semantic_character_plans(
        raw, prompt, anchors
    )] == ["none", "none"]


def test_readable_intent_schema_maps_cosplay_source_without_tag_guess() -> None:
    prompt = "丰川祥子在cosplay初音未来，千早爱音站在旁边"
    raw = json.dumps(
        {
            "characters": [
                {
                    "name": "丰川祥子",
                    "clothing": "cosplay初音未来",
                    "clothing_source": "初音未来",
                    "clothing_changes": [],
                    "appearance_changes": [],
                },
                {
                    "name": "千早爱音",
                    "clothing": None,
                    "clothing_source": None,
                    "clothing_changes": [],
                    "appearance_changes": [],
                },
            ]
        },
        ensure_ascii=False,
    )

    anchors = parse_semantic_plan(raw, prompt)
    plans = parse_semantic_character_plans(raw, prompt, anchors)

    assert semantic_plan_validation_issues(raw, prompt) == ()
    assert [(anchor.role, anchor.source_text, anchor.candidates) for anchor in anchors] == [
        ("outfit_source", "初音未来", ()),
        ("target_character", "丰川祥子", ()),
        ("target_character", "千早爱音", ()),
    ]
    assert plans[0].wardrobe == SemanticWardrobe("outfit_source", anchors[0].anchor_id)
    assert plans[1].wardrobe == SemanticWardrobe("none")


def test_readable_intent_only_changes_uses_default_as_modification_base() -> None:
    prompt = "千早爱音的裙子变成蓝色"
    raw = json.dumps(
        {
            "characters": [
                {
                    "name": "千早爱音",
                    "clothing": None,
                    "clothing_source": None,
                    "clothing_changes": [
                        {
                            "operation": "replace_color",
                            "slots": ["lower_body.skirt"],
                            "color": "blue",
                            "source_text": "裙子变成蓝色",
                        }
                    ],
                    "appearance_changes": [],
                }
            ]
        },
        ensure_ascii=False,
    )

    plans = parse_semantic_character_plans(raw, prompt)

    assert plans[0].wardrobe == SemanticWardrobe("default_profile")
    assert plans[0].directives[0].operation == "replace_color"


def test_missing_outfit_source_reference_requests_llm_repair_without_guessing() -> None:
    prompt = (
        "千早爱音正在cosplay重音teto，但她的裙子不翼而飞，上衣也被撕破了。"
        "她绑着粉色的双钻头发型，就像重音teto那样。"
    )
    raw = json.dumps(
        {
            "anchors": [
                {
                    "id": "target_1",
                    "role": "target_character",
                    "group": "character",
                    "source_text": "千早爱音",
                    "description": "Chihaya Anon",
                    "candidates": ["chihaya_anon"],
                },
                {
                    "id": "source_1",
                    "role": "outfit_source",
                    "group": "character",
                    "source_text": "重音teto",
                    "description": "Kasane Teto outfit source",
                    "candidates": ["kasane_teto"],
                },
            ],
            "character_plans": [
                {
                    "target_anchor_id": "target_1",
                    "wardrobe": {"kind": "outfit_source"},
                    "directives": [
                        {
                            "operation": "remove",
                            "slots": ["lower_body.skirt"],
                            "source_text": "她的裙子不翼而飞",
                        },
                        {
                            "operation": "damage",
                            "slots": ["upper_body.primary"],
                            "source_text": "上衣也被撕破了",
                        },
                    ],
                }
            ],
        },
        ensure_ascii=False,
    )

    issues = semantic_plan_validation_issues(raw, prompt)
    repair = build_semantic_plan_repair_prompt(prompt, raw, issues)

    assert issues == (
        "character_plans[1].wardrobe.anchor_id is required when kind is outfit_source",
    )
    assert "anchor_id is required" in repair
    assert '"source_text": "重音teto"' in repair
    assert '"operation": "damage"' in repair


def test_compact_semantic_plan_builds_internal_ids_and_relationships() -> None:
    prompt = "千早爱音cosplay重音teto，上衣也被撕破了"
    raw = json.dumps(
        {
            "characters": [
                {
                    "name": "千早爱音",
                    "candidates": ["chihaya_anon"],
                    "wardrobe": {
                        "kind": "outfit_source",
                        "source": "重音teto",
                    },
                    "changes": [
                        {
                            "operation": "damage",
                            "slots": ["upper_body.primary"],
                            "source_text": "上衣也被撕破了",
                        }
                    ],
                }
            ],
            "lookups": [
                {
                    "text": "重音teto",
                    "role": "outfit_source",
                    "candidates": ["kasane_teto"],
                }
            ],
        },
        ensure_ascii=False,
    )

    anchors = parse_semantic_plan(raw, prompt)
    plans = parse_semantic_character_plans(raw, prompt, anchors)

    assert semantic_plan_validation_issues(raw, prompt) == ()
    assert [(item.anchor_id, item.role) for item in anchors] == [
        ("lookup_1", "outfit_source"),
        ("target_1", "target_character"),
    ]
    assert plans[0].target_anchor_id == "target_1"
    assert plans[0].wardrobe.anchor_id == "lookup_1"
    assert plans[0].directives[0].operation == "damage"


def test_compact_semantic_plan_only_repairs_missing_source_relationship() -> None:
    prompt = "千早爱音cosplay重音teto"
    raw = json.dumps(
        {
            "characters": [
                {
                    "name": "千早爱音",
                    "tag": "chihaya_anon",
                    "wardrobe": {
                        "kind": "outfit_source",
                        "source": "重音teto",
                    },
                }
            ],
            "lookups": [],
        },
        ensure_ascii=False,
    )

    assert semantic_plan_validation_issues(raw, prompt) == (
        "characters[1].wardrobe.source must exactly match one wardrobe lookup "
        "from the request",
    )


def test_compact_named_outfit_normalizes_unique_outfit_source_role() -> None:
    prompt = "丰川祥子和千早爱音穿着羽丘夏季校服"
    raw = json.dumps(
        {
            "characters": [
                {
                    "name": "丰川祥子",
                    "tag": "togawa_sakiko",
                    "wardrobe": {"kind": "named_outfit", "source": "羽丘夏季校服"},
                },
                {
                    "name": "千早爱音",
                    "tag": "chihaya_anon",
                    "wardrobe": {"kind": "named_outfit", "source": "羽丘夏季校服"},
                },
            ],
            "lookups": [
                {
                    "text": "羽丘夏季校服",
                    "role": "outfit_source",
                    "tag": "haneoka_school_uniform",
                }
            ],
        },
        ensure_ascii=False,
    )

    anchors = parse_semantic_plan(raw, prompt)
    plans = parse_semantic_character_plans(raw, prompt, anchors)

    assert semantic_plan_validation_issues(raw, prompt) == ()
    assert [(anchor.anchor_id, anchor.role) for anchor in anchors] == [
        ("lookup_1", "outfit"),
        ("target_1", "target_character"),
        ("target_2", "target_character"),
    ]
    assert [plan.wardrobe.anchor_id for plan in plans] == ["lookup_1", "lookup_1"]
    assert all(plan.wardrobe.kind == "named_outfit" for plan in plans)


def test_compact_shared_source_with_conflicting_wardrobe_kinds_is_rejected() -> None:
    prompt = "丰川祥子穿羽丘夏季校服，千早爱音cosplay羽丘夏季校服"
    raw = json.dumps(
        {
            "characters": [
                {
                    "name": "丰川祥子",
                    "tag": "togawa_sakiko",
                    "wardrobe": {"kind": "named_outfit", "source": "羽丘夏季校服"},
                },
                {
                    "name": "千早爱音",
                    "tag": "chihaya_anon",
                    "wardrobe": {"kind": "outfit_source", "source": "羽丘夏季校服"},
                },
            ],
            "lookups": [
                {
                    "text": "羽丘夏季校服",
                    "role": "outfit_source",
                    "tag": "haneoka_school_uniform",
                }
            ],
        },
        ensure_ascii=False,
    )

    issues = semantic_plan_validation_issues(raw, prompt)

    assert len(issues) == 2
    assert all("conflicting wardrobe roles" in issue for issue in issues)


def test_compact_validation_ignores_unreferenced_optional_lookup_noise() -> None:
    prompt = "千早爱音站在舞台上"
    raw = json.dumps(
        {
            "characters": [
                {
                    "name": "千早爱音",
                    "candidates": ["chihaya_anon"],
                    "wardrobe": {"kind": "none", "source": ""},
                    "changes": [{"unexpected": "discard later"}],
                }
            ],
            "lookups": [
                {"text": "not grounded", "role": "unknown", "candidates": []}
            ],
        },
        ensure_ascii=False,
    )

    assert semantic_plan_validation_issues(raw, prompt) == ()


def test_character_plans_require_unique_role_correct_anchor_references() -> None:
    prompt = "丰川祥子默认服装，千早爱音穿羽丘冬季校服，祥子下半身什么都没穿"
    raw = json.dumps(
        {
            "anchors": [
                {"id": "sakiko", "role": "target_character", "group": "character", "source_text": "丰川祥子", "description": "Sakiko", "candidates": ["togawa_sakiko"]},
                {"id": "anon", "role": "target_character", "group": "character", "source_text": "千早爱音", "description": "Anon", "candidates": ["chihaya_anon"]},
                {"id": "haneoka", "role": "outfit", "group": "outfit", "source_text": "羽丘冬季校服", "description": "winter uniform", "candidates": ["haneoka_school_uniform"]},
            ],
            "character_plans": [
                {"target_anchor_id": "sakiko", "wardrobe": {"kind": "default_profile"}, "directives": [{"operation": "remove", "slots": ["lower_body.all"], "source_text": "下半身什么都没穿"}]},
                {"target_anchor_id": "anon", "wardrobe": {"kind": "named_outfit", "anchor_id": "haneoka"}, "directives": []},
                {"target_anchor_id": "haneoka", "wardrobe": {"kind": "default_profile"}, "directives": []},
                {"target_anchor_id": "anon", "wardrobe": {"kind": "named_outfit", "anchor_id": "missing"}, "directives": []},
            ],
        },
        ensure_ascii=False,
    )

    plans = parse_semantic_character_plans(raw, prompt)

    assert [plan.target_anchor_id for plan in plans] == ["sakiko", "anon"]
    assert plans[0].directives[0].slots == ("lower_body.all",)
    assert plans[1].wardrobe.anchor_id == "haneoka"


def test_duplicate_anchor_id_makes_character_plan_reference_ambiguous() -> None:
    prompt = "丰川祥子穿羽丘校服"
    raw = json.dumps({
        "anchors": [
            {"id": "target", "role": "target_character", "group": "character", "source_text": "丰川祥子", "description": "Sakiko", "candidates": ["togawa_sakiko"]},
            {"id": "uniform", "role": "outfit", "group": "outfit", "source_text": "羽丘校服", "description": "uniform", "candidates": ["haneoka_school_uniform"]},
            {"id": "uniform", "role": "outfit", "group": "outfit", "source_text": "羽丘校服", "description": "duplicate", "candidates": ["school_uniform"]},
        ],
        "character_plans": [{"target_anchor_id": "target", "wardrobe": {"kind": "named_outfit", "anchor_id": "uniform"}, "directives": []}],
    }, ensure_ascii=False)

    assert parse_semantic_character_plans(raw, prompt) == ()


def test_semantic_source_phrase_matches_ascii_case_insensitively() -> None:
    plan = parse_semantic_plan(
        '{"anchors":[{"id":"source","role":"outfit_source",'
        '"group":"character","source_text":"oblivionis",'
        '"description":"Oblivionis outfit",'
        '"candidates":["oblivionis_(bang_dream!)"]}]}',
        "穿着Oblivionis服装",
    )

    assert plan and plan[0].role == "outfit_source"


def test_semantic_plan_accepts_only_keep_clothing_directive() -> None:
    directives = parse_semantic_outfit_directives(
        '{"anchors":[{"id":"target","role":"target_character"}],'
        '"outfit_directives":[{"operation":"keep_only",'
        '"slots":["outerwear","headwear"],'
        '"target_anchor_id":"target",'
        '"source_text":"除了最外面的大衣和头饰、头纱以外什么都没穿"}]}',
        "复仇者穿着常服，但除了最外面的大衣和头饰、头纱以外什么都没穿",
    )

    assert len(directives) == 1
    assert directives[0].operation == "keep_only"
    assert directives[0].slots == ("outerwear", "headwear")


def test_semantic_outfit_directive_rejects_unsourced_or_tag_level_claims() -> None:
    directives = parse_semantic_outfit_directives(
        '{"anchors":[{"id":"target","role":"target_character"}],'
        '"outfit_directives":[{"operation":"keep_only",'
        '"slots":["white_dress"],"source_text":"只保留大衣"},'
        '{"operation":"remove","slots":["outerwear"],'
        '"source_text":"用户没说过的话"}]}',
        "复仇者只保留大衣",
    )

    assert directives == ()


def test_semantic_plan_accepts_full_outfit_recolor_without_slots() -> None:
    directives = parse_semantic_outfit_directives(
        '{"anchors":[{"id":"target","role":"target_character"}],'
        '"outfit_directives":[{"operation":"recolor_all",'
        '"slots":[],"color":"black","source_text":"改成全黑色调"}]}',
        "复仇者的官方常服，改成全黑色调",
    )

    assert directives[0].operation == "recolor_all"
    assert directives[0].slots == ()
    assert directives[0].color == "black"


def test_semantic_plan_keeps_torn_garment_as_damage_not_removal() -> None:
    prompt = "千早爱音cosplay重音teto，上衣被撕破了"
    raw = json.dumps({
        "anchors": [
            {"id": "anon", "role": "target_character", "group": "character", "source_text": "千早爱音", "description": "Anon", "candidates": ["chihaya_anon"]},
            {"id": "teto", "role": "outfit_source", "group": "character", "source_text": "重音teto", "description": "Teto outfit", "candidates": ["kasane_teto"]},
        ],
        "character_plans": [{
            "target_anchor_id": "anon",
            "wardrobe": {"kind": "outfit_source", "anchor_id": "teto"},
            "directives": [{"operation": "damage", "slots": ["upper_body.primary"], "source_text": "上衣被撕破了"}],
        }],
    }, ensure_ascii=False)

    plans = parse_semantic_character_plans(raw, prompt)

    assert plans[0].directives[0].operation == "damage"
    assert "clothing_changes" in build_semantic_plan_prompt(prompt)
    assert "damaged clothing is a change, not removal" in build_semantic_plan_prompt(prompt)


def test_multi_target_outfit_directive_requires_a_valid_target_anchor() -> None:
    raw = (
        '{"anchors":[{"id":"sakiko","role":"target_character"},'
        '{"id":"anon","role":"target_character"}],'
        '"outfit_directives":[{"operation":"remove",'
        '"slots":["upper_body.primary"],"source_text":"爱音的上衣消失"},'
        '{"operation":"remove","slots":["upper_body.primary"],'
        '"target_anchor_id":"anon","source_text":"爱音的上衣消失"}]}'
    )

    directives = parse_semantic_outfit_directives(raw, "祥子穿常服，爱音的上衣消失")

    assert len(directives) == 1
    assert directives[0].target_anchor_id == "anon"


def test_parenthesized_english_alias_becomes_a_character_anchor() -> None:
    anchors = extract_parenthesized_character_aliases(
        "《黑夜君临》（Elden Ring Nightreign）的复仇者（revenant），双手抱胸"
    )

    assert len(anchors) == 1
    assert anchors[0].role == "target_character"
    assert anchors[0].source_text == "revenant"
    assert anchors[0].candidates == ("revenant",)


def test_parenthesized_english_title_becomes_a_copyright_anchor() -> None:
    anchors = extract_parenthesized_copyright_aliases(
        "《黑夜君临》（Elden Ring Nightreign）的复仇者（revenant）"
    )

    assert len(anchors) == 1
    assert anchors[0].role == "copyright"
    assert anchors[0].source_text == "Elden Ring Nightreign"
    assert anchors[0].candidates == ("elden_ring_nightreign",)


def test_parenthesized_english_title_preserves_danbooru_punctuation() -> None:
    anchors = extract_parenthesized_copyright_aliases(
        "《Re:从零开始的异世界生活》（Re:Zero kara Hajimeru Isekai Seikatsu）的蕾姆（rem）"
    )

    assert anchors[0].candidates == ("re:zero_kara_hajimeru_isekai_seikatsu",)


def test_parenthesized_alias_does_not_treat_a_localized_title_as_a_character() -> None:
    anchors = extract_parenthesized_character_aliases(
        "《黑夜君临》（Elden Ring Nightreign）的复仇者（revenant）"
    )

    assert [anchor.source_text for anchor in anchors] == ["revenant"]


def test_character_alias_prefers_the_explicit_copyright_scope(monkeypatch) -> None:
    anchors = (
        SemanticAnchor(
            anchor_id="alias",
            role="target_character",
            group="character",
            source_text="revenant",
            description="Explicit character alias",
            candidates=("revenant",),
        ),
        SemanticAnchor(
            anchor_id="work",
            role="copyright",
            group="series",
            source_text="黑夜君临",
            description="Elden Ring Nightreign",
            candidates=("elden_ring_nightreign",),
        ),
    )
    calls = []

    def fake_batch(queries, **_kwargs):
        calls.append(queries)
        if len(calls) == 1:
            return {
                "results": {
                    "a0_c0": {
                        "candidate_tags": {
                            "characters": [
                                {"tag": "revenant_(apex_legends)", "count": 354},
                                {"tag": "revenant_(elden_ring)", "count": 353},
                            ]
                        }
                    },
                    "a1_c0": {
                        "confirmed_tags": {
                            "series": [{"tag": "elden_ring_nightreign"}]
                        }
                    },
                }
            }
        return {
            "results": {
                "fallback_0": {
                    "confirmed_tags": {
                        "characters": [{"tag": "revenant_(elden_ring)"}]
                    }
                },
                "series_0": {"confirmed_tags": {"series": [{"tag": "elden_ring"}]}},
            }
        }

    monkeypatch.setattr(semantic_module, "_run_cli_batch", fake_batch)

    result = lookup_semantic_anchors(anchors, cli_path=PLUGIN_DIR / "unused.exe")

    assert result.confirmed_tags == (
        "revenant_(elden_ring)",
        "elden_ring_nightreign",
        "elden_ring",
    )


def test_contextual_character_aliases_cover_revenant_work_name_matrix() -> None:
    resolver = DanbooruResolver(
        logger=object(),
        cache={},
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
    )
    works = (
        "艾尔登法环",
        "艾尔登法环黑夜君临",
        "黑夜君临",
        "elden ring",
        "nightreign",
        "elden ring nightreign",
    )
    characters = ("复仇者", "revenant")

    for work in works:
        for character in characters:
            anchors = resolver.configured_character_anchors_for_prompt(
                f"{work}的{character}，双手抱胸"
            )
            assert [(anchor.role, anchor.candidates) for anchor in anchors] == [
                ("copyright", ("elden_ring",)),
                ("target_character", ("revenant_(elden_ring)",)),
            ]


def test_contextual_character_alias_requires_compatible_work_scope() -> None:
    resolver = DanbooruResolver(
        logger=object(),
        cache={},
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
        config={
            "danbooru_series_alias_mappings": [
                "作品甲 | work a sequel=work_a",
                "作品乙 | work b=work_b",
            ],
            "danbooru_character_alias_mappings": [
                "英雄甲 | hero a=hero_a_(work_a)",
            ],
        },
    )

    matched = resolver.configured_character_anchors_for_prompt(
        "work a sequel + hero a"
    )
    wrong_work = resolver.configured_character_anchors_for_prompt("work b + hero a")
    no_work = resolver.configured_character_anchors_for_prompt("hero a")

    assert [anchor.candidates for anchor in matched] == [
        ("work_a",),
        ("hero_a_(work_a)",),
    ]
    assert [anchor.candidates for anchor in wrong_work] == [("work_b",)]
    assert no_work == ()


def test_contextual_character_anchor_is_exactly_validated_locally(monkeypatch) -> None:
    resolver = DanbooruResolver(
        logger=object(),
        cache={},
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
    )
    anchors = resolver.configured_character_anchors_for_prompt(
        "elden ring nightreign + revenant"
    )

    def fake_batch(queries, **_kwargs):
        return {
            "results": {
                query["id"]: {
                    "confirmed_tags": {
                        "characters" if query["group"] == "character" else "series": [
                            {"tag": query["keyword"], "count": 100}
                        ]
                    }
                }
                for query in queries
            }
        }

    monkeypatch.setattr(semantic_module, "_run_cli_batch", fake_batch)

    result = lookup_semantic_anchors(anchors, cli_path=PLUGIN_DIR / "unused.exe")

    assert result.confirmed_tags == ("revenant_(elden_ring)", "elden_ring")
    assert result.missing_descriptions == ()


def test_configured_identity_drops_weaker_duplicate_and_misclassified_work_aliases() -> None:
    resolver = DanbooruResolver(
        logger=object(),
        cache={},
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
    )
    prompt = "艾尔登法环黑夜君临(nightreign)的复仇者(revenant)"
    configured = resolver.configured_character_anchors_for_prompt(prompt)
    discovered = (
        *extract_parenthesized_character_aliases(prompt),
        SemanticAnchor(
            "planner_character",
            "target_character",
            "character",
            "复仇者",
            "Revenant",
            ("revenant",),
        ),
        SemanticAnchor(
            "other_character",
            "target_character",
            "character",
            "梅琳娜",
            "Melina",
            ("melina_(elden_ring)",),
        ),
    )

    preferred = prefer_configured_character_anchors(configured, discovered)

    assert [(anchor.role, anchor.candidates) for anchor in preferred] == [
        ("copyright", ("elden_ring",)),
        ("target_character", ("revenant_(elden_ring)",)),
        ("target_character", ("melina_(elden_ring)",)),
    ]


def test_semantic_plan_accepts_independently_named_outfit_role() -> None:
    plan = parse_semantic_plan(
        '{"anchors":[{"id":"outfit","role":"outfit",'
        '"group":"outfit","source_text":"羽丘校服",'
        '"description":"Haneoka school uniform",'
        '"candidates":["haneoka_school_uniform"]}]}',
        "千早爱音穿着羽丘校服",
    )

    assert plan and plan[0].role == "outfit"


def test_semantic_plan_recovers_full_named_uniform_source_from_generic_source() -> None:
    plan = parse_semantic_plan(
        '{"anchors":[{"id":"outfit","role":"outfit",'
        '"group":"outfit","source_text":"校服",'
        '"description":"Tsukinomori school uniform",'
        '"candidates":["school_uniform"]}]}',
        "若叶睦和长崎素世穿着月之森校服",
    )

    assert plan and plan[0].source_text == "月之森校服"


def test_specific_named_uniform_is_not_satisfied_by_generic_uniform_tag(
    monkeypatch,
) -> None:
    anchor = SemanticAnchor(
        anchor_id="outfit",
        role="outfit",
        group="outfit",
        source_text="月之森校服",
        description="Tsukinomori school uniform",
        candidates=("school_uniform",),
    )
    monkeypatch.setattr(
        semantic_module,
        "_run_cli_batch",
        lambda *_args, **_kwargs: {
            "results": {
                "a0_c0": {
                    "confirmed_tags": {
                        "general": [{"tag": "school_uniform"}]
                    }
                }
            }
        },
    )

    result = lookup_semantic_anchors((anchor,), cli_path=PLUGIN_DIR / "unused.exe")

    assert result.confirmed_tags == ()
    assert result.named_outfit_tags == ()
    assert result.missing_descriptions == ("Tsukinomori school uniform",)


def test_exact_general_fallback_named_outfit_is_screened_and_promoted(
    monkeypatch,
) -> None:
    anchor = SemanticAnchor(
        anchor_id="outfit",
        role="outfit",
        group="outfit",
        source_text="羽丘校服",
        description="Haneoka school uniform",
        candidates=("haneoka_school_uniform",),
    )

    monkeypatch.setattr(
        semantic_module,
        "_run_cli_batch",
        lambda *_args, **_kwargs: {
            "results": {
                "a0_c0": {
                    "candidate_tags": {
                        "general": [
                            {
                                "tag": "haneoka_school_uniform",
                                "category": "general",
                                "source_category": "general",
                                "count": 3432,
                                "match_layer": "group_general_fallback",
                            }
                        ]
                    }
                }
            }
        },
    )

    result = lookup_semantic_anchors((anchor,), cli_path=PLUGIN_DIR / "unused.exe")

    assert result.confirmed_tags == ("haneoka_school_uniform",)
    assert result.named_outfit_tags == ("haneoka_school_uniform",)


def test_general_fallback_individual_garment_is_not_promoted(monkeypatch) -> None:
    anchor = SemanticAnchor(
        anchor_id="shirt",
        role="clothing",
        group="clothing",
        source_text="白衬衫",
        description="white shirt",
        candidates=("white_shirt",),
    )
    monkeypatch.setattr(
        semantic_module,
        "_run_cli_batch",
        lambda *_args, **_kwargs: {
            "results": {
                "a0_c0": {
                    "candidate_tags": {
                        "general": [
                            {
                                "tag": "white_shirt",
                                "source_category": "general",
                                "count": 100000,
                                "match_layer": "group_general_fallback",
                            }
                        ]
                    }
                }
            }
        },
    )

    result = lookup_semantic_anchors((anchor,), cli_path=PLUGIN_DIR / "unused.exe")

    assert result.confirmed_tags == ()
    assert result.named_outfit_tags == ()


def test_confirmed_named_outfit_is_marked_for_persistence(monkeypatch) -> None:
    anchor = SemanticAnchor(
        anchor_id="wedding",
        role="outfit",
        group="outfit",
        source_text="婚礼衣服",
        description="wedding dress outfit",
        candidates=("wedding_dress",),
    )
    monkeypatch.setattr(
        semantic_module,
        "_run_cli_batch",
        lambda *_args, **_kwargs: {
            "results": {
                "a0_c0": {
                    "confirmed_tags": {
                        "general": [{"tag": "wedding_dress"}]
                    }
                }
            }
        },
    )

    result = lookup_semantic_anchors((anchor,), cli_path=PLUGIN_DIR / "unused.exe")

    assert result.confirmed_tags == ("wedding_dress",)
    assert result.named_outfit_tags == ("wedding_dress",)


def test_semantic_cache_and_fresh_lookup_merge_without_shadowing() -> None:
    cached = SemanticLookupResult(
        confirmed_tags=("oblivionis_(bang_dream!)",),
        outfit_source_tags=("oblivionis_(bang_dream!)",),
        outfit_profile_tags=("red_shirt",),
        status="profile_cache",
    )
    fresh = SemanticLookupResult(
        confirmed_tags=("haneoka_school_uniform",),
        named_outfit_tags=("haneoka_school_uniform",),
        status="resolved",
    )

    merged = merge_semantic_results(cached, fresh)

    assert merged.status == "resolved"
    assert merged.outfit_profile_tags == ("red_shirt",)
    assert merged.named_outfit_tags == ("haneoka_school_uniform",)


def test_outfit_profile_cache_survives_resolver_recreation() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    def build(path: Path) -> DanbooruResolver:
        return DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        build(path).remember_outfit_summary(
            "初音未来",
            ("hatsune_miku",),
            ("grey_shirt", "detached_sleeves", "pleated_skirt"),
        )
        cached = build(path).cached_outfit_source("初音未来")

    assert cached is not None
    assert cached.status == "profile_cache"
    assert cached.outfit_source_tags == ("hatsune_miku",)
    assert cached.outfit_profile_tags == (
        "grey_shirt",
        "detached_sleeves",
        "pleated_skirt",
    )


def test_editable_outfit_profile_alias_is_a_hard_tag_trigger_for_normal_prompt() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )
        resolver.remember_outfit_summary(
            "amoris",
            ("amoris_(bang_dream!)",),
            ("black_corset", "red_shorts"),
            {"appearance_tags": ["blue_eyes"]},
        )
        snapshot = resolver.wardrobe_snapshot()
        snapshot["outfits"][0]["aliases"].extend(["阿莫莉丝", "amoris costume"])
        resolver.save_wardrobe(
            {
                "baseRevision": snapshot["revision"],
                "outfits": snapshot["outfits"],
                "outfitSets": snapshot["outfitSets"],
                "terms": snapshot["terms"],
            }
        )
        cached = resolver.cached_outfit_profiles_for_prompt(
            "让角色穿阿莫莉丝，在舞台上挥手"
        )

    assert cached is not None
    assert cached.confirmed_tags == ("amoris_(bang_dream!)", "bang_dream!")
    assert cached.outfit_profile_tags == ("black_corset", "red_shorts")
    assert cached.appearance_profile_tags == ("blue_eyes",)


def test_exact_visual_profile_alias_overrides_untrusted_outfit_source_candidate() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )
        resolver.remember_outfit_summary(
            "重音tetosv",
            ("kasane_teto_(sv)",),
            ("grey_jacket", "grey_skirt"),
        )
        anchor = SemanticAnchor(
            "source",
            "outfit_source",
            "character",
            "重音tetosv",
            "Kasane Teto SV outfit donor",
            ("kasane_teto",),
        )

        preferred = resolver.prefer_cached_outfit_source_anchors((anchor,))

    assert preferred[0].candidates == ("kasane_teto_(sv)",)


def test_visual_profile_alias_matches_inside_longer_source_phrase() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=Path(directory) / "profiles.json",
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )
        resolver.remember_outfit_summary(
            "kasane teto (sv)",
            ("kasane_teto_(sv)",),
            ("grey_jacket",),
        )
        anchor = SemanticAnchor(
            "source",
            "outfit_source",
            "character",
            "SynthV服装（也就是kasane teto (sv))",
            "outfit donor",
            ("kasane_teto",),
        )

        preferred = resolver.prefer_cached_character_anchors((anchor,))

    assert preferred[0].candidates == ("kasane_teto_(sv)",)


def test_visual_profile_alias_also_stabilizes_visible_target_identity() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=Path(directory) / "profiles.json",
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )
        resolver.remember_outfit_summary(
            "闪刀姬零衣",
            ("sky_striker_ace_-_raye",),
            ("two-tone_dress",),
        )
        anchor = SemanticAnchor(
            "target",
            "target_character",
            "character",
            "闪刀姬零衣",
            "visible character",
            ("raye",),
        )

        preferred = resolver.prefer_cached_character_anchors((anchor,))

    assert preferred[0].candidates == ("sky_striker_ace_-_raye",)


def test_visual_profile_alias_does_not_cross_wardrobe_qualifiers() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=Path(directory) / "profiles.json",
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )
        resolver.remember_outfit_summary(
            "角色甲的演出服",
            ("character_a",),
            ("stage_dress",),
            qualifier="stage",
        )
        anchor = SemanticAnchor(
            "source",
            "outfit_source",
            "character",
            "角色甲",
            "default outfit donor",
            ("wrong_character",),
        )

        preferred = resolver.prefer_cached_character_anchors((anchor,))

    assert preferred[0].candidates == ("wrong_character",)


def test_prompt_profile_scan_uses_longest_non_overlapping_alias_occurrences() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=Path(directory) / "profiles.json",
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )
        resolver.remember_outfit_summary(
            "黑魔导", ("dark_magician",), ("dark_robe",)
        )
        resolver.remember_outfit_summary(
            "黑魔导女孩", ("dark_magician_girl",), ("blue_dress",)
        )

        girl_only = resolver.cached_outfit_profiles_for_prompt(
            "祥子cosplay黑魔导女孩"
        )
        both = resolver.cached_outfit_profiles_for_prompt(
            "爱音cosplay黑魔导，祥子cosplay黑魔导女孩"
        )

    assert girl_only is not None
    assert girl_only.outfit_profile_tags == ("blue_dress",)
    assert both is not None
    assert set(both.outfit_profile_tags) == {"dark_robe", "blue_dress"}


def test_learning_refuses_second_canonical_source_for_exact_trigger_alias() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )
        resolver.remember_outfit_summary(
            "重音tetosv",
            ("kasane_teto_(sv)",),
            ("grey_jacket", "grey_skirt"),
        )
        resolver.remember_outfit_summary(
            "重音tetosv",
            ("kasane_teto",),
            ("white_shirt", "black_skirt"),
        )
        profiles = json.loads(path.read_text(encoding="utf-8"))["profiles"]

    assert list(profiles) == ["重音tetosv"]
    assert profiles["重音tetosv"]["source_tags"] == ["kasane_teto_(sv)"]
    assert profiles["重音tetosv"]["outfit_tags"] == ["grey_jacket", "grey_skirt"]


def test_scoped_source_suffix_disambiguates_existing_trigger_collision() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        path.write_text(
            json.dumps(
                {
                    "version": 3,
                    "profiles": {
                        "wrong": {
                            "kind": "character_outfit",
                            "qualifier": "default",
                            "aliases": ["重音tetosv"],
                            "source_tags": ["kasane_teto"],
                            "outfit_tags": ["white_shirt"],
                        },
                        "sv": {
                            "kind": "character_outfit",
                            "qualifier": "default",
                            "aliases": ["重音tetosv"],
                            "source_tags": ["kasane_teto_(sv)"],
                            "outfit_tags": ["grey_jacket"],
                        },
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )
        anchor = SemanticAnchor(
            "source", "outfit_source", "character", "重音tetosv", "donor",
            ("kasane_teto",),
        )

        preferred = resolver.prefer_cached_outfit_source_anchors((anchor,))

    assert preferred[0].candidates == ("kasane_teto_(sv)",)


def test_refresh_preserves_editor_aliases_outfit_and_stable_appearance() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )
        resolver.remember_outfit_summary(
            "sky_striker_ace_-_raye",
            ("sky_striker_ace_-_raye",),
            ("two-tone_dress",),
            {"appearance_tags": ["blonde_hair", "long_hair", "green_eyes"]},
        )
        snapshot = resolver.wardrobe_snapshot()
        snapshot["outfits"][0]["aliases"].extend(["闪刀姬零衣", "零衣"])
        snapshot["outfits"][0]["tags"] = ["manual_dress", "black_thighhighs"]
        resolver.save_wardrobe(
            {
                "baseRevision": snapshot["revision"],
                "outfits": snapshot["outfits"],
                "outfitSets": snapshot["outfitSets"],
                "terms": snapshot["terms"],
            }
        )

        resolver.remember_outfit_summary(
            "闪刀姬零衣",
            ("sky_striker_ace_-_raye",),
            ("refreshed_wrong_dress",),
            {"appearance_tags": ["short_blue_hair", "blue_eyes"]},
        )
        saved = json.loads(path.read_text(encoding="utf-8"))["profiles"]
        cached = resolver.cached_outfit_profiles_for_prompt("游戏王的闪刀姬零衣")

    profile = saved["sky_striker_ace_-_raye"]
    assert "闪刀姬零衣" in profile["aliases"]
    assert "零衣" in profile["aliases"]
    assert profile["outfit_tags"] == ["manual_dress", "black_thighhighs"]
    assert profile["evidence"]["appearance_tags"] == [
        "blonde_hair",
        "long_hair",
        "green_eyes",
    ]
    assert cached is not None
    assert cached.confirmed_tags == ("sky_striker_ace_-_raye",)


def test_wardrobe_editor_can_add_and_correct_stable_appearance() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    resolver = DanbooruResolver(
        logger=_Logger(),
        cache={},
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
    )
    initial = resolver.wardrobe_snapshot()
    saved = resolver.save_wardrobe(
        {
            "baseRevision": initial["revision"],
            "outfits": [
                {
                    "key": "hatsune_miku",
                    "aliases": ["初音未来"],
                    "sourceTags": ["hatsune_miku"],
                    "tags": ["necktie"],
                    "appearanceTags": ["aqua_hair", "aqua_eyes"],
                    "qualifier": "default",
                }
            ],
            "outfitSets": [],
            "terms": [],
        }
    )
    saved["outfits"][0]["appearanceTags"] = ["blue_hair", "blue_eyes"]
    corrected = resolver.save_wardrobe(
        {
            "baseRevision": saved["revision"],
            "outfits": saved["outfits"],
            "outfitSets": saved["outfitSets"],
            "terms": saved["terms"],
        }
    )

    evidence = resolver._profile_data()["profiles"]["hatsune_miku"]["evidence"]
    assert corrected["outfits"][0]["appearanceTags"] == [
        "blue_hair",
        "blue_eyes",
    ]
    assert evidence["appearance_tags"] == ["blue_hair", "blue_eyes"]
    assert evidence["appearance_manual_override"] is True


def test_cleared_stable_appearance_is_not_relearned_on_refresh() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    resolver = DanbooruResolver(
        logger=_Logger(),
        cache={},
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
    )
    resolver.remember_outfit_summary(
        "hatsune_miku",
        ("hatsune_miku",),
        ("necktie",),
        {"appearance_tags": ["wrong_hair", "wrong_eyes"]},
    )
    snapshot = resolver.wardrobe_snapshot()
    snapshot["outfits"][0]["appearanceTags"] = []
    resolver.save_wardrobe(
        {
            "baseRevision": snapshot["revision"],
            "outfits": snapshot["outfits"],
            "outfitSets": snapshot["outfitSets"],
            "terms": snapshot["terms"],
        }
    )
    resolver.remember_outfit_summary(
        "hatsune_miku",
        ("hatsune_miku",),
        ("new_outfit",),
        {"appearance_tags": ["relearned_hair", "relearned_eyes"]},
    )

    evidence = resolver._profile_data()["profiles"]["hatsune_miku"]["evidence"]
    assert resolver.wardrobe_snapshot()["outfits"][0]["appearanceTags"] == []
    assert evidence["appearance_manual_override"] is True


def test_stable_appearance_matches_when_requested_outfit_variant_is_missing() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    resolver = DanbooruResolver(
        logger=_Logger(),
        cache={},
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
    )
    resolver.remember_outfit_summary(
        "hatsune_miku",
        ("hatsune_miku",),
        ("default_outfit",),
        {"appearance_tags": ["aqua_hair", "aqua_eyes"]},
    )

    cached = resolver.cached_outfit_profiles_for_prompt("hatsune_miku in winter")

    assert cached is not None
    assert cached.outfit_profile_tags == ()
    assert cached.character_appearance_profiles == (
        (("hatsune_miku",), "hatsune_miku", ("aqua_hair", "aqua_eyes")),
    )


def test_editable_outfit_profile_does_not_match_inside_latin_word() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    resolver = DanbooruResolver(
        logger=_Logger(),
        cache={},
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
    )
    resolver._profile_cache_data = {
        "version": 3,
        "profiles": {
            "cat": {
                "kind": "character_outfit",
                "aliases": ["cat"],
                "source_tags": ["cat_costume"],
                "outfit_tags": ["cat_ears"],
            }
        },
    }

    assert resolver.cached_outfit_profiles_for_prompt("a concatenate test") is None


def test_named_outfit_profile_survives_recreation_and_matches_english_alias() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    def build(path: Path) -> DanbooruResolver:
        return DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        build(path).remember_named_outfit(
            "羽丘校服", "haneoka_school_uniform"
        )
        cached = build(path).cached_named_outfits_for_prompt(
            "chihaya anon wears haneoka school uniform"
        )

    assert cached is not None
    assert cached.confirmed_tags == ("haneoka_school_uniform",)
    assert cached.named_outfit_tags == ("haneoka_school_uniform",)


def test_named_outfit_learning_keeps_distinct_entities_that_share_one_tag() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )
        resolver.remember_named_outfit("羽丘校服", "haneoka_school_uniform")
        resolver.remember_named_outfit("羽丘学园校服", "haneoka_school_uniform")
        resolver.remember_named_outfit("羽丘校服", "haneoka_school_uniform")

        snapshot = resolver.wardrobe_snapshot()
        resolver.save_wardrobe(
            {
                "baseRevision": snapshot["revision"],
                "outfits": snapshot["outfits"],
                "outfitSets": snapshot["outfitSets"],
                "terms": snapshot["terms"],
            }
        )
        persisted = json.loads(path.read_text(encoding="utf-8"))["profiles"]

    learned = [
        item for item in snapshot["outfitSets"] if item["origin"] == "learned"
    ]
    stored = [
        item for item in persisted.values() if item.get("kind") == "named_outfit"
    ]
    assert len(learned) == 2
    assert {item["alias"] for item in learned} == {"羽丘校服", "羽丘学园校服"}
    assert {item["tag"] for item in learned} == {"haneoka_school_uniform"}
    assert len(stored) == 2
    assert {item["aliases"][0] for item in stored} == {"羽丘校服", "羽丘学园校服"}


def test_editor_alias_on_learned_outfit_survives_save_response_and_reload() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    def build(path: Path) -> DanbooruResolver:
        return DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        resolver = build(path)
        resolver.remember_named_outfit(
            "羽丘冬季校服", "haneoka_school_uniform", "winter"
        )
        snapshot = resolver.wardrobe_snapshot()
        item = snapshot["outfitSets"][0]
        item["aliases"].extend(["羽丘冬装", "Haneoka cold-weather look"])

        saved = resolver.save_wardrobe(
            {
                "baseRevision": snapshot["revision"],
                "outfits": snapshot["outfits"],
                "outfitSets": snapshot["outfitSets"],
                "terms": snapshot["terms"],
            }
        )
        reloaded = build(path).wardrobe_snapshot()

    assert "羽丘冬装" in saved["outfitSets"][0]["aliases"]
    assert "haneoka cold-weather look" in saved["outfitSets"][0]["aliases"]
    assert reloaded["outfitSets"][0]["aliases"] == saved["outfitSets"][0]["aliases"]


def test_configured_named_outfit_ui_groups_chinese_and_english_aliases() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    resolver = DanbooruResolver(
        logger=_Logger(),
        cache={},
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
        config={
            "danbooru_named_outfit_mappings": [
                "羽丘校服=haneoka_school_uniform"
            ]
        },
    )

    snapshot = resolver.wardrobe_snapshot()
    outfit_set = snapshot["outfitSets"][0]
    english_before_save = resolver.cached_named_outfits_for_prompt(
        "chihaya anon wears haneoka school uniform"
    )
    resolver.save_wardrobe(
        {
            "baseRevision": snapshot["revision"],
            "outfits": [],
            "outfitSets": snapshot["outfitSets"],
            "terms": [],
        }
    )
    english = resolver.cached_named_outfits_for_prompt(
        "chihaya anon wears haneoka school uniform"
    )

    assert outfit_set["tag"] == "haneoka_school_uniform"
    assert outfit_set["aliases"] == [
        "羽丘校服",
        "haneoka school uniform",
    ]
    assert "haneoka_school_uniform" not in outfit_set["aliases"]
    assert english_before_save is not None
    assert english_before_save.named_outfit_tags == ("haneoka_school_uniform",)
    assert resolver._config["danbooru_named_outfit_mappings"] == [
        "羽丘校服 | haneoka school uniform=haneoka_school_uniform",
    ]
    assert english is not None
    assert english.named_outfit_tags == ("haneoka_school_uniform",)


def test_configured_named_outfit_supports_many_manual_aliases_per_tag() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    resolver = DanbooruResolver(
        logger=_Logger(),
        cache={},
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
        config={
            "danbooru_named_outfit_mappings": [
                "羽丘校服 | 羽丘制服 | haneoka uniform | haneoka academy uniform=haneoka_school_uniform"
            ]
        },
    )

    snapshot = resolver.wardrobe_snapshot()
    aliases = snapshot["outfitSets"][0]["aliases"]
    matched = resolver.cached_named_outfits_for_prompt("穿羽丘制服拍照")

    assert aliases == [
        "羽丘校服",
        "羽丘制服",
        "haneoka uniform",
        "haneoka academy uniform",
        "haneoka school uniform",
    ]
    assert matched is not None
    assert matched.named_outfit_tags == ("haneoka_school_uniform",)
    assert len(matched.anchors) == 1
    assert matched.anchors[0].source_text == "羽丘制服"
    assert matched.anchors[0].candidates == ("haneoka_school_uniform",)


def test_editor_alias_on_configured_outfit_survives_config_reload() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    def build(config) -> DanbooruResolver:
        return DanbooruResolver(
            logger=_Logger(),
            cache={},
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
            config=config,
        )

    resolver = build(
        {"danbooru_named_outfit_mappings": ["羽丘校服=haneoka_school_uniform"]}
    )
    snapshot = resolver.wardrobe_snapshot()
    snapshot["outfitSets"][0]["aliases"].extend(
        ["羽丘学院套装", "Anon school look"]
    )
    saved = resolver.save_wardrobe(
        {
            "baseRevision": snapshot["revision"],
            "outfits": snapshot["outfits"],
            "outfitSets": snapshot["outfitSets"],
            "terms": snapshot["terms"],
        }
    )
    reloaded = build(resolver._config).wardrobe_snapshot()
    assert "羽丘学院套装" in saved["outfitSets"][0]["aliases"]
    assert "anon school look" in saved["outfitSets"][0]["aliases"]
    assert reloaded["outfitSets"][0]["aliases"] == saved["outfitSets"][0]["aliases"]


def test_configured_outfit_rows_sharing_one_tag_survive_save_and_reload() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    def build(config) -> DanbooruResolver:
        return DanbooruResolver(
            logger=_Logger(),
            cache={},
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
            config=config,
        )

    resolver = build(
        {
            "danbooru_named_outfit_mappings": [
                "羽丘夏季校服 | haneoka summer look=haneoka_school_uniform",
                "羽丘冬季校服 | haneoka winter look=haneoka_school_uniform",
            ]
        }
    )
    snapshot = resolver.wardrobe_snapshot()
    saved = resolver.save_wardrobe(
        {
            "baseRevision": snapshot["revision"],
            "outfits": snapshot["outfits"],
            "outfitSets": snapshot["outfitSets"],
            "terms": snapshot["terms"],
        }
    )
    reloaded = build(resolver._config).wardrobe_snapshot()
    matched = build(resolver._config).cached_named_outfits_for_prompt(
        "角色甲穿羽丘夏季校服，角色乙穿羽丘冬季校服"
    )

    assert len(saved["outfitSets"]) == 2
    assert len(reloaded["outfitSets"]) == 2
    assert {item["variant"] for item in reloaded["outfitSets"]} == {
        "summer",
        "winter",
    }
    assert {item["tag"] for item in reloaded["outfitSets"]} == {
        "haneoka_school_uniform"
    }
    assert matched is not None
    assert len(matched.anchors) == 2
    assert {anchor.source_text for anchor in matched.anchors} == {
        "羽丘夏季校服",
        "羽丘冬季校服",
    }


def test_seasonal_named_outfits_remain_separate_and_drop_poisoned_aliases() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        path.write_text(
            json.dumps(
                {
                    "version": 3,
                    "profiles": {
                        "两只大猫爪子": {
                            "kind": "named_outfit",
                            "aliases": [
                                "两只大猫爪子",
                                "羽丘冬季校服",
                                "haneoka school uniform",
                            ],
                            "outfit_tags": ["haneoka_school_uniform"],
                        },
                        "羽丘夏季校服": {
                            "kind": "named_outfit",
                            "variant": "summer",
                            "aliases": ["羽丘夏季校服"],
                            "outfit_tags": ["haneoka_school_uniform"],
                        },
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )

        snapshot = resolver.wardrobe_snapshot()
        winter_item = next(
            item for item in snapshot["outfitSets"] if item["variant"] == "winter"
        )
        winter_item["aliases"].append("羽丘冬季制服")
        saved = resolver.save_wardrobe(
            {
                "baseRevision": snapshot["revision"],
                "outfits": snapshot["outfits"],
                "outfitSets": snapshot["outfitSets"],
                "terms": snapshot["terms"],
            }
        )
        saved_again = resolver.save_wardrobe(
            {
                "baseRevision": saved["revision"],
                "outfits": saved["outfits"],
                "outfitSets": saved["outfitSets"],
                "terms": saved["terms"],
            }
        )
        persisted = json.loads(path.read_text(encoding="utf-8"))["profiles"]

    variants = {item["variant"]: item for item in snapshot["outfitSets"]}
    assert set(variants) == {"summer", "winter"}
    assert variants["winter"]["aliases"] == [
        "羽丘冬季校服",
        "haneoka school winter uniform",
        "羽丘冬季制服",
    ]
    assert "两只大猫爪子" not in variants["winter"]["aliases"]
    assert persisted["两只大猫爪子"]["variant"] == "winter"
    assert persisted["羽丘夏季校服"]["variant"] == "summer"
    assert saved["revision"] == saved_again["revision"]


def test_seasonal_outfit_profiles_use_distinct_keys_and_cache_entries() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )
        resolver.remember_outfit_summary(
            "羽丘夏季校服",
            ("haneoka_school_uniform",),
            ("summer_uniform", "white_shirt", "short_sleeves"),
            qualifier="summer",
        )
        resolver.remember_outfit_summary(
            "羽丘冬季校服",
            ("haneoka_school_uniform",),
            ("winter_uniform", "grey_jacket", "long_sleeves"),
            qualifier="winter",
        )
        summer = resolver.cached_outfit_source("羽丘夏季校服")
        winter = resolver.cached_outfit_source("羽丘冬季校服")
        saved = json.loads(path.read_text(encoding="utf-8"))["profiles"]

    assert "羽丘夏季校服::summer" in saved
    assert "羽丘冬季校服::winter" in saved
    assert summer is not None
    assert winter is not None
    assert summer.outfit_profile_tags[0] == "summer_uniform"
    assert winter.outfit_profile_tags[0] == "winter_uniform"


def test_casual_outfit_profile_does_not_fall_back_to_default_cache() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    resolver = DanbooruResolver(
        logger=_Logger(),
        cache={},
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
    )
    resolver.remember_outfit_summary(
        "角色甲",
        ("character_a",),
        ("school_uniform",),
    )
    resolver.remember_outfit_summary(
        "角色甲",
        ("character_a",),
        ("blue_cardigan", "jeans"),
        qualifier="casual",
    )

    casual = resolver.cached_outfit_profiles_for_prompt("角色甲穿官方常服")
    default = resolver.cached_outfit_profiles_for_prompt("角色甲站立")

    assert casual is not None
    assert casual.outfit_profile_tags == ("blue_cardigan", "jeans")
    assert default is not None
    assert default.outfit_profile_tags == ("school_uniform",)


def test_mixed_character_variants_are_selected_from_each_local_clause() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    resolver = DanbooruResolver(
        logger=_Logger(),
        cache={},
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
    )
    resolver.remember_outfit_summary(
        "千早爱音", ("chihaya_anon",), ("anon_school_uniform",)
    )
    resolver.remember_outfit_summary(
        "千早爱音", ("chihaya_anon",), ("pink_cardigan", "jeans"),
        qualifier="casual",
    )
    resolver.remember_outfit_summary(
        "丰川祥子", ("togawa_sakiko",), ("sakiko_school_uniform",)
    )
    resolver.remember_outfit_summary(
        "丰川祥子", ("togawa_sakiko",), ("blue_cardigan", "long_skirt"),
        qualifier="casual",
    )

    cached = resolver.cached_outfit_profiles_for_prompt(
        "千早爱音穿官方常服，丰川祥子穿默认服装"
    )

    assert cached is not None
    scoped = {
        alias: (tags, qualifier)
        for alias, _source, tags, qualifier in cached.source_outfit_profiles
    }
    assert scoped["千早爱音"] == (("pink_cardigan", "jeans"), "casual")
    assert scoped["丰川祥子"] == (("sakiko_school_uniform",), "default")


def test_target_casual_variant_uses_casual_post_query(monkeypatch) -> None:
    anchor = SemanticAnchor(
        "target",
        "target_character",
        "character",
        "角色甲",
        "A casual outfit variant",
        ("character_a",),
    )
    monkeypatch.setattr(
        resolver_module,
        "lookup_semantic_anchors",
        lambda *_args, **_kwargs: SemanticLookupResult(
            confirmed_tags=("character_a",),
            anchor_tags=(("target", "character_a"),),
            anchors=(anchor,),
            status="resolved",
        ),
    )
    calls: list[tuple[str, str]] = []

    def fake_profile(tag, *, outfit_kind, **_kwargs):
        calls.append((tag, outfit_kind))
        return VariantOutfitProfile(tags=("blue_shirt", "jeans"))

    monkeypatch.setattr(
        resolver_module, "fetch_variant_outfit_profile", fake_profile
    )

    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    resolver = DanbooruResolver(
        logger=_Logger(),
        cache={},
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
    )
    resolver._local_cli_path = lambda: PLUGIN_DIR / "unused.exe"

    result = asyncio.run(resolver.resolve_semantic_anchors((anchor,)))

    assert calls == [("character_a", "casual")]
    assert result.character_profiles == (
        ("target", "character_a", ("blue_shirt", "jeans"), ()),
    )


def test_fresh_editor_profile_is_used_without_resampling_outfit_source(
    monkeypatch,
) -> None:
    anchor = SemanticAnchor(
        "source",
        "outfit_source",
        "character",
        "重音tetosv",
        "outfit donor",
        ("kasane_teto_(sv)",),
    )
    monkeypatch.setattr(
        resolver_module,
        "lookup_semantic_anchors",
        lambda *_args, **_kwargs: SemanticLookupResult(
            confirmed_tags=("kasane_teto_(sv)",),
            outfit_source_tags=("kasane_teto_(sv)",),
            anchor_tags=(("source", "kasane_teto_(sv)"),),
            anchors=(anchor,),
            status="resolved",
        ),
    )

    def unexpected_fetch(*_args, **_kwargs):
        raise AssertionError("fresh editor profile must not be resampled")

    monkeypatch.setattr(
        resolver_module, "fetch_variant_outfit_profile", unexpected_fetch
    )

    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=Path(directory) / "profiles.json",
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )
        resolver._local_cli_path = lambda: PLUGIN_DIR / "unused.exe"
        resolver.remember_outfit_summary(
            "重音tetosv",
            ("kasane_teto_(sv)",),
            ("manual_grey_jacket", "manual_grey_skirt"),
            {"appearance_tags": ["red_hair", "drill_hair"]},
        )

        result = asyncio.run(resolver.resolve_semantic_anchors((anchor,)))

    assert result.anchor_outfit_profiles == ((
        "source",
        "kasane_teto_(sv)",
        ("manual_grey_jacket", "manual_grey_skirt"),
        "default",
    ),)


def test_named_outfit_persistence_stays_bound_to_matching_anchor(monkeypatch) -> None:
    anchors = (
        SemanticAnchor(
            "paws",
            "clothing",
            "clothing",
            "两只大猫爪子",
            "oversized cat paw gloves",
            ("cat_paws",),
        ),
        SemanticAnchor(
            "uniform",
            "outfit",
            "outfit",
            "羽丘冬季校服",
            "Haneoka winter school uniform",
            ("haneoka_school_uniform",),
        ),
    )
    result = SemanticLookupResult(
        confirmed_tags=("cat_paws", "haneoka_school_uniform"),
        named_outfit_tags=("haneoka_school_uniform",),
        anchors=anchors,
        status="resolved",
    )
    monkeypatch.setattr(
        resolver_module, "lookup_semantic_anchors", lambda *_args, **_kwargs: result
    )
    monkeypatch.setattr(
        resolver_module,
        "fetch_variant_outfit_profile",
        lambda *_args, **_kwargs: VariantOutfitProfile(),
    )

    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    resolver = DanbooruResolver(
        logger=_Logger(),
        cache={},
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
    )
    resolver._local_cli_path = lambda: PLUGIN_DIR / "unused.exe"
    remembered: list[tuple[str, str, str | None]] = []
    resolver.remember_named_outfit = (
        lambda alias, tag, variant=None: remembered.append((alias, tag, variant))
    )

    asyncio.run(resolver.resolve_semantic_anchors(anchors))

    assert remembered == [
        ("羽丘冬季校服", "haneoka_school_uniform", "winter")
    ]


def test_haneoka_summer_uniform_enters_concrete_profile_tag_query(
    monkeypatch,
) -> None:
    anchor = SemanticAnchor(
        "haneoka_summer",
        "outfit",
        "outfit",
        "羽丘夏季校服",
        "Haneoka summer school uniform",
        ("haneoka_school_uniform",),
    )
    monkeypatch.setattr(
        resolver_module,
        "lookup_semantic_anchors",
        lambda *_args, **_kwargs: SemanticLookupResult(
            confirmed_tags=("haneoka_school_uniform",),
            named_outfit_tags=("haneoka_school_uniform",),
            anchors=(anchor,),
            status="resolved",
        ),
    )
    calls: list[tuple[str, str]] = []

    def fake_profile(tag, *, outfit_kind, **_kwargs):
        calls.append((tag, outfit_kind))
        return VariantOutfitProfile(
            tags=("summer_uniform", "short_sleeves", "white_shirt"),
            sample_mode="summer_single_character_anchor",
            focused_posts=8,
        )

    monkeypatch.setattr(
        resolver_module, "fetch_variant_outfit_profile", fake_profile
    )

    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=Path(directory) / "profiles.json",
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )
        resolver._local_cli_path = lambda: PLUGIN_DIR / "unused.exe"
        result = asyncio.run(resolver.resolve_semantic_anchors((anchor,)))

    assert calls == [("haneoka_school_uniform", "summer")]
    assert result.anchor_outfit_profiles == (
        (
            "haneoka_summer",
            "haneoka_school_uniform",
            ("summer_uniform", "short_sleeves", "white_shirt"),
            "summer",
        ),
    )


def test_every_named_outfit_variant_uses_the_same_profile_query_policy(
    monkeypatch,
) -> None:
    anchors = (
        SemanticAnchor(
            "default_set",
            "outfit",
            "outfit",
            "默认套组",
            "default named outfit",
            ("default_named_set",),
        ),
        SemanticAnchor(
            "summer_set", "outfit", "outfit", "夏季套组", "summer named outfit", ("summer_named_set",)
        ),
        SemanticAnchor(
            "winter_set", "outfit", "outfit", "冬季套组", "winter named outfit", ("winter_named_set",)
        ),
        SemanticAnchor(
            "stage_set", "outfit", "outfit", "舞台套组", "stage outfit", ("stage_named_set",)
        ),
    )
    canonical_tags = tuple(anchor.candidates[0] for anchor in anchors)
    monkeypatch.setattr(
        resolver_module,
        "lookup_semantic_anchors",
        lambda *_args, **_kwargs: SemanticLookupResult(
            confirmed_tags=canonical_tags,
            named_outfit_tags=canonical_tags,
            anchors=anchors,
            status="resolved",
        ),
    )
    calls: list[tuple[str, str]] = []

    def fake_profile(tag, *, outfit_kind, **_kwargs):
        calls.append((tag, outfit_kind))
        return VariantOutfitProfile()

    monkeypatch.setattr(
        resolver_module, "fetch_variant_outfit_profile", fake_profile
    )

    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    resolver = DanbooruResolver(
        logger=_Logger(),
        cache={},
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
    )
    resolver._local_cli_path = lambda: PLUGIN_DIR / "unused.exe"
    asyncio.run(resolver.resolve_semantic_anchors(anchors))

    assert calls == [
        ("default_named_set", "default"),
        ("summer_named_set", "summer"),
        ("winter_named_set", "winter"),
        ("stage_named_set", "stage"),
    ]


def test_wardrobe_snapshot_does_not_collapse_distinct_rows_by_canonical_tag() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        path.write_text(
            json.dumps(
                {
                    "version": 3,
                    "profiles": {
                        "羽丘校服": {
                            "kind": "named_outfit",
                            "aliases": ["羽丘校服", "haneoka_school_uniform"],
                            "outfit_tags": ["haneoka_school_uniform"],
                        },
                        "羽丘学园校服": {
                            "kind": "named_outfit",
                            "aliases": ["羽丘学园校服", "haneoka school uniform"],
                            "outfit_tags": ["haneoka_school_uniform"],
                        },
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )

        snapshot = resolver.wardrobe_snapshot()

    assert len(snapshot["outfitSets"]) == 2
    assert {item["tag"] for item in snapshot["outfitSets"]} == {
        "haneoka_school_uniform"
    }
    assert {item["alias"] for item in snapshot["outfitSets"]} == {
        "羽丘校服",
        "羽丘学园校服",
    }


def test_generic_school_uniform_is_neither_persisted_nor_loaded_as_named_set() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    def build(path: Path) -> DanbooruResolver:
        return DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        path.write_text(
            json.dumps(
                {
                    "version": 3,
                    "profiles": {
                        "校服": {
                            "kind": "named_outfit",
                            "aliases": ["校服", "school_uniform"],
                            "outfit_tags": ["school_uniform"],
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        resolver = build(path)
        assert resolver.cached_named_outfits_for_prompt("羽丘校服") is None
        before = path.read_text(encoding="utf-8")
        resolver.remember_named_outfit("月之森校服", "school_uniform")
        after = path.read_text(encoding="utf-8")

    assert after == before


def test_configured_named_uniform_survives_unavailable_semantic_lookup() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=Path(directory) / "profiles.json",
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
            config={
                "danbooru_named_outfit_mappings": [
                    "月之森校服=tsukinomori_school_uniform"
                ]
            },
        )
        cached = resolver.cached_named_outfits_for_prompt(
            "若叶睦和长崎素世穿着月之森校服"
        )

    assert cached is not None
    assert cached.confirmed_tags == ("tsukinomori_school_uniform",)
    assert cached.named_outfit_tags == ("tsukinomori_school_uniform",)


def test_configured_named_outfit_mapping_validates_and_accepts_separators() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=Path(directory) / "profiles.json",
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
            config={
                "danbooru_named_outfit_mappings": [
                    "羽丘校服 → haneoka_school_uniform",
                    "坏映射=not a canonical tag?",
                ]
            },
        )

        matched = resolver.cached_named_outfits_for_prompt("千早爱音穿着羽丘校服")
        invalid = resolver.cached_named_outfits_for_prompt("角色穿着坏映射")

    assert matched is not None
    assert matched.named_outfit_tags == ("haneoka_school_uniform",)
    assert invalid is None


def test_configured_term_mapping_is_returned_as_request_hard_tag() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=Path(directory) / "profiles.json",
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
            config={"danbooru_term_mappings": ["舞会面具=masquerade_mask"]},
        )

        result = resolver.cached_term_mappings_for_prompt("少女戴着舞会面具")

    assert result is not None
    assert result.confirmed_tags == ("masquerade_mask",)


def test_configured_term_mapping_ignores_explicitly_negated_occurrence() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    resolver = DanbooruResolver(
        logger=_Logger(),
        cache={},
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
        config={
            "danbooru_term_mappings": [
                "白丝袜=white_pantyhose",
                "mask=masquerade_mask",
            ]
        },
    )

    assert resolver.cached_term_mappings_for_prompt("少女不要白丝袜") is None
    positive = resolver.cached_term_mappings_for_prompt(
        "不要白丝袜，另一名少女穿白丝袜"
    )
    assert positive is not None
    assert positive.confirmed_tags == ("white_pantyhose",)
    assert resolver.cached_term_mappings_for_prompt("not a mask") is None
    kimono = resolver.cached_term_mappings_for_prompt("kimono mask")
    assert kimono is not None
    assert kimono.confirmed_tags == ("masquerade_mask",)


def test_wardrobe_editor_round_trip_updates_profiles_and_mappings() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
            config={
                "danbooru_named_outfit_mappings": [
                    "月之森校服=tsukinomori_school_uniform"
                ],
                "danbooru_term_mappings": [],
            },
        )
        initial = resolver.wardrobe_snapshot()
        saved = resolver.save_wardrobe(
            {
                "baseRevision": initial["revision"],
                "outfits": [
                    {
                        "key": "amoris",
                        "aliases": ["Amoris"],
                        "sourceTags": ["amoris_(bang_dream!)"],
                        "tags": ["black_corset", "red_shorts"],
                        "qualifier": "default",
                    }
                ],
                "outfitSets": [
                    {
                        "alias": "月之森校服",
                        "tag": "tsukinomori_school_uniform",
                        "origin": "configured",
                        "profileKey": "",
                    }
                ],
                "terms": [{"alias": "舞会面具", "tag": "masquerade_mask"}],
            }
        )
        persisted = json.loads(path.read_text(encoding="utf-8"))

    assert saved["revision"] != initial["revision"]
    assert saved["outfits"][0]["tags"] == ["black_corset", "red_shorts"]
    assert persisted["profiles"]["amoris"]["source_tags"] == [
        "amoris_(bang_dream!)"
    ]
    assert resolver._config["danbooru_term_mappings"] == [
        "舞会面具=masquerade_mask"
    ]


def test_wardrobe_editor_rekeys_manual_casual_profile() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    resolver = DanbooruResolver(
        logger=_Logger(),
        cache={},
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
        config={},
    )
    initial = resolver.wardrobe_snapshot()

    resolver.save_wardrobe(
        {
            "baseRevision": initial["revision"],
            "outfits": [
                {
                    "key": "千早爱音",
                    "aliases": ["千早爱音", "chihaya_anon"],
                    "sourceTags": ["chihaya_anon"],
                    "tags": ["grey_dress", "pinafore_dress", "grey_belt"],
                    "qualifier": "casual",
                }
            ],
            "outfitSets": [],
            "terms": [],
        }
    )

    assert "千早爱音::casual" in resolver._profile_data()["profiles"]
    casual = resolver.cached_outfit_profiles_for_prompt("千早爱音穿官方常服")
    assert casual is not None
    assert casual.outfit_profile_tags == (
        "grey_dress",
        "pinafore_dress",
        "grey_belt",
    )


def test_character_outfit_profile_refreshes_once_per_algorithm_window() -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )
        assert resolver.outfit_source_refresh_needed("amoris") is True
        resolver.remember_outfit_summary(
            "amoris",
            ("amoris_(bang_dream!)",),
            ("black_corset", "red_shorts"),
            {
                "sample_mode": "single_character_anchor",
                "total_posts": 100,
                "selected_posts": 21,
                "focused_posts": 21,
                "anchor_tag": "black_corset",
            },
        )
        assert resolver.outfit_source_refresh_needed("amoris") is False
        saved = json.loads(path.read_text(encoding="utf-8"))

    evidence = saved["profiles"]["amoris"]["evidence"]
    assert evidence["algorithm_version"] == 4
    assert evidence["sample_mode"] == "single_character_anchor"
    assert evidence["created_at"] <= evidence["updated_at"]


def test_wardrobe_snapshot_exposes_stable_profile_creation_time() -> None:
    resolver = DanbooruResolver(
        logger=object(),
        cache={},
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
    )
    resolver._profile_cache_data = {
        "version": 3,
        "profiles": {
            "legacy": {
                "kind": "character_outfit",
                "aliases": ["旧档案"],
                "source_tags": ["legacy_character"],
                "outfit_tags": ["legacy_dress"],
                "evidence": {"updated_at": 100.0},
            },
            "new": {
                "kind": "character_outfit",
                "aliases": ["新档案"],
                "source_tags": ["new_character"],
                "outfit_tags": ["new_dress"],
                "evidence": {"created_at": 200.0, "updated_at": 300.0},
            },
        },
    }

    by_key = {item["key"]: item for item in resolver.wardrobe_snapshot()["outfits"]}

    assert by_key["legacy"]["evidence"]["createdAt"] == 100.0
    assert by_key["new"]["evidence"]["createdAt"] == 200.0


def test_multiple_outfit_sources_are_extracted_and_persisted_separately(
    monkeypatch,
) -> None:
    anchors = (
        SemanticAnchor(
            "amoris",
            "outfit_source",
            "character",
            "Amoris",
            "Amoris costume source",
            ("amoris_(bang_dream!)",),
        ),
        SemanticAnchor(
            "anon_stage",
            "outfit_source",
            "character",
            "千早爱音",
            "stage outfit of Anon Chihaya",
            ("chihaya_anon",),
        ),
    )
    monkeypatch.setattr(
        resolver_module,
        "lookup_semantic_anchors",
        lambda *_args, **_kwargs: SemanticLookupResult(
            confirmed_tags=("amoris_(bang_dream!)", "chihaya_anon"),
            outfit_source_tags=("amoris_(bang_dream!)", "chihaya_anon"),
            anchors=anchors,
            status="resolved",
        ),
    )

    def fake_profile(source_tag, *, outfit_kind, **_kwargs):
        if source_tag == "amoris_(bang_dream!)":
            assert outfit_kind == "default"
            return VariantOutfitProfile(
                tags=("black_corset", "red_shorts"),
                sample_mode="single_character_anchor",
                total_posts=100,
                selected_posts=21,
                focused_posts=21,
                anchor_tag="black_corset",
            )
        assert source_tag == "chihaya_anon"
        assert outfit_kind == "stage"
        return VariantOutfitProfile(
            tags=("blue_jacket", "white_skirt", "black_choker"),
            sample_mode="stage_single_character_anchor",
            total_posts=100,
            selected_posts=26,
            focused_posts=10,
            anchor_tag="blue_jacket",
        )

    monkeypatch.setattr(
        resolver_module, "fetch_variant_outfit_profile", fake_profile
    )

    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )
        monkeypatch.setattr(resolver, "_local_cli_path", lambda: Path("cli.exe"))
        result = asyncio.run(resolver.resolve_semantic_anchors(anchors))
        saved = json.loads(path.read_text(encoding="utf-8"))

    assert result.source_outfit_profiles == (
        (
            "Amoris",
            "amoris_(bang_dream!)",
            ("black_corset", "red_shorts"),
            "default",
        ),
        (
            "千早爱音",
            "chihaya_anon",
            ("blue_jacket", "white_skirt", "black_choker"),
            "stage",
        ),
    )
    assert saved["profiles"]["amoris"]["source_tags"] == [
        "amoris_(bang_dream!)"
    ]
    assert saved["profiles"]["amoris"]["outfit_tags"] == [
        "black_corset",
        "red_shorts",
    ]
    assert saved["profiles"]["千早爱音::stage"]["source_tags"] == [
        "chihaya_anon"
    ]
    assert "haneoka_school_uniform" not in saved["profiles"][
        "千早爱音::stage"
    ]["outfit_tags"]


def test_winter_outfit_source_does_not_overwrite_default_profile(
    monkeypatch,
) -> None:
    anchor = SemanticAnchor(
        "anon_winter",
        "outfit_source",
        "character",
        "千早爱音",
        "千早爱音穿着羽丘冬季校服 / winter school uniform",
        ("chihaya_anon",),
    )
    monkeypatch.setattr(
        resolver_module,
        "lookup_semantic_anchors",
        lambda *_args, **_kwargs: SemanticLookupResult(
            confirmed_tags=("chihaya_anon",),
            outfit_source_tags=("chihaya_anon",),
            anchors=(anchor,),
            status="resolved",
        ),
    )

    def fake_profile(_source_tag, *, outfit_kind, **_kwargs):
        assert outfit_kind == "winter"
        return VariantOutfitProfile(
            tags=("winter_uniform", "blazer", "plaid_skirt"),
            sample_mode="winter_single_character_anchor",
            total_posts=40,
            selected_posts=12,
            focused_posts=8,
            anchor_tag="winter_uniform",
        )

    monkeypatch.setattr(
        resolver_module, "fetch_variant_outfit_profile", fake_profile
    )

    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        resolver = DanbooruResolver(
            logger=_Logger(),
            cache={},
            profile_cache_path=path,
            get_bool=lambda _key, default: default,
            get_int=lambda _key, default: default,
            get_float=lambda _key, default: default,
            get_str=lambda _key, default: default,
        )
        resolver.remember_outfit_summary(
            "千早爱音",
            ("chihaya_anon",),
            ("summer_uniform", "short_sleeves", "plaid_skirt"),
        )
        monkeypatch.setattr(resolver, "_local_cli_path", lambda: Path("cli.exe"))
        asyncio.run(resolver.resolve_semantic_anchors((anchor,)))
        saved = json.loads(path.read_text(encoding="utf-8"))["profiles"]

    assert saved["千早爱音"]["outfit_tags"] == [
        "summer_uniform",
        "short_sleeves",
        "plaid_skirt",
    ]
    assert saved["千早爱音::winter"]["outfit_tags"] == [
        "winter_uniform",
        "blazer",
        "plaid_skirt",
    ]


def test_modified_cached_outfit_uses_effective_tags_without_hard_source_anchor() -> None:
    cached = SemanticLookupResult(
        confirmed_tags=("oblivionis_(bang_dream!)", "bang_dream!"),
        outfit_source_tags=("oblivionis_(bang_dream!)",),
        outfit_profile_tags=("red_shirt", "black_skirt"),
        status="profile_cache",
    )

    context = cached.prompt_context(
        include_outfit_source_anchor=False,
        effective_outfit_tags=("pink_shirt", "black_skirt"),
        removed_outfit_tags=("red_shirt",),
    )

    assert "confirmed hard tags: bang_dream!" in context
    assert "context only; do not emit as a hard tag" in context
    assert "pink_shirt, black_skirt" in context
    assert "never restore: red_shirt" in context
