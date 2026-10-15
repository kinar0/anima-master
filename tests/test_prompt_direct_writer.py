from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from danbooru_resolver import DanbooruResolveOutcome
from danbooru_semantic import SemanticLookupResult
from prompt_pipeline import PromptPipeline


class Logger:
    def info(self, *_args): pass
    def warning(self, *_args): pass
    def debug(self, *_args): pass


class Resolver:
    def required_core_tags_for_prompt(self, _prompt): return ()
    def required_profile_tags_for_prompt(self, _prompt): return ()
    def profile_hints_for_prompt(self, _prompt): return {}
    def semantic_lookup_available(self): return False

    async def resolve_semantic_anchors(self, _anchors):
        raise AssertionError("This test uses local evidence only")

    async def resolve_detailed(self, *, llm_content, **_kwargs):
        return DanbooruResolveOutcome(text=llm_content)

    def cached_outfit_profiles_for_prompt(self, prompt):
        profiles = tuple(row for row in (
            (("角色甲",), "character_a", ("pink_hair", "grey_eyes")),
            (("角色乙",), "character_b", ("blue_hair", "yellow_eyes")),
        ) if row[0][0] in prompt)
        return SemanticLookupResult(
            confirmed_tags=tuple(row[1] for row in profiles),
            character_appearance_profiles=profiles, status="profile_cache",
        )


def semantic_character(name, clothing=None, changes=()):
    return dict(name=name, aliases=[], clothing=clothing, clothing_source=None,
                clothing_changes=[], appearance_changes=list(changes))


def run(prompt, semantic, writer, config=None, resolver=None, original_prompt="", repair=None):
    config = config or {}

    class Context:
        def __init__(self):
            self.outputs = [json.dumps(semantic, ensure_ascii=False), writer]
            if repair is not None:
                self.outputs.append(repair)
            self.calls = []

        async def get_current_chat_provider_id(self, _umo): return "provider"

        async def llm_generate(self, **kwargs):
            self.calls.append(kwargs)
            assert self.outputs, "A valid direct plan must not cause a seven-field or extra review call"
            return type("Response", (), {"completion_text": self.outputs.pop(0)})()

    class Researcher:
        def plan(self, _prompt):
            return type("Plan", (), dict(use_web_search=False, use_deep_thinking=False,
                                        search_reason="", thinking_reason=""))()

    context = Context()
    pipeline = PromptPipeline(
        context=context, config=config, logger=Logger(), danbooru_resolver=resolver or Resolver(),
        researcher=Researcher(), get_bool=lambda k, d: bool(config.get(k, d)),
        get_int=lambda k, d: int(config.get(k, d)), get_float=lambda k, d: float(config.get(k, d)),
        get_str=lambda k, d: str(config.get(k, d)), shorten=lambda t, n=600: t[:n],
    )
    result = asyncio.run(pipeline.build(type("Event", (), {"unified_msg_origin": "session"})(),
                                       prompt, original_user_prompt=original_prompt))
    return result, context.calls


def plan(characters, global_prompt="2girls, garden"):
    return json.dumps(dict(
        composition_analysis=dict(viewpoint="front", layout="separate regions", relations="two instances"),
        global_prompt=global_prompt,
        characters=[dict(name=name, prompt=text, x=x, y=.5, interaction_tags=[],
                         position_reason="separate visible region") for name, text, x in characters],
    ))


def test_restored_scene_responsibilities_stay_inside_the_direct_nai_writer():
    output = plan([("character_a", "girl, pink hair, feeding character_b with a spoon", .3),
                   ("character_b", "girl, blue hair, accepting food from character_a", .7)])
    result, calls = run(
        "-r 角色甲用勺子喂角色乙", {"characters": [
            semantic_character("角色甲"), semantic_character("角色乙")
        ]}, output, {"debug_prompt_enabled": True,
                    "prompt_builder_template": "CUSTOM SEVEN FIELD WRITER {theme}"},
    )
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    writer = calls[1]
    assert '"global_prompt"' in writer["prompt"]
    assert "Before output, review scene completeness" in writer["prompt"]
    assert "tag ownership, positions" in writer["prompt"]
    assert "source#tag" in writer["prompt"] and "target#tag" in writer["prompt"]
    assert "mutual#tag" in writer["prompt"]
    assert "Actual canvas: 1024x1536" in writer["prompt"]
    assert "CUSTOM SEVEN FIELD WRITER" not in writer["prompt"]
    assert "Anima" not in writer["system_prompt"]
    assert "Chinese requests require Chinese" in writer["system_prompt"]
    assert "Chinese prose AND English detail tags" in writer["system_prompt"]
    assert "Never translate the tag layer into Chinese keywords" in writer["system_prompt"]
    assert "Lighting is opt-in" in writer["system_prompt"]
    assert "seven-field" not in writer["prompt"]
    assert writer["prompt"].count("角色甲用勺子喂角色乙") == 1
    # Character count is not a semantic contract or a reason to drop duties.
    # The checks above guard responsibilities and the direct two-call route.
    assert result.summary["prompt_output_mode"] == "nai_instances"
    assert result.summary["prompt_llm_initial_content"] == output
    assert "nai_full_prompt_before_split" not in result.summary


@pytest.mark.parametrize("repair_ok", [True, False])
def test_direct_direction_review_repairs_only_invalid_pairs_once(repair_ok):
    layout = json.loads(plan([("character_a", "girl, hugging character_b", .4),
                              ("character_b", "girl, receiving a hug", .6)]))
    layout["characters"][0]["interaction_tags"] = ["source#hug"]
    invalid = json.dumps(layout)
    layout["characters"][1]["interaction_tags"] = ["target#hug"]
    fixed = json.dumps(layout)
    result, calls = run("-r 角色甲抱住角色乙", {"characters": [
        semantic_character("角色甲"), semantic_character("角色乙")
    ]}, invalid, {"debug_prompt_enabled": True}, repair=fixed if repair_ok else invalid)
    assert len(calls) == 3  # semantic planner, direct writer, one pair repair
    assert result.summary["nai_character_plan_attempt_count"] == 2
    assert result.summary["prompt_llm_attempt_count"] == 2
    assert result.summary["nai_interaction_initial_validation_errors"]
    assert "source#hug" in calls[2]["prompt"] and "target#hug" in calls[2]["prompt"]
    assert "Lighting is opt-in" in calls[2]["system_prompt"]
    assert "affected body part's owner" in calls[2]["system_prompt"]
    assert "Chinese prose AND English detail tags" in calls[2]["system_prompt"]
    assert result.summary["nai_character_plan_initial_raw"] == invalid
    if repair_ok:
        assert result.final_prompt
        assert result.summary["nai_character_plan_accepted_attempt"] == "retry"
        assert result.summary["prompt_llm_initial_content"] == invalid
        assert result.summary["prompt_llm_retry_content"] == fixed
        assert result.summary["nai_interaction_retry_validation_errors"] == []
        assert "target#hug" in result.summary["nai_characters"][1]["prompt"]
    else:
        assert not result.final_prompt
        assert result.summary["skipped_reason"] == "nai_character_plan_failed"
        assert "nai_character_plan_invalid_interaction_pairs" in result.summary["llm_error"]
        assert "nai_characters" not in result.summary


@pytest.mark.parametrize("repair_needed", [False, True])
@pytest.mark.parametrize("outfit_tag", ["haneoka_school_uniform", "aurora_academy_attire"])
def test_nai_named_clothes_stay_with_wearers_including_after_pair_repair(outfit_tag, repair_needed):
    class NamedClothing(Resolver):
        def cached_outfit_profiles_for_prompt(self, prompt):
            from dataclasses import replace
            return replace(super().cached_outfit_profiles_for_prompt(prompt),
                           named_outfit_tags=(outfit_tag,))

    layout = json.loads(plan([
        ("character_a", f"girl, {outfit_tag}, 张嘴接受角色乙喂来的食物", .3),
        ("character_b", f"girl, {outfit_tag}, 用自己手里的勺子喂角色甲", .7),
    ], f"2girls, {outfit_tag.replace('_', ' ')}, bed, table, unclassified_glow"))
    layout["characters"][1]["interaction_tags"] = ["source#feeding"]
    if not repair_needed:
        layout["characters"][0]["interaction_tags"] = ["target#feeding"]
    initial = json.dumps(layout, ensure_ascii=False)
    layout["characters"][0]["interaction_tags"] = ["target#feeding"]
    fixed = json.dumps(layout, ensure_ascii=False)
    result, calls = run(
        "-r 角色甲和角色乙都穿命名校服，角色乙用勺子喂角色甲", {"characters": [
            semantic_character("角色甲", "命名校服"), semantic_character("角色乙", "命名校服")
        ]}, initial, {"debug_prompt_enabled": True}, resolver=NamedClothing(),
        repair=fixed if repair_needed else None,
    )
    assert result.final_prompt, result.summary
    assert len(calls) == (3 if repair_needed else 2)
    assert outfit_tag.replace('_', ' ') not in result.final_prompt
    assert result.summary["nai_dropped_global_character_tags"] == [outfit_tag.replace('_', ' ')]
    assert all(term in result.final_prompt for term in ("bed", "table", "unclassified glow"))
    recipient, actor = result.summary["nai_characters"]
    assert outfit_tag in recipient["prompt"] and outfit_tag in actor["prompt"]
    assert "张嘴接受角色乙喂来的食物" in recipient["prompt"]
    assert "用自己手里的勺子喂角色甲" in actor["prompt"]
    assert "target#feeding" in recipient["prompt"] and "source#feeding" in actor["prompt"]
    if not repair_needed:
        from agent_tools.comfyui_workflows import custom_t2i_workflow
        graph = custom_t2i_workflow(
            {"custom_workflow_path": "nai_api.json"}, result.final_prompt, "blurry",
            1024, 1536, 28, 6, 123, nai_characters=result.summary["nai_characters"],
        )
        selector = graph[graph["1"]["inputs"]["characterPrompts"][0]]["inputs"]
        assert graph[selector["character1"][0]]["inputs"]["comfyui_prompt"] == recipient["prompt"]
        assert graph[selector["character2"][0]]["inputs"]["comfyui_prompt"] == actor["prompt"]


def test_nai_shared_named_outfit_anchor_is_not_filtered_from_second_wearer():
    from danbooru_semantic import SemanticAnchor

    class SharedOutfit(Resolver):
        def cached_named_outfits_for_prompt(self, prompt):
            return SemanticLookupResult(
                named_outfit_tags=("aurora_academy_attire",),
                anchors=(SemanticAnchor("uniform", "outfit", "outfit", "命名校服", "",
                                        ("aurora_academy_attire",)),),
            )

    result, calls = run(
        "-r 角色甲穿命名校服，角色乙也穿命名校服", {"characters": [
            semantic_character("角色甲", "命名校服"), semantic_character("角色乙", "命名校服"),
        ]}, plan([("character_a", "girl, aurora_academy_attire, 穿命名校服站立", .3),
                  ("character_b", "girl, aurora_academy_attire, 也穿命名校服站立", .7)]),
        resolver=SharedOutfit(),
    )
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    assert all("aurora_academy_attire" in row["prompt"] for row in result.summary["nai_characters"])


def test_nai_ordinary_shirt_survives_shared_uniform_components_and_old_cache():
    from dataclasses import replace
    from danbooru_semantic import SemanticAnchor

    class MixedOutfits(Resolver):
        def cached_outfit_profiles_for_prompt(self, prompt):
            return replace(super().cached_outfit_profiles_for_prompt(prompt),
                           source_outfit_profiles=(("角色甲", "character_a",
                                                    ("old_school_uniform", "white_shirt"), "default"),))

        def cached_named_outfits_for_prompt(self, prompt):
            return SemanticLookupResult(
                named_outfit_tags=("aurora_academy_attire",),
                anchors=(SemanticAnchor("uniform", "outfit", "outfit", "命名校服", "",
                                        ("aurora_academy_attire",)),),
                anchor_outfit_profiles=(("uniform", "aurora_academy_attire", ("white_shirt",), "default"),),
            )

    result, calls = run(
        "-r 角色甲穿白衬衫，角色乙穿命名校服", {"characters": [
            semantic_character("角色甲", "白衬衫"), semantic_character("角色乙", "命名校服"),
        ]}, plan([("character_a", "girl, white_shirt, standing, 角色甲穿着白衬衫站立。", .3),
                  ("character_b", "girl, aurora_academy_attire, white_shirt, standing, 角色乙穿着命名校服站立。", .7)]),
        resolver=MixedOutfits(),
    )
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    a, b = result.summary["nai_characters"]
    assert "white_shirt" in a["prompt"] and "白衬衫站立。" in a["prompt"]
    assert "aurora_academy_attire" not in a["prompt"] and "old_school_uniform" not in a["prompt"]
    assert "aurora_academy_attire" in b["prompt"] and "white_shirt" in b["prompt"]


def test_nai_classified_clothes_are_not_assigned_to_a_wearer_by_host():
    class NamedClothing(Resolver):
        def cached_outfit_profiles_for_prompt(self, prompt):
            from dataclasses import replace
            return replace(super().cached_outfit_profiles_for_prompt(prompt),
                           named_outfit_tags=("unfamiliar_named_costume",))

    result, calls = run(
        "-r 角色甲和角色乙站在陈列命名服装的房间", {"characters": [
            semantic_character("角色甲"), semantic_character("角色乙")
        ]}, plan([("character_a", "girl, standing, 站在房间左边", .3),
                  ("character_b", "girl, sitting, 坐在房间右边", .7)],
                 "2girls, room, unfamiliar_named_costume"), resolver=NamedClothing(),
    )
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    assert "unfamiliar named costume" not in result.final_prompt
    assert all("unfamiliar_named_costume" not in row["prompt"] for row in result.summary["nai_characters"])
    assert "room" in result.final_prompt


def test_two_stages_keep_wearer_appearance_omission_and_distinct_clothes():
    request = "-r 角色甲穿白裙，上半张脸黑着(不画眼睛)，角色乙穿蓝裙站在旁边"
    semantic = {"characters": [
        semantic_character("角色甲", "白裙", [dict(dimension="eye_color", operation="omit",
                                             source_text="上半张脸黑着(不画眼睛)")]),
        semantic_character("角色乙", "蓝裙"),
    ]}
    output = plan([
        ("character_a", "girl, pink hair, grey eyes, no visible eyes, wearing a white dress, waving", .3),
        ("character_b", "girl, blue hair, yellow eyes, wearing a blue dress, sitting", .7),
    ])
    result, calls = run(request, semantic, output, {"keyword_prompt_rules": ["白裙 => preserve white dress"]})
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    assert '"characters"' in calls[0]["prompt"]
    assert '"global_prompt"' in calls[1]["prompt"]
    assert "Characters, Identity, Details and Nltags" not in calls[1]["prompt"]
    assert "preserve white dress" in calls[1]["prompt"]
    assert "Actual canvas: 1024x1536" in calls[1]["prompt"]
    first, second = result.summary["nai_characters"]
    assert "grey eyes" not in first["prompt"] and "grey_eyes" not in first["prompt"]
    assert "no visible eyes" in first["prompt"] and "white dress" in first["prompt"]
    assert "yellow eyes" in second["prompt"].replace("_", " ") and "blue dress" in second["prompt"]
    assert "sitting" in second["prompt"] and "waving" in first["prompt"]
    assert "white background" not in result.final_prompt


@pytest.mark.parametrize("user_text,allowed", [
    ("角色甲是扶她，站立", True), ("futanari 角色甲站立", True),
    ("角色甲不是扶她，站立", False), ("角色甲站立，不要futanari", False),
    ("角色甲穿透明外套，魅惑表情", False),
])
def test_nai_futa_guidance_and_authority_only_follow_positive_original_request(user_text, allowed):
    from prompt_templates import NAI_FUTA_INSTANCE_POLICY
    result, calls = run(
        "-r " + user_text, {"characters": [semantic_character("角色甲")]},
        plan([("character_a", "girl, futanari, pink hair, standing", .5)], "1girl, white background"),
    )
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    assert (NAI_FUTA_INSTANCE_POLICY in calls[1]["prompt"]) is allowed
    authority = result.summary["sexual_trait_authority"]
    assert authority["futa_allowed"] is allowed
    assert ("futanari" in result.summary["nai_characters"][0]["prompt"]) is allowed
    if not allowed:
        assert authority["removed"]
    if allowed:
        assert "1girl, 1boy" in calls[1]["prompt"]
        assert "不要用 `futa with female / futa with male` 代替 NAI 全局人数" in calls[1]["prompt"]


def test_reference_futa_traits_do_not_authorize_target_traits_or_futa_policy():
    from prompt_templates import NAI_FUTA_INSTANCE_POLICY
    original = "-r 角色甲参考图片的服装站立"
    expanded = original + "。参考图视觉反推 tags：futanari"
    result, calls = run(
        expanded, {"characters": [semantic_character("角色甲")]},
        plan([("character_a", "girl, futanari, pink hair, standing", .5)], "1girl, white background"),
        original_prompt=original,
    )
    assert result.final_prompt, result.summary
    assert NAI_FUTA_INSTANCE_POLICY not in calls[1]["prompt"]
    assert "Original user request (background and trait authority):" in calls[1]["prompt"]
    assert original.removeprefix("-r ") in calls[1]["prompt"]
    assert "参考图视觉反推 tags：futanari" in calls[1]["prompt"]
    assert result.summary["sexual_trait_authority"]["futa_allowed"] is False
    assert "futanari" not in result.summary["nai_characters"][0]["prompt"]


@pytest.mark.parametrize("config,user_text,enabled", [
    ({}, "角色甲穿透明外套，魅惑表情", True),
    ({"sensual_mode_enabled": False}, "角色甲穿透明外套，魅惑表情", False),
    ({"sensual_mode_markers": ["薄绸"]}, "角色甲穿薄绸衣服", True),
    ({"sensual_mode_markers": ["薄绸"]}, "角色甲穿透明外套", False),
    ({}, "角色甲是扶她，站立", False),
])
def test_nai_and_seven_field_share_sensual_policy_and_config(config, user_text, enabled):
    from prompt_templates import SENSUAL_EXPRESSION_POLICY, build_llm_prompt
    from nai_character_mode import NAI_WARDROBE_POLICY
    result, calls = run(
        "-r " + user_text, {"characters": [semantic_character("角色甲")]},
        plan([("character_a", "girl, standing", .5)], "1girl, white background"), config,
    )
    assert result.final_prompt, result.summary
    prompt = calls[1]["prompt"]
    ordinary = build_llm_prompt(user_text, sensual_mode=enabled)
    assert (SENSUAL_EXPRESSION_POLICY in ordinary) is enabled
    assert (SENSUAL_EXPRESSION_POLICY in prompt) is enabled
    assert result.summary["sensual_mode"] is enabled
    assert prompt.count(SENSUAL_EXPRESSION_POLICY) == int(enabled)
    assert prompt.count(NAI_WARDROBE_POLICY) == 1
    assert prompt.endswith(NAI_WARDROBE_POLICY)
    assert "Saved defaults are soft references" in prompt
    assert "selected named outfit or" in prompt
    assert "Unresolved explicit clothing does not permit default fallback" in prompt
    if enabled:
        assert prompt.index(SENSUAL_EXPRESSION_POLICY) < prompt.index(NAI_WARDROBE_POLICY)


@pytest.mark.parametrize("trait", ["futanari", "扶她", "阴茎", "penis"])
def test_unrequested_traits_in_english_or_chinese_do_not_bypass_nai_authority(trait):
    result, calls = run(
        "-r 角色甲穿透明外套，魅惑表情", {"characters": [semantic_character("角色甲")]},
        plan([("character_a", f"girl, {trait}, pink hair, standing", .5)], "1girl, white background"),
    )
    assert result.final_prompt, result.summary
    assert trait not in result.summary["nai_characters"][0]["prompt"]
    assert result.summary["sexual_trait_authority"]["removed"]


def test_partial_planner_keeps_all_local_profiles_without_fixed_character_hints():
    class ThreeProfiles(Resolver):
        def cached_outfit_profiles_for_prompt(self, _prompt):
            rows = (("角色甲", "character_a", "pink_hair", "grey_eyes", "white_shirt"),
                    ("角色乙", "character_b", "blue_hair", "yellow_eyes", "blue_dress"),
                    ("角色丙", "character_c", "green_hair", "green_eyes", "black_coat"))
            return SemanticLookupResult(
                confirmed_tags=tuple(row[1] for row in rows),
                source_outfit_profiles=tuple((name, tag, (clothes,), "default")
                                            for name, tag, hair, eyes, clothes in rows),
                character_appearance_profiles=tuple(((name,), tag, (hair, eyes))
                                                   for name, tag, hair, eyes, clothes in rows),
                status="profile_cache",
            )

    result, calls = run(
        "-r 角色甲应付着角色乙和角色丙的挑逗",
        {"characters": [semantic_character("角色甲")]},
        plan([("character_a", "1girl, pink hair, grey eyes, waving", .5),
              ("character_b", "1girl, blonde hair, red eyes, leaning", .2),
              ("character_c", "1girl, purple hair, yellow eyes, standing", .8)], "3girls, temple"),
        {"fixed_characters": [], "debug_prompt_enabled": True}, resolver=ThreeProfiles(),
    )
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    prompt = calls[1]["prompt"]
    assert "角色乙 => character_b; stable appearance: blue_hair, yellow_eyes" in prompt
    assert "角色丙 => character_c; stable appearance: green_hair, green_eyes" in prompt
    assert "角色乙: default reference = blue_dress" in prompt
    assert "角色丙: default reference = black_coat" in prompt
    assert "not a mandatory visible roster" in prompt
    assert "Other confirmed non-wardrobe hard tags: character_b" not in prompt
    assert len(result.summary["nai_bound_local_profile_references"]) == 2
    a, b, c = result.summary["nai_characters"]
    assert "blue_hair" in b["prompt"] and "yellow_eyes" in b["prompt"]
    assert "green_hair" in c["prompt"] and "green_eyes" in c["prompt"]
    assert "blonde hair" not in b["prompt"] and "purple hair" not in c["prompt"]
    assert all("1girl" not in item["prompt"] for item in (a, b, c))


def test_unplanned_reference_is_optional_and_explicit_clothing_never_loads_defaults():
    class CachedClothing(Resolver):
        def cached_outfit_profiles_for_prompt(self, prompt):
            return SemanticLookupResult(
                confirmed_tags=("character_a", "character_b"),
                character_appearance_profiles=((("角色甲",), "character_a", ("pink_hair",)),
                                               (("角色乙",), "character_b", ("blue_hair",))),
                source_outfit_profiles=(("角色乙", "character_b", ("school_uniform", "grey_jacket"), "default"),),
                status="profile_cache",
            )

    result, calls = run(
        "-r 角色甲穿角色乙的衣服，角色乙只作参考不出现在画面中",
        {"characters": [semantic_character("角色甲")]},
        plan([("character_a", "girl, wearing a new ceremonial robe", .5)], "1girl, white background"),
        resolver=CachedClothing(),
    )
    assert result.final_prompt, result.summary
    assert len(result.summary["nai_characters"]) == 1
    assert result.summary["nai_bound_local_profile_references"] == []
    assert result.summary["nai_local_profile_references"] == []
    assert "角色乙: default reference =" not in calls[1]["prompt"]
    assert "new ceremonial robe" in result.summary["nai_characters"][0]["prompt"]


def test_unplanned_visible_wearer_keeps_new_garment_without_default_restore():
    result, calls = run(
        "-r 角色甲挥手，角色乙穿星光斗篷站立",
        {"characters": [semantic_character("角色甲")]},
        plan([("character_a", "girl, waving", .25),
              ("character_b", "girl, blue hair, wearing a sparkling cloak, standing", .75)]),
    )
    assert result.final_prompt, result.summary
    assert "角色乙: explicit clothing unresolved" in calls[1]["prompt"]
    assert "sparkling cloak" in result.summary["nai_characters"][1]["prompt"]


def test_unplanned_wardrobe_only_reference_keeps_its_own_components():
    class WardrobeOnly(Resolver):
        def cached_outfit_profiles_for_prompt(self, _prompt):
            return SemanticLookupResult(
                confirmed_tags=("character_a", "character_b"),
                source_outfit_profiles=(("角色甲", "character_a", ("white_shirt",), "default"),
                                       ("角色乙", "character_b", ("blue_dress",), "default")),
                outfit_profile_tags=("white_shirt", "blue_dress"), status="profile_cache",
            )
    result, calls = run(
        "-r 角色甲和角色乙站立", {"characters": [semantic_character("角色甲")]},
        plan([("character_a", "girl, standing", .25), ("character_b", "girl, standing", .75)]),
        resolver=WardrobeOnly(),
    )
    assert result.final_prompt, result.summary
    assert "角色乙: default reference = blue_dress." in calls[1]["prompt"]
    assert "角色乙: default reference = white_shirt" not in calls[1]["prompt"]
    assert "blue_dress" not in result.summary["nai_characters"][1]["prompt"]


def test_nai_references_do_not_promote_generic_named_outfit_or_ambiguous_alias():
    from prompt_pipeline import nai_local_profile_references
    result = SemanticLookupResult(
        confirmed_tags=("outfit_tag", "character_a", "character_b"),
        named_outfit_tags=("outfit_tag",),
        source_outfit_profiles=(("命名制服", "outfit_tag", ("shirt",), "default"),),
        character_appearance_profiles=((("同名角色",), "character_a", ("pink_hair",)),
                                       (("同名角色",), "character_b", ("blue_hair",))),
    )
    assert nai_local_profile_references(
        "命名制服和同名角色", (), (), result, implicit_mode="default_profile",
    ) == ()


def test_unselected_local_reference_does_not_claim_clothing_from_visible_wearer():
    class CachedClothing(Resolver):
        def cached_outfit_profiles_for_prompt(self, prompt):
            return SemanticLookupResult(
                confirmed_tags=("character_a", "character_b"),
                character_appearance_profiles=((("角色甲",), "character_a", ("pink_hair",)),
                                               (("角色乙",), "character_b", ("blue_hair",))),
                source_outfit_profiles=(("角色乙", "character_b", ("blue_dress",), "default"),),
                status="profile_cache",
            )
    result, calls = run(
        "-r 角色甲站立，角色乙的名字写在牌子上", {"characters": [semantic_character("角色甲")]},
        plan([("character_a", "girl, blue_dress, standing", .5)], "1girl, white background"),
        resolver=CachedClothing(),
    )
    assert result.final_prompt, result.summary
    assert len(result.summary["nai_local_profile_references"]) == 1
    assert result.summary["nai_bound_local_profile_references"] == []
    assert "blue_dress" in result.summary["nai_characters"][0]["prompt"]


def test_same_identity_two_panels_keeps_distinct_actions_clothes_and_positions():
    request = "-r 左格角色甲穿白裙挥手，右格角色甲穿蓝裙坐下"
    semantic = {"characters": [semantic_character("角色甲", "白裙"), semantic_character("角色甲", "蓝裙")]}
    output = plan([
        ("character_a", "girl, wearing a white dress, waving", .25),
        ("character_a", "girl, wearing a blue dress, sitting", .75),
    ], "2girls, two panels, simple background, white background")
    result, calls = run(request, semantic, output)
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    first, second = result.summary["nai_characters"]
    assert first["name"] == second["name"] == "character_a"
    assert first["x"] == .25 and second["x"] == .75
    assert "white dress" in first["prompt"] and "blue dress" not in first["prompt"]
    assert "blue dress" in second["prompt"] and "white dress" not in second["prompt"]
    assert "waving" in first["prompt"] and "sitting" in second["prompt"]


def test_repeated_identity_aliases_share_the_same_baseline_without_extra_binding():
    class AliasResolver(Resolver):
        def cached_outfit_profiles_for_prompt(self, _prompt):
            return SemanticLookupResult(
                confirmed_tags=("character_a",),
                character_appearance_profiles=((("角色甲", "甲的别名"), "character_a", ("pink_hair", "grey_eyes")),),
                status="profile_cache",
            )

    result, calls = run(
        "-r 左格角色甲站立，右格甲的别名坐下",
        {"characters": [semantic_character("角色甲"), semantic_character("甲的别名")]},
        plan([("character_a", "girl, standing", .25), ("character_a", "girl, sitting", .75)]),
        resolver=AliasResolver(),
    )
    assert result.final_prompt, result.summary
    assert len(calls) == 2 and len(result.summary["nai_characters"]) == 2
    for character in result.summary["nai_characters"]:
        assert "pink_hair" not in character["prompt"] and "grey_eyes" not in character["prompt"]


@pytest.mark.parametrize("source_mode", ["valid", "missing", "invented"])
def test_repeated_identity_omission_applies_to_quoted_instance_only(source_mode):
    left = "左格角色甲上半张脸黑着(不画眼睛)"
    right = "右格角色甲微笑站立"
    semantic = {"characters": [
        semantic_character("角色甲", changes=[dict(dimension="eye_color", operation="omit",
                                              source_text="上半张脸黑着(不画眼睛)")]),
        semantic_character("角色甲"),
    ]}
    output = json.loads(plan([
        ("character_a", "girl, grey eyes, no visible eyes, upper face shadowed", .25),
        ("character_a", "girl, grey eyes, smiling, standing", .75),
    ], "2girls, two panels, white background"))
    if source_mode != "missing":
        output["characters"][0]["source_text"] = left if source_mode == "valid" else "虚构的左格要求"
        output["characters"][1]["source_text"] = right
    result, calls = run(f"-r {left}，{right}", semantic, json.dumps(output))
    assert len(calls) == 2
    if source_mode != "valid":
        assert result.final_prompt == ""
        assert result.summary["llm_error"] == ("nai_character_plan_missing_instance_source"
                                              if source_mode == "missing" else "nai_character_plan_ungrounded_instance_source")
        return
    assert result.final_prompt, result.summary
    first, second = result.summary["nai_characters"]
    assert "grey eyes" not in first["prompt"] and "grey_eyes" not in first["prompt"]
    assert "no visible eyes" in first["prompt"]
    assert "grey eyes" in second["prompt"].replace("_", " ") and "smiling" in second["prompt"]


def test_shared_and_local_clauses_can_both_bind_repeated_instances():
    left, right, shared = "左格角色甲站立", "右格角色甲坐下", "两格都不画眼睛"
    change = dict(dimension="eye_color", operation="omit", source_text=shared)
    semantic = {"characters": [semantic_character("角色甲", changes=[change]),
                               semantic_character("角色甲", changes=[change])]}
    output = json.loads(plan([
        ("character_a", "girl, grey eyes, no visible eyes, standing", .25),
        ("character_a", "girl, grey eyes, no visible eyes, sitting", .75),
    ], "2girls, two panels, white background"))
    for item, clause in zip(output["characters"], (left, right)):
        item["source_text"] = [clause, shared]
    result, calls = run(f"-r {left}，{right}。{shared}", semantic, json.dumps(output))
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    for character in result.summary["nai_characters"]:
        assert "grey eyes" not in character["prompt"] and "grey_eyes" not in character["prompt"]
        assert "no visible eyes" in character["prompt"]


@pytest.mark.parametrize("source", ["角色乙的挑逗", ["角色乙", "角色乙的挑逗"]])
@pytest.mark.parametrize("authority", ["unplanned", "unique_change", "repeated_baseline"])
def test_optional_paraphrased_source_does_not_block_or_grant_authority(source, authority):
    request = "-r 角色甲应付着角色乙和同伴的挑逗"
    semantic = {"characters": [semantic_character("角色甲")]}
    name = "character_b"
    if authority == "unique_change":
        request += "，角色乙不画眼睛"
        semantic["characters"].append(semantic_character("角色乙", changes=[
            dict(dimension="eye_color", operation="omit", source_text="不画眼睛")]))
    elif authority == "repeated_baseline":
        semantic["characters"].append(semantic_character("角色乙"))
        request += "，右格角色乙站立"
    output = json.loads(plan([
        ("character_a", "girl, standing", .25),
        (name, "girl, yellow eyes, no visible eyes, sitting", .75),
    ]))
    output["characters"][1]["source_text"] = source
    if authority == "repeated_baseline":
        output["characters"].append(dict(output["characters"][1], x=.9, prompt="girl, yellow eyes, standing"))
    result, calls = run(request, semantic, json.dumps(output))
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    dropped = result.summary["nai_dropped_instance_sources"]
    assert dropped[0]["name"] == name and dropped[0]["source_text"] == source
    assert all("source_text" not in character for character in result.summary["nai_characters"][1:])
    character = result.summary["nai_characters"][1]
    assert "sitting" in character["prompt"]
    if authority == "unique_change":
        assert "yellow eyes" not in character["prompt"] and "yellow_eyes" not in character["prompt"]
    elif authority == "repeated_baseline":
        assert "yellow eyes" in character["prompt"].replace("_", " ")


def test_anonymous_english_instance_name_can_bind_to_chinese_intent_anchor():
    output = json.loads(plan([("girl", "girl, grey eyes, no visible eyes, standing", .5)], "1girl, white background"))
    output["characters"][0]["anchor_id"] = "target_1"
    result, calls = run(
        "-r 一个女孩，不画眼睛",
        {"characters": [semantic_character("一个女孩", changes=[dict(dimension="eye_color", operation="omit", source_text="不画眼睛")])]},
        json.dumps(output),
    )
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    prompt = result.summary["nai_characters"][0]["prompt"]
    assert "grey eyes" not in prompt and "no visible eyes" in prompt


@pytest.mark.parametrize("donor_visible", [False, True, "also_visible"])
def test_clothing_donor_is_not_a_visible_character_or_target_appearance(donor_visible):
    class DonorResolver(Resolver):
        def cached_outfit_profiles_for_prompt(self, _prompt):
            return SemanticLookupResult(
                confirmed_tags=("character_a", "donor_character"),
                source_outfit_profiles=(("来源角色", "donor_character", ("black_dress",), "default"),),
                character_appearance_profiles=(
                    (("角色甲",), "character_a", ("pink_hair", "grey_eyes")),
                    (("来源角色",), "donor_character", ("blonde_hair", "green_eyes")),
                ), status="profile_cache",
            )

    semantic = {
        "anchors": [dict(id="a", role="target_character", group="character", source_text="角色甲", candidates=["character_a"]),
                    dict(id="source", role="outfit_source", group="character", source_text="来源角色", candidates=["donor_character"])],
        "character_plans": [dict(target_anchor_id="a", wardrobe=dict(kind="outfit_source", anchor_id="source"), directives=[])],
    }
    request = "-r 角色甲cosplay来源角色"
    if donor_visible == "also_visible":
        request += "，来源角色站在旁边"
        semantic["anchors"].append(dict(id="b", role="target_character", group="character", source_text="来源角色", candidates=["donor_character"]))
        semantic["character_plans"].append(dict(target_anchor_id="b", wardrobe=dict(kind="none"), directives=[]))
    characters = [("character_a", "girl, blonde hair, green eyes, wearing a black dress", .3)]
    if donor_visible:
        characters.append(("donor_character", "girl, blonde hair, green eyes, sitting", .7))
    result, calls = run(request, semantic,
                        plan(characters, ("2girls" if donor_visible else "1girl") + ", white background"), resolver=DonorResolver())
    assert len(calls) == 2
    if donor_visible is True:
        assert result.final_prompt == ""
        assert result.summary["llm_error"] == "nai_character_plan_visible_outfit_source"
    else:
        assert result.final_prompt, result.summary
        prompt = result.summary["nai_characters"][0]["prompt"]
        assert "blonde hair" not in prompt and "green eyes" not in prompt
        assert "pink_hair" in prompt and "grey_eyes" in prompt
        assert "black dress" in prompt
        assert r"donor character \(cosplay\)" in prompt


def test_noun_mapping_cannot_authorize_a_stale_default_wardrobe():
    from prompt_pipeline import CharacterEffectiveOutfit, build_wardrobe_authority
    from outfit_transfer import EffectiveOutfitPlan

    wearer = CharacterEffectiveOutfit(
        target_anchor_id="a", target_source_text="角色甲", target_candidates=("character_a",),
        wardrobe_kind="creative_fallback", wardrobe_anchor_id="", wardrobe_tag="",
        appearance_tags=(), effective=EffectiveOutfitPlan(),
    )
    evidence = SemanticLookupResult(outfit_profile_tags=("school_uniform",))
    authority = build_wardrobe_authority((wearer,), evidence, (), explicit_term_tags=("school_uniform",))
    assert "school_uniform" not in authority.explicit_tags
    assert authority.filter_tags(("school_uniform", "wedding_dress")) == ("wedding_dress",)


def test_direct_writer_keeps_request_artist_group_and_exact_nested_weight():
    result, calls = run(
        "-r -s2 一个女孩", {"characters": [semantic_character("一个女孩")]},
        plan([("girl", "girl, standing", .5)], "1girl, white background"),
        {"artist_presets": ["A=(first_artist:0.8)", "B=(second_artist_(scope):1.1)"],
         "active_artist_preset": "A", "default_artist_tags": "", "quality_prefix": "masterpiece, best quality"},
    )
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    assert "(second_artist_(scope):1.1)" in result.final_prompt
    assert "first_artist" not in result.final_prompt and "-s2" not in result.final_prompt
    assert "masterpiece" in result.final_prompt and "best quality" in result.final_prompt


@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("invented_source", [False, True])
def test_one_wearer_removing_shared_profile_component_does_not_remove_it_from_another(repeat, invented_source):
    class SharedProfileResolver(Resolver):
        def cached_outfit_profiles_for_prompt(self, prompt):
            return SemanticLookupResult(
                confirmed_tags=("character_a", "character_b"),
                outfit_profile_tags=("white_shirt", "masquerade_mask"),
                character_appearance_profiles=((("角色甲",), "character_a", ()), (("角色乙",), "character_b", ())),
                source_outfit_profiles=tuple((source, tag, ("white_shirt", "masquerade_mask"), "default")
                                            for source, tag in (("角色甲", "character_a"), ("角色乙", "character_b"))),
                status="profile_cache",
            )

    first = semantic_character("角色甲", "默认服装")
    first["clothing_changes"] = [dict(operation="remove", slots=["face_accessory.mask"], source_text="不戴舞会面具")]
    second_name = "角色甲" if repeat else "角色乙"
    semantic = {"characters": [first, semantic_character(second_name, "默认服装")]}
    output = json.loads(plan([
        ("character_a", "girl, white_shirt, masquerade_mask, waving", .3),
        ("character_a" if repeat else "character_b", "girl, white_shirt, masquerade_mask, sitting", .7),
    ]))
    left = "左格角色甲穿默认服装但不戴舞会面具"
    right = f"右格{second_name}穿默认服装"
    if repeat:
        for item, source, anchor in zip(output["characters"], (left, right), ("target_1", "target_2")):
            item.update(source_text=source, anchor_id=anchor)
    if invented_source:
        output["characters"][0]["source_text"] = "左格角色甲穿礼服但不戴面具"
    result, calls = run(f"-r {left}，{right}", semantic, json.dumps(output),
                        resolver=SharedProfileResolver())
    assert len(calls) == 2
    if repeat and invented_source:
        assert not result.final_prompt
        assert result.summary["llm_error"] == "nai_character_plan_ungrounded_instance_source"
        return
    assert result.final_prompt, result.summary
    if invented_source:
        assert result.summary["nai_dropped_instance_sources"]
    first, second = result.summary["nai_characters"]
    assert "masquerade_mask" not in first["prompt"], result.summary["semantic_character_outfits"]
    assert "masquerade_mask" in second["prompt"]
    assert "white_shirt" in first["prompt"] and "white_shirt" in second["prompt"]


@pytest.mark.parametrize("user_text,writer_tags,marker", [
    ("一个女孩", "simple background, white background", ""),
    ("一个女孩", "simple background, white background", "background_mode_explicit_scene"),
    ("一个女孩毕业时挥手", "school gate, graduation ceremony", "background_mode_default_portrait"),
    ("一个女孩毕业时挥手", "school gate, graduation ceremony", ""),
    ("一个女孩在海边", "beach, sunset", "background_mode_default_portrait"),
])
def test_ordinary_host_preserves_writer_background_even_if_marker_is_missing_or_conflicts(user_text, writer_tags, marker):
    writer = ("{Count: 1girl}\n{Characters:}\n{Copyright:}\n{Identity:}\n"
              "{Details: standing}\n{Tags: " + writer_tags + "}\n{Nltags:}" + marker)
    result, calls = run(user_text, {"characters": [semantic_character("一个女孩")]}, writer)
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    for tag in writer_tags.split(", "):
        assert tag in result.final_prompt
    assert result.summary["background_mode_source"] == ("llm_marker" if marker else "writer")
    assert "background_mode_" not in result.final_prompt
    if "white background" not in writer_tags:
        assert "white background" not in result.final_prompt
        assert "simple background" not in result.final_prompt
