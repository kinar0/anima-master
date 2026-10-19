from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from danbooru_semantic import (
    SemanticAppearanceChange, SemanticLookupResult, parse_semantic_appearance_changes,
    parse_semantic_plan, parse_semantic_character_plans, semantic_plan_validation_issues,
)
from outfit_transfer import EffectiveOutfitPlan, outfit_tag_slot
from prompt_background import framing_hidden_outfit_slots
from prompt_pipeline import (
    CharacterEffectiveOutfit, WardrobeAuthority, _without_omitted_appearance,
    appearance_dimensions, appearance_reference_evidence, bind_appearance_accessory_permissions,
    build_wardrobe_authority, filter_shared_appearance_tags, merge_authoritative_identity_block,
    filter_profile_tags_for_composition,
    _nai_instance_evidence,
    apply_framing_to_character_outfits, fallback_missing_unspecified_profiles,
    non_wardrobe_confirmed_tags,
    reference_appearance_issues,
)
from prompt_templates import build_llm_prompt
from prompt_visibility import VISUAL_FOCUS_POLICY
from test_prompt_direct_writer import Resolver, run, semantic_character


def local_plan(name="角色甲", identity="character_a"):
    return CharacterEffectiveOutfit(
        target_anchor_id=identity, target_source_text=name, target_candidates=(identity,),
        wardrobe_kind="default_reference", wardrobe_anchor_id="", wardrobe_tag="",
        appearance_tags=("long_hair", "pink_hair", "grey_eyes", "straight_hair"),
        effective=EffectiveOutfitPlan(subject=name), resolution_state="unspecified",
    )


@pytest.mark.parametrize("phrase", [
    "long pink hair styled in two side up twin tails",
    "half-up twin-tail-like silhouette",
    "twin-tail-like half-updo",
    "ponytail", "twintails", "two_side_up", "twin buns", "curly hair", "sidelocks",
])
def test_hair_shapes_are_not_bodily_tails(phrase):
    assert "hair_style" in appearance_dimensions(phrase)
    assert "tail" not in appearance_dimensions(phrase)
    text = f"character_a has {phrase}, grey eyes"
    merged = merge_authoritative_identity_block(
        "character_a", text, ("long_hair", "pink_hair", "grey_eyes"), frozenset(),
        advisory_writer_dimensions=frozenset({"hair_style"}),
        replacement_writer_dimensions=frozenset({"hair_style"}),
    )
    assert phrase in merged
    assert phrase in _without_omitted_appearance(text, frozenset({"tail"}))


@pytest.mark.parametrize("phrase", ["fox tail", "two tails", "twin tails and pink hair", "blue tail", "尾巴"])
def test_real_tail_permission_is_not_opened_by_hair_fix(phrase):
    assert "tail" in appearance_dimensions(phrase)
    merged = merge_authoritative_identity_block(
        "character_a", f"character_a has {phrase}", ("pink_hair",), frozenset(),
        advisory_writer_dimensions=frozenset({"hair_style"}),
    )
    assert not (appearance_dimensions(merged) & {"tail"})


def test_hair_and_body_tail_remain_separate_in_same_description():
    text = "long pink hair styled into twin tails, a fox tail"
    assert appearance_dimensions(text) >= {"hair_style", "tail"}
    omitted = _without_omitted_appearance(text, frozenset({"tail"}))
    assert "twin tails" in omitted and "fox tail" not in omitted


def reference_cache():
    return SemanticLookupResult(
        confirmed_tags=("character_a", "character_b", "character_c"),
        character_appearance_profiles=(
            (("角色甲",), "character_a", ("pink_hair", "grey_eyes", "straight_hair")),
            (("角色乙",), "character_b", ("blue_hair", "yellow_eyes", "two_side_up", "horns", "yellow_eyes_blue_hair")),
            (("角色丙",), "character_c", ("green_hair", "blue_eyes", "curly_hair")),
        ),
        source_outfit_profiles=(
            ("角色乙", "character_b", ("red_dress", "hair_ribbon", "boots"), "default"),
            ("角色丙", "character_c", ("white_shirt", "hair_bow"), "default"),
        ), status="profile_cache",
    )


@pytest.mark.parametrize("dimension,expected", [
    ("hair_style", {"two_side_up"}), ("eye_color", {"yellow_eyes"}),
    ("horns", {"horns"}), ("hair_color", {"blue_hair"}),
])
def test_source_evidence_is_dimension_scoped_for_any_recipient(dimension, expected):
    change = SemanticAppearanceChange("角色甲", "角色甲采用角色乙的特征", (dimension,), "replace", "角色乙")
    row, = appearance_reference_evidence((change,), reference_cache())
    assert set(row["tags"]) == expected
    assert row["optional_accessories"] == (["hair_ribbon"] if dimension == "hair_style" else [])
    assert "red_dress" not in str(row) and "boots" not in str(row)


def test_unseen_ambiguous_or_unbound_sources_do_not_select_saved_profile():
    change = SemanticAppearanceChange("角色甲", "角色甲采用陌生人的发型", ("hair_style",), "replace", "陌生人")
    assert not appearance_reference_evidence((change,), reference_cache())
    ambiguous = replace(reference_cache(), character_appearance_profiles=(
        (("陌生人",), "source_a", ("curly_hair",)),
        (("陌生人",), "source_b", ("twintails",)),
    ))
    assert not appearance_reference_evidence((change,), ambiguous)
    assert not appearance_reference_evidence((replace(change, reference_character="角色"),), reference_cache())


def test_independent_changes_keep_reference_and_accessory_permissions_per_wearer():
    plans = (local_plan(), local_plan("角色丙", "character_c"))
    changes = (SemanticAppearanceChange("角色甲", "角色甲采用角色乙的发型", ("hair_style",), "replace", "角色乙"),
               SemanticAppearanceChange("角色丙", "角色丙采用角色乙的瞳色", ("eye_color",), "replace", "角色乙"))
    a, c = bind_appearance_accessory_permissions(plans, changes)
    authority = build_wardrobe_authority((a, c), reference_cache(), ())
    assert a.appearance_accessory_slots == ("hair_accessory",)
    assert c.appearance_accessory_slots == ()
    assert authority.filter_tags(("hair_ribbon",)) == ()  # not shared by everyone
    assert authority.for_character(a).filter_tags(("hair_ribbon",)) == ("hair_ribbon",)
    assert authority.for_character(c).filter_tags(("hair_ribbon",)) == ()
    assert "hair_ribbon" in authority.filter_character_tags("hair_ribbon, red_dress", a)
    assert "red_dress" not in authority.filter_character_tags("hair_ribbon, red_dress", a)
    removed = replace(authority, removed_tags=("hair_ribbon",))
    assert removed.for_character(a).filter_tags(("hair_ribbon",)) == ()
    assert replace(authority, removed_slots=("hair_accessory",)).for_character(a).filter_tags(("hair_ribbon",)) == ()


def test_opened_hair_dimension_survives_shared_tags_without_unlocking_other_facts():
    plan = local_plan()
    change = SemanticAppearanceChange("角色甲", "角色甲采用角色乙的发型", ("hair_style",), "replace", "角色乙")
    assert filter_shared_appearance_tags(
        "two_side_up, blue_hair, yellow_eyes, sky", (plan,), user_prompt="", appearance_changes=(change,),
    ) == "two_side_up, sky"
    other = local_plan("角色丙", "character_c")
    assert filter_shared_appearance_tags(
        "two_side_up, sky", (plan, other), user_prompt="", appearance_changes=(change,),
    ) == "sky"


def test_appearance_source_cannot_select_clothing_even_when_repair_fails():
    prompt = "角色甲采用角色乙的发型"
    row = semantic_character("角色甲", changes=[dict(
        dimension="hair_style", operation="replace", source_text=prompt, reference_character="角色乙")])
    row["clothing_source"] = "角色乙"
    raw = json.dumps(dict(characters=[row]), ensure_ascii=False)
    assert any("clothing_source requires" in issue for issue in semantic_plan_validation_issues(raw, prompt))
    anchors = parse_semantic_plan(raw, prompt)
    assert all(anchor.role != "outfit_source" for anchor in anchors)
    assert parse_semantic_character_plans(raw, prompt, anchors)[0].wardrobe.kind == "none"
    parsed, = parse_semantic_appearance_changes(raw, prompt)
    assert parsed.reference_character == "角色乙"


def test_garment_source_and_appearance_source_can_differ():
    prompt = "角色甲穿角色丙的陌生礼服，采用角色乙的发型"
    row = semantic_character("角色甲", "角色丙的陌生礼服", [dict(
        dimension="hair_style", operation="replace", source_text="采用角色乙的发型", reference_character="角色乙")])
    row["clothing_source"] = "角色丙"
    raw = json.dumps(dict(characters=[row]), ensure_ascii=False)
    assert semantic_plan_validation_issues(raw, prompt) == ()
    anchors = parse_semantic_plan(raw, prompt)
    assert any(anchor.role == "outfit_source" and anchor.source_text == "角色丙" for anchor in anchors)
    assert parse_semantic_appearance_changes(raw, prompt)[0].reference_character == "角色乙"


def test_reference_name_elsewhere_does_not_grant_appearance_transfer():
    prompt = "角色甲改成卷发，角色乙在旁边站立"
    raw = json.dumps(dict(characters=[semantic_character("角色甲", changes=[dict(
        dimension="hair_style", operation="replace", source_text="改成卷发", reference_character="角色乙")])]), ensure_ascii=False)
    assert any("reference_character must occur" in issue for issue in semantic_plan_validation_issues(raw, prompt))
    assert not parse_semantic_appearance_changes(raw, prompt)[0].reference_character


@pytest.mark.parametrize("framing", ["角色甲大头照", "超级大头照", "证件照", "脸部肖像", "headshot"])
def test_portrait_crop_omits_inherited_torso_attributes_but_keeps_neck_and_hair(framing):
    tags = ("white_shirt", "grey_jacket", "green_necktie", "neck_ribbon", "long_sleeves", "hair_ribbon", "black_choker")
    kept, omitted = filter_profile_tags_for_composition(tags, framing)
    assert set(kept) == {"hair_ribbon", "black_choker"}
    assert set(omitted) == set(tags) - set(kept)


def test_upper_body_and_clothing_mentions_do_not_trigger_head_crop():
    assert "upper_body.neckwear" not in framing_hidden_outfit_slots("上半身肖像")
    assert not framing_hidden_outfit_slots("下半身没有衣服")
    assert outfit_tag_slot("black_choker") != "upper_body.neckwear"


def test_cropped_default_is_not_missing_profile_and_does_not_restore_full_uniform_identity():
    baseline = ("academy_school_uniform", "winter_uniform", "white_shirt", "long_sleeves", "green_necktie")
    plan = replace(local_plan(), effective=EffectiveOutfitPlan(subject="角色甲", base_tags=baseline, effective_tags=baseline))
    cropped, = apply_framing_to_character_outfits((plan,), "大头照")
    assert not cropped.effective.effective_tags
    assert set(cropped.effective.removed_tags) == set(baseline)
    assert "outfit.identity" in cropped.effective.forbidden_slots
    unchanged, missing = fallback_missing_unspecified_profiles((cropped,), explicit_wardrobe_evidence=False)
    assert unchanged == (cropped,) and not missing
    explicit = replace(plan, wardrobe_kind="named_outfit", wardrobe_tag="academy_school_uniform")
    kept, = apply_framing_to_character_outfits((explicit,), "大头照")
    assert "academy_school_uniform" in kept.effective.effective_tags


def test_reference_copyright_is_scoped_without_dropping_independently_visible_owner():
    evidence = replace(reference_cache(), confirmed_tags=("character_a", "character_b", "series_a", "series_b"),
                       profile_copyrights=(("character_a", ("series_a",)), ("character_b", ("series_b",))))
    single = non_wardrobe_confirmed_tags(evidence, (local_plan(),), include_source_anchor=True)
    assert "series_a" in single and "series_b" not in single
    pair = non_wardrobe_confirmed_tags(evidence, (local_plan(), local_plan("角色乙", "character_b")), include_source_anchor=True)
    assert "series_a" in pair and "series_b" in pair


@pytest.mark.parametrize("custom", ["", "自定义任务书：{theme}"])
def test_writer_receives_focus_rules_even_with_custom_template(custom):
    prompt = build_llm_prompt("角色甲采用角色乙的发型", prompt_builder_template=custom)
    assert VISUAL_FOCUS_POLICY in prompt
    assert "Nltags adds" in prompt and "every requested person" in prompt


class ReferenceResolver(Resolver):
    def cached_outfit_profiles_for_prompt(self, _prompt):
        return reference_cache()


def test_full_pipeline_passes_source_shape_without_borrowing_wardrobe_or_losing_accessory():
    prompt = "角色甲大头照，采用角色乙的发型，脖子戴项圈"
    change = dict(dimension="hair_style", operation="replace", source_text="采用角色乙的发型", reference_character="角色乙")
    writer = ("{Count: 1girl}\n{Characters: character_a}\n{Copyright:}\n"
              "{Identity: character_a has pink hair styled in two side up twin tails and grey eyes}\n"
              "{Details: character_a wears a black choker and a hair ribbon}\n"
              "{Tags: headshot, two_side_up, hair_ribbon, white background}\n{Nltags:}")
    result, calls = run(prompt, dict(characters=[semantic_character("角色甲", changes=[change])]),
                        writer, resolver=ReferenceResolver(), config={"debug_prompt_enabled": True})
    assert result.final_prompt and not result.summary.get("llm_failed"), result.summary
    assert len(calls) == 2
    taskbook = calls[1]["prompt"]
    reference_line = next(line for line in taskbook.splitlines() if "角色甲 <= 角色乙" in line)
    assert "two_side_up" in reference_line and "blue_hair" not in reference_line
    assert "red_dress" not in reference_line and "boots" not in reference_line
    assert "cosplay source =" not in taskbook
    assert "two side up" in result.final_prompt
    assert "twin tails" in result.final_prompt
    assert "hair ribbon" in result.final_prompt
    assert "red dress" not in result.final_prompt
    assert "character_b" not in result.summary["required_core_tags"]
    assert not result.summary["global_outfit_reinforcement_tags"]


def test_raw_remains_byte_for_byte_without_source_interpretation():
    tags = "1girl, character_a, {{{two side up}}}, hair ribbon, 0.7::close shot::"
    result, calls = run("无优化 " + tags, {}, "", resolver=ReferenceResolver())
    assert result.final_prompt == tags and not calls


def test_same_identity_different_panels_do_not_share_accessory_allowance():
    request = "左格角色甲采用角色乙的发型。右格角色甲维持原样。"
    change = SemanticAppearanceChange("角色甲", "左格角色甲采用角色乙的发型", ("hair_style",), "replace", "角色乙")
    inherited, = bind_appearance_accessory_permissions((local_plan(),), (change,))
    left, _ = _nai_instance_evidence(
        dict(name="character_a", source_text="左格角色甲采用角色乙的发型"),
        (inherited,), (change,), request, repeated=True,
    )
    right, _ = _nai_instance_evidence(
        dict(name="character_a", source_text="右格角色甲维持原样"),
        (inherited,), (change,), request, repeated=True,
    )
    assert left.appearance_accessory_slots == ("hair_accessory",)
    assert not right.appearance_accessory_slots


def test_nai_appearance_donor_review_does_not_require_a_garment_relationship():
    prompt = "-r 角色甲大头照，采用角色乙的发型，角色乙只提供发型参考，不出场。"
    change = dict(dimension="hair_style", operation="replace", source_text="采用角色乙的发型", reference_character="角色乙")
    writer = json.dumps(dict(
        composition_analysis=dict(viewpoint="front", layout="portrait", relations="one visible character"),
        global_prompt="1girl, headshot, white background, 角色甲的单人白底大头照。",
        characters=[dict(name="character_a", prompt="girl, character_a, two_side_up, pink_hair, grey_eyes, hair_ribbon, 角色甲保持粉发灰眼，梳两侧扎起的发型并戴发带。",
                         x=.5, y=.5, position_reason="portrait centre", interaction_tags=[])],
        identity_review=[dict(name="character_b", visibility="donor", source_text="采用角色乙的发型")],
    ), ensure_ascii=False)
    result, calls = run(prompt, dict(characters=[semantic_character("角色甲", changes=[change])]),
                        writer, resolver=ReferenceResolver())
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    assert "hair_ribbon" in result.summary["nai_characters"][0]["prompt"]


def test_reference_review_preserves_independent_edits_and_does_not_force_hidden_features():
    change = SemanticAppearanceChange("角色甲", "采用角色乙的发型", ("hair_style",), "replace", "角色乙")
    rows = appearance_reference_evidence((change,), reference_cache())
    assert reference_appearance_issues("pink hair in twin buns", rows)
    assert not reference_appearance_issues("pink hair, two side up", rows)
    assert not reference_appearance_issues("pink hair in a two-side-up arrangement", rows)
    assert not reference_appearance_issues("face only, grey eyes", rows)
    modified = SemanticAppearanceChange("角色甲", "改成卷发", ("hair_style",), "replace")
    assert not reference_appearance_issues(
        "pink hair, curly hair", appearance_reference_evidence((change, modified), reference_cache()),
    )


def test_reference_review_does_not_guess_between_two_sources_for_same_dimension():
    changes = tuple(SemanticAppearanceChange("角色甲", f"采用{source}的发型", ("hair_style",), "replace", source)
                    for source in ("角色乙", "角色丙"))
    assert not reference_appearance_issues("pink hair", appearance_reference_evidence(changes, reference_cache()))


@pytest.mark.parametrize("independently_visible", [False, True])
def test_source_only_mentions_do_not_become_required_people_but_independent_occurrences_do(independently_visible):
    phrase = "角色甲采用角色乙的瞳色"
    prompt = phrase + ("，角色乙在旁边坐着" if independently_visible else "，白底")
    first = semantic_character("角色甲", changes=[dict(
        dimension="eye_color", operation="replace", source_text="采用角色乙的瞳色", reference_character="角色乙")])
    raw = json.dumps(dict(characters=[first, semantic_character("角色乙")]), ensure_ascii=False)
    issues = semantic_plan_validation_issues(raw, prompt)
    assert any("without an independent visible occurrence" in issue for issue in issues) != independently_visible
    anchors = parse_semantic_plan(raw, prompt)
    assert any(anchor.role == "target_character" and anchor.source_text == "角色乙" for anchor in anchors) == independently_visible


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_known_source_fact_missing_in_own_field_uses_only_existing_writer_repair(repair_succeeds):
    change = dict(dimension="hair_style", operation="replace", source_text="采用角色乙的发型", reference_character="角色乙")
    # A global tag alone does not specify who wears the known source shape.
    writer = ("{Count: 1girl}{Characters: character_a}{Copyright:}"
              "{Identity: character_a has pink hair in buns and grey eyes}"
              "{Details: character_a wears a black choker}"
              "{Tags: two_side_up, headshot, white background}{Nltags:}")
    repair = writer.replace("in buns", "in two side up") if repair_succeeds else writer
    result, calls = run("角色甲大头照，采用角色乙的发型", dict(characters=[semantic_character("角色甲", changes=[change])]),
                        writer, resolver=ReferenceResolver(), repair=repair)
    assert len(calls) == 3
    assert any("OWN" in issue for issue in result.summary["structured_initial_validation_errors"])
    assert bool(result.final_prompt) == repair_succeeds
    if repair_succeeds:
        assert result.summary["prompt_llm_accepted_attempt"] == "retry"
        assert "two side up" in result.final_prompt
    else:
        assert result.summary["structured_retry_validation_errors"]


def test_nai_known_source_fact_repair_is_instance_scoped():
    change = dict(dimension="hair_style", operation="replace", source_text="左格角色甲采用角色乙的发型", reference_character="角色乙")
    original = "-r 左格角色甲采用角色乙的发型。右格角色甲维持原样。"
    data = dict(
        composition_analysis=dict(viewpoint="front", layout="two panels", relations="same person twice"),
        global_prompt="2girls, comic, white background, 左右两格分别显示角色甲。",
        characters=[dict(name="character_a", prompt="girl, character_a, pink_hair, grey_eyes, buns, 角色甲粉发灰眼。",
                         source_text="左格角色甲采用角色乙的发型", x=.25, y=.5, position_reason="left", interaction_tags=[]),
                    dict(name="character_a", prompt="girl, character_a, pink_hair, grey_eyes, straight_hair, 角色甲保持原发型。",
                         source_text="右格角色甲维持原样", x=.75, y=.5, position_reason="right", interaction_tags=[])],
        identity_review=[dict(name="character_b", visibility="donor", source_text="左格角色甲采用角色乙的发型")],
    )
    initial = json.dumps(data, ensure_ascii=False)
    data["characters"][0]["prompt"] = data["characters"][0]["prompt"].replace("buns", "two_side_up")
    result, calls = run(original, dict(characters=[semantic_character("角色甲", changes=[change])]),
                        initial, resolver=ReferenceResolver(), repair=json.dumps(data, ensure_ascii=False))
    assert len(calls) == 3 and result.final_prompt, result.summary
    assert result.summary["nai_appearance_initial_validation_errors"]
    assert not result.summary["nai_appearance_retry_validation_errors"]
    assert "straight_hair" in result.summary["nai_characters"][1]["prompt"]
