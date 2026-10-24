"""General regressions for the 2026-10-20 borrowed-outfit failures."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import replace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from danbooru_semantic import (
    SemanticLookupResult, parse_semantic_character_plans, parse_semantic_plan, semantic_plan_validation_issues,
)
from outfit_transfer import UserOutfitPatch, OutfitTransferPlan, build_effective_outfit_plan
from prompt_pipeline import (
    CharacterEffectiveOutfit, apply_framing_to_character_outfits, build_wardrobe_authority,
    character_wardrobe_authority_context, controlled_character_outfit_detail,
    WardrobeAuthority,
)


def intent(name, clothing=None, donor=None, changes=()):
    return dict(name=name, aliases=[], clothing=clothing, clothing_source=donor,
                clothing_changes=list(changes), appearance_changes=[])


@pytest.mark.parametrize("name,source", [("若叶睦", "mortis"), ("角色乙", "Astral Knight")])
def test_donor_mentioned_only_in_accepted_clothing_is_not_a_second_person(name, source):
    clothing = f"{source}的衣服"
    prompt = f"{name}穿着{clothing}站立"
    raw = json.dumps(dict(characters=[intent(name, clothing, source), intent(source)]))
    issues = semantic_plan_validation_issues(raw, prompt)
    assert any("clothing_source, not a visible character" in issue for issue in issues)
    anchors = parse_semantic_plan(raw, prompt)
    assert [a.source_text for a in anchors if a.role == "target_character"] == [name]
    assert any(a.role == "outfit_source" and a.source_text == source for a in anchors)


def test_donor_with_an_independent_occurrence_still_appears():
    prompt = "甲穿着乙的衣服，乙站在旁边"
    raw = json.dumps(dict(characters=[intent("甲", "乙的衣服", "乙"), intent("乙")]))
    assert not semantic_plan_validation_issues(raw, prompt)
    assert len([a for a in parse_semantic_plan(raw, prompt) if a.role == "target_character"]) == 2


@pytest.mark.parametrize("evidence", [
    "裙子经过剪裁，只留下身体侧面的部分", "外套只保留袖口部分",
    "the dress retains only the side panels",
])
def test_partial_retention_never_authorizes_a_whole_outfit_allowlist(evidence):
    prompt = "角色甲穿舞台服，" + evidence
    change = dict(operation="keep_only", slots=["lower_body.skirt"], source_text=evidence)
    raw = json.dumps(dict(characters=[intent("角色甲", "舞台服", changes=[change])]))
    assert any("partial garment" in issue for issue in semantic_plan_validation_issues(raw, prompt))
    assert not parse_semantic_character_plans(raw, prompt)[0].directives
    change["operation"] = "modify"
    raw = json.dumps(dict(characters=[intent("角色甲", "舞台服", changes=[change])]))
    assert not semantic_plan_validation_issues(raw, prompt)
    assert parse_semantic_character_plans(raw, prompt)[0].directives[0].source_text == evidence


def test_modified_skirt_preserves_other_garments_through_context_and_final_filter():
    base = ("red_shirt", "black_corset", "black_skirt", "black_gloves", "black_pantyhose", "black_mask")
    effective = build_effective_outfit_plan(
        OutfitTransferPlan(enabled=True, target_character="角色甲"),
        user_prompt="角色甲穿舞台服（没有面具），裙子只留下侧面的部分，双腿被黑丝包裹",
        base_tags=base,
        semantic_patches=(UserOutfitPatch("角色甲", "modify", "lower_body.skirt",
                                        evidence="裙子只留下侧面的部分"),),
    )
    assert effective.removed_tags == ("black_mask",)
    assert "black_pantyhose" in effective.effective_tags
    wearer = CharacterEffectiveOutfit(
        target_anchor_id="t", target_source_text="角色甲", target_candidates=("character_a",),
        wardrobe_kind="outfit_source", wardrobe_anchor_id="s", wardrobe_tag="stage_persona",
        appearance_tags=(), effective=effective, complete_named_profile=True,
    )
    context = character_wardrobe_authority_context((wearer,))
    assert "black_pantyhose" in context and "modify lower_body.skirt" in context
    wearer = apply_framing_to_character_outfits((wearer,), "full body, foot focus")[0]
    authority = build_wardrobe_authority((wearer,), SemanticLookupResult(), ())
    detail = controlled_character_outfit_detail(
        "character_a wears a red shirt, black corset, side-panel black skirt, black gloves and black pantyhose",
        "character_a", wearer, wardrobe_authority=authority,
    )
    assert "black pantyhose" in detail and "red shirt" in detail and "black gloves" in detail


def test_whole_outfit_keep_only_still_respects_aggregate_lower_body_slot():
    effective = build_effective_outfit_plan(
        OutfitTransferPlan(target_character="甲"), user_prompt="甲只穿下半身衣物",
        base_tags=("red_shirt", "black_skirt", "black_pantyhose"),
        semantic_patches=(UserOutfitPatch("甲", "keep_only", "lower_body.all", evidence="只穿下半身衣物"),),
    )
    assert effective.effective_tags == ("black_skirt", "black_pantyhose")


@pytest.mark.parametrize("description", ["甲穿舞台服（没有面具）", "甲没戴面罩", "甲不戴帽子"])
def test_accessory_absence_has_a_grounded_fallback_when_planner_misses_it(description):
    effective = build_effective_outfit_plan(
        OutfitTransferPlan(target_character="甲"), user_prompt=description,
        base_tags=("white_shirt", "black_mask", "masquerade_mask", "red_hat"),
    )
    missing = "red_hat" if "帽子" in description else "black_mask"
    assert missing in effective.removed_tags
    assert "white_shirt" in effective.effective_tags


def test_one_persons_accessory_absence_does_not_remove_anothers():
    effective = build_effective_outfit_plan(
        OutfitTransferPlan(target_character="甲"), user_prompt="甲穿舞台服，乙没有面具",
        known_character_names=("甲", "乙"), base_tags=("black_mask", "red_shirt"),
    )
    assert effective.effective_tags == ("black_mask", "red_shirt")


def test_other_wearers_generic_component_does_not_delete_own_specific_component():
    wearer = CharacterEffectiveOutfit(
        "a", "甲", ("character_a",), "outfit_source", "s", "source_a", (),
        build_effective_outfit_plan(OutfitTransferPlan(target_character="甲"), user_prompt="",
                                   base_tags=("green_brooch", "red_shirt")),
    )
    authority = WardrobeAuthority(
        has_character_plans=True,
        character_selected_tags=(("a", ("green_brooch", "red_shirt")),
                                 ("b", ("brooch", "shirt", "blue_skirt"))),
    )
    text = "character_a wears a red shirt, a green brooch and a blue skirt"
    actual = authority.filter_character_prose(text, wearer)
    assert "red shirt" in actual and "green brooch" in actual
    assert "blue skirt" not in actual
    # An explicit deletion is still enforced when the component is no longer selected.
    removed = replace(authority, character_selected_tags=(("a", ("red_shirt",)),),
                      removed_tags=("green_brooch",))
    assert "green brooch" not in removed.filter_character_prose(text, wearer)
