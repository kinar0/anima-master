"""A selected costume source must reach the actual wearer prompt, even when edited."""
from dataclasses import replace

import pytest

from danbooru_semantic import SemanticAnchor, SemanticLookupResult
from outfit_transfer import EffectiveOutfitPlan
from prompt_pipeline import (
    CharacterEffectiveOutfit, append_outfit_source_tags, selected_outfit_source_tags,
)
from test_prompt_direct_writer import Resolver, plan, run, with_identity_review


class CostumeResolver(Resolver):
    def cached_outfit_profiles_for_prompt(self, _prompt):
        return SemanticLookupResult(
            confirmed_tags=("character_a", "character_b", "costume_a", "costume_b"),
            outfit_source_tags=("costume_a", "costume_b"),
            source_outfit_profiles=(("来源甲", "costume_a", ("red_shirt", "black_mask"), "default"),
                                    ("来源乙", "costume_b", ("blue_dress",), "default")),
            character_appearance_profiles=((("角色甲",), "character_a", ("pink_hair", "grey_eyes")),
                                           (("角色乙",), "character_b", ("blue_hair", "yellow_eyes"))),
            status="profile_cache",
        )


def semantics():
    return {"anchors": [
        dict(id="a", role="target_character", group="character", source_text="角色甲", candidates=["character_a"]),
        dict(id="b", role="target_character", group="character", source_text="角色乙", candidates=["character_b"]),
        dict(id="sa", role="outfit_source", group="character", source_text="来源甲", candidates=["costume_a"]),
        dict(id="sb", role="outfit_source", group="character", source_text="来源乙", candidates=["costume_b"]),
    ], "character_plans": [
        dict(target_anchor_id="a", wardrobe=dict(kind="outfit_source", anchor_id="sa"), directives=[
            dict(operation="remove", slots=["face_accessory.mask"], source_text="角色甲没有面具")]),
        dict(target_anchor_id="b", wardrobe=dict(kind="outfit_source", anchor_id="sb"), directives=[]),
    ]}


@pytest.mark.parametrize("nai", [False, True])
def test_actual_prompts_retain_each_borrowed_source_despite_missing_writer_tags(nai):
    request = "角色甲穿来源甲的衣服，角色乙穿来源乙的衣服，角色甲没有面具"
    if nai:
        request = "-r " + request
        output = plan([("character_a", "girl, pink_hair, red_shirt, 角色甲穿红衬衫。", .3),
                       ("character_b", "girl, blue_hair, blue_dress, 角色乙穿蓝裙。", .7)], "2girls, white_background")
        output = with_identity_review(output, "costume_a", "donor", "角色甲穿来源甲的衣服")
        import json
        data = json.loads(output)
        data["identity_review"].append(dict(name="costume_b", visibility="donor", source_text="角色乙穿来源乙的衣服"))
        output = json.dumps(data, ensure_ascii=False)
    else:
        output = ("{Count: 2girls}{Characters: character_a, character_b}{Copyright: }"
                  "{Identity: character_a has pink hair; character_b has blue hair}"
                  "{Details: character_a wears a red shirt; character_b wears a blue dress}"
                  "{Tags: full body, white background}{Nltags: }")
    result, calls = run(request, semantics(), output, resolver=CostumeResolver())
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    if nai:
        a, b = result.summary["nai_characters"]
        assert "costume a" in a["prompt"] and "costume b" not in a["prompt"]
        assert "costume b" in b["prompt"] and "costume a" not in b["prompt"]
        assert "costume a" not in result.final_prompt and "costume b" not in result.final_prompt
        assert "mask" not in a["prompt"]
    else:
        details = next(line for line in result.final_prompt.splitlines() if "character a wears" in line)
        a, b = details.split(";")
        assert "costume a" in a and "costume b" not in a
        assert "costume b" in b and "costume a" not in b
        assert result.summary["structured_character_count"] == 2
        assert "black mask" not in result.final_prompt
    assert result.summary["outfit_source_anchor_emitted"] is True


def test_unknown_or_unbound_source_cannot_be_injected_and_edits_do_not_remove_known_source():
    source = SemanticAnchor("s", "outfit_source", "character", "来源", "", ("costume_a",))
    outfit = CharacterEffectiveOutfit("a", "甲", ("character_a",), "outfit_source", "s", "costume_a", (),
                                     EffectiveOutfitPlan(removed_tags=("mask",)))
    assert selected_outfit_source_tags((source,), (outfit,)) == ("costume_a",)
    assert not selected_outfit_source_tags((), (outfit,))
    assert not selected_outfit_source_tags((source,), (replace(outfit, resolution_state="explicit_but_unresolved"),))
    assert not selected_outfit_source_tags((source,), (replace(outfit, wardrobe_kind="creative_fallback"),))
    assert not selected_outfit_source_tags((replace(source, role="appearance"),), (outfit,))
    assert append_outfit_source_tags("character_a, costume_a", ("costume_a",)) == "character_a, costume_a"
