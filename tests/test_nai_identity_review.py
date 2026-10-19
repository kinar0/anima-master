"""Identity completeness without turning every mentioned person into a target."""
import pytest

from danbooru_semantic import SemanticLookupResult
from prompt_pipeline import saved_appearance_for_names
from test_prompt_direct_writer import Resolver, plan, run, semantic_character, with_identity_review


VARIANT = "character_a_(alternate)"


class Profiles(Resolver):
    def cached_outfit_profiles_for_prompt(self, _prompt):
        return SemanticLookupResult(
            confirmed_tags=("character_a", VARIANT, "character_b"),
            character_appearance_profiles=(
                (("角色甲",), "character_a", ("pink_hair", "grey_eyes")),
                (("变体甲",), VARIANT, ("blue_hair", "yellow_eyes", "horns")),
                (("角色乙",), "character_b", ("green_hair",)),
            ), status="profile_cache",
        )


REQUEST = "-r 角色甲和变体甲衣服略显凌乱地坐在沙发上，角色乙在旁边拿着自拍杆。"
TWO = [("character_a", "girl, sitting", .25), ("character_b", "girl, standing", .75)]
THREE = [TWO[0], (VARIANT, "girl, brown_hair, sitting, 变体甲坐在沙发上。", .5), TWO[1]]


@pytest.mark.parametrize("source_misclassified", [False, True])
def test_omitted_identity_repairs_once_then_binds_its_own_profile(source_misclassified):
    first = semantic_character("角色甲", "变体甲衣服" if source_misclassified else None)
    if source_misclassified:
        first["clothing_source"] = "变体甲"
    semantic = {"characters": [first, semantic_character("角色乙")]}
    repaired = plan(THREE)
    if source_misclassified:
        repaired = with_identity_review(repaired, VARIANT, "visible", "角色甲和变体甲衣服略显凌乱地坐在沙发上")
    result, calls = run(REQUEST, semantic, plan(TWO), resolver=Profiles(), repair=repaired)
    assert result.final_prompt, result.summary
    assert len(calls) == 3
    assert result.summary["nai_identity_initial_validation_errors"]
    assert result.summary["nai_identity_retry_validation_errors"] == []
    rows = result.summary["nai_characters"]
    assert len(rows) == 3 and rows[1]["name"] == VARIANT
    assert "blue_hair" in rows[1]["prompt"] and "brown_hair" not in rows[1]["prompt"]
    assert "horns" not in rows[1]["prompt"]  # Unexpressed traits remain omitted.
    assert result.summary["nai_anchor_roles"]


def test_known_alias_in_returned_box_canonicalizes_without_extra_call():
    output = plan([TWO[0], ("变体甲", THREE[1][1], .5), TWO[1]])
    result, calls = run(REQUEST, {"characters": [semantic_character("角色甲"), semantic_character("角色乙")]},
                        output, resolver=Profiles())
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    row = result.summary["nai_characters"][1]
    assert row["name"] == VARIANT and row["prompt"].startswith(VARIANT + ",")
    assert "变体甲坐在沙发上" in row["prompt"]
    assert "blue_hair" in row["prompt"] and "brown_hair" not in row["prompt"]


@pytest.mark.parametrize("review", [None, ("offscreen", "变体甲"),
                                   ("offscreen", "变体甲在画外"),
                                   ("visible", "角色甲和变体甲衣服略显凌乱地坐在沙发上")])
def test_missing_or_ungrounded_or_inconsistent_review_cannot_silently_succeed(review):
    output = plan(TWO)
    if review:
        output = with_identity_review(output, VARIANT, *review)
    result, calls = run(REQUEST, {"characters": [semantic_character("角色甲"), semantic_character("角色乙")]},
                        output, resolver=Profiles(), repair=output)
    assert not result.final_prompt
    assert len(calls) == 3
    assert "nai_identity_plan_invalid" in result.summary["llm_error"]


@pytest.mark.parametrize("suffix,visibility", [
    ("变体甲在画外说话", "offscreen"),
    ("变体甲的名字写在牌子上", "quoted"),
    ("角色甲穿变体甲的衣服", "donor"),
])
def test_grounded_nonvisible_mentions_do_not_create_extra_people(suffix, visibility):
    request = "-r 角色甲和角色乙坐在沙发上，" + suffix
    output = with_identity_review(plan(TWO), VARIANT, visibility, suffix)
    result, calls = run(request, {"characters": [semantic_character("角色甲"), semantic_character("角色乙")]},
                        output, resolver=Profiles())
    assert result.final_prompt, result.summary
    assert len(calls) == 2 and len(result.summary["nai_characters"]) == 2
    assert result.summary["nai_bound_local_profile_references"] == []
    assert result.summary["nai_identity_review"][0]["visibility"] == visibility


def test_shared_scene_quote_does_not_silently_prove_donor_exclusion():
    first = dict(semantic_character("角色甲", "变体甲衣服"), clothing_source="变体甲")
    evidence = "角色甲和变体甲衣服略显凌乱地坐在沙发上"
    initial = with_identity_review(plan(TWO), VARIANT, "donor", evidence)
    repaired = with_identity_review(plan(THREE), VARIANT, "visible", evidence)
    result, calls = run(REQUEST, {"characters": [first, semantic_character("角色乙")]},
                        initial, resolver=Profiles(), repair=repaired)
    assert result.final_prompt, result.summary
    assert len(calls) == 3 and len(result.summary["nai_characters"]) == 3
    assert "No direct wearer-to-source relationship" in calls[2]["prompt"]


def test_unfamiliar_borrowing_is_rechecked_without_host_grammar_becoming_a_veto():
    phrase = "角色甲的装束沿用变体甲的设计，变体甲仅供衣装参考"
    request = "-r " + phrase + "，角色乙站在旁边"
    initial = with_identity_review(plan(TWO), VARIANT, "donor", phrase)
    result, calls = run(request, {"characters": [semantic_character("角色甲"), semantic_character("角色乙")]},
                        initial, resolver=Profiles(), repair=initial)
    assert result.final_prompt, result.summary
    assert len(calls) == 3 and len(result.summary["nai_characters"]) == 2
    assert result.summary["nai_identity_review"][0]["visibility"] == "donor"


@pytest.mark.parametrize("form", ["ellipsis", "array", "invalid_fragment"])
def test_discontinuous_donor_evidence_preserves_only_independently_exact_excerpts(form):
    first = "角色甲穿变体甲的衣服"
    last = "变体甲只作服装来源，不出现在画面中"
    request = f"-r {first}坐在沙发上，角色乙站在旁边。{last}。"
    excerpts = [first, last if form != "invalid_fragment" else "变体甲在原文没有说过的地方"]
    evidence = "……".join(excerpts) if form == "ellipsis" else excerpts
    output = with_identity_review(plan(TWO), VARIANT, "donor", evidence)
    result, calls = run(request, {"characters": [semantic_character("角色甲"), semantic_character("角色乙")]},
                        output, resolver=Profiles(), repair=output)
    if form == "invalid_fragment":
        assert not result.final_prompt and len(calls) == 3
    else:
        assert result.final_prompt, result.summary
        assert len(calls) == 2
        assert result.summary["nai_identity_review"][0]["source_text"] == excerpts


def test_exact_identity_beats_base_name_prefix_but_conflicting_exact_alias_stays_ambiguous():
    result = Profiles().cached_outfit_profiles_for_prompt("")
    assert saved_appearance_for_names(result, ("变体甲", VARIANT)) == (VARIANT, ("blue_hair", "yellow_eyes", "horns"))
    assert saved_appearance_for_names(result, ("角色甲", "character_a"))[0] == "character_a"
    from dataclasses import replace
    ambiguous = replace(result, character_appearance_profiles=(
        (("同名",), "character_a", ("pink_hair",)),
        (("同名",), VARIANT, ("blue_hair",)),
    ))
    assert saved_appearance_for_names(ambiguous, ("同名",)) == ("", ())


def test_ambiguous_alias_is_not_guessed_or_required_as_two_people():
    class Ambiguous(Resolver):
        def cached_outfit_profiles_for_prompt(self, _prompt):
            return SemanticLookupResult(confirmed_tags=("character_a", VARIANT),
                character_appearance_profiles=((("同名",), "character_a", ("pink_hair",)),
                                               (("同名",), VARIANT, ("blue_hair",))), status="profile_cache")
    result, calls = run("-r 同名站立", {"characters": [semantic_character("同名")]},
                        plan([("同名", "girl, standing", .5)]), resolver=Ambiguous())
    assert result.final_prompt, result.summary
    assert len(calls) == 2 and result.summary["nai_characters"][0]["name"] == "同名"
