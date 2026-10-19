"""Multi-panel instances stay separate from identity-level review records."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from danbooru_semantic import SemanticLookupResult
from prompt_pipeline import nai_cross_panel_instance_issues
from nai_character_mode import reconcile_nai_instance_population
from test_prompt_direct_writer import Resolver, plan, run, semantic_character


class Aliases(Resolver):
    def cached_outfit_profiles_for_prompt(self, _prompt):
        return SemanticLookupResult(
            confirmed_tags=("character_a", "character_b"),
            character_appearance_profiles=(
                (("角色甲", "小甲"), "character_a", ("pink_hair",)),
                (("角色乙", "小乙"), "character_b", ("blue_hair",)),
            ), status="profile_cache",
        )


def reviewed(rows, reviews, sources=()):
    output = json.loads(plan(rows))
    output["identity_review"] = [dict(name=name, visibility=role, source_text=evidence)
                                 for name, role, evidence in reviews]
    for character, evidence in zip(output["characters"], sources):
        character["source_text"] = evidence
    return json.dumps(output, ensure_ascii=False)


def test_repeated_visible_review_rows_merge_without_merging_panel_instances():
    first, second = "第一格：角色甲坐下", "第二格：小甲挥手"
    output = reviewed([
        ("character_a", "girl, sitting, 角色甲坐下。", .25),
        ("character_a", "girl, waving, 小甲挥手。", .75),
    ], [("character_a", "visible", first), ("小甲", "visible", second)], [first, second])
    result, calls = run(f"-r 两格漫画。{first}。{second}。",
                        {"characters": [semantic_character("角色甲")]}, output, resolver=Aliases())
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    assert len(result.summary["nai_characters"]) == 2
    assert result.summary["nai_identity_review"] == [dict(
        name="character_a", visibility="visible", source_text=[first, second])]


def test_collapsed_panel_evidence_uses_existing_single_repair_budget():
    first, second = "第一格：角色甲坐下", "第二格：小甲挥手"
    reviews = [("character_a", "visible", first), ("character_a", "visible", second)]
    initial = reviewed([("character_a", "girl, sitting, waving", .5)], reviews, [[first, second]])
    repair = reviewed([("character_a", "girl, sitting", .25), ("character_a", "girl, waving", .75)],
                      reviews, [first, second])
    result, calls = run(f"-r 两格漫画。{first}。{second}。",
                        {"characters": [semantic_character("角色甲")]}, initial,
                        resolver=Aliases(), repair=repair)
    assert result.final_prompt, result.summary
    assert len(calls) == 3
    assert any("nai_character_cross_panel_instance" in issue
               for issue in result.summary["nai_identity_initial_validation_errors"])
    assert result.summary["nai_identity_retry_validation_errors"] == []
    assert len(result.summary["nai_characters"]) == 2
    assert "Split each visible appearance" in calls[-1]["prompt"]


def test_duplicate_context_in_single_scene_does_not_invent_another_person():
    first, second = "角色甲坐在窗边", "小甲拿着一本书"
    output = reviewed([("character_a", "girl, sitting, holding book", .5)],
                      [("character_a", "visible", first), ("character_a", "visible", second)])
    result, calls = run(f"-r {first}，{second}。", {"characters": [semantic_character("角色甲")]},
                        output, resolver=Aliases())
    assert result.final_prompt, result.summary
    assert len(calls) == 2 and len(result.summary["nai_characters"]) == 1


@pytest.mark.parametrize("second_role,second_context,error", [
    ("offscreen", "第二格：小甲挥手", "conflicting visibility"),
    ("donor", "第二格：小甲根本不存在的动作", "exact contextual evidence"),
])
def test_duplicate_reviews_do_not_relax_visibility_or_grounding(second_role, second_context, error):
    first, second = "第一格：角色甲坐下", "第二格：小甲挥手"
    output = reviewed([("character_a", "girl, sitting", .25), ("character_a", "girl, waving", .75)],
                      [("character_a", "visible", first), ("character_a", second_role, second_context)],
                      [first, second])
    result, calls = run(f"-r 两格漫画。{first}。{second}。", {"characters": [semantic_character("角色甲")]},
                        output, resolver=Aliases(), repair=output)
    assert not result.final_prompt and len(calls) == 3
    assert error in result.summary["llm_error"]


def test_invalid_optional_visible_review_does_not_veto_known_returned_instances():
    from dataclasses import replace
    class ConfirmedAliases(Aliases):
        def prefer_cached_character_anchors(self, anchors):
            return tuple(replace(anchor, candidates=("character_a",))
                         if anchor.role == "target_character" and anchor.source_text == "角色甲" else anchor
                         for anchor in anchors)

    first, second = "第一格：角色甲坐下", "第二格：小甲挥手"
    output = reviewed([("character_a", "girl, sitting", .25), ("character_a", "girl, waving", .75)],
                      [("character_a", "visible", "第一格：角色甲省略了中间文字的改写")], [first, second])
    result, calls = run(f"-r 两格漫画。{first}。{second}。", {"characters": [semantic_character("角色甲")]},
                        output, resolver=ConfirmedAliases())
    assert result.final_prompt, result.summary
    assert len(calls) == 2 and len(result.summary["nai_characters"]) == 2
    assert result.summary["nai_identity_review"] == []
    assert result.summary["nai_dropped_identity_reviews"][0]["reason"] == "invalid_optional_visible_review"


def test_ungrounded_required_visible_review_cannot_promote_a_clothing_donor():
    request = "-r 角色甲穿角色乙的衣服，角色乙站在旁边。"
    semantic = {"characters": [dict(semantic_character("角色甲", "角色乙的衣服"), clothing_source="角色乙")]}
    output = reviewed([("character_a", "girl, standing", .25), ("character_b", "girl, standing", .75)],
                      [("character_b", "visible", "角色乙出现在不存在的原文")])
    result, calls = run(request, semantic, output, resolver=Aliases(), repair=output)
    assert not result.final_prompt and len(calls) == 3
    assert "exact contextual evidence" in result.summary["llm_error"]


@pytest.mark.parametrize("headings", [
    ("第一格", "第二格"), ("左上小格子", "右下B格"), ("Panel A", "Panel B"),
])
def test_only_explicit_grounded_cross_panel_sources_require_split(headings):
    first, second = (f"{label}：角色甲微笑" for label in headings)
    request = f"两格漫画。{first}。{second}。共享要求为白衬衫。"
    def issues(evidence):
        return nai_cross_panel_instance_issues({"characters": [dict(name="character_a", source_text=evidence)]}, request)
    assert issues([first, second])
    assert issues(first + "。" + second)
    assert not issues([first, "共享要求为白衬衫"])
    assert not issues([first, first])
    assert not issues([first, "不存在的第三格：角色甲微笑"])
    assert not issues("")


def test_clothing_grid_labels_are_not_panel_headings():
    excerpts = ["衬衫的格子：蓝白相间", "裙子的格子：红黑相间"]
    assert not nai_cross_panel_instance_issues(
        {"characters": [dict(name="character_a", source_text=excerpts)]}, "。".join(excerpts))


@pytest.mark.parametrize("genders,initial,expected", [
    (["girl"] * 7, "6girls, 6koma, comic", "7girls, 6koma, comic"),
    (["girl", "boy"], "2girls, white background", "1girl, 1boy, white background"),
    (["girl", "other"], "2girls, comic", "2people, comic"),
    (["girl, boy"], "1girl, white background", "1people, white background"),
    ([], "no humans, title card", "no humans, title card"),
])
def test_population_uses_instances_and_keeps_panel_counts(genders, initial, expected):
    data = {"global_prompt": initial, "characters": [dict(name="same_identity", prompt=gender) for gender in genders]}
    reconcile_nai_instance_population(data)
    assert data["global_prompt"] == expected
    assert reconcile_nai_instance_population(data) is None


def test_exact_failed_comic_response_repairs_to_seven_independent_instances():
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads((root / "diagnostics/nai-comic-identity-review/original-failure.json").read_text(encoding="utf-8-sig"))
    summary = evidence["summary"]

    class RecordedProfiles(Resolver):
        def cached_outfit_profiles_for_prompt(self, _prompt):
            return SemanticLookupResult(
                confirmed_tags=tuple(summary["danbooru_semantic_confirmed_tags"]),
                character_appearance_profiles=tuple(
                    (tuple(row["aliases"]), row["source_tag"], tuple(row["appearance_tags"]))
                    for row in summary["danbooru_semantic_character_appearance_profiles"]
                ), named_outfit_tags=tuple(summary["danbooru_semantic_named_outfit_tags"]),
                status="profile_cache",
            )

    result, calls = run(evidence["original_prompt"], json.loads(summary["semantic_plan_raw"]),
                        summary["nai_character_plan_initial_raw"], {"debug_prompt_enabled": True},
                        resolver=RecordedProfiles(), repair=summary["nai_character_plan_retry_raw"])
    assert result.final_prompt, result.summary
    assert len(calls) == 3
    assert result.summary["nai_character_plan_accepted_attempt"] == "retry"
    assert any("nai_character_cross_panel_instance" in error
               for error in result.summary["nai_identity_initial_validation_errors"])
    assert result.summary["nai_identity_retry_validation_errors"] == []
    characters = result.summary["nai_characters"]
    assert len(characters) == 7
    assert [row["name"] for row in characters].count("togawa_sakiko") == 4
    assert [row["name"] for row in characters].count("chihaya_anon") == 3
    assert len({(row["x"], row["y"]) for row in characters}) == 7
    assert len(result.summary["nai_identity_review"]) == 2
    assert len(calls) == result.summary["semantic_plan_attempt_count"] + result.summary["nai_character_plan_attempt_count"]
    from agent_tools.comfyui_workflows import custom_t2i_workflow
    workflow = custom_t2i_workflow({"custom_workflow_path": str(root / "nai_api.json")},
                                  result.final_prompt, "blurry", 832, 1216, 28, 6, 123,
                                  nai_characters=characters)
    selector = next(node["inputs"] for node in workflow.values() if node.get("class_type") == "CharacterPromptSelect")
    assert selector["character7_enable"] is True
    assert not selector.get("character8_enable", False)
    for index, character in enumerate(characters, 1):
        converter = workflow[selector[f"character{index}"][0]]
        assert converter["inputs"]["comfyui_prompt"] == character["prompt"]
        assert (selector[f"character{index}_x"], selector[f"character{index}_y"]) == (character["x"], character["y"])
    assert "7girls" in result.final_prompt
