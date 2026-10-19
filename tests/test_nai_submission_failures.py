"""Submission prerequisites recover once without weakening identity/text authority."""
from __future__ import annotations

from dataclasses import replace
import json

import pytest

from danbooru_semantic import SemanticAnchor, SemanticLookupResult
from nai_character_mode import parse_nai_character_plan
from nai_text import split_nai_text_tail
from prompt_pipeline import nai_local_identity_mentions, review_nai_identities
from task_state import TaskRecorder
from test_prompt_direct_writer import plan, run, semantic_character, with_identity_review
from test_nai_text import _nai_config, _with_regions


GOOD = plan([("character_a", "girl, sitting, 角色甲坐在花园中。", .3),
             ("character_b", "girl, standing, 角色乙站在花园中。", .7)])
REQUEST = "-r 角色甲坐在花园中，角色乙站在花园中。"
SEMANTIC = {"characters": [semantic_character("角色甲"), semantic_character("角色乙")]}


def broken_plan(defect):
    if defect == "json":
        return GOOD.replace(', "interaction_tags"', '}, "interaction_tags"', 1)
    data = json.loads(GOOD)
    if defect == "position":
        data["characters"][0].pop("x")
    elif defect == "layout":
        data["composition_analysis"]["projection"] = data["composition_analysis"].pop("layout")
    elif defect == "relations":
        data["composition_analysis"]["关系"] = data["composition_analysis"].pop("relations")
    elif defect == "reason":
        data["characters"][0].pop("position_reason")
    elif defect == "count":
        data["characters"] = []
    return json.dumps(data, ensure_ascii=False)


@pytest.mark.parametrize("defect,error", [
    ("json", "invalid_json"), ("position", "invalid_position"),
    ("layout", "missing_composition_analysis"), ("relations", "missing_composition_analysis"),
    ("reason", "missing_position_reason"), ("count", "character_count"),
])
@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_structure_uses_one_shared_repair_and_preserves_the_original_error(defect, error, repair_succeeds):
    broken = broken_plan(defect)
    result, calls = run(REQUEST, SEMANTIC, broken, {"debug_prompt_enabled": True},
                        repair=GOOD if repair_succeeds else broken)
    assert len(calls) == 3
    s = result.summary
    assert s["nai_character_plan_attempt_count"] == 2
    assert s["nai_structure_initial_validation_errors"] == ["nai_character_plan_" + error]
    assert s["nai_character_plan_initial_raw"] == broken
    assert "numeric x/y" in calls[-1]["prompt"] and "Previous JSON:" in calls[-1]["prompt"]
    if repair_succeeds:
        assert result.final_prompt, s
        assert s["nai_structure_retry_validation_errors"] == []
        assert s["nai_character_plan_accepted_attempt"] == "retry"
        assert [row["x"] for row in s["nai_characters"]] == [.3, .7]
    else:
        assert not result.final_prompt
        assert s["llm_error"] == "nai_character_plan_" + error
        assert s["llm_failed"] is True
        recorder = TaskRecorder.__new__(TaskRecorder)
        task = {}
        recorder.mark_prompt_built(task, s)
        assert task["prompt_summary"]["llm_ok"] is False


def test_structure_repair_cannot_get_another_budget_for_missing_known_identity():
    only_a = plan([("character_a", "girl, sitting", .3)])
    result, calls = run(REQUEST, SEMANTIC, broken_plan("json"), repair=only_a)
    assert not result.final_prompt and len(calls) == 3
    assert result.summary["nai_identity_retry_validation_errors"]


def test_structure_only_failure_does_not_skip_first_usable_donor_semantic_review():
    from test_nai_identity_review import Profiles, VARIANT

    source = "角色甲和变体甲衣服略显凌乱地坐在一起，角色乙在旁边站立。"
    output = with_identity_review(GOOD, VARIANT, "donor", source)
    result, calls = run("-r " + source, SEMANTIC, broken_plan("json"),
                        resolver=Profiles(), repair=output)
    assert not result.final_prompt and len(calls) == 3
    assert "Re-evaluate donor-only exclusion" in result.summary["llm_error"]


def identity_evidence(*, confirmed_b=True, source_b=False):
    a = SemanticAnchor("target_1", "target_character", "character", "角色甲", "", ("character_a",))
    b = SemanticAnchor("source_1" if source_b else "target_2",
                       "outfit_source" if source_b else "target_character",
                       "character", "访客乙", "", ("character_b",), source_category=4)
    return SemanticLookupResult(
        confirmed_tags=("character_a", "character_b") if confirmed_b else ("character_a",),
        anchors=(a, b), anchor_tags=((a.anchor_id, "character_a"), (b.anchor_id, "character_b")) if confirmed_b
        else ((a.anchor_id, "character_a"),),
        character_appearance_profiles=((("角色甲",), "character_a", ("pink_hair",)),),
    )


@pytest.mark.parametrize("role", ["visible", "offscreen", "quoted"])
def test_lookup_confirmed_identity_needs_no_saved_profile(role):
    source = "角色甲站立，访客乙在旁边站立。" if role == "visible" else "角色甲站立，访客乙仅在画外或引文提及。"
    result = identity_evidence(source_b=role != "visible")
    mentions = nai_local_identity_mentions(source, result)
    assert any(row.candidates == ("character_b",) for row in mentions)
    output = json.loads(GOOD)
    if role != "visible":
        output["characters"] = output["characters"][:1]
    output["identity_review"] = [dict(name="访客乙", visibility=role, source_text=source)]
    parsed = parse_nai_character_plan(json.dumps(output))
    assert review_nai_identities(parsed, mentions, result, result.anchors, source) == ()
    assert parsed["validated_identity_review"][0]["name"] == "character_b"


@pytest.mark.parametrize("kind", ["missing", "bad_source", "contradiction", "source_promotion"])
def test_lookup_only_identity_keeps_required_review_and_visibility_checks(kind):
    source = "角色甲站立，访客乙站在旁边。"
    result = identity_evidence(source_b=kind == "source_promotion")
    output = json.loads(GOOD)
    if kind == "missing":
        output["characters"] = output["characters"][:1]
    else:
        output["identity_review"] = [dict(name="访客乙", visibility="offscreen" if kind == "contradiction" else "visible",
                                          source_text="不存在的原文" if kind in {"bad_source", "source_promotion"} else source)]
    parsed = parse_nai_character_plan(json.dumps(output))
    issues = review_nai_identities(parsed, nai_local_identity_mentions(source, result), result, result.anchors, source)
    # A returned confirmed target's optional visible quote can be dropped; its
    # source-only promotion cannot. Contradictions and missing targets still fail.
    if kind == "bad_source":
        assert not issues and parsed["dropped_identity_reviews"]
    else:
        assert issues


def test_unconfirmed_candidate_or_general_outfit_tag_cannot_become_identity_authority():
    result = identity_evidence(confirmed_b=False)
    assert len(nai_local_identity_mentions("角色甲和访客乙站立", result)) == 1
    result = identity_evidence()
    result = replace(result, anchors=(result.anchors[0], replace(result.anchors[1], source_category=0)))
    assert len(nai_local_identity_mentions("角色甲和访客乙站立", result)) == 1


def test_conflicting_lookup_alias_does_not_override_profile_owner():
    result = identity_evidence()
    result = replace(result, character_appearance_profiles=(
        *result.character_appearance_profiles,
        (("访客乙",), "different_character", ("blue_hair",)),
    ), confirmed_tags=(*result.confirmed_tags, "different_character"))
    assert not any(row.source_text == "访客乙"
                   for row in nai_local_identity_mentions("角色甲和访客乙站立", result))


@pytest.mark.parametrize("visible", [True, False])
def test_optional_unknown_review_is_diagnostic_and_grants_no_identity_or_profile(visible):
    source = "-r 角色甲站立，陌生访客在旁边。" if visible else "-r 角色甲站立，牌子上只提及陌生访客的名字。"
    rows = [("character_a", "girl, standing", .3)]
    if visible:
        rows.append(("陌生访客", "other, standing", .7))
    output = with_identity_review(plan(rows), "陌生访客", "visible" if visible else "quoted", source[3:])
    result, calls = run(source, {"characters": [semantic_character("角色甲")]}, output)
    assert result.final_prompt and len(calls) == 2
    assert result.summary["nai_identity_review"] == []
    assert result.summary["nai_dropped_identity_reviews"][0]["reason"] == "unconfirmed_identity_review"
    assert len(result.summary["nai_characters"]) == len(rows)


def test_unknown_review_cannot_excuse_missing_known_identity_or_contradict_visible_box():
    result, calls = run(REQUEST, SEMANTIC,
                        with_identity_review(plan([("character_a", "girl, standing", .3)]),
                                             "陌生人", "quoted", "不存在的句子"),
                        repair=with_identity_review(plan([("character_a", "girl, standing", .3)]),
                                                    "陌生人", "quoted", "不存在的句子"))
    assert not result.final_prompt and len(calls) == 3
    output = with_identity_review(plan([("character_a", "girl, standing", .3),
                                       ("陌生访客", "other, standing", .7)]),
                                  "陌生访客", "offscreen", "陌生访客在旁边")
    result, calls = run("-r 角色甲站立，陌生访客在旁边", {"characters": [semantic_character("角色甲")]},
                        output, repair=output)
    assert not result.final_prompt and len(calls) == 3
    assert "visibility disagrees" in result.summary["llm_error"]


def test_conflicting_unknown_review_rows_are_not_silently_discarded():
    data = json.loads(plan([("character_a", "girl, standing", .3)]))
    data["identity_review"] = [dict(name="陌生人", visibility=role, source_text="陌生人只是引用")
                               for role in ("visible", "quoted")]
    output = json.dumps(data, ensure_ascii=False)
    result, calls = run("-r 角色甲站立，陌生人只是引用。", {"characters": [semantic_character("角色甲")]},
                        output, repair=output)
    assert not result.final_prompt and len(calls) == 3
    assert "conflicting visibility" in result.summary["llm_error"]


def test_unnamed_writer_designed_panels_repair_to_empty_source_labels_with_distinct_owners():
    source = '两格漫画，角色甲先挥手说“你好”，随后坐下说“再见”。'
    output = plan([("character_a", "girl, waving, 上格的角色甲挥手。", .25),
                   ("character_a", "girl, sitting, 下格的角色甲坐下。", .75)], "2girls, comic, 上下两格。")
    regions = [(0, "speech", 0, "第一格", "上格角色甲头顶的气泡写NAILITERAL0000TOKEN。"),
               (1, "speech", 1, "第二格", "下格角色甲头顶的气泡写NAILITERAL0001TOKEN。")]
    initial = _with_regions(output, source, regions)
    fixed = _with_regions(output, source, [(i, kind, owner, "", placement)
                                          for i, kind, owner, _panel, placement in regions])
    result, calls = run("-r " + source, {"characters": [semantic_character("角色甲")]},
                        initial, _nai_config(), repair=fixed)
    assert result.final_prompt and len(calls) == 3
    assert result.summary["nai_text_initial_validation_errors"] == ["nai_text_ungrounded_panel: 0", "nai_text_ungrounded_panel: 1"]
    assert result.summary["nai_text_retry_validation_errors"] == []
    assert 'use panel=""' in calls[-1]["prompt"]
    assert [split_nai_text_tail(row["prompt"])[1] for row in result.summary["nai_characters"]] == ["你好", "再见"]
