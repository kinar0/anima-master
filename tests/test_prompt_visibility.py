from __future__ import annotations

import sys
import re
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompt_constraints import build_constraint_plan_prompt
from prompt_pipeline import appearance_dimension, appearance_dimensions, filter_character_appearance_prose, merge_authoritative_identity_block
from prompt_templates import build_llm_prompt
from prompt_visibility import APPEARANCE_VISIBILITY_POLICY
from multi_person_prompt import build_multi_person_plan_prompt
from test_prompt_direct_writer import run, semantic_character, plan


@pytest.mark.parametrize("text,tags", [
    ("粉色长发灰眼女孩", ("pink_hair", "long_hair", "grey_eyes")),
    ("蓝色短发金瞳，微笑", ("blue_hair", "short_hair", "golden_eyes")),
    ("银色头发，红眼，白皙皮肤", ("silver_hair", "red_eyes", "fair_skin")),
    ("双马尾，平胸，狐耳", ("twintails", "flat_chest", "fox_ears")),
    ("long pink hair, gray eyes, holding a book", ("long_hair", "pink_hair", "grey_eyes")),
    ("金色头发，灰色眼睛", ("blonde_hair", "gray_eyes")),
    ("平胸，粉色长发", ("(flat_chest:1.4)", "pink_hair", "long_hair")),
])
def test_nai_matching_profile_prose_keeps_original_language(text, tags):
    assert filter_character_appearance_prose(text, tags, frozenset(), preserve_matching=True) == text


@pytest.mark.parametrize("bad,good,tag", [
    ("蓝眼", "灰眼", "grey_eyes"),
    ("红色头发", "粉色头发", "pink_hair"),
    ("大胸", "平胸", "flat_chest"),
    ("双马尾", "马尾", "ponytail"),
])
def test_nai_preserving_chinese_does_not_open_conflicting_profile_facts(bad, good, tag):
    assert bad not in filter_character_appearance_prose(
        bad + "，手扶着同伴的肩膀", (tag,), frozenset(), preserve_matching=True,
    )
    assert filter_character_appearance_prose(good, (tag,), frozenset(), preserve_matching=True) == good
    dimension = appearance_dimension(tag)
    assert filter_character_appearance_prose(
        bad, (tag,), frozenset((dimension,)), preserve_matching=True,
    ) == bad


@pytest.mark.parametrize("text", ["粉色短发灰眼女孩", "short pink hair, gray eyes"])
def test_nai_compound_profile_span_checks_length_as_well_as_color(text):
    filtered = filter_character_appearance_prose(
        text, ("pink_hair", "long_hair", "grey_eyes"), frozenset(), preserve_matching=True,
    )
    assert "短" not in filtered and "short" not in filtered
    assert "灰眼" in filtered or "gray eyes" in filtered


def test_nai_camera_and_room_corner_are_not_profile_horns():
    text = "视角从上方，角色站在墙角，身后角落里有一张桌子"
    assert "horns" not in appearance_dimensions(text)
    assert filter_character_appearance_prose(text, ("horns",), frozenset(), preserve_matching=True) == text


def test_nai_matching_weighted_profile_tag_is_not_doubled_or_damaged():
    result, _ = run(
        "-r 角色甲站立", {"characters": [semantic_character("角色甲")]},
        plan([("character_a", "girl, (pink_hair:1.3), 灰眼女孩站立", .5)], "1girl"),
    )
    text = result.summary["nai_characters"][0]["prompt"]
    assert "(pink_hair:1.3)" in text and text.count("pink_hair") == 1
    assert "灰眼女孩站立" in text


def test_nai_chinese_appearance_and_global_relations_reach_generation_inputs():
    from agent_tools.comfyui_workflows import custom_t2i_workflow
    request = "-r 角色甲在左，角色乙在右，角色甲用手扶住角色乙肩膀"
    global_text = "2girls, room, 角色甲在左侧，角色乙在右侧。角色甲用自己的手扶住角色乙的肩膀。"
    output = plan([
        ("character_a", "girl, 粉色长发灰眼女孩用自己的手扶住角色乙的肩膀, pink_hair, grey_eyes", .35),
        ("character_b", "girl, 蓝色长发黄眼女孩被角色甲扶住肩膀, blue_hair, yellow_eyes", .65),
    ], global_text)
    result, calls = run(request, {"characters": [semantic_character("角色甲"), semantic_character("角色乙")]}, output)
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    assert "角色甲用自己的手扶住角色乙的肩膀" in result.final_prompt
    a, b = result.summary["nai_characters"]
    assert "粉色长发灰眼女孩" in a["prompt"]
    assert "蓝色长发黄眼女孩" in b["prompt"]
    graph = custom_t2i_workflow(
        {"custom_workflow_path": "nai_api.json"}, result.final_prompt, "blurry",
        1024, 1536, 28, 6, 123, nai_characters=result.summary["nai_characters"],
    )
    selector = graph[graph["1"]["inputs"]["characterPrompts"][0]]["inputs"]
    assert graph[selector["character1"][0]]["inputs"]["comfyui_prompt"] == a["prompt"]
    assert graph[selector["character2"][0]]["inputs"]["comfyui_prompt"] == b["prompt"]


@pytest.mark.parametrize("source_index", [0, 1])
def test_nai_contact_state_belongs_to_mouth_owner_not_hand_owner(source_index):
    from tag_cleaner import split_tags
    names = ("character_a", "character_b")
    layout = json.loads(plan([
        (name, "girl", .35 + .3 * i) for i, name in enumerate(names)
    ], "2girls, room, 两人面对面，一人将自己的手伸入另一人的嘴中。"))
    source, target = layout["characters"][source_index], layout["characters"][1-source_index]
    source.update(prompt="girl, 将自己的手伸进同伴的嘴巴", interaction_tags=["source#hand_in_mouth"])
    target.update(prompt="girl, 嘴里有同伴伸入的手, hand_in_mouth, mouth_open", interaction_tags=["target#hand_in_mouth"])
    result, _ = run("-r 角色甲和角色乙互动", {"characters": [semantic_character("角色甲"), semantic_character("角色乙")]},
                    json.dumps(layout, ensure_ascii=False))
    assert result.final_prompt, result.summary
    source = result.summary["nai_characters"][source_index]
    target = result.summary["nai_characters"][1-source_index]
    assert "hand_in_mouth" not in split_tags(source["prompt"])
    assert "source#hand_in_mouth" in split_tags(source["prompt"])
    assert "hand_in_mouth" in split_tags(target["prompt"])
    assert "target#hand_in_mouth" in split_tags(target["prompt"])


def test_nai_self_contact_and_anatomy_direction_tags_survive_profile_filtering():
    from dataclasses import replace
    from test_prompt_direct_writer import Resolver
    class Horns(Resolver):
        def cached_outfit_profiles_for_prompt(self, prompt):
            return replace(super().cached_outfit_profiles_for_prompt(prompt),
                           character_appearance_profiles=((("角色甲",), "character_a", ("horns",)),))
    output = json.loads(plan([("character_a", "girl, 自己的手放在自己的嘴里, hand_in_mouth, horns", .35),
                                          ("character_b", "girl", .65)]))
    output["characters"][0]["interaction_tags"] = ["target#touching_horns"]
    output["characters"][1]["interaction_tags"] = ["source#touching_horns"]
    result, _ = run("-r 角色甲将自己的手放在自己嘴里，角色乙摸角色甲的角",
                    {"characters": [semantic_character("角色甲"), semantic_character("角色乙")]},
                    json.dumps(output, ensure_ascii=False), resolver=Horns())
    assert result.final_prompt, result.summary
    assert "hand_in_mouth" in result.summary["nai_characters"][0]["prompt"]
    assert "target#touching_horns" in result.summary["nai_characters"][0]["prompt"]


@pytest.mark.parametrize("mode", ["txt2img", "img2img"])
def test_custom_template_without_context_placeholder_receives_visibility_policy(mode):
    prompt = build_llm_prompt("a portrait", mode=mode,
                              prompt_builder_template="CUSTOM {theme}")
    assert prompt.startswith("CUSTOM a portrait")
    assert prompt.count(APPEARANCE_VISIBILITY_POLICY) == 1


def test_other_writers_receive_same_visibility_policy():
    assert APPEARANCE_VISIBILITY_POLICY in build_multi_person_plan_prompt("two people")
    assert APPEARANCE_VISIBILITY_POLICY in build_constraint_plan_prompt("a portrait", "pink hair")


def test_low_cfg_writer_sees_final_viewpoint_without_treating_it_as_priority_tags():
    context = "character_a has pink hair; character_a faces away with her face hidden"
    prompt = build_constraint_plan_prompt("a portrait", "medium shot",
                                          visible_character_context=context)
    assert context in prompt
    assert "visibility context, not extra priority tags" in prompt


@pytest.mark.parametrize("visible", [
    "pink hair", "pink hair and fox ears", "fang",
])
def test_identity_merge_does_not_complete_unexpressed_profile(visible):
    merged = merge_authoritative_identity_block(
        "character_a", f"character_a has {visible}",
        ("pink_hair", "grey_eyes", "long_hair", "flat_chest", "fox_ears", "fang"),
        frozenset(),
    )
    assert "grey" not in merged and "long_hair" not in merged
    assert "flat_chest" not in merged
    assert ("fox ears" in merged) == ("fox ears" in visible)
    assert ("fang" in merged) == ("fang" in visible)


def test_identity_corrects_written_wrong_color_without_completing_other_features():
    merged = merge_authoritative_identity_block(
        "character_a", "character_a has blue eyes", ("pink_hair", "grey_eyes", "fang"),
        frozenset(),
    )
    assert "blue eyes" not in merged and "grey_eyes" in merged
    assert "pink_hair" not in merged and "fang" not in merged


@pytest.mark.parametrize("nai", [False, True])
@pytest.mark.parametrize("face,eyes", [
    ("from behind, face hidden", ""),
    ("head out of frame, hands only", ""),
    ("closed eyes", ""),
    ("hair covering eyes", ""),
    ("from behind, looking back, one eye visible", "grey eyes"),
    ("from behind, face visible in a clear mirror reflection", "grey eyes"),
    ("side face, eye without highlights", "grey eyes"),
])
def test_writer_visibility_is_retained_per_wearer_in_both_modes(nai, face, eyes):
    semantic = {"characters": [semantic_character("角色甲"), semantic_character("角色乙")]}
    visible_hair = "head out of frame" not in face
    a = ", ".join(item for item in ("girl", "pink hair" if visible_hair else "", eyes, face) if item)
    if nai:
        writer = plan([("character_a", a, .25),
                       ("character_b", "girl, blue hair, yellow eyes, front view", .75)])
    else:
        identity_a = (f"character_a has pink hair{(' and ' + eyes) if eyes else ''}"
                      if visible_hair else "character_a is a girl")
        writer = (
            "{Count: 2girls}{Characters: character_a, character_b}{Copyright: }"
            f"{{Identity: {identity_a}; "
            "character_b has blue hair and yellow eyes}"
            f"{{Details: character_a is {face}; character_b is facing the camera}}"
            "{Tags: two people, medium shot}{Nltags: }"
        )
    # No explicit omit in LLM1: the writer chooses the final viewpoint itself.
    result, calls = run(("-r " if nai else "") + "角色甲和角色乙合影", semantic, writer,
                        {"debug_prompt_enabled": True})
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    assert APPEARANCE_VISIBILITY_POLICY in calls[1]["prompt"]
    if nai:
        a_text, b_text = (item["prompt"] for item in result.summary["nai_characters"])
    else:
        a_text = re.search(r"character a (?:has|is) ([^;\n]+)", result.final_prompt).group(1)
        b_text = re.search(r"character b has ([^;\n]+)", result.final_prompt).group(1)
    assert ("grey" in a_text) == bool(eyes)
    assert ("pink" in a_text) == visible_hair
    assert "yellow" in b_text
    assert "blue" in b_text


def test_same_identity_instances_do_not_share_implicit_visibility():
    result, calls = run(
        "-r 左格角色甲背向，右格角色甲正面",
        {"characters": [semantic_character("角色甲"), semantic_character("角色甲")]},
        plan([("character_a", "girl, pink hair, from behind", .25),
              ("character_a", "girl, pink hair, blue eyes, front view", .75)]),
    )
    assert result.final_prompt, result.summary
    assert len(calls) == 2
    first, second = result.summary["nai_characters"]
    assert "grey" not in first["prompt"] and "blue eyes" not in second["prompt"]
    assert "grey_eyes" in second["prompt"]


def test_replacement_fact_does_not_force_hidden_eyes_into_prompt():
    result, _ = run(
        "-r 角色甲的眼睛改成蓝色，背向观众",
        {"characters": [semantic_character("角色甲", changes=[dict(
            dimension="eye_color", operation="replace", source_text="眼睛改成蓝色")])]},
        plan([("character_a", "girl, pink hair, from behind", .5)], "1girl"),
    )
    assert result.final_prompt, result.summary
    assert "eyes" not in result.summary["nai_characters"][0]["prompt"]
