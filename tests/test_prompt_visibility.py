from __future__ import annotations

import sys
import re
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompt_constraints import build_constraint_plan_prompt
from prompt_pipeline import merge_authoritative_identity_block
from prompt_templates import build_llm_prompt
from prompt_visibility import APPEARANCE_VISIBILITY_POLICY
from multi_person_prompt import build_multi_person_plan_prompt
from test_prompt_direct_writer import run, semantic_character, plan


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
