from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from agent_tools.comfyui_command_runner import run_cli_action  # noqa: E402
from agent_tools.comfyui_workflows import custom_t2i_workflow, workflow  # noqa: E402
from generation_task import GenerationTaskRunner  # noqa: E402
from nai_character_mode import (  # noqa: E402
    build_nai_character_plan_prompt,
    downgrade_nai_interaction_tags,
    missing_nai_noun_tags,
    nai_interaction_validation_issues,
    parse_nai_character_plan,
    preserve_nai_global_artist_tags,
    resolve_nai_canvas,
    strip_nai_character_switch,
)


@pytest.mark.parametrize("tags,valid", [
    ([[], []], True),  # individual actions, proximity, gaze: no invented relation
    ([["source#feeding"], ["target#feeding"]], True),
    ([["source#Hug"], ["target#hug"]], True),
    ([["source#helping_up"], ["target#helping up"]], True),
    ([["mutual#hug"], ["mutual#hug"]], True),
    ([["source#hug"], []], False),
    ([[], ["target#hug"]], False),
    ([["source#hug"], ["target#kiss"]], False),
    ([["mutual#hug"], []], False),
    ([["source#hug", "target#hug"], []], False),
    ([["mutual#hug", "source#hug"], ["mutual#hug", "target#hug"]], False),
    ([["source#feeding"], ["target#feeding"], ["mutual#hug"], ["mutual#hug"]], True),
    ([["source#hug"], ["target#hug"], ["source#hug"], ["target#hug"]], True),
])
def test_interaction_review_checks_pairs_without_a_fixed_action_dictionary(tags, valid):
    parsed = parse_nai_character_plan(json.dumps({
        "composition_analysis": {"viewpoint": "side", "layout": "contact", "relations": "reviewed"},
        "global_prompt": f"{len(tags)}people, indoors",
        "characters": [{"name": f"instance_{index}", "prompt": "girl",
                        "interaction_tags": actions, "position_reason": "visible region",
                        "x": .2 + .15 * index, "y": .5}
                       for index, actions in enumerate(tags)],
    }))
    assert bool(nai_interaction_validation_issues(parsed)) is not valid
    original_prompts = [row["prompt"] for row in parsed["characters"]]
    warnings = downgrade_nai_interaction_tags(parsed)
    assert bool(warnings) is not valid
    assert nai_interaction_validation_issues(parsed) == ()
    if valid:
        assert [row["prompt"] for row in parsed["characters"]] == original_prompts
    assert downgrade_nai_interaction_tags(parsed) == ()  # idempotent


def test_uncertain_direction_group_preserves_valid_other_actions_and_opaque_prose():
    parsed = parse_nai_character_plan(json.dumps({
        "composition_analysis": dict(viewpoint="front", layout="pair", relations="help and wave"),
        "global_prompt": "2girls, indoors",
        "characters": [
            dict(name="A", prompt="girl, helping up, (white shirt:1.2), 扶起B并挥手。NAILITERAL0000TOKEN",
                 interaction_tags=["source#helping_up", "source#waving"],
                 position_reason="beside B", x=.4, y=.5),
            dict(name="B", prompt="girl, standing, 被A扶起。",
                 interaction_tags=["target#helping up"],
                 position_reason="beside A", x=.6, y=.5),
        ],
    }, ensure_ascii=False))
    assert downgrade_nai_interaction_tags(parsed)
    assert parsed["characters"][0]["interaction_tags"] == ["source#helping_up"]
    assert parsed["characters"][1]["interaction_tags"] == ["target#helping up"]
    assert "source#waving" not in parsed["characters"][0]["prompt"]
    for text in ("source#helping_up", "(white shirt:1.2)", "扶起B并挥手。NAILITERAL0000TOKEN"):
        assert text in parsed["characters"][0]["prompt"]
    assert parsed["dropped_interaction_tags"] == ["source#waving"]


@pytest.mark.parametrize("action", ["holding_hands", "hug", "helping_up"])
def test_directional_action_deduplicates_ordinary_action_and_keeps_relation_prose(action):
    from tag_cleaner import split_tags
    parsed = parse_nai_character_plan(json.dumps({
        "composition_analysis": {"viewpoint": "side", "layout": "pair", "relations": "contact"},
        "global_prompt": "3girls, bed",
        "characters": [
            {"name": "A", "prompt": f"girl, {action.replace('_', ' ')}, holding a spoon, 用手帮助B",
             "interaction_tags": [f"source#{action}"], "position_reason": "left", "x": .3, "y": .5},
            {"name": "B", "prompt": f"girl, {action}, mouth_open, 接受A的帮助",
             "interaction_tags": [f"target#{action}"], "position_reason": "right", "x": .7, "y": .5},
            {"name": "C", "prompt": f"girl, {action}", "interaction_tags": [],
             "position_reason": "behind", "x": .5, "y": .6},
        ],
    }, ensure_ascii=False))
    source, target, individual = parsed["characters"]
    assert downgrade_nai_interaction_tags(parsed) == ()
    for row, role in ((source, "source"), (target, "target")):
        tags = split_tags(row["prompt"])
        assert action not in tags and action.replace('_', ' ') not in tags
        assert f"{role}#{action}" in tags
    assert "holding a spoon" in source["prompt"] and "用手帮助B" in source["prompt"]
    assert "mouth_open" in target["prompt"] and "接受A的帮助" in target["prompt"]
    assert action in split_tags(individual["prompt"])
    assert missing_nai_noun_tags(parsed, (action,)) == ()


@pytest.mark.parametrize("action", ["hand_in_mouth", "examining_unseen_detail", "hug"])
def test_paired_actions_remove_only_exact_duplicates_after_downgrade(action):
    from tag_cleaner import split_tags
    ordinary = action.replace("_", " ")
    parsed = parse_nai_character_plan(json.dumps({
        "composition_analysis": dict(viewpoint="close", layout="contact", relations="A acts on B"),
        "global_prompt": f"3girls, bed, {action}",
        "characters": [
            dict(name="A", prompt=f"girl, ({ordinary}:1.2), {action}_focus, A帮助B。",
                 interaction_tags=[f"source#{action}"], position_reason="near B", x=.4, y=.5),
            dict(name="B", prompt=f"girl, {action}, open_mouth, B接受A的帮助。",
                 interaction_tags=[f"target#{action}"], position_reason="near A", x=.6, y=.5),
            dict(name="C", prompt=f"girl, {action}, C独自做动作。", interaction_tags=[],
                 position_reason="separate", x=.8, y=.7),
        ],
    }, ensure_ascii=False))
    assert downgrade_nai_interaction_tags(parsed) == ()
    a, b, c = parsed["characters"]
    assert f"({ordinary}:1.2)" not in split_tags(a["prompt"])
    assert action not in split_tags(b["prompt"])
    assert action in split_tags(c["prompt"])
    assert f"{action}_focus" in a["prompt"]
    assert "open_mouth" in b["prompt"] and "A帮助B。" in a["prompt"]
    assert action not in split_tags(parsed["global_prompt"])
    records = list(parsed["dropped_redundant_action_tags"])
    assert downgrade_nai_interaction_tags(parsed) == ()
    assert parsed["dropped_redundant_action_tags"] == records


def test_uncertain_pair_retains_ordinary_body_state_and_does_not_guess_synonyms():
    parsed = parse_nai_character_plan(json.dumps({
        "composition_analysis": dict(viewpoint="close", layout="pair", relations="contact"),
        "global_prompt": "2girls, bed",
        "characters": [
            dict(name="A", prompt="girl, finger_in_mouth, 自己的手指放在嘴里。",
                 interaction_tags=["source#finger_in_mouth"], position_reason="left", x=.4, y=.5),
            dict(name="B", prompt="girl, hand_in_mouth, 接触关系不明确。",
                 interaction_tags=["target#hand_in_mouth"], position_reason="right", x=.6, y=.5),
        ],
    }, ensure_ascii=False))
    assert downgrade_nai_interaction_tags(parsed)
    assert "finger_in_mouth" in parsed["characters"][0]["prompt"]
    assert "hand_in_mouth" in parsed["characters"][1]["prompt"]
    assert parsed["dropped_redundant_action_tags"] == []


def test_direct_writer_reviews_composition_tag_ownership_and_directions_in_one_call():
    instruction = build_nai_character_plan_prompt(
        "A feeds B", "local identity evidence",
        {"width": 1216, "height": 832, "aspect_ratio": "19:13", "orientation": "landscape"},
    )
    assert "Before output, review scene completeness" in instruction
    assert "source#tag" in instruction and "target#tag" in instruction and "mutual#tag" in instruction
    assert "same action tag" in instruction
    assert "Reassign misplaced global instance conditions" in instruction
    assert "actual canvas" in instruction and "panel boundaries" in instruction
    assert "coherent natural language" in instruction
    assert "seven-field" not in instruction and "same writing call" not in instruction
    assert "seven semantic responsibilities" not in instruction



def test_nai_writer_contract_separates_counts_and_requires_chinese_prompt_prose() -> None:
    instruction = build_nai_character_plan_prompt(
        "书生角色甲站立", "角色甲 => character_a",
        {"width": 832, "height": 1216, "aspect_ratio": "13:19", "orientation": "portrait"},
    )
    assert "unnumbered gender tag girl, boy, or other" in instruction
    assert "Ngirls, Mboys for a mixed group" in instruction
    assert "do not infer a gender change from titles" in instruction
    assert "retain the user's original Chinese" in instruction
    assert "Chinese prose supplements English tags; neither replaces the other" in instruction
    assert "never translate the tag layer into Chinese keywords" in instruction
    assert "Identity, gender and interaction_tags alone do not satisfy" in instruction
    assert "Chinese prose must be complete sentences" in instruction
    assert "three female instances use 3girls, not a list" in instruction
    assert "inside each actual instance prompt" in instruction
    assert "Do not translate all prose into English or output tags alone" in instruction
    assert "without a one-sentence limit" in instruction
    assert "Refer to other participants by name or position" in instruction
    assert "Chinese only in source_text, analysis or position_reason does not satisfy" in instruction
    assert "whose limb contacts whose body part" in instruction
    assert "A source prefix grants agency, not ownership" in instruction
    assert "only to the owner of the affected body part" in instruction
    assert "without repeating the same action as a bare or weighted" in instruction
    assert "A shared outfit does not imply shared personal accessories" in instruction
    assert "First identify the visual focus" in instruction
    assert "compare close cropping, an oblique view and an over-shoulder" in instruction
    assert "closed or clenched mouth" in instruction
    assert "Include scene-level prose stating who is where" in instruction
    assert "For a Chinese request this prose is Chinese" in instruction
    assert "Add lighting descriptions only when requested" in instruction
    assert "Compare tags with prose for each expressed appearance" in instruction
    assert "even when everyone wears the same named outfit" in instruction
    assert "source_text must quote the exact applicable request clause" in instruction
    assert "Otherwise leave it empty" in instruction
    assert "original language, without translation" in instruction
    assert "Write all prompt strings in English" not in instruction
    assert "artist tags, quality" not in instruction


@pytest.mark.parametrize("tag,expected", [("1girl", "girl"), ("1boy", "boy"), ("2girls", "girl"), ("1person", "other")])
def test_nai_instance_population_tags_are_normalized_keeping_chinese_content(tag, expected):
    raw = json.dumps({
        "composition_analysis": {"viewpoint": "正面", "layout": "中央", "relations": "单人"},
        "global_prompt": "1girl, 白色背景",
        "characters": [{"name": "character_a", "source_text": "角色甲身穿道袍",
                        "prompt": tag + ", 身穿道袍，抬手作揖", "position_reason": "位于中央", "x": .5, "y": .5}],
    }, ensure_ascii=False)
    parsed = parse_nai_character_plan(raw)
    character = parsed["characters"][0]
    assert character["prompt"] == "character_a, " + expected + ", 身穿道袍, 抬手作揖"
    assert character["source_text"] == "角色甲身穿道袍"
    assert parsed["global_prompt"] == "1girl, 白色背景"


def test_nai_noun_coverage_diagnostic_tracks_standalone_character_scope() -> None:
    canvas = {"width": 1024, "height": 1536, "aspect_ratio": "2:3", "orientation": "portrait"}
    instruction = build_nai_character_plan_prompt(
        "巨乳女孩站立", "1girl, large breasts", canvas,
        noun_mappings=(("巨乳", "large_breasts"),),
    )
    assert "巨乳 => large_breasts" in instruction
    assert "not a required tag checklist" in instruction
    assert "does not replace the standalone tag" not in instruction
    plan = {
        "global_prompt": "1girl, white background, large breasts",
        "characters": [{"name": "girl", "prompt": "girl has large breasts"}],
    }
    assert missing_nai_noun_tags(plan, ("large_breasts",)) == ("large_breasts",)
    plan["characters"][0]["prompt"] = "girl, large breasts"
    assert missing_nai_noun_tags(plan, ("large_breasts",)) == ()
    plan["global_prompt"] = "1girl, garden"
    assert missing_nai_noun_tags(plan, ("garden",)) == ()

    raw = json.dumps({
        "composition_analysis": {
            "viewpoint": "front", "layout": "pair", "relations": "standing"
        },
        "global_prompt": "2girls, tutu, garden",
        "characters": [
            {"name": "A", "prompt": "A, tutu", "position_reason": "left", "x": .3, "y": .5},
            {"name": "B", "prompt": "B, suit", "position_reason": "right", "x": .7, "y": .5},
        ],
    })
    parsed = parse_nai_character_plan(raw, character_scoped_tags=("tutu",))
    assert parsed["global_prompt"] == "2girls, garden"
    assert parsed["dropped_global_character_tags"] == ["tutu"]
    assert missing_nai_noun_tags(
        parsed, ("tutu",), character_scoped_tags=("tutu",)
    ) == ()


@pytest.mark.parametrize("direction", ["source", "target", "mutual"])
@pytest.mark.parametrize("action", ["interlocked_fingers", "holding_hands", "hug"])
def test_nai_noun_action_survives_valid_directional_character_tag(direction, action):
    raw = json.dumps({
        "composition_analysis": {
            "viewpoint": "side", "layout": "pair", "relations": "interaction"
        },
        "global_prompt": "2girls, underwater",
        "characters": [{
            "name": "A", "prompt": "girl, reaching toward B",
            "interaction_tags": [f"{direction}#{action.replace('_', ' ')}"],
            "position_reason": "left", "x": .3, "y": .5,
        }, {
            "name": "B", "prompt": "girl", "position_reason": "right",
            "x": .7, "y": .5,
        }],
    })
    plan = parse_nai_character_plan(raw)

    assert missing_nai_noun_tags(
        plan, (action,), character_scoped_tags=(action,)
    ) == ()
    # A related action is not evidence for a different canonical mapping.
    assert missing_nai_noun_tags(plan, (f"{action}_other",)) == (f"{action}_other",)


@pytest.mark.parametrize("prompt", [
    "girl, hands joined in interlocked_fingers",
    "girl, source#interlocked_fingers#hug",
    "girl, source#interlocked_fingers, hug",
    "girl, source#interlocked_fingers and hug",
])
def test_nai_noun_action_rejects_prose_and_discarded_directional_copies(prompt):
    raw = json.dumps({
        "composition_analysis": {
            "viewpoint": "side", "layout": "pair", "relations": "interaction"
        },
        "global_prompt": "2girls, underwater, mutual#interlocked_fingers",
        "characters": [{
            "name": "A", "prompt": prompt, "interaction_tags": [],
            "position_reason": "left", "x": .3, "y": .5,
        }],
    })
    plan = parse_nai_character_plan(raw)

    assert missing_nai_noun_tags(plan, ("interlocked_fingers",)) == (
        "interlocked_fingers",
    )


def test_nai_noun_action_requires_directional_tag_in_emitted_prompt():
    plan = {
        "global_prompt": "2girls, underwater",
        "characters": [{
            "prompt": "girl, reaching",
            "interaction_tags": ["mutual#interlocked_fingers"],
        }],
    }
    assert missing_nai_noun_tags(plan, ("interlocked_fingers",)) == (
        "interlocked_fingers",
    )


def test_nai_global_restores_exact_configured_artist_tags():
    artists = "(dishwasher1910:0.864), (yd_(orange_maru):1.1), year 2024"
    missing = preserve_nai_global_artist_tags("2girls, bedroom", artists)
    assert missing == f"(dishwasher1910:0.864), (yd_(orange_maru):1.1), year 2024, 2girls, bedroom"
    malformed = preserve_nai_global_artist_tags(
        r"2girls, \(yd \(orange maru\)\:1.1\), bedroom", artists
    )
    assert malformed == missing
    assert preserve_nai_global_artist_tags(
        r"2girls, \(yd \(orange maru\)\):1.1, bedroom", artists
    ) == missing


def test_nai_global_restores_numeric_emphasis_artist_tags():
    artists = "1.2::artist:banpai akira ::, -0.5:: lips::"

    assert preserve_nai_global_artist_tags("2girls, bedroom", artists) == (
        f"{artists}, 2girls, bedroom"
    )


def nai_graph() -> dict:
    return json.loads((PLUGIN_DIR / "nai_api.json").read_text(encoding="utf-8"))


def build(
    tmp_path: Path, graph: dict, *, override=False, size=False, width=1216, height=832
) -> dict:
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(graph), encoding="utf-8")
    return workflow(
        {
            "custom_workflow_enabled": True,
            "custom_workflow_path": str(path),
            "custom_workflow_override_parameters": override,
            "sampler_name": "er_sde",
            "scheduler": "normal",
        },
        "(watercolor:1.2), 1girl",
        "blurry",
        width,
        height,
        20,
        4.5,
        42,
        override_size=size,
    )


def test_nai_replaces_prompt_through_conversion_and_preserves_export(tmp_path):
    graph = nai_graph()
    before = copy.deepcopy(graph)
    result = build(tmp_path, graph)
    assert result["3"]["inputs"]["text"] == "(watercolor:1.2), 1girl"
    assert result["11"]["inputs"]["text"] == "blurry"
    assert result["9"] == before["9"]
    assert result["12"] == before["12"]
    expected = dict(before["1"]["inputs"], seed=42)
    assert result["1"]["inputs"] == expected
    assert result["10"]["inputs"]["images"] == ["1", 0]
    assert result["10"]["inputs"]["filename_prefix"].startswith("astrbot/")
    assert graph == before
    # Loading/patching must not overwrite the user's saved workflow.
    assert json.loads((tmp_path / "workflow.json").read_text()) == before


def test_nai_r_switch_and_repeated_character_instances(tmp_path):
    enabled, prompt = strip_nai_character_switch(
        "-r 左格高松灯穿校服，右格高松灯穿礼服"
    )
    assert enabled and prompt == "左格高松灯穿校服，右格高松灯穿礼服"
    assert strip_nai_character_switch("girl-ribbon, -raw") == (False, "girl-ribbon, -raw")
    plan = parse_nai_character_plan(json.dumps({
        "composition_analysis": {
            "viewpoint": "front view", "layout": "two panels", "relations": "separate poses",
        },
        "global_prompt": "2girls, takamatsu tomori, artist:foo, split screen, white background",
        "characters": [
            {"name": "takamatsu tomori", "prompt": "school uniform, kneeling", "position_reason": "left panel", "x": 0.2, "y": 0.5},
            {"name": "takamatsu tomori", "prompt": "evening dress, standing", "position_reason": "right panel", "x": 0.8, "y": 0.5},
        ],
    }))
    assert "takamatsu tomori" not in plan["global_prompt"]
    assert "artist:foo" in plan["global_prompt"]
    result = custom_t2i_workflow(
        {"custom_workflow_path": "nai_api.json"},
        plan["global_prompt"], "blurry", 1024, 1536, 28, 6, 123,
        nai_characters=plan["characters"],
    )
    selector_id = result["1"]["inputs"]["characterPrompts"][0]
    selector = result[selector_id]["inputs"]
    assert result["3"]["inputs"]["text"] == plan["global_prompt"]
    assert selector["character2_enable"] is True
    assert selector["character3_enable"] is False
    assert (selector["character1_x"], selector["character2_x"]) == (0.2, 0.8)
    first = result[selector["character1"][0]]["inputs"]["comfyui_prompt"]
    second = result[selector["character2"][0]]["inputs"]["comfyui_prompt"]
    assert "school uniform" in first and "evening dress" in second
    assert result[selector_id]["class_type"] == "CharacterPromptSelect"
    converter_ids = []
    for index in (1, 2):
        link = selector[f"character{index}"]
        assert link[1] == 0
        assert result[link[0]]["class_type"] == "ComfyUIToNovelAIV4"
        converter_ids.append(link[0])
    assert len(set(converter_ids)) == 2
    assert selector["character3"] == ""


def test_character_conversion_keeps_weight_escape_prose_and_direction_inputs():
    characters = [
        {"name": "A", "prompt": r"girl, character \(series\), (pink hair:1.2), source#hug, holding B gently", "x": .2, "y": .4},
        {"name": "B", "prompt": r"girl, (blue hair:0.9), target#hug, leaning against A", "x": .8, "y": .6},
    ]
    graph = custom_t2i_workflow(
        {"custom_workflow_path": "nai_api.json"}, "2girls", "blurry",
        1024, 1536, 28, 6, 123, nai_characters=characters,
    )
    selector = graph[graph["1"]["inputs"]["characterPrompts"][0]]["inputs"]
    for index, character in enumerate(characters, 1):
        node_id, port = selector[f"character{index}"]
        assert port == 0 and graph[node_id]["class_type"] == "ComfyUIToNovelAIV4"
        assert graph[node_id]["inputs"]["comfyui_prompt"] == character["prompt"]
        assert selector[f"character{index}_x"] == character["x"]
        assert selector[f"character{index}_y"] == character["y"]


def test_nai_directional_action_tags_reach_the_correct_character_boxes():
    plan = parse_nai_character_plan(json.dumps({
        "composition_analysis": {
            "viewpoint": "front view", "layout": "two figures", "relations": "A hugs B",
        },
        "global_prompt": "2girls, source#hug, indoors",
        "characters": [
            {"name": "A", "prompt": "girl, blue hair, hugging B", "interaction_tags": ["source#hug"],
             "position_reason": "left", "x": .3, "y": .5},
            {"name": "B", "prompt": "girl, pink hair, target#hug", "interaction_tags": ["target#hug"],
             "position_reason": "right", "x": .7, "y": .5},
        ],
    }))
    assert plan["global_prompt"] == "2girls, indoors"
    assert plan["characters"][0]["prompt"].endswith("source#hug")
    assert plan["characters"][1]["prompt"].count("target#hug") == 1
    result = custom_t2i_workflow(
        {"custom_workflow_path": "nai_api.json"}, plan["global_prompt"],
        "blurry", 1024, 1536, 28, 6, 123, nai_characters=plan["characters"],
    )
    selector = result[result["1"]["inputs"]["characterPrompts"][0]]["inputs"]
    first = result[selector["character1"][0]]["inputs"]["comfyui_prompt"]
    second = result[selector["character2"][0]]["inputs"]["comfyui_prompt"]
    assert "source#hug" in first and "target#hug" not in first
    assert "target#hug" in second and "source#hug" not in second


def test_nai_gaze_tags_are_character_scoped_and_removed_from_global_prompt():
    plan = parse_nai_character_plan(json.dumps({
        "composition_analysis": {
            "viewpoint": "front", "layout": "close pair", "relations": "different gazes",
        },
        "global_prompt": (
            "2girls, looking at viewer, eye_contact, looking down, sideways glance, "
            "indoors, close-up, eye-level"
        ),
        "characters": [
            {"name": "togawa sakiko", "prompt": "face toward viewer, looking at viewer",
             "position_reason": "left", "x": .36, "y": .42},
            {"name": "chihaya anon", "prompt": (
                "face toward viewer, eyes glancing down-left toward togawa sakiko's chest"
             ), "position_reason": "right", "x": .62, "y": .56},
        ],
    }))

    assert plan["global_prompt"] == "2girls, indoors, close-up, eye-level"
    assert plan["dropped_global_character_tags"] == [
        "looking at viewer", "eye_contact", "looking down", "sideways glance",
    ]
    assert "looking at viewer" in plan["characters"][0]["prompt"]
    assert "eyes glancing down-left" in plan["characters"][1]["prompt"]


def test_nai_all_character_instance_conditions_are_removed_from_global_prompt():
    plan = parse_nai_character_plan(json.dumps({
        "composition_analysis": {
            "viewpoint": "wide shot", "layout": "group", "relations": "classroom",
        },
        "global_prompt": (
            "2girls, girl, blue_hair, large_breasts, white_shirt, school_uniform, "
            "smiling, holding_guitar, sitting_on_table, hug, bed, classroom, "
            "cinematic lighting, wide shot"
        ),
        "characters": [
            {
                "name": "character a",
                "prompt": (
                    "girl, blue_hair, large_breasts, white_shirt, school_uniform, "
                    "smiling, holding_guitar, sitting_on_table, hug, bed"
                ),
                "position_reason": "sitting on the left desk", "x": .3, "y": .55,
            },
            {
                "name": "character b",
                "prompt": "girl, pink_hair, school_uniform, standing",
                "position_reason": "standing at right", "x": .7, "y": .5,
            },
        ],
    }))

    assert plan["global_prompt"] == (
        "2girls, bed, classroom, cinematic lighting, wide shot"
    )
    assert plan["dropped_global_character_tags"] == [
        "girl", "blue_hair", "large_breasts", "white_shirt", "school_uniform",
        "smiling", "holding_guitar", "sitting_on_table", "hug",
    ]


def test_nai_position_prompt_requires_per_character_gaze_assignment():
    canvas = {
        "width": 1216, "height": 832, "aspect_ratio": "19:13",
        "orientation": "landscape", "character_limit": 22,
    }
    instruction = build_nai_character_plan_prompt(
        "两人的脸朝向viewer，但爱音看向祥子的胸口", "2girls, looking at viewer", canvas
    )

    assert "Put individual conditions only in its prompt" in instruction
    assert "every applicable prompt" in instruction
    assert "gaze toward the viewer" not in instruction
    assert "2girls, looking at viewer" in instruction
    assert "Keep face direction separate from eye gaze" in instruction
    assert "held/used props" in instruction and "local environment contact" in instruction
    assert "A shared bed can stay global" in instruction
    assert "lying on that bed belongs to its occupant" in instruction
    assert "panel-specific camera/crop belongs to its instance" in instruction



def test_nai_mutual_action_tags_and_legacy_plan_without_tags():
    base = {
        "composition_analysis": {
            "viewpoint": "front", "layout": "pair", "relations": "mutual embrace",
        },
        "global_prompt": "2girls, indoors",
        "characters": [
            {"name": "A", "prompt": "girl", "interaction_tags": ["mutual#hug"],
             "position_reason": "left", "x": .3, "y": .5},
            {"name": "B", "prompt": "girl", "interaction_tags": ["mutual#hug"],
             "position_reason": "right", "x": .7, "y": .5},
        ],
    }
    mutual = parse_nai_character_plan(json.dumps(base))
    assert all("mutual#hug" in item["prompt"] for item in mutual["characters"])
    for item in base["characters"]:
        del item["interaction_tags"]
    legacy = parse_nai_character_plan(json.dumps(base))
    assert all(item["interaction_tags"] == [] for item in legacy["characters"])


def test_nai_directional_action_accepts_one_multiword_danbooru_tag():
    plan = parse_nai_character_plan(json.dumps({
        "composition_analysis": {"viewpoint": "front", "layout": "pair", "relations": "pointing"},
        "global_prompt": "2girls",
        "characters": [
            {"name": "A", "prompt": "girl", "interaction_tags": ["source#pointing at another"],
             "position_reason": "left", "x": .3, "y": .5},
            {"name": "B", "prompt": "girl", "interaction_tags": ["target#pointing"],
             "position_reason": "right", "x": .7, "y": .5},
        ],
    }))
    assert "source#pointing at another" in plan["characters"][0]["prompt"]
    assert "target#pointing" in plan["characters"][1]["prompt"]


@pytest.mark.parametrize("invalid", ["source#hug, target#hug", "source#hug#kiss", "source#", "source#亲吻", "hug"])
def test_nai_drops_invalid_directional_action_tag_without_losing_layout(invalid):
    plan = parse_nai_character_plan(json.dumps({
        "composition_analysis": {"viewpoint": "front", "layout": "pair", "relations": "hug"},
        "global_prompt": "2girls",
        "characters": [
            {"name": "A", "prompt": "girl, smiling", "interaction_tags": [invalid],
             "position_reason": "left", "x": .3, "y": .5},
            {"name": "B", "prompt": "girl", "interaction_tags": [],
             "position_reason": "right", "x": .7, "y": .5},
        ],
    }))
    assert plan["characters"][0]["prompt"] == "A, girl, smiling"
    assert plan["characters"][0]["x"] == .3
    assert plan["dropped_interaction_tags"] == [invalid]


def test_nai_ordinary_four_panel_request_omits_directional_tags():
    request = (
        "4格漫画：第1格爱音掀开暖帘，祥子跟在身后；第2格爱音回头微笑；"
        "第3格祥子点头；第4格两人走向座位。"
    )
    canvas = {"width": 832, "height": 1216, "aspect_ratio": "13:19", "orientation": "portrait"}
    instruction = build_nai_character_plan_prompt(request, "2girls", canvas)
    assert '"interaction_tags":[]' in instruction
    assert "Leave interaction_tags empty" in instruction
    plan = parse_nai_character_plan(json.dumps({
        "composition_analysis": {"viewpoint": "front", "layout": "four panels", "relations": "walking"},
        "global_prompt": "2girls, restaurant",
        "characters": [{"name": "A", "prompt": "A, lifting curtain, source#lifting curtain",
                        "interaction_tags": [],
                        "position_reason": "upper panel", "x": .5, "y": .2}],
    }))
    assert plan["characters"][0]["prompt"] == "A, lifting curtain"
    assert plan["characters"][0]["interaction_tags"] == []
    assert plan["dropped_interaction_tags"] == ["source#lifting curtain"]


def test_nai_unlisted_interaction_is_not_blocked_by_a_trigger_word_list():
    request = "A踢了B，B摔倒在地"
    canvas = {"width": 832, "height": 1216, "aspect_ratio": "13:19", "orientation": "portrait"}
    instruction = build_nai_character_plan_prompt(request, "2girls", canvas)
    assert "actions beyond common examples" in instruction
    assert request in instruction
    plan = parse_nai_character_plan(json.dumps({
        "composition_analysis": {"viewpoint": "front", "layout": "pair", "relations": "A kicks B"},
        "global_prompt": "2girls, street",
        "characters": [
            {"name": "A", "prompt": "girl, kicking B", "interaction_tags": ["source#kicking"],
             "position_reason": "left", "x": .3, "y": .5},
            {"name": "B", "prompt": "girl, falling", "interaction_tags": ["target#kicking"],
             "position_reason": "right", "x": .7, "y": .5},
        ],
    }))
    assert plan["characters"][0]["prompt"].endswith("source#kicking")
    assert plan["characters"][1]["prompt"].endswith("target#kicking")


def test_nai_r_requires_nai_workflow():
    with pytest.raises(SystemExit, match="nai_character_mode_requires_nai_workflow"):
        workflow({}, "1girl", "blurry", 1024, 1536, 20, 5, 42,
                 nai_characters=[{"name": "girl", "prompt": "girl", "x": .5, "y": .5}])


def test_nai_r_supports_22_instances_and_rejects_23():
    characters = [
        {"name": f"girl {i}", "prompt": f"girl {i}, standing", "position_reason": "panel", "x": .5, "y": .5}
        for i in range(1, 23)
    ]
    plan = {"composition_analysis": {"viewpoint": "front", "layout": "grid", "relations": "separate"},
            "global_prompt": "22girls, grid", "characters": characters}
    parsed = parse_nai_character_plan(json.dumps(plan))
    graph = custom_t2i_workflow(
        {"custom_workflow_path": "nai_api.json"}, parsed["global_prompt"],
        "blurry", 1024, 1536, 28, 6, 123, nai_characters=parsed["characters"],
    )
    selector = graph[graph["1"]["inputs"]["characterPrompts"][0]]["inputs"]
    assert selector["character22_enable"] is True
    assert "girl 22" in graph[selector["character22"][0]]["inputs"]["comfyui_prompt"]
    plan["characters"].append(characters[0])
    with pytest.raises(ValueError, match="nai_character_plan_character_count"):
        parse_nai_character_plan(json.dumps(plan))


def test_nai_r_uses_v4_model_limit_even_with_explicit_canvas(tmp_path):
    graph = nai_graph()
    graph["1"]["inputs"]["model"] = "NAI Diffusion V4.5 Full"
    path = tmp_path / "nai-v4.json"
    path.write_text(json.dumps(graph), encoding="utf-8")
    canvas = resolve_nai_canvas(
        {"custom_workflow_enabled": True, "custom_workflow_path": str(path)},
        (832, 1216), explicit_size=True,
    )
    assert canvas["character_limit"] == 6
    assert "at most 6 instances" in build_nai_character_plan_prompt("six girls", "6girls", canvas)
    characters = [{"name": "girl", "prompt": "girl", "x": .5, "y": .5} for _ in range(7)]
    with pytest.raises(SystemExit, match="nai_character_plan_character_count"):
        custom_t2i_workflow(
            {"custom_workflow_path": str(path)}, "7girls", "blurry",
            832, 1216, 28, 6, 123, nai_characters=characters,
        )


@pytest.mark.parametrize("chair,floor", [
    ((.27, .29), (.76, .75)), ((.74, .25), (.23, .77)),
    ((.3427, .5173), (.3491, .5279)), ((.5, .5), (.5, .5)),
    ((0., 0.), (1., 1.)),
])
def test_nai_complex_side_view_preserves_model_layout(chair, floor):
    source = "丰川祥子坐在椅子上，千早爱音跪坐在丰川祥子面前的地上。侧视图。"
    canvas = resolve_nai_canvas(
        {"custom_workflow_enabled": True, "custom_workflow_path": "nai_api.json"},
        (1216, 832),
        plugin_root=PLUGIN_DIR,
    )
    instruction = build_nai_character_plan_prompt(source, "2girls, side view", canvas)
    assert "Interpret camera, relative height/depth" in instruction
    assert "horizontal and vertical separation" in instruction
    assert "do not use a fixed coordinate template" in instruction
    assert "Use spatial reasoning" in instruction
    assert "ALL requested actions" in instruction
    assert "simultaneously feasible" in instruction
    assert "a participant's multiple actions together" in instruction
    assert "do not snap to tenths or a grid" in instruction
    assert "panel coordinates still refer to the whole canvas" in instruction
    assert "do not pull unrelated people together" in instruction
    assert "Leave interaction_tags empty" in instruction
    directed = build_nai_character_plan_prompt("A 拥抱 B", "2girls", canvas)
    assert "source#tag" in directed and "target#tag" in directed
    assert "mutual#tag" in directed
    assert "Actual canvas: 1024x1536 pixels, aspect ratio 2:3 (portrait)" in instruction
    plan = parse_nai_character_plan(json.dumps({
        "composition_analysis": {
            "viewpoint": "side view",
            "layout": "chair sitter higher, floor kneeler lower, staggered diagonally",
            "relations": "the kneeler faces the sitter at close range",
        },
        "global_prompt": "2girls, side view, chair, indoors",
        "characters": [
            {"name": "togawa sakiko", "prompt": "sitting on a chair, looking at anon", "position_reason": "upper seat level", "x": chair[0], "y": chair[1]},
            {"name": "chihaya anon", "prompt": "kneeling on the floor, facing sakiko", "position_reason": "lower foreground", "x": floor[0], "y": floor[1]},
        ],
    }))
    assert (plan["characters"][0]["x"], plan["characters"][0]["y"]) == chair
    assert (plan["characters"][1]["x"], plan["characters"][1]["y"]) == floor
    assert plan["composition_analysis"]["viewpoint"] == "side view"
    graph = custom_t2i_workflow(
        {"custom_workflow_path": "nai_api.json"}, plan["global_prompt"], "blurry",
        1024, 1536, 28, 6, 123, nai_characters=plan["characters"],
    )
    selector = graph[graph["1"]["inputs"]["characterPrompts"][0]]["inputs"]
    assert (selector["character1_x"], selector["character1_y"]) == chair
    assert (selector["character2_x"], selector["character2_y"]) == floor


def test_nai_canvas_uses_explicit_size_or_override():
    config = {
        "custom_workflow_enabled": True,
        "custom_workflow_path": "nai_api.json",
    }
    explicit = resolve_nai_canvas(
        config, (1536, 1024), explicit_size=True, plugin_root=PLUGIN_DIR
    )
    assert explicit == {
        "width": 1536, "height": 1024, "aspect_ratio": "3:2",
        "orientation": "landscape", "source": "request",
        "model": "NAI Diffusion V5 Full", "character_limit": 22,
    }
    config["custom_workflow_override_parameters"] = True
    overridden = resolve_nai_canvas(config, (832, 1216), plugin_root=PLUGIN_DIR)
    assert (overridden["width"], overridden["height"]) == (832, 1216)
    assert overridden["source"] == "configuration"


def test_nai_r_passes_this_request_size_to_planner():
    class Recorder:
        def build_generation_start(self, **_kwargs):
            return {"task_id": "nai-size"}

        def mark_prompt_built(self, *_args):
            pass

        def mark_completed(self, *_args, **_kwargs):
            pass

        def write(self, *_args):
            pass

    observed = []
    summary = {
        "nai_r_mode": True,
        "nai_characters": [{"name": "girl", "prompt": "girl", "x": .5, "y": .5}],
        "nai_missing_noun_tags": ["heart"],
    }

    async def build_prompt(_event, _prompt, **kwargs):
        observed.append(kwargs)
        return "1girl, side view"

    submitted = []

    async def run_tool(args):
        submitted.append(args)
        return {"ok": True, "outputs": []}

    runner = GenerationTaskRunner(
        task_recorder=Recorder(),
        image_inputs=SimpleNamespace(last_summary={}),
        reference_context=SimpleNamespace(last_summary={}),
        is_allowed=lambda _event: True,
        ensure_ready=lambda _event: asyncio.sleep(0, result={"ok": True}),
        wants_reference_image=lambda _prompt: False,
        augment_reference_image=lambda _event, prompt: asyncio.sleep(0, result=prompt),
        augment_quoted_spell=lambda _event, prompt: prompt,
        build_prompt=build_prompt,
        prompt_summary=lambda: summary,
        run_tool=run_tool,
        get_bool=lambda _key, default: default,
        get_int=lambda _key, default: default,
        get_float=lambda _key, default: default,
        get_str=lambda _key, default: default,
        shorten=lambda value, limit: value[:limit],
    )
    result = asyncio.run(runner.generate_payload(
        object(), "-r 丰川祥子坐在椅子上，千早爱音跪在面前", width=1536, height=1024
    ))
    assert result["ok"] is True
    assert observed[0]["canvas_size"] == (1536, 1024)
    assert observed[0]["canvas_size_explicit"] is True
    assert len(submitted) == 1
    assert submitted[0][submitted[0].index("--prompt") + 1] == "1girl, side view"
    assert json.loads(submitted[0][submitted[0].index("--nai-characters") + 1]) == summary["nai_characters"]


@pytest.mark.parametrize("override,size", [(False, True), (True, False)])
def test_nai_size_and_parameter_override_preserves_nai_sampler(
    tmp_path, override, size
):
    result = build(tmp_path, nai_graph(), override=override, size=size)
    inputs = result["1"]["inputs"]
    assert (inputs["width"], inputs["height"]) == (1216, 832)
    assert inputs["steps"] == (20 if override else 28)
    assert inputs["cfg_scale"] == (4.5 if override else 6)
    assert inputs["sampler"] == "k_euler"
    assert inputs["scheduler"] == "karras"


def test_nai_direct_strings_and_inline_converter(tmp_path):
    graph = nai_graph()
    graph["1"]["inputs"]["prompt"] = "old positive"
    graph["12"]["inputs"]["comfyui_prompt"] = "old negative"
    result = build(tmp_path, graph)
    assert result["1"]["inputs"]["prompt"] == "(watercolor:1.2), 1girl"
    assert result["12"]["inputs"]["comfyui_prompt"] == "blurry"
    assert result["3"] == graph["3"]


def test_nai_follows_textbox_passthrough(tmp_path):
    graph = nai_graph()
    graph["3"]["inputs"]["passthrough"] = ["20", 0]
    graph["20"] = {"class_type": "Textbox", "inputs": {"text": "upstream"}}
    result = build(tmp_path, graph)
    assert result["20"]["inputs"]["text"] == "(watercolor:1.2), 1girl"
    assert result["3"] == graph["3"]


@pytest.mark.parametrize("invalid", ["shared", "cycle", "unknown", "missing"])
def test_nai_rejects_ambiguous_or_unsupported_links(tmp_path, invalid):
    graph = nai_graph()
    if invalid == "shared":
        graph["1"]["inputs"]["negative_prompt"] = ["9", 0]
    elif invalid == "cycle":
        graph["9"]["inputs"]["comfyui_prompt"] = ["9", 0]
    elif invalid == "unknown":
        graph["3"]["class_type"] = "UnrecognizedStringJoin"
    else:
        del graph["3"]
    error = (
        "custom_workflow_prompt_nodes_ambiguous"
        if invalid == "shared"
        else "nai_prompt_link_not_supported"
    )
    with pytest.raises(SystemExit, match=error):
        build(tmp_path, graph)


def test_nai_rejects_invalid_explicit_size_before_submission(tmp_path):
    with pytest.raises(SystemExit, match="nai_size_requires_multiple_of_64"):
        build(tmp_path, nai_graph(), size=True, width=836)


def test_ui_workflow_has_actionable_error(tmp_path):
    with pytest.raises(SystemExit, match="custom_workflow_requires_api_export"):
        build(tmp_path, {"nodes": [], "links": []})


def test_nai_api_can_be_selected_by_plugin_relative_path():
    result = custom_t2i_workflow(
        {"custom_workflow_path": "nai_api.json"},
        "1girl",
        "blurry",
        832,
        1216,
        20,
        5,
        123,
    )
    assert result["1"]["inputs"]["seed"] == 123
    assert result["3"]["inputs"]["text"] == "1girl"


def test_nai_validation_error_reaches_cli_payload(tmp_path):
    result = run_cli_action(lambda: build(tmp_path, nai_graph(), size=True, width=836))
    assert result == {"ok": False, "error": "nai_size_requires_multiple_of_64"}


def test_cli_does_not_swallow_numeric_exit():
    def stop():
        raise SystemExit(2)

    with pytest.raises(SystemExit) as exc:
        run_cli_action(stop)
    assert exc.value.code == 2


def test_nai_generation_payload_submits_adapted_graph_and_reports_actual_settings(
    tmp_path,
    monkeypatch,
):
    monkeypatch.syspath_prepend(str(PLUGIN_DIR / "agent_tools"))
    import comfyui_operations

    submitted = []

    def run_prompt(config, outputs, graph):
        submitted.append(graph)
        return "test-prompt-id", {"status": {"status_str": "success"}}

    monkeypatch.setattr(comfyui_operations, "_run_prompt", run_prompt)
    monkeypatch.setattr(
        comfyui_operations,
        "_save_history_images",
        lambda *args: ([tmp_path / "test.png"], 1),
    )
    result = comfyui_operations.generate_payload(
        {"custom_workflow_enabled": True, "custom_workflow_path": "nai_api.json"},
        {
            "width": 832,
            "height": 1216,
            "steps": 20,
            "cfg": 5,
            "allowed_sizes": ["832x1216"],
            "negative_prompt": "blurry",
        },
        tmp_path,
        SimpleNamespace(
            width=None,
            height=None,
            steps=None,
            cfg=None,
            seed=123,
            negative_prompt=None,
            override_size=False,
        ),
        "1girl, white dress",
    )
    assert result["ok"] is True
    assert (result["width"], result["height"], result["steps"], result["cfg"]) == (
        1024,
        1536,
        28,
        6,
    )
    assert result["seed"] == 123
    assert submitted[0]["3"]["inputs"]["text"] == "1girl, white dress"
    assert submitted[0]["11"]["inputs"]["text"] == "blurry"
    assert result["outputs"] == [str(tmp_path / "test.png")]
