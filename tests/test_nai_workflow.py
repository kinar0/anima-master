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
    has_explicit_nai_interaction,
    parse_nai_character_plan,
    preserve_nai_global_artist_tags,
    resolve_nai_canvas,
    strip_nai_character_switch,
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

    assert "local environment relationships in global_prompt" in instruction
    assert "repeat it in every applicable character prompt" in instruction
    assert "only Anon's eyes glance down-left toward Sakiko's chest" in instruction
    assert "looking at viewer only in Sakiko's prompt" in instruction
    assert "sex/gender, identity tag" in instruction
    assert "sitting on a table" in instruction
    assert "classroom belongs globally" in instruction
    assert "sitting on classroom desk belongs to that character" in instruction


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
    assert not has_explicit_nai_interaction(request)
    canvas = {"width": 832, "height": 1216, "aspect_ratio": "13:19", "orientation": "portrait"}
    instruction = build_nai_character_plan_prompt(request, "2girls", canvas)
    assert '"interaction_tags"' not in instruction
    assert "Omit interaction_tags entirely" in instruction
    plan = parse_nai_character_plan(json.dumps({
        "composition_analysis": {"viewpoint": "front", "layout": "four panels", "relations": "walking"},
        "global_prompt": "2girls, restaurant",
        "characters": [{"name": "A", "prompt": "A, lifting curtain, source#lift(curtain)",
                        "interaction_tags": ["target#walk"],
                        "position_reason": "upper panel", "x": .5, "y": .2}],
    }), allow_interaction_tags=False)
    assert plan["characters"][0]["prompt"] == "A, lifting curtain"
    assert len(plan["dropped_interaction_tags"]) == 2


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


@pytest.mark.parametrize("chair,floor", [((.27, .29), (.76, .75)), ((.74, .25), (.23, .77))])
def test_nai_complex_side_view_preserves_model_layout(chair, floor):
    source = "丰川祥子坐在椅子上，千早爱音跪坐在丰川祥子面前的地上。侧视图。"
    canvas = resolve_nai_canvas(
        {"custom_workflow_enabled": True, "custom_workflow_path": "nai_api.json"},
        (1216, 832),
        plugin_root=PLUGIN_DIR,
    )
    instruction = build_nai_character_plan_prompt(source, "2girls, side view", canvas)
    assert "camera angle" in instruction
    assert "relative positions" in instruction
    assert "Do not use a fixed coordinate template" in instruction
    assert "Omit interaction_tags entirely" in instruction
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
    }

    async def build_prompt(_event, _prompt, **kwargs):
        observed.append(kwargs)
        return "1girl, side view"

    async def run_tool(_args):
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
