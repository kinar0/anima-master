"""Request-local NAI character prompt planning and validation."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


_R_SWITCH = re.compile(r"(?<!\S)-r(?=$|[\s,，;；])", re.I)


def strip_nai_character_switch(prompt: str) -> tuple[bool, str]:
    """Recognize a standalone -r in the optimizable part of a request."""
    text = str(prompt or "")
    enabled = bool(_R_SWITCH.search(text))
    if enabled:
        text = _R_SWITCH.sub(" ", text)
    return enabled, re.sub(r"\s+", " ", text).strip(" ,，;；")


def resolve_nai_canvas(
    config: dict[str, Any],
    requested_size: tuple[int, int],
    *,
    explicit_size: bool = False,
    plugin_root: Path | None = None,
) -> dict[str, Any]:
    """Use the NAI workflow canvas unless this request overrides its size."""
    width, height = (int(requested_size[0]), int(requested_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("nai_canvas_invalid_size")
    source = "request" if explicit_size else "configuration"
    if (
        not explicit_size
        and not bool(config.get("custom_workflow_override_parameters", False))
        and bool(config.get("custom_workflow_enabled", False))
    ):
        path_text = str(config.get("custom_workflow_path") or "").strip()
        if path_text:
            path = Path(path_text).expanduser()
            if not path.is_absolute():
                path = (plugin_root or Path(__file__).resolve().parent) / path
            try:
                raw = json.loads(path.read_text(encoding="utf-8-sig"))
                graph = raw.get("prompt", raw) if isinstance(raw, dict) else {}
                if not isinstance(graph, dict):
                    graph = {}
                generators = [
                    node.get("inputs", {})
                    for node in graph.values()
                    if isinstance(node, dict)
                    and node.get("class_type") == "NovelAIGenerator"
                ]
                if len(generators) == 1:
                    graph_width = int(generators[0]["width"])
                    graph_height = int(generators[0]["height"])
                    if graph_width > 0 and graph_height > 0:
                        width, height = graph_width, graph_height
                        source = "nai_workflow"
            except (OSError, TypeError, ValueError, KeyError):
                # Workflow validation still reports the real error at submission.
                pass
    divisor = math.gcd(width, height) or 1
    return {
        "width": width,
        "height": height,
        "aspect_ratio": f"{width // divisor}:{height // divisor}",
        "orientation": "landscape" if width > height else "portrait" if width < height else "square",
        "source": source,
    }


def build_nai_character_plan_prompt(
    user_prompt: str, final_prompt: str, canvas: dict[str, Any]
) -> str:
    width, height = int(canvas["width"]), int(canvas["height"])
    return (
        "Plan the composition for this NAI image request before assigning any "
        "character coordinates. First identify every visible character instance, "
        "including repeated appearances of the same identity. Then interpret "
        "the user's camera angle, poses, relative positions, height, depth, "
        "facing directions and interactions as one coherent scene. Choose a "
        "composition that shows those relations clearly. Finally assign each "
        "instance the center of its visible region as normalized x/y coordinates. "
        "Do not use a fixed coordinate template for particular words or poses. "
        "Briefly state your layout decision and why each position fits. "
        "Return JSON only:\n"
        '{"composition_analysis":{"viewpoint":"...","layout":"...",'
        '"relations":"..."},"global_prompt":"...","characters":['
        '{"name":"canonical character tag or short identity",'
        '"prompt":"identity, appearance, clothing, pose, action",'
        '"position_reason":"...",'
        '"x":0.5,"y":0.5}]}\n'
        "Create one characters entry per visible INSTANCE, not per unique name. "
        "If the same character appears in two panels/grid cells, make two entries "
        "with separate actions, clothing and coordinates. Preserve the user's "
        "explicit differences between instances. The user request is authoritative "
        "for those differences even if the validated full prompt merged them. "
        "Use up to five instances. "
        "Coordinates range from 0 to 1, with x increasing left to right and y "
        "increasing top to bottom. Respect explicit left/right or panel placement; "
        "when direction is unstated, choose it to suit the whole composition. "
        "Use the actual canvas aspect ratio when balancing spacing, framing, "
        "overlap and negative space; normalized coordinates refer to this canvas. "
        "For example, someone seated on furniture and someone on the floor may "
        "need both horizontal and vertical separation in a side view; decide "
        "their actual positions from the request and your composition analysis. "
        "Put each character's identity tag, stable appearance, clothing, pose "
        "and action only in that instance's prompt. Put artist tags, quality, "
        "background, lighting, layout, interaction, shared scene and the "
        "correct total person-count tag only in global_prompt. Do not put "
        "character identity, wardrobe or individual actions in global_prompt. "
        "Write all prompt strings in English. Keep established Danbooru tags "
        "and escaped weight syntax when useful. "
        "Do not invent another visible person from viewer or an off-screen source.\n"
        f"Actual canvas: {width}x{height} pixels, aspect ratio "
        f"{canvas['aspect_ratio']} ({canvas['orientation']}); "
        f"width/height={width / height:.4f}.\n"
        f"User request:\n{user_prompt}\n"
        f"Validated full prompt and character evidence:\n{final_prompt}"
    )


def parse_nai_character_plan(raw: str) -> dict[str, Any]:
    """Validate model output before any NAI workflow is submitted."""
    value = str(raw or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I).strip()
    try:
        data = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("nai_character_plan_invalid_json") from exc
    if not isinstance(data, dict):
        raise ValueError("nai_character_plan_invalid_structure")
    global_prompt = data.get("global_prompt")
    characters = data.get("characters")
    analysis = data.get("composition_analysis")
    if not isinstance(analysis, dict) or any(
        not isinstance(analysis.get(key), str) or not analysis[key].strip()
        for key in ("viewpoint", "layout", "relations")
    ):
        raise ValueError("nai_character_plan_missing_composition_analysis")
    if not isinstance(global_prompt, str) or not global_prompt.strip():
        raise ValueError("nai_character_plan_missing_global_prompt")
    if not isinstance(characters, list) or not 1 <= len(characters) <= 5:
        raise ValueError("nai_character_plan_character_count")
    cleaned: list[dict[str, Any]] = []
    for item in characters:
        if not isinstance(item, dict):
            raise ValueError("nai_character_plan_invalid_character")
        name, prompt = item.get("name"), item.get("prompt")
        if not isinstance(name, str) or not name.strip() or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("nai_character_plan_invalid_character")
        try:
            x, y = float(item["x"]), float(item["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("nai_character_plan_invalid_position") from exc
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ValueError("nai_character_plan_invalid_position")
        character_prompt = prompt.strip()
        if name.casefold() not in character_prompt.casefold():
            character_prompt = f"{name.strip()}, {character_prompt}"
        position_reason = item.get("position_reason")
        if not isinstance(position_reason, str) or not position_reason.strip():
            raise ValueError("nai_character_plan_missing_position_reason")
        cleaned.append({
            "name": name.strip(), "prompt": character_prompt,
            "position_reason": position_reason.strip(), "x": x, "y": y,
        })
    # Exact character identity tags belong to the instance boxes even if the
    # planner copied them from the validated seven-field prompt.
    global_parts = [part.strip() for part in global_prompt.split(",")]
    names = {re.sub(r"[\s_]+", "", item["name"].casefold()) for item in cleaned}
    global_parts = [
        part for part in global_parts
        if re.sub(r"[\s_]+", "", part.casefold()) not in names
    ]
    global_clean = ", ".join(part for part in global_parts if part)
    if not global_clean:
        raise ValueError("nai_character_plan_missing_global_prompt")
    return {
        "composition_analysis": {
            key: analysis[key].strip() for key in ("viewpoint", "layout", "relations")
        },
        "global_prompt": global_clean,
        "characters": cleaned,
    }
