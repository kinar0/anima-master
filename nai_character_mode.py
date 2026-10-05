"""Request-local NAI character prompt planning and validation."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

try:
    from .tag_cleaner import split_tags, display_tag_text
except ImportError:  # pragma: no cover - direct script-style imports.
    from tag_cleaner import split_tags, display_tag_text


_R_SWITCH = re.compile(r"(?<!\S)-r(?=$|[\s,，;；])", re.I)
MAX_NAI_CHARACTERS = 22
_EXPLICIT_INTERACTION_RE = re.compile(
    r"拥抱|抱住|搂住|接吻|亲吻|牵.{0,4}手|握手|击掌|对视|推.{0,4}(?:他|她|对方)|"
    r"(?:hug|kiss|hold hands|shake hands|high.five|look at each other|pushes? (?:him|her|them))",
    re.I,
)


def has_explicit_nai_interaction(user_prompt: str) -> bool:
    """Conservatively opt into optional person-to-person direction tags."""
    return bool(_EXPLICIT_INTERACTION_RE.search(str(user_prompt or "")))


def nai_character_limit(model: str) -> int:
    """Match the selected generator model's character-caption limit."""
    name = str(model or "").casefold()
    if "v5" in name or "diffusion-5" in name:
        return MAX_NAI_CHARACTERS
    if "v4" in name or "diffusion-4" in name:
        return 6
    return 0
_DIRECTIONAL_ACTION_TAG_RE = re.compile(
    r"(?:source|target|mutual)#[a-z0-9][a-z0-9_]*(?:[ _-][a-z0-9_]+)*",
    re.I,
)
_CHARACTER_SCOPED_GAZE_RE = re.compile(
    r"^(?:"
    r"looking(?:\s|$)|glancing(?:\s|$)|gazing(?:\s|$)|staring(?:\s|$)|"
    r"eyes?\s+(?:looking|glancing|gazing|staring|directed|turned|averted|downcast)(?:\s|$)|"
    r"gaze(?:\s|$)|eye contact$|direct gaze$|averted gaze$|"
    r"sideways glance$"
    r")",
    re.I,
)
_CHARACTER_SCOPED_TAG_RE = re.compile(
    r"^(?:"
    # Per-instance sex/gender. Count tags such as 2girls do not match because
    # they start with a digit and remain valid global population anchors.
    r"(?:girl|boy|woman|man|female|male|futanari|androgynous)(?:\s|$)|"
    # Common appearance and anatomy forms, including Danbooru underscore tags
    # after normalization by _normalized_nai_tag().
    r"(?:long|short|medium|very long|blue|pink|red|green|yellow|black|white|"
    r"brown|blonde|grey|gray|purple|orange|aqua|silver|multicolored)\s+"
    r"(?:hair|eyes?|skin)(?:\s|$)|"
    r"(?:breasts?|chest|body|face|mouth|lips?|ears?|tail|horns?)\s+|"
    r"(?:twintails?|two side up|ponytail|braid|drill hair|sidelocks|ahoge|"
    r"heterochromia|freckles|fangs?|animal ears?|cat ears?|dog ears?|fox ears?|"
    r"large breasts?|small breasts?|flat chest|mature|child|teenager)(?:\s|$)|"
    # Clothing and worn accessories.
    r"(?:wearing|dressed in|undressed|nude|topless|bottomless)(?:\s|$)|"
    r"(?:school|military|maid|sailor|business)\s+uniform(?:\s|$)|"
    r"(?:shirt|blouse|jacket|coat|dress|skirt|pants|trousers|shorts|uniform|"
    r"swimsuit|bikini|underwear|bra|panties|stockings|socks|shoes|boots|gloves|"
    r"hat|cap|ribbon|necktie|tie|scarf|mask|veil|armor|apron|hoodie|sweater|"
    r"cardigan|kimono|robe)(?:\s|$)|"
    r"(?:white|black|blue|pink|red|green|yellow|brown|grey|gray|purple)\s+"
    r"(?:shirt|blouse|jacket|coat|dress|skirt|pants|trousers|shorts|uniform|"
    r"swimsuit|bikini|stockings|socks|shoes|boots|gloves|ribbon|necktie|tie)(?:\s|$)|"
    # Expression, pose, action, prop use and local character/environment contact.
    r"(?:smile|smiling|grin|grinning|blush|blushing|crying|tears|angry|"
    r"sad|surprised|embarrassed|expressionless|open mouth|closed mouth)(?:\s|$)|"
    r"(?:sit|sitting|stand|standing|kneel|kneeling|crouch|crouching|squat|"
    r"squatting|lie|lying|walk|walking|run|running|jump|jumping|lean|leaning|"
    r"bending|reaching|holding|carrying|touching|grabbing|"
    r"hug|hugging|kiss|kissing|point|pointing|wave|waving|arms?|hands?|legs?|"
    r"head|face)\b|"
    r"(?:on|against|under|inside|beside|behind|in front of)\s+"
    r"(?:a\s+|the\s+)?[a-z0-9]"
    r")",
    re.I,
)


def _validated_directional_action_tag(value: Any) -> str:
    tag = str(value or "").strip()
    if len(tag) > 80 or not _DIRECTIONAL_ACTION_TAG_RE.fullmatch(tag):
        raise ValueError("nai_character_plan_invalid_interaction_tag")
    return tag


def _normalized_nai_tag(value: Any) -> str:
    normalized = re.sub(r"[_\s-]+", " ", str(value or "").strip().casefold())
    return normalized.strip(" ()")


def _is_character_scoped_tag(value: Any) -> bool:
    """Return whether a tag belongs to one NAI character instance."""
    normalized = _normalized_nai_tag(value)
    return bool(
        _CHARACTER_SCOPED_GAZE_RE.match(normalized)
        or _CHARACTER_SCOPED_TAG_RE.match(normalized)
    )


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
    model = ""
    if bool(config.get("custom_workflow_enabled", False)):
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
                    model = str(generators[0].get("model") or "")
                    graph_width = int(generators[0]["width"])
                    graph_height = int(generators[0]["height"])
                    if (not explicit_size
                        and not bool(config.get("custom_workflow_override_parameters", False))
                        and graph_width > 0 and graph_height > 0):
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
        "model": model,
        "character_limit": nai_character_limit(model) if model else MAX_NAI_CHARACTERS,
    }


def build_nai_character_plan_prompt(
    user_prompt: str, final_prompt: str, canvas: dict[str, Any]
) -> str:
    width, height = int(canvas["width"]), int(canvas["height"])
    interaction_requested = has_explicit_nai_interaction(user_prompt)
    interaction_example = (
        '"interaction_tags":["source#hug"],'
        if interaction_requested else ""
    )
    interaction_rule = (
        "For an explicitly directed interaction, add interaction_tags to the "
        "participating character entries: the actor gets source#tag and the "
        "recipient gets target#tag. For a genuinely mutual action, give each "
        "participant mutual#tag. Each string must contain exactly one "
        "Danbooru action tag after # (spaces within one tag are fine), never "
        "a comma-separated phrase or a whole sentence. Keep these tags out "
        "of global_prompt and the ordinary prompt field. "
        if interaction_requested else
        "The user did not request a person-to-person directed action. Omit "
        "interaction_tags entirely; keep individual actions in each prompt. "
    )
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
        '"prompt":"gender, identity, appearance, clothing, expression, pose, '
        'action, local environment relation",'
        f'{interaction_example}'
        '"position_reason":"...",'
        '"x":0.5,"y":0.5}]}\n'
        "Create one characters entry per visible INSTANCE, not per unique name. "
        "If the same character appears in two panels/grid cells, make two entries "
        "with separate actions, clothing and coordinates. Preserve the user's "
        "explicit differences between instances. The user request is authoritative "
        "for those differences even if the validated full prompt merged them. "
        f"Use at most {canvas.get('character_limit', MAX_NAI_CHARACTERS)} instances for this model. "
        "Coordinates range from 0 to 1, with x increasing left to right and y "
        "increasing top to bottom. Respect explicit left/right or panel placement; "
        "when direction is unstated, choose it to suit the whole composition. "
        "Use the actual canvas aspect ratio when balancing spacing, framing, "
        "overlap and negative space; normalized coordinates refer to this canvas. "
        "For example, someone seated on furniture and someone on the floor may "
        "need both horizontal and vertical separation in a side view; decide "
        "their actual positions from the request and your composition analysis. "
        "Treat every condition that applies to a character instance as character "
        "prompt content. Put that character's sex/gender, identity tag, stable "
        "appearance, clothing and worn accessories, expression, pose, action, "
        "held/touched/occupied props, body orientation, head/face orientation, "
        "gaze, and local relationship to the environment only in that instance's "
        "prompt. Local environment relationships include sitting on a table, "
        "leaning against a wall, standing in water, lying on a bed, being under "
        "an umbrella, or occupying a chair. A face can point toward the "
        "viewer while its eyes look elsewhere; preserve those as separate facts. "
        "Put only whole-image conditions in global_prompt: artist tags, quality, "
        "copyright, total person-count, the shared background/scene itself, "
        "camera/framing, lighting, color grading and layout. A scene may be global "
        "while a character's contact with it is local: classroom belongs globally, "
        "but sitting on classroom desk belongs to that character. Do not put sex/"
        "gender, character identity, appearance, wardrobe, expression, pose, worn "
        "or handled objects, body/head orientation, gaze, individual actions, or "
        "local environment relationships in global_prompt. The validated "
        "full prompt may contain unscoped character tags; redistribute them "
        "instead of copying them globally. Gaze tags such as looking at viewer, "
        "looking at another, looking down, looking away, eye contact, sideways "
        "glance, glancing, gazing or staring must always be written into the "
        "applicable character prompt, never global_prompt. Even when every "
        "character shares one gaze or expression, repeat it in every applicable "
        "character prompt. Apply the same rule to all other instance conditions; "
        "even a condition shared by all characters must be repeated in each "
        "applicable character prompt instead of placed globally. Example: if both "
        "characters wear school uniforms in a classroom, keep classroom global "
        "and repeat school uniform in both character prompts. If one sits on a "
        "desk, put sitting on desk only in that character prompt. If both faces "
        "point toward the viewer but "
        "only Anon's eyes glance down-left toward Sakiko's chest, put face toward "
        "viewer in both character prompts, looking at viewer only in Sakiko's "
        "prompt, and the down-left chest gaze only in Anon's prompt; put none of "
        "those gaze facts in global_prompt. "
        "Write all prompt strings in English. Keep established Danbooru tags "
        "and escaped weight syntax when useful. "
        f"{interaction_rule}"
        "Do not invent an interaction or assign its direction from character "
        "order or position alone. "
        "Do not invent another visible person from viewer or an off-screen source.\n"
        f"Actual canvas: {width}x{height} pixels, aspect ratio "
        f"{canvas['aspect_ratio']} ({canvas['orientation']}); "
        f"width/height={width / height:.4f}.\n"
        f"User request:\n{user_prompt}\n"
        f"Validated full prompt and character evidence:\n{final_prompt}"
    )


def parse_nai_character_plan(
    raw: str, *, character_limit: int = MAX_NAI_CHARACTERS,
    allow_interaction_tags: bool = True,
) -> dict[str, Any]:
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
    if not isinstance(characters, list) or not 1 <= len(characters) <= character_limit:
        raise ValueError("nai_character_plan_character_count")
    cleaned: list[dict[str, Any]] = []
    dropped_interaction_tags: list[str] = []
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
        raw_interaction_tags = item.get("interaction_tags", [])
        interaction_tags: list[str] = []

        def collect_interaction_tag(tag: Any) -> None:
            if not allow_interaction_tags or len(interaction_tags) >= 8:
                dropped_interaction_tags.append(str(tag)[:80])
                return
            try:
                validated = _validated_directional_action_tag(tag)
            except ValueError:
                dropped_interaction_tags.append(str(tag)[:80])
                return
            if validated not in interaction_tags:
                interaction_tags.append(validated)

        if isinstance(raw_interaction_tags, list):
            for tag in raw_interaction_tags:
                collect_interaction_tag(tag)
        elif raw_interaction_tags:
            dropped_interaction_tags.append(str(raw_interaction_tags)[:80])
        character_prompt = prompt.strip()
        # The list is authoritative; a copied directional tag in the prose
        # field should not duplicate the same tag in the submitted character box.
        prompt_parts = []
        for part in split_tags(character_prompt):
            if re.match(r"^(?:source|target|mutual)#", part, re.I):
                collect_interaction_tag(part)
            else:
                prompt_parts.append(part)
        character_prompt = ", ".join(prompt_parts)
        if name.casefold() not in character_prompt.casefold():
            character_prompt = f"{name.strip()}, {character_prompt}"
        if interaction_tags:
            character_prompt += ", " + ", ".join(interaction_tags)
        position_reason = item.get("position_reason")
        if not isinstance(position_reason, str) or not position_reason.strip():
            raise ValueError("nai_character_plan_missing_position_reason")
        cleaned.append({
            "name": name.strip(), "prompt": character_prompt,
            "interaction_tags": interaction_tags,
            "position_reason": position_reason.strip(), "x": x, "y": y,
        })
    # Exact character identity tags belong to the instance boxes even if the
    # planner copied them from the validated seven-field prompt.
    global_parts = split_tags(global_prompt)
    names = {re.sub(r"[\s_]+", "", item["name"].casefold()) for item in cleaned}
    dropped_global_character_tags: list[str] = []
    retained_global_parts: list[str] = []
    for part in global_parts:
        if _is_character_scoped_tag(part):
            dropped_global_character_tags.append(part)
            continue
        retained_global_parts.append(part)
    global_parts = retained_global_parts
    global_parts = [
        part for part in global_parts
        if re.sub(r"[\s_]+", "", part.casefold()) not in names
        and not re.match(r"^(?:source|target|mutual)#", part, re.I)
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
        "dropped_interaction_tags": dropped_interaction_tags,
        "dropped_global_character_tags": dropped_global_character_tags,
    }


def preserve_nai_global_artist_tags(global_prompt: str, artist_tags: str) -> str:
    """Restore configured artist anchors after the NAI layout LLM rewrites globals."""
    configured = split_tags(artist_tags)
    if not configured:
        return global_prompt

    def artist_key(value: str) -> str:
        # Ignore punctuation, escaping, separators and a trailing weight so a
        # model's malformed copy of the same artist can be replaced.
        value = value.strip().replace("\\", "")
        value = re.sub(r":\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)\)?$", "", value)
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    configured_keys = {artist_key(tag) for tag in configured}
    retained = [
        part for part in split_tags(global_prompt)
        if artist_key(part) not in configured_keys
    ]
    return ", ".join([*(display_tag_text(tag) for tag in configured), *retained])
