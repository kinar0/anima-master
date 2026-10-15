"""Request-local NAI character prompt planning and validation."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

try:
    from .tag_cleaner import split_tags, display_tag_text
    from .prompt_visibility import APPEARANCE_VISIBILITY_POLICY
except ImportError:  # pragma: no cover - direct script-style imports.
    from tag_cleaner import split_tags, display_tag_text
    from prompt_visibility import APPEARANCE_VISIBILITY_POLICY


_R_SWITCH = re.compile(r"(?<!\S)-r(?=$|[\s,，;；])", re.I)
MAX_NAI_CHARACTERS = 22

NAI_WARDROBE_POLICY = """

Wardrobe authority: explicit user edits take priority. A selected named outfit or
explicit profile defines the base; keep recognizable unmodified parts and apply
all requested changes. When a selected named outfit still visibly applies, include
its confirmed canonical outfit tag in that wearer's actual prompt; its Chinese
name alone is not a tag anchor. Add verified visible components when known, without
guessing unknown ones. Saved defaults are soft references
only when clothing is unstated. Express visible parts, not a full component checklist.
Never restore removed/replaced clothes, assign another wearer's clothes, or copy a
donor's appearance. Unresolved explicit clothing does not permit default fallback.
"""


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
    user_prompt: str, final_prompt: str, canvas: dict[str, Any],
    *, noun_mappings: tuple[tuple[str, str], ...] = (),
) -> str:
    """Describe the NAI instance contract once, followed by request evidence."""
    width, height = int(canvas["width"]), int(canvas["height"])
    noun_rule = ""
    if noun_mappings:
        noun_rule = (
            "\nVocabulary references (source phrase => canonical tag):\n"
            + "\n".join(f"- {alias} => {tag}" for alias, tag in noun_mappings)
            + "\nUse only mappings that fit the request and the correct owner; natural "
            "language or a directional action tag may express the meaning instead. "
            "This is not a required tag checklist.\n"
        )
    return (
        "Design a complete, coherent NAI image from the request and local evidence. "
        "Preserve explicit identities, count, clothing, actions, expressions, props "
        "and instance differences. Express abstract themes through relevant visible "
        "anchors. Use concise Danbooru tags and coherent natural language; preserve "
        "supported weights without padding a fixed tag count or repeating synonyms.\n"
        "Output JSON only, using this schema:\n"
        '{"composition_analysis":{"viewpoint":"...","layout":"...",'
        '"relations":"..."},"global_prompt":"shared tags, scene-level prose in the user language","characters":['
        '{"name":"canonical character tag or short identity",'
        '"anchor_id":"matching local target anchor, or empty",'
        '"source_text":"exact request clause or clause array, or empty",'
        '"prompt":"girl / boy / other, identity, instance sentence in the user language, matching visible-detail tags",'
        '"interaction_tags":[],"position_reason":"...","x":0.5,"y":0.5}]}\n'
        "Instances and evidence:\n"
        "Create one entry per visible instance, including anonymous people and "
        "each appearance of a repeated identity in different panels. Do not add "
        "visible people from viewer, quoted names or off-screen clothing sources. "
        "Use the same confirmed canonical spelling in name and prompt; anonymous "
        "people use a short identity, not an invented canonical tag. "
        "Different local wardrobe plans for one identity require the matching "
        "anchor_id; never choose by array order. For repeated identities with "
        "locally recorded clothing edits or appearance changes, source_text must "
        "quote the exact applicable request clause, or an array of exact clauses "
        "for shared and local requirements. Otherwise leave it empty. Quote in "
        "the original language, without translation, paraphrase or joining "
        "non-contiguous text. User-specified instance differences override "
        "identity-level baselines.\n"
        "Scope and population:\n"
        "global_prompt contains total population, confirmed copyright, shared "
        "scene/environment objects, whole-image camera/framing, requested lighting, "
        "colour and layout. Include scene-level prose stating who is where and "
        "who acts on whom, using names or positions without individual appearance "
        "or clothing. For a Chinese request this prose is Chinese. Analysis and "
        "position reasons are metadata, not substitutes for this generated text. "
        "Count every visible instance once: 1girl/1boy for one known "
        "gender, Ngirls/Nboys for one gender, Ngirls, Mboys for a mixed group, "
        "or Npeople when gender is unknown. Each instance uses the unnumbered "
        "gender tag girl, boy, or other, never a numeric population tag. Follow "
        "explicit user gender, then reliable identity facts; do not infer a "
        "gender change from titles, clothing, poses or chest size. "
        "Put individual conditions only in its prompt: identity, visible appearance, "
        "clothing/accessories, expression, pose, held/used props, body/head "
        "orientation, gaze and local environment contact. Its own action and receiving state "
        "belong there too; global prose connects those actions into one scene. Repeat a shared instance "
        "condition in every applicable prompt, never globally, even when everyone "
        "wears the same named outfit. A shared bed can stay global; "
        "lying on that bed belongs to its occupant. A panel-specific camera/crop "
        "belongs to its instance; only whole-page framing belongs globally. "
        "Keep face direction separate from eye gaze. "
        "Reassign misplaced global instance conditions without losing "
        "their supported meaning or copying all evidence into every box.\n"
        "Composition:\n"
        "Interpret camera, relative height/depth, facing and contact together "
        "before placing instances. Use the actual canvas aspect ratio and "
        "respect explicit left/right and panel boundaries. Set x/y to the centre "
        "of each visible region, in [0,1], x left-to-right and y top-to-bottom; "
        "do not use a fixed coordinate template for pose words. Distinguish "
        "horizontal and vertical separation when the view requires it. For contact, "
        "check that projected bodies and limbs can meet at this framing and scale; "
        "do not place them as disconnected figures. Coordinates guide composition, "
        "not exact anatomy; preserve intentional cropping. Keep "
        "composition_analysis and position_reason brief.\n"
        "Interactions:\n"
        "Identify explicit actions between visible people from meaning, including "
        "actions beyond common examples. Allowed prefixes are exactly source#, "
        "target#, mutual#. For each directed action, use source#tag for the "
        "initiator and target#tag for the recipient with the same action tag on "
        "different instances. Use mutual#tag on every participant only for a "
        "genuinely reciprocal action; do not mix directed and mutual roles for "
        "the same action on one instance. Each interaction_tags item is one "
        "English Danbooru action tag after # (spaces within a tag are allowed), "
        "never multiple tags or a sentence. Put these only in interaction_tags. "
        "Evaluate each ordinary action tag as a standalone assertion about this "
        "instance. A source prefix grants agency, not ownership of the action's "
        "body location. Bare body-location/state tags belong only to the owner of "
        "the affected body part, never automatically to both participants or to "
        "the owner of a contacting limb. Remove an unsafe bare tag while retaining "
        "the act in named prose and paired direction tags. Both may keep an ordinary "
        "action only when its standalone meaning truly applies to both. "
        "Leave interaction_tags empty for individual actions, proximity, one-sided "
        "gaze or off-screen people. Write who acts on whom, direction and contact "
        "in the actual instance prompts too, naming whose limb contacts whose "
        "body part. Review every requested action, not only the dominant one. "
        "Prefer explicit prose over an ambiguous or invented tag. "
        "Distinguish recipients in separate groups, and never infer action direction "
        "from position or array order.\n"
        "Writing and review:\n"
        "Use the user's language for descriptive prose. For a Chinese request, "
        "retain the user's original Chinese details in concise but complete Chinese "
        "prose inside each actual instance prompt, covering its visible appearance, "
        "clothing, expression, pose, actions and relations without a one-sentence "
        "limit. Do not translate all prose into English or output tags alone. "
        "Refer to other participants by name or position, never by describing "
        "their appearance, body traits or clothing inside this instance's prompt. "
        "Write that prose before choosing its detail tags, then derive tags "
        "from the stated owners and states. Do not summarize a whole interaction "
        "as a body-state tag on the initiator. If the full request still leaves "
        "an owner ambiguous, preserve the original clause without inventing an "
        "ownership-specific tag to resolve it. "
        "Chinese only in source_text, analysis or position_reason does not satisfy "
        "this requirement. Keep canonical tags and identities accurately spelled "
        "in English alongside the prose. Use references only for requested aspects; "
        "when borrowing clothing, pose or style alone, preserve the target identity. "
        "Default to simple background and white background only when no environment "
        "is needed. Supply a setting required by the request or action without "
        "conflicting white-background tags; explicit simple/white background wins. "
        "Do not invent unrelated locations or weather. Add lighting descriptions "
        "only when requested; do not add soft/cinematic lighting by default or "
        "add a light source to explicit darkness. Respect explicit absence, plain "
        "background, no makeup and neutral expression without embellishment. "
        "Do not invent sex characteristics without positive user evidence. "
        "Quality and artist tags are supplied by the host; do not output them. "
        "Before output, review scene completeness, tag ownership, positions, "
        "visibility and action pairs against the actual prompt strings, not just "
        "the analysis. Compare tags with prose for each expressed appearance, "
        "clothing state, expression, gaze, pose/contact, camera, background and "
        "lighting dimension. Keep the grounded specific state; remove contradictory "
        "generic claims and redundant synonyms. Correct omissions, inverted roles "
        "and missing counterparts within local authority. Record reviewed relations "
        "in composition_analysis.relations "
        "as brief actor/action/recipient/body-owner statements, covering every "
        "requested interaction and marking any unresolved owner.\n"
        + APPEARANCE_VISIBILITY_POLICY
        + f"\nActual canvas: {width}x{height} pixels, aspect ratio "
        f"{canvas['aspect_ratio']} ({canvas['orientation']}); "
        f"width/height={width / height:.4f}. "
        f"Use at most {canvas.get('character_limit', MAX_NAI_CHARACTERS)} instances for this model.\n"
        + noun_rule
        + f"\nUser request:\n{user_prompt}\n"
        f"Local character evidence:\n{final_prompt}"
    )


def nai_interaction_validation_issues(plan: dict[str, Any]) -> tuple[str, ...]:
    """Check direction-pair structure without guessing actions from request words."""
    actions: dict[str, dict[str, set[int]]] = {}
    for index, character in enumerate(plan["characters"]):
        for tag in character.get("interaction_tags", ()):
            direction, _, action = tag.casefold().partition("#")
            roles = actions.setdefault(_normalized_nai_tag(action), {})
            roles.setdefault(direction, set()).add(index)
    issues = []
    for action, roles in actions.items():
        sources, targets = roles.get("source", set()), roles.get("target", set())
        mutual = roles.get("mutual", set())
        if any(not (targets - {index}) for index in sources):
            issues.append(f"source#{action} has no target#{action} on another visible instance")
        if any(not (sources - {index}) for index in targets):
            issues.append(f"target#{action} has no source#{action} on another visible instance")
        if mutual and len(mutual) < 2:
            issues.append(f"mutual#{action} needs at least two visible participants")
        if mutual & (sources | targets):
            issues.append(f"{action} assigns mutual and directed roles to the same instance")
    return tuple(issues)


def missing_nai_noun_tags(
    plan: dict[str, Any], tags: tuple[str, ...],
    *, character_scoped_tags: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Report exact mapped-tag coverage for diagnostics, never plan validity."""
    def key(value: str) -> str:
        return re.sub(r"\s+", "_", str(value or "").replace("\\", "").strip().casefold())

    global_keys = {key(part) for part in split_tags(plan["global_prompt"])}
    character_keys = {
        key(part)
        for character in plan["characters"]
        for part in split_tags(character["prompt"])
    }
    # parse_nai_character_plan emits only validated interaction_tags into the
    # character prompt. Their action is already conditioned with its direction;
    # an ordinary duplicate is unnecessary for exact coverage diagnostics.
    character_keys.update(
        key(part.partition("#")[2])
        for character in plan["characters"]
        for part in split_tags(character["prompt"])
        if _DIRECTIONAL_ACTION_TAG_RE.fullmatch(part)
    )
    scoped_keys = {key(tag) for tag in character_scoped_tags}
    return tuple(
        tag for tag in tags
        if key(tag) not in (
            character_keys
            if key(tag) in scoped_keys or _is_character_scoped_tag(tag)
            else character_keys | global_keys
        )
    )


def parse_nai_character_plan(
    raw: str, *, character_limit: int = MAX_NAI_CHARACTERS,
    character_scoped_tags: tuple[str, ...] = (),
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
            if len(interaction_tags) >= 8:
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
        # Only the explicit array can supply direction tags. Strip copies from
        # the prose field so an empty array really means no interaction tag.
        prompt_parts = []
        for part in split_tags(character_prompt):
            if re.match(r"^(?:source|target|mutual)#", part, re.I):
                if part not in interaction_tags:
                    dropped_interaction_tags.append(part[:80])
            else:
                count = re.fullmatch(r"\d+\s*(girls?|boys?|others?|people|persons?)", part.strip(), re.I)
                if count:
                    gender = count.group(1).casefold()
                    part = "girl" if gender.startswith("girl") else "boy" if gender.startswith("boy") else "other"
                prompt_parts.append(part)
        character_prompt = ", ".join(prompt_parts)
        if name.casefold() not in character_prompt.casefold():
            character_prompt = f"{name.strip()}, {character_prompt}"
        if interaction_tags:
            character_prompt += ", " + ", ".join(interaction_tags)
        position_reason = item.get("position_reason")
        if not isinstance(position_reason, str) or not position_reason.strip():
            raise ValueError("nai_character_plan_missing_position_reason")
        evidence = {}
        for field in ("anchor_id", "source_text"):
            value = item.get(field, "")
            if field == "source_text" and isinstance(value, list):
                if not 1 <= len(value) <= 12 or any(not isinstance(clause, str) or not clause.strip() for clause in value):
                    raise ValueError("nai_character_plan_invalid_instance_evidence")
                evidence[field] = [clause.strip() for clause in value]
            elif not isinstance(value, str):
                raise ValueError("nai_character_plan_invalid_instance_evidence")
            elif value.strip():
                evidence[field] = value.strip()
        cleaned.append({
            **evidence, "name": name.strip(), "prompt": character_prompt,
            "interaction_tags": interaction_tags,
            "position_reason": position_reason.strip(), "x": x, "y": y,
        })
    # Exact character identity tags belong to the instance boxes even if the
    # writer copied them from local evidence.
    global_parts = split_tags(global_prompt)
    names = {re.sub(r"[\s_]+", "", item["name"].casefold()) for item in cleaned}
    dropped_global_character_tags: list[str] = []
    retained_global_parts: list[str] = []
    scoped_keys = {_normalized_nai_tag(tag) for tag in character_scoped_tags}
    for part in global_parts:
        if _is_character_scoped_tag(part) or _normalized_nai_tag(part) in scoped_keys:
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
