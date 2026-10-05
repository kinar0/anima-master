"""Semantic planner boundary and tag-trust vocabulary.

The planner receives the user's image request as plain text and returns compact
per-character intent JSON. The host creates internal anchor IDs, attempts local
character/outfit matching, and supplies verified evidence to the prompt writer.
The former candidates/lookups and anchors/character_plans schemas remain accepted
compatibility inputs. Planner output is never tag evidence.

Terminology used by the prompt pipeline:

* A *soft tag* is an LLM/search candidate or descriptive hint. It may guide prose
  or a lookup, but it must not establish character identity or wardrobe facts.
* A *validated tag* has supporting local-index, configured, or persisted-profile
  evidence. Validation describes provenance, not whether the tag will be emitted.
* A *hard tag* is deterministically injected into the final prompt. This describes
  placement, not provenance: identity and named-outfit hard tags require validated
  or user-configured evidence, while generic garment fallbacks may be synthesized
  from bounded user operations or filtered reference/search evidence. Conversely,
  a validated tag may be withheld when it belongs to an outfit donor, a different
  character, or a garment explicitly removed by the user.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


DEFAULT_SEMANTIC_PLAN_SYSTEM_PROMPT = (
    "Extract only what the user said about each visible character. Copy phrases "
    "exactly, preserve the user's complete visible-character roster without "
    "merging or omitting anyone, use null for unknown clothing, never guess tags "
    "or lore, and return JSON only."
)

MAX_SEMANTIC_CHARACTERS = 22
MAX_SEMANTIC_ANCHORS = MAX_SEMANTIC_CHARACTERS * 2

_COUNT_WORD_VALUES = {
    **{str(value): value for value in range(1, MAX_SEMANTIC_CHARACTERS + 1)},
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6,
    "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12,
    "十三": 13, "十四": 14, "十五": 15, "十六": 16, "十七": 17,
    "十八": 18, "十九": 19, "二十": 20, "二十一": 21, "二十二": 22,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "twenty-one": 21, "twenty two": 22, "twenty-two": 22,
}
_COUNT_WORD_PATTERN = "|".join(
    re.escape(value)
    for value in sorted(_COUNT_WORD_VALUES, key=len, reverse=True)
)


def _explicit_visible_character_count(user_prompt: str) -> int | None:
    """Read an explicit visible-person total without trying to identify names."""
    text = str(user_prompt or "")
    total_patterns = (
        rf"(?<![第零一二两三四五六七八九十\d])"
        rf"(?P<count>{_COUNT_WORD_PATTERN})\s*(?:个|名|位)?\s*"
        rf"(?:可见的?)?(?:角色|人物|人)(?!称)",
        rf"\b(?P<count>{_COUNT_WORD_PATTERN})\s+(?:visible\s+)?"
        rf"(?:characters?|people|persons?)\b",
    )
    for pattern in total_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return _COUNT_WORD_VALUES.get(match.group("count").casefold())

    gender_pattern = (
        rf"(?<![第零一二两三四五六七八九十\d])"
        rf"(?P<count>{_COUNT_WORD_PATTERN})\s*(?:个|名|位)?\s*"
        rf"(?:女孩|女生|女性|少女|男孩|男生|男性|少年)"
        rf"|\b(?P<english>{_COUNT_WORD_PATTERN})\s*"
        rf"(?:girls?|boys?|women|men)\b"
    )
    values: list[int] = []
    for match in re.finditer(gender_pattern, text, re.I):
        value = match.group("count") or match.group("english")
        parsed = _COUNT_WORD_VALUES.get(str(value).casefold())
        if parsed:
            values.append(parsed)
    total = sum(values)
    return total if 1 <= total <= MAX_SEMANTIC_CHARACTERS else None

LEGACY_SEMANTIC_PLAN_SYSTEM_PROMPTS = frozenset({
    "You extract semantic lookup anchors for a local Danbooru index. "
    "Return valid JSON only. Never claim that a candidate is verified.",
    "You build a character-to-wardrobe relation graph and bounded lookup anchors "
    "for a local Danbooru index. Relationship binding is the primary task: emit "
    "exactly one character_plans item for every visible target_character, preserve "
    "the wearer and its wardrobe/outfit_source even when the request also changes "
    "hair, color, pose, expression, or scene, and keep separate clauses scoped to "
    "their own wearers. Never merge overlapping source names or request-wide "
    "wardrobes. Return valid JSON only. Candidates are unverified lookup hints.",
    "You identify visible characters, what each one wears, and a few Danbooru "
    "lookup hints. Return the compact characters/lookups JSON requested by the "
    "user prompt. Copy every name and evidence phrase exactly from the request. "
    "Do not invent IDs, groups, descriptions, or cross-reference keys. Return "
    "valid JSON only. Candidate tags are unverified lookup hints.",
    "You only extract per-character intent from the user's request. Report each "
    "visible character, stated clothing or unknown clothing, explicit clothing "
    "changes, and explicit appearance changes. Copy evidence exactly; never guess "
    "Danbooru tags, character identities, outfit contents, or missing clothes. "
    "Return only the JSON requested by the user prompt.",
})


_ALLOWED_GROUPS = {
    "character",
    "series",
    "appearance",
    "expression",
    "pose",
    "action",
    "clothing",
    "outfit",
    "accessory",
    "prop",
    "scene",
    "background",
    "lighting",
}
_ALLOWED_ROLES = {
    "target_character",
    "outfit_source",
    "copyright",
    "appearance",
    "expression",
    "pose",
    "action",
    "clothing",
    "outfit",
    "accessory",
    "prop",
    "scene",
    "lighting",
}


@dataclass(frozen=True)
class SemanticAnchor:
    """One source-grounded lookup question proposed by the semantic planner.

    ``source_text`` must occur in the request. ``role`` says how a confirmed result
    may be used, while ``group`` restricts the local index category. ``candidates``
    are soft spelling guesses only; they become hard-tag candidates only after the
    local lookup returns a matching canonical record.
    """

    anchor_id: str
    role: str
    group: str
    source_text: str
    description: str
    candidates: tuple[str, ...]
    profile_key: str = ""
    literal_name: str = ""
    source_category: int | None = None


@dataclass(frozen=True)
class SemanticOutfitDirective:
    """A bounded clothing operation proposed by the planning LLM.

    This is deliberately semantic rather than tag-level: the planner may say
    which *slot* to keep or remove, but local verified outfit data decides the
    actual Danbooru tags affected.
    """

    operation: str
    slots: tuple[str, ...]
    color: str = ""
    source_text: str = ""
    target_anchor_id: str = ""


@dataclass(frozen=True)
class SemanticWardrobe:
    """One target character's authoritative wardrobe selection."""

    kind: str
    anchor_id: str = ""


@dataclass(frozen=True)
class SemanticCharacterPlan:
    """A strictly reference-checked visual/outfit plan for one character.

    Outfit ownership is keyed by ``target_anchor_id`` rather than stored in a
    request-wide tag bag. Clothing is visually attached to a wearer, so flattening
    two profiles would let the final LLM swap uniforms or apply one character's
    removal/recolor directive to another.
    """

    target_anchor_id: str
    wardrobe: SemanticWardrobe
    directives: tuple[SemanticOutfitDirective, ...] = ()
    clothing_anchor_id: str = ""


@dataclass(frozen=True)
class SemanticAppearanceChange:
    """One LLM1-detected identity modification kept as advisory evidence only."""

    character_name: str
    source_text: str
    dimensions: tuple[str, ...] = ()
    operation: str = "unspecified"


@dataclass(frozen=True)
class SemanticLookupResult:
    """Locally validated semantic evidence for one image request.

    ``confirmed_tags`` and ``anchor_tags`` contain canonical lookup results.
    ``candidate_tags`` remain soft diagnostics and must never be injected merely
    because they were returned by fuzzy search. ``character_profiles`` and
    ``anchor_outfit_profiles`` retain anchor IDs so outfit/appearance evidence can
    be assigned to one wearer; the request-wide profile fields exist for legacy
    single-character results only.
    """

    confirmed_tags: tuple[str, ...] = ()
    outfit_source_tags: tuple[str, ...] = ()
    outfit_profile_tags: tuple[str, ...] = ()
    appearance_profile_tags: tuple[str, ...] = ()
    named_outfit_tags: tuple[str, ...] = ()
    source_outfit_profiles: tuple[
        tuple[str, str, tuple[str, ...], str], ...
    ] = ()
    character_appearance_profiles: tuple[
        tuple[tuple[str, ...], str, tuple[str, ...]], ...
    ] = ()
    anchor_tags: tuple[tuple[str, str], ...] = ()
    character_profiles: tuple[
        tuple[str, str, tuple[str, ...], tuple[str, ...]], ...
    ] = ()
    anchor_outfit_profiles: tuple[
        tuple[str, str, tuple[str, ...], str], ...
    ] = ()
    missing_descriptions: tuple[str, ...] = ()
    candidate_tags: tuple[str, ...] = ()
    anchors: tuple[SemanticAnchor, ...] = ()
    status: str = "not_available"

    def prompt_context(
        self,
        *,
        include_outfit_source_anchor: bool = True,
        effective_outfit_tags: tuple[str, ...] = (),
        removed_outfit_tags: tuple[str, ...] = (),
        suppress_profile_outfit_tags: bool = False,
    ) -> str:
        """Render verified evidence for the final prompt-writing LLM."""
        if self.status == "not_available" or (
            not self.anchors
            and not self.confirmed_tags
            and not self.outfit_source_tags
            and not self.outfit_profile_tags
            and not self.appearance_profile_tags
            and not self.named_outfit_tags
            and not self.source_outfit_profiles
            and not self.character_appearance_profiles
            and not effective_outfit_tags
        ):
            return ""
        lines = [
            "Local Danbooru validation (authoritative for hard tags):",
            "Use confirmed tags exactly. Do not transliterate missing concepts into invented tags.",
        ]
        confirmed = tuple(
            tag
            for tag in self.confirmed_tags
            if include_outfit_source_anchor or tag not in self.outfit_source_tags
        )
        if confirmed:
            lines.append("confirmed hard tags: " + ", ".join(confirmed))
        anchor_tag_map = dict(self.anchor_tags)
        confirmed_characters = tuple(
            (anchor.source_text, anchor_tag_map.get(anchor.anchor_id, ""))
            for anchor in self.anchors
            if anchor.role == "target_character"
            and anchor_tag_map.get(anchor.anchor_id, "")
        )
        if confirmed_characters:
            lines.append("confirmed visible character roster (authoritative):")
            lines.extend(
                f"- {source}: {tag}" for source, tag in confirmed_characters
            )
            lines.append(
                "Characters must use these exact canonical tags. Do not rename, "
                "translate, re-scope, or independently guess them."
            )
        if self.outfit_source_tags:
            if include_outfit_source_anchor:
                lines.append(
                    "costume/cosplay source anchors (not extra visible people): "
                    + ", ".join(self.outfit_source_tags)
                )
                if not self.outfit_profile_tags:
                    lines.append(
                        "The source anchor must still appear in Tags. It is sufficient "
                        "on its own: do not invent or infer clothing attributes when no "
                        "post-derived outfit profile is available."
                    )
            else:
                lines.append(
                    "costume source identity (context only; do not emit as a hard tag): "
                    + ", ".join(self.outfit_source_tags)
                )
        visible_outfit_tags = (
            effective_outfit_tags
            if suppress_profile_outfit_tags
            else effective_outfit_tags or self.outfit_profile_tags
        )
        if visible_outfit_tags:
            lines.append(
                "authoritative visible outfit tags for this request: "
                + ", ".join(visible_outfit_tags)
            )
            lines.append(
                "Do not add clothing outside this authoritative list. Explicit user "
                "changes have already been applied to it."
            )
        if self.appearance_profile_tags:
            lines.append(
                "authoritative recurring appearance tags for this character: "
                + ", ".join(self.appearance_profile_tags)
            )
        if self.character_appearance_profiles:
            lines.append("Character-scoped stable appearance profiles:")
            for aliases, source_tag, tags in self.character_appearance_profiles:
                label = aliases[0] if aliases else source_tag
                lines.append(f"- {label} / {source_tag}: " + ", ".join(tags))
        if self.source_outfit_profiles:
            lines.append("Character-scoped outfit-source profiles (never mix them):")
            for alias, source_tag, tags, qualifier in self.source_outfit_profiles:
                label = f"{alias} [{qualifier}]" if qualifier != "default" else alias
                lines.append(
                    f"- {label} / {source_tag}: " + ", ".join(tags)
                )
        if self.named_outfit_tags:
            lines.append(
                "user-requested named outfit-set hard tags: "
                + ", ".join(self.named_outfit_tags)
            )
            lines.append(
                "Keep each named outfit set on the character explicitly wearing it."
            )
            lines.append(
                "A named outfit-set tag is complete on its own. Do not infer or add "
                "component garments such as shirt, jacket, skirt, pleated_skirt, "
                "necktie, hosiery, or shoes unless the user explicitly modifies them "
                "or an authoritative visible outfit profile lists them."
            )
        if removed_outfit_tags:
            lines.append(
                "outfit tags explicitly replaced or removed by the user; never restore: "
                + ", ".join(removed_outfit_tags)
            )
        if self.missing_descriptions:
            lines.append(
                "unresolved concepts; express these only in Nltags: "
                + "; ".join(self.missing_descriptions)
            )
        return "\n".join(lines)

    def outfit_tags_for_source(self, source_text: str) -> tuple[str, ...]:
        """Return only the profile belonging to one requested costume source."""
        key = re.sub(r"\s+", " ", str(source_text or "").strip().lower())
        for alias, source_tag, tags, _qualifier in self.source_outfit_profiles:
            alias_key = re.sub(r"\s+", " ", alias.strip().lower())
            tag_key = source_tag.split("_(", 1)[0].replace("_", " ").lower()
            if key and (key in alias_key or alias_key in key or key == tag_key):
                return tags
        return ()


def build_semantic_plan_prompt(user_prompt: str) -> str:
    """Build the semantic planner's complete input/output protocol.

    Args:
        user_prompt: Original image request used both as planner input and as the
            source-of-truth text against which returned evidence is checked.

    Returns:
        A prompt requiring per-character intent JSON with no model-authored tags.
        The host derives IDs and performs local evidence lookup afterward.
    """
    expected_count = _explicit_visible_character_count(user_prompt)
    count_contract = (
        f"The request explicitly requires exactly {expected_count} visible "
        f"characters. The characters array MUST contain exactly {expected_count} "
        "items. Do not merge, summarize, sample, or omit characters. "
        if expected_count is not None else
        "Preserve the complete visible-character roster. Do not merge, summarize, "
        "sample, or omit characters. "
    )
    return (
        "List each visible character once. " + count_contract +
        "A character used only as a cosplay or "
        "clothing reference is not visible. Copy all text values exactly from the "
        "request; never translate, guess tags, or fill missing clothes.\n\n"
        "Return only:\n"
        '{"characters":[{"name":"exact visible entity mention","aliases":[],'
        '"clothing":null,'
        '"clothing_source":null,"clothing_changes":[],'
        '"appearance_changes":[]}]}\n\n'
        "name = only the exact entity mention for the visible character, never a "
        "whole descriptive clause. aliases = exact alternate names explicitly "
        "paired with that same entity in A (B), otherwise []. For example, "
        "复仇者（revenant） uses name=复仇者 and aliases=[revenant], while "
        "巨乳（huge breasts）的粉发美少女 uses name=粉发美少女 and aliases=[]; "
        "appearance, clothing, pose, action, and translation glosses are not "
        "character aliases.\n"
        "clothing = the exact stated garment/set phrase, or null when unstated.\n"
        "clothing_source = the exact character/persona whose outfit is copied, or "
        "null. For A cosplay C, A is visible and C is clothing_source.\n"
        "clothing_changes = explicit changes only. Every item MUST be an object "
        "with operation, slots, color, source_text; source_text must be exact. "
        "operation MUST be one of remove, damage, replace_color, recolor_all, add, "
        "keep_only. slots MUST use only upper_body.primary, lower_body.skirt, "
        "one_piece.dress, outerwear, headwear, face_accessory.mask, handwear, "
        "legwear, footwear, lower_body.all, misc. If an item does not clearly fit "
        "a named slot, ALWAYS use misc instead of inventing another slot. Use "
        "remove (never removed/removal) for an item the character no longer wears; "
        "use damage for torn/ripped clothing, not remove. Torn or damaged clothing "
        "is a change, not removal. Use [] only when there is no explicit clothing "
        "change.\n"
        "appearance_changes = explicit request phrases only. Prefer objects with "
        "dimension (eye_color, hair_color, hair_length, hair_style, skin_color, "
        "chest_size, animal_ears, tail, horns, age_presentation, or "
        "gender_presentation), operation (replace or additive), and exact "
        "source_text; do not output tags. operation=replace means the new value "
        "supersedes the old value; operation=additive means both remain visibly "
        "present. A trait immediately before a name belongs only to that "
        "person (扶她千早爱音 means only 千早爱音 is futanari); futanari is not "
        "chest_size. Shared "
        "words such as both/all/双方/两人/都 must be copied onto every affected "
        "character.\n\n"
        f"User request: {user_prompt}"
    )


def parse_semantic_plan(raw: str, user_prompt: str) -> tuple[SemanticAnchor, ...]:
    """Convert an untrusted planner response into source-grounded lookup anchors.

    ``raw`` may contain a fenced or prose-wrapped JSON object. The result contains
    at most ``MAX_SEMANTIC_ANCHORS`` anchors whose source text occurs in
    ``user_prompt`` and whose
    candidate spellings are safe to pass to the local Danbooru lookup CLI. An
    invalid response degrades to an empty tuple rather than aborting generation.
    """
    data = _semantic_json(raw, user_prompt)
    items = data.get("anchors") if isinstance(data, dict) else None
    intent_only = bool(data.get("_intent_only")) if isinstance(data, dict) else False
    if not isinstance(items, list):
        return ()
    anchors: list[SemanticAnchor] = []
    for index, item in enumerate(items[:MAX_SEMANTIC_ANCHORS]):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        group = str(item.get("group") or "").strip().lower()
        source_text = str(item.get("source_text") or "").strip()
        if role == "outfit" and source_text in {"校服", "制服"}:
            # Some planners preserve the outfit role but discard the proper-name
            # modifier. Recover the complete phrase after an explicit wear verb so
            # 月之森校服 cannot later be cached globally under the alias "校服".
            specialized = re.search(
                r"(?:穿着|身穿|身着|换上|换成)\s*[\"“']?"
                r"([^，。；;、\s\"”']{2,24}(?:校服|制服))",
                user_prompt,
                flags=re.I,
            )
            if specialized:
                source_text = specialized.group(1)
        description = re.sub(r"\s+", " ", str(item.get("description") or "").strip())
        if role not in _ALLOWED_ROLES or group not in _ALLOWED_GROUPS:
            continue
        if role in {"target_character", "outfit_source"} and group != "character":
            continue
        if role == "copyright" and group != "series":
            continue
        if not source_text or (
            source_text not in user_prompt
            and source_text.lower() not in user_prompt.lower()
        ):
            continue
        if not description or len(description) > 180:
            description = source_text
        candidates: list[str] = []
        raw_candidates = item.get("candidates")
        if not isinstance(raw_candidates, list):
            continue
        for value in raw_candidates[:3]:
            candidate = re.sub(r"\s+", "_", str(value or "").strip().lower())
            if not re.fullmatch(r"[a-z0-9_.'():!\-]{2,120}", candidate):
                continue
            if candidate not in candidates:
                candidates.append(candidate)
        if not candidates and not (
            intent_only
            and role in {
                "target_character", "outfit", "outfit_source", "clothing"
            }
        ):
            continue
        anchor_id = re.sub(
            r"[^a-z0-9_-]+", "_", str(item.get("id") or f"anchor_{index}").lower()
        ).strip("_") or f"anchor_{index}"
        anchors.append(
            SemanticAnchor(
                anchor_id=anchor_id,
                role=role,
                group=group,
                source_text=source_text,
                description=description,
                candidates=tuple(candidates),
            )
        )
    return tuple(anchors)


_OUTFIT_DIRECTIVE_OPERATIONS = {
    "remove", "damage", "replace_color", "recolor_all", "add", "keep_only"
}
_OUTFIT_DIRECTIVE_SLOTS = {
    "upper_body.primary",
    "lower_body.skirt",
    "one_piece.dress",
    "outerwear",
    "headwear",
    "face_accessory.mask",
    "handwear",
    "legwear",
    "footwear",
    "lower_body.all",
    "misc",
}
_OUTFIT_DIRECTIVE_COLORS = {
    "pink", "red", "white", "black", "blue", "green", "yellow", "purple",
    "grey", "brown", "gold", "silver", "orange", "beige", "navy",
}

# LLM1 output is untrusted, but harmless morphology and everyday garment names
# are not ambiguity. Normalize those values before validation so a grounded user
# instruction is not silently lost merely because the model wrote ``removed``
# instead of ``remove`` or ``boots`` instead of ``footwear``.
_OUTFIT_DIRECTIVE_OPERATION_ALIASES = {
    "remove": "remove", "removed": "remove", "removal": "remove",
    "delete": "remove", "deleted": "remove", "omit": "remove",
    "omitted": "remove", "take_off": "remove", "taken_off": "remove",
    "damage": "damage", "damaged": "damage", "torn": "damage",
    "ripped": "damage",
    "replace_color": "replace_color", "replace_colour": "replace_color",
    "recolor": "replace_color", "recolored": "replace_color",
    "recoloured": "replace_color", "change_color": "replace_color",
    "change_colour": "replace_color",
    "recolor_all": "recolor_all", "recolour_all": "recolor_all",
    "add": "add", "added": "add",
    "keep_only": "keep_only", "only_keep": "keep_only",
}
_OUTFIT_DIRECTIVE_SLOT_ALIASES = {
    "shirt": "upper_body.primary", "top": "upper_body.primary",
    "upper_garment": "upper_body.primary", "upper_body": "upper_body.primary",
    "skirt": "lower_body.skirt",
    "dress": "one_piece.dress", "one_piece": "one_piece.dress",
    "coat": "outerwear", "jacket": "outerwear", "cloak": "outerwear",
    "cape": "outerwear",
    "hat": "headwear", "headdress": "headwear",
    "mask": "face_accessory.mask", "face_mask": "face_accessory.mask",
    "glove": "handwear", "gloves": "handwear",
    "stocking": "legwear", "stockings": "legwear", "sock": "legwear",
    "socks": "legwear", "pantyhose": "legwear",
    "boot": "footwear", "boots": "footwear", "shoe": "footwear",
    "shoes": "footwear", "heels": "footwear",
    "lower_body": "lower_body.all", "lower_body_clothing": "lower_body.all",
    "accessory": "misc", "accessories": "misc", "ornament": "misc",
    "jewelry": "misc", "jewellery": "misc", "other": "misc",
    "unknown": "misc", "miscellaneous": "misc",
}

_WARDROBE_KINDS = {
    "default_profile", "casual_profile", "summer_profile", "winter_profile",
    "stage_profile", "named_outfit", "outfit_source", "creative_fallback",
    "none"
}


def _raw_semantic_json(raw: str) -> dict[str, Any]:
    """Extract one JSON object from a possibly fenced planner response.

    Only mappings are accepted because every semantic-plan consumer expects named
    fields. Malformed or differently shaped model output is represented by an
    empty mapping so callers can use their normal no-plan fallback.
    """
    text = re.sub(r"^```(?:json)?\s*", "", str(raw or "").strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


_APPEARANCE_CHANGE_DIMENSIONS = {
    "eye_color": "eye_color",
    "eye_traits": "eye_traits",
    "eyes": "eye_color",
    "heterochromia": "eye_color",
    "hair_color": "hair_color",
    "hair_length": "hair_length",
    "hair_style": "hair_style",
    "skin_color": "skin_color",
    "chest_size": "chest_size",
    "animal_ears": "animal_ears",
    "tail": "tail",
    "horns": "horns",
    "age_presentation": "age_presentation",
    "gender_presentation": "gender_presentation",
}


def _appearance_source_is_grounded(
    source_text: str, user_prompt: str, character_name: str
) -> bool:
    """Ground LLM1 evidence through generic normalized clause alignment.

    Exact normalized substrings remain the strongest evidence. Minor planner
    paraphrases are accepted only when one request clause has high sequence
    agreement with the complete source phrase. The character itself must already
    be grounded separately; this function never infers an appearance dimension.
    """

    def normalized(value: str) -> str:
        return re.sub(
            r"[^\w\u4e00-\u9fff]+", "", str(value or "").casefold()
        )

    prompt_key = normalized(user_prompt)
    source_key = normalized(source_text)
    if not source_key:
        return False
    # A bare sexual-trait phrase is easy for a planner to copy onto every
    # character because it occurs somewhere in the request.  Require the
    # phrase to be locally attached to this character instead.  This is only
    # an evidence check; it does not infer new appearance dimensions.
    if source_key in {"扶她", "futa", "futanari"}:
        name_key = normalized(character_name)
        if not name_key:
            return False
        return bool(
            re.search(re.escape(source_key) + re.escape(name_key), prompt_key)
            or re.search(re.escape(name_key) + r"(?:是|为|变成|成为)?" + re.escape(source_key), prompt_key)
        )
    if source_key in prompt_key:
        return True
    if not str(character_name or "").strip() or len(source_key) < 4:
        return False
    clauses = tuple(
        key
        for clause in re.split(r"[。！？!?;；\n]+", str(user_prompt or ""))
        if (key := normalized(clause))
    )
    return any(
        SequenceMatcher(None, source_key, clause).ratio() >= 0.72
        and SequenceMatcher(None, source_key, clause).find_longest_match().size
        / min(len(source_key), len(clause))
        >= 0.65
        for clause in clauses
    )


def parse_semantic_appearance_changes(
    raw: str, user_prompt: str
) -> tuple[SemanticAppearanceChange, ...]:
    """Read LLM1 identity-change hints without making them hard tag authority.

    The planner is allowed to understand wording that host regexes miss.  Its
    result is deliberately advisory: every name and source phrase must still be
    traceable to the user request, but an unknown dimension or malformed row is
    retained as an LLM2 reminder rather than becoming a validation failure.
    """
    data = _raw_semantic_json(raw)
    characters = data.get("characters") if isinstance(data, dict) else None
    if not isinstance(characters, list):
        return ()
    changes: list[SemanticAppearanceChange] = []
    for character in characters[:MAX_SEMANTIC_CHARACTERS]:
        if not isinstance(character, dict):
            continue
        name = str(character.get("name") or "").strip()
        if not name or (name not in user_prompt and name.lower() not in user_prompt.lower()):
            continue
        rows = character.get("appearance_changes", [])
        if not isinstance(rows, list):
            continue
        for row in rows[:8]:
            if isinstance(row, dict):
                source_text = str(
                    row.get("source_text") or row.get("text") or row.get("change") or ""
                ).strip()
                raw_dimensions = row.get("dimensions", row.get("dimension", ()))
                raw_operation = str(row.get("operation") or "").strip().lower()
            else:
                source_text = str(row or "").strip()
                raw_dimensions = ()
                raw_operation = ""
            if not _appearance_source_is_grounded(source_text, user_prompt, name):
                continue
            if isinstance(raw_dimensions, str):
                raw_dimensions = (raw_dimensions,)
            dimensions = tuple(
                dict.fromkeys(
                    _APPEARANCE_CHANGE_DIMENSIONS.get(
                        str(value or "").strip().lower(), ""
                    )
                    for value in raw_dimensions
                    if _APPEARANCE_CHANGE_DIMENSIONS.get(
                        str(value or "").strip().lower(), ""
                    )
                )
            ) if isinstance(raw_dimensions, (list, tuple)) else ()
            if re.sub(r"[^\w\u4e00-\u9fff]+", "", source_text.casefold()) in {
                "扶她", "futa", "futanari"
            }:
                # Futanari is a sex characteristic, not a breast-size edit.
                dimensions = tuple(d for d in dimensions if d != "chest_size")
            operation = {
                "replace": "replace",
                "replacement": "replace",
                "additive": "additive",
                "add": "additive",
                "augment": "additive",
                "coexist": "additive",
            }.get(raw_operation, "unspecified")
            change = SemanticAppearanceChange(name, source_text, dimensions, operation)
            if change not in changes:
                changes.append(change)
    return tuple(changes)


def _candidate_values(item: dict[str, Any]) -> list[Any]:
    """Return compact-schema candidates without treating strings as lists."""
    values = item.get("candidates", item.get("tags"))
    if isinstance(values, list):
        return values
    tag = item.get("tag")
    return [tag] if tag not in (None, "") else []


def _simple_semantic_to_legacy(
    data: dict[str, Any], user_prompt: str
) -> dict[str, Any]:
    """Normalize compact LLM1 JSON to the established internal graph schema.

    The current protocol contains only per-character intent. The host creates
    deterministic target/source anchors with empty candidate lists so later local
    profile matching—not LLM1—supplies tag evidence. The older compact
    characters/lookups form remains accepted for compatibility.
    """
    if "anchors" in data or "character_plans" in data:
        return data
    characters = data.get("characters")
    lookups = data.get("lookups")
    intent_only = "lookups" not in data
    if not isinstance(characters, list) or (
        not intent_only and not isinstance(lookups, list)
    ):
        return data
    if intent_only:
        lookups = []

    def normalized_source(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    def normalized_wardrobe(item: dict[str, Any]) -> dict[str, Any]:
        """Translate the readable intent schema into one internal wardrobe row."""
        legacy = item.get("wardrobe")
        if isinstance(legacy, dict):
            return legacy
        clothing = str(item.get("clothing") or "").strip()
        source = str(item.get("clothing_source") or "").strip()
        changes = item.get("clothing_changes", item.get("changes", []))
        if source:
            kind = "outfit_source"
            wardrobe_source = source
        elif re.search(r"(?:默认|原始|初始|default|original).{0,6}(?:衣|服|outfit)", clothing, re.I):
            kind = "default_profile"
            wardrobe_source = ""
        elif re.search(r"(?:常服|便服|日常服|casual)", clothing, re.I):
            kind = "casual_profile"
            wardrobe_source = ""
        elif re.search(r"(?:夏季服装|夏装|夏服|summer\s+(?:clothes?|clothing|outfit|wear|attire))", clothing, re.I):
            kind = "summer_profile"
            wardrobe_source = clothing
        elif re.search(r"(?:冬季服装|冬装|冬服|winter\s+(?:clothes?|clothing|outfit|wear|attire))", clothing, re.I):
            kind = "winter_profile"
            wardrobe_source = clothing
        elif re.search(r"(?:演出服|舞台服|舞台装|表演服|(?:stage|performance|concert)\s+(?:costume|outfit|wear|attire))", clothing, re.I):
            kind = "stage_profile"
            wardrobe_source = clothing
        elif clothing:
            kind = "creative_fallback"
            wardrobe_source = clothing
        elif isinstance(changes, list) and changes:
            kind = "default_profile"
            wardrobe_source = ""
        else:
            kind = "none"
            wardrobe_source = ""
        return {"kind": kind, "source": wardrobe_source, "changes": changes}

    wardrobe_role_demands: dict[str, set[str]] = {}
    for item in characters[:MAX_SEMANTIC_CHARACTERS]:
        if not isinstance(item, dict):
            continue
        wardrobe = normalized_wardrobe(item)
        kind = str(wardrobe.get("kind") or "none").strip().lower()
        if kind not in {"named_outfit", "outfit_source"}:
            continue
        source = normalized_source(
            wardrobe.get("source", wardrobe.get("source_text", ""))
        )
        if source:
            expected_role = "outfit" if kind == "named_outfit" else "outfit_source"
            wardrobe_role_demands.setdefault(source, set()).add(expected_role)

    wardrobe_lookup_counts: dict[str, int] = {}
    for item in lookups[:8]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        source = normalized_source(item.get("text", item.get("source_text", "")))
        if role in {"outfit", "outfit_source"} and source and _candidate_values(item):
            wardrobe_lookup_counts[source] = wardrobe_lookup_counts.get(source, 0) + 1

    anchors: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    lookup_ids: dict[tuple[str, str], list[str]] = {}
    role_groups = {
        "outfit_source": "character",
        "copyright": "series",
        "appearance": "appearance",
        "expression": "expression",
        "pose": "pose",
        "action": "action",
        "clothing": "clothing",
        "outfit": "outfit",
        "accessory": "accessory",
        "prop": "prop",
        "scene": "scene",
        "lighting": "lighting",
    }
    for index, item in enumerate(lookups[:8], start=1):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        source_text = str(
            item.get("text", item.get("source_text", "")) or ""
        ).strip()
        if role not in role_groups:
            continue
        source_key = normalized_source(source_text)
        demanded_roles = wardrobe_role_demands.get(source_key, set())
        if (
            role in {"outfit", "outfit_source"}
            and wardrobe_lookup_counts.get(source_key) == 1
            and len(demanded_roles) == 1
        ):
            role = next(iter(demanded_roles))
        anchor_id = f"lookup_{index}"
        anchors.append(
            {
                "id": anchor_id,
                "role": role,
                "group": role_groups[role],
                "source_text": source_text,
                "description": source_text,
                "candidates": _candidate_values(item),
            }
        )
        key = (role, source_key)
        lookup_ids.setdefault(key, []).append(anchor_id)

    # Intent-only LLM1 output names clothing but never guesses its tag. Preserve
    # that exact phrase as a source-grounded anchor even for ordinary garments
    # and character-owned profile variants. Besides making explicit wardrobe
    # evidence observable, this prevents a valid clothing request from being
    # mistaken for an unspecified creative fallback.
    if intent_only:
        for item in characters[:MAX_SEMANTIC_CHARACTERS]:
            if not isinstance(item, dict):
                continue
            wardrobe = normalized_wardrobe(item)
            kind = str(wardrobe.get("kind") or "none").strip().lower()
            if kind not in {
                "named_outfit", "outfit_source", "creative_fallback",
                "summer_profile", "winter_profile", "stage_profile",
            }:
                continue
            role = (
                "outfit"
                if kind == "named_outfit"
                else "outfit_source"
                if kind == "outfit_source"
                else "clothing"
            )
            source_text = str(
                wardrobe.get("source", wardrobe.get("source_text", "")) or ""
            ).strip()
            clothing_text = str(item.get("clothing") or "").strip()
            key = (role, normalized_source(source_text))
            if not source_text or key in lookup_ids:
                continue
            anchor_id = f"intent_source_{len(anchors) + 1}"
            anchors.append({
                "id": anchor_id,
                "role": role,
                "group": role_groups[role],
                "source_text": source_text,
                # Keep source identity and outfit qualifier as separate facts.
                # For "A wears B's stage outfit", source_text must remain B so
                # the character alias is grounded, while the full clothing phrase
                # carries "stage" into variant selection.
                "description": (
                    clothing_text
                    if kind == "outfit_source" and clothing_text
                    else source_text
                ),
                "candidates": [],
            })
            lookup_ids[key] = [anchor_id]

    for index, item in enumerate(
        characters[:MAX_SEMANTIC_CHARACTERS], start=1
    ):
        if not isinstance(item, dict):
            continue
        source_text = str(
            item.get("name", item.get("source_text", "")) or ""
        ).strip()
        target_id = f"target_{index}"
        anchors.append(
            {
                "id": target_id,
                "role": "target_character",
                "group": "character",
                "source_text": source_text,
                "description": source_text,
                "candidates": _candidate_values(item),
            }
        )
        wardrobe = normalized_wardrobe(item)
        kind = str(wardrobe.get("kind") or "none").strip().lower()
        source = str(
            wardrobe.get("source", wardrobe.get("source_text", "")) or ""
        ).strip()
        legacy_wardrobe: dict[str, Any] = {"kind": kind}
        if kind in {"named_outfit", "outfit_source"}:
            expected_role = "outfit" if kind == "named_outfit" else "outfit_source"
            matches = lookup_ids.get(
                (expected_role, normalized_source(source)), []
            )
            if len(matches) == 1:
                legacy_wardrobe["anchor_id"] = matches[0]
        changes = wardrobe.get("changes", item.get(
            "clothing_changes", item.get("changes", item.get("directives", []))
        ))
        plans.append(
            {
                "target_anchor_id": target_id,
                "wardrobe": legacy_wardrobe,
                "clothing_anchor_id": next(iter(lookup_ids.get((
                    "outfit_source" if kind == "outfit_source" else
                    "outfit" if kind == "named_outfit" else "clothing",
                    normalized_source(source),
                ), [])), ""),
                "directives": changes if isinstance(changes, list) else [],
            }
        )
    result = {"anchors": anchors, "character_plans": plans}
    if intent_only:
        result["_intent_only"] = True
    return result


def _semantic_json(raw: str, user_prompt: str = "") -> dict[str, Any]:
    """Return legacy or compact planner JSON in one internal representation."""
    data = _raw_semantic_json(raw)
    return _simple_semantic_to_legacy(data, user_prompt) if data else {}


def _normalize_outfit_directive_operation(value: Any) -> str:
    key = re.sub(r"[\s-]+", "_", str(value or "").strip().lower())
    return _OUTFIT_DIRECTIVE_OPERATION_ALIASES.get(key, "")


def _normalize_outfit_directive_slots(
    value: Any, *, unknown_to_misc: bool = True
) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    normalized: list[str] = []
    for raw_slot in value[:4]:
        slot = re.sub(r"[\s-]+", "_", str(raw_slot or "").strip().lower())
        slot = _OUTFIT_DIRECTIVE_SLOT_ALIASES.get(slot, slot)
        if unknown_to_misc and slot and slot not in _OUTFIT_DIRECTIVE_SLOTS:
            slot = "misc"
        if slot and slot not in normalized:
            normalized.append(slot)
    return tuple(normalized)


def _parse_outfit_directive_item(
    item: Any, *, user_prompt: str, target_anchor_id: str
) -> SemanticOutfitDirective | None:
    """Validate one target-bound wardrobe operation from untrusted planner JSON.

    The operation is accepted only when its evidence is present verbatim in the
    user request and its slot/color cardinality matches the operation contract.
    This prevents planner-invented removals or recolors from becoming hard tags.
    """
    if not isinstance(item, dict):
        return None
    operation = _normalize_outfit_directive_operation(item.get("operation"))
    source_text = str(item.get("source_text") or "").strip()
    raw_slots = item.get("slots")
    if not isinstance(raw_slots, list):
        return None
    slots = _normalize_outfit_directive_slots(raw_slots)
    color = str(item.get("color") or "").strip().lower()
    if (
        operation not in _OUTFIT_DIRECTIVE_OPERATIONS
        or (not slots and operation != "recolor_all")
        or any(slot not in _OUTFIT_DIRECTIVE_SLOTS for slot in slots)
        or not source_text
        or (source_text not in user_prompt and source_text.lower() not in user_prompt.lower())
    ):
        return None
    if operation == "recolor_all" and slots:
        return None
    if operation in {"remove", "damage", "replace_color", "add"} and len(slots) != 1:
        return None
    if operation in {"replace_color", "recolor_all", "add"} and color not in _OUTFIT_DIRECTIVE_COLORS:
        return None
    return SemanticOutfitDirective(operation, slots, color, source_text, target_anchor_id)


def parse_semantic_character_plans(
    raw: str,
    user_prompt: str,
    anchors: tuple[SemanticAnchor, ...] | None = None,
) -> tuple[SemanticCharacterPlan, ...]:
    """Build character-scoped wardrobe plans from validated anchor references.

    Duplicate target IDs and dangling or role-incompatible wardrobe IDs are
    discarded. Returning no plan is intentional: downstream code can then use its
    conservative creative fallback instead of assigning one character's clothes
    to another.
    """
    items = _semantic_json(raw, user_prompt).get("character_plans")
    if not isinstance(items, list):
        return ()
    validated = anchors if anchors is not None else parse_semantic_plan(raw, user_prompt)
    counts: dict[str, int] = {}
    by_id: dict[str, SemanticAnchor] = {}
    for anchor in validated:
        key = anchor.anchor_id.lower()
        counts[key] = counts.get(key, 0) + 1
        by_id[key] = anchor
    plans: list[SemanticCharacterPlan] = []
    referenced_targets: list[str] = []
    for item in items[:MAX_SEMANTIC_CHARACTERS]:
        if not isinstance(item, dict):
            continue
        target_id = str(item.get("target_anchor_id") or "").strip().lower()
        target = by_id.get(target_id)
        wardrobe_data = item.get("wardrobe")
        if (
            counts.get(target_id) != 1
            or target is None
            or target.role != "target_character"
            or not isinstance(wardrobe_data, dict)
        ):
            continue
        kind = str(wardrobe_data.get("kind") or "").strip().lower()
        wardrobe_anchor_id = str(wardrobe_data.get("anchor_id") or "").strip().lower()
        if kind not in _WARDROBE_KINDS:
            continue
        if kind in {"named_outfit", "outfit_source"}:
            wardrobe_anchor = by_id.get(wardrobe_anchor_id)
            expected_role = "outfit" if kind == "named_outfit" else "outfit_source"
            if (
                counts.get(wardrobe_anchor_id) != 1
                or wardrobe_anchor is None
                or wardrobe_anchor.role != expected_role
            ):
                continue
        elif wardrobe_anchor_id:
            continue
        raw_directives = item.get("directives", [])
        if not isinstance(raw_directives, list):
            continue
        directives = tuple(
            directive
            for raw_directive in raw_directives[:6]
            if (directive := _parse_outfit_directive_item(
                raw_directive,
                user_prompt=user_prompt,
                target_anchor_id=target_id,
            )) is not None
        )
        plans.append(SemanticCharacterPlan(
            target_anchor_id=target_id,
            wardrobe=SemanticWardrobe(kind=kind, anchor_id=wardrobe_anchor_id),
            directives=tuple(dict.fromkeys(directives)),
            clothing_anchor_id=(str(item.get("clothing_anchor_id") or "")
                                if str(item.get("clothing_anchor_id") or "") in by_id else ""),
        ))
        referenced_targets.append(target_id)
    ambiguous = {target for target in referenced_targets if referenced_targets.count(target) > 1}
    return tuple(plan for plan in plans if plan.target_anchor_id not in ambiguous)


def semantic_plan_validation_issues(raw: str, user_prompt: str) -> tuple[str, ...]:
    """Report structural relationship errors that warrant one planner retry.

    Compact output is checked only at its public boundary: grounded character
    names, a supported wardrobe kind, and an exact source-to-lookup relationship.
    Optional/unreferenced lookup details and malformed changes are simply filtered
    later rather than making the entire response fail. Legacy graph JSON retains
    its reference validation for backward compatibility.
    """
    raw_data = _raw_semantic_json(raw)
    if not raw_data:
        return ("response is not a valid JSON object",)
    if "characters" in raw_data and "lookups" not in raw_data:
        characters = raw_data.get("characters")
        if not isinstance(characters, list):
            return ("characters must be an array",)
        readable_intent = any(
            isinstance(item, dict)
            and any(
                key in item
                for key in (
                    "clothing", "clothing_source", "clothing_changes",
                    "appearance_changes",
                )
            )
            and "wardrobe" not in item
            for item in characters[:MAX_SEMANTIC_CHARACTERS]
        )
        issues: list[str] = []
        if len(characters) > MAX_SEMANTIC_CHARACTERS:
            issues.append(
                f"characters must not exceed {MAX_SEMANTIC_CHARACTERS} visible "
                f"characters; got {len(characters)}"
            )
        expected_count = _explicit_visible_character_count(user_prompt)
        if expected_count is not None and len(characters) != expected_count:
            issues.append(
                "characters must contain exactly "
                f"{expected_count} visible characters; got {len(characters)}"
            )
        request_has_parentheses = bool(re.search(r"[（(][^（）()]{1,120}[）)]", user_prompt))
        for index, item in enumerate(
            characters[:MAX_SEMANTIC_CHARACTERS], start=1
        ):
            if not isinstance(item, dict):
                issues.append(f"characters[{index}] must be an object")
                continue
            name = str(item.get("name") or "").strip()
            if not name or (
                name not in user_prompt and name.lower() not in user_prompt.lower()
            ):
                issues.append(
                    f"characters[{index}].name must be an exact phrase from the request"
                )
            if readable_intent:
                aliases = item.get("aliases")
                if aliases is None and request_has_parentheses:
                    issues.append(f"characters[{index}].aliases must be an array")
                elif aliases is not None and not isinstance(aliases, list):
                    issues.append(f"characters[{index}].aliases must be an array")
                elif isinstance(aliases, list):
                    for alias_index, alias_value in enumerate(aliases[:4], start=1):
                        alias = str(alias_value or "").strip()
                        if not alias or not _explicit_character_alias_pair(
                            user_prompt, name, alias
                        ):
                            issues.append(
                                f"characters[{index}].aliases[{alias_index}] must "
                                "form an explicit A (B) pair with that character name"
                            )
                for field in ("clothing", "clothing_source"):
                    value = item.get(field)
                    if value is None:
                        continue
                    text = str(value).strip()
                    if not text or (
                        text not in user_prompt
                        and text.lower() not in user_prompt.lower()
                    ):
                        issues.append(
                            f"characters[{index}].{field} must be null or an exact "
                            "phrase from the request"
                        )
                for field in ("clothing_changes", "appearance_changes"):
                    value = item.get(field, [])
                    if not isinstance(value, list):
                        issues.append(f"characters[{index}].{field} must be an array")
                        continue
                    if field == "clothing_changes":
                        for change_index, change in enumerate(value[:8], start=1):
                            if _parse_outfit_directive_item(
                                change,
                                user_prompt=user_prompt,
                                target_anchor_id=f"target_{index}",
                            ) is None:
                                issues.append(
                                    f"characters[{index}].clothing_changes"
                                    f"[{change_index}] must use a supported grounded "
                                    "operation, slot, color, and exact source_text"
                                )
                    else:
                        for change_index, change in enumerate(value[:8], start=1):
                            if not isinstance(change, dict):
                                issues.append(
                                    f"characters[{index}].appearance_changes"
                                    f"[{change_index}] must be an object"
                                )
                                continue
                            operation = str(change.get("operation") or "").lower()
                            if operation not in {"replace", "additive"}:
                                issues.append(
                                    f"characters[{index}].appearance_changes"
                                    f"[{change_index}].operation must be replace or "
                                    "additive"
                                )
                            source_text = str(change.get("source_text") or "").strip()
                            if not _appearance_source_is_grounded(
                                source_text, user_prompt, name
                            ):
                                issues.append(
                                    f"characters[{index}].appearance_changes"
                                    f"[{change_index}].source_text must align with "
                                    "the request"
                                )
                continue
            wardrobe = item.get("wardrobe")
            if not isinstance(wardrobe, dict):
                issues.append(f"characters[{index}].wardrobe must be an object")
                continue
            kind = str(wardrobe.get("kind") or "none").strip().lower()
            if kind not in _WARDROBE_KINDS:
                issues.append(f"characters[{index}].wardrobe.kind is invalid")
                continue
            source = str(wardrobe.get("source") or "").strip()
            grounded = bool(
                source
                and (source in user_prompt or source.lower() in user_prompt.lower())
            )
            if kind in {"named_outfit", "outfit_source", "creative_fallback"}:
                if not grounded:
                    issues.append(
                        f"characters[{index}].wardrobe.source must be an exact phrase "
                        "from the request"
                    )
            elif source and not grounded:
                issues.append(
                    f"characters[{index}].wardrobe.source must be empty or grounded"
                )
        return tuple(dict.fromkeys(issues))
    if "characters" in raw_data or "lookups" in raw_data:
        characters = raw_data.get("characters")
        lookups = raw_data.get("lookups")
        if not isinstance(characters, list) or not isinstance(lookups, list):
            return ("characters and lookups must both be arrays",)
        lookup_rows = [item for item in lookups[:8] if isinstance(item, dict)]
        issues: list[str] = []
        if len(characters) > MAX_SEMANTIC_CHARACTERS:
            issues.append(
                f"characters must not exceed {MAX_SEMANTIC_CHARACTERS} visible "
                f"characters; got {len(characters)}"
            )
        wardrobe_demands: dict[str, set[str]] = {}
        for item in characters[:MAX_SEMANTIC_CHARACTERS]:
            if not isinstance(item, dict) or not isinstance(item.get("wardrobe"), dict):
                continue
            wardrobe = item["wardrobe"]
            kind = str(wardrobe.get("kind") or "none").strip().lower()
            if kind not in {"named_outfit", "outfit_source"}:
                continue
            source = str(wardrobe.get("source") or "").strip()
            if source:
                source_key = re.sub(r"\s+", " ", source.lower())
                expected_role = "outfit" if kind == "named_outfit" else "outfit_source"
                wardrobe_demands.setdefault(source_key, set()).add(expected_role)
        expected_count = _explicit_visible_character_count(user_prompt)
        if expected_count is not None and len(characters) != expected_count:
            issues.append(
                "characters must contain exactly "
                f"{expected_count} visible characters; got {len(characters)}"
            )
        for index, item in enumerate(
            characters[:MAX_SEMANTIC_CHARACTERS], start=1
        ):
            if not isinstance(item, dict):
                issues.append(f"characters[{index}] must be an object")
                continue
            name = str(item.get("name") or "").strip()
            candidates = _candidate_values(item)
            if not name or (
                name not in user_prompt and name.lower() not in user_prompt.lower()
            ):
                issues.append(
                    f"characters[{index}].name must be an exact phrase from the request"
                )
            if not candidates:
                issues.append(f"characters[{index}].candidates must not be empty")
            wardrobe = item.get("wardrobe")
            if wardrobe is None:
                wardrobe = {"kind": "none"}
            if not isinstance(wardrobe, dict):
                issues.append(f"characters[{index}].wardrobe must be an object")
                continue
            kind = str(wardrobe.get("kind") or "none").strip().lower()
            if kind not in _WARDROBE_KINDS:
                issues.append(f"characters[{index}].wardrobe.kind is invalid")
                continue
            if kind not in {"named_outfit", "outfit_source"}:
                continue
            source = str(wardrobe.get("source") or "").strip()
            source_key = re.sub(r"\s+", " ", source.lower())
            matches = [
                lookup
                for lookup in lookup_rows
                if re.sub(
                    r"\s+", " ", str(lookup.get("text") or "").strip().lower()
                ) == source_key
                and str(lookup.get("role") or "").strip().lower()
                in {"outfit", "outfit_source"}
                and _candidate_values(lookup)
            ]
            grounded = bool(
                source
                and (source in user_prompt or source.lower() in user_prompt.lower())
            )
            if not grounded or len(matches) != 1:
                issues.append(
                    f"characters[{index}].wardrobe.source must exactly match one "
                    "wardrobe lookup from the request"
                )
            elif len(wardrobe_demands.get(source_key, set())) != 1:
                issues.append(
                    f"characters[{index}].wardrobe.source has conflicting wardrobe roles"
                )
        return tuple(dict.fromkeys(issues))

    data = _semantic_json(raw, user_prompt)
    if not data:
        return ("response is not a valid JSON object",)
    anchors = parse_semantic_plan(raw, user_prompt)
    by_id = {anchor.anchor_id.lower(): anchor for anchor in anchors}
    target_ids = [
        anchor.anchor_id.lower()
        for anchor in anchors
        if anchor.role == "target_character"
    ]
    raw_plans = data.get("character_plans")
    # An omitted plan is the planner's semantic abstention and must continue to
    # degrade to explicit_but_unresolved.  Retry only a plan it actually attempted
    # but encoded with broken references; otherwise ordinary no-plan responses
    # would unexpectedly add another paid LLM call.
    if not isinstance(raw_plans, list):
        return ()
    issues: list[str] = []
    if isinstance(raw_plans, list):
        seen_targets: list[str] = []
        for index, item in enumerate(
            raw_plans[:MAX_SEMANTIC_CHARACTERS], start=1
        ):
            if not isinstance(item, dict):
                issues.append(f"character_plans[{index}] is not an object")
                continue
            target_id = str(item.get("target_anchor_id") or "").strip().lower()
            target = by_id.get(target_id)
            if target is None or target.role != "target_character":
                issues.append(
                    f"character_plans[{index}].target_anchor_id does not reference "
                    "one target_character anchor"
                )
            else:
                seen_targets.append(target_id)
            wardrobe = item.get("wardrobe")
            if not isinstance(wardrobe, dict):
                issues.append(f"character_plans[{index}].wardrobe must be an object")
                continue
            kind = str(wardrobe.get("kind") or "").strip().lower()
            anchor_id = str(wardrobe.get("anchor_id") or "").strip().lower()
            if kind not in _WARDROBE_KINDS:
                issues.append(f"character_plans[{index}].wardrobe.kind is invalid")
            elif kind in {"named_outfit", "outfit_source"}:
                expected = "outfit" if kind == "named_outfit" else "outfit_source"
                wardrobe_anchor = by_id.get(anchor_id)
                if not anchor_id:
                    issues.append(
                        f"character_plans[{index}].wardrobe.anchor_id is required "
                        f"when kind is {kind}"
                    )
                elif wardrobe_anchor is None or wardrobe_anchor.role != expected:
                    issues.append(
                        f"character_plans[{index}].wardrobe.anchor_id must reference "
                        f"one {expected} anchor"
                    )
            elif anchor_id:
                issues.append(
                    f"character_plans[{index}].wardrobe.anchor_id must be omitted "
                    f"when kind is {kind}"
                )
        for target_id in target_ids:
            count = seen_targets.count(target_id)
            if count == 0:
                issues.append(f"target_character {target_id} has no character_plans item")
            elif count > 1:
                issues.append(f"target_character {target_id} has duplicate character_plans items")
    return tuple(dict.fromkeys(issues))


def build_semantic_plan_repair_prompt(
    user_prompt: str, previous_raw: str, issues: tuple[str, ...]
) -> str:
    """Ask the same planner to repair references without changing user semantics."""
    issue_lines = "\n".join(f"- {issue}" for issue in issues)
    return (
        build_semantic_plan_prompt(user_prompt)
        + "\n\nYour previous response failed structural validation:\n"
        + issue_lines
        + "\nRepair the complete intent JSON object. Preserve every correctly "
        "understood character, clothing change, and appearance request. Do not "
        "invent tags, candidates, lookups, IDs, or unstated facts. Every name and "
        "source_text must be copied from the request. Return the corrected complete "
        "JSON object only.\n\nPrevious response:\n"
        + str(previous_raw or "")
    )


def parse_semantic_outfit_directives(
    raw: str, user_prompt: str
) -> tuple[SemanticOutfitDirective, ...]:
    """Parse only explicit, source-grounded outfit operations from a plan."""
    text = re.sub(r"^```(?:json)?\s*", "", str(raw or "").strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return ()
    items = data.get("outfit_directives") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return ()
    anchors = data.get("anchors") if isinstance(data, dict) else None
    target_anchor_ids = {
        str(anchor.get("id") or "").strip().lower()
        for anchor in anchors or ()
        if isinstance(anchor, dict)
        and str(anchor.get("role") or "").strip().lower() == "target_character"
        and str(anchor.get("id") or "").strip()
    }
    if not target_anchor_ids:
        return ()
    directives: list[SemanticOutfitDirective] = []
    for item in items[:6]:
        if not isinstance(item, dict):
            continue
        operation = _normalize_outfit_directive_operation(item.get("operation"))
        source_text = str(item.get("source_text") or "").strip()
        target_anchor_id = str(item.get("target_anchor_id") or "").strip().lower()
        if not target_anchor_id and len(target_anchor_ids) == 1:
            target_anchor_id = next(iter(target_anchor_ids))
        raw_slots = item.get("slots")
        if not isinstance(raw_slots, list):
            continue
        slots = _normalize_outfit_directive_slots(raw_slots, unknown_to_misc=False)
        color = str(item.get("color") or "").strip().lower()
        if (
            operation not in _OUTFIT_DIRECTIVE_OPERATIONS
            or (not slots and operation != "recolor_all")
            or any(slot not in _OUTFIT_DIRECTIVE_SLOTS for slot in slots)
            or not source_text
            or target_anchor_id not in target_anchor_ids
            or (source_text not in user_prompt and source_text.lower() not in user_prompt.lower())
        ):
            continue
        if operation == "recolor_all" and slots:
            continue
        if operation == "keep_only" and len(slots) > 4:
            continue
        if operation in {"remove", "damage", "replace_color", "add"} and len(slots) != 1:
            continue
        if operation in {"replace_color", "recolor_all", "add"} and color not in _OUTFIT_DIRECTIVE_COLORS:
            continue
        directive = SemanticOutfitDirective(
            operation, slots, color, source_text, target_anchor_id
        )
        if directive not in directives:
            directives.append(directive)
    return tuple(directives)


def _literal_phrase_pattern(value: str) -> str:
    """Return a case-insensitive regex fragment for one exact copied phrase."""
    parts = re.split(r"(\s+)", str(value or "").strip())
    return "".join(r"\s+" if part.isspace() else re.escape(part) for part in parts)


def _explicit_character_alias_pair(
    user_prompt: str, character_name: str, alias: str
) -> bool:
    """Check that two planner fields are the two sides of one literal A (B)."""
    name_pattern = _literal_phrase_pattern(character_name)
    alias_pattern = _literal_phrase_pattern(alias)
    if not name_pattern or not alias_pattern or character_name.strip() == alias.strip():
        return False
    pair_patterns = (
        rf"{name_pattern}\s*[（(]\s*{alias_pattern}\s*[）)]",
        rf"{alias_pattern}\s*[（(]\s*{name_pattern}\s*[）)]",
    )
    return any(re.search(pattern, user_prompt, re.I) for pattern in pair_patterns)


def parse_semantic_character_aliases(
    raw: str, user_prompt: str
) -> tuple[tuple[str, str], ...]:
    """Read source-grounded character alias pairs explicitly classified by LLM1."""
    data = _raw_semantic_json(raw)
    characters = data.get("characters") if isinstance(data, dict) else None
    if not isinstance(characters, list):
        return ()
    pairs: list[tuple[str, str]] = []
    for item in characters[:MAX_SEMANTIC_CHARACTERS]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        aliases = item.get("aliases", [])
        if not name or not isinstance(aliases, list):
            continue
        for value in aliases[:4]:
            alias = str(value or "").strip()
            pair = (name, alias)
            if (
                alias
                and _explicit_character_alias_pair(user_prompt, name, alias)
                and pair not in pairs
            ):
                pairs.append(pair)
    return tuple(pairs)


def extract_parenthesized_character_aliases(
    user_prompt: str,
    character_alias_pairs: tuple[tuple[str, str], ...] = (),
) -> tuple[SemanticAnchor, ...]:
    """Create character anchors from LLM1-classified, locally paired aliases.

    The host does not guess the left edge of ``A (B)``. LLM1 supplies the exact
    character entity and its aliases; this function independently verifies that
    the two strings are adjacent sides of the same parenthesized pair. This keeps
    new-character learning available without promoting appearance translations.
    """
    anchors: list[SemanticAnchor] = []
    seen_candidates: set[str] = set()
    for character_name, alias in character_alias_pairs:
        if not _explicit_character_alias_pair(user_prompt, character_name, alias):
            continue
        lookup_text = alias
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9 _.'():!\-]{1,119}", lookup_text):
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9 _.'():!\-]{1,119}", character_name):
                lookup_text = character_name
            else:
                continue
        candidate = re.sub(r"\s+", "_", lookup_text.lower()).strip("_")
        if (
            not re.fullmatch(r"[a-z0-9_.'():!\-]{2,120}", candidate)
            or candidate in seen_candidates
        ):
            continue
        seen_candidates.add(candidate)
        anchors.append(
            SemanticAnchor(
                anchor_id=f"parenthesized_character_alias_{len(anchors) + 1}",
                role="target_character",
                group="character",
                source_text=lookup_text,
                description=f"Explicit character alias for {character_name}",
                candidates=(candidate,),
            )
        )
    return tuple(anchors)


def bind_parenthesized_character_aliases(
    anchors: tuple[SemanticAnchor, ...],
    user_prompt: str,
    pairs: tuple[tuple[str, str], ...],
) -> tuple[SemanticAnchor, ...]:
    """Attach verified alias syntax to its owner's existing wardrobe-plan ID."""
    result = list(anchors)
    for name, alias in pairs:
        aliases = extract_parenthesized_character_aliases(user_prompt, ((name, alias),))
        if not aliases:
            continue
        evidence = aliases[0]
        owners = [i for i, anchor in enumerate(result)
                  if anchor.role == "target_character"
                  and anchor.source_text.casefold() == name.casefold()]
        if len(owners) == 1:
            index = owners[0]
            # Two Latin sides may be a canonical qualified name, not a
            # translation pair. Keep the literal whole name for exact lookup;
            # never silently reduce ``name (costume)`` to the base character.
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9 _.'!:()\-]{1,119}", name) and re.fullmatch(
                r"[A-Za-z][A-Za-z0-9 _.'!:()\-]{1,119}", alias
            ):
                whole = re.search(
                    rf"{_literal_phrase_pattern(name)}\s*[（(]\s*{_literal_phrase_pattern(alias)}\s*[）)]",
                    user_prompt, re.I,
                )
                if whole:
                    literal = whole.group().replace("（", "(").replace("）", ")")
                    candidate = re.sub(r"\s+", "_", literal.lower())
                    candidate = re.sub(r"_*\(", "_(", candidate)
                    result[index] = replace(result[index], source_text=literal, literal_name=literal, candidates=(candidate,))
                    continue
            result[index] = replace(result[index], candidates=evidence.candidates,
                                    literal_name=evidence.source_text if "_(" in evidence.candidates[0] else "")
        elif not owners:
            result.append(replace(evidence, anchor_id=f"parenthesized_character_alias_{len(result) + 1}"))
    return tuple(result)


def extract_parenthesized_copyright_aliases(user_prompt: str) -> tuple[SemanticAnchor, ...]:
    """Extract explicit English work titles paired with a Chinese book-title mark.

    Args:
        user_prompt: The original image request.

    Returns:
        Copyright lookup anchors independent of the semantic-planning LLM.
    """
    anchors: list[SemanticAnchor] = []
    prompt = str(user_prompt or "")
    for match in re.finditer(
        r"《(?P<title>[^》]{1,80})》\s*[（(]\s*"
        r"(?P<alias>[A-Za-z][A-Za-z0-9 _.'!: -]{1,78})\s*[）)]",
        prompt,
    ):
        title = match.group("title").strip()
        alias = match.group("alias").strip()
        candidate = re.sub(r"\s+", "_", alias.lower()).strip("_")
        if not re.fullmatch(r"[a-z0-9_.'():!\-]{2,120}", candidate):
            continue
        anchors.append(
            SemanticAnchor(
                anchor_id=f"parenthesized_copyright_alias_{len(anchors) + 1}",
                role="copyright",
                group="series",
                source_text=alias,
                description=f"Explicit work title for {title}",
                candidates=(candidate,),
            )
        )
    return tuple(anchors)


def resolve_local_cli_path(configured_path: str = "") -> Path | None:
    """Resolve one explicit/well-known local CLI path without filesystem search."""
    candidates: list[Path] = []
    if str(configured_path or "").strip():
        candidates.append(Path(str(configured_path).strip()).expanduser())
    skills_root = str(os.environ.get("COMFYUI_GOOD_ANIMA_SKILLS_DIR") or "").strip()
    if skills_root:
        candidates.append(Path(skills_root) / "danbooru-tags" / "bin" / "danbooru-tags.exe")
    candidates.append(
        Path.home() / ".codex" / "skills" / "danbooru-tags" / "bin" / "danbooru-tags.exe"
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _run_cli_batch(
    queries: list[dict[str, Any]],
    *,
    cli_path: Path,
    timeout: float,
    exact_only: bool = False,
) -> dict[str, Any] | None:
    """Run one bounded local CLI batch and return decoded JSON."""
    payload = json.dumps({"queries": queries}, ensure_ascii=False)
    batch_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", delete=False
        ) as handle:
            handle.write(payload)
            batch_path = handle.name
        command = [
            str(cli_path),
            "--batch-file",
            batch_path,
            "--batch-workers",
            "4",
            "--for-prompt",
            "--json",
            "--compact",
        ]
        if exact_only:
            command.extend(("--match-mode", "exact"))
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1.0, min(float(timeout), 20.0)),
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        if batch_path:
            try:
                Path(batch_path).unlink(missing_ok=True)
            except OSError:
                pass
    if completed.returncode != 0:
        return None
    try:
        data = json.loads(completed.stdout)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def lookup_semantic_anchors(
    anchors: tuple[SemanticAnchor, ...],
    *,
    cli_path: Path,
    timeout: float = 8.0,
) -> SemanticLookupResult:
    """Validate planner candidates and return only locally supported hard tags.

    Args:
        anchors: Source-grounded concepts with untrusted candidate spellings.
        cli_path: Local Danbooru lookup executable or script.
        timeout: Maximum duration for each bounded batch lookup.

    Returns:
        A result whose ``confirmed_tags`` may be emitted as hard tags. Candidate
        tags remain diagnostic hints; CLI failure is reported through ``status``
        and does not raise into the generation pipeline.
    """
    if not anchors:
        return SemanticLookupResult(status="empty_plan")
    queries: list[dict[str, Any]] = []
    query_map: dict[str, tuple[int, int, bool, str]] = {}
    for anchor_index, anchor in enumerate(anchors):
        primary_candidates = list(anchor.candidates[:2])
        raw_phrase = str(anchor.source_text or "").strip()
        fallback_phrase = ""
        if re.fullmatch(r"[A-Za-z0-9_.'():!\- ]{2,80}", raw_phrase):
            fallback_phrase = raw_phrase
        query_candidates = list(primary_candidates)
        if fallback_phrase and re.sub(r"\s+", "_", fallback_phrase.lower()) not in {
            re.sub(r"\s+", "_", item.lower()) for item in query_candidates
        }:
            query_candidates.append(fallback_phrase)
        for candidate_index, candidate in enumerate(query_candidates[:3]):
            query_id = f"a{anchor_index}_c{candidate_index}"
            candidate_key = re.sub(r"\s+", "_", candidate.lower()).strip("_")
            source_key = re.sub(r"\s+", "_", raw_phrase.lower()).strip("_")
            # A raw source-word query needs intent-filtered candidate handling
            # even when the planner happened to include that same word among
            # its primary guesses.
            is_fallback = bool(source_key and candidate_key == source_key)
            query_map[query_id] = (
                anchor_index,
                candidate_index,
                is_fallback,
                candidate,
            )
            queries.append(
                {
                    "id": query_id,
                    "group": anchor.group,
                    "keyword": candidate,
                    "limit": 5 if anchor.group in {"character", "series"} else 10,
                }
            )
    data = _run_cli_batch(queries, cli_path=cli_path, timeout=timeout)
    if data is None:
        return SemanticLookupResult(anchors=anchors, status="tool_failed")
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, dict):
        return SemanticLookupResult(anchors=anchors, status="tool_failed")

    confirmed_by_anchor: dict[int, str] = {}
    inferred_series_tags: list[str] = []
    candidates: list[str] = []
    fallback_candidates: dict[int, list[tuple[str, int]]] = {}
    named_outfit_by_anchor: dict[int, str] = {}

    def is_specific_school_uniform(anchor: SemanticAnchor) -> bool:
        source = re.sub(r"[\s\"“”'‘’]+", "", anchor.source_text).lower()
        if not re.search(r"(?:校服|制服|schooluniform)$", source, flags=re.I):
            return False
        modifier = re.sub(
            r"(?:校服|制服|schooluniform)$", "", source, flags=re.I
        ).strip()
        return len(modifier) >= 2

    def outfit_tag_has_required_specificity(
        anchor: SemanticAnchor, tag: str
    ) -> bool:
        tag_key = re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")
        if is_specific_school_uniform(anchor) and tag_key in {
            "school_uniform",
            "uniform",
        }:
            return False
        return True

    def is_named_outfit_anchor(anchor: SemanticAnchor, tag: str) -> bool:
        tag_key = re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")
        return outfit_tag_has_required_specificity(anchor, tag) and (
            anchor.role == "outfit" or bool(
            anchor.role == "clothing"
            and tag_key.endswith("_uniform")
            and re.search(r"(?:校服|制服|uniform)", anchor.source_text, flags=re.I)
            )
        )

    for query_id, result in results.items():
        mapping = query_map.get(str(query_id))
        if not isinstance(result, dict):
            continue
        if mapping is None:
            continue
        anchor_index, candidate_index, is_fallback, query_phrase = mapping
        confirmed = result.get("confirmed_tags")
        if isinstance(confirmed, dict):
            records = [
                record
                for values in confirmed.values()
                if isinstance(values, list)
                for record in values
                if isinstance(record, dict)
            ]
            if records and (
                anchor_index not in confirmed_by_anchor
                or candidate_index == 0
            ):
                tag = next(
                    (
                        str(record.get("tag") or "").strip()
                        for record in records
                        if outfit_tag_has_required_specificity(
                            anchors[anchor_index],
                            str(record.get("tag") or "").strip(),
                        )
                    ),
                    "",
                )
                if tag:
                    confirmed_by_anchor[anchor_index] = tag
                    if is_named_outfit_anchor(anchors[anchor_index], tag):
                        named_outfit_by_anchor[anchor_index] = tag
        raw_candidates = result.get("candidate_tags")
        if isinstance(raw_candidates, dict):
            for values in raw_candidates.values():
                if not isinstance(values, list):
                    continue
                for record in values[:3]:
                    if not isinstance(record, dict):
                        continue
                    tag = str(record.get("tag") or "").strip()
                    if tag and tag not in candidates:
                        candidates.append(tag)
                    anchor = anchors[anchor_index]
                    tag_key = re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")
                    query_key = re.sub(
                        r"[^a-z0-9]+", "_", query_phrase.lower()
                    ).strip("_")
                    try:
                        count = int(record.get("count") or 0)
                    except (TypeError, ValueError):
                        count = 0
                    match_layer = str(record.get("match_layer") or "").strip()
                    source_category = str(
                        record.get("source_category") or record.get("category") or ""
                    ).strip()
                    # The local index classifies many canonical clothing-set tags as
                    # general tags.  They therefore arrive as candidates even when
                    # their spelling exactly matches the planner's explicit lookup.
                    # Promote only a high-evidence, user-requested complete ensemble;
                    # ordinary garments and fuzzy candidates remain non-authoritative.
                    if (
                        is_named_outfit_anchor(anchor, tag)
                        and tag_key == query_key
                        and match_layer == "group_general_fallback"
                        and source_category == "general"
                        and count >= 10
                    ):
                        confirmed_by_anchor[anchor_index] = tag
                        named_outfit_by_anchor[anchor_index] = tag
                        continue
                    if not tag or not is_fallback:
                        continue
                    if anchor.group != "character":
                        continue
                    if tag_key != query_key and not tag.lower().startswith(
                        query_key + "_("
                    ):
                        continue
                    values = fallback_candidates.setdefault(anchor_index, [])
                    if (tag, count) not in values:
                        values.append((tag, count))

    confirmed_copyright_tags = {
        tag
        for index, tag in confirmed_by_anchor.items()
        if anchors[index].role == "copyright"
    }
    selected_fallbacks: dict[int, tuple[str, int]] = {}
    for index, values in fallback_candidates.items():
        compatible = []
        for tag, count in values:
            scoped = re.fullmatch(r".+_\(([^)]+)\)", tag)
            scope = scoped.group(1) if scoped else ""
            if any(
                copyright == scope
                or copyright.startswith(scope + "_")
                or scope.startswith(copyright + "_")
                for copyright in confirmed_copyright_tags
            ):
                compatible.append((tag, count))
        if confirmed_copyright_tags:
            selected = compatible
        elif len(values) == 1:
            selected = values
        else:
            continue
        selected_fallbacks[index] = max(selected, key=lambda item: item[1])

    unresolved_fallbacks = {
        index: value[0]
        for index, value in selected_fallbacks.items()
        if index not in confirmed_by_anchor
    }
    # Fuzzy lookup is used only to discover a plausible scoped character tag.
    # Re-query it in exact-only mode before promotion; otherwise a popular fuzzy
    # match could silently become authoritative identity evidence.
    fallback_queries = [
            {
                "id": f"fallback_{index}",
                "group": anchors[index].group,
                "keyword": tag,
                "limit": 5,
            }
            for index, tag in unresolved_fallbacks.items()
        ]
    series_candidates: list[str] = []
    for tag in (*confirmed_by_anchor.values(), *unresolved_fallbacks.values()):
        scoped = re.fullmatch(r".+_\(([^)]+)\)", tag)
        if scoped and scoped.group(1) not in series_candidates:
            series_candidates.append(scoped.group(1))
    fallback_queries.extend(
        {
            "id": f"series_{index}",
            "group": "series",
            "keyword": series,
            "limit": 5,
        }
        for index, series in enumerate(series_candidates)
    )
    if fallback_queries:
        fallback_data = _run_cli_batch(
            fallback_queries,
            cli_path=cli_path,
            timeout=timeout,
            exact_only=True,
        )
        fallback_results = (
            fallback_data.get("results") if isinstance(fallback_data, dict) else None
        )
        if isinstance(fallback_results, dict):
            for index, selected_tag in unresolved_fallbacks.items():
                result = fallback_results.get(f"fallback_{index}")
                confirmed = result.get("confirmed_tags") if isinstance(result, dict) else None
                records = [
                    record
                    for values in confirmed.values()
                    if isinstance(confirmed, dict) and isinstance(values, list)
                    for record in values
                    if isinstance(record, dict)
                ] if isinstance(confirmed, dict) else []
                if any(str(record.get("tag") or "") == selected_tag for record in records):
                    confirmed_by_anchor[index] = selected_tag
            for index, series in enumerate(series_candidates):
                result = fallback_results.get(f"series_{index}")
                confirmed = result.get("confirmed_tags") if isinstance(result, dict) else None
                if not isinstance(confirmed, dict):
                    continue
                for values in confirmed.values():
                    if not isinstance(values, list):
                        continue
                    for record in values:
                        if not isinstance(record, dict):
                            continue
                        tag = str(record.get("tag") or "").strip()
                        if tag == series and tag not in inferred_series_tags:
                            inferred_series_tags.append(tag)

    confirmed_tags: list[str] = []
    source_tags: list[str] = []
    missing: list[str] = []
    role_priority = {
        "target_character": 0,
        "outfit_source": 1,
        "copyright": 2,
        "clothing": 3,
        "outfit": 3,
        "appearance": 4,
        "accessory": 5,
        "prop": 6,
        "expression": 7,
        "pose": 8,
        "action": 8,
        "scene": 9,
        "lighting": 10,
    }
    ordered_indices = sorted(
        range(len(anchors)),
        key=lambda index: (role_priority.get(anchors[index].role, 50), index),
    )
    series_inserted = False
    for index in ordered_indices:
        anchor = anchors[index]
        tag = confirmed_by_anchor.get(index, "")
        if tag:
            if tag not in confirmed_tags:
                confirmed_tags.append(tag)
            if anchor.role == "outfit_source" and tag not in source_tags:
                source_tags.append(tag)
                for series_tag in inferred_series_tags:
                    if series_tag not in confirmed_tags:
                        confirmed_tags.append(series_tag)
                series_inserted = True
        elif anchor.description not in missing:
            missing.append(anchor.description)
    if not series_inserted:
        for tag in inferred_series_tags:
            if tag not in confirmed_tags:
                confirmed_tags.append(tag)
    return SemanticLookupResult(
        confirmed_tags=tuple(confirmed_tags),
        outfit_source_tags=tuple(source_tags),
        named_outfit_tags=tuple(
            dict.fromkeys(named_outfit_by_anchor.values())
        ),
        missing_descriptions=tuple(missing),
        candidate_tags=tuple(candidates[:12]),
        anchors=anchors,
        anchor_tags=tuple(
            (anchors[index].anchor_id, tag)
            for index, tag in confirmed_by_anchor.items()
        ),
        status="resolved",
    )


def merge_semantic_results(
    *results: SemanticLookupResult | None,
) -> SemanticLookupResult:
    """Merge cache and request lookup evidence without letting either shadow the other."""
    present = tuple(result for result in results if result is not None)
    if not present:
        return SemanticLookupResult()

    def merged(field: str) -> tuple[Any, ...]:
        return tuple(
            dict.fromkeys(
                item
                for result in present
                for item in getattr(result, field, ())
            )
        )

    statuses = tuple(result.status for result in present if result.status)
    if "resolved" in statuses:
        status = "resolved"
    elif "profile_cache" in statuses:
        status = "profile_cache"
    else:
        status = statuses[-1] if statuses else "not_available"
    return SemanticLookupResult(
        confirmed_tags=merged("confirmed_tags"),
        outfit_source_tags=merged("outfit_source_tags"),
        outfit_profile_tags=merged("outfit_profile_tags"),
        appearance_profile_tags=merged("appearance_profile_tags"),
        named_outfit_tags=merged("named_outfit_tags"),
        source_outfit_profiles=merged("source_outfit_profiles"),
        character_appearance_profiles=merged("character_appearance_profiles"),
        anchor_tags=merged("anchor_tags"),
        character_profiles=merged("character_profiles"),
        anchor_outfit_profiles=merged("anchor_outfit_profiles"),
        missing_descriptions=merged("missing_descriptions"),
        candidate_tags=merged("candidate_tags"),
        anchors=merged("anchors"),
        status=status,
    )


def prefer_configured_character_anchors(
    configured: tuple[SemanticAnchor, ...],
    discovered: tuple[SemanticAnchor, ...],
) -> tuple[SemanticAnchor, ...]:
    """Resolve identity aliases in configured, explicit, then planner order.

    Configured mappings are inserted first and suppress equivalent discovered
    anchors. The caller orders parenthesized aliases before planner output, so an
    explicit ``中文名 (canonical name)`` survives ahead of an LLM guess. Unrelated
    discovered anchors keep request order. This precedence prevents a plausible
    hallucinated spelling from replacing a user-maintained or explicit identity.
    """
    if not configured:
        return tuple(dict.fromkeys(discovered))

    def normalized(value: str) -> str:
        return re.sub(
            r"[^a-z0-9\u3400-\u9fff]+", "_", value.lower()
        ).strip("_")

    def terms(anchor: SemanticAnchor) -> set[str]:
        values = {
            normalized(anchor.source_text),
            *(normalized(item) for item in anchor.candidates),
        }
        for candidate in anchor.candidates:
            scoped = re.fullmatch(r"(.+)_\([^)]+\)", candidate.strip().lower())
            if scoped:
                values.add(normalized(scoped.group(1)))
        return {value for value in values if value}

    configured_by_role = {
        role: tuple(anchor for anchor in configured if anchor.role == role)
        for role in {anchor.role for anchor in configured}
    }
    configured_copyright_terms = {
        value
        for anchor in configured_by_role.get("copyright", ())
        for value in terms(anchor)
    }
    # Character plans refer to host-generated planner IDs. When a configured
    # alias supplies authoritative identity evidence, keep that discovered ID
    # while replacing only its candidates; otherwise the valid plan would dangle.
    substituted_discovered: set[SemanticAnchor] = set()
    kept: list[SemanticAnchor] = []
    for configured_anchor in dict.fromkeys(configured):
        equivalent = next(
            (
                anchor
                for anchor in discovered
                if anchor.role == configured_anchor.role
                and anchor.role == "target_character"
                and terms(anchor) & terms(configured_anchor)
            ),
            None,
        )
        if equivalent is None:
            kept.append(configured_anchor)
            continue
        kept.append(replace(
            equivalent,
            description=configured_anchor.description,
            candidates=configured_anchor.candidates,
        ))
        substituted_discovered.add(equivalent)
    for anchor in discovered:
        if anchor in substituted_discovered:
            continue
        anchor_terms = terms(anchor)
        if (
            anchor.role == "target_character"
            and anchor_terms & configured_copyright_terms
        ):
            # A loose ``作品(alias)的角色(alias)`` extractor can mistake the
            # work alias for a second visible character.
            continue
        equivalents = configured_by_role.get(anchor.role, ())
        if any(anchor_terms & terms(candidate) for candidate in equivalents):
            continue
        if anchor not in kept:
            kept.append(anchor)
    return tuple(kept)
