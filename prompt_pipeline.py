from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

try:
    from .danbooru_resolver import DanbooruResolveOutcome, DanbooruResolver
    from .danbooru_semantic import (
        DEFAULT_SEMANTIC_PLAN_SYSTEM_PROMPT,
        LEGACY_SEMANTIC_PLAN_SYSTEM_PROMPTS,
        SemanticAnchor,
        SemanticAppearanceChange,
        SemanticCharacterPlan,
        SemanticWardrobe,
        SemanticOutfitDirective,
        SemanticLookupResult,
        build_semantic_plan_prompt,
        build_semantic_plan_repair_prompt,
        bind_parenthesized_character_aliases,
        extract_parenthesized_copyright_aliases,
        merge_semantic_results,
        prefer_configured_character_anchors,
        parse_semantic_plan,
        parse_semantic_appearance_changes,
        parse_semantic_character_aliases,
        parse_semantic_character_plans,
        parse_semantic_outfit_directives,
        semantic_plan_validation_issues,
    )
    from .multi_person_prompt import (
        build_multi_person_plan_prompt,
        parse_multi_person_plan,
        render_multi_person_character,
    )
    from .nai_character_mode import (
        build_nai_character_plan_prompt,
        parse_nai_character_plan,
        preserve_nai_global_artist_tags,
        resolve_nai_canvas,
        strip_nai_character_switch,
    )
    from .outfit_transfer import (
        build_effective_outfit_plan,
        build_outfit_constraint_narrative,
        build_outfit_summary_prompt,
        build_outfit_transfer_block,
        bind_explicit_outfit_patch_target,
        detect_outfit_transfer,
        extract_reference_tag_text,
        filter_outfit_tags,
        keep_only_verified_outfit_tags,
        outfit_tag_slot,
        parse_user_outfit_patches,
        preferred_search_prompt,
        rewrite_target_outfit_detail,
        UserOutfitPatch,
        EffectiveOutfitPlan,
        OutfitTransferPlan,
    )
    from .prompt_background import (
        DEFAULT_PORTRAIT,
        EXPLICIT_SCENE,
        enforce_user_background_intent,
        extract_background_mode,
        framing_hidden_outfit_slots,
        has_generated_scene,
        strip_default_portrait_prose,
        strip_default_portrait_tags,
        strip_unrequested_default_background_prose,
        user_requests_explicit_background,
    )
    from .prompt_builder import (
        build_final_prompt,
    )
    from .prompt_constraints import (
        build_constraint_plan_prompt,
        parse_constraint_plan,
    )
    from .prompt_presets import (
        active_artist_tags,
        apply_config_preset,
        extract_artist_preset_switch,
        fixed_character_tags,
        mentioned_fixed_characters,
        selected_fixed_character,
        strip_raw_prefix,
        wants_sensual_mode,
    )
    from .prompt_keyword_rules import (
        build_keyword_rule_block,
        match_keyword_prompt_rules,
    )
    from .prompt_research import PromptResearcher
    from .prompt_templates import build_llm_prompt, has_positive_futa_request
    from .tag_cleaner import clean_content_tags, split_tags
except ImportError:  # pragma: no cover - fallback for direct script-style imports.
    from danbooru_resolver import DanbooruResolveOutcome, DanbooruResolver
    from danbooru_semantic import (
        DEFAULT_SEMANTIC_PLAN_SYSTEM_PROMPT,
        LEGACY_SEMANTIC_PLAN_SYSTEM_PROMPTS,
        SemanticAnchor,
        SemanticAppearanceChange,
        SemanticCharacterPlan,
        SemanticWardrobe,
        SemanticOutfitDirective,
        SemanticLookupResult,
        build_semantic_plan_prompt,
        build_semantic_plan_repair_prompt,
        bind_parenthesized_character_aliases,
        extract_parenthesized_copyright_aliases,
        merge_semantic_results,
        prefer_configured_character_anchors,
        parse_semantic_plan,
        parse_semantic_appearance_changes,
        parse_semantic_character_aliases,
        parse_semantic_character_plans,
        parse_semantic_outfit_directives,
        semantic_plan_validation_issues,
    )
    from multi_person_prompt import (
        build_multi_person_plan_prompt,
        parse_multi_person_plan,
        render_multi_person_character,
    )
    from nai_character_mode import (
        build_nai_character_plan_prompt,
        parse_nai_character_plan,
        preserve_nai_global_artist_tags,
        resolve_nai_canvas,
        strip_nai_character_switch,
    )
    from outfit_transfer import (
        build_effective_outfit_plan,
        build_outfit_constraint_narrative,
        build_outfit_summary_prompt,
        build_outfit_transfer_block,
        bind_explicit_outfit_patch_target,
        detect_outfit_transfer,
        extract_reference_tag_text,
        filter_outfit_tags,
        keep_only_verified_outfit_tags,
        outfit_tag_slot,
        parse_user_outfit_patches,
        preferred_search_prompt,
        rewrite_target_outfit_detail,
        UserOutfitPatch,
        EffectiveOutfitPlan,
        OutfitTransferPlan,
    )
    from prompt_background import (
        DEFAULT_PORTRAIT,
        EXPLICIT_SCENE,
        enforce_user_background_intent,
        extract_background_mode,
        framing_hidden_outfit_slots,
        has_generated_scene,
        strip_default_portrait_prose,
        strip_default_portrait_tags,
        strip_unrequested_default_background_prose,
        user_requests_explicit_background,
    )
    from prompt_builder import (
        build_final_prompt,
    )
    from prompt_constraints import (
        build_constraint_plan_prompt,
        parse_constraint_plan,
    )
    from prompt_presets import (
        active_artist_tags,
        apply_config_preset,
        extract_artist_preset_switch,
        fixed_character_tags,
        mentioned_fixed_characters,
        selected_fixed_character,
        strip_raw_prefix,
        wants_sensual_mode,
    )
    from prompt_keyword_rules import (
        build_keyword_rule_block,
        match_keyword_prompt_rules,
    )
    from prompt_research import PromptResearcher
    from prompt_templates import build_llm_prompt, has_positive_futa_request
    from tag_cleaner import clean_content_tags, split_tags


def _extract_completion_text(response: Any) -> str:
    """Extract LLM completion text across provider response shapes.

    Reasoning models frequently park the visible answer in a dedicated field
    while ``completion_text`` stays empty (for example when a long chain of
    thought consumed the token budget).  Try the common field names before
    falling back to the generic message-chain conventions.

    Args:
        response: Object returned by AstrBot ``llm_generate``.

    Returns:
        Stripped completion text, or an empty string when unavailable.
    """
    candidates = (
        "completion_text",
        "text",
        "content",
        "completion",
        "answer",
        "response",
        "output",
        "result",
        "reasoning_content",
        "reasoning",
        "thinking_content",
        "message",
    )
    for name in candidates:
        try:
            value = getattr(response, name, None)
        except AttributeError:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            parts: list[str] = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                    continue
                if isinstance(item, dict):
                    for key in ("text", "content", "message"):
                        chunk = item.get(key)
                        if isinstance(chunk, str) and chunk.strip():
                            parts.append(chunk.strip())
                elif hasattr(item, "content"):
                    chunk = getattr(item, "content", None)
                    if isinstance(chunk, str) and chunk.strip():
                        parts.append(chunk.strip())
            if parts:
                return "\n".join(parts).strip()
    # Generic message-chain convention used by some AstrBot wrappers.
    for name in ("messages", "chain", "choices"):
        try:
            chain = getattr(response, name, None)
        except AttributeError:
            continue
        if not isinstance(chain, (list, tuple)) or not chain:
            continue
        for item in chain:
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, dict):
                message = item.get("message") or item
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return ""


@dataclass(frozen=True)
class PromptPipelineResult:
    """Prompt pipeline output and debug summary.

    Args:
        final_prompt: Prompt sent to the ComfyUI tool.
        summary: Non-secret prompt build summary for logs and last_task.json.
    """

    final_prompt: str
    summary: dict[str, Any]


@dataclass(frozen=True)
class StructuredPromptCharacter:
    """One character section returned by the prompt-building LLM.

    Args:
        name: User-facing source name or Danbooru candidate from the roster.
        identity_tags: Stable identity traits, such as hair and eye color.
        detail_tags: Mutable costume, expression, pose, and prop tags.
    """

    name: str
    identity_tags: str
    detail_tags: str


@dataclass(frozen=True)
class StructuredPromptParseResult:
    """Lossless best-effort parse of the seven-field writer response.

    Field-envelope and count failures are validation errors. Character scope is
    deliberately advisory: clauses are assigned when possible, continuations
    inherit the previous explicit owner, and still-unscoped text is preserved
    in Nltags instead of invalidating or discarding the response.
    """

    roster_tags: tuple[str, ...]
    copyright_tags: tuple[str, ...]
    characters: tuple[StructuredPromptCharacter, ...]
    scene: str
    nltags: str
    validation_errors: tuple[str, ...] = ()
    scope_warnings: tuple[str, ...] = ()
    unscoped_identity: tuple[str, ...] = ()
    unscoped_details: tuple[str, ...] = ()


def confirmed_semantic_character_tags(
    result: SemanticLookupResult,
) -> tuple[str, ...]:
    """Return locally confirmed visible characters in request order."""
    anchor_tag_map = dict(result.anchor_tags)
    return tuple(
        dict.fromkeys(
            anchor_tag_map.get(anchor.anchor_id, "")
            for anchor in result.anchors
            if anchor.role == "target_character"
            and anchor_tag_map.get(anchor.anchor_id, "")
        )
    )


def bind_confirmed_character_anchors(
    result: SemanticLookupResult, anchors: tuple[SemanticAnchor, ...]
) -> SemanticLookupResult:
    """Bind exact locally confirmed owner candidates even when lookup missed them."""
    tags = dict(result.anchor_tags)
    confirmed = set(result.confirmed_tags)
    for anchor in anchors:
        if anchor.role != "target_character" or anchor.anchor_id in tags:
            continue
        matches = set(anchor.candidates).intersection(confirmed)
        if len(matches) == 1:
            tags[anchor.anchor_id] = matches.pop()
    resolved_sources = {
        anchor.source_text for anchor in anchors
        if anchor.role == "target_character" and anchor.anchor_id in tags
    }
    return replace(
        result,
        anchors=anchors,
        anchor_tags=tuple(tags.items()),
        missing_descriptions=tuple(
            item for item in result.missing_descriptions if item not in resolved_sources
        ),
    )


def reconcile_confirmed_semantic_characters(
    characters: tuple[StructuredPromptCharacter, ...],
    nltags: str,
    confirmed_tags: tuple[str, ...],
) -> tuple[tuple[StructuredPromptCharacter, ...], str]:
    """Canonically rewrite only writer names uniquely grounded by LLM1 evidence.

    The prompt-writer already receives the locally confirmed character tags, but
    can still decorate a name (for example, ``young_example_character``).  Keep
    the writer's character-to-Identity/Details association and only replace a
    name when exactly one confirmed tag matches. Reserve exact matches before
    decorated names. A complete roster with one unmatched name and one missing
    identity has a unique remaining association; never assign multiple unknowns
    by position.
    """
    if not characters or not confirmed_tags:
        return characters, nltags

    def tokens(value: str) -> tuple[str, ...]:
        return tuple(re.findall(r"[a-z0-9]+", str(value or "").lower()))

    def is_whole_token_subsequence(
        writer_tokens: tuple[str, ...], canonical_tokens: tuple[str, ...]
    ) -> bool:
        if not canonical_tokens or len(canonical_tokens) > len(writer_tokens):
            return False
        width = len(canonical_tokens)
        return any(
            writer_tokens[index : index + width] == canonical_tokens
            for index in range(len(writer_tokens) - width + 1)
        )

    def matching_canonicals(writer_name: str, available: tuple[str, ...]) -> tuple[str, ...]:
        writer_key = _normalized_character_key(writer_name)
        writer_tokens = tokens(writer_name)
        exact = tuple(
            tag
            for tag in available
            if writer_key and writer_key == _normalized_character_key(tag)
        )
        if exact:
            return exact
        return tuple(
            tag
            for tag in available
            if is_whole_token_subsequence(writer_tokens, tokens(tag))
        )

    def replace_name(text: str) -> str:
        # One pass prevents a base name from rewriting an already replaced
        # qualified name (or a replacement from being substituted again).
        variants = {
            variant.lower(): canonical
            for writer_name, canonical in replacements
            for variant in (writer_name, writer_name.replace("_", " "))
        }
        pattern = "|".join(
            re.escape(value) for value in sorted(variants, key=len, reverse=True)
        )
        return re.sub(
            rf"(?<!\w)(?:{pattern})(?!\w)",
            lambda match: variants[match.group(0).lower()],
            str(text or ""), flags=re.I,
        )

    available = tuple(dict.fromkeys(tag for tag in confirmed_tags if tag))
    replacements: list[tuple[str, str]] = []
    assignments: dict[int, str] = {}
    # Reserve exact identities globally, independent of writer ordering.
    for index, character in enumerate(characters):
        matches = tuple(
            tag for tag in available
            if _normalized_character_key(tag) == _normalized_character_key(character.name)
        )
        if len(matches) == 1:
            assignments[index] = matches[0]
            available = tuple(tag for tag in available if tag != matches[0])
    for index, character in enumerate(characters):
        if index in assignments:
            continue
        # Preserve the established single-character contract: once LLM1 has
        # confirmed the only visible character, its canonical tag wins even if
        # the writer invented a wholly unrelated name.  Multiple characters
        # need the stricter unique-match rule below so no one is reassigned by
        # position or deletion.
        matches = (
            available
            if len(characters) == 1 and len(available) == 1
            else matching_canonicals(character.name, available)
        )
        if len(matches) != 1:
            continue
        canonical = matches[0]
        available = tuple(tag for tag in available if tag != canonical)
        assignments[index] = canonical

    unmatched = [index for index in range(len(characters)) if index not in assignments]
    if (
        len(characters) == len(set(confirmed_tags))
        and len(unmatched) == len(available) == 1
        and len({_normalized_character_key(item.name) for item in characters}) == len(characters)
    ):
        assignments[unmatched[0]] = available[0]
    replacements = [
        (character.name, assignments[index])
        for index, character in enumerate(characters) if index in assignments
    ]
    reconciled = tuple(
        replace(character, name=assignments.get(index, character.name))
        for index, character in enumerate(characters)
    )

    if not replacements:
        return characters, nltags
    return (
        tuple(
            StructuredPromptCharacter(
                name=character.name,
                identity_tags=replace_name(character.identity_tags),
                detail_tags=replace_name(character.detail_tags),
            )
            for character in reconciled
        ),
        replace_name(nltags),
    )


def validate_confirmed_character_roster(
    parsed: StructuredPromptParseResult, confirmed_tags: tuple[str, ...]
) -> StructuredPromptParseResult:
    """Repair unique identities, then reject missing confirmed visible owners."""
    characters, nltags = reconcile_confirmed_semantic_characters(
        parsed.characters, parsed.nltags, confirmed_tags
    )
    missing = set(confirmed_tags).difference(item.name for item in characters)
    errors = list(parsed.validation_errors)
    if missing:
        errors.append(
            "Characters is missing confirmed visible identities: " + ", ".join(sorted(missing))
        )
    return replace(
        parsed, characters=characters, nltags=nltags, validation_errors=tuple(errors)
    )


def _normalized_character_key(value: str) -> str:
    """Normalize a display name or Danbooru tag for local-hint alignment."""
    normalized = str(value or "").strip().lower().replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", normalized)


def _configured_character_anchor(tags: str) -> str:
    """Return the likely character-name prefix from a chat-authored entry."""
    return next(iter(split_tags(tags)), "").strip()


def _match_fixed_character_hint(
    character_name: str,
    hints: dict[str, str],
    used_names: set[str],
    *,
    structured_character_count: int,
) -> str:
    """Match an LLM roster name to a locally mentioned character hint."""
    target = _normalized_character_key(character_name)
    if target:
        for local_name, tags in hints.items():
            if local_name in used_names:
                continue
            variants = {
                _normalized_character_key(local_name),
                _normalized_character_key(_configured_character_anchor(tags)),
            }
            if target in variants:
                return local_name
    remaining = [name for name in hints if name not in used_names]
    if structured_character_count == 1 and len(remaining) == 1:
        return remaining[0]
    return ""


_UNBOUND_DIRECTIONAL_TAGS = {
    "embrace",
    "embracing",
    "cuddle",
    "cuddling",
    "hug",
    "hugging",
    "lying",
    "sitting",
}


def filter_unbound_directional_tags(tag_text: str) -> tuple[str, tuple[str, ...]]:
    """Remove bare shared tags that cannot encode interaction ownership."""
    kept: list[str] = []
    removed: list[str] = []
    for tag in split_tags(tag_text):
        if tag.strip().lower() in _UNBOUND_DIRECTIONAL_TAGS:
            removed.append(tag)
        else:
            kept.append(tag)
    return ", ".join(kept), tuple(removed)


def is_chinese_model_refusal(text: str) -> bool:
    """Return whether an expected prompt response is a Chinese/English refusal.

    The historical public name is retained for compatibility with tests and
    external imports, although providers may answer the Chinese task in English.
    """
    response = re.sub(r"\s+", " ", str(text or "")).strip()
    refusal_patterns = (
        r"(?:抱歉|对不起|很遗憾).{0,48}(?:不能|无法|不可以|没法|不便)",
        r"(?:不能|无法|不可以|没法|不便).{0,36}"
        r"(?:满足|帮助|协助|生成|创作|提供|完成|处理|遵循|支持).{0,20}"
        r"(?:要求|请求|内容|指令)?",
        r"(?:拒绝|不能接受|无法接受).{0,24}(?:要求|请求|生成|创作|内容)",
    )
    if len(re.findall(r"[\u3400-\u9fff]", response)) >= 4 and any(
        re.search(pattern, response) for pattern in refusal_patterns
    ):
        return True
    english = response.replace("’", "'")
    english_patterns = (
        r"\b(?:i(?:'m| am)?\s+sorry\b.{0,80})?"
        r"(?:i\s+)?(?:cannot|can't|won't|am unable to)\s+"
        r"(?:help|assist|comply|fulfill|generate|create|provide|complete|process|follow)\b",
        r"\b(?:i\s+)?(?:must|have to)\s+(?:refuse|decline)\b",
        r"\b(?:i\s+)?(?:refuse|decline)\s+(?:this|the|your)\s+request\b",
    )
    return any(re.search(pattern, english, flags=re.I) for pattern in english_patterns)


_FUTA_COUNT_KEYS = frozenset(
    {"futa", "futanari", "1futa", "1futanari", "1 futa", "1 futanari"}
)
_FUTA_WITH_FEMALE_KEYS = frozenset(
    {"futa with female", "female with futa", "female with futanari"}
)
_FUTA_WITH_MALE_KEYS = frozenset(
    {"futa with male", "male with futa", "male with futanari"}
)
_GIRL_COUNT_RE = re.compile(r"(\d+)\s*girls?")
_BOY_COUNT_RE = re.compile(r"(\d+)\s*boys?")


def _count_tag(count: int, noun: str) -> str:
    """Render a Danbooru people-count tag with the correct plural form.

    ``1girl``/``1boy`` stay singular while every other count gets ``s``
    appended (``2girls``, ``3boys``).  The futa paths must never emit an
    impossible form such as ``1boys``.
    """
    return f"{count}{noun}{'' if count == 1 else 's'}"


def normalize_anima_count_tags(
    tags: tuple[str, ...] | list[str], character_count: int
) -> tuple[str, ...]:
    """Align futa count tags with Anima's people-count semantics.

    Anima counts a futanari inside the girls count: one female plus one futa
    is `2girls, futa with female`, two females plus one futa is
    `3girls, futa with female`, and a lone futa is `1girl, futanari`. A futa
    plus a male uses `futa with male`; rosters larger than that pair also keep
    an explicit `Npeople` anchor. The
    previous normalization collapsed `2girls, futanari` into a bare
    `futa with female`, which dropped the people-count anchor Anima needs.

    Args:
        tags: Raw tags from the Count block.
        character_count: Number of characters in the Characters roster.

    Returns:
        Tags with a canonical people-count tag and a normalized futa tag.
    """
    values = tuple(str(tag).strip() for tag in tags if str(tag).strip())
    if not values:
        return values
    normalized = tuple(tag.lower().replace("_", " ") for tag in values)
    has_futa = any(tag in _FUTA_COUNT_KEYS for tag in normalized)
    has_futa_with_female = any(
        tag in _FUTA_WITH_FEMALE_KEYS for tag in normalized
    )
    has_futa_with_male = any(tag in _FUTA_WITH_MALE_KEYS for tag in normalized)
    if not (has_futa or has_futa_with_female or has_futa_with_male):
        return values
    girl_count = max(
        (
            int(match.group(1))
            for tag in normalized
            if (match := _GIRL_COUNT_RE.fullmatch(tag))
        ),
        default=0,
    )
    boy_count = max(
        (
            int(match.group(1))
            for tag in normalized
            if (match := _BOY_COUNT_RE.fullmatch(tag))
        ),
        default=0,
    )
    kept = tuple(
        value
        for value, tag in zip(values, normalized, strict=True)
        if (
            tag not in _FUTA_COUNT_KEYS
            and tag not in _FUTA_WITH_FEMALE_KEYS
            and tag not in _FUTA_WITH_MALE_KEYS
            and not _GIRL_COUNT_RE.fullmatch(tag)
            and not _BOY_COUNT_RE.fullmatch(tag)
        )
    )
    if has_futa_with_male:
        count_anchor = f"{character_count}people" if character_count > 2 else ""
        return tuple(
            dict.fromkeys(
                tag for tag in (count_anchor, "futa with male", *kept) if tag
            )
        )
    if character_count == 1:
        return tuple(dict.fromkeys(("1girl", "futanari", *kept)))
    if boy_count and not girl_count:
        # A bare futa plus boys: the futa is not counted among boys, so the
        # pair relation is sufficient for two people. Larger rosters still need
        # an explicit total-count anchor.
        count_anchor = f"{character_count}people" if character_count > 2 else ""
        return tuple(
            dict.fromkeys(
                tag for tag in (count_anchor, "futa with male", *kept) if tag
            )
        )
    # Anima's `Ngirls` count already includes the futa.  When the roster is
    # authoritative (`character_count` > 0) the girls total is derived from it
    # instead of blindly adding one, so `2girls, futanari` stays `2girls` and
    # a mixed `2girls, 1boy, futanari` roster (1 female + 1 futa + 1 boy)
    # becomes `2girls, 1boy, futa with female` rather than a fake `3girls`.
    if boy_count:
        total_girls = (
            character_count - boy_count if character_count else girl_count + 1
        )
        return tuple(
            dict.fromkeys(
                (
                    _count_tag(total_girls, "girl") if total_girls else "",
                    _count_tag(boy_count, "boy"),
                    "futa with female",
                    *kept,
                )
            )
        )
    total_girls = character_count if character_count else girl_count + 1
    return tuple(
        dict.fromkeys(
            (_count_tag(total_girls, "girl"), "futa with female", *kept)
        )
    )


def structured_count_tags_match_roster(
    tags: tuple[str, ...], character_count: int
) -> bool:
    """Return whether structured Count tags encode the whole visible roster.

    Mixed-gender counts are additive: ``1girl, 1boy`` is the correct count for
    two roster entries even though neither individual tag equals two.
    """
    numeric_counts = tuple(
        (int(match.group(1)), match.group(2).lower())
        for item in tags
        if (
            match := re.fullmatch(
                r"\s*(\d+)\s*(girls?|boys?|people|persons?)\s*",
                item,
                flags=re.IGNORECASE,
            )
        )
    )
    gender_counts = tuple(
        (count, noun)
        for count, noun in numeric_counts
        if noun.startswith(("girl", "boy"))
    )
    people_counts = tuple(
        (count, noun)
        for count, noun in numeric_counts
        if noun.startswith(("people", "person"))
    )
    if gender_counts:
        return (
            not people_counts
            and len({noun.rstrip("s") for _count, noun in gender_counts})
            == len(gender_counts)
            and sum(count for count, _noun in gender_counts) == character_count
        )
    if people_counts:
        return len(people_counts) == 1 and people_counts[0][0] == character_count
    return (
        character_count == 2
        and any(
            item.lower().replace("_", " ") == "futa with male"
            for item in tags
        )
    )


_MALE_GENITAL_REQUEST_RE = re.compile(
    r"阴茎|肉棒|鸡巴|(?<![a-z])(?:penis|cock|dick|phallus|erection)(?![a-z])",
    re.I,
)
_SEXUAL_TRAIT_NEGATION_BEFORE_RE = re.compile(
    r"(?:不是(?:一个)?|并非|不要(?:出现|包含|带有)?|禁止(?:出现|包含)?|不许|不能|别|没有|无|"
    r"not(?:\s+(?:a|an))?|no|without|never|exclude|remove)\s*$",
    re.I,
)
_FUTA_OUTPUT_RE = re.compile(
    r"(?<![a-z])(?:female|male)\s+with\s+futa(?:nari)?(?![a-z])|"
    r"(?<![a-z])futa(?:nari)?\s+with\s+(?:female|male)(?![a-z])|"
    r"(?<![a-z])(?:1\s*)?futa(?:nari)?(?![a-z])",
    re.I,
)
_MALE_GENITAL_OUTPUT_RE = re.compile(
    r"(?<![a-z])(?:penis|cock|dick|phallus|erection|erect|precum|scrotum|testicles?)"
    r"(?![a-z])",
    re.I,
)


def has_positive_male_genital_request(text: str) -> bool:
    """Return whether the user positively requested visible male genital anatomy."""
    value = str(text or "")
    for match in _MALE_GENITAL_REQUEST_RE.finditer(value):
        prefix = value[max(0, match.start() - 16) : match.start()]
        if not _SEXUAL_TRAIT_NEGATION_BEFORE_RE.search(prefix):
            return True
    return has_positive_futa_request(value)


def _strip_unrequested_sexual_trait_text(
    text: str, *, allow_futa: bool, allow_male_genitals: bool
) -> tuple[str, tuple[str, ...]]:
    """Remove unsupported writer inventions while preserving surrounding content."""
    removed: list[str] = []
    kept: list[str] = []
    for fragment in re.split(r"\s*[,;]\s*", str(text or "")):
        fragment = fragment.strip()
        if not fragment:
            continue
        searchable = fragment.replace("_", " ")
        if not allow_male_genitals and _MALE_GENITAL_OUTPUT_RE.search(searchable):
            removed.append(fragment)
            continue
        cleaned = fragment
        if not allow_futa and _FUTA_OUTPUT_RE.search(searchable):
            cleaned = _FUTA_OUTPUT_RE.sub("", searchable)
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
            removed.append(fragment)
        if cleaned:
            kept.append(cleaned)
    return ", ".join(kept), tuple(removed)


def enforce_structured_sexual_trait_authority(
    parsed: StructuredPromptParseResult, user_prompt: str
) -> tuple[StructuredPromptParseResult, tuple[str, ...]]:
    """Require positive user evidence for futa and male-genital LLM2 output."""
    allow_futa = has_positive_futa_request(user_prompt)
    allow_male_genitals = has_positive_male_genital_request(user_prompt)
    if allow_futa and allow_male_genitals:
        return parsed, ()

    removed: list[str] = []
    roster_tags: list[str] = []
    for tag in parsed.roster_tags:
        searchable = tag.lower().replace("_", " ")
        if not allow_futa and (
            searchable in _FUTA_COUNT_KEYS
            or searchable in _FUTA_WITH_FEMALE_KEYS
            or searchable in _FUTA_WITH_MALE_KEYS
        ):
            removed.append(tag)
            continue
        roster_tags.append(tag)
    if not roster_tags and parsed.characters:
        roster_tags.append(f"{len(parsed.characters)}people")

    characters: list[StructuredPromptCharacter] = []
    for character in parsed.characters:
        identity, identity_removed = _strip_unrequested_sexual_trait_text(
            character.identity_tags,
            allow_futa=allow_futa,
            allow_male_genitals=allow_male_genitals,
        )
        detail, detail_removed = _strip_unrequested_sexual_trait_text(
            character.detail_tags,
            allow_futa=allow_futa,
            allow_male_genitals=allow_male_genitals,
        )
        removed.extend((*identity_removed, *detail_removed))
        characters.append(replace(character, identity_tags=identity, detail_tags=detail))

    scene, scene_removed = _strip_unrequested_sexual_trait_text(
        parsed.scene,
        allow_futa=allow_futa,
        allow_male_genitals=allow_male_genitals,
    )
    nltags, nltags_removed = _strip_unrequested_sexual_trait_text(
        parsed.nltags,
        allow_futa=allow_futa,
        allow_male_genitals=allow_male_genitals,
    )
    removed.extend((*scene_removed, *nltags_removed))
    return (
        replace(
            parsed,
            roster_tags=tuple(roster_tags),
            characters=tuple(characters),
            scene=scene,
            nltags=nltags,
        ),
        tuple(dict.fromkeys(removed)),
    )


def _parse_structured_prompt(text: str) -> StructuredPromptParseResult:
    """Parse seven fields without treating natural-language scope as syntax."""
    raw = str(text or "").replace("\r\n", "\n").strip()
    field_names = {
        "count",
        "characters",
        "copyright",
        "identity",
        "details",
        "tags",
        "nltags",
    }
    matches = re.findall(
        r"\{\s*(Count|Characters|Copyright|Identity|Details|Tags|Nltags)\s*:\s*(.*?)\s*\}",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    values_by_field: dict[str, list[str]] = {}
    for key, value in matches:
        values_by_field.setdefault(key.lower(), []).append(value.strip())
    missing = sorted(field_names - set(values_by_field))
    # Some providers repeat a completed block verbatim.  Reusing one identical
    # value is unambiguous; competing values still require a format retry.
    duplicates = sorted(
        key for key, values in values_by_field.items()
        if len(set(values)) > 1
    )
    validation_errors: list[str] = []
    if missing:
        validation_errors.append("missing fields: " + ", ".join(missing))
    if duplicates:
        validation_errors.append("duplicate fields: " + ", ".join(duplicates))
    if validation_errors:
        return StructuredPromptParseResult(
            (), (), (), raw, "", tuple(validation_errors)
        )
    blocks = {key: values[0] for key, values in values_by_field.items()}
    roster = [
        item.strip()
        for item in blocks["characters"].split(",")
        if item.strip()
    ]
    raw_roster_tags = tuple(
        item.strip() for item in blocks["count"].split(",") if item.strip()
    )
    # DeepSeek occasionally returns a bare roster cardinality (``2``) even
    # though every other field is valid. The Characters block already gives
    # the authoritative total, so normalize the harmless shorthand locally.
    raw_roster_tags = tuple(
        f"{len(roster)}people"
        if re.fullmatch(r"\d+", item) and int(item) == len(roster)
        else item
        for item in raw_roster_tags
    )
    roster_tags = list(normalize_anima_count_tags(raw_roster_tags, len(roster)))
    has_count_tag = False
    for item in roster_tags:
        # Count is a structured hard-tag block.  Once it contains a valid
        # people-count anchor, preserve every tag in the block; the content
        # cleaner must not silently discard relationship/focus/count metadata.
        normalized = item.lower().replace("_", " ")
        if re.fullmatch(
            r"(?:[1-9]\d*\+? ?(?:girls?|boys?|people|persons?|others?)|"
            r"multiple (?:girls|boys|people)|no humans|futa with (?:female|male)|"
            r"futanari|1futanari)",
            normalized,
        ):
            has_count_tag = True
    copyright_tags = tuple(
        item.strip()
        for item in blocks["copyright"].split(",")
        if item.strip()
    )
    if not has_count_tag:
        validation_errors.append(
            "Count must include a valid people-count tag such as 1girl, "
            "2girls, 1boy, Npeople, or no humans"
        )
    if validation_errors:
        return StructuredPromptParseResult(
            (), (), (), raw, "", tuple(validation_errors)
        )

    def name_pattern(name: str) -> str:
        variants = tuple(dict.fromkeys((name, name.replace("_", " "))))
        return "(?:" + "|".join(re.escape(item) for item in variants) + ")"

    def scoped_section(
        section_name: str, section: str
    ) -> tuple[dict[str, list[str]], tuple[str, ...], tuple[str, ...]]:
        # A pipe is an established alternate role separator. Semicolons split
        # clauses, but do not imply that an unlabelled continuation has lost
        # its owner: it inherits the preceding explicit/unique role.
        clauses = [
            clause.strip()
            for clause in re.split(r"\s*(?:;|\|)\s*", section)
            if clause.strip()
        ]
        assigned = {name: [] for name in roster}
        unscoped: list[str] = []
        current_owner = ""
        for clause in clauses:
            starts = [
                name
                for name in roster
                if re.match(
                    rf"^\s*{name_pattern(name)}(?!\w)",
                    clause,
                    flags=re.IGNORECASE,
                )
            ]
            mentions = [
                name
                for name in roster
                if re.search(
                    rf"(?<!\w){name_pattern(name)}(?!\w)",
                    clause,
                    flags=re.IGNORECASE,
                )
            ]
            owner = starts[0] if len(starts) == 1 else ""
            if not owner and len(mentions) == 1:
                # Handles framing such as ``photo one shows hatsune_miku``
                # without forcing a rewrite or losing the surrounding words.
                owner = mentions[0]
            if not owner and current_owner:
                # Pronouns, photo-by-photo narration, and other continuation
                # clauses keep the last unambiguous role scope.
                owner = current_owner
            if owner:
                assigned[owner].append(clause)
                current_owner = owner
            else:
                unscoped.append(clause)

        warnings = [
            f"{section_name} has no attributable clause for {name}"
            for name, clauses_for_name in assigned.items()
            if not clauses_for_name
        ]
        if unscoped:
            warnings.append(
                f"{section_name} preserved {len(unscoped)} unscoped clause(s) in Nltags"
            )
        return assigned, tuple(unscoped), tuple(warnings)

    identity_by_name, unscoped_identity, identity_warnings = scoped_section(
        "Identity", blocks["identity"]
    )
    details_by_name, unscoped_details, details_warnings = scoped_section(
        "Details", blocks["details"]
    )

    characters = tuple(
        StructuredPromptCharacter(
            name=name,
            identity_tags="; ".join(identity_by_name[name]),
            detail_tags="; ".join(details_by_name[name]),
        )
        for name in roster
    )
    preserved_nltags = "; ".join(
        part
        for part in (
            blocks["nltags"],
            *unscoped_identity,
            *unscoped_details,
        )
        if part
    )
    return StructuredPromptParseResult(
        tuple(roster_tags),
        copyright_tags,
        characters,
        blocks["tags"],
        preserved_nltags,
        (),
        (*identity_warnings, *details_warnings),
        unscoped_identity,
        unscoped_details,
    )


def extract_structured_prompt(
    text: str,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[StructuredPromptCharacter, ...],
    str,
    str,
]:
    """Return the established five-part public structured-prompt tuple."""
    parsed = _parse_structured_prompt(text)
    return (
        parsed.roster_tags,
        parsed.copyright_tags,
        parsed.characters,
        parsed.scene,
        parsed.nltags,
    )


def extract_nltags(text: str) -> tuple[str, str]:
    """Separate an optional natural-language block from LLM-generated tags.

    Args:
        text: Raw LLM completion containing Danbooru tags and optional Nltags.

    Returns:
        A pair containing tag text and a natural-language description. Empty or
        invalid Nltags declarations preserve the existing tag-only behavior.
    """
    raw = str(text or "").strip()
    match = re.search(r"(?:^|[\n,])\s*nltags\s*:\s*", raw, flags=re.I)
    if not match:
        return raw, ""
    tags = raw[: match.start()].strip(" \t\r\n,;")
    nltags = " ".join(raw[match.end() :].split()).strip(" ,;:")
    return tags, nltags


def normalize_structured_nltags(
    nltags: str, character_names: tuple[str, ...]
) -> str:
    """Align structured Nltags names with the ComfyUI-facing character names.

    Args:
        nltags: Natural-language text produced by the LLM.
        character_names: Character roster labels before display normalization.

    Returns:
        Nltags containing every roster name with underscores rendered as spaces.
    """
    normalized_names = tuple(
        re.sub(r"\s+", " ", name.replace("_", " ")).strip()
        for name in character_names
        if name.strip()
    )
    result = str(nltags or "").strip()
    for source, display in zip(character_names, normalized_names, strict=True):
        result = re.sub(
            rf"(?<!\w){re.escape(source)}(?!\w)",
            display,
            result,
            flags=re.IGNORECASE,
        )
    missing = [
        name
        for name in normalized_names
        if not re.search(rf"(?<!\w){re.escape(name)}(?!\w)", result, re.I)
    ]
    if missing:
        prefix = ", ".join(missing)
        result = f"{prefix}. {result}".strip() if result else prefix
    return result


_OUTFIT_NARRATIVE_RE = re.compile(
    r"\b(?:wears?|wearing|costume|outfit|clothing|dress|gown|skirt|shirt|"
    r"sleeves?|lace|frills?|ruffles?|ornament|mask|ribbon|bow|stockings?|"
    r"thighhighs?|pantyhose|boots?|shoes?|uniform|school|academy|"
    r"bottomless|topless|nude|naked|gothic lolita)\b",
    re.I,
)

_OUTFIT_ASSERTION_RE = re.compile(
    r"\b(?:wears?|wearing|dressed|clad)\b|"
    r"\b(?:is|are|appears?|becomes?)\s+(?:fully\s+|completely\s+)?"
    r"(?:bottomless|topless|nude|naked)\b|"
    r"\b(?:costume|outfit|clothing|uniform|dress|gown|skirt|shirt|"
    r"stockings?|thighhighs?|pantyhose|boots?|shoes?)\s+"
    r"(?:is|are|has|have|looks?|appears?)\b",
    re.I,
)


def minimal_verified_outfit_nltags(
    nltags: str,
    source_tags: tuple[str, ...],
    *,
    include_source_cue: bool = True,
) -> str:
    """Keep a verified costume cue without retaining guessed outfit prose."""
    source = str(source_tags[0] if source_tags else "costume").split("_(", 1)[0]
    source_label = source.replace("_", " ").strip().title() or "Costume"
    parts = [f"Costume based on the character {source_label}."] if include_source_cue else []
    for sentence in re.split(r"(?<=[.!?])\s+", " ".join(str(nltags or "").split())):
        sentence = sentence.strip()
        if not sentence or _OUTFIT_NARRATIVE_RE.search(sentence):
            continue
        if sentence not in parts:
            parts.append(sentence)
    return " ".join(parts)


def strip_outfit_narrative(nltags: str) -> str:
    """Drop actual wardrobe assertions, not every sentence naming a garment.

    Garment words can locate an unrelated requested fact (for example an erection
    beneath a skirt).  Treating those nouns as proof that the whole sentence is
    wardrobe prose silently discarded user semantics.
    """
    return " ".join(
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(str(nltags or "").split()))
        if sentence.strip() and not _OUTFIT_ASSERTION_RE.search(sentence)
    )


def scoped_outfit_narrative(detail: str) -> str:
    """Keep wearer-bound wardrobe clauses from one post-processed Details entry.

    The prompt writer's shared Nltags wardrobe prose is untrusted because it can
    assign one character's clothes to another. Details has already passed the
    per-character wardrobe authority rewrite, so wardrobe clauses extracted here
    are the safe source for reinforcing the final character-to-outfit binding.
    """
    clauses = []
    for clause in re.split(r"\s*;\s*", " ".join(str(detail or "").split())):
        clause = clause.strip(" ,.;")
        if clause and _OUTFIT_NARRATIVE_RE.search(clause):
            clauses.append(clause + ".")
    return " ".join(dict.fromkeys(clauses))


@dataclass(frozen=True)
class CharacterEffectiveOutfit:
    """Resolved request-scoped wardrobe data owned by exactly one character.

    The semantic anchor IDs preserve ownership through lookup, user overrides,
    prompt cleanup, and final hard-tag injection. This avoids a global outfit list,
    which cannot represent who wears what in multi-character requests.
    """

    target_anchor_id: str
    target_source_text: str
    target_candidates: tuple[str, ...]
    wardrobe_kind: str
    wardrobe_anchor_id: str
    wardrobe_tag: str
    appearance_tags: tuple[str, ...]
    effective: EffectiveOutfitPlan
    complete_named_profile: bool = False
    composition_omitted_tags: tuple[str, ...] = ()
    resolution_state: str = "resolved"
    unresolved_evidence: tuple[str, ...] = ()


_STABLE_APPEARANCE_REQUEST_RE = re.compile(
    r"(?:头发|发色|发型|长发|短发|眼睛|瞳色|肤色|皮肤颜色|兽耳|耳朵|尾巴|角|"
    r"\bhair\b|\bhair\s*colou?r\b|\bhairstyle\b|\beyes?\b|"
    r"\beye\s*colou?r\b|\bskin\s*colou?r\b|\bears?\b|\btails?\b|\bhorns?\b)",
    re.I,
)
_EXPLICIT_IDENTITY_CHANGE_RE = re.compile(
    r"(?:(?:改|变|染|换|剪|修改|设为|成为|变成|长出|戴上).{0,18}"
    r"(?:头发|发色|发型|眼睛|瞳色|肤色|耳朵|尾巴|角)|"
    r"(?:头发|发色|发型|眼睛|瞳色|肤色|耳朵|尾巴|角).{0,18}"
    r"(?:改|变|染|换|剪|修改|设为|成为|变成|长出|戴上|红色|蓝色|金色|"
    r"银色|白色|黑色|绿色|紫色|粉色|棕色|灰色|长发|短发)|"
    r"(?:红|蓝|金|银|白|黑|绿|紫|粉|棕|灰)(?:色)?(?:长|短)?发|"
    r"(?:红|蓝|金|银|白|黑|绿|紫|粉|棕|灰)(?:色)?(?:眼睛|瞳)|"
    r"\b(?:change|alter|replace|dye|cut|grow|give|make|turn)\b.{0,28}"
    r"\b(?:hair|eyes?|skin|ears?|tails?|horns?)\b|"
    r"\b(?:long|short|red|blue|blonde|green|yellow|gold(?:en)?|silver|white|"
    r"black|purple|pink|brown|grey|gray)\b.{0,10}"
    r"\b(?:hair|eyes?|skin|ears?|tails?|horns?)\b)",
    re.I,
)
def explicit_identity_override_requested(
    user_prompt: str,
    target_names: tuple[str, ...],
    *,
    character_count: int,
) -> bool:
    """Return whether the user explicitly changes this character's appearance.

    Saved appearance is authoritative by default.  A single-character request can
    state an appearance change with a pronoun, while a multi-character request must
    identify the affected character in the same sentence to avoid leaking one
    person's override to everyone else.
    """
    text = str(user_prompt or "")
    if not _EXPLICIT_IDENTITY_CHANGE_RE.search(text):
        return False
    if character_count <= 1:
        return True

    def normalized(value: str) -> str:
        return re.sub(r"[^\w]+", " ", value.lower().replace("_", " ")).strip()

    aliases = tuple(
        key for value in target_names if (key := normalized(str(value or "")))
    )
    for sentence in re.split(r"[。！？!?;；\n]+", text):
        if not _EXPLICIT_IDENTITY_CHANGE_RE.search(sentence):
            continue
        normalized_sentence = normalized(sentence)
        if any(alias in normalized_sentence for alias in aliases):
            return True
    return False


def authoritative_profile_identity_block(
    character_name: str, appearance_tags: tuple[str, ...]
) -> str:
    """Render a saved visual profile as the complete structured Identity clause."""
    tags = tuple(dict.fromkeys(tag.strip() for tag in appearance_tags if tag.strip()))
    return f"{character_name} has {', '.join(tags)}" if tags else ""


_APPEARANCE_DIMENSION_PATTERNS: dict[str, re.Pattern[str]] = {
    "eye_color": re.compile(
        r"(?:red|blue|green|yellow|gold(?:en)?|silver|white|black|purple|pink|"
        r"brown|grey|gray|amber|aqua|cyan|orange|violet|hazel)"
        r"(?:[ _-]+eyes?|[ _-]+eyed)\b|"
        r"(?:红|蓝|绿|黄|金|银|白|黑|紫|粉|棕|褐|灰|琥珀|青|橙)(?:色)?(?:眼睛|眼|瞳)|"
        r"(?:眼睛|瞳色|瞳).{0,8}(?:红|蓝|绿|黄|金|银|白|黑|紫|粉|棕|褐|灰|琥珀|青|橙)(?:色)?",
        re.I,
    ),
    "hair_color": re.compile(
        r"(?:red|blue|green|yellow|gold(?:en)?|blonde|silver|white|black|purple|"
        r"pink|brown|grey|gray|aqua|cyan|orange|violet)[ _-]+hair\b|"
        r"(?:红|蓝|绿|黄|金|银|白|黑|紫|粉|棕|褐|灰|青|橙)(?:色)?.{0,8}(?:头发|发色|发型)|"
        r"(?:头发|发色|发型).{0,8}(?:红|蓝|绿|黄|金|银|白|黑|紫|粉|棕|褐|灰|青|橙)(?:色)?",
        re.I,
    ),
    "hair_length": re.compile(
        r"\b(?:very[ _-]+long|long|medium|short)[ _-]+hair\b|"
        r"(?:超长|及腰|长|中长|短)(?:发|头发)",
        re.I,
    ),
    "hair_style": re.compile(
        r"\b(?:twintails?|twin[ _-]+drills?|drill[ _-]+hair|ponytails?|"
        r"braids?|braided[ _-]+hair|straight[ _-]+hair|wavy[ _-]+hair|"
        r"curly[ _-]+hair|bob[ _-]+cut|hair[ _-]+bun|double[ _-]+bun)\b|"
        r"(?:双马尾|双钻头|钻头发|马尾|辫子|麻花辫|直发|卷发|波浪发|丸子头|发型)",
        re.I,
    ),
    "skin_color": re.compile(
        r"\b(?:pale|fair|tan(?:ned)?|dark|brown|black|white)[ _-]+skin\b|"
        r"(?:苍白|白皙|小麦|晒黑|深色|棕色|褐色|黑色)(?:皮肤|肤色)",
        re.I,
    ),
    "chest_size": re.compile(
        r"\b(?:flat[ _-]+chest|small|medium|large|huge)[ _-]+breasts?\b|"
        r"(?:平胸|贫乳|小胸|巨乳|大胸|胸部.{0,5}(?:变大|变小|丰满))",
        re.I,
    ),
    "animal_ears": re.compile(r"\b(?:animal|cat|dog|fox|rabbit)[ _-]+ears?\b|(?:兽耳|猫耳|狗耳|狐耳|兔耳)", re.I),
    "tail": re.compile(r"\b(?:animal|cat|dog|fox|rabbit)?[ _-]*tails?\b|尾巴", re.I),
    "horns": re.compile(r"\bhorns?\b|(?:犄角|兽角|头角|长角|角(?!色))", re.I),
}


def appearance_dimensions(value: str) -> frozenset[str]:
    """Classify every visual dimension expressed by one tag/prose fragment."""
    normalized = str(value or "").strip().lower().replace("_", " ")
    dimensions = {
        dimension
        for dimension, pattern in _APPEARANCE_DIMENSION_PATTERNS.items()
        if pattern.search(normalized)
    }
    # Natural writer prose often inserts a colour between the length and noun:
    # ``long green hair``.  The atomic profile tag is still ``long_hair``.
    if re.search(r"\bhair\b", normalized) and re.search(
        r"\b(?:very\s+long|long|medium|short)\b", normalized
    ):
        dimensions.add("hair_length")
    return frozenset(dimensions)


def appearance_dimension(value: str) -> str:
    """Return the first known dimension for compatibility with atomic-tag callers."""
    dimensions = appearance_dimensions(value)
    return next(
        (
            dimension
            for dimension in _APPEARANCE_DIMENSION_PATTERNS
            if dimension in dimensions
        ),
        "",
    )


_STANDALONE_APPEARANCE_TAG_PATTERNS: dict[str, re.Pattern[str]] = {
    "eye_color": re.compile(
        r"(?:red|blue|green|yellow|gold(?:en)?|silver|white|black|purple|pink|"
        r"brown|grey|gray|amber|aqua|cyan|orange|violet) eyes?",
        re.I,
    ),
    "hair_color": re.compile(
        r"(?:red|blue|green|yellow|gold(?:en)?|blonde|silver|white|black|purple|"
        r"pink|brown|grey|gray|aqua|cyan|orange|violet) hair",
        re.I,
    ),
    "hair_length": re.compile(r"(?:very long|long|medium|short) hair", re.I),
    "hair_style": re.compile(
        r"(?:twintails?|twin drills?|drill hair|ponytails?|braids?|braided hair|"
        r"straight hair|wavy hair|curly hair|bob cut|hair bun|double bun)",
        re.I,
    ),
    "skin_color": re.compile(
        r"(?:pale|fair|tan(?:ned)?|dark|brown|black|white) skin", re.I
    ),
    "chest_size": re.compile(
        r"(?:flat chest|(?:small|medium|large|huge) breasts?)", re.I
    ),
    "animal_ears": re.compile(r"(?:animal|cat|dog|fox|rabbit) ears?", re.I),
    "tail": re.compile(r"(?:(?:animal|cat|dog|fox|rabbit) )?tails?", re.I),
    "horns": re.compile(r"horns?", re.I),
}


def standalone_appearance_dimension(value: str) -> str:
    """Classify only a complete subject-appearance tag.

    Compound props and accessories such as ``red_hair_ornament``,
    ``blue_eyes_symbol`` and ``comet_tail`` are not character appearance facts
    and must survive shared-tag cleanup.
    """
    normalized = re.sub(
        r"\s+", " ", str(value or "").strip().lower().replace("_", " ")
    )
    for dimension, pattern in _STANDALONE_APPEARANCE_TAG_PATTERNS.items():
        if pattern.fullmatch(normalized):
            return dimension
    return ""


def appearance_override_dimensions(
    user_prompt: str,
    target_names: tuple[str, ...],
    *,
    character_count: int,
) -> frozenset[str]:
    """Disabled compatibility hook for the former raw-prompt regex parser.

    User prose must not be interpreted by host regexes: words such as ``视角``
    are not evidence that a character has horns.  LLM1's source-grounded
    ``appearance_changes`` is the only optional identity-change hint for LLM2.
    """
    del user_prompt, target_names, character_count
    return frozenset()


def merge_authoritative_identity_block(
    character_name: str,
    writer_identity: str,
    appearance_tags: tuple[str, ...],
    override_dimensions: frozenset[str],
    *,
    advisory_writer_dimensions: frozenset[str] = frozenset(),
    replacement_writer_dimensions: frozenset[str] = frozenset(),
) -> str:
    """Merge writer prose with only the missing stable profile dimensions.

    Writer facts that agree with a locked profile dimension remain in their
    natural prose form instead of being discarded and blindly re-appended as
    underscore tags. Dimensions explicitly classified by LLM1 as replacements
    suppress their old profile values. Additive and legacy-unspecified changes
    conservatively keep the saved value alongside newly written facts.
    """
    name_pattern = re.escape(character_name).replace("_", r"(?:_|\s)")
    prefix = re.compile(
        rf"^\s*{name_pattern}\s+(?=(?:has|is|are|with)\s+)",
        re.I,
    )
    body = prefix.sub("", str(writer_identity or "").strip()).strip(" ,.;")
    # Keep the copula, but separate its accompanying attributes so they still
    # pass through the same profile authority as a leading ``has`` sentence.
    body = re.sub(
        r"\b((?:is|are)\s+[^,;]+?)\s+with\s+",
        r"\1 and has ",
        body,
        flags=re.I,
    )
    fragments = [
        re.sub(r"^and\s+", "", item.strip(" ,.;"), flags=re.I)
        for item in re.split(r"\s*,\s*|\s+and\s+", body, flags=re.I)
        if item.strip(" ,.;")
    ]
    standalone_color = re.compile(
        r"^(?:red|blue|green|yellow|gold(?:en)?|blonde|silver|white|black|"
        r"purple|pink|brown|grey|gray|amber|aqua|cyan|orange|violet)$",
        re.I,
    )
    kept: list[str] = []
    predicate_clauses: list[str] = []
    stable_by_dimension: dict[str, tuple[str, ...]] = {
        dimension: tuple(
            tag.strip()
            for tag in appearance_tags
            if tag.strip() and appearance_dimension(tag) == dimension
        )
        for dimension in _APPEARANCE_DIMENSION_PATTERNS
    }
    stable_covered_dimensions: set[str] = set()

    def fact_tokens(value: str) -> set[str]:
        return {
            token[:-1]
            if len(token) > 3 and token.endswith("s") and not token.endswith("ss")
            else token
            for token in re.findall(r"[a-z0-9]+", value.lower().replace("_", " "))
        }

    def agrees_with_stable(fragment: str, dimension: str) -> bool:
        fragment_tokens = fact_tokens(fragment)
        return any(
            fact_tokens(tag) <= fragment_tokens
            for tag in stable_by_dimension.get(dimension, ())
        )

    for index, fragment in enumerate(fragments):
        # Identity is normally a ``<name> has ...`` sentence, but writers
        # regularly mix predicates in one list: ``has pink hair ... and is a
        # loli``.  Do not later place ``is a loli`` after a generated ``has``;
        # retain its predicate and assemble a grammatical identity sentence.
        predicate_match = re.fullmatch(r"(?:is|are)\s+(.+)", fragment, re.I)
        if predicate_match:
            fragment = predicate_match.group(1).strip(" ,.;")
        # A writer may also repeat the leading connective after ``and``.
        # These are attributes, not a literal value named ``with glasses``.
        fragment = re.sub(r"^(?:has|with)\s+", "", fragment, flags=re.I)
        if not fragment:
            continue
        dimensions = set(appearance_dimensions(fragment))
        if not dimensions and standalone_color.fullmatch(fragment):
            # LLM prose often writes ``blonde and blue hair`` or ``yellow and
            # green eyes``.  Splitting on ``and`` leaves the first colour without
            # its noun, so inherit the adjacent explicit colour dimension rather
            # than treating it as an unrestricted signature trait.
            next_dimension = (
                appearance_dimension(fragments[index + 1])
                if index + 1 < len(fragments)
                else ""
            )
            previous_dimension = (
                appearance_dimension(fragments[index - 1]) if index else ""
            )
            # In ``yellow and green eyes`` the governing noun is on the right;
            # use the previous fragment only for a trailing orphan adjective.
            inherited_dimension = next_dimension or previous_dimension
            if inherited_dimension in {"hair_color", "eye_color"}:
                dimensions.add(inherited_dimension)
        locked_dimensions = dimensions - set(override_dimensions) - set(
            advisory_writer_dimensions
        )
        if any(
            not agrees_with_stable(fragment, dimension)
            for dimension in locked_dimensions
        ):
            continue
        if predicate_match:
            predicate_clauses.append(f"is {fragment}")
        else:
            kept.append(fragment)
        stable_covered_dimensions.update(
            dimension
            for dimension in dimensions
            if agrees_with_stable(fragment, dimension)
        )
    replacement_dimensions = set(override_dimensions) | set(
        replacement_writer_dimensions
    )
    stable = [
        tag.strip()
        for tag in appearance_tags
        if tag.strip()
        and appearance_dimension(tag) not in replacement_dimensions
        and appearance_dimension(tag) not in stable_covered_dimensions
    ]
    traits = tuple(dict.fromkeys((*kept, *stable)))
    predicates = tuple(dict.fromkeys(predicate_clauses))
    if predicates and traits:
        return (
            f"{character_name} {' and '.join(predicates)} and has "
            f"{', '.join(traits)}"
        )
    if predicates:
        return f"{character_name} {' and '.join(predicates)}"
    return f"{character_name} has {', '.join(traits)}" if traits else writer_identity


def filter_character_appearance_prose(
    text: str,
    appearance_tags: tuple[str, ...],
    override_dimensions: frozenset[str],
) -> str:
    """Remove LLM-authored prose for profile-locked appearance dimensions."""
    locked_dimensions = {
        dimension
        for tag in appearance_tags
        if (dimension := appearance_dimension(tag))
        and dimension not in override_dimensions
    }
    result = str(text or "")
    for dimension in locked_dimensions:
        result = _APPEARANCE_DIMENSION_PATTERNS[dimension].sub("", result)
    result = re.sub(r"\s+,", ",", result)
    result = re.sub(r",\s*(?=,|;|$)", "", result)
    result = re.sub(r"\s+and\s+(?=,|;|$)", "", result, flags=re.I)
    return re.sub(r"\s{2,}", " ", result).strip(" ,;")


def filter_shared_appearance_tags(
    text: str,
    plans: tuple[CharacterEffectiveOutfit, ...],
    *,
    user_prompt: str,
) -> str:
    """Keep character appearance out of ambiguous shared Tags fields.

    With one profiled wearer, only user-opened dimensions may remain in Tags.
    With multiple wearers, every character-specific appearance dimension belongs
    in that wearer's Identity and is removed from the shared tag stream.
    """
    profiled = tuple(plan for plan in plans if plan.appearance_tags)
    if not profiled:
        return text
    if len(plans) > 1:
        per_wearer_overrides = tuple(
            appearance_override_dimensions(
                user_prompt,
                (plan.target_source_text, *plan.target_candidates),
                character_count=len(plans),
            )
            for plan in plans
        )
        shared_overrides = (
            set.intersection(*(set(item) for item in per_wearer_overrides))
            if per_wearer_overrides
            else set()
        )
        blocked_dimensions = set(_APPEARANCE_DIMENSION_PATTERNS) - shared_overrides
    else:
        plan = profiled[0]
        override_dimensions = appearance_override_dimensions(
            user_prompt,
            (plan.target_source_text, *plan.target_candidates),
            character_count=1,
        )
        blocked_dimensions = {
            dimension
            for tag in plan.appearance_tags
            if (dimension := appearance_dimension(tag))
            and dimension not in override_dimensions
        }
    return ", ".join(
        tag
        for tag in split_tags(text)
        if standalone_appearance_dimension(tag) not in blocked_dimensions
    )


def saved_appearance_for_names(
    semantic_result: SemanticLookupResult, names: tuple[str, ...]
) -> tuple[str, tuple[str, ...]]:
    """Resolve one local canonical character and its stable appearance."""
    normalized_names = {
        re.sub(r"[^\w]+", " ", str(name).lower().replace("_", " ")).strip()
        for name in names
        if str(name).strip()
    }
    matches: list[tuple[str, tuple[str, ...]]] = []
    for aliases, source_tag, tags in semantic_result.character_appearance_profiles:
        profile_names = {
            re.sub(r"[^\w]+", " ", str(value).lower().replace("_", " ")).strip()
            for value in (*aliases, source_tag)
            if str(value).strip()
        }
        if any(
            left == right
            or (len(left) >= 3 and left in right)
            or (len(right) >= 3 and right in left)
            for left in normalized_names
            for right in profile_names
        ):
            matches.append((source_tag, tags))
    unique = tuple(dict.fromkeys(matches))
    return unique[0] if len(unique) == 1 else ("", ())


def strip_structured_block_duplicates_from_nltags(
    nltags: str,
    identity_blocks: tuple[str, ...],
    detail_blocks: tuple[str, ...],
) -> str:
    """Keep Nltags narrative that is not already present in protected blocks."""

    def comparable(text: str) -> str:
        return re.sub(
            r"[^\w]+", " ", str(text or "").lower().replace("_", " ")
        ).strip()

    protected = tuple(
        key
        for block in (*identity_blocks, *detail_blocks)
        for clause in re.split(r"\s*;\s*", str(block or ""))
        if (key := comparable(clause))
    )
    kept: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", " ".join(str(nltags or "").split())):
        sentence = sentence.strip()
        key = comparable(sentence)
        if not key:
            continue
        # A shorter Nltags sentence is also redundant when it is simply one
        # clause copied from a richer Identity/Details block.
        if any(key == block_key or key in block_key for block_key in protected):
            continue
        if sentence not in kept:
            kept.append(sentence)
    return " ".join(kept)


def strip_writer_guessed_identity_from_nltags(nltags: str) -> str:
    """Remove stale appearance prose without copying Identity into Nltags."""
    return " ".join(
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+", " ".join(str(nltags or "").split())
        )
        if sentence.strip() and not _STABLE_APPEARANCE_REQUEST_RE.search(sentence)
    )


def safe_global_outfit_tags(
    plans: tuple[CharacterEffectiveOutfit, ...],
) -> tuple[str, ...]:
    """Never promote cached wardrobe components into final global hard tags.

    Per-character profile tags are evidence for LLM2 and a conflict-removal
    boundary, not content the host may independently add to the image. Even a
    component shared by every wearer can be irrelevant to the user's requested
    variation, and automatic promotion previously restored masks and accessories
    the writer deliberately omitted.
    """
    return ()


@dataclass(frozen=True)
class WardrobeAuthority:
    """Single request-level decision about which wardrobe evidence may survive."""

    has_character_plans: bool = False
    all_creative: bool = False
    cached_tags: tuple[str, ...] = ()
    explicit_tags: tuple[str, ...] = ()
    selected_tags: tuple[str, ...] = ()
    source_grounding_tags: tuple[str, ...] = ()
    unbound_grounding_tags: tuple[str, ...] = ()
    removed_tags: tuple[str, ...] = ()
    removed_slots: tuple[str, ...] = ()
    stale_cached_tags: tuple[str, ...] = ()
    character_selected_tags: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @staticmethod
    def _key(tag: str) -> str:
        return tag.strip().lower().replace(" ", "_")

    @staticmethod
    def _matches_slot(tag: str, slot: str) -> bool:
        actual = outfit_tag_slot(tag)
        if slot == "lower_body.all":
            return actual.startswith("lower_body.") or actual == "legwear"
        return bool(actual) and actual == slot

    @classmethod
    def _prose_pattern(cls, tag: str) -> str:
        """Match a grounded tag in prose, allowing ordinary outfit qualifiers."""
        words = cls._key(tag).replace("_", " ").split()
        if len(words) >= 3 and words[-2:] == ["school", "uniform"]:
            prefix = r"\s+".join(re.escape(word) for word in words[:-2])
            return rf"{prefix}(?:\s+[a-z-]+){{0,2}}\s+school\s+uniform"
        return r"\s+".join(re.escape(word) for word in words)

    @classmethod
    def _remove_prose_tag(cls, text: str, tag: str) -> str:
        """Remove one known-conflicting wardrobe assertion conservatively.

        Match actual wearing/list grammar, not every occurrence of a garment
        noun.  This keeps action tags and location phrases such as ``skirt
        lift`` or ``beneath her skirt`` intact.
        """
        phrase = cls._prose_pattern(tag)
        action_suffix = r"(?:lift|tug|pull|grab|adjustment|adjusting|flutter|spread)"
        if cls._key(tag) in {"bottomless", "topless", "nude", "naked"}:
            return re.sub(
                rf"\b(?:and\s+)?(?:is|are|appears?|becomes?)\s+{phrase}\b",
                "",
                text,
                flags=re.I,
            )
        # Writer prose commonly inserts color/material/style adjectives before a
        # forbidden grounded garment (``green pleated skirt`` while the removed
        # tag is ``pleated_skirt``). Consume the complete list item rather than
        # requiring the tag phrase to begin immediately after ``wears``/a comma.
        optional_modifiers = (
            r"(?:(?!(?:and|or|with)\b)[a-z][a-z-]*\s+){0,3}"
        )
        result = re.sub(
            rf"\b(?:wears?|wearing|dressed in|clad in|in)\s+"
            rf"(?:(?:a|an|the)\s+)?{optional_modifiers}{phrase}\b"
            rf"(?!\s+{action_suffix}\b)",
            "",
            text,
            flags=re.I,
        )
        if outfit_tag_slot(tag) == "face_accessory.mask":
            # A removed mask must not return as a held prop. This is the common
            # LLM2 escape hatch after "not wearing a mask" and contradicts the
            # request-level removal authority.
            result = re.sub(
                rf"\b(?:holds?|holding|carries|carrying)\s+"
                rf"(?:(?:a|an|the)\s+)?{optional_modifiers}{phrase}\b",
                "",
                result,
                flags=re.I,
            )
        return re.sub(
            rf"(?:,\s*|\s+and\s+)(?:(?:a|an|the)\s+)?"
            rf"{optional_modifiers}{phrase}\b"
            rf"(?!\s+{action_suffix}\b)",
            "",
            result,
            flags=re.I,
        )

    def filter_tags(self, tags: tuple[str, ...]) -> tuple[str, ...]:
        stale = {
            self._key(tag)
            for tag in (*self.stale_cached_tags, *self.removed_tags)
        }
        return tuple(dict.fromkeys(
            tag
            for tag in tags
            if self._key(tag) not in stale
            and not any(self._matches_slot(tag, slot) for slot in self.removed_slots)
        ))

    def filter_tag_text(self, text: str) -> str:
        return ", ".join(self.filter_tags(tuple(split_tags(text))))

    def filter_prose(self, text: str) -> str:
        result = str(text or "")
        for tag in sorted(
            (*self.stale_cached_tags, *self.removed_tags),
            key=len,
            reverse=True,
        ):
            result = self._remove_prose_tag(result, tag)
        result = re.sub(r"\bwears?\s*(?=,|;|\band\b|$)", "", result, flags=re.I)
        result = re.sub(r"\s+,", ",", result)
        return re.sub(r"\s{2,}", " ", result).strip(" ,;")

    def filter_character_prose(
        self, text: str, plan: CharacterEffectiveOutfit
    ) -> str:
        """Remove only grounded outfit evidence owned by another wearer."""
        result = self.filter_prose(text)
        own_keys = {
            self._key(tag)
            for target_id, tags in self.character_selected_tags
            if target_id.lower() == plan.target_anchor_id.lower()
            for tag in tags
        }
        other_tags = tuple(
            tag
            for target_id, tags in self.character_selected_tags
            if target_id.lower() != plan.target_anchor_id.lower()
            for tag in tags
            if self._key(tag) not in own_keys
        )
        for tag in sorted(other_tags, key=len, reverse=True):
            result = self._remove_prose_tag(result, tag)
        result = re.sub(r"\s+and\s+(?=(?:is|are)\b)", " ", result, flags=re.I)
        return re.sub(r"\s{2,}", " ", result).strip(" ,;")


def build_wardrobe_authority(
    plans: tuple[CharacterEffectiveOutfit, ...],
    semantic_result: SemanticLookupResult,
    anchors: tuple[SemanticAnchor, ...],
    *,
    explicit_term_tags: tuple[str, ...] = (),
) -> WardrobeAuthority:
    cached_tags = tuple(
        dict.fromkeys(
            (
                *semantic_result.outfit_profile_tags,
                *(
                    tag
                    for _alias, _source_tag, tags, _qualifier
                    in semantic_result.source_outfit_profiles
                    for tag in tags
                ),
            )
        )
    )
    explicit_tags = tuple(
        dict.fromkeys(
            (
                *explicit_term_tags,
                *(
                    candidate
                    for anchor in anchors
                    if anchor.role in {"clothing", "outfit"}
                    for candidate in anchor.candidates
                ),
            )
        )
    )
    removed_tags = tuple(
        dict.fromkeys(
            tag for plan in plans for tag in plan.effective.removed_tags if tag
        )
    )
    removed_slots = tuple(dict.fromkeys(
        slot
        for plan in plans
        for slot in plan.effective.forbidden_slots
        if slot
    ))
    removed_keys = {WardrobeAuthority._key(tag) for tag in removed_tags}
    selected_tags = tuple(
        dict.fromkeys(
            tag
            for plan in plans
            for tag in (
                *(
                    ()
                    if plan.wardrobe_kind == "outfit_source"
                    else (plan.wardrobe_tag,)
                ),
                *plan.effective.effective_tags,
            )
            if tag and WardrobeAuthority._key(tag) not in removed_keys
        )
    )
    character_selected_tags = tuple(
        (
            plan.target_anchor_id,
            tuple(
                dict.fromkeys(
                    tag
                    for tag in (
                        *(
                            ()
                            if plan.wardrobe_kind == "outfit_source"
                            else (plan.wardrobe_tag,)
                        ),
                        *plan.effective.effective_tags,
                    )
                    if tag and WardrobeAuthority._key(tag) not in removed_keys
                )
            ),
        )
        for plan in plans
    )
    anchor_by_id = {anchor.anchor_id.lower(): anchor for anchor in anchors}
    source_grounding_tags = tuple(
        dict.fromkeys(
            tag
            for plan in plans
            if plan.wardrobe_kind == "outfit_source"
            for source_anchor in (
                anchor_by_id.get(plan.wardrobe_anchor_id.lower()),
            )
            if source_anchor is not None
            for alias, source_tag, tags, _qualifier
            in semantic_result.source_outfit_profiles
            if (
                source_tag.lower()
                in {candidate.lower() for candidate in source_anchor.candidates}
                or re.sub(r"\s+", " ", alias.strip().lower())
                == re.sub(r"\s+", " ", source_anchor.source_text.strip().lower())
            )
            for tag in tags
            if tag and WardrobeAuthority._key(tag) not in removed_keys
        )
    )
    unresolved_exists = any(
        plan.resolution_state == "explicit_but_unresolved" for plan in plans
    )
    unbound_source_grounding_tags = tuple(
        dict.fromkeys(
            tag
            for anchor in anchors
            if unresolved_exists
            and anchor.role in {"clothing", "outfit", "outfit_source"}
            for alias, source_tag, tags, _qualifier
            in semantic_result.source_outfit_profiles
            if (
                source_tag.lower() in {candidate.lower() for candidate in anchor.candidates}
                or re.sub(r"\s+", " ", alias.strip().lower())
                == re.sub(r"\s+", " ", anchor.source_text.strip().lower())
            )
            for tag in (source_tag, *tags)
            if tag and WardrobeAuthority._key(tag) not in removed_keys
        )
    )
    unresolved_anchor_ids = {
        anchor.anchor_id.lower()
        for anchor in anchors
        if unresolved_exists and anchor.role in {"clothing", "outfit", "outfit_source"}
    }
    unbound_anchor_grounding_tags = tuple(
        dict.fromkeys(
            tag
            for anchor_id, outfit_tag, tags, _qualifier
            in semantic_result.anchor_outfit_profiles
            if anchor_id.lower() in unresolved_anchor_ids
            for tag in (outfit_tag, *tags)
            if tag and WardrobeAuthority._key(tag) not in removed_keys
        )
    )
    unbound_grounding_tags = tuple(
        dict.fromkeys(
            (*unbound_source_grounding_tags, *unbound_anchor_grounding_tags)
        )
    )
    allowed_keys = {
        WardrobeAuthority._key(tag)
        for tag in (
            *explicit_tags,
            *selected_tags,
            *source_grounding_tags,
            *unbound_grounding_tags,
        )
    }
    stale = (
        tuple(
            tag
            for tag in cached_tags
            if WardrobeAuthority._key(tag) not in allowed_keys
        )
        if plans
        else ()
    )
    return WardrobeAuthority(
        has_character_plans=bool(plans),
        all_creative=bool(plans)
        and all(plan.wardrobe_kind == "creative_fallback" for plan in plans),
        cached_tags=cached_tags,
        explicit_tags=explicit_tags,
        selected_tags=selected_tags,
        source_grounding_tags=source_grounding_tags,
        unbound_grounding_tags=unbound_grounding_tags,
        removed_tags=removed_tags,
        removed_slots=removed_slots,
        stale_cached_tags=stale,
        character_selected_tags=character_selected_tags,
    )


def non_wardrobe_confirmed_tags(
    semantic_result: SemanticLookupResult,
    plans: tuple[CharacterEffectiveOutfit, ...],
    *,
    include_source_anchor: bool,
) -> tuple[str, ...]:
    """Keep hard semantic tags without reviving an unselected cached wardrobe."""
    selected_wardrobe_tags = {
        tag
        for plan in plans
        for tag in (
            plan.wardrobe_tag,
            *plan.effective.base_tags,
            *plan.effective.effective_tags,
        )
        if tag
    }
    cached_profile_outfit_tags = {
        tag for tag in semantic_result.outfit_profile_tags if tag
    }
    cached_profile_outfit_tags.update(
        tag
        for _alias, _source_tag, tags, _qualifier
        in semantic_result.source_outfit_profiles
        for tag in tags
        if tag
    )
    return tuple(
        tag
        for tag in semantic_result.confirmed_tags
        if (include_source_anchor or tag not in semantic_result.outfit_source_tags)
        and (not plans or tag not in selected_wardrobe_tags)
        and (not plans or tag not in cached_profile_outfit_tags)
    )


_OFFICIAL_DEFAULT_OUTFIT_RE = re.compile(
    r"(?:官方默认服装|原作(?:默认)?服装|默认服装|标准服装)", re.I
)
_OFFICIAL_CASUAL_PROFILE_RE = re.compile(r"(?:官方常服|官方casual(?:套装|服装)?)", re.I)
_CREATIVE_PRIVATE_OUTFIT_RE = re.compile(
    r"(?:居家私服|私服|居家服|日常便服|日常服装|休闲服(?:装)?|休闲穿搭|"
    r"随意(?:的)?生活服|homewear|home clothes?|private clothes?|off-duty clothes?)",
    re.I,
)
_CASUAL_PROFILE_RE = re.compile(
    r"(?:常服|"
    r"(?<![a-z0-9_])casual(?:\s+(?:clothes?|clothing|outfit|wear|attire|look))?"
    r"(?![a-z0-9_]))",
    re.I,
)
_SUMMER_PROFILE_RE = re.compile(
    r"(?:夏季服装|夏装|夏服|summer\s+(?:clothes?|clothing|outfit|wear|attire))",
    re.I,
)
_WINTER_PROFILE_RE = re.compile(
    r"(?:冬季服装|冬装|冬服|winter\s+(?:clothes?|clothing|outfit|wear|attire))",
    re.I,
)
_STAGE_PROFILE_RE = re.compile(
    r"(?:演出服|舞台服|舞台装|表演服|"
    r"(?:stage|performance|concert)\s+(?:costume|outfit|wear|attire))",
    re.I,
)

CHARACTER_PROFILE_WARDROBE_KINDS = {
    "default_profile",
    "default_reference",
    "casual_profile",
    "summer_profile",
    "winter_profile",
    "stage_profile",
}

WARDROBE_PROFILE_QUALIFIERS = {
    "default_profile": "default",
    "default_reference": "default",
    "casual_profile": "casual",
    "summer_profile": "summer",
    "winter_profile": "winter",
    "stage_profile": "stage",
}

UNSPECIFIED_WARDROBE_POLICIES = {
    "default_profile",
    "creative_fallback",
    "scene_adaptive",
}

DEFAULT_SCENE_ADAPTIVE_WARDROBE_MARKERS = (
    "刚刚出浴",
    "刚出浴",
    "洗完澡",
    "浴后",
    "刚起床",
    "准备睡觉",
    "运动结束后",
    "训练结束后",
    "比赛结束后",
    "比赛",
    "刚跑完步",
    "正在健身",
    "泳池边",
    "泳池旁",
    "在泳池",
    "游泳结束后",
    "演出结束后",
    "刚结束演出",
    "post-bath",
    "after a bath",
    "just woke up",
    "getting ready for bed",
    "after exercising",
    "after training",
    "after the competition",
    "just finished running",
    "working out",
    "poolside",
    "by the pool",
    "at the pool",
    "after swimming",
    "after performing",
)


def requested_wardrobe_mode(user_prompt: str) -> str:
    """Classify an explicit character-owned wardrobe profile request."""
    text = str(user_prompt or "")
    if _OFFICIAL_DEFAULT_OUTFIT_RE.search(text):
        return "default_profile"
    if _OFFICIAL_CASUAL_PROFILE_RE.search(text):
        return "casual_profile"
    if _CREATIVE_PRIVATE_OUTFIT_RE.search(text):
        return "creative_fallback"
    if _CASUAL_PROFILE_RE.search(text):
        return "casual_profile"
    if _SUMMER_PROFILE_RE.search(text):
        return "summer_profile"
    if _WINTER_PROFILE_RE.search(text):
        return "winter_profile"
    if _STAGE_PROFILE_RE.search(text):
        return "stage_profile"
    return ""


def requested_wardrobe_modes_by_target(
    user_prompt: str, targets: tuple[SemanticAnchor, ...]
) -> dict[str, str]:
    """Return explicit profile/private choices scoped to each wearer.

    Request-level mode detection cannot represent mixed assignments such as
    ``A穿官方常服，B穿默认服装``.  Use punctuation-delimited clauses containing
    each target's exact source phrase so one character's choice does not unlock
    or invalidate another character's saved profile.
    """
    clauses = tuple(
        clause.strip()
        for clause in re.split(
            r"[，,。！？!?；;\n]+|(?<!不)(?:而(?:是)?|但(?:是)?|然后|同时)",
            str(user_prompt or ""),
        )
        if clause.strip()
    )
    scoped_modes: dict[str, list[str]] = {}
    shared_scope = re.compile(
        r"(?:双方|两人|二人|全员|所有人|都|各自|\bboth\b|\ball\b)",
        re.I,
    )
    for clause in clauses:
        lowered = clause.lower()
        present = sorted(
            (
                (lowered.find(source.lower()), target, source)
                for target in targets
                if (source := str(target.source_text or "").strip())
                and source.lower() in lowered
            ),
            key=lambda item: item[0],
        )
        if not present:
            continue
        clause_mode = requested_wardrobe_mode(clause)
        if clause_mode and shared_scope.search(clause):
            for _position, target, _source in present:
                scoped_modes.setdefault(target.anchor_id.lower(), []).append(
                    clause_mode
                )
            continue
        # ``A和B穿常服`` is shared even without 都.  In contrast,
        # ``A穿常服和B在泳池边`` binds the mode only to A because the mode occurs
        # inside A's local segment before B begins.
        if len(present) > 1:
            first_position = present[0][0]
            last_position, _last_target, last_source = present[-1]
            roster_text = lowered[first_position : last_position + len(last_source)]
            for _position, _target, source in present:
                roster_text = roster_text.replace(source.lower(), "")
            if (
                not re.sub(r"[\s、和与及跟同&+]+", "", roster_text)
                and requested_wardrobe_mode(lowered[last_position + len(last_source) :])
            ):
                for _position, target, _source in present:
                    scoped_modes.setdefault(target.anchor_id.lower(), []).append(
                        clause_mode
                    )
                continue
        for index, (position, target, source) in enumerate(present):
            next_position = (
                present[index + 1][0] if index + 1 < len(present) else len(clause)
            )
            local_segment = clause[position:next_position]
            if mode := requested_wardrobe_mode(local_segment):
                scoped_modes.setdefault(target.anchor_id.lower(), []).append(mode)
    scoped: dict[str, str] = {}
    for target_id, modes in scoped_modes.items():
        if len(set(modes)) == 1:
            scoped[target_id] = modes[0]
    return scoped


def annotate_requested_profile_variants(
    anchors: tuple[SemanticAnchor, ...], user_prompt: str
) -> tuple[SemanticAnchor, ...]:
    """Carry wearer-scoped profile qualifiers into resolver character queries."""
    targets = tuple(
        anchor for anchor in anchors if anchor.role == "target_character"
    )
    scoped_modes = requested_wardrobe_modes_by_target(user_prompt, targets)
    return tuple(
        replace(
            anchor,
            description=(
                f"{anchor.description} "
                f"{WARDROBE_PROFILE_QUALIFIERS[mode]} outfit variant"
            ).strip(),
        )
        if anchor.role == "target_character"
        and (mode := scoped_modes.get(anchor.anchor_id.lower(), ""))
        in WARDROBE_PROFILE_QUALIFIERS
        and mode != "default_profile"
        else anchor
        for anchor in anchors
    )


def scene_adaptive_wardrobe_marker(
    user_prompt: str,
    markers: tuple[str, ...] = DEFAULT_SCENE_ADAPTIVE_WARDROBE_MARKERS,
) -> str:
    """Return the first configured high-confidence creative wardrobe cue."""
    lowered = str(user_prompt or "").lower()
    return next(
        (
            marker
            for raw_marker in markers
            if (marker := str(raw_marker or "").strip().lower())
            and marker in lowered
            and not (
                marker == "比赛"
                and re.search(r"(?:观看|看|围观|观赏)\s*比赛|观赛", lowered)
            )
        ),
        "",
    )


def resolve_unspecified_wardrobe_mode(
    policy: str,
    *,
    sensual_mode: bool = False,
    scene_marker: str = "",
) -> tuple[str, str]:
    """Resolve the host-side baseline for a request with no clothing intent.

    Scene-adaptive mode still exposes the character's default profile as a soft
    reference. The final writer may replace or adapt it when the scene strongly
    suggests different clothes; explicit creative_fallback remains the opt-out.
    """
    normalized = str(policy or "").strip().lower()
    if normalized not in UNSPECIFIED_WARDROBE_POLICIES:
        normalized = "scene_adaptive"
    if normalized == "creative_fallback":
        return "creative_fallback", "configured_creative"
    if normalized == "scene_adaptive" and (sensual_mode or scene_marker):
        return "default_profile", "scene_adaptive_reference"
    return "default_profile", "configured_default"


def has_explicit_wardrobe_evidence(
    *,
    requested_mode: str,
    outfit_transfer_enabled: bool,
    anchors: tuple[SemanticAnchor, ...],
    plans: tuple[SemanticCharacterPlan, ...],
) -> bool:
    """Return whether config must yield to request-grounded wardrobe evidence."""
    # Anchors have already been schema-validated and their source_text is required
    # to be an exact substring of the request. Their semantic role is therefore
    # stronger and more extensible evidence than a language-specific garment list.
    # This covers unknown proper-name outfits and new garment vocabulary without
    # teaching the host every Chinese or English clothing noun.
    explicit_anchor_ids = {
        anchor.anchor_id.lower()
        for anchor in anchors
        if anchor.role in {"clothing", "outfit", "outfit_source"}
    }
    return bool(
        requested_mode
        or outfit_transfer_enabled
        or explicit_anchor_ids
        or any(
            (
                plan.wardrobe.kind in {"named_outfit", "outfit_source"}
                and plan.wardrobe.anchor_id.lower() in explicit_anchor_ids
            )
            or bool(plan.directives)
            for plan in plans
        )
    )


_HOST_WARDROBE_CUE_RE = re.compile(
    r"(?:穿(?:着|上)?|换(?:上|成)?|服装|衣服|衣物|套装|校服|制服|常服|私服|"
    r"裙|裤|袜|衫|上衣|外套|夹克|礼服|婚纱|睡衣|泳装|内衣|鞋|靴|手套|"
    r"cosplay|costume|outfit|clothes?|clothing|wear(?:s|ing)?)",
    re.I,
)


def has_host_wardrobe_cue(user_prompt: str) -> bool:
    """Conservatively detect clothing intent when the semantic planner fails."""
    return bool(_HOST_WARDROBE_CUE_RE.search(str(user_prompt or "")))


def fallback_target_anchors_from_semantic_evidence(
    user_prompt: str,
    configured_anchors: tuple[SemanticAnchor, ...],
    semantic_result: SemanticLookupResult,
) -> tuple[SemanticAnchor, ...]:
    """Recover only target identities already grounded by local configuration.

    This fallback never guesses an LLM1 relationship.  It merely keeps known
    wearer identities available so an explicit clothing request can degrade to
    ``explicit_but_unresolved`` and stale cached wardrobes can be removed.
    """
    targets = [
        anchor
        for anchor in configured_anchors
        if anchor.role == "target_character"
    ]
    seen_sources = {
        re.sub(r"\s+", " ", anchor.source_text.strip().lower())
        for anchor in targets
    }
    source_profiles = tuple(
        (aliases, source_tag)
        for aliases, source_tag, _tags
        in semantic_result.character_appearance_profiles
    ) + tuple(
        ((alias,), source_tag)
        for alias, source_tag, _tags, _qualifier
        in semantic_result.source_outfit_profiles
        if source_tag not in semantic_result.outfit_source_tags
    )
    for aliases, source_tag in source_profiles:
        matched_alias = next(
            (
                alias
                for alias in sorted(aliases, key=len, reverse=True)
                if alias
                and DanbooruResolver._alias_match_spans(user_prompt, alias)
            ),
            "",
        )
        source_key = re.sub(r"\s+", " ", matched_alias.strip().lower())
        if not matched_alias or source_key in seen_sources or not source_tag:
            continue
        targets.append(
            SemanticAnchor(
                anchor_id=f"fallback_target_{len(targets) + 1}",
                role="target_character",
                group="character",
                source_text=matched_alias,
                description=f"Locally cached character: {matched_alias}",
                candidates=(source_tag,),
            )
        )
        seen_sources.add(source_key)
    return tuple(targets)


_DIRECT_WEAR_VERB = r"(?:穿着?|身穿|身着|换上|换成|wears?|wearing|dressed\s+in)"


def explicit_worn_clothing_phrases(user_prompt: str, wearer: str) -> tuple[str, ...]:
    """Find direct wearer-to-clothing evidence without classifying the garment.

    The text after a wear verb stays intact for the writer. A name elsewhere in
    the same clause is not enough to assign that clothing to the character.
    """
    if not wearer.strip():
        return ()
    phrases: list[str] = []
    for clause in re.split(r"[，,。；;\n]+", str(user_prompt or "")):
        for match in re.finditer(
            rf"{re.escape(wearer.strip())}\s*(?:正|正在)?\s*"
            rf"{_DIRECT_WEAR_VERB}\s*[\"“']?(?P<clothing>.+)",
            clause,
            flags=re.I,
        ):
            clothing = re.split(_DIRECT_WEAR_VERB, match.group("clothing"), maxsplit=1, flags=re.I)[0]
            clothing = clothing.strip(" \t\"”'的")
            if clothing:
                phrases.append(clothing)
    return tuple(dict.fromkeys(phrases))


def repair_single_target_cached_named_outfit_plan(
    plans: tuple[SemanticCharacterPlan, ...],
    anchors: tuple[SemanticAnchor, ...],
    cached_named_result: SemanticLookupResult | None,
    user_prompt: str,
) -> tuple[SemanticCharacterPlan, ...]:
    """Repair locally known named outfits from wearer-scoped explicit clauses.

    A repair requires one exact cached outfit alias after a wearing verb in a
    clause that also names the target. This supports multiple characters without
    distributing one wearer's outfit to another or treating scene context as an
    assignment. Existing outfit-source cosplay relations are never rewritten.
    """
    if cached_named_result is None or not plans:
        return plans
    targets = {
        anchor.anchor_id: anchor
        for anchor in anchors
        if anchor.role == "target_character"
    }
    cached_tags = {
        tag.lower() for tag in cached_named_result.named_outfit_tags if tag
    }
    outfit_anchors: dict[str, SemanticAnchor] = {}
    for anchor in anchors:
        if anchor.role != "outfit" or not anchor.source_text.strip():
            continue
        if cached_tags and not any(
            candidate.lower() in cached_tags for candidate in anchor.candidates
        ):
            continue
        source_key = re.sub(r"\s+", " ", anchor.source_text.strip().lower())
        # Prefer the evidence-bearing cached anchor over the empty intent anchor.
        if anchor.candidates:
            outfit_anchors[source_key] = anchor
    if not outfit_anchors:
        return plans
    repaired: list[SemanticCharacterPlan] = []
    for plan in plans:
        target = targets.get(plan.target_anchor_id)
        if target is None or plan.wardrobe.kind == "outfit_source":
            repaired.append(plan)
            continue
        matches: dict[str, SemanticAnchor] = {}
        for phrase in explicit_worn_clothing_phrases(user_prompt, target.source_text):
            # A later scene mention is not the object of the wear verb.
            worn_object = re.split(
                r"(?:站在|坐在|看着|望着|走到|跑到|放在|挂在|陈列在|"
                r"\b(?:standing|sitting|looking|hanging)\b)",
                phrase,
                maxsplit=1,
                flags=re.I,
            )[0]
            for source_key, outfit_anchor in outfit_anchors.items():
                if DanbooruResolver._alias_match_spans(
                    worn_object[:96], outfit_anchor.source_text
                ):
                    matches[source_key] = outfit_anchor
        if len(matches) == 1:
            outfit_anchor = next(iter(matches.values()))
            plan = replace(
                plan,
                wardrobe=SemanticWardrobe("named_outfit", outfit_anchor.anchor_id),
            )
        repaired.append(plan)
    return tuple(repaired)


def preserve_unresolved_worn_clothing_plans(
    plans: tuple[SemanticCharacterPlan, ...],
    anchors: tuple[SemanticAnchor, ...],
    user_prompt: str,
) -> tuple[SemanticCharacterPlan, ...]:
    """Prevent a missed ordinary outfit from falling back to a saved uniform."""
    targets = {
        anchor.anchor_id: anchor for anchor in anchors
        if anchor.role == "target_character"
    }
    return tuple(
        replace(plan, wardrobe=SemanticWardrobe("creative_fallback"))
        if plan.wardrobe.kind in {"none", "default_profile", "default_reference"}
        and (target := targets.get(plan.target_anchor_id)) is not None
        and explicit_worn_clothing_phrases(user_prompt, target.source_text)
        else plan
        for plan in plans
    )


def explicit_cosplay_assignments(
    anchors: tuple[SemanticAnchor, ...], user_prompt: str
) -> tuple[tuple[str, str], ...]:
    """Extract only explicit wearer/source pairs from character-scoped clauses."""
    targets = tuple(anchor for anchor in anchors if anchor.role == "target_character")
    clauses = tuple(
        clause.strip()
        for clause in re.split(r"[，。；;,;\n]+", str(user_prompt or ""))
        if clause.strip()
    )
    source_token = r"[\u3400-\u9fffA-Za-z0-9_.'!:+\-]{1,40}?"
    patterns = (
        re.compile(
            # Do not reinterpret the noun in ``C 的 cosplay 服装`` as a
            # second verb whose source is the following word ``服装``.
            rf"(?:正在|正|在)?\s*(?<!的)(?:cosplay(?:ing)?(?:\s+as)?|cos|扮演|扮成|"
            rf"装扮成|打扮成)\s*[\"“']?(?P<source>{source_token})"
            rf"(?=(?:的)?(?:cosplay|服装|衣服|服饰|造型|costume|outfit)|"
            rf"[\"”'\s]|$)",
            re.I,
        ),
        re.compile(
            rf"(?:穿着?|身穿|换上|套上)\s*[\"“']?(?P<source>{source_token})"
            rf"(?:的(?:cosplay|角色扮演)?|(?:cosplay|角色扮演))"
            rf"(?:服装|衣服|服饰|造型|costume|outfit)",
            re.I,
        ),
    )
    assignments: list[tuple[str, str]] = []
    for target in targets:
        for clause in clauses:
            if not DanbooruResolver._alias_match_spans(clause, target.source_text):
                continue
            target_match = re.search(re.escape(target.source_text), clause, re.I)
            scoped = clause[target_match.end():] if target_match else clause
            match = None
            for pattern in patterns:
                match = pattern.search(scoped)
                if match:
                    break
            if not match:
                continue
            source = match.group("source").strip(" \"“”'")
            if source and source.lower() != target.source_text.strip().lower():
                pair = (target.anchor_id, source)
                if pair not in assignments:
                    assignments.append(pair)
    return tuple(assignments)


def add_explicit_cosplay_source_anchors(
    anchors: tuple[SemanticAnchor, ...], user_prompt: str
) -> tuple[SemanticAnchor, ...]:
    """Add source-grounded anchors before local profile matching."""
    result = list(anchors)
    existing = {
        re.sub(r"\s+", " ", anchor.source_text.strip().lower())
        for anchor in anchors
        if anchor.role == "outfit_source"
    }
    for _target_id, source in explicit_cosplay_assignments(anchors, user_prompt):
        key = re.sub(r"\s+", " ", source.lower())
        if key in existing:
            continue
        result.append(SemanticAnchor(
            anchor_id=f"host_cosplay_source_{len(result) + 1}",
            role="outfit_source",
            group="character",
            source_text=source,
            description=source,
            candidates=(),
        ))
        existing.add(key)
    return tuple(result)


def repair_explicit_cosplay_plans(
    plans: tuple[SemanticCharacterPlan, ...],
    anchors: tuple[SemanticAnchor, ...],
    user_prompt: str,
) -> tuple[SemanticCharacterPlan, ...]:
    """Make explicit A-cosplays-C wording authoritative over a weak LLM1 enum."""
    sources = {
        re.sub(r"\s+", " ", anchor.source_text.strip().lower()): anchor
        for anchor in anchors
        if anchor.role == "outfit_source"
    }
    assignments = dict(explicit_cosplay_assignments(anchors, user_prompt))
    repaired: list[SemanticCharacterPlan] = []
    for plan in plans:
        source_text = assignments.get(plan.target_anchor_id, "")
        source = sources.get(re.sub(r"\s+", " ", source_text.lower()))
        if source is not None:
            plan = replace(
                plan,
                wardrobe=SemanticWardrobe("outfit_source", source.anchor_id),
            )
        repaired.append(plan)
    existing = {plan.target_anchor_id for plan in repaired}
    for target_id, source_text in assignments.items():
        if target_id in existing:
            continue
        source = sources.get(re.sub(r"\s+", " ", source_text.lower()))
        if source is not None:
            repaired.append(SemanticCharacterPlan(
                target_anchor_id=target_id,
                wardrobe=SemanticWardrobe("outfit_source", source.anchor_id),
            ))
    return tuple(repaired)


def add_host_outfit_changes_to_plans(
    plans: tuple[SemanticCharacterPlan, ...],
    anchors: tuple[SemanticAnchor, ...],
    user_prompt: str,
) -> tuple[SemanticCharacterPlan, ...]:
    """Merge deterministic user clothing changes into LLM1's intent rows.

    LLM1 is allowed to be terse or omit an operation object. The host already
    has conservative, character-scoped parsing for explicit removals/recolors;
    keeping it here prevents a valid source/profile match from being undone by
    one weak planner response while preserving the learning/profile pipeline.
    """
    targets = {
        anchor.anchor_id: anchor
        for anchor in anchors
        if anchor.role == "target_character"
    }
    known_names = tuple(anchor.source_text for anchor in targets.values())
    by_target = {plan.target_anchor_id: plan for plan in plans}
    ordered_ids = [plan.target_anchor_id for plan in plans]
    for target_id, target in targets.items():
        patches = parse_user_outfit_patches(
            user_prompt,
            target.source_text,
            known_character_names=known_names,
        )
        directives = tuple(
            SemanticOutfitDirective(
                operation=(
                    "replace_color" if patch.operation == "replace" else patch.operation
                ),
                slots=tuple(slot for slot in patch.slot.split(",") if slot),
                color=patch.value,
                source_text=patch.evidence,
                target_anchor_id=target_id,
            )
            for patch in patches
            if patch.operation in {"remove", "replace", "recolor_all", "keep_only", "add", "damage"}
            and patch.evidence
        )
        if not directives:
            continue
        current = by_target.get(target_id)
        if current is None:
            current = SemanticCharacterPlan(
                target_anchor_id=target_id,
                wardrobe=SemanticWardrobe("default_profile"),
            )
            ordered_ids.append(target_id)
        by_target[target_id] = replace(
            current,
            directives=tuple(dict.fromkeys((*current.directives, *directives))),
        )
    return tuple(by_target[target_id] for target_id in ordered_ids)


def requests_casual_life_outfit(user_prompt: str) -> bool:
    """Return whether the request selects the evidence-backed casual variant."""
    return requested_wardrobe_mode(user_prompt) == "casual_profile"


def apply_requested_wardrobe_mode(
    plans: tuple[SemanticCharacterPlan, ...],
    anchors: tuple[SemanticAnchor, ...],
    mode: str,
) -> tuple[tuple[SemanticCharacterPlan, ...], tuple[SemanticAnchor, ...]]:
    """Apply deterministic request semantics when the planning LLM is incomplete."""
    supported_modes = {
        "default_profile", "casual_profile", "summer_profile",
        "winter_profile", "stage_profile", "creative_fallback",
    }
    if mode not in supported_modes:
        return plans, anchors
    if plans:
        plans = tuple(
            replace(plan, wardrobe=SemanticWardrobe(mode))
            if plan.wardrobe.kind
            in (*supported_modes, "none")
            else plan
            for plan in plans
        )
    else:
        plans = tuple(
            SemanticCharacterPlan(
                target_anchor_id=anchor.anchor_id,
                wardrobe=SemanticWardrobe(mode),
            )
            for anchor in anchors
            if anchor.role == "target_character"
        )
    if mode in WARDROBE_PROFILE_QUALIFIERS and mode != "default_profile":
        target_ids = {
            plan.target_anchor_id.lower()
            for plan in plans
            if plan.wardrobe.kind == mode
        }
        variant = WARDROBE_PROFILE_QUALIFIERS[mode]
        anchors = tuple(
            replace(
                anchor,
                description=(
                    f"{anchor.description} {variant} outfit variant"
                ).strip(),
            )
            if anchor.anchor_id.lower() in target_ids
            else anchor
            for anchor in anchors
        )
    return plans, anchors


def apply_character_wardrobe_baselines(
    plans: tuple[SemanticCharacterPlan, ...],
    anchors: tuple[SemanticAnchor, ...],
    *,
    implicit_mode: str,
    user_prompt: str,
    create_missing: bool,
) -> tuple[SemanticCharacterPlan, ...]:
    """Choose each character's base wardrobe without flattening request scope.

    ``none`` means the request did not choose clothes for that wearer.  The usual
    fallback is a soft ``default_reference``: LLM2 sees the saved default pieces
    but may adapt or replace them for the scene. A clothing mutation without a
    named/default/casual or ordinary-garment base uses the same soft reference and
    keeps the mutation mandatory. Explicit per-character modes remain stronger.
    """
    targets = tuple(
        anchor for anchor in anchors if anchor.role == "target_character"
    )
    wardrobe_anchors = tuple(
        anchor
        for anchor in anchors
        if anchor.role in {"clothing", "outfit", "outfit_source"}
        and anchor.source_text.strip()
    )
    scoped_modes = requested_wardrobe_modes_by_target(user_prompt, targets)
    explicit_anchor_targets: set[str] = set()
    clauses = tuple(
        clause.strip()
        for clause in re.split(r"[，,。！？!?；;\n]+", str(user_prompt or ""))
        if clause.strip()
    )
    shared_scope = re.compile(
        r"(?:双方|两人|二人|全员|所有人|都|各自|\bboth\b|\ball\b)",
        re.I,
    )
    for clause in clauses:
        lowered = clause.lower()
        present_targets = sorted(
            (
                (lowered.find(target.source_text.lower()), target)
                for target in targets
                if target.source_text.strip()
                and target.source_text.lower() in lowered
            ),
            key=lambda item: item[0],
        )
        anchor_positions = tuple(
            lowered.find(anchor.source_text.lower())
            for anchor in wardrobe_anchors
            if anchor.source_text.lower() in lowered
        )
        if not present_targets or not anchor_positions:
            continue
        if len(targets) == 1 or shared_scope.search(clause):
            explicit_anchor_targets.update(
                target.anchor_id.lower() for _position, target in present_targets
            )
            continue
        if len(present_targets) > 1:
            first_position = present_targets[0][0]
            last_position, last_target = present_targets[-1]
            roster_text = lowered[
                first_position : last_position + len(last_target.source_text)
            ]
            for _position, target in present_targets:
                roster_text = roster_text.replace(target.source_text.lower(), "")
            if (
                not re.sub(r"[\s、和与及跟同&+]+", "", roster_text)
                and any(position > last_position for position in anchor_positions)
            ):
                explicit_anchor_targets.update(
                    target.anchor_id.lower() for _position, target in present_targets
                )
                continue
        for index, (position, target) in enumerate(present_targets):
            next_position = (
                present_targets[index + 1][0]
                if index + 1 < len(present_targets)
                else len(clause)
            )
            if any(
                position < anchor_position < next_position
                for anchor_position in anchor_positions
            ):
                explicit_anchor_targets.add(target.anchor_id.lower())
    updated: list[SemanticCharacterPlan] = []
    existing_ids: set[str] = set()
    for plan in plans:
        target_id = plan.target_anchor_id.lower()
        existing_ids.add(target_id)
        kind = plan.wardrobe.kind
        scoped_mode = scoped_modes.get(target_id, "")
        if kind not in {"named_outfit", "outfit_source"}:
            if scoped_mode:
                kind = scoped_mode
            elif kind == "default_profile" and target_id not in explicit_anchor_targets:
                # "skirt becomes blue" / "wears no top" names a mutation,
                # not an explicit request to wear the pristine default. Expose a
                # modified default reference while keeping the mutation mandatory.
                # Likewise, an unscoped LLM1 default_profile is only a fallback:
                # the strong default selection requires an explicit per-character
                # default phrase in the user's request.
                # creative_fallback is intentionally excluded: LLM1 uses it when
                # the request already supplies an ordinary garment base such as a
                # one-piece swimsuit, which must not load the saved default at all.
                kind = "default_reference"
            elif plan.directives and kind == "none":
                kind = "default_reference"
            elif kind == "none":
                kind = (
                    "default_reference"
                    if implicit_mode == "default_profile"
                    else "creative_fallback"
                )
        updated.append(
            replace(plan, wardrobe=SemanticWardrobe(kind, plan.wardrobe.anchor_id))
        )
    for target in targets:
        if target.anchor_id.lower() in existing_ids:
            continue
        scoped_mode = scoped_modes.get(target.anchor_id.lower(), "")
        if create_missing or scoped_mode:
            kind = scoped_mode or (
                "default_reference"
                if implicit_mode == "default_profile"
                else "creative_fallback"
            )
            updated.append(
                SemanticCharacterPlan(
                    target_anchor_id=target.anchor_id,
                    wardrobe=SemanticWardrobe(kind),
                )
            )
    return tuple(updated)


def filter_profile_tags_for_composition(
    tags: tuple[str, ...], user_prompt: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Omit saved wardrobe components that cannot be visible in the crop.

    This applies only to local profile evidence before user patches. Explicitly
    requested additions are synthesized afterwards and therefore remain intact.
    """
    hidden_slots = framing_hidden_outfit_slots(user_prompt)
    if not hidden_slots:
        return tags, ()
    kept: list[str] = []
    omitted: list[str] = []
    for tag in tags:
        if outfit_tag_slot(tag) in hidden_slots:
            omitted.append(tag)
        else:
            kept.append(tag)
    return tuple(kept), tuple(omitted)


def apply_framing_to_character_outfits(
    plans: tuple[CharacterEffectiveOutfit, ...], framing_text: str
) -> tuple[CharacterEffectiveOutfit, ...]:
    """Apply an LLM-selected crop to profile components without erasing patches.

    The first pass sees user-authored framing before LLM2. This second reusable
    pass also handles framing chosen by LLM2 itself. Tags explicitly synthesized
    by a user patch remain protected even when their saved-profile slot is outside
    the crop; only inherited profile components are optional composition evidence.
    """
    hidden_slots = framing_hidden_outfit_slots(framing_text)
    if not hidden_slots:
        return plans
    updated: list[CharacterEffectiveOutfit] = []
    for plan in plans:
        added_keys = {
            WardrobeAuthority._key(tag) for tag in plan.effective.added_tags
        }
        omitted = tuple(
            tag
            for tag in plan.effective.base_tags
            if outfit_tag_slot(tag) in hidden_slots
            and WardrobeAuthority._key(tag) not in added_keys
        )
        if not omitted:
            updated.append(plan)
            continue
        omitted_keys = {WardrobeAuthority._key(tag) for tag in omitted}
        effective = replace(
            plan.effective,
            base_tags=tuple(
                tag
                for tag in plan.effective.base_tags
                if WardrobeAuthority._key(tag) not in omitted_keys
            ),
            effective_tags=tuple(
                tag
                for tag in plan.effective.effective_tags
                if WardrobeAuthority._key(tag) not in omitted_keys
            ),
            removed_tags=tuple(
                dict.fromkeys((*plan.effective.removed_tags, *omitted))
            ),
        )
        updated.append(
            replace(
                plan,
                effective=effective,
                composition_omitted_tags=tuple(
                    dict.fromkeys((*plan.composition_omitted_tags, *omitted))
                ),
            )
        )
    return tuple(updated)


def build_character_effective_outfits(
    anchors: tuple[SemanticAnchor, ...],
    plans: tuple[SemanticCharacterPlan, ...],
    semantic_result: SemanticLookupResult,
    *,
    user_prompt: str,
    known_character_names: tuple[str, ...] = (),
) -> tuple[CharacterEffectiveOutfit, ...]:
    """Resolve each validated character plan without flattening profile ownership."""
    by_id = {anchor.anchor_id.lower(): anchor for anchor in anchors}
    anchor_tags = {anchor_id.lower(): tag for anchor_id, tag in semantic_result.anchor_tags}
    character_profiles = {
        anchor_id.lower(): (character_tag, outfit_tags, appearance_tags)
        for anchor_id, character_tag, outfit_tags, appearance_tags
        in semantic_result.character_profiles
    }
    outfit_profiles = {
        anchor_id.lower(): (outfit_tag, outfit_tags, qualifier)
        for anchor_id, outfit_tag, outfit_tags, qualifier
        in semantic_result.anchor_outfit_profiles
    }

    def cached_profile_for_target(
        target: SemanticAnchor, qualifier: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
        """Return a cache profile only when it belongs uniquely to ``target``."""
        target_text = re.sub(r"\s+", " ", target.source_text.strip().lower())
        target_candidates = {candidate.lower() for candidate in target.candidates}
        matches = [
            tags
            for alias, source_tag, tags, profile_qualifier
            in semantic_result.source_outfit_profiles
            if (not qualifier or profile_qualifier == qualifier) and (
                source_tag.lower() in target_candidates
                or (
                    target_text
                    and re.sub(r"\s+", " ", alias.strip().lower()) == target_text
                )
            )
        ]
        if len(matches) != 1:
            return None
        appearances = (
            semantic_result.appearance_profile_tags
            if len(semantic_result.source_outfit_profiles) == 1
            else ()
        )
        return matches[0], appearances

    def cached_profile_for_wardrobe(
        wardrobe_anchor: SemanticAnchor,
    ) -> tuple[str, tuple[str, ...]] | None:
        """Bind a prompt-matched saved outfit profile back to its LLM1 anchor.

        Prompt-wide cache matching can resolve an editor alias even when local
        Danbooru lookup classifies the same phrase differently or cannot produce
        an ``anchor_outfit_profiles`` row.  The saved profile is still scoped to
        the exact source phrase and must not be demoted to request-wide context.
        """
        source_text = re.sub(
            r"\s+", " ", wardrobe_anchor.source_text.strip().lower()
        )
        candidates = {candidate.lower() for candidate in wardrobe_anchor.candidates}
        requested_mode = requested_wardrobe_mode(
            f"{wardrobe_anchor.source_text} {wardrobe_anchor.description}"
        )
        requested_qualifier = WARDROBE_PROFILE_QUALIFIERS.get(
            requested_mode, ""
        )
        matches = [
            (source_tag, tags)
            for alias, source_tag, tags, qualifier
            in semantic_result.source_outfit_profiles
            if (not requested_qualifier or qualifier == requested_qualifier)
            and (
                (source_text and re.sub(r"\s+", " ", alias.strip().lower()) == source_text)
                or (source_tag and source_tag.lower() in candidates)
            )
        ]
        unique = tuple(dict.fromkeys(matches))
        return unique[0] if len(unique) == 1 else None

    built: list[CharacterEffectiveOutfit] = []
    scoped_character_names = tuple(dict.fromkeys((
        *known_character_names,
        *(
            anchor.source_text
            for anchor in anchors
            if anchor.role == "target_character" and anchor.source_text
        ),
    )))
    for plan in plans:
        target = by_id.get(plan.target_anchor_id.lower())
        if target is None:
            continue
        wardrobe_tag = ""
        wardrobe_kind = plan.wardrobe.kind
        base_tags: tuple[str, ...] = ()
        appearance_tags: tuple[str, ...] = ()
        complete_named_profile = False
        # Stable appearance belongs to the target character, not to the selected
        # wardrobe mode.  In particular, sensual/scene-adaptive requests often
        # switch clothing to creative_fallback; that must not also discard the
        # character's saved hair/eye/body anchors.
        target_profile = character_profiles.get(plan.target_anchor_id.lower())
        requested_qualifier = WARDROBE_PROFILE_QUALIFIERS.get(
            plan.wardrobe.kind, ""
        )
        cached_target_profile = (
            cached_profile_for_target(target, requested_qualifier)
            if requested_qualifier
            else None
        )
        cached_target_appearance_profile = cached_profile_for_target(target, "")
        cached_character_tag, cached_target_appearance = saved_appearance_for_names(
            semantic_result, (target.source_text, *target.candidates)
        )
        if cached_target_appearance:
            appearance_tags = cached_target_appearance
        elif target_profile and target_profile[2]:
            appearance_tags = target_profile[2]
        elif cached_target_profile:
            appearance_tags = cached_target_profile[1]
        elif cached_target_appearance_profile:
            appearance_tags = cached_target_appearance_profile[1]
        if plan.wardrobe.kind in CHARACTER_PROFILE_WARDROBE_KINDS:
            # Prompt-matched cache rows carry an explicit qualifier, so prefer
            # them over an unqualified legacy character_profiles row. The latter
            # remains a compatibility fallback for resolver results whose target
            # query was already annotated with the requested variant.
            if cached_target_profile:
                base_tags, _ = cached_target_profile
            elif target_profile:
                _character_tag, base_tags, _ = target_profile
            elif len(plans) == 1:
                # Older lookup results expose only one request-wide profile. It is
                # safe to inherit solely when there is one wearer; with multiple
                # plans this fallback would flatten character ownership.
                base_tags = semantic_result.outfit_profile_tags
                appearance_tags = semantic_result.appearance_profile_tags
        elif plan.wardrobe.kind in {"named_outfit", "outfit_source"}:
            profile = outfit_profiles.get(plan.wardrobe.anchor_id.lower())
            if profile:
                wardrobe_tag, profile_tags, _qualifier = profile
                base_tags = tuple(dict.fromkeys(
                    profile_tags
                    if plan.wardrobe.kind == "outfit_source"
                    else (wardrobe_tag, *profile_tags)
                ))
                complete_named_profile = bool(profile_tags)
            else:
                wardrobe_anchor = by_id.get(plan.wardrobe.anchor_id.lower())
                cached_wardrobe = (
                    cached_profile_for_wardrobe(wardrobe_anchor)
                    if wardrobe_anchor is not None
                    else None
                )
                if cached_wardrobe:
                    wardrobe_tag, profile_tags = cached_wardrobe
                    base_tags = tuple(dict.fromkeys(
                        profile_tags
                        if plan.wardrobe.kind == "outfit_source"
                        else (wardrobe_tag, *profile_tags)
                    ))
                    complete_named_profile = bool(profile_tags)
                else:
                    wardrobe_tag = anchor_tags.get(plan.wardrobe.anchor_id.lower(), "")
                    if wardrobe_tag and plan.wardrobe.kind != "outfit_source":
                        base_tags = (wardrobe_tag,)
            named_outfit_keys = {
                WardrobeAuthority._key(tag)
                for tag in semantic_result.named_outfit_tags
            }
            if (
                wardrobe_kind == "outfit_source"
                and named_outfit_keys
                and any(
                    WardrobeAuthority._key(tag) in named_outfit_keys
                    for tag in base_tags
                )
            ):
                # LLM1 can mislabel a proper named uniform as a character
                # outfit_source.  Locally confirmed named-outfit evidence is
                # stronger than that soft relation kind and has complete saved
                # components, so it must use named-set semantics rather than
                # open-ended cosplay completion.
                wardrobe_kind = "named_outfit"
        base_tags, composition_omitted_tags = filter_profile_tags_for_composition(
            base_tags, user_prompt
        )
        patches = tuple(
            UserOutfitPatch(
                subject=target.source_text,
                operation="replace" if directive.operation == "replace_color" else directive.operation,
                slot=",".join(directive.slots),
                value=(
                    "bottomless"
                    if directive.operation == "remove" and directive.slots == ("lower_body.all",)
                    else directive.color
                ),
                evidence=directive.source_text,
            )
            for directive in plan.directives
        )
        transfer = OutfitTransferPlan(
            enabled=plan.wardrobe.kind in {"named_outfit", "outfit_source"},
            source_subject=(
                by_id[plan.wardrobe.anchor_id].source_text
                if plan.wardrobe.anchor_id in by_id else ""
            ),
            target_character=target.source_text,
        )
        effective = build_effective_outfit_plan(
            transfer,
            user_prompt=user_prompt,
            base_tags=base_tags,
            known_character_names=scoped_character_names,
            semantic_patches=patches,
        )
        built.append(CharacterEffectiveOutfit(
            target_anchor_id=plan.target_anchor_id,
            target_source_text=target.source_text,
            target_candidates=tuple(dict.fromkeys((
                *target.candidates,
                *((cached_character_tag,) if cached_character_tag else ()),
                *((anchor_tags[plan.target_anchor_id.lower()],)
                  if plan.target_anchor_id.lower() in anchor_tags else ()),
            ))),
            wardrobe_kind=wardrobe_kind,
            wardrobe_anchor_id=plan.wardrobe.anchor_id,
            wardrobe_tag=wardrobe_tag,
            appearance_tags=appearance_tags,
            effective=effective,
            complete_named_profile=complete_named_profile,
            composition_omitted_tags=composition_omitted_tags,
        ))
    return tuple(built)


def complete_character_wardrobe_states(
    plans: tuple[CharacterEffectiveOutfit, ...],
    anchors: tuple[SemanticAnchor, ...],
    semantic_result: SemanticLookupResult,
    *,
    explicit_wardrobe_evidence: bool,
    requested_mode: str = "",
    user_prompt: str = "",
) -> tuple[CharacterEffectiveOutfit, ...]:
    """Represent every visible target with one explicit wardrobe resolution state.

    A real missing LLM1 edge is never reconstructed in code. An existing ``none``
    edge, however, has already established the wearer and is represented as a soft
    default reference even when another character has explicit clothing. This is
    distinct from ``explicit_but_unresolved``, where the planner failed to bind
    request evidence to any wearer.
    """
    targets = tuple(anchor for anchor in anchors if anchor.role == "target_character")
    wardrobe_anchors = tuple(
        anchor
        for anchor in anchors
        if anchor.role in {"clothing", "outfit", "outfit_source"}
    )
    wardrobe_roles = {anchor.role for anchor in wardrobe_anchors}
    scoped_requested_modes = requested_wardrobe_modes_by_target(
        user_prompt, targets
    )

    def accepted_explicit_plan(plan: CharacterEffectiveOutfit) -> bool:
        kind = plan.wardrobe_kind
        scoped_mode = scoped_requested_modes.get(
            plan.target_anchor_id.lower(), ""
        )
        if kind in {"named_outfit", "outfit_source"}:
            return bool(plan.wardrobe_anchor_id)
        if kind == "default_reference":
            return True
        if kind == "creative_fallback":
            if scoped_mode == "creative_fallback":
                return True
            return (
                requested_mode == "creative_fallback" and not wardrobe_roles
            ) or bool(wardrobe_roles and wardrobe_roles <= {"clothing"})
        if kind in CHARACTER_PROFILE_WARDROBE_KINDS - {"default_reference"}:
            if scoped_mode:
                return scoped_mode == kind
            if kind == "default_profile" and plan.effective.modified:
                return True
            return requested_mode == kind and (
                not wardrobe_roles or plan.effective.modified
            )
        return False

    completed: list[CharacterEffectiveOutfit] = []
    existing_ids = {plan.target_anchor_id.lower() for plan in plans}

    def evidence_for(anchor: SemanticAnchor) -> str:
        tags: list[str] = []
        anchor_key = anchor.anchor_id.lower()
        for profile_anchor_id, outfit_tag, profile_tags, _qualifier in (
            semantic_result.anchor_outfit_profiles
        ):
            if profile_anchor_id.lower() == anchor_key:
                tags.extend((outfit_tag, *profile_tags))
        candidates = {candidate.lower() for candidate in anchor.candidates}
        source_text = re.sub(r"\s+", " ", anchor.source_text.strip().lower())
        for alias, source_tag, profile_tags, _qualifier in (
            semantic_result.source_outfit_profiles
        ):
            if source_tag.lower() in candidates or re.sub(
                r"\s+", " ", alias.strip().lower()
            ) == source_text:
                tags.extend((source_tag, *profile_tags))
        tag_text = ", ".join(dict.fromkeys(tag for tag in tags if tag))
        base = (
            f"{anchor.role} evidence from user text {anchor.source_text!r}: "
            f"{anchor.description}"
        )
        return f"{base}; locally grounded candidates: {tag_text}" if tag_text else base

    unresolved_evidence = tuple(evidence_for(anchor) for anchor in wardrobe_anchors)
    for plan in plans:
        if plan.wardrobe_kind == "default_reference":
            completed.append(replace(plan, resolution_state="unspecified"))
        elif explicit_wardrobe_evidence and not accepted_explicit_plan(plan):
            completed.append(
                replace(
                    plan,
                    wardrobe_kind="explicit_but_unresolved",
                    wardrobe_anchor_id="",
                    wardrobe_tag="",
                    effective=EffectiveOutfitPlan(subject=plan.target_source_text),
                    resolution_state="explicit_but_unresolved",
                    unresolved_evidence=unresolved_evidence,
                )
            )
        else:
            completed.append(
                replace(
                    plan,
                    resolution_state=(
                        "resolved"
                        if explicit_wardrobe_evidence
                        or plan.effective.modified
                        else "unspecified"
                    ),
                )
            )
    for target in targets:
        if target.anchor_id.lower() in existing_ids:
            continue
        source_tag, appearance_tags = saved_appearance_for_names(
            semantic_result, (target.source_text, *target.candidates)
        )
        completed.append(
            CharacterEffectiveOutfit(
                target_anchor_id=target.anchor_id,
                target_source_text=target.source_text,
                target_candidates=tuple(dict.fromkeys((
                    *target.candidates, *((source_tag,) if source_tag else ()),
                ))),
                wardrobe_kind=(
                    "explicit_but_unresolved"
                    if explicit_wardrobe_evidence
                    else "creative_fallback"
                ),
                wardrobe_anchor_id="",
                wardrobe_tag="",
                appearance_tags=appearance_tags,
                effective=EffectiveOutfitPlan(subject=target.source_text),
                resolution_state=(
                    "explicit_but_unresolved"
                    if explicit_wardrobe_evidence
                    else "unspecified"
                ),
                unresolved_evidence=(
                    unresolved_evidence if explicit_wardrobe_evidence else ()
                ),
            )
        )
    return tuple(completed)


def fallback_missing_unspecified_profiles(
    plans: tuple[CharacterEffectiveOutfit, ...],
    *,
    explicit_wardrobe_evidence: bool,
) -> tuple[tuple[CharacterEffectiveOutfit, ...], bool]:
    """Let the writer design clothes when an implicit default profile is absent."""
    changed = False
    updated: list[CharacterEffectiveOutfit] = []
    for plan in plans:
        if (
            plan.wardrobe_kind in {"default_profile", "default_reference"}
            and not plan.effective.effective_tags
            and (
                plan.resolution_state == "unspecified"
                or (
                    plan.wardrobe_kind == "default_profile"
                    and not explicit_wardrobe_evidence
                )
            )
        ):
            updated.append(replace(plan, wardrobe_kind="creative_fallback"))
            changed = True
        else:
            updated.append(plan)
    return tuple(updated), changed


_CREATIVE_TAG_DENY_RE = re.compile(
    r"(?:^|_)(?:bottomless|topless|nude|naked)(?:$|_)|(?:^|_)(?:school_)?uniform(?:$|_)",
    re.I,
)


def keep_safe_creative_wardrobe_tags(text: str) -> str:
    """Keep ordinary creative garments but reject named uniforms and nudity."""
    return ", ".join(
        tag
        for tag in split_tags(text)
        if not _CREATIVE_TAG_DENY_RE.search(tag.strip().lower().replace(" ", "_"))
    )


_UNTRUSTED_OUTFIT_DETAIL_RE = re.compile(
    r"\b(?:wears?|wearing|dressed|clothed|outfit|costume|uniform|school uniform|"
    r"dress|gown|skirt|shirt|blouse|jacket|coat|pants|trousers|shorts|underwear|"
    r"panties|stockings?|pantyhose|thighhighs?|socks?|boots?|shoes?|gloves?|"
    r"mask|nude|naked|bottomless|topless)\b",
    re.I,
)


def strip_untrusted_outfit_detail(detail: str) -> str:
    """Remove LLM-authored wardrobe/nudity clauses while retaining other details."""
    text = " ".join(str(detail or "").split()).strip(" ,;")
    if not text:
        return ""
    parts = re.split(r"\s*;\s*|\s*,\s*(?=[a-z_]+\s)|\s+and\s+(?=[a-z_]+\s)", text)
    kept = [part.strip(" ,;") for part in parts if part.strip() and not _UNTRUSTED_OUTFIT_DETAIL_RE.search(part)]
    if kept:
        return "; ".join(dict.fromkeys(kept))
    subject = text.split(None, 1)[0]
    return subject if re.fullmatch(r"[A-Za-z0-9_.'()!-]+", subject) else ""


def _character_outfit_matches(
    character_name: str, plan: CharacterEffectiveOutfit
) -> bool:
    """Match a writer roster name to a plan's source name or canonical candidates."""
    key = _normalized_character_key(character_name)
    candidates = {
        _normalized_character_key(plan.target_source_text),
        *(_normalized_character_key(candidate) for candidate in plan.target_candidates),
    }
    return key in candidates


def _semantic_appearance_changes_for_character(
    changes: tuple[SemanticAppearanceChange, ...],
    plan: CharacterEffectiveOutfit,
) -> tuple[SemanticAppearanceChange, ...]:
    """Return LLM1 advisory identity changes belonging to one profiled wearer."""
    aliases = {
        _normalized_character_key(value)
        for value in (plan.target_source_text, *plan.target_candidates)
        if str(value).strip()
    }
    return tuple(
        change
        for change in changes
        if _normalized_character_key(change.character_name) in aliases
    )


def _advisory_writer_dimensions(
    changes: tuple[SemanticAppearanceChange, ...],
) -> frozenset[str]:
    """Map LLM1-only identity hints to dimensions LLM2 may express in prose."""
    return frozenset(
        "eye_color" if dimension == "eye_traits" else dimension
        for change in changes
        for dimension in change.dimensions
        if dimension in _APPEARANCE_DIMENSION_PATTERNS or dimension == "eye_traits"
    )


def _replacement_writer_dimensions(
    changes: tuple[SemanticAppearanceChange, ...],
) -> frozenset[str]:
    """Return dimensions LLM1 explicitly classified as replacing old values."""
    return frozenset(
        "eye_color" if dimension == "eye_traits" else dimension
        for change in changes
        if change.operation in {"replace", "omit"}
        for dimension in change.dimensions
        if dimension in _APPEARANCE_DIMENSION_PATTERNS or dimension == "eye_traits"
    )


def _omitted_appearance_dimensions(
    changes: tuple[SemanticAppearanceChange, ...],
) -> frozenset[str]:
    """A request to hide a feature outranks its saved visual profile."""
    return frozenset(
        "eye_color" if dimension == "eye_traits" else dimension
        for change in changes
        if change.operation == "omit"
        for dimension in change.dimensions
        if dimension in _APPEARANCE_DIMENSION_PATTERNS or dimension == "eye_traits"
    )


def _without_omitted_appearance(
    text: str, dimensions: frozenset[str]
) -> str:
    """Remove positive appearance assertions, retaining visibility instructions."""
    result = str(text or "")
    for dimension in dimensions:
        result = _APPEARANCE_DIMENSION_PATTERNS[dimension].sub("", result)
    result = re.sub(r"\b(?:and|with)\s*(?=[,;]|$)", "", result, flags=re.I)
    result = re.sub(r"\s+,", ",", result)
    result = re.sub(r",\s*(?=,|;|$)", "", result)
    result = re.sub(r"\s{2,}", " ", result)
    return result.strip(" ,;")


def _enforce_nai_appearance_omissions(
    characters: list[dict[str, Any]],
    plans: tuple[CharacterEffectiveOutfit, ...],
    changes: tuple[SemanticAppearanceChange, ...],
) -> None:
    """Apply wearer-scoped visibility authority after the NAI planner rewrites tags."""
    for character in characters:
        wearer = next(
            (
                item for item in plans
                if _character_outfit_matches(character["name"], item)
            ), None,
        )
        if wearer is None:
            continue
        omitted = _omitted_appearance_dimensions(
            _semantic_appearance_changes_for_character(changes, wearer)
        )
        if omitted:
            character["prompt"] = _without_omitted_appearance(
                character["prompt"], omitted
            )


def reconcile_character_hints_with_appearance(
    hints: dict[str, str], plans: tuple[CharacterEffectiveOutfit, ...]
) -> dict[str, str]:
    """Replace stale configured prose with the current wearer-scoped profile."""
    reconciled: dict[str, str] = {}
    for hint_name, hint_text in hints.items():
        matching_plan = next(
            (
                item
                for item in plans
                if _character_outfit_matches(hint_name, item) and item.appearance_tags
            ),
            None,
        )
        if matching_plan is None:
            reconciled[hint_name] = hint_text
            continue
        canonical = next(
            (
                candidate
                for candidate in matching_plan.target_candidates
                if candidate.strip()
            ),
            hint_name,
        )
        reconciled[hint_name] = ", ".join(
            dict.fromkeys((canonical, *matching_plan.appearance_tags))
        )
    return reconciled


def controlled_character_outfit_detail(
    detail: str,
    character_name: str,
    plan: CharacterEffectiveOutfit | None,
    *,
    user_prompt: str = "",
    wardrobe_authority: WardrobeAuthority | None = None,
) -> str:
    """Filter one writer Details clause without injecting cached wardrobe tags."""
    raw_writer_has_outfit = bool(_UNTRUSTED_OUTFIT_DETAIL_RE.search(str(detail or "")))
    if wardrobe_authority is not None:
        authority_for_detail = wardrobe_authority
        if (
            plan is not None
            and plan.resolution_state == "unspecified"
            and raw_writer_has_outfit
            and isinstance(wardrobe_authority, WardrobeAuthority)
        ):
            # LLM2 sees the complete raw request. If it supplies an outfit where
            # LLM1 reported no relation, the implicit fallback must yield. Keep
            # explicit removals and cross-wearer ownership filtering, but do not
            # delete the writer's choice merely because it matches a cached default.
            authority_for_detail = replace(
                wardrobe_authority, stale_cached_tags=()
            )
        detail = (
            authority_for_detail.filter_character_prose(detail, plan)
            if plan is not None
            and hasattr(authority_for_detail, "filter_character_prose")
            else authority_for_detail.filter_prose(detail)
        )
    if plan is not None:
        result = " ".join(str(detail or "").split()).strip(" ,;")
        result = re.sub(r"\b(?:and\s+)?(?:is|are)\s*(?=$|[,;.])", "", result, flags=re.I)
        return re.sub(r"\s{2,}", " ", result).strip(" ,;") or character_name
    return " ".join(str(detail or "").split()).strip(" ,;")


def character_wardrobe_authority_context(
    plans: tuple[CharacterEffectiveOutfit, ...]
) -> str:
    """Render per-character wardrobe ownership rules for the prompt writer.

    Grounded plans provide evidence floors; unresolved plans explicitly delegate
    relationship interpretation to the writer without restoring cached defaults.
    """
    if not plans:
        return ""
    lines = ["Per-character clothing evidence:"]
    for plan in plans:
        mutations = "; ".join(
            f"{patch.operation} {patch.slot} (user evidence: {patch.evidence})"
            for patch in plan.effective.patches
            if patch.operation and patch.slot
        )
        mutation_rule = (
            f" Changes: {mutations}."
            if mutations
            else ""
        )
        crop_rule = (
            " Saved profile components outside the requested framing were omitted; "
            "do not restore them."
            if plan.composition_omitted_tags
            else ""
        )
        if plan.resolution_state == "explicit_but_unresolved":
            evidence = " | ".join(plan.unresolved_evidence) or (
                "the request has explicit wardrobe intent but no accepted binding"
            )
            lines.append(
                f"- {plan.target_source_text}: explicit clothing unresolved; infer only "
                f"from the user request, never restore defaults. Evidence: {evidence}"
            )
        elif plan.wardrobe_kind == "default_reference":
            tags = ", ".join(plan.effective.effective_tags) or "creative fallback"
            lines.append(
                f"- {plan.target_source_text}: default reference = {tags}. Use when "
                f"context fits; otherwise adapt to the scene.{mutation_rule}{crop_rule}"
            )
        elif plan.resolution_state == "unspecified":
            lines.append(
                f"- {plan.target_source_text}: clothing unstated; choose from context."
            )
        elif plan.wardrobe_kind == "creative_fallback":
            lines.append(
                f"- {plan.target_source_text}: explicit ordinary clothing; complete its "
                "visible details without inventing a named uniform."
                f"{mutation_rule}{crop_rule}"
            )
        elif plan.wardrobe_kind == "named_outfit" and plan.complete_named_profile:
            tags = ", ".join(plan.effective.effective_tags) or "no grounded garments"
            lines.append(
                f"- {plan.target_source_text}: named outfit base = {tags}. Keep its "
                "recognizable unmodified components; apply every user-requested "
                "change to the base even when a saved component conflicts. Do not "
                "substitute another outfit."
                f"{mutation_rule}{crop_rule}"
            )
        elif plan.wardrobe_kind == "outfit_source":
            tags = ", ".join(plan.effective.effective_tags) or "no verified garments"
            source = plan.wardrobe_tag or "unresolved source"
            lines.append(
                f"- {plan.target_source_text}: cosplay source = {source}; "
                f"verified outfit anchors = {tags}. "
                "Complete recognizable source clothing only; keep the wearer's own face, "
                "hair, body and identity."
                f"{mutation_rule}{crop_rule}"
            )
        elif plan.wardrobe_kind in CHARACTER_PROFILE_WARDROBE_KINDS - {
            "default_reference"
        }:
            tags = ", ".join(plan.effective.effective_tags) or "no grounded garments"
            qualifier = WARDROBE_PROFILE_QUALIFIERS[plan.wardrobe_kind]
            label = f"EXPLICIT {qualifier.upper()} WARDROBE"
            lines.append(
                f"- {plan.target_source_text}: {label.lower()} = {tags}."
                f"{mutation_rule}{crop_rule}"
            )
        else:
            tags = ", ".join(plan.effective.effective_tags) or "no grounded garments"
            lines.append(
                f"- {plan.target_source_text}: verified clothing anchors = {tags}."
                f"{mutation_rule}{crop_rule}"
            )
    return "\n".join(lines)


class PromptPipeline:
    """Build Anima prompts from chat text while keeping plugin main thin."""

    def __init__(
        self,
        *,
        context: Any,
        config: dict[str, Any],
        logger: Any,
        danbooru_resolver: DanbooruResolver,
        researcher: PromptResearcher,
        get_bool: Callable[[str, bool], bool],
        get_int: Callable[[str, int], int],
        get_float: Callable[[str, float], float],
        get_str: Callable[[str, str], str],
        shorten: Callable[[str, int], str],
    ):
        """Store dependencies supplied by the AstrBot plugin.

        Args:
            context: AstrBot plugin context used for provider lookup and LLM calls.
            config: Plugin configuration dict.
            logger: Logger compatible with AstrBot logger methods.
            danbooru_resolver: Danbooru core tag resolver.
            researcher: Optional web/deep-thinking research planner.
            get_bool: Config boolean accessor.
            get_int: Config integer accessor.
            get_float: Config float accessor.
            get_str: Config string accessor.
            shorten: Text-shortening helper for summaries.
        """
        self.context = context
        self.config = config
        self.logger = logger
        self._danbooru_resolver = danbooru_resolver
        self._researcher = researcher
        self._bool = get_bool
        self._int = get_int
        self._float = get_float
        self._str = get_str
        self._shorten = shorten

    def _model_refusal_result(
        self,
        summary: dict[str, Any],
        response: str,
        *,
        multi_person: bool = False,
    ) -> PromptPipelineResult:
        """Build an empty prompt result that stops generation after refusal."""
        self.logger.warning("[comfyui_agent] prompt model refused generation")
        summary.update(
            {
                "model_refused_generation": True,
                "model_refusal_head": self._shorten(response, 300),
                "skipped_reason": "model_refused_generation",
                "final_prompt_head": "",
                "final_prompt_chars": 0,
            }
        )
        if multi_person:
            summary.update(
                {
                    "multi_person_mode": True,
                    "multi_person_plan_failed": False,
                    "multi_person_error": "",
                }
            )
        return PromptPipelineResult("", summary)

    async def _current_chat_provider_id(self, event: Any) -> str:
        """Resolve the prompt provider from explicit config, chat state, then default.

        Failure to inspect the active chat provider is non-fatal because AstrBot's
        per-origin default remains a valid final fallback.
        """
        configured = self._str("prompt_builder_provider_id", "").strip()
        if configured:
            return configured
        try:
            provider_id = await self.context.get_current_chat_provider_id(
                event.unified_msg_origin
            )
            return str(provider_id or "").strip()
        except Exception as exc:
            self.logger.warning(
                "[comfyui_agent] failed to get current chat provider: %s", exc
            )
        cfg = self.context.get_config(umo=event.unified_msg_origin)
        return str(
            cfg.get("provider_settings", {}).get("default_provider_id") or ""
        ).strip()

    async def _generate_prompt_tags_with_llm(
        self,
        *,
        provider_id: str,
        llm_prompt: str,
        use_deep_thinking: bool,
        fixed_character: bool,
        character_name: str = "",
        allow_futa: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        """Ask the configured provider for the seven-block Anima prompt payload.

        Args:
            provider_id: AstrBot provider selected for prompt construction.
            llm_prompt: Fully rendered user and validation context.
            use_deep_thinking: Whether to request provider reasoning controls.
            fixed_character: Whether a configured fixed character is active.
            character_name: Fixed identity whose built-in appearance must be omitted.
            allow_futa: Whether the user's own text positively requests futa traits.

        Returns:
            Provider completion text normalized across supported response shapes.
        """
        sexual_trait_rule = (
            " When the user explicitly requests a futa character, count that "
            "character inside the girls total: a lone futa is `1girl, futanari`; "
            "one futa plus one female is `2girls, futa with female`; one futa "
            "plus one male is `futa with male`."
            if allow_futa
            else " Never invent sex characteristics or genital anatomy that the "
            "user did not positively request; erotic acts and masturbation are "
            "not evidence for them, and explicitly negated traits are forbidden."
        )
        kwargs: dict[str, Any] = {
            "chat_provider_id": provider_id,
            "prompt": llm_prompt,
            "system_prompt": (
                "你是 Anima 模型的 Danbooru tag 提示词助手。"
                "请在内部充分推理和校验参考对象的视觉特征，但不要输出思考过程。"
                "只输出英文 danbooru tags，用英文逗号分隔。"
                "不要解释，不要 Markdown，不要输出质量词或画师词。"
                "严格按用户原始文字判断背景；未提背景时使用白底立绘。"
                "按用户是否明确要求场景，在最后输出且只输出一个背景控制标记。"
                "Follow the user prompt's seven-brace-block response format exactly. "
                "Count must contain an exact Danbooru people-count tag; never write "
                "a bare number, omit Count, or merge it into Characters. Characters "
                "contains names only. Every character appears exactly once in both "
                "Identity and Details; keep its clothing in its own Details clause."
                + sexual_trait_rule
            ),
            "max_tokens": max_tokens or self._int("prompt_builder_max_tokens", 700),
            "thinking": {
                "type": "enabled" if use_deep_thinking else "disabled"
            },
        }
        if use_deep_thinking:
            kwargs["reasoning_effort"] = (
                self._str("prompt_builder_reasoning_effort", "high") or "high"
            )
        response = await self.context.llm_generate(**kwargs)
        return _extract_completion_text(response)

    async def _generate_outfit_summary_with_llm(
        self,
        *,
        provider_id: str,
        summary_prompt: str,
        use_deep_thinking: bool,
    ) -> str:
        """Extract outfit-only tags from reference evidence with a bounded response.

        Identity, scene, quality, and artist tags are excluded by contract because
        the summary may later be applied to a different visible character.
        """
        kwargs: dict[str, Any] = {
            "chat_provider_id": provider_id,
            "prompt": summary_prompt,
            "system_prompt": (
                "你是二次元服装解析助手。"
                "请在内部充分推理来源服装结构，但不要输出思考过程。"
                "只输出英文 danbooru tags，用英文逗号分隔。"
                "不要解释，不要 Markdown，不要输出质量词、画师词或角色身份词。"
            ),
            "max_tokens": min(self._int("prompt_builder_max_tokens", 700), 500),
            "thinking": {
                "type": "enabled" if use_deep_thinking else "disabled"
            },
        }
        if use_deep_thinking:
            kwargs["reasoning_effort"] = (
                self._str("prompt_builder_reasoning_effort", "high") or "high"
            )
        response = await self.context.llm_generate(**kwargs)
        return _extract_completion_text(response)

    async def _generate_character_candidates_with_llm(
        self,
        *,
        provider_id: str,
        user_prompt: str,
        rejected_content: str,
        target_name: str = "",
        copyright_hints: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        """Extract bounded Danbooru character candidates from the user request.

        Args:
            provider_id: Active AstrBot provider identifier.
            user_prompt: Original request containing the named character.
            rejected_content: Initial LLM tags that online lookup could not verify.
            target_name: Optional character name selected by a multi-person plan.
            copyright_hints: First-round work scopes used only for disambiguation.

        Returns:
            Normalized candidate tags proposed for evidence-based lookup.
        """
        response = await self.context.llm_generate(
            chat_provider_id=provider_id,
            prompt=(
                "Extract the explicitly named existing anime/game character from "
                "the user request and propose up to 6 possible canonical Danbooru "
                "character tags. Preserve the exact source-language name and infer "
                "the work/copyright separately. Candidate tags must use lowercase "
                "ASCII, underscores, and a work disambiguation suffix when known.\n\n"
                'Return JSON only: {"source_name":"原文中的名字",'
                '"copyright":"work name","tag_candidates":["name_(work)"]}.\n'
                "The source_name must be an exact substring of the user request. "
                "Do not invent a character when none is explicitly named. A localized "
                "class/title may be the official localized name of a character: recover "
                "that character's official English or romanized proper name instead of "
                "literally translating it as a generic noun.\n\n"
                f"Target name: {target_name or 'not separately specified'}\n"
                "Known work/copyright scope hints: "
                f"{', '.join(copyright_hints) or 'none'}\n"
                f"User request: {user_prompt}\n"
                f"Initial character tags to cross-check: "
                f"{self._shorten(rejected_content, 500)}"
            ),
            system_prompt=(
                "You resolve named anime and game characters to candidate Danbooru "
                "tags. Return valid JSON only. Your candidates are search hints, "
                "not authoritative answers."
            ),
            max_tokens=350,
            thinking={"type": "disabled"},
        )
        raw = _extract_completion_text(response)
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if match:
            raw = match.group(0)
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return ()
        if not isinstance(data, dict):
            return ()
        source_name = str(data.get("source_name") or "").strip()
        if not source_name or source_name not in user_prompt:
            return ()
        raw_candidates = data.get("tag_candidates")
        if not isinstance(raw_candidates, list):
            return ()
        candidates: list[str] = []
        for item in raw_candidates:
            candidate = str(item or "").strip().lower()
            candidate = re.sub(r"\s+", "_", candidate)
            if not re.fullmatch(r"[a-z0-9_.'():-]{3,100}", candidate):
                continue
            if candidate not in candidates:
                candidates.append(candidate)
        return tuple(candidates[:6])

    async def _refine_unresolved_semantic_characters(
        self,
        *,
        provider_id: str,
        user_prompt: str,
        anchors: tuple[SemanticAnchor, ...],
        result: SemanticLookupResult,
    ) -> tuple[SemanticAnchor, ...]:
        """Add bounded canonical guesses for unresolved visible characters."""
        anchor_tag_map = dict(result.anchor_tags)

        def confirmed_tag_for(anchor: SemanticAnchor) -> str:
            mapped = anchor_tag_map.get(anchor.anchor_id, "")
            if mapped:
                return mapped
            return next(
                (
                    tag
                    for tag in result.confirmed_tags
                    if any(
                        tag == candidate or tag.startswith(candidate + "_(")
                        for candidate in anchor.candidates
                    )
                ),
                "",
            )

        copyright_hints = tuple(
            dict.fromkeys(
                tag
                for anchor in anchors
                if anchor.role == "copyright"
                for tag in (
                    confirmed_tag_for(anchor),
                    *anchor.candidates,
                )
                if tag
            )
        )
        refined: list[SemanticAnchor] = []
        refinement_attempts = 0
        for anchor in anchors:
            if (
                anchor.role != "target_character"
                or confirmed_tag_for(anchor)
                or any("_(" in candidate for candidate in anchor.candidates)
                or refinement_attempts >= 4
            ):
                refined.append(anchor)
                continue
            refinement_attempts += 1
            try:
                candidate_hints = await self._generate_character_candidates_with_llm(
                    provider_id=provider_id,
                    user_prompt=user_prompt,
                    rejected_content=", ".join(anchor.candidates),
                    target_name=anchor.source_text,
                    copyright_hints=copyright_hints,
                )
            except Exception as exc:
                self.logger.warning(
                    "[comfyui_agent] semantic character refinement failed target=%s: %s",
                    anchor.source_text,
                    exc,
                )
                candidate_hints = ()
            merged_candidates = tuple(
                dict.fromkeys((*candidate_hints, *anchor.candidates))
            )[:6]
            if candidate_hints and merged_candidates != anchor.candidates:
                anchor = replace(anchor, candidates=merged_candidates)
            refined.append(anchor)
        return tuple(refined)

    async def _generate_semantic_plan_with_llm(
        self, *, provider_id: str, user_prompt: str, max_tokens: int | None = None
    ) -> str:
        """Propose bounded lookup anchors; the local index remains authoritative."""
        planner_prompt = build_semantic_plan_prompt(user_prompt)
        configured_system_prompt = self._str(
            "danbooru_semantic_system_prompt", ""
        ).strip()
        planner_system_prompt = (
            DEFAULT_SEMANTIC_PLAN_SYSTEM_PROMPT
            if not configured_system_prompt
            or configured_system_prompt in LEGACY_SEMANTIC_PLAN_SYSTEM_PROMPTS
            else configured_system_prompt
        )
        if self._bool("debug_prompt_enabled", False):
            self.logger.info(
                "[comfyui_agent] semantic planner LLM prompt:\n%s",
                planner_prompt,
            )
            self.logger.info(
                "[comfyui_agent] semantic planner LLM system prompt:\n%s",
                planner_system_prompt,
            )
        response = await self.context.llm_generate(
            chat_provider_id=provider_id,
            prompt=planner_prompt,
            system_prompt=planner_system_prompt,
            max_tokens=max_tokens or min(self._int("prompt_builder_max_tokens", 900), 900),
            thinking={"type": "disabled"},
        )
        raw_plan = _extract_completion_text(response)
        if self._bool("debug_prompt_enabled", False):
            self.logger.info(
                "[comfyui_agent] semantic planner LLM output:\n%s",
                raw_plan,
            )
        return raw_plan

    async def _repair_semantic_plan_with_llm(
        self,
        *,
        provider_id: str,
        user_prompt: str,
        previous_raw: str,
        issues: tuple[str, ...],
        max_tokens: int | None = None,
    ) -> str:
        """Run one bounded relationship-repair attempt after schema validation."""
        repair_prompt = build_semantic_plan_repair_prompt(
            user_prompt, previous_raw, issues
        )
        configured_system_prompt = self._str(
            "danbooru_semantic_system_prompt", ""
        ).strip()
        planner_system_prompt = (
            DEFAULT_SEMANTIC_PLAN_SYSTEM_PROMPT
            if not configured_system_prompt
            or configured_system_prompt in LEGACY_SEMANTIC_PLAN_SYSTEM_PROMPTS
            else configured_system_prompt
        )
        if self._bool("debug_prompt_enabled", False):
            self.logger.info(
                "[comfyui_agent] semantic planner repair issues:\n%s",
                "\n".join(issues),
            )
            self.logger.info(
                "[comfyui_agent] semantic planner repair prompt:\n%s", repair_prompt
            )
        response = await self.context.llm_generate(
            chat_provider_id=provider_id,
            prompt=repair_prompt,
            system_prompt=planner_system_prompt,
            max_tokens=max_tokens or min(self._int("prompt_builder_max_tokens", 900), 900),
            thinking={"type": "disabled"},
        )
        repaired = _extract_completion_text(response)
        if self._bool("debug_prompt_enabled", False):
            self.logger.info(
                "[comfyui_agent] semantic planner repair output:\n%s", repaired
            )
        return repaired

    async def _generate_constraint_plan_with_llm(
        self,
        *,
        provider_id: str,
        plan_prompt: str,
    ) -> str:
        """Ask the active provider for a low-CFG constraint plan.

        Args:
            provider_id: AstrBot provider identifier.
            plan_prompt: Structured constraint planning prompt.

        Returns:
            Raw JSON text returned by the provider.
        """
        response = await self.context.llm_generate(
            chat_provider_id=provider_id,
            prompt=plan_prompt,
            system_prompt=(
                "You are a strict JSON planner for anime image prompt constraints. "
                "Return only valid JSON. Do not output Markdown or explanations."
            ),
            max_tokens=450,
            thinking={"type": "disabled"},
        )
        return _extract_completion_text(response)

    async def _generate_multi_person_plan_with_llm(
        self,
        *,
        provider_id: str,
        plan_prompt: str,
        use_deep_thinking: bool,
    ) -> str:
        """Ask the active provider for a structured multi-person scene plan.

        Args:
            provider_id: AstrBot provider identifier.
            plan_prompt: Structured planning request.
            use_deep_thinking: Whether to request provider reasoning support.

        Returns:
            Raw JSON text returned by the provider.
        """
        kwargs: dict[str, Any] = {
            "chat_provider_id": provider_id,
            "prompt": plan_prompt,
            "system_prompt": (
                "You plan multi-character Anima illustrations. "
                "Return only valid JSON matching the requested schema. "
                "Keep every character's identity and attributes in its own block."
            ),
            "max_tokens": min(self._int("prompt_builder_max_tokens", 700), 900),
            "thinking": {
                "type": "enabled" if use_deep_thinking else "disabled"
            },
        }
        if use_deep_thinking:
            kwargs["reasoning_effort"] = (
                self._str("prompt_builder_reasoning_effort", "high") or "high"
            )
        response = await self.context.llm_generate(**kwargs)
        return _extract_completion_text(response)

    async def _build_multi_person_prompt(
        self,
        *,
        provider_id: str,
        prompt: str,
        prompt_config: dict[str, Any],
        use_deep_thinking: bool,
        summary: dict[str, Any],
        original_user_prompt: str = "",
    ) -> PromptPipelineResult | None:
        """Build a hybrid tag and natural-language prompt for 2–4 people.

        Args:
            provider_id: Active chat provider identifier.
            prompt: User's multi-person scene request.
            prompt_config: Effective preset-aware plugin configuration.
            use_deep_thinking: Whether the provider should use reasoning mode.
            summary: Mutable request summary populated by this branch.
            original_user_prompt: User text before reference-context expansion.

        Returns:
            Completed prompt result, or None when planning fails and generation
            must stop without entering the ordinary prompt path.
        """
        configured_characters = fixed_character_tags(prompt_config)
        mentioned_fixed_characters = {
            name: tags
            for name, tags in configured_characters.items()
            if name and name in prompt
        }
        plan_prompt = build_multi_person_plan_prompt(
            prompt,
            fixed_characters=mentioned_fixed_characters,
            original_user_prompt=original_user_prompt,
        )
        keyword_rules = match_keyword_prompt_rules(
            original_user_prompt or prompt, prompt_config
        )
        plan_prompt += build_keyword_rule_block(keyword_rules)
        summary["keyword_prompt_rule_markers"] = [
            rule.marker for rule in keyword_rules
        ]
        raw_plan = ""
        plan = None
        planner_error = "invalid_plan"
        planner_retry_count = 0
        for attempt in range(2):
            try:
                raw_plan = await self._generate_multi_person_plan_with_llm(
                    provider_id=provider_id,
                    plan_prompt=plan_prompt,
                    use_deep_thinking=use_deep_thinking,
                )
            except Exception as exc:
                planner_error = self._shorten(str(exc), 300)
                self.logger.warning(
                    "[comfyui_agent] multi-person planner attempt %s failed: %s",
                    attempt + 1,
                    exc,
                )
            else:
                if is_chinese_model_refusal(raw_plan):
                    return self._model_refusal_result(
                        summary,
                        raw_plan,
                        multi_person=True,
                    )
                if not str(raw_plan or "").strip():
                    planner_error = "empty_llm_response"
                    self.logger.warning(
                        "[comfyui_agent] multi-person planner returned empty "
                        "content on attempt %s",
                        attempt + 1,
                    )
                    candidate = None
                else:
                    candidate = parse_multi_person_plan(raw_plan)
                if candidate is not None:
                    allowed_aliases = {
                        f"CHARACTER {letter}"
                        for letter in "ABCD"[: len(candidate.characters)]
                    }
                    interaction_aliases = {
                        alias.upper()
                        for interaction in candidate.interactions
                        for alias in re.findall(
                            r"\bCharacter\s+[A-D]\b",
                            interaction,
                            flags=re.IGNORECASE,
                        )
                    }
                    if (
                        candidate.interactions
                        and interaction_aliases
                        and not interaction_aliases.issubset(allowed_aliases)
                    ):
                        planner_error = "invalid_interaction_aliases"
                    else:
                        plan = candidate
                        break
                else:
                    planner_error = "invalid_plan"
            if attempt == 0:
                planner_retry_count = 1
                plan_prompt += (
                    "\nThe previous response was invalid. Return corrected JSON only. "
                    "Keep 2 to 4 characters, reference only defined Character aliases "
                    "inside interactions, and preserve one coherent shared scene."
                )
        if plan is None:
            self.logger.warning(
                "[comfyui_agent] multi-person planner stopped: %s", planner_error
            )
            summary.update(
                {
                    "multi_person_mode": True,
                    "multi_person_plan_failed": True,
                    "multi_person_error": planner_error,
                    "multi_person_planner_retry_count": planner_retry_count,
                }
            )
            return None

        character_blocks: list[str] = []
        character_entity_names: list[set[str]] = []
        resolved_count = 0
        fixed_character_count = 0
        danbooru_resolved_count = 0
        unresolved_character_count = 0
        character_resolution_statuses: list[dict[str, Any]] = []
        character_slots: list[str] = []
        character_roles: list[str] = []
        emphasized_anchor_count = 0
        used_fixed_names: set[str] = set()
        fixed_genders: list[str] = []
        aliases = ("Character A", "Character B", "Character C", "Character D")
        explicit_position_requested = bool(
            re.search(
                r"左边|右边|左侧|右侧|前景|后方|前后站位|"
                r"\bon\s+the\s+(?:left|right)\b|\bforeground\b|\bbackground\b",
                prompt,
                flags=re.IGNORECASE,
            )
        )
        spatial_mode = plan.spatial_mode
        if explicit_position_requested:
            spatial_mode = "explicit_positions"
        elif plan.interactions:
            spatial_mode = "shared_contact"
        elif spatial_mode == "explicit_positions":
            spatial_mode = "shared_scene"
        grouped_contact = spatial_mode == "shared_contact"
        for index, character in enumerate(plan.characters):
            character_slots.append(character.slot)
            fixed_name = next(
                (
                    name
                    for name in mentioned_fixed_characters
                    if name not in used_fixed_names
                    and (
                        name == character.name
                        or name in character.name
                        or character.name in name
                    )
                ),
                "",
            )
            fixed_tags = ""
            resolved_identity = ""
            resolution_status = "not_requested"
            candidate_hints: tuple[str, ...] = ()
            if fixed_name:
                used_fixed_names.add(fixed_name)
                configured_tags = split_tags(configured_characters[fixed_name])
                normalized_configured_tags = {
                    tag.lower().replace(" ", "") for tag in configured_tags
                }
                if normalized_configured_tags & {
                    "futa",
                    "futanari",
                    "1futa",
                    "1futanari",
                }:
                    fixed_genders.append("futa")
                elif "1girl" in normalized_configured_tags:
                    fixed_genders.append("girl")
                elif "1boy" in normalized_configured_tags:
                    fixed_genders.append("boy")
                fixed_tags = ", ".join(
                    tag
                    for tag in configured_tags
                    if tag.lower()
                    not in {
                        "1girl",
                        "1 girl",
                        "1boy",
                        "1 boy",
                        "solo",
                    }
                )
                resolved_count += 1
                fixed_character_count += 1
                resolution_status = "fixed"
            elif character.danbooru_candidate:
                resolution = await self._danbooru_resolver.resolve_detailed(
                    llm_content=character.danbooru_candidate,
                    user_prompt=character.name or prompt,
                    fixed_character=False,
                )
                if resolution.status == "unresolved" or resolution.explicit_request:
                    try:
                        candidate_hints = (
                            await self._generate_character_candidates_with_llm(
                                provider_id=provider_id,
                                user_prompt=prompt,
                                rejected_content=character.danbooru_candidate,
                                target_name=character.name,
                            )
                        )
                    except Exception as exc:
                        self.logger.warning(
                            "[comfyui_agent] multi-person character candidate "
                            "planner failed for %s: %s",
                            character.name or character.danbooru_candidate,
                            exc,
                        )
                    if candidate_hints:
                        resolution = await self._danbooru_resolver.resolve_detailed(
                            llm_content=character.danbooru_candidate,
                            user_prompt=character.name or prompt,
                            fixed_character=False,
                            candidate_hints=candidate_hints,
                        )
                resolution_status = resolution.status
                if resolution.status == "resolved":
                    resolved_identity = resolution.canonical_tag or next(
                        iter(split_tags(resolution.text)), ""
                    )
                    if len(resolution.identity_tags) > 1:
                        fixed_tags = ", ".join(resolution.identity_tags[1:])
                    resolved_count += 1
                    danbooru_resolved_count += 1
                else:
                    resolved_identity = character.danbooru_candidate
                    unresolved_character_count += 1
            available_identity_tags = [
                tag.strip(" ()")
                for tag in split_tags(fixed_tags or character.appearance)
                if tag.strip(" ()")
                and tag.strip(" ()").lower()
                not in {"1girl", "1 girl", "1boy", "1 boy", "solo"}
            ]
            proposed_identity_tags = [
                str(tag).strip(" ()")
                for tag in character.identity_anchors
                if str(tag).strip(" ()")
            ]
            if fixed_name or resolution_status == "resolved":
                fixed_source = re.sub(
                    r"[^a-z0-9]+",
                    " ",
                    (
                        configured_characters[fixed_name] if fixed_name else fixed_tags
                    ).lower(),
                )
                identity_tags = [
                    tag
                    for tag in proposed_identity_tags
                    if re.sub(r"[^a-z0-9]+", " ", tag.lower()).strip() in fixed_source
                ][:6]
            else:
                identity_tags = proposed_identity_tags[:6]
            if not identity_tags:
                identity_tags = available_identity_tags[:6]
            visual_label = str(character.visual_label or "").strip().lower()
            if visual_label and (
                fixed_name
                or resolution_status in {"resolved", "profile_cache"}
            ):
                label_terms = re.sub(
                    r"[^a-z0-9]+",
                    " ",
                    visual_label.replace("haired", "hair")
                    .replace("eyed", "eyes")
                    .replace("eared", "ears"),
                ).split()
                label_terms = [
                    term
                    for term in label_terms
                    if term not in {"girl", "boy", "woman", "man", "person"}
                ]
                label_source = re.sub(
                    r"[^a-z0-9]+",
                    " ",
                    (
                        configured_characters[fixed_name] if fixed_name else fixed_tags
                    ).lower(),
                )
                if not label_terms or any(
                    term not in label_source for term in label_terms
                ):
                    visual_label = ""
            if not visual_label or re.search(
                r"\b(?:character\s+[a-d]|first|second|third|fourth|rider|"
                r"support(?:ing|er)?|left|right|top|bottom)\b",
                visual_label,
                flags=re.IGNORECASE,
            ):
                descriptors: list[str] = []
                for tag in identity_tags[:2]:
                    descriptor = re.sub(r"[()_:]+", " ", tag.lower())
                    descriptor = re.sub(r"\s+", " ", descriptor).strip()
                    descriptor = re.sub(r"\s+hair$", "-haired", descriptor)
                    descriptor = re.sub(r"\s+eyes$", "-eyed", descriptor)
                    descriptor = re.sub(r"\s+ears$", "-eared", descriptor)
                    if descriptor:
                        descriptors.append(descriptor.replace(" ", "-"))
                gender_label = (
                    "girl"
                    if any("girl" in tag.lower() for tag in plan.count_tags)
                    else "person"
                )
                visual_label = " ".join((*descriptors, gender_label)).strip()
            if not visual_label:
                visual_label = str(character.role or aliases[index]).strip().lower()
            if visual_label in character_roles:
                visual_label = f"{visual_label} {index + 1}"
            character_roles.append(visual_label)

            emphasized = {
                re.sub(r"[^a-z0-9]+", " ", str(tag).lower()).strip()
                for tag in character.emphasized_anchors
                if str(tag).strip()
            }
            rendered_identity_tags = [
                f"({tag}:1.3)"
                if re.sub(r"[^a-z0-9]+", " ", tag.lower()).strip() in emphasized
                else tag
                for tag in identity_tags
            ]
            emphasized_anchor_count += sum(
                rendered != original
                for rendered, original in zip(
                    rendered_identity_tags, identity_tags, strict=True
                )
            )
            character_resolution_statuses.append(
                {
                    "alias": aliases[index],
                    "role": visual_label,
                    "name": character.name,
                    "status": resolution_status,
                    "canonical_tag": resolved_identity
                    if resolution_status in {"resolved", "profile_cache"}
                    else "",
                    "identity_tags": identity_tags,
                    "emphasized_identity_tags": [
                        tag
                        for tag in identity_tags
                        if re.sub(r"[^a-z0-9]+", " ", tag.lower()).strip() in emphasized
                    ],
                    "candidate_hints": list(candidate_hints),
                }
            )
            character_entity_names.append(
                {
                    value
                    for value in (
                        character.name,
                        character.danbooru_candidate,
                        fixed_name,
                        resolved_identity,
                    )
                    if value
                }
            )
            character_blocks.append(
                render_multi_person_character(
                    character,
                    alias=visual_label,
                    resolved_identity=resolved_identity,
                    fixed_tags=fixed_tags if fixed_name else "",
                    grouped_contact=grouped_contact,
                    explicit_positions=spatial_mode == "explicit_positions",
                    identity_anchors=tuple(rendered_identity_tags),
                    # Sitting/lying/leaning ownership is essential in close
                    # contact scenes. The planner already keeps the directed
                    # interaction out of pose, so retaining pose here does not
                    # duplicate the relationship.
                    include_pose=True,
                    outfit_constraints=(),
                )
            )

        blocked_positive_markers = (
            "split screen",
            "panel",
            "multiple view",
            "alternate view",
            "character sheet",
            "duplicate character",
            "cloned character",
        )
        character_count = len(plan.characters)
        deterministic_count_tags: tuple[str, ...] = ()
        if len(fixed_genders) == character_count:
            girl_count = fixed_genders.count("girl")
            boy_count = fixed_genders.count("boy")
            futa_count = fixed_genders.count("futa")
            if character_count == 1 and futa_count == 1:
                deterministic_count_tags = ("1girl", "futanari")
            elif futa_count == 1 and boy_count and not girl_count:
                deterministic_count_tags = tuple(
                    tag
                    for tag in (
                        f"{character_count}people" if character_count > 2 else "",
                        "futa with male",
                    )
                    if tag
                )
            elif futa_count and not boy_count:
                # Anima counts a futa inside the girls count.
                total_girls = girl_count + futa_count
                deterministic_count_tags = (
                    _count_tag(total_girls, "girl"),
                    "futa with female",
                )
            else:
                deterministic_count_tags = tuple(
                    tag
                    for tag in (
                        _count_tag(girl_count, "girl") if girl_count else "",
                        _count_tag(boy_count, "boy") if boy_count else "",
                    )
                    if tag
                )
        if not deterministic_count_tags:
            normalized_plan_count_tags = normalize_anima_count_tags(
                plan.count_tags, character_count
            )
            relationship_count_tags = tuple(
                tag
                for tag in normalized_plan_count_tags
                if tag.lower().replace("_", " ")
                in {"futa with female", "futa with male"}
            )
            numeric_count_tags = tuple(
                tag
                for tag in normalized_plan_count_tags
                if (
                    (
                        match := re.fullmatch(
                            r"\s*(\d+)\s*(girls?|boys?|people|persons?)\s*",
                            tag,
                            flags=re.IGNORECASE,
                        )
                    )
                    and int(match.group(1)) == character_count
                )
            )[:1]
            deterministic_count_tags = (
                tuple(
                    dict.fromkeys(
                        (*numeric_count_tags, *relationship_count_tags[:1])
                    )
                )
                if numeric_count_tags
                else relationship_count_tags[:1] or (f"{character_count}people",)
            )
        filtered_common_tags = tuple(
            tag
            for tag in plan.common_tags
            if not any(marker in tag.lower() for marker in blocked_positive_markers)
            and tag.strip().lower() != str(plan.relationship_tag or "").strip().lower()
            and not (
                plan.interactions
                and tag.strip().lower() in _UNBOUND_DIRECTIONAL_TAGS
            )
            and not re.fullmatch(
                r"\s*\d+\s*(girls?|boys?|people|persons?)\s*",
                tag,
                flags=re.IGNORECASE,
            )
        )
        background_request = str(original_user_prompt or prompt).strip()
        filtered_background_tags, effective_background_mode, background_overridden = (
            enforce_user_background_intent(
                ", ".join(filtered_common_tags),
                plan.background_mode,
                background_request,
            )
        )
        filtered_common_tags = tuple(split_tags(filtered_background_tags))
        generated_scene_selected = bool(
            not user_requests_explicit_background(background_request)
            and has_generated_scene(
                ", ".join(filtered_common_tags), ", ".join(character_blocks)
            )
        )
        if generated_scene_selected:
            effective_background_mode = EXPLICIT_SCENE
            filtered_common_tags = tuple(split_tags(
                strip_default_portrait_tags(", ".join(filtered_common_tags))
            ))
        if effective_background_mode == DEFAULT_PORTRAIT:
            filtered_common_tags = tuple(
                dict.fromkeys(
                    (
                        *filtered_common_tags,
                        "full body",
                        "centered",
                        "simple background",
                        "white background",
                    )
                )
            )
        planned_relationship_tag = str(plan.relationship_tag or "").strip()
        relationship_tag_suppressed = bool(
            plan.interactions
            and planned_relationship_tag.lower() in _UNBOUND_DIRECTIONAL_TAGS
        )
        relationship_tag = "" if relationship_tag_suppressed else planned_relationship_tag
        common_content = ", ".join(
            (
                *deterministic_count_tags,
                *(("duo",) if character_count == 2 else ()),
                *((relationship_tag,) if relationship_tag else ()),
                *filtered_common_tags,
            )
        )
        low_cfg_harness = bool(prompt_config.get("low_cfg_harness_enabled", False))
        constraint_raw = ""
        constraint_plan = parse_constraint_plan("")
        if low_cfg_harness:
            # DEFERRED(Turbo wardrobe authority): constraint-plan tags are not
            # post-filtered through the ordinary LLM1/WardrobeAuthority graph.
            # Keep this compatibility behavior until Turbo is deliberately
            # redesigned and covered by dedicated ownership/conflict tests.
            try:
                constraint_raw = await self._generate_constraint_plan_with_llm(
                    provider_id=provider_id,
                    plan_prompt=build_constraint_plan_prompt(
                        user_prompt=prompt,
                        llm_content=common_content,
                    ),
                )
                constraint_plan = parse_constraint_plan(constraint_raw)
            except Exception as exc:
                self.logger.warning(
                    "[comfyui_agent] multi-person constraint planner failed: %s",
                    exc,
                )

        normalized_interactions: list[str] = []
        for interaction in plan.interactions:
            normalized = interaction
            replacements = sorted(
                (
                    (name, aliases[index])
                    for index, names in enumerate(character_entity_names)
                    for name in names
                    if name.lower() != aliases[index].lower()
                ),
                key=lambda item: len(item[0]),
                reverse=True,
            )
            for name, alias in replacements:
                if any(ord(char) > 127 for char in name):
                    normalized = normalized.replace(name, f" {alias} ")
                else:
                    normalized = re.sub(
                        rf"(?<![\w]){re.escape(name)}(?![\w])",
                        alias,
                        normalized,
                        flags=re.IGNORECASE,
                    )
            normalized = re.sub(r"\s+", " ", normalized).strip()
            normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
            normalized_interactions.append(normalized)

        normalized_aliases = {
            alias.upper()
            for interaction in normalized_interactions
            for alias in re.findall(
                r"\bCharacter\s+[A-D]\b",
                interaction,
                flags=re.IGNORECASE,
            )
        }
        allowed_aliases = {alias.upper() for alias in aliases[: len(plan.characters)]}
        if normalized_interactions and (
            not normalized_aliases.issubset(allowed_aliases)
            or len(normalized_aliases) < 2
        ):
            summary.update(
                {
                    "multi_person_mode": True,
                    "multi_person_plan_failed": True,
                    "multi_person_error": "invalid_interaction_aliases",
                    "multi_person_planner_retry_count": planner_retry_count,
                }
            )
            return None

        display_interactions: list[str] = []
        for interaction in normalized_interactions:
            displayed = interaction
            for alias, role in zip(
                aliases[: len(plan.characters)], character_roles, strict=True
            ):
                displayed = re.sub(
                    rf"\b{re.escape(alias)}\b",
                    f"the {role}",
                    displayed,
                    flags=re.IGNORECASE,
                )
            display_interactions.append(displayed)

        scene_guard = "The composition shows one shared continuous moment."
        relative_position = ""
        if spatial_mode == "explicit_positions" and len(plan.characters) == 2:
            slot_aliases = {
                character.slot: f"the {character_roles[index]}"
                for index, character in enumerate(plan.characters)
            }
            if {"left", "right"}.issubset(slot_aliases):
                relative_position = (
                    f"{slot_aliases['left']} stands immediately beside "
                    f"{slot_aliases['right']}, to {slot_aliases['right']}'s left, "
                    "while both remain in the same central group."
                )
            elif {"foreground", "background"}.issubset(slot_aliases):
                relative_position = (
                    f"{slot_aliases['foreground']} stands slightly in front of "
                    f"{slot_aliases['background']} while both remain together in "
                    "the same continuous scene."
                )
        narrative_blocks = tuple(
            block
            for block in (
                *character_blocks,
                *display_interactions,
                relative_position,
                scene_guard,
            )
            if block
        )
        built = build_final_prompt(
            user_prompt=prompt,
            llm_content=common_content,
            config=prompt_config,
            constraint_plan=constraint_plan,
            narrative_blocks=narrative_blocks,
            suppress_fixed_character=True,
            force_multi_character=True,
        )
        content_tag_count = len(split_tags(built.content_tags))
        summary.update(
            {
                "multi_person_mode": True,
                "multi_person_plan_failed": False,
                "multi_person_planner_retry_count": planner_retry_count,
                "planned_character_count": len(plan.characters),
                "resolved_character_count": resolved_count,
                "fixed_character_count": fixed_character_count,
                "danbooru_resolved_count": danbooru_resolved_count,
                "unresolved_character_count": unresolved_character_count,
                "character_resolution_statuses": character_resolution_statuses,
                "named_character_detected": bool(plan.characters),
                "character_slots": character_slots,
                "character_roles": character_roles,
                "interaction_count": len(plan.interactions),
                "relationship_tag": relationship_tag,
                "planned_relationship_tag": planned_relationship_tag,
                "relationship_tag_suppressed": relationship_tag_suppressed,
                "emphasized_anchor_count": emphasized_anchor_count,
                "grouped_contact": grouped_contact,
                "spatial_mode": spatial_mode,
                "background_mode": effective_background_mode,
                "background_mode_source": (
                    "user_prompt_override"
                    if background_overridden
                    else "llm_generated_scene" if generated_scene_selected else "llm_plan"
                ),
                "explicit_position_requested": explicit_position_requested,
                "interaction_aliases_normalized": (
                    tuple(normalized_interactions) != plan.interactions
                ),
                "composition_source": "deterministic",
                "hybrid_prompt": True,
                "raw_mode": False,
                "deep_thinking": use_deep_thinking,
                "fixed_character": bool(used_fixed_names),
                "fixed_character_name": ", ".join(sorted(used_fixed_names)),
                "sensual_mode": wants_sensual_mode(prompt, prompt_config),
                "default_style": built.used_default_style,
                "low_cfg_harness": low_cfg_harness,
                "constraint_mode": built.constraint_mode,
                "weighted_style_tags": list(built.weighted_style_tags),
                "constraint_tags": list(built.constraint_tags),
                "removed_constraint_tags": list(built.removed_constraint_tags),
                "constraint_reason": built.constraint_reason,
                "llm_failed": False,
                "llm_error": "",
                "llm_content_tag_count": content_tag_count,
                "removed_content_tag_count": max(
                    0,
                    len(split_tags(common_content)) - content_tag_count,
                ),
                "llm_content_chars": len(built.content_tags),
                "final_prompt_chars": len(built.final_prompt),
                "final_prompt_head": self._shorten(built.final_prompt, 600),
            }
        )
        if self._bool("debug_prompt_enabled", False):
            summary.update(
                {
                    "multi_person_plan_prompt": plan_prompt,
                    "multi_person_plan": raw_plan,
                    "constraint_plan": constraint_raw,
                    "final_prompt": built.final_prompt,
                }
            )
        return PromptPipelineResult(built.final_prompt, summary)

    async def build(
        self,
        event: Any,
        user_prompt: str,
        mode: str = "txt2img",
        *,
        multi_person: bool = False,
        original_user_prompt: str = "",
        canvas_size: tuple[int, int] | None = None,
        canvas_size_explicit: bool = False,
    ) -> PromptPipelineResult:
        """Build the final prompt and summary for one generation request.

        Args:
            event: AstrBot message event for provider and per-chat config lookup.
            user_prompt: User prompt after reference-image augmentation.
            mode: Generation mode, such as `txt2img` or `img2img`.
            multi_person: Whether `/anm 多人` requested structured planning.
            original_user_prompt: User text before reference-context augmentation.
            canvas_size: This request's configured or explicit width and height.
            canvas_size_explicit: Whether the user selected this size for the request.

        Returns:
            Final prompt plus a serializable summary dict.
        """
        original_prompt = str(user_prompt or "").strip()
        nai_r_mode, original_prompt = strip_nai_character_switch(original_prompt)
        _, background_intent_prompt = strip_nai_character_switch(
            str(original_user_prompt or original_prompt).strip()
        )
        legacy_creative_flag_re = re.compile(
            r"(?<!\S)--(?:自由发挥|自由拓展|创意拓展|创意扩展|creative)"
            r"(?=$|\s|[,，;；:：])",
            re.IGNORECASE,
        )
        prompt = legacy_creative_flag_re.sub(" ", original_prompt).strip()
        prompt = re.sub(r"^[\s,，;；:：]+|[\s,，;；:：]+$", "", prompt)
        prompt = re.sub(r"([,，;；])\s*[,，;；]+", r"\1", prompt)
        prompt = re.sub(r"\s+", " ", prompt)
        requested_outfit_mode = requested_wardrobe_mode(prompt)
        casual_life_requested = requested_outfit_mode == "casual_profile"
        creative_private_requested = requested_outfit_mode == "creative_fallback"
        summary: dict[str, Any] = {
            "prompt_optimize_enabled": self._bool("prompt_optimize_enabled", True),
            "mode": mode,
            "requested_outfit_mode": requested_outfit_mode,
            "multi_person_mode": bool(multi_person),
            "nai_r_mode": nai_r_mode,
            "original_prompt_head": self._shorten(original_prompt, 600),
        }
        preset_config = apply_config_preset(dict(self.config))
        artist_settings = getattr(self, "artist_session_settings", None)
        if artist_settings is not None:
            session = str(getattr(event, "unified_msg_origin", "") or "").strip()
            selected = artist_settings.get(session)
            if selected is not None:
                preset_config["active_artist_preset"] = selected
        preset_index, prompt, switch_error = extract_artist_preset_switch(
            prompt, preset_config
        )
        if switch_error is not None:
            preset_index = None
            summary["artist_preset_switch_error"] = switch_error
        if not self._bool("prompt_optimize_enabled", True):
            if nai_r_mode:
                summary["skipped_reason"] = "nai_character_mode_requires_optimization"
                return PromptPipelineResult("", summary)
            summary.update(
                {
                    "skipped_reason": "prompt_optimize_disabled",
                    "final_prompt_head": self._shorten(prompt, 600),
                    "final_prompt_chars": len(prompt),
                }
            )
            return PromptPipelineResult(prompt, summary)
        raw_mode, raw_prompt = strip_raw_prefix(prompt)
        if raw_mode:
            if nai_r_mode:
                summary["skipped_reason"] = "nai_character_mode_requires_optimization"
                return PromptPipelineResult("", summary)
            self.logger.info("[comfyui_agent] prompt builder skipped: raw tags mode")
            summary.update(
                {
                    "raw_mode": True,
                    "skipped_reason": "raw_tags_mode",
                    "final_prompt_head": self._shorten(raw_prompt, 600),
                    "final_prompt_chars": len(raw_prompt),
                }
            )
            return PromptPipelineResult(raw_prompt, summary)

        prompt_config = preset_config
        if preset_index is not None:
            prompt_config["_artist_preset_index"] = preset_index
        local_character_hints = mentioned_fixed_characters(prompt, prompt_config)
        profile_tags_getter = getattr(
            self._danbooru_resolver, "required_profile_tags_for_prompt", None
        )
        required_profile_tags = (
            tuple(profile_tags_getter(prompt)) if profile_tags_getter else ()
        )
        profile_hints_getter = getattr(
            self._danbooru_resolver, "profile_hints_for_prompt", None
        )
        profile_hints = (
            dict(profile_hints_getter(prompt)) if profile_hints_getter else {}
        )
        profile_outfit_rule = "\n".join(
            f"{name}: {hint}" for name, hint in profile_hints.items()
        )
        fixed_character = selected_fixed_character(prompt, prompt_config)
        fixed_character_name = fixed_character[0] if fixed_character else ""
        use_fixed_character = fixed_character is not None
        use_sensual_mode = wants_sensual_mode(prompt, prompt_config)
        unspecified_wardrobe_policy = self._str(
            "unspecified_wardrobe_policy", "scene_adaptive"
        ).strip().lower()
        raw_scene_markers = self.config.get(
            "scene_adaptive_wardrobe_markers",
            list(DEFAULT_SCENE_ADAPTIVE_WARDROBE_MARKERS),
        )
        if raw_scene_markers is None:
            configured_scene_markers = ()
        elif isinstance(raw_scene_markers, str):
            configured_scene_markers = tuple(
                item.strip()
                for item in re.split(r"[,，\n]", raw_scene_markers)
                if item.strip()
            )
        else:
            configured_scene_markers = tuple(
                str(item).strip() for item in raw_scene_markers if str(item).strip()
            )
        matched_wardrobe_scene_marker = scene_adaptive_wardrobe_marker(
            prompt, configured_scene_markers
        )
        implicit_wardrobe_mode, wardrobe_source = resolve_unspecified_wardrobe_mode(
            unspecified_wardrobe_policy,
            sensual_mode=use_sensual_mode,
            scene_marker=matched_wardrobe_scene_marker,
        )
        explicit_wardrobe_evidence = False
        summary.update(
            {
                "unspecified_wardrobe_policy": unspecified_wardrobe_policy,
                "scene_adaptive_wardrobe_marker": matched_wardrobe_scene_marker,
            }
        )
        keyword_rules = match_keyword_prompt_rules(
            background_intent_prompt, prompt_config
        )
        summary["keyword_prompt_rule_markers"] = [
            rule.marker for rule in keyword_rules
        ]
        outfit_plan = detect_outfit_transfer(
            prompt,
            fixed_character_name,
            tuple(local_character_hints),
        )
        # Ordinary "A cosplays/wears C" text belongs to the per-character LLM1
        # graph and local profile resolver. The legacy transfer path is only for
        # explicit reference-image or search workflows; running both paths makes
        # one detected target masquerade as the sole fixed subject in group art.
        if outfit_plan.enabled and not (
            outfit_plan.source_from_reference or outfit_plan.source_from_search
        ):
            outfit_plan = OutfitTransferPlan(directive_prompt=outfit_plan.directive_prompt)
        outfit_plan = bind_explicit_outfit_patch_target(
            outfit_plan,
            user_prompt=prompt,
            known_character_names=tuple(local_character_hints),
            fallback_character=fixed_character_name,
        )

        provider_id = await self._current_chat_provider_id(event)
        if not provider_id:
            self.logger.warning(
                "[comfyui_agent] prompt builder has no provider; aborting optimized generation"
            )
            summary.update(
                {
                    "skipped_reason": "no_chat_provider",
                    "llm_failed": True,
                    "llm_error": "no_chat_provider",
                    "final_prompt_head": "",
                    "final_prompt_chars": 0,
                }
            )
            return PromptPipelineResult("", summary)

        if multi_person and nai_r_mode:
            summary["skipped_reason"] = "nai_character_mode_incompatible_multi_person"
            return PromptPipelineResult("", summary)
        if multi_person:
            # `/anm 多人` is the original independent compatibility route.  It
            # plans anonymous Character A/B/C/D visual blocks directly and must
            # not inherit the newer LLM1 character/wardrobe graph.  Mixing the
            # two planners changes its identity model and makes a failed name
            # join capable of leaking unscoped clothing into a character block.
            multi_research_plan = self._researcher.plan(prompt)
            multi_result = await self._build_multi_person_prompt(
                provider_id=provider_id,
                prompt=prompt,
                prompt_config=prompt_config,
                use_deep_thinking=multi_research_plan.use_deep_thinking,
                summary=summary,
                original_user_prompt=background_intent_prompt,
            )
            if multi_result is not None:
                return multi_result
            summary.setdefault("multi_person_mode", True)
            summary.setdefault("multi_person_plan_failed", True)
            summary.setdefault("multi_person_error", "invalid_plan")
            summary.update(
                {
                    "skipped_reason": "multi_person_plan_failed",
                    "final_prompt_head": "",
                    "final_prompt_chars": 0,
                }
            )
            return PromptPipelineResult("", summary)

        semantic_result = SemanticLookupResult()
        semantic_plan_raw = ""
        semantic_plan_initial_raw = ""
        semantic_plan_validation_errors: tuple[str, ...] = ()
        semantic_plan_attempt_count = 0
        semantic_anchors: tuple[SemanticAnchor, ...] = ()
        semantic_character_plans: tuple[SemanticCharacterPlan, ...] = ()
        semantic_appearance_changes: tuple[SemanticAppearanceChange, ...] = ()
        semantic_outfit_directives: tuple[SemanticOutfitDirective, ...] = ()
        semantic_outfit_patches: tuple[UserOutfitPatch, ...] = ()
        cached_source_getter = getattr(
            self._danbooru_resolver, "cached_outfit_source", None
        )
        cached_source_result = None
        if (
            outfit_plan.enabled
            and outfit_plan.source_subject
            and callable(cached_source_getter)
        ):
            cached_result = cached_source_getter(outfit_plan.source_subject)
            if cached_result is not None:
                cached_source_result = cached_result
        cached_named_getter = getattr(
            self._danbooru_resolver, "cached_named_outfits_for_prompt", None
        )
        cached_named_result = (
            cached_named_getter(prompt) if callable(cached_named_getter) else None
        )
        cached_term_getter = getattr(
            self._danbooru_resolver, "cached_term_mappings_for_prompt", None
        )
        cached_term_result = (
            cached_term_getter(prompt) if callable(cached_term_getter) else None
        )
        explicit_term_tags = tuple(
            dict.fromkeys(
                getattr(cached_term_result, "confirmed_tags", ())
                if cached_term_result is not None
                else ()
            )
        )
        cached_profile_getter = getattr(
            self._danbooru_resolver, "cached_outfit_profiles_for_prompt", None
        )
        cached_profile_result = (
            cached_profile_getter(prompt)
            if callable(cached_profile_getter)
            else None
        )
        configured_character_anchor_getter = getattr(
            self._danbooru_resolver,
            "configured_character_anchors_for_prompt",
            None,
        )
        configured_character_anchors = (
            tuple(configured_character_anchor_getter(prompt))
            if callable(configured_character_anchor_getter)
            else ()
        )
        bound_wardrobe_result = None
        semantic_result = merge_semantic_results(
            cached_source_result,
            cached_profile_result,
            cached_named_result,
            cached_term_result,
        )
        semantic_available = getattr(
            self._danbooru_resolver, "semantic_lookup_available", None
        )
        semantic_resolve = getattr(
            self._danbooru_resolver, "resolve_semantic_anchors", None
        )
        semantic_lookup_ready = bool(
            callable(semantic_available)
            and semantic_available()
            and callable(semantic_resolve)
        )
        # Wardrobe ownership planning only needs the chat provider. Danbooru lookup
        # enriches/validates tags when available, but must never gate the first LLM:
        # otherwise a temporary CLI/tool outage silently falls back to cached clothes.
        if callable(semantic_available) and callable(semantic_resolve):
            try:
                semantic_plan_raw = await self._generate_semantic_plan_with_llm(
                    provider_id=provider_id,
                    user_prompt=prompt,
                    max_tokens=7200 if nai_r_mode else None,
                )
                semantic_plan_attempt_count = 1
                semantic_plan_initial_raw = semantic_plan_raw
                semantic_plan_validation_errors = semantic_plan_validation_issues(
                    semantic_plan_raw, prompt
                )
                if semantic_plan_validation_errors:
                    semantic_plan_attempt_count = 2
                    try:
                        repaired_plan = await self._repair_semantic_plan_with_llm(
                            provider_id=provider_id,
                            user_prompt=prompt,
                            previous_raw=semantic_plan_raw,
                            issues=semantic_plan_validation_errors,
                            max_tokens=7200 if nai_r_mode else None,
                        )
                    except Exception as repair_exc:
                        # Keep the usable, source-grounded subset of the first
                        # response. A failed repair must not discard it.
                        self.logger.warning(
                            "[comfyui_agent] semantic planner repair failed; "
                            "keeping initial plan: %s",
                            repair_exc,
                        )
                    else:
                        repaired_errors = semantic_plan_validation_issues(
                            repaired_plan, prompt
                        )
                        # Never replace a partially usable first response with
                        # a merely 'less invalid' second response.
                        if not repaired_errors:
                            semantic_plan_raw = repaired_plan
                            semantic_plan_validation_errors = ()
                planner_anchors = parse_semantic_plan(semantic_plan_raw, prompt)
                planner_character_aliases = parse_semantic_character_aliases(
                    semantic_plan_raw, prompt
                )
                semantic_anchors = prefer_configured_character_anchors(
                    configured_character_anchors,
                    (
                        *extract_parenthesized_copyright_aliases(prompt),
                        *bind_parenthesized_character_aliases(
                            planner_anchors, prompt, planner_character_aliases,
                        ),
                    ),
                )
                semantic_anchors = add_explicit_cosplay_source_anchors(
                    semantic_anchors, prompt
                )
                cached_complete_anchor_ids: set[str] = set()
                if cached_named_result is not None:
                    for cached_anchor in cached_named_result.anchors:
                        cached_source_key = re.sub(
                            r"\s+", " ", cached_anchor.source_text.strip().lower()
                        )
                        if any(
                            anchor.role in {"outfit", "clothing"}
                            and re.sub(
                                r"\s+", " ", anchor.source_text.strip().lower()
                            )
                            == cached_source_key
                            and bool(
                                set(candidate.lower() for candidate in anchor.candidates)
                                & set(
                                    candidate.lower()
                                    for candidate in cached_anchor.candidates
                                )
                            )
                            for anchor in semantic_anchors
                        ):
                            continue
                        semantic_anchors = (*semantic_anchors, cached_anchor)
                semantic_anchors = annotate_requested_profile_variants(
                    semantic_anchors, prompt
                )
                prefer_cached_characters = getattr(
                    self._danbooru_resolver,
                    "prefer_cached_character_anchors",
                    None,
                )
                if not callable(prefer_cached_characters):
                    prefer_cached_characters = getattr(
                        self._danbooru_resolver,
                        "prefer_cached_outfit_source_anchors",
                        None,
                    )
                if callable(prefer_cached_characters):
                    semantic_anchors = tuple(
                        prefer_cached_characters(semantic_anchors)
                    )
                semantic_outfit_directives = parse_semantic_outfit_directives(
                    semantic_plan_raw, prompt
                )
                semantic_appearance_changes = parse_semantic_appearance_changes(
                    semantic_plan_raw, prompt
                )
                semantic_character_plans = parse_semantic_character_plans(
                    semantic_plan_raw, prompt, semantic_anchors
                )
                semantic_character_plans = repair_explicit_cosplay_plans(
                    semantic_character_plans, semantic_anchors, prompt
                )
                semantic_character_plans = add_host_outfit_changes_to_plans(
                    semantic_character_plans, semantic_anchors, prompt
                )
                semantic_character_plans = repair_single_target_cached_named_outfit_plan(
                    semantic_character_plans,
                    semantic_anchors,
                    cached_named_result,
                    prompt,
                )
                semantic_character_plans = preserve_unresolved_worn_clothing_plans(
                    semantic_character_plans, semantic_anchors, prompt
                )
                bind_wardrobes = getattr(self._danbooru_resolver, "bind_character_wardrobes", None)
                if callable(bind_wardrobes):
                    if semantic_lookup_ready:
                        semantic_anchors = await self._danbooru_resolver.classify_literal_targets(semantic_anchors)
                    semantic_anchors, semantic_character_plans = bind_wardrobes(
                        semantic_anchors, semantic_character_plans
                    )
                    bound_wardrobe_result = self._danbooru_resolver.cached_bound_wardrobes(semantic_anchors)
                    semantic_result = merge_semantic_results(semantic_result, bound_wardrobe_result)
                if self._bool("debug_prompt_enabled", False):
                    self.logger.info(
                        "[comfyui_agent] semantic planner parsed anchors:\n%r",
                        semantic_anchors,
                    )
                    self.logger.info(
                        "[comfyui_agent] semantic planner parsed character plans:\n%r",
                        semantic_character_plans,
                    )
                if self._bool("debug_prompt_enabled", False):
                    self.logger.info(
                        "[comfyui_agent] semantic planner accepted character plans "
                        "without host-side relationship recovery:\n%r",
                        semantic_character_plans,
                    )
                explicit_wardrobe_evidence = has_explicit_wardrobe_evidence(
                    requested_mode=requested_outfit_mode,
                    outfit_transfer_enabled=outfit_plan.enabled,
                    anchors=semantic_anchors,
                    plans=semantic_character_plans,
                ) or any(
                    explicit_worn_clothing_phrases(prompt, anchor.source_text)
                    for anchor in semantic_anchors
                    if anchor.role == "target_character"
                )
                if explicit_wardrobe_evidence:
                    if requested_outfit_mode == "default_profile":
                        wardrobe_source = "explicit_default_profile"
                    elif requested_outfit_mode == "casual_profile":
                        wardrobe_source = "explicit_casual_profile"
                    elif requested_outfit_mode in {
                        "summer_profile", "winter_profile", "stage_profile"
                    }:
                        wardrobe_source = f"explicit_{requested_outfit_mode}"
                    elif requested_outfit_mode == "creative_fallback":
                        wardrobe_source = "explicit_creative"
                    elif outfit_plan.enabled or any(
                        plan.wardrobe.kind in {"named_outfit", "outfit_source"}
                        for plan in semantic_character_plans
                    ):
                        wardrobe_source = "explicit_named_outfit"
                    else:
                        wardrobe_source = "explicit_clothing"
                semantic_character_plans = apply_character_wardrobe_baselines(
                    semantic_character_plans,
                    semantic_anchors,
                    implicit_mode=implicit_wardrobe_mode,
                    user_prompt=prompt,
                    create_missing=not explicit_wardrobe_evidence,
                )
                if semantic_character_plans:
                    semantic_outfit_directives = tuple(
                        directive
                        for character_plan in semantic_character_plans
                        for directive in character_plan.directives
                    )
                semantic_outfit_patches = tuple(
                    UserOutfitPatch(
                        subject="semantic_target",
                        operation=(
                            "replace"
                            if directive.operation == "replace_color"
                            else directive.operation
                        ),
                        slot=",".join(directive.slots),
                        value=directive.color,
                        evidence=directive.source_text,
                    )
                    for directive in semantic_outfit_directives
                )
                if cached_named_result is not None:
                    cached_named_keys = {
                        tag.lower() for tag in cached_named_result.named_outfit_tags
                    }
                    cached_named_profiles_by_anchor = {
                        anchor_id.lower(): tuple(profile_tags)
                        for anchor_id, _tag, profile_tags, _variant
                        in cached_named_result.anchor_outfit_profiles
                        if profile_tags
                    }
                    cached_named_profiles_by_tag = {
                        tag.lower(): tuple(profile_tags)
                        for _anchor_id, tag, profile_tags, _variant
                        in cached_named_result.anchor_outfit_profiles
                        if tag and profile_tags
                    }

                    def cached_named_anchor_is_complete(
                        anchor: SemanticAnchor,
                    ) -> bool:
                        """Skip lookup when the generic named set has components.

                        Named sets no longer live in the character-outfit cache, so
                        asking ``cached_outfit_source`` about their display alias
                        always returns ``None``.  Completeness belongs to the
                        named-set profile row itself.
                        """
                        candidates = tuple(
                            candidate.lower() for candidate in anchor.candidates
                        )
                        if not any(
                            candidate in cached_named_keys
                            for candidate in candidates
                        ):
                            return False
                        return bool(
                            cached_named_profiles_by_anchor.get(
                                anchor.anchor_id.lower()
                            )
                            or any(
                                cached_named_profiles_by_tag.get(candidate)
                                for candidate in candidates
                            )
                        )

                    cached_complete_anchor_ids |= {
                        anchor.anchor_id
                        for anchor in semantic_anchors
                        if anchor.role in {"outfit", "clothing"}
                        and cached_named_anchor_is_complete(anchor)
                    }
                if bound_wardrobe_result is not None:
                    cached_complete_anchor_ids.update(
                        anchor_id for anchor_id, _tag, _tags, _variant
                        in bound_wardrobe_result.anchor_outfit_profiles
                    )
                if outfit_plan.enabled and outfit_plan.source_subject:
                    source_text = outfit_plan.source_subject.strip()
                    source_candidates: list[str] = []
                    if re.fullmatch(
                        r"[A-Za-z0-9_.'():!\- ]{2,80}", source_text
                    ):
                        source_candidates.append(
                            re.sub(r"\s+", "_", source_text.lower())
                        )
                    source_key = source_text.lower()
                    for anchor in semantic_anchors:
                        if source_key not in (
                            anchor.source_text + " " + anchor.description
                        ).lower():
                            continue
                        for candidate in anchor.candidates:
                            if candidate not in source_candidates:
                                source_candidates.append(candidate)
                    if source_candidates:
                        # Host-side transfer detection knows which person is the
                        # donor. Replace overlapping planner roles with one explicit
                        # outfit_source anchor so the donor cannot also become a
                        # visible character or an unscoped clothing set.
                        semantic_anchors = tuple(
                            anchor
                            for anchor in semantic_anchors
                            if not (
                                source_key
                                in (anchor.source_text + " " + anchor.description).lower()
                                and anchor.role
                                in {"outfit_source", "outfit", "clothing"}
                            )
                        )
                        refresh_getter = getattr(
                            self._danbooru_resolver,
                            "outfit_source_refresh_needed",
                            None,
                        )
                        refresh_source = cached_source_result is None or (
                            callable(refresh_getter)
                            and refresh_getter(source_text)
                        )
                        if refresh_source:
                            semantic_anchors = (
                                *semantic_anchors,
                                SemanticAnchor(
                                    anchor_id="host_outfit_source",
                                    role="outfit_source",
                                    group="character",
                                    source_text=source_text,
                                    description=f"{source_text} costume source",
                                    candidates=tuple(source_candidates[:3]),
                                ),
                            )
                if semantic_lookup_ready:
                    lookup_anchors = tuple(
                        anchor
                        for anchor in semantic_anchors
                        if anchor.anchor_id not in cached_complete_anchor_ids
                    )
                    resolved_semantic = await semantic_resolve(lookup_anchors)
                    refined_anchors = await self._refine_unresolved_semantic_characters(
                        provider_id=provider_id,
                        user_prompt=prompt,
                        anchors=semantic_anchors,
                        result=resolved_semantic,
                    )
                    if refined_anchors != semantic_anchors:
                        semantic_anchors = refined_anchors
                        lookup_anchors = tuple(
                            anchor
                            for anchor in semantic_anchors
                            if anchor.anchor_id not in cached_complete_anchor_ids
                        )
                        resolved_semantic = await semantic_resolve(lookup_anchors)
                    semantic_result = merge_semantic_results(
                        cached_source_result,
                        cached_profile_result,
                        cached_named_result,
                        cached_term_result,
                        resolved_semantic,
                        bound_wardrobe_result,
                    )
            except Exception as exc:
                self.logger.warning(
                    "[comfyui_agent] semantic wardrobe planning/lookup failed: %s",
                    exc,
                )
        semantic_fallback_used = False
        if has_host_wardrobe_cue(prompt) and not semantic_character_plans:
            if not any(
                anchor.role == "target_character" for anchor in semantic_anchors
            ):
                fallback_targets = fallback_target_anchors_from_semantic_evidence(
                    prompt,
                    configured_character_anchors,
                    semantic_result,
                )
                semantic_anchors = tuple(
                    dict.fromkeys((*semantic_anchors, *fallback_targets))
                )
            if any(
                anchor.role == "target_character" for anchor in semantic_anchors
            ):
                explicit_wardrobe_evidence = True
                wardrobe_source = "explicit_but_unresolved"
                semantic_fallback_used = True
        semantic_result = bind_confirmed_character_anchors(semantic_result, semantic_anchors)
        semantic_character_tags = confirmed_semantic_character_tags(semantic_result)
        confirmed_anchor_tags = dict(semantic_result.anchor_tags)
        summary["confirmed_character_bindings"] = [
            {"source": anchor.source_text, "canonical_tag": confirmed_anchor_tags[anchor.anchor_id]}
            for anchor in semantic_anchors
            if anchor.role == "target_character" and anchor.anchor_id in confirmed_anchor_tags
        ]
        semantic_required_tags: tuple[str, ...] = ()
        semantic_context = semantic_result.prompt_context()
        summary.update(
            {
                "danbooru_semantic_status": semantic_result.status,
                "semantic_plan_attempt_count": semantic_plan_attempt_count,
                "semantic_plan_validation_errors": list(
                    semantic_plan_validation_errors
                ),
                "semantic_wardrobe_fallback_used": semantic_fallback_used,
                "danbooru_semantic_confirmed_tags": list(
                    semantic_result.confirmed_tags
                ),
                "danbooru_explicit_term_tags": list(explicit_term_tags),
                "danbooru_semantic_outfit_sources": list(
                    semantic_result.outfit_source_tags
                ),
                "danbooru_semantic_outfit_tags": list(
                    tuple(
                        dict.fromkeys(
                            tag
                            for _alias, _source_tag, tags, _qualifier
                            in semantic_result.source_outfit_profiles
                            for tag in tags
                        )
                    )
                    or semantic_result.outfit_profile_tags
                ),
                "danbooru_semantic_appearance_tags": list(
                    semantic_result.appearance_profile_tags
                ),
                "danbooru_semantic_character_appearance_profiles": [
                    {
                        "aliases": list(aliases),
                        "source_tag": source_tag,
                        "appearance_tags": list(tags),
                    }
                    for aliases, source_tag, tags
                    in semantic_result.character_appearance_profiles
                ],
                "danbooru_semantic_source_outfit_profiles": [
                    {
                        "alias": alias,
                        "source_tag": source_tag,
                        "qualifier": qualifier,
                        "outfit_tags": list(tags),
                    }
                    for alias, source_tag, tags, qualifier
                    in semantic_result.source_outfit_profiles
                ],
                "danbooru_semantic_named_outfit_tags": list(
                    semantic_result.named_outfit_tags
                ),
                "danbooru_semantic_missing": list(
                    semantic_result.missing_descriptions
                ),
                "danbooru_semantic_candidates": list(
                    semantic_result.candidate_tags
                ),
            }
        )

        required_core_tags = (
            self._danbooru_resolver.required_core_tags_for_prompt(prompt)
            if not use_fixed_character
            else ()
        )
        required_count_tags: tuple[str, ...] = ()
        self.logger.info(
            "[comfyui_agent] prompt builder input fixed_characters=%s sensual=%s required_core_tags=%s prompt=%s",
            ",".join(local_character_hints) or "none",
            use_sensual_mode,
            ",".join(required_core_tags) or "none",
            prompt[:180],
        )
        research_plan = self._researcher.plan(prompt)
        self.logger.info(
            "[comfyui_agent] prompt strategy web_search=%s deep_thinking=%s search_reason=%s thinking_reason=%s",
            research_plan.use_web_search,
            research_plan.use_deep_thinking,
            research_plan.search_reason or "none",
            research_plan.thinking_reason or "none",
        )
        search_query_prompt = preferred_search_prompt(outfit_plan, prompt)
        search_context = (
            await self._researcher.search_context(
                event,
                prompt,
                search_query_prompt=search_query_prompt,
            )
            if research_plan.use_web_search
            else ""
        )
        outfit_summary = ""
        outfit_summary_source = ""
        reference_tag_text = (
            extract_reference_tag_text(prompt) if outfit_plan.enabled else ""
        )
        # Outfit evidence is selected from most request-specific to least: tags
        # embedded with this reference image, a donor-scoped local profile, the
        # legacy request-wide profile, then an LLM summary of search context. The
        # latter starts as soft evidence: filtering may retain generic garments,
        # but it never certifies donor identity or a named outfit set.
        if reference_tag_text:
            outfit_summary = filter_outfit_tags(reference_tag_text, max_tags=42)
            if outfit_summary:
                outfit_summary_source = "reference_filter"
        source_specific_outfit_tags = semantic_result.outfit_tags_for_source(
            outfit_plan.source_subject
        )
        if not outfit_summary and source_specific_outfit_tags:
            outfit_summary = ", ".join(source_specific_outfit_tags)
            outfit_summary_source = "danbooru_source_posts"
        elif not outfit_summary and semantic_result.outfit_profile_tags:
            outfit_summary = ", ".join(semantic_result.outfit_profile_tags)
            outfit_summary_source = "danbooru_source_posts"
        if not outfit_summary and outfit_plan.enabled and search_context:
            summary_prompt = build_outfit_summary_prompt(
                outfit_plan,
                original_prompt=prompt,
                source_context=search_context or semantic_context,
            )
            try:
                raw_outfit_summary = await self._generate_outfit_summary_with_llm(
                    provider_id=provider_id,
                    summary_prompt=summary_prompt,
                    use_deep_thinking=research_plan.use_deep_thinking,
                )
                outfit_summary = filter_outfit_tags(raw_outfit_summary, max_tags=48)
                if outfit_summary:
                    outfit_summary_source = (
                        "search_summary"
                        if search_context
                        else "validated_source_llm_summary"
                    )
                if self._bool("debug_prompt_enabled", False):
                    self.logger.info(
                        "[comfyui_agent] outfit summary LLM output:\n%s",
                        raw_outfit_summary,
                    )
            except Exception as exc:
                self.logger.warning(
                    "[comfyui_agent] outfit summary build failed: %s", exc
                )
        if (
            outfit_summary
            and outfit_plan.source_subject
            and semantic_result.outfit_source_tags
            and not semantic_result.source_outfit_profiles
            and outfit_summary_source == "danbooru_source_posts"
        ):
            remember_outfit = getattr(
                self._danbooru_resolver, "remember_outfit_summary", None
            )
            if callable(remember_outfit):
                remember_outfit(
                    outfit_plan.source_subject,
                    semantic_result.outfit_source_tags,
                    tuple(split_tags(outfit_summary)),
                )
        effective_outfit = build_effective_outfit_plan(
            outfit_plan,
            user_prompt=prompt,
            base_tags=tuple(split_tags(outfit_summary)),
            known_character_names=tuple(local_character_hints),
            # Target-bound semantic patches are applied through character plans.
            # When LLM1 abstains from a relationship, do not reinterpret those
            # patches as request-global mutations; LLM2 receives the evidence.
            semantic_patches=(),
        )
        if effective_outfit.modified:
            outfit_summary = ", ".join(effective_outfit.effective_tags)
        scoped_profile_tags = tuple(
            dict.fromkeys(
                tag
                for _alias, _source_tag, tags, _qualifier
                in semantic_result.source_outfit_profiles
                for tag in tags
            )
        )
        profile_tags_for_request = (
            scoped_profile_tags or semantic_result.outfit_profile_tags
        )
        character_effective_outfits = build_character_effective_outfits(
            semantic_anchors,
            semantic_character_plans,
            semantic_result,
            user_prompt=prompt,
            known_character_names=tuple(local_character_hints),
        )
        character_effective_outfits = complete_character_wardrobe_states(
            character_effective_outfits,
            semantic_anchors,
            semantic_result,
            explicit_wardrobe_evidence=explicit_wardrobe_evidence,
            requested_mode=requested_outfit_mode,
            user_prompt=prompt,
        )
        if self._bool("debug_prompt_enabled", False):
            self.logger.info(
                "[comfyui_agent] per-character wardrobe resolution states:\n%r",
                tuple(
                    (
                        item.target_source_text,
                        item.resolution_state,
                        item.wardrobe_kind,
                        item.unresolved_evidence,
                    )
                    for item in character_effective_outfits
                ),
            )
        if any(
            item.resolution_state == "explicit_but_unresolved"
            for item in character_effective_outfits
        ):
            wardrobe_source = "explicit_but_unresolved"
        character_effective_outfits, missing_profile_fallback = (
            fallback_missing_unspecified_profiles(
                character_effective_outfits,
                explicit_wardrobe_evidence=explicit_wardrobe_evidence,
            )
        )
        if missing_profile_fallback:
            wardrobe_source = "missing_profile_fallback"
        wardrobe_authority = build_wardrobe_authority(
            character_effective_outfits,
            semantic_result,
            semantic_anchors,
            explicit_term_tags=explicit_term_tags,
        )
        # Once wearer-scoped plans exist, the request-wide summary is no longer an
        # authority. Keeping it alive would create a second, conflicting wardrobe
        # channel later in prompt assembly.
        if wardrobe_authority.has_character_plans and not outfit_plan.enabled:
            outfit_summary = ""
            outfit_summary_source = "suppressed_by_character_authority"
        required_profile_tags = wardrobe_authority.filter_tags(required_profile_tags)
        global_outfit_reinforcement_tags = safe_global_outfit_tags(
            character_effective_outfits
        )
        semantic_visible_outfit_tags = wardrobe_authority.filter_tags(tuple(
            tag
            for tag in profile_tags_for_request
            if tag not in effective_outfit.removed_tags
        ))
        include_source_anchor = (
            not effective_outfit.has_destructive_override
            and not any(
                item.resolution_state == "explicit_but_unresolved"
                for item in character_effective_outfits
            )
            and not any(
                item.wardrobe_kind == "outfit_source"
                for item in character_effective_outfits
            )
        )
        semantic_confirmed_tags = non_wardrobe_confirmed_tags(
            semantic_result,
            character_effective_outfits,
            include_source_anchor=include_source_anchor,
        )
        semantic_confirmed_tags = wardrobe_authority.filter_tags(
            semantic_confirmed_tags
        )
        semantic_required_tags = tuple(
            dict.fromkeys(
                (
                    *explicit_term_tags,
                    *semantic_confirmed_tags,
                    *(() if character_effective_outfits else effective_outfit.effective_tags),
                    *(() if character_effective_outfits else semantic_visible_outfit_tags),
                    *(() if character_effective_outfits else semantic_result.appearance_profile_tags),
                    *(("casual",) if casual_life_requested else ()),
                    *(() if character_effective_outfits else semantic_result.named_outfit_tags),
                )
            )
        )
        # This is the soft/validated -> hard placement boundary. Confirmed/profile
        # evidence and bounded outfit fallbacks enter only after wearer scoping and
        # explicit removals. Planner candidates and unresolved descriptions stay in
        # context for the writer and can never enter this tuple directly.
        semantic_context = semantic_result.prompt_context(
            include_outfit_source_anchor=include_source_anchor,
            effective_outfit_tags=effective_outfit.effective_tags,
            removed_outfit_tags=effective_outfit.removed_tags,
            suppress_profile_outfit_tags=False,
        )
        # Configured fixed-character prose predates learned visual profiles and
        # can therefore contain stale traits.  Once a wearer-scoped profile is
        # available, expose one consistent identity source to the writer instead
        # of asking it to arbitrate contradictory blue/grey-eye style hints.
        local_character_hints = reconcile_character_hints_with_appearance(
            local_character_hints, character_effective_outfits
        )
        summary["appearance_omissions"] = [
            {"character": item.target_source_text, "dimensions": sorted(omitted)}
            for item in character_effective_outfits
            if (omitted := _omitted_appearance_dimensions(
                _semantic_appearance_changes_for_character(
                    semantic_appearance_changes, item
                )
            ))
        ]
        if character_effective_outfits:
            context_lines = []
            visible_roster = tuple(
                dict.fromkeys(
                    tag
                    for anchor_id, tag in semantic_result.anchor_tags
                    if any(
                        anchor.anchor_id.lower() == anchor_id.lower()
                        and anchor.role == "target_character"
                        for anchor in semantic_anchors
                    )
                )
            )
            extra_hard_tags = tuple(
                tag for tag in semantic_confirmed_tags if tag not in visible_roster
            )
            if extra_hard_tags:
                context_lines.append(
                    "Other confirmed non-wardrobe hard tags: "
                    + ", ".join(extra_hard_tags)
                )
            if visible_roster:
                bindings = tuple(summary["confirmed_character_bindings"])
                context_lines.extend(
                    (
                        "confirmed visible character roster (authoritative):",
                        "Use these canonical names in Characters, Identity, Details and Nltags:",
                        *(
                            f"- {item['source']} => {item['canonical_tag']}"
                            for item in bindings
                        ),
                        *(
                            tag for tag in visible_roster
                            if tag not in {item["canonical_tag"] for item in bindings}
                        ),
                    )
                )
            appearance_plans = tuple(
                item for item in character_effective_outfits if item.appearance_tags
            )
            if appearance_plans:
                context_lines.append("Stable appearance by wearer (user edits take priority):")
                context_lines.extend(
                    f"- {item.target_source_text}: "
                    + ", ".join(
                        tag for tag in item.appearance_tags
                        if appearance_dimension(tag) not in _omitted_appearance_dimensions(
                            _semantic_appearance_changes_for_character(
                                semantic_appearance_changes, item
                            )
                        )
                    )
                    for item in appearance_plans
                )
                advisory_change_lines = []
                for item in appearance_plans:
                    changes = _semantic_appearance_changes_for_character(
                        semantic_appearance_changes, item
                    )
                    if not changes:
                        continue
                    descriptions = "; ".join(
                        change.source_text
                        + (
                            f" ({', '.join(change.dimensions)}; "
                            f"operation={change.operation})"
                            if change.dimensions
                            else ""
                        )
                        for change in changes
                    )
                    advisory_change_lines.append(
                        f"- {item.target_source_text}: {descriptions}"
                    )
                if advisory_change_lines:
                    context_lines.append(
                        "Explicit user appearance changes by wearer; never assign "
                        "one person's change to another:"
                    )
                    context_lines.extend(advisory_change_lines)
                    context_lines.append(
                        "Follow each change operation exactly: additive keeps the "
                        "compatible baseline value and also states the new visible "
                        "value; replace supersedes only that dimension; omit "
                        "means the feature is not visible and its saved color or "
                        "shape must not be mentioned anywhere."
                    )
            if semantic_result.missing_descriptions:
                context_lines.append(
                    "unresolved concepts; express only when assigned by the character plan: "
                    + "; ".join(semantic_result.missing_descriptions)
                )
            semantic_context = "\n".join(line for line in context_lines if line.rstrip(": "))
        llm_prompt = build_llm_prompt(
            prompt,
            search_context=search_context,
            # The structured protocol owns the full roster.  Do not tell the
            # LLM that a first matched character will be added later: doing so
            # makes its Identity/Details blocks incomplete in multi-person
            # requests.  The legacy fixed-character fallback remains below.
            fixed_character=False,
            character_name="",
            fixed_character_hints={
                name: hint for name, hint in local_character_hints.items()
                if not any(
                    _character_outfit_matches(name, item) and item.appearance_tags
                    for item in character_effective_outfits
                )
            },
            sensual_mode=use_sensual_mode,
            mode=mode,
            prompt_builder_template=self._str("prompt_builder_template", ""),
            outfit_transfer_rule="\n".join(
                part
                for part in (
                    build_outfit_transfer_block(
                        outfit_plan, outfit_summary, effective_outfit
                    ),
                    profile_outfit_rule,
                    semantic_context,
                    character_wardrobe_authority_context(
                        character_effective_outfits
                    ),
                )
                if part
            ),
            original_theme=background_intent_prompt,
            keyword_prompt_rules=keyword_rules,
        )
        allow_futa = has_positive_futa_request(prompt)
        if self._bool("debug_prompt_enabled", False):
            self.logger.info(
                "[comfyui_agent] prompt builder LLM prompt:\n%s", llm_prompt
            )
        llm_content = ""
        llm_error = ""
        try:
            llm_content = await self._generate_prompt_tags_with_llm(
                provider_id=provider_id,
                llm_prompt=llm_prompt,
                use_deep_thinking=research_plan.use_deep_thinking,
                fixed_character=False,
                character_name="",
                allow_futa=allow_futa,
                max_tokens=7200 if nai_r_mode else None,
            )
        except Exception as exc:
            if not research_plan.use_deep_thinking:
                self.logger.warning(
                    "[comfyui_agent] prompt builder LLM failed: %s", exc
                )
                llm_error = str(exc)
                llm_content = ""
            else:
                self.logger.warning(
                    "[comfyui_agent] prompt builder deep thinking failed, retrying without it: %s",
                    exc,
                )
                try:
                    llm_content = await self._generate_prompt_tags_with_llm(
                        provider_id=provider_id,
                        llm_prompt=llm_prompt,
                        use_deep_thinking=False,
                        fixed_character=False,
                        character_name="",
                        allow_futa=allow_futa,
                        max_tokens=7200 if nai_r_mode else None,
                    )
                except Exception as retry_exc:
                    self.logger.warning(
                        "[comfyui_agent] prompt builder LLM failed: %s", retry_exc
                    )
                    llm_error = str(retry_exc)
                    llm_content = ""

        if is_chinese_model_refusal(llm_content):
            return self._model_refusal_result(summary, llm_content)
        # An empty completion is not a usable prompt.  Reasoning models often
        # spend the whole token budget on the chain of thought and return no
        # visible tags, and AstrBot may surface the text in a non-standard
        # field.  Retry once without deep thinking before giving up; never let
        # an empty LLM result fall through to the raw Chinese user prompt that
        # ComfyUI cannot consume.
        if not str(llm_content or "").strip() and research_plan.use_deep_thinking:
            self.logger.warning(
                "[comfyui_agent] prompt builder LLM returned empty content, "
                "retrying without deep thinking"
            )
            try:
                llm_content = await self._generate_prompt_tags_with_llm(
                    provider_id=provider_id,
                    llm_prompt=llm_prompt,
                    use_deep_thinking=False,
                    fixed_character=False,
                    character_name="",
                    allow_futa=allow_futa,
                    max_tokens=7200 if nai_r_mode else None,
                )
            except Exception as retry_exc:
                self.logger.warning(
                    "[comfyui_agent] prompt builder LLM retry failed: %s", retry_exc
                )
                llm_error = str(retry_exc)
                llm_content = ""
        if not str(llm_content or "").strip():
            if not llm_error:
                llm_error = "empty_llm_response"
            self.logger.warning(
                "[comfyui_agent] prompt builder LLM returned empty content; "
                "aborting generation instead of sending raw user text"
            )
            summary.update(
                {
                    "llm_failed": True,
                    "llm_error": llm_error,
                    "skipped_reason": "empty_llm_response",
                    "final_prompt_head": "",
                    "final_prompt_chars": 0,
                }
            )
            return PromptPipelineResult("", summary)

        initial_llm_content = llm_content
        retry_llm_content = ""
        if self._bool("debug_prompt_enabled", False):
            self.logger.info(
                "[comfyui_agent] prompt builder LLM initial output:\n%s", llm_content
            )
        # The background control marker is specified as the final item in the
        # LLM response, so it commonly sits outside the seven brace blocks.
        # Extract it before structured parsing, which intentionally retains
        # only the declared field values and would otherwise discard it.
        llm_content, llm_background_mode = extract_background_mode(llm_content)
        structured_parse = validate_confirmed_character_roster(
            _parse_structured_prompt(llm_content), semantic_character_tags
        )
        structured_roster_tags = structured_parse.roster_tags
        structured_copyright_tags = structured_parse.copyright_tags
        structured_characters = structured_parse.characters
        structured_scene = structured_parse.scene
        structured_nltags = structured_parse.nltags
        # Characters is a roster of named/canonical identities, not a second
        # people-count field. Anonymous, cropped, or obscured people may have a
        # valid Count while Characters, Identity, and Details remain empty.
        structured_prompt_mode = not structured_parse.validation_errors
        summary.update(
            {
                "prompt_llm_attempt_count": 1,
                "prompt_llm_accepted_attempt": (
                    "initial" if structured_prompt_mode else ""
                ),
                "structured_initial_validation_errors": list(
                    structured_parse.validation_errors
                ),
                "structured_scope_warnings": list(
                    structured_parse.scope_warnings
                ),
                "structured_unscoped_identity": list(
                    structured_parse.unscoped_identity
                ),
                "structured_unscoped_details": list(
                    structured_parse.unscoped_details
                ),
            }
        )
        structured_field_pattern = (
            r"\{\s*(?:Count|Characters|Copyright|Identity|Details|Tags|Nltags)\s*:"
        )
        has_structured_fields = bool(
            re.search(structured_field_pattern, str(llm_content or ""), re.IGNORECASE)
        )
        # A parenthesized canonical character tag (for example
        # ``revenant_(elden_ring)``) says nothing about whether a response that
        # visibly contains the seven-field envelope parsed successfully.  Keep
        # the old fast path for genuinely flat legacy tag streams, but never use
        # it to suppress validation/retry of brace-block output.
        if not structured_prompt_mode and (
            has_structured_fields or "_(" not in llm_content
        ):
            validation_reasons = structured_parse.validation_errors or (
                "response did not contain a complete valid seven-field envelope",
            )
            strict_format_prompt = (
                llm_prompt
                + "\n\nYour previous response was invalid for these exact reasons:\n- "
                + "\n- ".join(validation_reasons)
                + "\n\nPrevious response:\n"
                + initial_llm_content
                + "\n\nReturn exactly the seven "
                "brace blocks {Count: ...} {Characters: ...} {Copyright: ...} "
                "{Identity: ...} {Details: ...} {Tags: ...} {Nltags: ...}; Count "
                "must include an exact people-count tag, Copyright must contain "
                "only work/IP tags, and every named character must appear once "
                "in both Identity and Details."
            )
            try:
                retry_llm_content = await self._generate_prompt_tags_with_llm(
                    provider_id=provider_id,
                    llm_prompt=strict_format_prompt,
                    use_deep_thinking=False,
                    fixed_character=False,
                    character_name="",
                    allow_futa=allow_futa,
                    max_tokens=7200 if nai_r_mode else None,
                )
                summary["prompt_llm_attempt_count"] = 2
                if self._bool("debug_prompt_enabled", False):
                    self.logger.info(
                        "[comfyui_agent] prompt builder LLM retry output:\n%s",
                        retry_llm_content,
                    )
                if is_chinese_model_refusal(retry_llm_content):
                    return self._model_refusal_result(summary, retry_llm_content)
                retry_content = retry_llm_content
                retry_content, retry_background_mode = extract_background_mode(
                    retry_content
                )
                retry_parse = validate_confirmed_character_roster(
                    _parse_structured_prompt(retry_content), semantic_character_tags
                )
                structured_roster_tags = retry_parse.roster_tags
                structured_copyright_tags = retry_parse.copyright_tags
                structured_characters = retry_parse.characters
                structured_scene = retry_parse.scene
                structured_nltags = retry_parse.nltags
                retry_structured_prompt_mode = not retry_parse.validation_errors
                summary["structured_retry_validation_errors"] = list(
                    retry_parse.validation_errors
                )
                if retry_structured_prompt_mode:
                    structured_parse = retry_parse
                    llm_content = retry_content
                    llm_background_mode = retry_background_mode
                    structured_prompt_mode = True
                    summary["structured_format_retry"] = True
                    summary["prompt_llm_accepted_attempt"] = "retry"
                    summary["structured_scope_warnings"] = list(
                        retry_parse.scope_warnings
                    )
                    summary["structured_unscoped_identity"] = list(
                        retry_parse.unscoped_identity
                    )
                    summary["structured_unscoped_details"] = list(
                        retry_parse.unscoped_details
                    )
                else:
                    summary["structured_format_retry"] = False
            except Exception as exc:
                self.logger.warning(
                    "[comfyui_agent] structured prompt format retry failed: %s", exc
                )
                summary["structured_format_retry"] = False
                summary["structured_retry_error"] = str(exc)
        if not structured_prompt_mode:
            # Never reinterpret a failed seven-field response as a flat legacy
            # tag stream.  The legacy cleanup below deliberately strips brace
            # blocks; allowing a structured-looking response through therefore
            # turns a validation error into a successful but almost empty image
            # request.  Stop instead, preserving the failure in diagnostics.
            structured_field_count = len(
                re.findall(
                    structured_field_pattern,
                    str(llm_content or ""),
                    flags=re.IGNORECASE,
                )
            )
            if structured_field_count:
                llm_error = "invalid_structured_prompt"
                self.logger.warning(
                    "[comfyui_agent] prompt builder returned %s structured fields "
                    "but validation failed; aborting instead of stripping them",
                    structured_field_count,
                )
                summary.update(
                    {
                        "llm_failed": True,
                        "llm_error": llm_error,
                        "structured_field_count": structured_field_count,
                        "skipped_reason": llm_error,
                        "final_prompt_head": "",
                        "final_prompt_chars": 0,
                    }
                )
                if self._bool("debug_prompt_enabled", False):
                    summary["prompt_llm_initial_content"] = initial_llm_content
                    summary["prompt_llm_retry_content"] = retry_llm_content
                    summary["llm_raw_content"] = (
                        retry_llm_content or initial_llm_content
                    )
                return PromptPipelineResult("", summary)
        if structured_prompt_mode:
            structured_parse, removed_sexual_traits = (
                enforce_structured_sexual_trait_authority(structured_parse, prompt)
            )
            structured_roster_tags = structured_parse.roster_tags
            structured_copyright_tags = structured_parse.copyright_tags
            structured_characters = structured_parse.characters
            structured_scene = structured_parse.scene
            structured_nltags = structured_parse.nltags
            summary["sexual_trait_authority"] = {
                "futa_allowed": allow_futa,
                "male_genitals_allowed": has_positive_male_genital_request(prompt),
                "removed": list(removed_sexual_traits),
            }
        llm_framing_text = ", ".join(
            part
            for part in (
                prompt,
                structured_scene if structured_prompt_mode else llm_content,
            )
            if part
        )
        framed_character_outfits = apply_framing_to_character_outfits(
            character_effective_outfits, llm_framing_text
        )
        if framed_character_outfits != character_effective_outfits:
            character_effective_outfits = framed_character_outfits
            wardrobe_authority = build_wardrobe_authority(
                character_effective_outfits,
                semantic_result,
                semantic_anchors,
                explicit_term_tags=explicit_term_tags,
            )
            required_profile_tags = wardrobe_authority.filter_tags(
                required_profile_tags
            )
            semantic_required_tags = wardrobe_authority.filter_tags(
                semantic_required_tags
            )
            semantic_visible_outfit_tags = wardrobe_authority.filter_tags(
                tuple(
                    tag
                    for tag in profile_tags_for_request
                    if tag not in effective_outfit.removed_tags
                )
            )
            global_outfit_reinforcement_tags = safe_global_outfit_tags(
                character_effective_outfits
            )
        semantic_character_tags = confirmed_semantic_character_tags(semantic_result)
        structured_characters, structured_nltags = (
            reconcile_confirmed_semantic_characters(
                structured_characters,
                structured_nltags,
                semantic_character_tags,
            )
        )
        structured_character_mode = bool(structured_characters)
        structured_required_character_tags: list[str] = []
        structured_identity_blocks: tuple[str, ...] = ()
        structured_detail_blocks: tuple[str, ...] = ()
        profile_fallback_overridden_tags: list[str] = []
        outfit_target_display = outfit_plan.target_character
        if (
            not structured_prompt_mode
            and len(semantic_result.character_appearance_profiles) == 1
        ):
            aliases, source_tag, saved_tags = (
                semantic_result.character_appearance_profiles[0]
            )
            override_dimensions = appearance_override_dimensions(
                prompt, (*aliases, source_tag), character_count=1
            )
            stable_tags = tuple(
                tag
                for tag in saved_tags
                if appearance_dimension(tag) not in override_dimensions
            )
            semantic_required_tags = tuple(
                dict.fromkeys((*semantic_required_tags, source_tag, *stable_tags))
            )
        if structured_prompt_mode:
            # Details already carries ownership-aware prose, but that is not a
            # reason to discard useful Danbooru action/pose tags from Tags.
            # Preserve the structured Tags block and let only the regular exact
            # deduplication/conflict cleaner process it.
            removed_unbound_tags: tuple[str, ...] = ()
            structured_scene = filter_shared_appearance_tags(
                structured_scene,
                character_effective_outfits,
                user_prompt=prompt,
            )
            if (
                effective_outfit.modified
                or outfit_plan.enabled
                or semantic_result.named_outfit_tags
                or character_effective_outfits
                or casual_life_requested
            ):
                structured_scene = keep_only_verified_outfit_tags(
                    structured_scene,
                    tuple(
                        dict.fromkeys(
                            (
                                *explicit_term_tags,
                                *effective_outfit.effective_tags,
                                *(
                                    ()
                                    if character_effective_outfits
                                    else semantic_visible_outfit_tags
                                ),
                                *(
                                    ()
                                    if character_effective_outfits
                                    else semantic_result.named_outfit_tags
                                ),
                                *global_outfit_reinforcement_tags,
                            )
                        )
                    ),
                    effective_outfit.forbidden_slots,
                    # A valid seven-field response is authoritative. Only apply
                    # explicit forbidden-slot removals here; database evidence is
                    # not a whitelist for the writer's actions, props or scenery.
                    strict_allowlist=False,
                )
                structured_scene = wardrobe_authority.filter_tag_text(structured_scene)
            resolution_statuses: list[dict[str, Any]] = []
            effective_detail_blocks: list[str] = []
            effective_identity_blocks: list[str] = []
            used_fixed_names: set[str] = set()
            for character in structured_characters:
                semantic_confirmed = character.name in semantic_character_tags
                fixed_name = ""
                if semantic_confirmed:
                    identity_tags = (character.name,)
                    status = "semantic_confirmed"
                    canonical_tag = character.name
                else:
                    fixed_name = _match_fixed_character_hint(
                        character.name,
                        local_character_hints,
                        used_fixed_names,
                        structured_character_count=len(structured_characters),
                    )
                    if fixed_name:
                        used_fixed_names.add(fixed_name)
                        # The local text is an LLM hint, not a schema.  It may begin
                        # with natural language rather than a canonical tag, so the
                        # final roster anchor remains the LLM's structured name.
                        identity_tags = (character.name,)
                        status = "fixed"
                        canonical_tag = character.name
                    else:
                        resolution = await self._danbooru_resolver.resolve_detailed(
                            llm_content=character.name,
                            user_prompt=character.name,
                            fixed_character=False,
                        )
                        identity_tags = resolution.identity_tags or tuple(
                            split_tags(resolution.text)
                        )[:1]
                        status = resolution.status
                        canonical_tag = resolution.canonical_tag
                character_outfit = next(
                    (
                        candidate
                        for candidate in character_effective_outfits
                        if _character_outfit_matches(character.name, candidate)
                    ),
                    None,
                )
                planner_changes = (
                    _semantic_appearance_changes_for_character(
                        semantic_appearance_changes, character_outfit
                    ) if character_outfit else ()
                )
                omitted_dimensions = _omitted_appearance_dimensions(planner_changes)
                detail = character.detail_tags
                if omitted_dimensions:
                    detail = _without_omitted_appearance(detail, omitted_dimensions)
                override_dimensions = frozenset()
                if character_outfit and character_outfit.appearance_tags:
                    override_dimensions = appearance_override_dimensions(
                        prompt,
                        tuple(
                            dict.fromkeys(
                                (
                                    character.name,
                                    character_outfit.target_source_text,
                                    *character_outfit.target_candidates,
                                )
                            )
                        ),
                        character_count=len(structured_characters),
                    )
                if (
                    character_outfit is not None
                    and _UNTRUSTED_OUTFIT_DETAIL_RE.search(detail)
                    and character_outfit.resolution_state == "unspecified"
                ):
                    profile_fallback_overridden_tags.extend(
                        character_outfit.effective.effective_tags
                    )
                if character_effective_outfits or casual_life_requested:
                    detail = controlled_character_outfit_detail(
                        detail,
                        character.name,
                        character_outfit,
                        user_prompt=prompt,
                        wardrobe_authority=wardrobe_authority,
                    )
                elif (
                    effective_outfit.modified
                    and (
                        fixed_name == outfit_plan.target_character
                        or len(structured_characters) == 1
                    )
                ):
                    detail = rewrite_target_outfit_detail(detail, effective_outfit)
                    outfit_target_display = character.name.replace("_", " ")
                effective_detail_blocks.append(detail)
                identity_block = character.identity_tags
                if character_outfit and character_outfit.appearance_tags:
                    if omitted_dimensions:
                        identity_block = _without_omitted_appearance(
                            identity_block, omitted_dimensions
                        )
                    identity_block = merge_authoritative_identity_block(
                        character.name,
                        identity_block,
                        character_outfit.appearance_tags,
                        override_dimensions,
                        advisory_writer_dimensions=_advisory_writer_dimensions(
                            planner_changes
                        ),
                        replacement_writer_dimensions=_replacement_writer_dimensions(
                            planner_changes
                        ),
                    )
                effective_identity_blocks.append(identity_block)
                for identity_tag in identity_tags:
                    if identity_tag and identity_tag not in structured_required_character_tags:
                        structured_required_character_tags.append(identity_tag)
                resolution_statuses.append(
                    {
                        "name": character.name,
                        "status": status,
                        "canonical_tag": canonical_tag,
                        "identity_tags": list(identity_tags),
                    }
                )
            structured_identity_blocks = tuple(
                dict.fromkeys(
                    block for block in effective_identity_blocks if block
                )
            )
            structured_detail_blocks = tuple(
                dict.fromkeys(block for block in effective_detail_blocks if block)
            )
            if profile_fallback_overridden_tags:
                overridden_keys = {
                    WardrobeAuthority._key(tag)
                    for tag in profile_fallback_overridden_tags
                }
                required_profile_tags = tuple(
                    tag
                    for tag in required_profile_tags
                    if WardrobeAuthority._key(tag) not in overridden_keys
                )
                semantic_required_tags = tuple(
                    tag
                    for tag in semantic_required_tags
                    if WardrobeAuthority._key(tag) not in overridden_keys
                )
                summary["profile_fallback_overridden_tags"] = list(
                    dict.fromkeys(profile_fallback_overridden_tags)
                )
            # Only the seventh block's Danbooru tag section goes through the
            # tag cleaner.  The other structured fields are protected sections
            # assembled in their declared order by prompt_builder.
            llm_content = structured_scene
            nltags = normalize_structured_nltags(
                structured_nltags,
                tuple(character.name for character in structured_characters),
            )
            omitted_profile_tags = tuple(
                tag.replace("_", " ")
                for item in character_effective_outfits
                for tag in item.appearance_tags
                if appearance_dimension(tag) in _omitted_appearance_dimensions(
                    _semantic_appearance_changes_for_character(
                        semantic_appearance_changes, item
                    )
                )
            )
            for tag in omitted_profile_tags:
                # Nltags has no reliable character grammar, so remove only
                # suppressed saved values. Another wearer's different eye
                # color must remain available in their prose.
                nltags = re.sub(re.escape(tag), "", nltags, flags=re.I)
            if omitted_profile_tags:
                nltags = re.sub(r"\s+,", ",", nltags)
                nltags = re.sub(r"\s{2,}", " ", nltags)
            if semantic_result.outfit_source_tags and not character_effective_outfits:
                nltags = minimal_verified_outfit_nltags(
                    nltags,
                    semantic_result.outfit_source_tags,
                    include_source_cue=not effective_outfit.modified,
                )
            outfit_constraint_narrative = build_outfit_constraint_narrative(
                effective_outfit,
                subject=outfit_target_display,
                source_tags=semantic_result.outfit_source_tags,
            )
            if outfit_constraint_narrative and not character_effective_outfits:
                nltags = " ".join(
                    part for part in (nltags, outfit_constraint_narrative) if part
                )
            nltags = strip_structured_block_duplicates_from_nltags(
                nltags,
                structured_identity_blocks,
                structured_detail_blocks,
            )
            summary["structured_character_mode"] = structured_character_mode
            summary["structured_prompt_mode"] = True
            summary["structured_character_count"] = len(structured_characters)
            summary["structured_roster_tags"] = list(structured_roster_tags)
            summary["structured_copyright_tags"] = list(
                structured_copyright_tags
            )
            summary["required_profile_tags"] = list(required_profile_tags)
            summary["removed_unbound_directional_tags"] = list(
                removed_unbound_tags
            )
            summary["character_resolution_statuses"] = resolution_statuses
            summary["local_character_hints"] = list(local_character_hints)
            use_fixed_character = False
            fixed_character_name = ""
            required_core_tags = tuple(
                dict.fromkeys(
                    (
                        *structured_required_character_tags,
                        *semantic_required_tags,
                    )
                )
            )
            required_count_tags = structured_roster_tags
            if structured_characters:
                # The Count block must agree with the Characters roster.  A
                # reasoning model may stall on recounting characters and emit a
                # mismatched or duplicated count tag; the deterministic roster
                # length is the authoritative source for Anima's people count.
                # `futa with female` / `futa with male` and relationship tags
                # are kept, but any numeric people-count tag that disagrees
                # with the roster is replaced by a plain people-count tag.
                if not structured_count_tags_match_roster(
                    required_count_tags, len(structured_characters)
                ):
                    kept_count_tags = tuple(
                        item
                        for item in required_count_tags
                        if item.lower().replace("_", " ")
                        not in {
                            "futa with female",
                            "futa with male",
                            "futanari",
                            "1futanari",
                        }
                        and not re.fullmatch(
                            r"\s*\d+\s*(girls?|boys?|people|persons?)\s*",
                            item,
                            flags=re.IGNORECASE,
                        )
                    )
                    required_count_tags = tuple(
                        dict.fromkeys(
                            (
                                f"{len(structured_characters)}people",
                                *kept_count_tags,
                            )
                        )
                    )
                    summary["structured_count_repaired"] = True
        else:
            llm_content, nltags = extract_nltags(llm_content)
            if (
                effective_outfit.modified
                or outfit_plan.enabled
                or semantic_result.named_outfit_tags
            ):
                llm_content = keep_only_verified_outfit_tags(
                    llm_content,
                    tuple(
                        dict.fromkeys(
                            (
                                *explicit_term_tags,
                                *effective_outfit.effective_tags,
                                *semantic_visible_outfit_tags,
                                *semantic_result.named_outfit_tags,
                            )
                        )
                    ),
                    effective_outfit.forbidden_slots,
                    strict_allowlist=effective_outfit.has_allowlist_override,
                )
            if semantic_result.outfit_source_tags:
                nltags = minimal_verified_outfit_nltags(
                    nltags,
                    semantic_result.outfit_source_tags,
                    include_source_cue=not effective_outfit.modified,
                )
            outfit_constraint_narrative = build_outfit_constraint_narrative(
                effective_outfit,
                subject=outfit_plan.target_character,
                source_tags=semantic_result.outfit_source_tags,
            )
            if outfit_constraint_narrative:
                nltags = " ".join(
                    part for part in (nltags, outfit_constraint_narrative) if part
                )
            llm_content = re.sub(
                r"\{\s*(?:Count|Characters|Copyright|Identity|Details|Tags|Nltags)\s*:[^}]*\}",
                "",
                llm_content,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip(" ,\n")
            summary["structured_character_mode"] = False
        llm_content = wardrobe_authority.filter_tag_text(llm_content)
        nltags_authority = wardrobe_authority
        if profile_fallback_overridden_tags:
            overridden_keys = {
                WardrobeAuthority._key(tag)
                for tag in profile_fallback_overridden_tags
            }
            nltags_authority = replace(
                wardrobe_authority,
                stale_cached_tags=tuple(
                    tag
                    for tag in wardrobe_authority.stale_cached_tags
                    if WardrobeAuthority._key(tag) not in overridden_keys
                ),
            )
        nltags = nltags_authority.filter_prose(nltags)
        background_mode = ""
        background_mode_source = "not_applicable"
        generated_scene_selected = False
        if mode == "txt2img":
            background_mode = llm_background_mode
            if background_mode:
                background_mode_source = "llm_marker"
            else:
                background_mode = DEFAULT_PORTRAIT
                background_mode_source = "missing_marker_default"
            llm_content, background_mode, background_overridden = (
                enforce_user_background_intent(
                    llm_content, background_mode, background_intent_prompt
                )
            )
            if background_overridden:
                background_mode_source = "user_prompt_override"
            nltags = strip_unrequested_default_background_prose(
                nltags, background_intent_prompt
            )
            generated_scene_selected = bool(
                not user_requests_explicit_background(background_intent_prompt)
                and has_generated_scene(llm_content, nltags)
            )
            if generated_scene_selected:
                if background_mode == DEFAULT_PORTRAIT:
                    background_mode = EXPLICIT_SCENE
                    background_mode_source = "llm_generated_scene"
                llm_content = strip_default_portrait_tags(llm_content)
                nltags = strip_default_portrait_prose(nltags)
        llm_failed = bool(llm_error and not str(llm_content or "").strip())
        if structured_prompt_mode:
            character_resolution = DanbooruResolveOutcome(text=llm_content)
        else:
            character_resolution = await self._danbooru_resolver.resolve_detailed(
                llm_content=llm_content,
                user_prompt=prompt,
                fixed_character=use_fixed_character,
            )
        if not structured_prompt_mode and not use_fixed_character and (
            character_resolution.status == "unresolved"
            or (character_resolution.explicit_request and not required_core_tags)
        ):
            try:
                candidate_hints = await self._generate_character_candidates_with_llm(
                    provider_id=provider_id,
                    user_prompt=prompt,
                    rejected_content=llm_content,
                )
            except Exception as exc:
                self.logger.warning(
                    "[comfyui_agent] character candidate planner failed: %s",
                    exc,
                )
                candidate_hints = ()
            if candidate_hints:
                character_resolution = await self._danbooru_resolver.resolve_detailed(
                    llm_content=llm_content,
                    user_prompt=prompt,
                    fixed_character=False,
                    candidate_hints=candidate_hints,
                )
        llm_content = character_resolution.text
        if outfit_summary and not character_effective_outfits:
            llm_content = ", ".join(
                part for part in (llm_content, outfit_summary) if part
            )
        if character_resolution.identity_tags:
            required_core_tags = tuple(
                dict.fromkeys(
                    (*character_resolution.identity_tags, *required_profile_tags)
                )
            )
        elif required_profile_tags:
            required_core_tags = tuple(
                dict.fromkeys((*required_core_tags, *required_profile_tags))
            )
        if semantic_required_tags:
            required_core_tags = tuple(
                dict.fromkeys((*required_core_tags, *semantic_required_tags))
            )
        required_core_tags = wardrobe_authority.filter_tags(required_core_tags)
        low_cfg_harness = bool(prompt_config.get("low_cfg_harness_enabled", False))
        constraint_raw = ""
        constraint_plan = parse_constraint_plan("")
        if low_cfg_harness:
            # DEFERRED(Turbo wardrobe authority): do not casually insert an
            # extra wardrobe filter here. Constraint tags and priority tags
            # have low-CFG semantics that need a dedicated, end-to-end fix.
            try:
                constraint_raw = await self._generate_constraint_plan_with_llm(
                    provider_id=provider_id,
                    plan_prompt=build_constraint_plan_prompt(
                        user_prompt=prompt,
                        llm_content=llm_content or prompt,
                        fixed_character_name=fixed_character_name,
                    ),
                )
                constraint_plan = parse_constraint_plan(constraint_raw)
            except Exception as constraint_exc:
                self.logger.warning(
                    "[comfyui_agent] prompt constraint planner failed: %s",
                    constraint_exc,
                )
        structured_extra_tags = wardrobe_authority.filter_tags(tuple(
            dict.fromkeys(
                (
                    *required_profile_tags,
                    *semantic_required_tags,
                    *global_outfit_reinforcement_tags,
                )
            )
        ))
        if generated_scene_selected:
            structured_extra_tags = tuple(split_tags(
                strip_default_portrait_tags(", ".join(structured_extra_tags))
            ))
        built = build_final_prompt(
            user_prompt=prompt,
            llm_content=llm_content,
            config=prompt_config,
            required_count_tags=required_count_tags,
            required_core_tags=required_core_tags,
            structured_character_tags=tuple(structured_required_character_tags),
            structured_copyright_tags=structured_copyright_tags,
            structured_identity_blocks=structured_identity_blocks,
            structured_detail_blocks=structured_detail_blocks,
            structured_tag_tags=structured_extra_tags,
            preserve_structured_order=structured_prompt_mode,
            constraint_plan=constraint_plan,
            background_mode=background_mode,
            nltags=nltags,
            suppress_fixed_character=structured_prompt_mode,
            force_multi_character=structured_character_mode,
        )
        initial_input_tag_count = len(split_tags(llm_content))
        content_tag_count = len(split_tags(built.content_tags))
        removed_tag_count = max(0, initial_input_tag_count - content_tag_count)
        self.logger.info(
            "[comfyui_agent] prompt built raw=%s web_search=%s deep_thinking=%s character=%s sensual=%s fixed_character=%s default_style=%s low_cfg_harness=%s constraint=%s weighted_style=%s required_count_tags=%s required_core_tags=%s content_tags=%s content_chars=%s final_chars=%s final_head=%s",
            built.raw_mode,
            bool(search_context),
            research_plan.use_deep_thinking,
            built.character_name or "none",
            built.used_sensual_mode,
            built.used_fixed_character,
            built.used_default_style,
            low_cfg_harness,
            built.constraint_mode,
            ",".join(built.weighted_style_tags) or "none",
            ",".join(built.required_count_tags) or "none",
            ",".join(built.required_core_tags) or "none",
            content_tag_count,
            len(built.content_tags),
            len(built.final_prompt),
            built.final_prompt[:300],
        )
        summary.update(
            {
                "raw_mode": built.raw_mode,
                "web_search": bool(search_context),
                "deep_thinking": research_plan.use_deep_thinking,
                "search_reason": research_plan.search_reason or "",
                "thinking_reason": research_plan.thinking_reason or "",
                "fixed_character": built.used_fixed_character,
                "fixed_character_name": built.character_name,
                "sensual_mode": built.used_sensual_mode,
                "default_style": built.used_default_style,
                "low_cfg_harness": low_cfg_harness,
                "background_mode": background_mode or "unresolved",
                "background_mode_source": background_mode_source,
                "constraint_mode": built.constraint_mode,
                "weighted_style_tags": list(built.weighted_style_tags),
                "constraint_tags": list(built.constraint_tags),
                "removed_constraint_tags": list(built.removed_constraint_tags),
                "constraint_reason": built.constraint_reason,
                "required_count_tags": list(built.required_count_tags),
                "required_core_tags": list(built.required_core_tags),
                "required_profile_tags": list(required_profile_tags),
                "named_character_detected": character_resolution.status
                in {"resolved", "unresolved", "source_unavailable"},
                "character_resolution_status": character_resolution.status,
                "character_canonical_tag": character_resolution.canonical_tag,
                "character_identity_tags": list(character_resolution.identity_tags),
                "character_candidate_hints": list(character_resolution.candidate_hints),
                "character_resolution_evidence": list(character_resolution.evidence),
                "outfit_transfer": outfit_plan.enabled,
                "outfit_transfer_source": outfit_plan.source_subject,
                "outfit_transfer_target": outfit_plan.target_character,
                "outfit_summary_source": outfit_summary_source,
                "outfit_summary_chars": len(outfit_summary),
                "wardrobe_source": wardrobe_source,
                "explicit_wardrobe_evidence": explicit_wardrobe_evidence,
                "creative_shared_tags_allowed": bool(
                    character_effective_outfits
                    and all(
                        item.wardrobe_kind == "creative_fallback"
                        for item in character_effective_outfits
                    )
                ),
                "outfit_effective_tags": list(
                    wardrobe_authority.selected_tags
                    if character_effective_outfits
                    else wardrobe_authority.filter_tags(effective_outfit.effective_tags)
                ),
                "outfit_removed_tags": list(
                    wardrobe_authority.removed_tags
                    if character_effective_outfits
                    else effective_outfit.removed_tags
                ),
                "outfit_added_tags": list(
                    dict.fromkeys(
                        tag
                        for item in character_effective_outfits
                        for tag in item.effective.added_tags
                    )
                    if character_effective_outfits
                    else effective_outfit.added_tags
                ),
                "wardrobe_authority": {
                    "has_character_plans": wardrobe_authority.has_character_plans,
                    "all_creative": wardrobe_authority.all_creative,
                    "cached_tags": list(wardrobe_authority.cached_tags),
                    "explicit_tags": list(wardrobe_authority.explicit_tags),
                    "selected_tags": list(wardrobe_authority.selected_tags),
                    "source_grounding_tags": list(
                        wardrobe_authority.source_grounding_tags
                    ),
                    "unbound_grounding_tags": list(
                        wardrobe_authority.unbound_grounding_tags
                    ),
                    "removed_tags": list(wardrobe_authority.removed_tags),
                    "stale_cached_tags": list(
                        wardrobe_authority.stale_cached_tags
                    ),
                },
                "semantic_character_outfits": [
                    {
                        "target_anchor_id": item.target_anchor_id,
                        "target": item.target_source_text,
                        "wardrobe_kind": item.wardrobe_kind,
                        "wardrobe_anchor_id": item.wardrobe_anchor_id,
                        "effective_tags": list(item.effective.effective_tags),
                        "removed_tags": list(item.effective.removed_tags),
                        "appearance_tags": list(item.appearance_tags),
                        "complete_named_profile": item.complete_named_profile,
                        "resolution_state": item.resolution_state,
                        "unresolved_evidence": list(item.unresolved_evidence),
                    }
                    for item in character_effective_outfits
                ],
                "wardrobe_resolution_states": {
                    item.target_source_text: item.resolution_state
                    for item in character_effective_outfits
                },
                "global_outfit_reinforcement_tags": list(
                    global_outfit_reinforcement_tags
                ),
                "outfit_user_patches": [
                    {
                        "subject": patch.subject,
                        "operation": patch.operation,
                        "slot": patch.slot,
                        "value": patch.value,
                        "evidence": patch.evidence,
                    }
                    for patch in effective_outfit.patches
                ],
                "outfit_source_anchor_emitted": include_source_anchor,
                "llm_failed": llm_failed,
                "llm_error": self._shorten(llm_error, 300),
                "llm_content_tag_count": content_tag_count,
                "removed_content_tag_count": removed_tag_count,
                "llm_content_chars": len(built.content_tags),
                "nltags_chars": len(nltags),
                "final_prompt_chars": len(built.final_prompt),
                "final_prompt_head": self._shorten(built.final_prompt, 600),
                "prompt_builder_template_customized": bool(
                    self._str("prompt_builder_template", "").strip()
                ),
            }
        )
        if self._bool("debug_prompt_enabled", False):
            self.logger.info(
                "[comfyui_agent] prompt builder final prompt:\n%s", built.final_prompt
            )
            summary.update(
                {
                    "semantic_plan_prompt": build_semantic_plan_prompt(prompt),
                    "semantic_plan_initial_raw": semantic_plan_initial_raw,
                    "semantic_plan_repair_prompt": (
                        build_semantic_plan_repair_prompt(
                            prompt,
                            semantic_plan_initial_raw,
                            semantic_plan_validation_issues(
                                semantic_plan_initial_raw, prompt
                            ),
                        )
                        if semantic_plan_attempt_count > 1
                        else ""
                    ),
                    "semantic_plan_raw": semantic_plan_raw,
                    "llm_prompt": llm_prompt,
                    "prompt_llm_initial_content": initial_llm_content,
                    "prompt_llm_retry_content": retry_llm_content,
                    "outfit_summary": outfit_summary,
                    "constraint_plan": constraint_raw,
                    "llm_content": llm_content,
                    "final_prompt": built.final_prompt,
                }
            )
        if nai_r_mode:
            requested_canvas = canvas_size or (
                self._int("width", 1024), self._int("height", 1536)
            )
            debug_nai_plan = self._bool("debug_prompt_enabled", False)
            plan_prompt = ""
            plan_raw = ""
            try:
                nai_canvas = resolve_nai_canvas(
                    prompt_config, requested_canvas, explicit_size=canvas_size_explicit
                )
                plan_prompt = build_nai_character_plan_prompt(
                    prompt, built.final_prompt, nai_canvas
                )
                if debug_nai_plan:
                    self.logger.info(
                        "[comfyui_agent] NAI -r position planner canvas:\n%s",
                        json.dumps(nai_canvas, ensure_ascii=False, indent=2),
                    )
                    self.logger.info(
                        "[comfyui_agent] NAI -r position planner prompt:\n%s",
                        plan_prompt,
                    )
                plan_response = await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=plan_prompt,
                    system_prompt="Plan NAI character instances precisely. Return only valid JSON.",
                    max_tokens=max(
                        1200,
                        min(7200, max(
                            self._int("prompt_builder_max_tokens", 1800),
                            1200 + 260 * nai_canvas["character_limit"],
                        )),
                    ),
                    thinking={"type": "disabled"},
                )
                plan_raw = _extract_completion_text(plan_response)
                if debug_nai_plan:
                    self.logger.info(
                        "[comfyui_agent] NAI -r position planner raw output:\n%s",
                        plan_raw,
                    )
                nai_plan = parse_nai_character_plan(
                    plan_raw,
                    character_limit=nai_canvas["character_limit"],
                )
                _enforce_nai_appearance_omissions(
                    nai_plan["characters"],
                    character_effective_outfits,
                    semantic_appearance_changes,
                )
                if background_mode == DEFAULT_PORTRAIT and has_generated_scene(
                    nai_plan["global_prompt"],
                    ", ".join(item["prompt"] for item in nai_plan["characters"]),
                ):
                    nai_plan["global_prompt"] = strip_default_portrait_tags(
                        strip_default_portrait_prose(nai_plan["global_prompt"])
                    )
                    for character in nai_plan["characters"]:
                        character["prompt"] = strip_default_portrait_tags(
                            character["prompt"]
                        )
                    summary["background_mode"] = EXPLICIT_SCENE
                    summary["background_mode_source"] = "nai_generated_scene"
                if built.used_default_style:
                    nai_plan["global_prompt"] = preserve_nai_global_artist_tags(
                        nai_plan["global_prompt"],
                        active_artist_tags(prompt_config, preset_index),
                    )
            except Exception as exc:
                summary.update(
                    skipped_reason="nai_character_plan_failed", llm_error=str(exc)
                )
                if debug_nai_plan:
                    summary["nai_character_plan_prompt"] = plan_prompt
                    summary["nai_character_plan_raw"] = plan_raw
                    self.logger.info(
                        "[comfyui_agent] NAI -r position planner failed: %s", exc
                    )
                return PromptPipelineResult("", summary)
            summary["nai_characters"] = nai_plan["characters"]
            summary["nai_canvas"] = nai_canvas
            summary["nai_composition_analysis"] = nai_plan["composition_analysis"]
            summary["nai_global_prompt"] = nai_plan["global_prompt"]
            summary["nai_character_count"] = len(nai_plan["characters"])
            summary["nai_dropped_interaction_tags"] = nai_plan["dropped_interaction_tags"]
            summary["nai_dropped_global_character_tags"] = nai_plan[
                "dropped_global_character_tags"
            ]
            summary["final_prompt_head"] = self._shorten(nai_plan["global_prompt"], 600)
            summary["final_prompt_chars"] = len(nai_plan["global_prompt"])
            if self._bool("debug_prompt_enabled", False):
                summary["nai_character_plan_prompt"] = plan_prompt
                summary["nai_character_plan_raw"] = plan_raw
                summary["nai_full_prompt_before_split"] = built.final_prompt
                summary["final_prompt"] = nai_plan["global_prompt"]
                self.logger.info(
                    "[comfyui_agent] NAI -r position planner final plan:\n%s",
                    json.dumps(
                        {"canvas": nai_canvas, **nai_plan},
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
            return PromptPipelineResult(nai_plan["global_prompt"], summary)
        return PromptPipelineResult(built.final_prompt, summary)
