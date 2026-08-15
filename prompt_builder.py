from dataclasses import dataclass
from typing import Any

try:
    from .prompt_background import DEFAULT_PORTRAIT, apply_default_portrait_tags
    from .prompt_constraints import PromptConstraintPlan, apply_prompt_constraints
    from .prompt_presets import (
        DEFAULT_CHARACTER_TAGS,
        DEFAULT_QUALITY_TAGS,
        active_artist_tags,
        active_style_tags,
        apply_config_preset,
        selected_fixed_character,
        strip_raw_prefix,
        wants_default_style,
        wants_sensual_mode,
    )
    from .tag_cleaner import (
        DEFAULT_MAX_CONTENT_TAGS,
        clean_content_tags,
        display_tag_text,
        join_prompt_parts,
    )
except ImportError:  # pragma: no cover - fallback for direct script-style imports.
    from prompt_background import DEFAULT_PORTRAIT, apply_default_portrait_tags
    from prompt_constraints import PromptConstraintPlan, apply_prompt_constraints
    from prompt_presets import (
        DEFAULT_CHARACTER_TAGS,
        DEFAULT_QUALITY_TAGS,
        active_artist_tags,
        active_style_tags,
        apply_config_preset,
        selected_fixed_character,
        strip_raw_prefix,
        wants_default_style,
        wants_sensual_mode,
    )
    from tag_cleaner import (
        DEFAULT_MAX_CONTENT_TAGS,
        clean_content_tags,
        display_tag_text,
        join_prompt_parts,
    )


@dataclass(frozen=True)
class PromptBuildResult:
    final_prompt: str
    content_tags: str
    raw_mode: bool
    used_fixed_character: bool
    used_default_style: bool
    required_count_tags: tuple[str, ...] = ()
    required_core_tags: tuple[str, ...] = ()
    character_name: str = ""
    used_sensual_mode: bool = False
    constraint_mode: bool = False
    weighted_style_tags: tuple[str, ...] = ()
    constraint_tags: tuple[str, ...] = ()
    removed_constraint_tags: tuple[str, ...] = ()
    constraint_reason: str = ""


def build_final_prompt(
    *,
    user_prompt: str,
    llm_content: str,
    config: dict[str, Any],
    required_count_tags: tuple[str, ...] = (),
    required_core_tags: tuple[str, ...] = (),
    structured_character_tags: tuple[str, ...] = (),
    structured_copyright_tags: tuple[str, ...] = (),
    structured_identity_blocks: tuple[str, ...] = (),
    structured_detail_blocks: tuple[str, ...] = (),
    structured_tag_tags: tuple[str, ...] = (),
    preserve_structured_order: bool = False,
    constraint_plan: PromptConstraintPlan | None = None,
    narrative_blocks: tuple[str, ...] = (),
    nltags: str = "",
    suppress_fixed_character: bool = False,
    force_multi_character: bool = False,
    background_mode: str = "",
) -> PromptBuildResult:
    config = apply_config_preset(config)
    raw_mode, raw_prompt = strip_raw_prefix(user_prompt)
    if raw_mode:
        final = join_prompt_parts([raw_prompt])
        return PromptBuildResult(
            final_prompt=final,
            content_tags=final,
            raw_mode=True,
            used_fixed_character=False,
            used_default_style=False,
            required_count_tags=(),
            required_core_tags=(),
            character_name="",
            used_sensual_mode=False,
        )

    fixed_character = (
        None
        if suppress_fixed_character
        else selected_fixed_character(user_prompt, config)
    )
    use_character = fixed_character is not None
    artist = active_artist_tags(config)
    style_tags = active_style_tags(config).strip()
    use_style = wants_default_style(user_prompt, bool(artist.strip() or style_tags))
    use_sensual = wants_sensual_mode(user_prompt, config)
    # `preset_suppress_quality` lets a preset ask for no quality prefix at all.
    # Without it an empty `quality_prefix` would fall back to the default tags.
    if config.get("preset_suppress_quality"):
        quality = ""
    else:
        quality = str(config.get("quality_prefix") or DEFAULT_QUALITY_TAGS)
    character_name = ""
    if fixed_character is not None:
        character_name, character = fixed_character
    else:
        character = DEFAULT_CHARACTER_TAGS
    if use_character and not character.strip():
        use_character = False
        character_name = ""
    if use_style and not (artist.strip() or style_tags):
        use_style = False
    prompt_lower = str(user_prompt or "").lower()
    try:
        max_content_tags = int(
            config.get("prompt_builder_max_content_tags", DEFAULT_MAX_CONTENT_TAGS)
        )
    except (TypeError, ValueError):
        max_content_tags = DEFAULT_MAX_CONTENT_TAGS
    if max_content_tags <= 0:
        max_content_tags = DEFAULT_MAX_CONTENT_TAGS
    allow_multi_character = force_multi_character or any(
        marker in prompt_lower
        for marker in (
            "2girls",
            "2 girls",
            "3girls",
            "3 girls",
            "multiple girls",
            "multiple people",
            "crowd",
            "group",
            "双人",
            "两人",
            "二人",
            "多人",
            "群像",
            "一群",
        )
    )
    has_non_user_content = bool(
        required_count_tags
        or required_core_tags
        or structured_character_tags
        or structured_copyright_tags
        or structured_identity_blocks
        or structured_detail_blocks
        or structured_tag_tags
        or narrative_blocks
        or nltags
    )
    fallback_to_user_prompt = not has_non_user_content
    content = clean_content_tags(
        llm_content or (user_prompt if fallback_to_user_prompt else ""),
        max_tags=max_content_tags,
        # Structured Identity/Details are prose sections, not a reliable set of
        # Danbooru equivalents.  Do not erase appearance tags merely because a
        # structured character exists; exact tag deduplication still happens in
        # clean_content_tags and join_prompt_parts.
        strip_character_tags=(
            not preserve_structured_order
            and (use_character or bool(required_core_tags))
        ),
        protected_core_tags=tuple(
            dict.fromkeys((*required_core_tags, *structured_tag_tags))
        ),
        allow_multi_character=allow_multi_character,
    )
    constraint_result = apply_prompt_constraints(content, constraint_plan)
    content = constraint_result.content_tags
    if background_mode == DEFAULT_PORTRAIT:
        content = apply_default_portrait_tags(
            content,
            include_full_body=not any(
                tag.lower().replace("_", " ") == "no humans"
                for tag in required_count_tags
            ),
        )
    parts = [quality]
    if preserve_structured_order:
        # The seven structured blocks have different semantics.  In particular,
        # Identity/Details are natural-language clauses and must never be folded
        # into the final Nltags paragraph or passed through comma-based tag
        # cleaning.  Keep their original block order all the way to ComfyUI:
        # Count -> Characters -> Copyright -> Identity -> Details -> Tags.
        if required_count_tags:
            parts.append(", ".join(required_count_tags))
        if structured_character_tags:
            parts.append(", ".join(structured_character_tags))
        if structured_copyright_tags:
            parts.append(", ".join(structured_copyright_tags))
        # Artist tags are global authorship/style anchors.  In structured mode
        # they belong immediately after Copyright and before character prose.
        if use_style and artist.strip():
            parts.append(artist)
        prefix = join_prompt_parts(parts)

        ordered_sections = [prefix]
        for blocks in (structured_identity_blocks, structured_detail_blocks):
            section = "; ".join(
                display_tag_text(str(block).strip())
                for block in blocks
                if str(block or "").strip()
            )
            if section:
                ordered_sections.append(section)

        tag_parts: list[str] = []
        if constraint_result.weighted_style_tags:
            tag_parts.append(", ".join(constraint_result.weighted_style_tags))
        if use_character:
            tag_parts.append(character)
        if use_style:
            if style_tags:
                tag_parts.append(style_tags)
        if structured_tag_tags:
            tag_parts.append(", ".join(structured_tag_tags))
        tag_parts.append(content or (user_prompt if fallback_to_user_prompt else ""))
        tag_section = join_prompt_parts(tag_parts)
        if tag_section:
            ordered_sections.append(tag_section)
        final_prompt = ", ".join(section for section in ordered_sections if section)
    else:
        if required_count_tags:
            parts.append(", ".join(required_count_tags))
        if required_core_tags:
            parts.append(", ".join(required_core_tags))
        if constraint_result.weighted_style_tags:
            parts.append(", ".join(constraint_result.weighted_style_tags))
        if use_character:
            parts.append(character)
        if use_style:
            if artist.strip():
                parts.append(artist)
            if style_tags:
                parts.append(style_tags)
        parts.append(content or (user_prompt if fallback_to_user_prompt else ""))
        final_prompt = join_prompt_parts(parts)
    narrative = tuple(
        str(block or "").strip()
        for block in narrative_blocks
        if str(block or "").strip()
    )
    if narrative:
        final_prompt += ", " + ", ".join(
            display_tag_text(block) for block in narrative
        )
    nltags = " ".join(str(nltags or "").split()).strip(" ,;:")
    if nltags:
        final_prompt += f", Nltags: {display_tag_text(nltags)}"
    return PromptBuildResult(
        final_prompt=final_prompt,
        content_tags=content,
        raw_mode=False,
        used_fixed_character=use_character,
        used_default_style=use_style,
        required_count_tags=tuple(required_count_tags),
        required_core_tags=tuple(required_core_tags),
        character_name=character_name,
        used_sensual_mode=use_sensual,
        constraint_mode=constraint_result.triggered,
        weighted_style_tags=constraint_result.weighted_style_tags,
        constraint_tags=constraint_result.priority_tags,
        removed_constraint_tags=constraint_result.removed_tags,
        constraint_reason=constraint_result.reason,
    )
