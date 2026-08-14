from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:
    from .danbooru_resolver import DanbooruResolveOutcome, DanbooruResolver
    from .multi_person_prompt import (
        build_multi_person_plan_prompt,
        parse_multi_person_plan,
        render_multi_person_character,
    )
    from .outfit_transfer import (
        build_outfit_summary_prompt,
        build_outfit_transfer_block,
        detect_outfit_transfer,
        extract_reference_tag_text,
        filter_outfit_tags,
        preferred_search_prompt,
    )
    from .prompt_background import (
        DEFAULT_PORTRAIT,
        extract_background_mode,
    )
    from .prompt_builder import (
        build_final_prompt,
    )
    from .prompt_constraints import (
        build_constraint_plan_prompt,
        parse_constraint_plan,
    )
    from .prompt_presets import (
        apply_config_preset,
        fixed_character_tags,
        mentioned_fixed_characters,
        selected_fixed_character,
        strip_raw_prefix,
        wants_sensual_mode,
    )
    from .prompt_research import PromptResearcher
    from .prompt_templates import build_llm_prompt
    from .tag_cleaner import clean_content_tags, split_tags
except ImportError:  # pragma: no cover - fallback for direct script-style imports.
    from danbooru_resolver import DanbooruResolveOutcome, DanbooruResolver
    from multi_person_prompt import (
        build_multi_person_plan_prompt,
        parse_multi_person_plan,
        render_multi_person_character,
    )
    from outfit_transfer import (
        build_outfit_summary_prompt,
        build_outfit_transfer_block,
        detect_outfit_transfer,
        extract_reference_tag_text,
        filter_outfit_tags,
        preferred_search_prompt,
    )
    from prompt_background import (
        DEFAULT_PORTRAIT,
        extract_background_mode,
    )
    from prompt_builder import (
        build_final_prompt,
    )
    from prompt_constraints import (
        build_constraint_plan_prompt,
        parse_constraint_plan,
    )
    from prompt_presets import (
        apply_config_preset,
        fixed_character_tags,
        mentioned_fixed_characters,
        selected_fixed_character,
        strip_raw_prefix,
        wants_sensual_mode,
    )
    from prompt_research import PromptResearcher
    from prompt_templates import build_llm_prompt
    from tag_cleaner import clean_content_tags, split_tags


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
    """Return whether an expected prompt response is a Chinese refusal."""
    response = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(re.findall(r"[\u3400-\u9fff]", response)) < 4:
        return False
    refusal_patterns = (
        r"(?:抱歉|对不起|很遗憾).{0,48}(?:不能|无法|不可以|没法|不便)",
        r"(?:不能|无法|不可以|没法|不便).{0,36}"
        r"(?:满足|帮助|协助|生成|创作|提供|完成|处理|遵循|支持).{0,20}"
        r"(?:要求|请求|内容|指令)?",
        r"(?:拒绝|不能接受|无法接受).{0,24}(?:要求|请求|生成|创作|内容)",
    )
    return any(re.search(pattern, response) for pattern in refusal_patterns)


def extract_structured_prompt(
    text: str,
) -> tuple[tuple[str, ...], tuple[StructuredPromptCharacter, ...], str, str]:
    """Parse the unified character-first LLM response format.

    Args:
        text: LLM completion containing Count, Characters, per-character
            sections, shared tags, and Nltags.

    Returns:
        Roster count/relationship tags, character sections, shared scene tags,
        and natural-language tags. Empty character sections signal that a legacy
        tag-only completion was returned.
    """
    raw = str(text or "").replace("\r\n", "\n").strip()
    blocks = {
        key.lower(): value.strip()
        for key, value in re.findall(
            r"\{\s*(Count|Characters|Identity|Details|Tags|Nltags)\s*:\s*(.*?)\s*\}",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
    }
    if set(blocks) != {
        "count",
        "characters",
        "identity",
        "details",
        "tags",
        "nltags",
    }:
        return (), (), raw, ""
    roster_tags: list[str] = []
    has_count_tag = False
    for item in (item.strip() for item in blocks["count"].split(",") if item.strip()):
        normalized = item.lower()
        if re.fullmatch(r"(?:[1-9]\+?(?:girls?|boys?)|multiple (?:girls|boys|people))", normalized):
            roster_tags.append(item)
            has_count_tag = True
        elif normalized in {"solo", "duo", "hetero", "yuri", "yaoi"}:
            roster_tags.append(item)
    roster = [
        item.strip()
        for item in blocks["characters"].split(",")
        if item.strip()
    ]
    if not has_count_tag or not roster:
        return (), (), raw, ""

    def sentence_for(name: str, section: str) -> str:
        display = re.escape(name.replace("_", " "))
        source = re.escape(name)
        match = re.search(
            # A role name mentioned inside somebody else's clause (for example
            # "chihaya_anon lies in togawa_sakiko's arms") is not that
            # character's own Details entry.  Require a clause to *start* with
            # the role name, either at the section start or after a semicolon.
            rf"(?:^|;)\s*(?:{source}|{display})(?!\w)[^;]*",
            section,
            flags=re.IGNORECASE,
        )
        return match.group(0).strip(" ,;") if match else ""

    characters = tuple(
        StructuredPromptCharacter(
            name=name,
            identity_tags=sentence_for(name, blocks["identity"]),
            detail_tags=sentence_for(name, blocks["details"]),
        )
        for name in roster
    )
    if any(not character.identity_tags or not character.detail_tags for character in characters):
        return (), (), raw, ""
    return tuple(roster_tags), characters, blocks["tags"], blocks["nltags"]


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
    ) -> str:
        if character_name:
            character_rule = f"不要输出固定角色“{character_name}”的固有外观设定。"
        else:
            character_rule = (
                "用户没有使用固定角色时，如果用户明确点名现有作品角色，"
                "第一项必须输出最可信的标准 Danbooru 角色 tag，使用罗马字和下划线，必要时带作品消歧括号；"
                "禁止省略角色 tag 而只写外观，后续程序会联网查询 character 分类并校正。"
                "之后可以并且应该输出主体所需的固有外观设定。"
            )
        creative_rule = (
            "默认采用自由创作策略：在保留用户明确要求和角色身份的前提下，"
            "可以主动发展统一主题并补充有助于最终画面表现的可见内容，但不得在用户未提背景时创造场景；"
            "以协调、精致和好看为优先，不机械追求固定tag数量。"
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
                "Follow the user prompt's six-brace-block response format exactly. "
                "Count must contain the exact Danbooru people-count tag, while "
                "Characters contains names only; never omit Count or merge it into "
                "Characters."
                f"{character_rule}"
                f"{creative_rule}"
            ),
            "max_tokens": self._int("prompt_builder_max_tokens", 700),
        }
        if use_deep_thinking:
            kwargs["reasoning_effort"] = (
                self._str("prompt_builder_reasoning_effort", "high") or "high"
            )
            kwargs["thinking"] = {"type": "enabled"}
        response = await self.context.llm_generate(**kwargs)
        return str(getattr(response, "completion_text", "") or "").strip()

    async def _generate_outfit_summary_with_llm(
        self,
        *,
        provider_id: str,
        summary_prompt: str,
        use_deep_thinking: bool,
    ) -> str:
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
        }
        if use_deep_thinking:
            kwargs["reasoning_effort"] = (
                self._str("prompt_builder_reasoning_effort", "high") or "high"
            )
            kwargs["thinking"] = {"type": "enabled"}
        response = await self.context.llm_generate(**kwargs)
        return str(getattr(response, "completion_text", "") or "").strip()

    async def _generate_character_candidates_with_llm(
        self,
        *,
        provider_id: str,
        user_prompt: str,
        rejected_content: str,
        target_name: str = "",
    ) -> tuple[str, ...]:
        """Extract bounded Danbooru character candidates from the user request.

        Args:
            provider_id: Active AstrBot provider identifier.
            user_prompt: Original request containing the named character.
            rejected_content: Initial LLM tags that online lookup could not verify.
            target_name: Optional character name selected by a multi-person plan.

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
                "Do not invent a character when none is explicitly named.\n\n"
                f"Target name: {target_name or 'not separately specified'}\n"
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
        )
        raw = str(getattr(response, "completion_text", "") or "").strip()
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
        )
        return str(getattr(response, "completion_text", "") or "").strip()

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
        }
        if use_deep_thinking:
            kwargs["reasoning_effort"] = (
                self._str("prompt_builder_reasoning_effort", "high") or "high"
            )
            kwargs["thinking"] = {"type": "enabled"}
        response = await self.context.llm_generate(**kwargs)
        return str(getattr(response, "completion_text", "") or "").strip()

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
                if "1girl" in normalized_configured_tags:
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
            if visual_label and (fixed_name or resolution_status == "resolved"):
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
                    if resolution_status == "resolved"
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
            deterministic_count_tags = tuple(
                tag
                for tag in (
                    f"{girl_count}girls" if girl_count else "",
                    f"{boy_count}boys" if boy_count else "",
                )
                if tag
            )
        if not deterministic_count_tags:
            deterministic_count_tags = tuple(
                tag
                for tag in plan.count_tags
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
            )[:1] or (f"{character_count}people",)
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
        if plan.background_mode == DEFAULT_PORTRAIT:
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
                "background_mode": plan.background_mode,
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
    ) -> PromptPipelineResult:
        """Build the final prompt and summary for one generation request.

        Args:
            event: AstrBot message event for provider and per-chat config lookup.
            user_prompt: User prompt after reference-image augmentation.
            mode: Generation mode, such as `txt2img` or `img2img`.
            multi_person: Whether `/anm 多人` requested structured planning.
            original_user_prompt: User text before reference-context augmentation.

        Returns:
            Final prompt plus a serializable summary dict.
        """
        original_prompt = str(user_prompt or "").strip()
        background_intent_prompt = str(original_user_prompt or original_prompt).strip()
        legacy_creative_flag_re = re.compile(
            r"(?<!\S)--(?:自由发挥|自由拓展|创意拓展|创意扩展|creative)"
            r"(?=$|\s|[,，;；:：])",
            re.IGNORECASE,
        )
        prompt = legacy_creative_flag_re.sub(" ", original_prompt).strip()
        prompt = re.sub(r"^[\s,，;；:：]+|[\s,，;；:：]+$", "", prompt)
        prompt = re.sub(r"([,，;；])\s*[,，;；]+", r"\1", prompt)
        prompt = re.sub(r"\s+", " ", prompt)
        summary: dict[str, Any] = {
            "prompt_optimize_enabled": self._bool("prompt_optimize_enabled", True),
            "mode": mode,
            "multi_person_mode": bool(multi_person),
            "original_prompt_head": self._shorten(original_prompt, 600),
        }
        if not self._bool("prompt_optimize_enabled", True):
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

        prompt_config = apply_config_preset(dict(self.config))
        local_character_hints = mentioned_fixed_characters(prompt, prompt_config)
        fixed_character = selected_fixed_character(prompt, prompt_config)
        fixed_character_name = fixed_character[0] if fixed_character else ""

        provider_id = await self._current_chat_provider_id(event)
        if not provider_id:
            self.logger.warning(
                "[comfyui_agent] prompt builder has no provider; using original prompt"
            )
            summary.update(
                {
                    "skipped_reason": "no_chat_provider",
                    "llm_failed": True,
                    "llm_error": "no_chat_provider",
                    "final_prompt_head": self._shorten(prompt, 600),
                    "final_prompt_chars": len(prompt),
                }
            )
            return PromptPipelineResult(prompt, summary)

        use_fixed_character = fixed_character is not None
        use_sensual_mode = wants_sensual_mode(prompt, prompt_config)
        outfit_plan = detect_outfit_transfer(prompt, fixed_character_name)
        required_core_tags = (
            self._danbooru_resolver.required_core_tags_for_prompt(prompt)
            if not use_fixed_character
            else ()
        )
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
        if multi_person:
            # Preserve the explicit legacy command while ordinary generation uses
            # the unified character-first response protocol below.
            multi_result = await self._build_multi_person_prompt(
                provider_id=provider_id,
                prompt=prompt,
                prompt_config=prompt_config,
                use_deep_thinking=research_plan.use_deep_thinking,
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
        outfit_summary = ""
        outfit_summary_source = ""
        reference_tag_text = (
            extract_reference_tag_text(prompt) if outfit_plan.enabled else ""
        )
        if reference_tag_text:
            outfit_summary = filter_outfit_tags(reference_tag_text, max_tags=42)
            if outfit_summary:
                outfit_summary_source = "reference_filter"
        if outfit_plan.enabled and not outfit_summary and search_context:
            summary_prompt = build_outfit_summary_prompt(
                outfit_plan,
                original_prompt=prompt,
                source_context=search_context,
            )
            try:
                raw_outfit_summary = await self._generate_outfit_summary_with_llm(
                    provider_id=provider_id,
                    summary_prompt=summary_prompt,
                    use_deep_thinking=research_plan.use_deep_thinking,
                )
                outfit_summary = filter_outfit_tags(raw_outfit_summary, max_tags=48)
                if outfit_summary:
                    outfit_summary_source = "search_summary"
                if self._bool("debug_prompt_enabled", False):
                    self.logger.info(
                        "[comfyui_agent] outfit summary LLM output:\n%s",
                        raw_outfit_summary,
                    )
            except Exception as exc:
                self.logger.warning(
                    "[comfyui_agent] outfit summary build failed: %s", exc
                )
        llm_prompt = build_llm_prompt(
            prompt,
            search_context=search_context,
            # The structured protocol owns the full roster.  Do not tell the
            # LLM that a first matched character will be added later: doing so
            # makes its Identity/Details blocks incomplete in multi-person
            # requests.  The legacy fixed-character fallback remains below.
            fixed_character=False,
            character_name="",
            fixed_character_hints=local_character_hints,
            sensual_mode=use_sensual_mode,
            mode=mode,
            prompt_builder_template=self._str("prompt_builder_template", ""),
            outfit_transfer_rule=build_outfit_transfer_block(
                outfit_plan, outfit_summary
            ),
            original_theme=background_intent_prompt,
        )
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
                    )
                except Exception as retry_exc:
                    self.logger.warning(
                        "[comfyui_agent] prompt builder LLM failed: %s", retry_exc
                    )
                    llm_error = str(retry_exc)
                    llm_content = ""

        if is_chinese_model_refusal(llm_content):
            return self._model_refusal_result(summary, llm_content)

        if self._bool("debug_prompt_enabled", False):
            self.logger.info(
                "[comfyui_agent] prompt builder LLM output:\n%s", llm_content
            )
        (
            structured_roster_tags,
            structured_characters,
            structured_scene,
            structured_nltags,
        ) = (
            extract_structured_prompt(llm_content)
        )
        if not structured_characters and "_(" not in llm_content:
            strict_format_prompt = (
                llm_prompt
                + "\n\nYour previous response was invalid. Return exactly the six "
                "brace blocks {Count: ...} {Characters: ...} {Identity: ...} "
                "{Details: ...} {Tags: ...} {Nltags: ...}; Count must include an "
                "exact people-count tag and every named character must appear once "
                "in both Identity and Details."
            )
            try:
                retry_content = await self._generate_prompt_tags_with_llm(
                    provider_id=provider_id,
                    llm_prompt=strict_format_prompt,
                    use_deep_thinking=False,
                    fixed_character=False,
                    character_name="",
                )
                if is_chinese_model_refusal(retry_content):
                    return self._model_refusal_result(summary, retry_content)
                (
                    structured_roster_tags,
                    structured_characters,
                    structured_scene,
                    structured_nltags,
                ) = extract_structured_prompt(retry_content)
                if structured_characters:
                    llm_content = retry_content
                    summary["structured_format_retry"] = True
                else:
                    summary["structured_format_retry"] = False
            except Exception as exc:
                self.logger.warning(
                    "[comfyui_agent] structured prompt format retry failed: %s", exc
                )
                summary["structured_format_retry"] = False
        structured_character_mode = bool(structured_characters)
        if structured_character_mode:
            structured_scene, removed_unbound_tags = filter_unbound_directional_tags(
                structured_scene
            )
            rendered_characters: list[str] = []
            resolution_statuses: list[dict[str, Any]] = []
            used_fixed_names: set[str] = set()
            for character in structured_characters:
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
                declared_identity_tags = clean_content_tags(
                    character.identity_tags,
                    max_tags=12,
                    strip_character_tags=False,
                    allow_multi_character=True,
                )
                detail_tags = clean_content_tags(
                    character.detail_tags,
                    max_tags=24,
                    strip_character_tags=False,
                    allow_multi_character=True,
                )
                rendered = ", ".join(
                    (*identity_tags, declared_identity_tags, detail_tags)
                ).strip(" ,")
                if rendered:
                    rendered_characters.append(rendered)
                resolution_statuses.append(
                    {
                        "name": character.name,
                        "status": status,
                        "canonical_tag": canonical_tag,
                        "identity_tags": list(identity_tags),
                    }
                )
            llm_content = ", ".join(
                part
                for part in (
                    *structured_roster_tags,
                    *rendered_characters,
                    structured_scene,
                )
                if part
            )
            nltags = normalize_structured_nltags(
                structured_nltags,
                tuple(character.name for character in structured_characters),
            )
            summary["structured_character_mode"] = True
            summary["structured_character_count"] = len(structured_characters)
            summary["structured_roster_tags"] = list(structured_roster_tags)
            summary["removed_unbound_directional_tags"] = list(
                removed_unbound_tags
            )
            summary["character_resolution_statuses"] = resolution_statuses
            summary["local_character_hints"] = list(local_character_hints)
            use_fixed_character = False
            fixed_character_name = ""
            required_core_tags = ()
        else:
            llm_content, nltags = extract_nltags(llm_content)
            llm_content = re.sub(
                r"\{\s*(?:Count|Characters|Identity|Details|Tags|Nltags)\s*:[^}]*\}",
                "",
                llm_content,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip(" ,\n")
            summary["structured_character_mode"] = False
        background_mode = ""
        background_mode_source = "not_applicable"
        if mode == "txt2img":
            llm_content, background_mode = extract_background_mode(llm_content)
            if background_mode:
                background_mode_source = "llm_marker"
            else:
                background_mode = DEFAULT_PORTRAIT
                background_mode_source = "missing_marker_default"
        llm_failed = bool(llm_error and not str(llm_content or "").strip())
        if structured_character_mode:
            character_resolution = DanbooruResolveOutcome(text=llm_content)
        else:
            character_resolution = await self._danbooru_resolver.resolve_detailed(
                llm_content=llm_content,
                user_prompt=prompt,
                fixed_character=use_fixed_character,
            )
        if not structured_character_mode and not use_fixed_character and (
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
        if character_resolution.identity_tags:
            required_core_tags = character_resolution.identity_tags
        low_cfg_harness = bool(prompt_config.get("low_cfg_harness_enabled", False))
        constraint_raw = ""
        constraint_plan = parse_constraint_plan("")
        if low_cfg_harness:
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
        built = build_final_prompt(
            user_prompt=prompt,
            llm_content=llm_content,
            config=prompt_config,
            required_core_tags=required_core_tags,
            constraint_plan=constraint_plan,
            background_mode=background_mode,
            nltags=nltags,
            suppress_fixed_character=structured_character_mode,
            force_multi_character=structured_character_mode,
        )
        initial_input_tag_count = len(split_tags(llm_content))
        content_tag_count = len(split_tags(built.content_tags))
        removed_tag_count = max(0, initial_input_tag_count - content_tag_count)
        self.logger.info(
            "[comfyui_agent] prompt built raw=%s web_search=%s deep_thinking=%s character=%s sensual=%s fixed_character=%s default_style=%s low_cfg_harness=%s constraint=%s weighted_style=%s required_core_tags=%s content_tags=%s content_chars=%s final_chars=%s final_head=%s",
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
                "required_core_tags": list(built.required_core_tags),
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
                    "llm_prompt": llm_prompt,
                    "outfit_summary": outfit_summary,
                    "constraint_plan": constraint_raw,
                    "llm_content": llm_content,
                    "final_prompt": built.final_prompt,
                }
            )
        return PromptPipelineResult(built.final_prompt, summary)
