from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_RULE_SEPARATORS = ("=>", "->", "＝＞", "→")
_MARKER_SPLIT_RE = re.compile(r"[|｜]")


@dataclass(frozen=True)
class MatchedPromptRule:
    """One configured prompt instruction matched by the user's text."""

    marker: str
    instruction: str


def match_keyword_prompt_rules(
    user_prompt: str, config: dict[str, Any]
) -> tuple[MatchedPromptRule, ...]:
    """Return configured rules whose markers occur in the user's own text.

    Rules use ``marker1|marker2 => instruction``.  Matching is case-insensitive,
    and at most one result is returned for each configured rule.
    """
    if not bool(config.get("keyword_prompt_rules_enabled", True)):
        return ()
    configured = config.get("keyword_prompt_rules")
    if not isinstance(configured, list):
        return ()
    prompt = str(user_prompt or "").casefold()
    if not prompt:
        return ()

    matches: list[MatchedPromptRule] = []
    for raw_rule in configured[:50]:
        parsed = _parse_rule(raw_rule)
        if parsed is None:
            continue
        markers, instruction = parsed
        marker = next(
            (candidate for candidate in markers if candidate.casefold() in prompt),
            "",
        )
        if marker:
            matches.append(MatchedPromptRule(marker=marker, instruction=instruction))
    return tuple(matches)


def build_keyword_rule_block(rules: tuple[MatchedPromptRule, ...]) -> str:
    """Render matched rules as a forceful instruction block for the LLM."""
    if not rules:
        return ""
    lines = "\n".join(
        f'- 命中触发词“{rule.marker}”：{rule.instruction}' for rule in rules
    )
    return f"""

-----------
关键词强制规则（最高优先级，必须逐条执行）：
{lines}

这些规则是用户配置的硬性输出要求，不是建议。你必须在最终结果中逐条落实；若规则要求添加 tags，必须把相应英文 tags 明确写入对应字段，不得仅用近义词暗示、不得遗漏、不得擅自弱化或删除。若规则要求其他操作，也必须按原文完成。输出前逐条自检，但不要输出自检过程或解释。
"""


def _parse_rule(raw_rule: Any) -> tuple[tuple[str, ...], str] | None:
    text = str(raw_rule or "").strip()
    if not text:
        return None
    left = ""
    instruction = ""
    for separator in _RULE_SEPARATORS:
        if separator in text:
            left, instruction = text.split(separator, 1)
            break
    markers = tuple(
        marker.strip()
        for marker in _MARKER_SPLIT_RE.split(left)
        if marker.strip()
    )
    instruction = instruction.strip()
    if not markers or not instruction:
        return None
    return markers, instruction
