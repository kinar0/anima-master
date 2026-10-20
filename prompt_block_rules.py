"""Deterministic checks against the user's original generation request."""

import unicodedata
from typing import Any


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _matched_combination_terms(prompt: str, rules: Any) -> list[str]:
    if not isinstance(rules, (list, tuple)):
        return []
    text = _normalize(str(prompt or ""))
    matched = []
    for rule in rules:
        if not isinstance(rule, str) or "&&" not in rule:
            continue
        terms = [part.strip() for part in rule.split("&&")]
        if (
            len(terms) >= 2
            and all(terms)
            and all(_normalize(term) in text for term in terms)
        ):
            matched.extend(terms)
    return matched


def matches_blocked_combination(prompt: str, rules: Any) -> bool:
    """Return whether every term of any configured ``&&`` rule appears."""
    return bool(_matched_combination_terms(prompt, rules))


def find_blocked_prompt_terms(prompt: str, config: Any) -> list[str]:
    """Return only terms participating in fully triggered rules, deduplicated.

    Older two-list and ``&&`` rules remain active for compatibility.
    """
    if not isinstance(config, dict):
        return []
    text = _normalize(str(prompt or ""))

    def matching_terms(terms: Any) -> list[str]:
        if not isinstance(terms, (list, tuple)):
            return []
        return [
            term.strip()
            for term in terms
            if isinstance(term, str)
            if (normalized := _normalize(term.strip())) and normalized in text
        ]

    matched = []
    rules = config.get("blocked_prompt_rules", [])
    if isinstance(rules, (list, tuple)):
        for rule in rules:
            if not isinstance(rule, dict) or rule.get("__template_key") not in (
                None,
                "group_pair",
            ):
                continue
            left = matching_terms(rule.get("group_a"))
            right = matching_terms(rule.get("group_b"))
            if left and right:
                matched.extend(left + right)
    left = matching_terms(config.get("blocked_prompt_group_a"))
    right = matching_terms(config.get("blocked_prompt_group_b"))
    if left and right:
        matched.extend(left + right)
    matched.extend(
        _matched_combination_terms(text, config.get("blocked_prompt_combinations", []))
    )
    unique = {}
    for term in matched:
        unique.setdefault(_normalize(term), term)
    return list(unique.values())


def matches_blocked_prompt(prompt: str, config: Any) -> bool:
    """Block when any configured pair or legacy combination is triggered."""
    return bool(find_blocked_prompt_terms(prompt, config))


def blocked_prompt_message(prompt: str, config: Any) -> str:
    """Build the shared rejection reply, or return empty text when allowed."""
    terms = find_blocked_prompt_terms(prompt, config)
    if not terms:
        return ""
    words = "、".join(f"「{term}」" for term in terms)
    return (
        "请求包含禁止的词语组合，已拒绝生成。\n"
        f"本次命中词语：{words}。\n"
        "请修改后再试。"
    )
