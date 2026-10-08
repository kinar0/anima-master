"""Deterministic checks against the user's original generation request."""

import unicodedata
from typing import Any


def matches_blocked_combination(prompt: str, rules: Any) -> bool:
    """Return whether every term of any configured ``&&`` rule appears."""
    if not isinstance(rules, (list, tuple)):
        return False
    text = unicodedata.normalize("NFKC", str(prompt or "")).casefold()
    for rule in rules:
        if not isinstance(rule, str) or "&&" not in rule:
            continue
        terms = [
            unicodedata.normalize("NFKC", part.strip()).casefold()
            for part in rule.split("&&")
        ]
        if len(terms) >= 2 and all(terms) and all(term in text for term in terms):
            return True
    return False


def matches_blocked_prompt(prompt: str, config: Any) -> bool:
    """Block when any configured pair has a match in both of its groups.

    Older two-list and ``&&`` rules remain active for compatibility.
    """
    if not isinstance(config, dict):
        return False
    text = unicodedata.normalize("NFKC", str(prompt or "")).casefold()

    def matching_terms(terms: Any) -> bool:
        if not isinstance(terms, (list, tuple)):
            return False
        return any(
            normalized in text
            for term in terms
            if isinstance(term, str)
            if (normalized := unicodedata.normalize("NFKC", term.strip()).casefold())
        )

    rules = config.get("blocked_prompt_rules", [])
    if isinstance(rules, (list, tuple)):
        for rule in rules:
            if not isinstance(rule, dict) or rule.get("__template_key") not in (
                None,
                "group_pair",
            ):
                continue
            if matching_terms(rule.get("group_a")) and matching_terms(
                rule.get("group_b")
            ):
                return True
    if matching_terms(config.get("blocked_prompt_group_a")) and matching_terms(
        config.get("blocked_prompt_group_b")
    ):
        return True
    return matches_blocked_combination(
        text, config.get("blocked_prompt_combinations", [])
    )
