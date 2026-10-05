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
