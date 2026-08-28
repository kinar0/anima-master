from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

try:
    from .danbooru_tags import (
        DEFAULT_DONMAI_BASE_URLS,
        DEFAULT_USER_AGENT,
        character_resolution_requested,
        fetch_variant_outfit_profile,
        profile_hints_for_prompt,
        required_core_tags_for_prompt,
        required_profile_tags_for_prompt,
        resolve_core_tags,
    )
    from .danbooru_semantic import (
        SemanticAnchor,
        SemanticLookupResult,
        lookup_semantic_anchors,
        resolve_local_cli_path,
    )
except ImportError:  # pragma: no cover - fallback for direct script-style imports.
    from danbooru_tags import (
        DEFAULT_DONMAI_BASE_URLS,
        DEFAULT_USER_AGENT,
        character_resolution_requested,
        fetch_variant_outfit_profile,
        profile_hints_for_prompt,
        required_core_tags_for_prompt,
        required_profile_tags_for_prompt,
        resolve_core_tags,
    )
    from danbooru_semantic import (
        SemanticAnchor,
        SemanticLookupResult,
        lookup_semantic_anchors,
        resolve_local_cli_path,
    )


@dataclass(frozen=True)
class DanbooruResolveOutcome:
    """Request-scoped character resolution result."""

    text: str
    status: str = "not_requested"
    canonical_tag: str = ""
    identity_tags: tuple[str, ...] = ()
    candidate_hints: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    explicit_request: bool = False


class WardrobeValidationError(ValueError):
    """Raised when a visual wardrobe editor payload is unsafe or stale."""


class DanbooruResolver:
    """Configuration-aware resolver for Danbooru character core tags."""

    def __init__(
        self,
        *,
        logger: Any,
        cache: dict[str, Any],
        get_bool: Callable[[str, bool], bool],
        get_int: Callable[[str, int], int],
        get_float: Callable[[str, float], float],
        get_str: Callable[[str, str], str],
        config: dict[str, Any] | None = None,
        profile_cache_path: Path | None = None,
    ):
        """Store Danbooru lookup dependencies.

        Args:
            logger: Logger compatible with AstrBot logger methods.
            cache: Shared timestamped Danbooru lookup cache.
            get_bool: Config boolean accessor.
            get_int: Config integer accessor.
            get_float: Config float accessor.
            get_str: Config string accessor.
        """
        self.logger = logger
        self._cache = cache
        self._bool = get_bool
        self._int = get_int
        self._float = get_float
        self._str = get_str
        self._config = dict(config or {})
        self._profile_cache_path = profile_cache_path
        self._profile_cache_data: dict[str, Any] | None = None

    @staticmethod
    def _profile_alias_key(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    @classmethod
    def _alias_match_spans(cls, text: str, alias: str) -> tuple[tuple[int, int], ...]:
        """Return trigger spans without letting short Latin aliases hit in words."""
        haystack = cls._profile_alias_key(text)
        needle = cls._profile_alias_key(alias)
        if not needle:
            return ()
        if re.search(r"[\u3400-\u9fff]", needle):
            return tuple(
                (match.start(), match.end())
                for match in re.finditer(re.escape(needle), haystack, flags=re.I)
            )
        return tuple(
            (match.start(), match.end())
            for match in re.finditer(
                rf"(?<![a-z0-9_]){re.escape(needle)}(?![a-z0-9_])",
                haystack,
                flags=re.I,
            )
        )

    @classmethod
    def _text_mentions_alias(cls, text: str, alias: str) -> bool:
        return bool(cls._alias_match_spans(text, alias))

    @staticmethod
    def _outfit_variant(value: Any) -> str:
        """Classify text into the small wardrobe-variant vocabulary used on disk."""
        text = str(value or "").strip().lower()
        if re.search(r"(?:冬季|冬装|冬服|winter)", text, flags=re.I):
            return "winter"
        if re.search(r"(?:夏季|夏装|夏服|summer)", text, flags=re.I):
            return "summer"
        if re.search(
            r"(?:演出服|舞台服|stage outfit|performance outfit|concert outfit)",
            text,
            flags=re.I,
        ):
            return "stage"
        if re.search(
            r"(?:官方常服|常服|"
            r"(?<![a-z0-9_])casual(?:\s+(?:clothes?|clothing|outfit|wear|attire|look))?"
            r"(?![a-z0-9_]))",
            text,
            flags=re.I,
        ):
            return "casual"
        return "default"

    @classmethod
    def _outfit_variant_near_span(
        cls, text: str, start: int, end: int
    ) -> str:
        """Classify the clause that owns one matched character/outfit alias.

        A request may assign different saved variants to different wearers, for
        example ``A穿常服，B穿默认服装``.  Classifying the complete request once
        collapses both records into a single variant, so isolate the nearest
        comma/semicolon/sentence-delimited clause around each alias instead.
        """
        source = str(text or "")
        left = max(
            (source.rfind(separator, 0, start) for separator in "，,。；;\n"),
            default=-1,
        )
        right_candidates = [
            position
            for separator in "，,。；;\n"
            if (position := source.find(separator, end)) >= 0
        ]
        right = min(right_candidates) if right_candidates else len(source)
        return cls._outfit_variant(source[left + 1 : right])

    @classmethod
    def _mapping_alias_is_negated(
        cls, text: str, start: int
    ) -> bool:
        """Return whether a configured noun occurrence is explicitly negated."""
        prefix = str(text or "")[max(0, start - 24) : start]
        return bool(
            re.search(
                r"(?:(?:不要|不穿|不戴|不带|不使用|别穿|别戴|别带|"
                r"去掉|移除|删除|没有|并非|不是)\s*(?:任何|一切)?|"
                r"\b(?:without|remove|delete|no|not)\b\s*(?:the|any|an?|one)?)\s*$",
                prefix,
                flags=re.I,
            )
        )

    @classmethod
    def _normalize_outfit_variant(cls, value: Any, *hints: Any) -> str:
        """Keep a valid stored variant or infer one from legacy textual hints."""
        explicit = str(value or "").strip().lower()
        if explicit in {"default", "casual", "summer", "winter", "stage"}:
            return explicit
        return cls._outfit_variant(" ".join(str(hint or "") for hint in hints))

    @classmethod
    def _outfit_profile_key(cls, source_text: str, variant: str) -> str:
        """Return the cache key that isolates non-default seasonal profiles."""
        source = cls._profile_alias_key(source_text)
        return source if variant == "default" else f"{source}::{variant}"

    @classmethod
    def _alias_fits_named_outfit(
        cls, alias: str, canonical_tag: str, variant: str
    ) -> bool:
        """Reject learned aliases that would erase outfit identity or season."""
        alias_key = cls._profile_alias_key(alias)
        tag_key = str(canonical_tag or "").strip().lower()
        if not alias_key or alias_key == tag_key:
            return False
        if tag_key.endswith("_uniform") and not re.search(
            r"(?:校服|制服|uniform)", alias_key, flags=re.I
        ):
            return False
        alias_variant = cls._outfit_variant(alias_key)
        if variant == "default":
            return alias_variant == "default"
        return alias_variant == variant

    @staticmethod
    def _variant_english_alias(canonical_tag: str, variant: str) -> str:
        spaced = canonical_tag.replace("_", " ")
        if variant in {"summer", "winter"} and spaced.endswith(" school uniform"):
            return spaced.removesuffix(" uniform") + f" {variant} uniform"
        if variant in {"casual", "summer", "winter", "stage"}:
            return f"{spaced} {variant}"
        return spaced

    @classmethod
    def _named_outfit_aliases(
        cls,
        primary: str,
        canonical_tag: str,
        *collections: Any,
        variant: str = "default",
    ) -> list[str]:
        """Build stable trigger aliases without treating them as separate sets.

        Aliases entered by the user are authoritative. Learned aliases are
        screened separately by ``_safe_learned_outfit_aliases``.
        """
        variant = cls._normalize_outfit_variant(variant, primary, *collections)
        values: list[str] = [primary]
        for collection in collections:
            if isinstance(collection, (list, tuple)):
                values.extend(str(item) for item in collection)
        values.append(cls._variant_english_alias(canonical_tag, variant))
        canonical_key = cls._profile_alias_key(canonical_tag)
        aliases: list[str] = []
        seen: set[str] = set()
        for value in values:
            alias = cls._profile_alias_key(value)
            if (
                alias
                and alias != canonical_key
                and alias not in seen
            ):
                seen.add(alias)
                aliases.append(alias)
        return aliases

    @classmethod
    def _preferred_named_outfit_alias(
        cls,
        profile_key: str,
        aliases: list[str],
        canonical_tag: str,
        variant: str = "default",
    ) -> str:
        """Choose one human-facing alias while retaining all triggers in storage."""
        generated = {cls._profile_alias_key(canonical_tag)}
        primary = aliases[0] if aliases else profile_key
        candidates = cls._named_outfit_aliases(
            primary, canonical_tag, aliases, variant=variant
        )
        human = [item for item in candidates if item not in generated]
        if not human:
            return cls._profile_alias_key(profile_key) or canonical_tag
        # Preserve the editor's explicit alias order.  Re-sorting by shortest
        # label made the visible primary alias (and therefore the persisted
        # profile key) change again immediately after a successful save.
        return human[0]

    def _profile_data(self) -> dict[str, Any]:
        """Load the versioned profile cache once, or return an empty v3 cache.

        Version 3 stores ``{"version": 3, "profiles": {key: record}}``. A
        ``character_outfit`` record contains ``qualifier``, trigger ``aliases``,
        donor-identity ``source_tags``, related ``copyright_tags``, copyable
        ``outfit_tags``, and optional post-derived ``evidence``. A ``named_outfit``
        record contains ``variant``, learned ``aliases``, editor-owned
        ``manual_aliases``, and one complete ensemble tag in ``outfit_tags``.
        Non-default character profiles use ``source::variant`` keys so summer,
        winter, stage, and casual evidence cannot overwrite the default wardrobe.

        Missing, corrupt, and obsolete files are deliberately treated alike. They
        must not block image generation, and older schemas are unsafe to interpret
        as current wardrobe evidence.
        """
        if self._profile_cache_data is not None:
            return self._profile_cache_data
        data: dict[str, Any] = {"version": 3, "profiles": {}}
        path = self._profile_cache_path
        if path is not None and path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if (
                    isinstance(loaded, dict)
                    and loaded.get("version") == 3
                    and isinstance(loaded.get("profiles"), dict)
                ):
                    data = loaded
            except (OSError, ValueError):
                pass
        self._profile_cache_data = data
        return data

    def _save_profile_data(self) -> None:
        """Best-effort persist the in-memory profile cache via file replacement."""
        path = self._profile_cache_path
        if path is None or self._profile_cache_data is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(self._profile_cache_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError as exc:
            self.logger.warning(
                "[comfyui_agent] failed to save Danbooru profile cache: %s", exc
            )

    def cached_outfit_source(self, source_text: str) -> SemanticLookupResult | None:
        """Return a persistent outfit-source profile without invoking an LLM."""
        variant = self._outfit_variant(source_text)
        profiles = self._profile_data().get("profiles", {})
        key = self._outfit_profile_key(source_text, variant)
        profile = profiles.get(key)
        if not isinstance(profile, dict):
            profile = next(
                (
                    candidate
                    for profile_key, candidate in profiles.items()
                    if isinstance(candidate, dict)
                    and candidate.get("kind") != "named_outfit"
                    and self._normalize_outfit_variant(
                        candidate.get("qualifier"),
                        profile_key,
                        candidate.get("aliases", []),
                    )
                    == variant
                    and any(
                        self._profile_alias_key(source_text)
                        == self._profile_alias_key(alias)
                        for alias in (profile_key, *candidate.get("aliases", []))
                    )
                ),
                None,
            )
        if not isinstance(profile, dict):
            return None
        source_tags = tuple(
            str(tag) for tag in profile.get("source_tags", []) if str(tag).strip()
        )
        if not source_tags:
            return None
        confirmed = tuple(
            dict.fromkeys(
                (
                    *source_tags,
                    *(
                        str(tag)
                        for tag in profile.get("copyright_tags", [])
                        if str(tag).strip()
                    ),
                )
            )
        )
        return SemanticLookupResult(
            confirmed_tags=confirmed,
            outfit_source_tags=source_tags,
            outfit_profile_tags=tuple(
                str(tag)
                for tag in profile.get("outfit_tags", [])
                if str(tag).strip()
            ),
            status="profile_cache",
        )

    def cached_outfit_profiles_for_prompt(
        self, user_prompt: str
    ) -> SemanticLookupResult | None:
        """Return every editable clothing profile explicitly named in a prompt."""
        text = str(user_prompt or "")
        confirmed: list[str] = []
        outfit_tags: list[str] = []
        appearance_tags: list[str] = []
        scoped: list[tuple[str, str, tuple[str, ...], str]] = []
        scoped_appearances: dict[
            str, tuple[list[str], list[str]]
        ] = {}
        profile_items = [
            (str(profile_key), profile)
            for profile_key, profile in self._profile_data().get("profiles", {}).items()
            if isinstance(profile, dict) and profile.get("kind") != "named_outfit"
        ]
        occurrence_profiles: dict[
            tuple[int, int, str], list[tuple[str, str]]
        ] = {}
        for profile_key, profile in profile_items:
            aliases = tuple(
                dict.fromkeys(
                    self._profile_alias_key(alias)
                    for alias in (
                        str(profile_key).split("::", 1)[0],
                        *profile.get("aliases", []),
                    )
                    if self._profile_alias_key(alias)
                )
            )
            for alias in aliases:
                for start, end in self._alias_match_spans(text, alias):
                    occurrence_profiles.setdefault(
                        (start, end, alias), []
                    ).append((profile_key, alias))
        accepted_spans: list[tuple[int, int]] = []
        matched_alias_by_profile: dict[str, str] = {}
        matched_span_by_profile: dict[str, tuple[int, int]] = {}
        for (start, end, _alias), owners in sorted(
            occurrence_profiles.items(),
            key=lambda item: (-(item[0][1] - item[0][0]), item[0][0], item[0][2]),
        ):
            if any(start < used_end and used_start < end for used_start, used_end in accepted_spans):
                continue
            accepted_spans.append((start, end))
            for profile_key, alias in owners:
                matched_alias_by_profile.setdefault(profile_key, alias)
                matched_span_by_profile.setdefault(profile_key, (start, end))

        # A character-owned variant must not require one contiguous composite
        # alias such as ``千早爱音演出服``.  Once a character alias is grounded in
        # its local clause, use the requested qualifier to select the unique
        # sibling profile with the same canonical source tag.  This turns
        # ``千早爱音穿着演出服`` into (chihaya_anon, stage) while keeping
        # ``千早爱音站在舞台上`` on the default profile.
        directly_matched = tuple(matched_alias_by_profile.items())
        profile_by_key = dict(profile_items)
        for matched_key, matched_alias in directly_matched:
            matched_profile = profile_by_key.get(matched_key)
            matched_span = matched_span_by_profile.get(matched_key)
            if not isinstance(matched_profile, dict) or matched_span is None:
                continue
            requested_variant = self._outfit_variant_near_span(
                text, *matched_span
            )
            matched_sources = {
                self._profile_alias_key(tag)
                for tag in matched_profile.get("source_tags", [])
                if str(tag).strip()
            }
            if not matched_sources:
                continue
            sibling_keys = [
                candidate_key
                for candidate_key, candidate in profile_items
                if candidate_key != matched_key
                and self._normalize_outfit_variant(
                    candidate.get("qualifier"),
                    candidate_key,
                    candidate.get("aliases", []),
                )
                == requested_variant
                and matched_sources.intersection(
                    self._profile_alias_key(tag)
                    for tag in candidate.get("source_tags", [])
                    if str(tag).strip()
                )
            ]
            if len(sibling_keys) != 1:
                continue
            sibling_key = sibling_keys[0]
            matched_alias_by_profile.setdefault(sibling_key, matched_alias)
            matched_span_by_profile.setdefault(sibling_key, matched_span)

        for profile_key, profile in profile_items:
            if not isinstance(profile, dict) or profile.get("kind") == "named_outfit":
                continue
            variant = self._normalize_outfit_variant(
                profile.get("qualifier"), profile_key, profile.get("aliases", [])
            )
            aliases = tuple(
                dict.fromkeys(
                    self._profile_alias_key(alias)
                    for alias in (
                        str(profile_key).split("::", 1)[0],
                        *profile.get("aliases", []),
                    )
                    if self._profile_alias_key(alias)
                )
            )
            matched_alias = matched_alias_by_profile.get(str(profile_key), "")
            if not matched_alias:
                continue
            matched_span = matched_span_by_profile.get(str(profile_key))
            requested_variant = (
                self._outfit_variant_near_span(text, *matched_span)
                if matched_span is not None
                else self._outfit_variant(text)
            )
            sources = tuple(
                str(tag).strip()
                for tag in profile.get("source_tags", [])
                if str(tag).strip()
            )
            tags = tuple(
                str(tag).strip()
                for tag in profile.get("outfit_tags", [])
                if str(tag).strip()
            )
            evidence = (
                profile.get("evidence")
                if isinstance(profile.get("evidence"), dict)
                else {}
            )
            appearances = tuple(
                str(tag).strip()
                for tag in evidence.get("appearance_tags", [])
                if str(tag).strip()
            )
            for tag in (*sources, *profile.get("copyright_tags", [])):
                value = str(tag).strip()
                if value and value not in confirmed:
                    confirmed.append(value)
            for tag in appearances:
                if tag not in appearance_tags:
                    appearance_tags.append(tag)
            source_tag = sources[0] if sources else ""
            if source_tag and appearances:
                alias_bucket, appearance_bucket = scoped_appearances.setdefault(
                    source_tag, ([], [])
                )
                for alias in aliases:
                    if alias not in alias_bucket:
                        alias_bucket.append(alias)
                for tag in appearances:
                    if tag not in appearance_bucket:
                        appearance_bucket.append(tag)
            if variant != requested_variant:
                continue
            for tag in tags:
                if tag not in outfit_tags:
                    outfit_tags.append(tag)
            scoped.append((matched_alias, source_tag, tags, variant))
        if not scoped and not scoped_appearances:
            return None
        return SemanticLookupResult(
            confirmed_tags=tuple(confirmed),
            outfit_profile_tags=tuple(outfit_tags),
            appearance_profile_tags=tuple(appearance_tags),
            source_outfit_profiles=tuple(scoped),
            character_appearance_profiles=tuple(
                (tuple(aliases), source_tag, tuple(tags))
                for source_tag, (aliases, tags) in scoped_appearances.items()
            ),
            status="profile_cache",
        )

    def prefer_cached_character_anchors(
        self, anchors: tuple[SemanticAnchor, ...]
    ) -> tuple[SemanticAnchor, ...]:
        """Let the best persisted trigger alias determine character identity.

        Planner candidates are only lookup guesses.  Once an editor-visible visual
        profile owns a target/source phrase, its validated ``source_tags`` must be
        queried instead.  Matching is role-local, variant-local, and longest-first
        so ``重音teto`` cannot eclipse ``重音tetosv`` inside the same anchor phrase.
        """
        profiles = self._profile_data().get("profiles", {})
        preferred: list[SemanticAnchor] = []
        for anchor in anchors:
            if anchor.role not in {"target_character", "outfit_source"}:
                preferred.append(anchor)
                continue
            source_key = self._profile_alias_key(anchor.source_text)
            qualifier = self._outfit_variant(
                f"{anchor.source_text} {anchor.description}"
            )
            matches: list[tuple[int, str, dict[str, Any]]] = []
            for profile_key, profile in profiles.items():
                if not isinstance(profile, dict) or profile.get("kind") == "named_outfit":
                    continue
                if self._normalize_outfit_variant(
                    profile.get("qualifier"), profile_key, profile.get("aliases", [])
                ) != qualifier:
                    continue
                aliases = {
                    self._profile_alias_key(str(profile_key).split("::", 1)[0]),
                    *(
                        self._profile_alias_key(alias)
                        for alias in profile.get("aliases", [])
                    ),
                }
                matched_aliases = [
                    alias
                    for alias in aliases
                    if alias and self._text_mentions_alias(source_key, alias)
                ]
                if matched_aliases:
                    matches.append(
                        (max(len(alias) for alias in matched_aliases), str(profile_key), profile)
                    )
            if not matches:
                preferred.append(anchor)
                continue

            longest = max(length for length, _profile_key, _profile in matches)
            matches = [match for match in matches if match[0] == longest]

            distinct_sources = {
                tuple(
                    str(tag).strip().lower()
                    for tag in profile.get("source_tags", [])
                    if str(tag).strip()
                )
                for _length, _profile_key, profile in matches
            }
            distinct_sources.discard(())
            chosen: tuple[str, ...] = ()
            if len(distinct_sources) == 1:
                chosen = next(iter(distinct_sources))
            elif distinct_sources:
                # Existing poisoned caches may contain one alias on a base character
                # and a scoped variant.  A suffix such as ``(sv)`` is authoritative
                # only when that suffix is visibly present in the exact source phrase.
                compact_source = re.sub(r"[^a-z0-9]+", "", source_key)
                scored: list[tuple[int, tuple[str, ...]]] = []
                for source_tags in distinct_sources:
                    score = 0
                    for tag in source_tags:
                        scoped = re.fullmatch(r".+_\(([^)]+)\)", tag)
                        if not scoped:
                            continue
                        suffix = re.sub(r"[^a-z0-9]+", "", scoped.group(1).lower())
                        if suffix and suffix in compact_source:
                            score += len(suffix)
                    scored.append((score, source_tags))
                best_score = max((score for score, _tags in scored), default=0)
                winners = [tags for score, tags in scored if score == best_score and score > 0]
                if len(winners) == 1:
                    chosen = winners[0]
            if chosen:
                preferred.append(replace(anchor, candidates=chosen))
            else:
                self.logger.warning(
                    "[comfyui_agent] ambiguous cached character trigger alias=%s sources=%s; "
                    "keeping planner candidates without learning another alias mapping",
                    anchor.source_text,
                    sorted(distinct_sources),
                )
                preferred.append(anchor)
        return tuple(preferred)

    def prefer_cached_outfit_source_anchors(
        self, anchors: tuple[SemanticAnchor, ...]
    ) -> tuple[SemanticAnchor, ...]:
        """Backward-compatible name for the generalized character-anchor pass."""
        return self.prefer_cached_character_anchors(anchors)

    def outfit_source_refresh_needed(self, source_text: str) -> bool:
        """Refresh only absent, legacy, or week-old character outfit evidence."""
        variant = self._outfit_variant(source_text)
        key = self._outfit_profile_key(source_text, variant)
        profiles = self._profile_data().get("profiles", {})
        profile = profiles.get(key)
        if not isinstance(profile, dict):
            profile = next(
                (
                    candidate
                    for profile_key, candidate in profiles.items()
                    if isinstance(candidate, dict)
                    and candidate.get("kind") != "named_outfit"
                    and self._normalize_outfit_variant(
                        candidate.get("qualifier"),
                        profile_key,
                        candidate.get("aliases", []),
                    )
                    == variant
                    and any(
                        self._profile_alias_key(source_text)
                        == self._profile_alias_key(alias)
                        for alias in (profile_key, *candidate.get("aliases", []))
                    )
                ),
                None,
            )
        if not isinstance(profile, dict):
            return True
        return self._stored_profile_refresh_needed(profile)

    @staticmethod
    def _stored_profile_refresh_needed(profile: dict[str, Any]) -> bool:
        """Return whether one persisted visual profile needs evidence refresh."""
        evidence = profile.get("evidence")
        if not isinstance(evidence, dict) or evidence.get("algorithm_version") != 4:
            return True
        try:
            updated_at = float(evidence.get("updated_at") or 0.0)
        except (TypeError, ValueError):
            return True
        return time.time() - updated_at >= 7 * 86400

    def _stored_profile_for_source_tag(
        self, source_tag: str, qualifier: str
    ) -> dict[str, Any] | None:
        """Find one variant-local editable profile by validated donor identity."""
        source_key = self._profile_alias_key(source_tag)
        matches = [
            profile
            for profile_key, profile in self._profile_data().get("profiles", {}).items()
            if isinstance(profile, dict)
            and profile.get("kind") != "named_outfit"
            and self._normalize_outfit_variant(
                profile.get("qualifier"), profile_key, profile.get("aliases", [])
            )
            == qualifier
            and source_key
            in {
                self._profile_alias_key(tag)
                for tag in profile.get("source_tags", [])
                if str(tag).strip()
            }
        ]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _stored_profile_components(
        profile: dict[str, Any],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Read editor-authoritative outfit and stable appearance components."""
        evidence = profile.get("evidence")
        evidence = evidence if isinstance(evidence, dict) else {}
        return (
            tuple(
                str(tag).strip()
                for tag in profile.get("outfit_tags", [])
                if str(tag).strip()
            ),
            tuple(
                str(tag).strip()
                for tag in evidence.get("appearance_tags", [])
                if str(tag).strip()
            ),
        )

    @classmethod
    def _trusted_uniform_alias_tags(
        cls, profiles: dict[str, Any]
    ) -> dict[str, str]:
        """Map evidence-backed uniform aliases to their known canonical tag."""
        trusted: dict[str, str] = {}
        for key, profile in profiles.items():
            if not isinstance(profile, dict) or profile.get("kind") == "named_outfit":
                continue
            source_tag = next(
                (
                    str(tag).strip().lower()
                    for tag in profile.get("source_tags", [])
                    if str(tag).strip().lower().endswith("_uniform")
                ),
                "",
            )
            if not source_tag:
                continue
            for alias in (
                str(key).split("::", 1)[0],
                *profile.get("aliases", []),
            ):
                alias_key = cls._profile_alias_key(alias)
                if alias_key:
                    trusted[alias_key] = source_tag
        return trusted

    @classmethod
    def _named_profile_variant(cls, key: str, profile: dict[str, Any]) -> str:
        """Read a named outfit's variant with compatibility for legacy records."""
        return cls._normalize_outfit_variant(
            profile.get("variant"), key, profile.get("aliases", [])
        )

    @classmethod
    def _safe_learned_outfit_aliases(
        cls,
        key: str,
        profile: dict[str, Any],
        canonical_tag: str,
        variant: str,
        trusted_uniforms: dict[str, str],
    ) -> list[str]:
        """Drop aliases that lack outfit evidence or conflict with a trusted profile."""
        key_alias = cls._profile_alias_key(str(key).split("::", 1)[0])
        generated = cls._profile_alias_key(
            cls._variant_english_alias(canonical_tag, variant)
        )
        # Aliases explicitly saved in the editor are user-owned data.  They
        # must survive the stricter screening used for machine-learned aliases.
        # Older cache records have no ``manual_aliases`` field and therefore
        # still receive the legacy poison-data cleanup below.
        safe: list[str] = []
        for raw_alias in profile.get("manual_aliases", []):
            alias = cls._profile_alias_key(raw_alias)
            if alias and alias != cls._profile_alias_key(canonical_tag) and alias not in safe:
                safe.append(alias)
        for raw_alias in (key_alias, *profile.get("aliases", [])):
            alias = cls._profile_alias_key(raw_alias)
            if not cls._alias_fits_named_outfit(alias, canonical_tag, variant):
                continue
            if trusted_uniforms.get(alias, canonical_tag) != canonical_tag:
                continue
            if alias == generated and key_alias != generated:
                continue
            if alias not in safe:
                safe.append(alias)
        if not safe:
            return []
        return cls._named_outfit_aliases(
            safe[0], canonical_tag, safe, variant=variant
        )

    def cached_named_outfits_for_prompt(
        self, user_prompt: str
    ) -> SemanticLookupResult | None:
        """Resolve complete outfit-set aliases using user config before learned data.

        Explicit configured mappings are checked first and therefore own a
        duplicate alias/variant anchor identity. Persisted learned records are
        considered only after poison-data screening; within either source an alias
        mention wins, while canonical spelling is a variant-compatible fallback.
        Canonical tags are deduplicated, but conflicting distinct tags are retained
        as separate evidence rather than silently choosing one.
        """
        text = str(user_prompt or "").lower()
        requested_variant = self._outfit_variant(text)
        matched: list[str] = []
        matched_anchors: list[SemanticAnchor] = []
        anchor_identities: set[tuple[str, str]] = set()

        def register_match(alias: str, tag: str, variant: str) -> None:
            if tag not in matched:
                matched.append(tag)
            identity = (self._profile_alias_key(alias), variant)
            if identity in anchor_identities:
                return
            anchor_identities.add(identity)
            matched_anchors.append(
                SemanticAnchor(
                    anchor_id=f"cached_named_outfit_{len(matched_anchors) + 1}",
                    role="outfit",
                    group="outfit",
                    source_text=alias,
                    description=f"{variant} named outfit: {alias}",
                    candidates=(tag,),
                )
            )

        for tag, variant, aliases in self._configured_named_outfit_groups():
            matched_alias = next(
                (
                    alias
                    for alias in aliases
                    if self._text_mentions_alias(text, alias)
                ),
                "",
            )
            if (
                not matched_alias
                and variant == requested_variant
                and self._text_mentions_alias(text, tag.replace("_", " "))
            ):
                matched_alias = tag.replace("_", " ")
            if matched_alias:
                register_match(matched_alias, tag, variant)
        profile_tags_for_request: list[str] = []
        profiles = self._profile_data().get("profiles", {})
        trusted_uniforms = self._trusted_uniform_alias_tags(profiles)
        for alias, profile in profiles.items():
            if not isinstance(profile, dict) or profile.get("kind") != "named_outfit":
                continue
            profile_tags = tuple(
                str(tag).strip().lower()
                for tag in profile.get("outfit_tags", [])
                if str(tag).strip()
            )
            # Legacy planners could persist the generic alias 校服 ->
            # school_uniform as though it were an independently named set. Ignore
            # that poisoned entry so it cannot match every later school request.
            if set(profile_tags).issubset({"school_uniform", "uniform"}):
                continue
            canonical = profile_tags[0] if profile_tags else ""
            variant = self._named_profile_variant(str(alias), profile)
            aliases = self._safe_learned_outfit_aliases(
                str(alias), profile, canonical, variant, trusted_uniforms
            )
            alias_matches = any(
                self._text_mentions_alias(text, value) for value in aliases
            )
            derived_matches = variant == requested_variant and (
                self._text_mentions_alias(text, canonical)
                or self._text_mentions_alias(text, canonical.replace("_", " "))
            )
            if not derived_matches and not alias_matches:
                continue
            matched_alias = next(
                (
                    value
                    for value in aliases
                    if self._text_mentions_alias(text, value)
                ),
                str(alias).split("::", 1)[0],
            )
            register_match(matched_alias, canonical, variant)
            for tag in profile.get("outfit_tags", []):
                value = str(tag).strip()
                if value and value not in matched:
                    matched.append(value)
            for candidate in profiles.values():
                if not isinstance(candidate, dict) or candidate.get("kind") == "named_outfit":
                    continue
                candidate_variant = self._normalize_outfit_variant(
                    candidate.get("qualifier"), candidate.get("aliases", [])
                )
                if candidate_variant != variant:
                    continue
                if canonical not in {
                    str(tag).strip().lower()
                    for tag in candidate.get("source_tags", [])
                }:
                    continue
                for tag in candidate.get("outfit_tags", []):
                    value = str(tag).strip()
                    if value and value not in profile_tags_for_request:
                        profile_tags_for_request.append(value)
        if not matched:
            return None
        anchor_outfit_profiles = ()
        if len(matched_anchors) == 1 and profile_tags_for_request:
            anchor = matched_anchors[0]
            anchor_outfit_profiles = ((
                anchor.anchor_id,
                anchor.candidates[0] if anchor.candidates else "",
                tuple(profile_tags_for_request),
                requested_variant,
            ),)
        return SemanticLookupResult(
            confirmed_tags=tuple(matched),
            named_outfit_tags=tuple(matched),
            outfit_profile_tags=tuple(profile_tags_for_request),
            anchors=tuple(matched_anchors),
            anchor_outfit_profiles=anchor_outfit_profiles,
            status="profile_cache",
        )

    def _configured_named_outfits(self) -> dict[str, str]:
        """Parse user-defined alias=canonical_tag mappings from plugin config."""
        return {
            alias.lower(): tag
            for tag, _variant, aliases in self._configured_named_outfit_groups()
            for alias in aliases
        }

    def _configured_named_outfit_groups(
        self,
    ) -> list[tuple[str, str, list[str]]]:
        """Parse one canonical outfit record with any number of trigger aliases."""
        configured = self._config.get("danbooru_named_outfit_mappings", [])
        raw_pairs: list[tuple[str, str]] = []
        if isinstance(configured, dict):
            raw_pairs.extend(
                (str(alias), str(tag)) for alias, tag in configured.items()
            )
        elif isinstance(configured, (list, tuple)):
            for item in configured:
                match = re.fullmatch(
                    r"\s*(.+?)\s*(?:=>|=|＝|→|：|:)\s*(.+?)\s*",
                    str(item or ""),
                )
                if match:
                    raw_pairs.append((match.group(1), match.group(2)))
        groups: list[tuple[str, str, list[str]]] = []
        for raw_aliases, raw_tag in raw_pairs:
            tag = str(raw_tag or "").strip().lower().replace(" ", "_")
            if not re.fullmatch(r"[a-z0-9_.'():!\-]{2,120}", tag):
                continue
            aliases = [
                self._profile_alias_key(alias)
                for alias in re.split(r"\s*[|｜]\s*", str(raw_aliases or ""))
                if self._profile_alias_key(alias)
            ]
            if not aliases or any(len(alias) > 80 for alias in aliases):
                continue
            variant = self._normalize_outfit_variant("", aliases)
            groups.append((tag, variant, list(dict.fromkeys(aliases))))
        return groups

    def _configured_mappings(self, key: str) -> dict[str, str]:
        """Parse a bounded config mapping list into normalized aliases and tags."""
        configured = self._config.get(key, [])
        pairs: list[tuple[Any, Any]] = []
        if isinstance(configured, dict):
            pairs.extend(configured.items())
        elif isinstance(configured, (list, tuple)):
            for item in configured:
                match = re.fullmatch(
                    r"\s*(.+?)\s*(?:=>|=|＝|→|：|:)\s*(.+?)\s*",
                    str(item or ""),
                )
                if match:
                    pairs.append((match.group(1), match.group(2)))
        mappings: dict[str, str] = {}
        for raw_alias, raw_tag in pairs:
            alias = str(raw_alias or "").strip().lower()
            tag = str(raw_tag or "").strip().lower().replace(" ", "_")
            if not alias or len(alias) > 80:
                continue
            if not re.fullmatch(r"[a-z0-9_.'():!\-]{2,120}", tag):
                continue
            mappings[alias] = tag
        return mappings

    @staticmethod
    def _scope_compatible(left: str, right: str) -> bool:
        """Return whether two copyright scopes belong to the same tag family."""
        left_key = str(left or "").strip().lower().replace(" ", "_")
        right_key = str(right or "").strip().lower().replace(" ", "_")
        return bool(
            left_key
            and right_key
            and (
                left_key == right_key
                or left_key.startswith(right_key + "_")
                or right_key.startswith(left_key + "_")
            )
        )

    def _configured_alias_groups(
        self,
        key: str,
        defaults: tuple[str, ...] = (),
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Parse ``alias 1 | alias 2=canonical_tag`` records."""
        configured = self._config.get(key, defaults)
        raw_pairs: list[tuple[str, str]] = []
        if isinstance(configured, dict):
            raw_pairs.extend(
                (str(raw_aliases), str(tag))
                for raw_aliases, tag in configured.items()
            )
        elif isinstance(configured, (list, tuple)):
            for item in configured:
                match = re.fullmatch(
                    r"\s*(.+?)\s*(?:=>|=|＝|→|：|:)\s*(.+?)\s*",
                    str(item or ""),
                )
                if match:
                    raw_pairs.append((match.group(1), match.group(2)))
        groups: list[tuple[str, tuple[str, ...]]] = []
        for raw_aliases, raw_tag in raw_pairs:
            tag = str(raw_tag or "").strip().lower().replace(" ", "_")
            if not re.fullmatch(r"[a-z0-9_.'():!\-]{2,120}", tag):
                continue
            aliases = tuple(
                dict.fromkeys(
                    self._profile_alias_key(alias)
                    for alias in re.split(r"\s*[|｜]\s*", raw_aliases)
                    if self._profile_alias_key(alias)
                    and len(self._profile_alias_key(alias)) <= 80
                )
            )
            if aliases:
                groups.append((tag, aliases))
        return tuple(groups)

    def configured_character_anchors_for_prompt(
        self, user_prompt: str
    ) -> tuple[SemanticAnchor, ...]:
        """Build context-scoped character anchors from extensible alias records.

        A scoped character alias is authoritative only when the same request also
        names a compatible work family.  This lets localized sequel/spinoff names
        disambiguate a character whose Danbooru suffix still uses the parent work.
        Aliases are tested longest-first so a specific multi-word/localized title
        wins over its shorter substring.
        """
        default_series = (
            "艾尔登法环 | 艾尔登法环黑夜君临 | 黑夜君临 | "
            "elden ring | nightreign | elden ring nightreign=elden_ring",
        )
        default_characters = (
            "复仇者 | revenant=revenant_(elden_ring)",
        )
        series_groups = self._configured_alias_groups(
            "danbooru_series_alias_mappings", default_series
        )
        character_groups = self._configured_alias_groups(
            "danbooru_character_alias_mappings", default_characters
        )
        matched_series: list[tuple[str, str]] = []
        for canonical, aliases in series_groups:
            matched_alias = next(
                (
                    alias
                    for alias in sorted(aliases, key=len, reverse=True)
                    if self._text_mentions_alias(user_prompt, alias)
                ),
                "",
            )
            if matched_alias and canonical not in {
                item[0] for item in matched_series
            }:
                matched_series.append((canonical, matched_alias))

        anchors: list[SemanticAnchor] = [
            SemanticAnchor(
                anchor_id=f"configured_copyright_{index}",
                role="copyright",
                group="series",
                source_text=alias,
                description=f"Configured work alias: {alias}",
                candidates=(canonical,),
            )
            for index, (canonical, alias) in enumerate(matched_series, start=1)
        ]
        seen_characters: set[str] = set()
        for canonical, aliases in character_groups:
            matched_alias = next(
                (
                    alias
                    for alias in sorted(aliases, key=len, reverse=True)
                    if self._text_mentions_alias(user_prompt, alias)
                ),
                "",
            )
            if not matched_alias or canonical in seen_characters:
                continue
            scoped = re.fullmatch(r".+_\(([^)]+)\)", canonical)
            if scoped and not any(
                self._scope_compatible(scoped.group(1), series)
                for series, _alias in matched_series
            ):
                continue
            seen_characters.add(canonical)
            anchors.append(
                SemanticAnchor(
                    anchor_id=f"configured_character_{len(seen_characters)}",
                    role="target_character",
                    group="character",
                    source_text=matched_alias,
                    description=f"Configured character alias: {matched_alias}",
                    candidates=(canonical,),
                )
            )
        return tuple(anchors)

    def cached_term_mappings_for_prompt(
        self, user_prompt: str
    ) -> SemanticLookupResult | None:
        """Return user-trusted noun translations explicitly present in a request."""
        text = str(user_prompt or "").lower()
        matched = tuple(
            dict.fromkeys(
                tag
                for alias, tag in self._configured_mappings(
                    "danbooru_term_mappings"
                ).items()
                if any(
                    not self._mapping_alias_is_negated(text, start)
                    for start, _end in self._alias_match_spans(text, alias)
                )
            )
        )
        if not matched:
            return None
        return SemanticLookupResult(
            confirmed_tags=matched,
            status="profile_cache",
        )

    @staticmethod
    def _clean_tag_list(value: Any, *, limit: int = 80) -> list[str]:
        """Validate and deduplicate an editor-supplied canonical-tag list."""
        if not isinstance(value, list) or len(value) > limit:
            raise WardrobeValidationError("Tag 列表格式无效或数量过多。")
        cleaned: list[str] = []
        for item in value:
            tag = str(item or "").strip().lower().replace(" ", "_")
            if not re.fullmatch(r"[a-z0-9_.'():!\-]{2,120}", tag):
                raise WardrobeValidationError(f"无效的 canonical tag：{item}")
            if tag not in cleaned:
                cleaned.append(tag)
        return cleaned

    @staticmethod
    def _clean_alias(value: Any, *, label: str = "名称") -> str:
        """Normalize one editor label while rejecting unsafe control characters."""
        alias = re.sub(r"\s+", " ", str(value or "").strip())
        if not alias or len(alias) > 80 or any(ord(char) < 32 for char in alias):
            raise WardrobeValidationError(f"{label}为空、过长或含控制字符。")
        return alias

    def wardrobe_snapshot(self) -> dict[str, Any]:
        """Return editable outfit profiles, named sets, and noun translations."""
        profiles = self._profile_data().get("profiles", {})
        outfits: list[dict[str, Any]] = []
        learned_sets: list[dict[str, Any]] = []
        trusted_uniforms = self._trusted_uniform_alias_tags(profiles)
        for key, raw in profiles.items():
            if not isinstance(raw, dict):
                continue
            kind = str(raw.get("kind") or "character_outfit")
            aliases = [
                str(item).strip()
                for item in raw.get("aliases", [])
                if str(item).strip()
            ]
            tags = [
                str(item).strip()
                for item in raw.get("outfit_tags", [])
                if str(item).strip()
            ]
            if kind == "named_outfit":
                canonical = tags[0] if tags else ""
                if not canonical or canonical in {"school_uniform", "uniform"}:
                    continue
                variant = self._named_profile_variant(str(key), raw)
                cleaned_aliases = self._safe_learned_outfit_aliases(
                    str(key), raw, canonical, variant, trusted_uniforms
                )
                if not cleaned_aliases:
                    continue
                preferred = self._preferred_named_outfit_alias(
                    str(key), cleaned_aliases, canonical, variant
                )
                learned_sets.append(
                    {
                        "alias": preferred,
                        "aliases": self._named_outfit_aliases(
                            preferred,
                            canonical,
                            cleaned_aliases,
                            variant=variant,
                        ),
                        "tag": canonical,
                        "variant": variant,
                        "origin": "learned",
                        "profileKey": str(key),
                    }
                )
                continue
            if not raw.get("source_tags") and not tags:
                continue
            evidence = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {}
            outfits.append(
                {
                    "key": str(key),
                    "aliases": aliases or [str(key)],
                    "sourceTags": [
                        str(item).strip()
                        for item in raw.get("source_tags", [])
                        if str(item).strip()
                    ],
                    "tags": tags,
                    "appearanceTags": [
                        str(item).strip()
                        for item in evidence.get("appearance_tags", [])
                        if str(item).strip()
                    ],
                    "qualifier": self._normalize_outfit_variant(
                        raw.get("qualifier"), str(key), aliases
                    ),
                    "evidence": {
                        "sampleMode": str(evidence.get("sample_mode") or ""),
                        "sampleCount": int(evidence.get("focused_posts") or 0),
                        # Older caches did not have a creation timestamp. Their
                        # earliest available evidence time is a stable migration
                        # baseline; newly saved/learned profiles keep a distinct
                        # created_at even when their evidence is later refreshed.
                        "createdAt": float(
                            evidence.get("created_at")
                            or evidence.get("updated_at")
                            or 0
                        ),
                        "updatedAt": float(evidence.get("updated_at") or 0),
                    },
                }
            )
        configured_sets: list[dict[str, Any]] = []
        for tag, variant, configured_aliases in self._configured_named_outfit_groups():
            aliases = self._named_outfit_aliases(
                configured_aliases[0],
                tag,
                configured_aliases,
                variant=variant,
            )
            configured_sets.append(
                {
                    "alias": self._preferred_named_outfit_alias(
                        configured_aliases[0], aliases, tag, variant
                    ),
                    "aliases": aliases,
                    "tag": tag,
                    "variant": variant,
                    "origin": "configured",
                    "profileKey": "",
                }
            )
        terms = [
            {"alias": alias, "tag": tag}
            for alias, tag in self._configured_mappings("danbooru_term_mappings").items()
        ]
        payload = {
            "outfits": sorted(outfits, key=lambda item: item["key"]),
            "outfitSets": sorted(
                configured_sets + learned_sets,
                key=lambda item: (item["alias"], item["origin"]),
            ),
            "terms": sorted(terms, key=lambda item: item["alias"]),
        }
        revision = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {**payload, "revision": revision}

    def save_wardrobe(self, payload: Any) -> dict[str, Any]:
        """Validate and atomically replace editable wardrobe collections."""
        if not isinstance(payload, dict):
            raise WardrobeValidationError("请求正文必须是 JSON 对象。")
        current = self.wardrobe_snapshot()
        if payload.get("baseRevision") != current["revision"]:
            raise WardrobeValidationError("服装词库已发生变化，请刷新后重试。")
        raw_outfits = payload.get("outfits")
        raw_sets = payload.get("outfitSets")
        raw_terms = payload.get("terms")
        if not all(isinstance(value, list) for value in (raw_outfits, raw_sets, raw_terms)):
            raise WardrobeValidationError("服装词库列表格式无效。")
        if len(raw_outfits) > 300 or len(raw_sets) > 500 or len(raw_terms) > 500:
            raise WardrobeValidationError("服装词库条目数量过多。")

        old_profiles = self._profile_data().get("profiles", {})
        new_profiles: dict[str, Any] = {}
        seen_keys: set[str] = set()
        for item in raw_outfits:
            if not isinstance(item, dict):
                raise WardrobeValidationError("服装档案条目格式无效。")
            key = self._profile_alias_key(self._clean_alias(item.get("key"), label="档案名称"))
            aliases = [self._clean_alias(value, label="档案别名") for value in item.get("aliases", [])]
            source_tags = self._clean_tag_list(item.get("sourceTags", []), limit=20)
            tags = self._clean_tag_list(item.get("tags", []), limit=80)
            qualifier = self._normalize_outfit_variant(
                item.get("qualifier"), key, aliases
            )
            base_key = key.split("::", 1)[0]
            storage_key = self._outfit_profile_key(base_key, qualifier)
            if storage_key in seen_keys:
                raise WardrobeValidationError(f"重复的服装档案：{storage_key}")
            seen_keys.add(storage_key)
            old_candidate = old_profiles.get(key, old_profiles.get(storage_key))
            old = old_candidate if isinstance(old_candidate, dict) else {}
            old_evidence = (
                old.get("evidence")
                if isinstance(old.get("evidence"), dict)
                else {}
            )
            old_appearance_tags = self._clean_tag_list(
                old_evidence.get("appearance_tags", []), limit=80
            )
            if "appearanceTags" in item:
                appearance_tags = self._clean_tag_list(
                    item.get("appearanceTags"), limit=80
                )
                appearance_changed = appearance_tags != old_appearance_tags
            else:
                # Keep older API clients backward compatible: omitting the field
                # means preserve it, while an explicit empty list means clear it.
                appearance_tags = old_appearance_tags
                appearance_changed = False
            record: dict[str, Any] = {
                "kind": "character_outfit",
                "qualifier": qualifier,
                "aliases": list(dict.fromkeys(aliases or [key])),
                "source_tags": source_tags,
                "copyright_tags": list(old.get("copyright_tags", [])),
                "outfit_tags": tags,
            }
            # Every editor-saved profile gets a creation timestamp, including
            # a completely manual row with no learned evidence or appearance
            # tags. It remains stable across later edits and evidence refreshes.
            record["evidence"] = {
                **old_evidence,
                "appearance_tags": appearance_tags,
                "created_at": float(
                    old_evidence.get("created_at")
                    or old_evidence.get("updated_at")
                    or time.time()
                ),
            }
            if appearance_changed or old_evidence.get(
                "appearance_manual_override"
            ):
                record["evidence"]["appearance_manual_override"] = True
            new_profiles[storage_key] = record

        configured_sets: list[str] = []
        learned_by_key: dict[str, dict[str, Any]] = {}
        seen_set_aliases: dict[str, str] = {}
        for item in raw_sets:
            if not isinstance(item, dict):
                raise WardrobeValidationError("服装套组条目格式无效。")
            raw_aliases = item.get("aliases")
            aliases = (
                [
                    self._clean_alias(value, label="套组别名")
                    for value in raw_aliases
                ]
                if isinstance(raw_aliases, list) and raw_aliases
                else [self._clean_alias(item.get("alias"), label="套组名称")]
            )
            tag = self._clean_tag_list([item.get("tag")], limit=1)[0]
            variant = self._normalize_outfit_variant(
                item.get("variant"), aliases
            )
            aliases = self._named_outfit_aliases(
                aliases[0], tag, aliases, variant=variant
            )
            if not aliases:
                raise WardrobeValidationError(
                    "服装套组没有与 canonical tag/变体相符的有效别名。"
                )
            for alias in aliases:
                alias_key = alias.lower()
                previous_tag = seen_set_aliases.get(alias_key)
                if previous_tag is not None and previous_tag != tag:
                    raise WardrobeValidationError(f"重复的服装套组名称：{alias}")
                seen_set_aliases[alias_key] = tag
            if item.get("origin") == "learned" and item.get("profileKey"):
                profile_key = self._profile_alias_key(
                    self._clean_alias(item.get("profileKey"), label="套组条目 ID")
                )
                base_key = profile_key
                suffix = 2
                while profile_key in new_profiles or profile_key in learned_by_key:
                    profile_key = f"{base_key}::named{suffix}"
                    suffix += 1
                learned_by_key[profile_key] = {
                    "kind": "named_outfit",
                    "variant": variant,
                    "aliases": list(aliases),
                    "manual_aliases": list(aliases),
                    "source_tags": [],
                    "copyright_tags": [],
                    "outfit_tags": [tag],
                }
            else:
                configured_sets.append(f"{' | '.join(aliases)}={tag}")
        for profile_key, learned in learned_by_key.items():
            tag = learned["outfit_tags"][0]
            learned["aliases"] = self._named_outfit_aliases(
                learned["aliases"][0],
                tag,
                learned["aliases"],
                variant=learned["variant"],
            )
            learned["manual_aliases"] = list(
                dict.fromkeys(learned["manual_aliases"])
            )
            new_profiles[profile_key] = learned

        term_entries: list[str] = []
        seen_terms: set[str] = set()
        for item in raw_terms:
            if not isinstance(item, dict):
                raise WardrobeValidationError("名词翻译条目格式无效。")
            alias = self._clean_alias(item.get("alias"), label="名词")
            tag = self._clean_tag_list([item.get("tag")], limit=1)[0]
            if alias.lower() in seen_terms:
                raise WardrobeValidationError(f"重复的名词：{alias}")
            seen_terms.add(alias.lower())
            term_entries.append(f"{alias}={tag}")

        self._config["danbooru_named_outfit_mappings"] = configured_sets
        self._config["danbooru_term_mappings"] = term_entries
        self._profile_cache_data = {"version": 3, "profiles": new_profiles}
        self._save_profile_data()
        return self.wardrobe_snapshot()

    def remember_outfit_summary(
        self,
        source_text: str,
        source_tags: tuple[str, ...],
        outfit_tags: tuple[str, ...],
        evidence: dict[str, Any] | None = None,
        qualifier: str = "default",
    ) -> None:
        """Persist validated donor identity and reusable clothing as separate fields.

        ``source_tags`` identify the outfit donor and are not appearance tags for
        the visible target. ``outfit_tags`` are the transferable garment evidence;
        optional post statistics live under ``evidence`` and determine refresh age.
        """
        qualifier_key = self._normalize_outfit_variant(qualifier, source_text)
        key = self._outfit_profile_key(source_text, qualifier_key)
        if not key or not source_tags:
            return
        copyright_tags: list[str] = []
        for source_tag in source_tags:
            scoped = re.fullmatch(r".+_\(([^)]+)\)", source_tag)
            if scoped and scoped.group(1) not in copyright_tags:
                copyright_tags.append(scoped.group(1))
        profiles = self._profile_data().setdefault("profiles", {})
        source_tag_keys = {
            self._profile_alias_key(tag) for tag in source_tags if str(tag).strip()
        }
        source_match_key = next(
            (
                str(profile_key)
                for profile_key, candidate in profiles.items()
                if isinstance(candidate, dict)
                and candidate.get("kind") != "named_outfit"
                and self._normalize_outfit_variant(
                    candidate.get("qualifier"), profile_key, candidate.get("aliases", [])
                )
                == qualifier_key
                and source_tag_keys.intersection(
                    self._profile_alias_key(tag)
                    for tag in candidate.get("source_tags", [])
                    if str(tag).strip()
                )
            ),
            "",
        )
        exact_alias_matches = [
            str(profile_key)
            for profile_key, candidate in profiles.items()
            if isinstance(candidate, dict)
            and candidate.get("kind") != "named_outfit"
            and self._normalize_outfit_variant(
                candidate.get("qualifier"), profile_key, candidate.get("aliases", [])
            )
            == qualifier_key
            and key
            in {
                self._profile_alias_key(str(profile_key).split("::", 1)[0]),
                *(
                    self._profile_alias_key(alias)
                    for alias in candidate.get("aliases", [])
                ),
            }
        ]
        if not source_match_key and exact_alias_matches:
            self.logger.warning(
                "[comfyui_agent] refusing duplicate outfit profile alias=%s incoming_sources=%s "
                "existing_profiles=%s",
                source_text,
                sorted(source_tag_keys),
                exact_alias_matches,
            )
            return
        existing_key = source_match_key or key
        existing_value = profiles.get(existing_key)
        existing = existing_value if isinstance(existing_value, dict) else {}
        existing_evidence = (
            existing.get("evidence")
            if isinstance(existing.get("evidence"), dict)
            else {}
        )
        aliases = list(
            dict.fromkeys(
                str(alias).strip()
                for alias in (
                    *existing.get("aliases", []),
                    source_text,
                    *source_tags,
                    *(
                        (f"{source_text}的演出服", f"{source_text} stage outfit")
                        if qualifier_key == "stage"
                        else ()
                    ),
                )
                if str(alias).strip()
            )
        )
        stored_outfit_tags = tuple(
            str(tag).strip()
            for tag in existing.get("outfit_tags", [])
            if str(tag).strip()
        )
        stored_appearance_tags = tuple(
            str(tag).strip()
            for tag in existing_evidence.get("appearance_tags", [])
            if str(tag).strip()
        )
        appearance_manual_override = bool(
            existing_evidence.get("appearance_manual_override")
        )
        record: dict[str, Any] = {
            "kind": "character_outfit",
            "qualifier": qualifier_key,
            "aliases": aliases,
            "source_tags": list(
                dict.fromkeys((*existing.get("source_tags", []), *source_tags))
            ),
            "copyright_tags": list(
                dict.fromkeys((*existing.get("copyright_tags", []), *copyright_tags))
            ),
            # Once a profile exists, its editor-visible wardrobe is authoritative.
            # Refresh only supplies initial tags for a brand-new profile.
            "outfit_tags": list(
                stored_outfit_tags or tuple(dict.fromkeys(outfit_tags))
            ),
        }
        if evidence:
            record["evidence"] = {
                **evidence,
                "appearance_tags": list(
                    stored_appearance_tags
                    if appearance_manual_override
                    else stored_appearance_tags
                    or tuple(
                        dict.fromkeys(
                            str(tag).strip()
                            for tag in evidence.get("appearance_tags", [])
                            if str(tag).strip()
                        )
                    )
                ),
                **(
                    {"appearance_manual_override": True}
                    if appearance_manual_override
                    else {}
                ),
                "algorithm_version": 4,
                "created_at": float(
                    existing_evidence.get("created_at")
                    or existing_evidence.get("updated_at")
                    or time.time()
                ),
                "updated_at": time.time(),
            }
        elif existing_evidence:
            record["evidence"] = dict(existing_evidence)
        profiles[existing_key] = record
        if key != existing_key:
            profiles.pop(key, None)
        self._save_profile_data()

    def remember_named_outfit(
        self, alias: str, canonical_tag: str, variant: str | None = None
    ) -> None:
        """Persist one screened tag representing a complete named outfit set.

        Generic ``school_uniform``/``uniform`` tags are rejected because caching
        them under a proper-name alias would make unrelated schools resolve to the
        same outfit. Existing aliases are merged only inside the same variant.
        """
        key = self._profile_alias_key(alias)
        tag = str(canonical_tag or "").strip().lower()
        variant_key = self._normalize_outfit_variant(variant, alias)
        if (
            not key
            or not re.fullmatch(r"[a-z0-9_.'():!\-]{2,120}", tag)
            or tag in {"school_uniform", "uniform"}
        ):
            return
        profiles = self._profile_data().setdefault("profiles", {})
        configured_tag = self._configured_named_outfits().get(key)
        if configured_tag == tag:
            return
        matching_keys = [
            str(profile_key)
            for profile_key, profile in profiles.items()
            if isinstance(profile, dict)
            and profile.get("kind") == "named_outfit"
            and self._named_profile_variant(str(profile_key), profile) == variant_key
            and key
            in {
                self._profile_alias_key(str(profile_key).split("::", 1)[0]),
                *(
                    self._profile_alias_key(item)
                    for item in profile.get("aliases", [])
                ),
                *(
                    self._profile_alias_key(item)
                    for item in profile.get("manual_aliases", [])
                ),
            }
        ]
        target_key = matching_keys[0] if matching_keys else key
        existing_aliases: list[str] = []
        manual_aliases: list[str] = []
        removed_duplicate = False
        for profile_key in matching_keys:
            profile = profiles.get(profile_key, {})
            existing_aliases.extend(profile.get("aliases", []))
            manual_aliases.extend(profile.get("manual_aliases", []))
            if profile_key != target_key:
                profiles.pop(profile_key, None)
                removed_duplicate = True
        new_profile = {
            "kind": "named_outfit",
            "variant": variant_key,
            "aliases": self._named_outfit_aliases(
                key, tag, existing_aliases, manual_aliases, variant=variant_key
            ),
            "source_tags": [],
            "copyright_tags": [],
            "outfit_tags": [tag],
        }
        if manual_aliases:
            new_profile["manual_aliases"] = list(dict.fromkeys(manual_aliases))
        if not new_profile["aliases"]:
            return
        if profiles.get(target_key) == new_profile and not removed_duplicate:
            return
        profiles[target_key] = new_profile
        self._save_profile_data()

    def required_core_tags_for_prompt(self, user_prompt: str) -> tuple[str, ...]:
        """Return locally known character anchors explicitly requested by the user.

        Args:
            user_prompt: User prompt text.

        Returns:
            Required core tags.
        """
        return required_core_tags_for_prompt(user_prompt)

    def required_profile_tags_for_prompt(self, user_prompt: str) -> tuple[str, ...]:
        """Return deterministic tags for an explicitly requested variant."""
        return required_profile_tags_for_prompt(user_prompt)

    def profile_hints_for_prompt(self, user_prompt: str) -> dict[str, str]:
        """Return locally verified variant context for the prompt LLM."""
        return profile_hints_for_prompt(user_prompt)

    def _base_urls(self) -> tuple[str, ...]:
        base_urls_text = self._str("danbooru_tag_base_urls", "").strip()
        if not base_urls_text:
            return DEFAULT_DONMAI_BASE_URLS
        return tuple(
            item.strip()
            for item in re.split(r"[,;\n]+", base_urls_text)
            if item.strip()
        )

    def semantic_lookup_available(self) -> bool:
        """Return whether generalized local semantic validation can run."""
        if not self._bool("danbooru_semantic_lookup_enabled", True):
            return False
        return self._local_cli_path() is not None

    def _local_cli_path(self) -> Path | None:
        return resolve_local_cli_path(self._str("danbooru_local_cli_path", ""))

    async def resolve_semantic_anchors(
        self, anchors: tuple[SemanticAnchor, ...]
    ) -> SemanticLookupResult:
        """Run one local batch lookup and expand verified outfit sources."""
        cli_path = self._local_cli_path()
        if cli_path is None:
            return SemanticLookupResult(anchors=anchors, status="not_available")
        if not anchors:
            return SemanticLookupResult(anchors=anchors, status="empty_plan")
        timeout = max(
            1.0,
            min(self._float("danbooru_tag_lookup_timeout", 6.0), 20.0),
        )
        result = await asyncio.to_thread(
            lookup_semantic_anchors,
            anchors,
            cli_path=cli_path,
            timeout=timeout,
        )
        anchor_tag_map = dict(result.anchor_tags)
        for anchor in anchors:
            if anchor.role != "target_character" or anchor.anchor_id in anchor_tag_map:
                continue
            matched_tag = next(
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
            if matched_tag:
                anchor_tag_map[anchor.anchor_id] = matched_tag
        target_anchor_tags = tuple(
            (anchor.anchor_id, anchor_tag_map.get(anchor.anchor_id, ""))
            for anchor in anchors
            if anchor.role == "target_character" and anchor_tag_map.get(anchor.anchor_id)
        )
        learned_outfit_tags: list[str] = []
        learned_appearance_tags: list[str] = []
        character_profiles: list[
            tuple[str, str, tuple[str, ...], tuple[str, ...]]
        ] = []
        for target_anchor_id, target_tag in target_anchor_tags:
            target_anchor = next(
                anchor
                for anchor in anchors
                if anchor.anchor_id == target_anchor_id
            )
            target_variant = self._outfit_variant(
                f"{target_anchor.source_text} {target_anchor.description}"
            )
            stored_record = self._stored_profile_for_source_tag(
                target_tag, target_variant
            )
            stored_outfit, stored_appearance = (
                self._stored_profile_components(stored_record)
                if stored_record is not None
                else ((), ())
            )
            profile = None
            if stored_record is None or self._stored_profile_refresh_needed(stored_record):
                profile = await asyncio.to_thread(
                    fetch_variant_outfit_profile,
                    target_tag,
                    outfit_kind=target_variant,
                    timeout=min(2.5, timeout),
                    user_agent=(
                        self._str("danbooru_tag_user_agent", DEFAULT_USER_AGENT).strip()
                        or DEFAULT_USER_AGENT
                    ),
                    cache=self._cache,
                    donmai_base_urls=self._base_urls(),
                )
            effective_outfit = stored_outfit if stored_record is not None else profile.tags
            effective_appearance = (
                stored_appearance if stored_record is not None else profile.appearance_tags
            )
            learned_outfit_tags.extend(effective_outfit)
            learned_appearance_tags.extend(effective_appearance)
            character_profiles.append(
                (target_anchor_id, target_tag, effective_outfit, effective_appearance)
            )
            if profile is not None and (profile.tags or profile.appearance_tags):
                self.remember_outfit_summary(
                    target_anchor.source_text or target_tag,
                    (target_tag,),
                    profile.tags,
                    {
                        "sample_mode": profile.sample_mode,
                        "total_posts": profile.total_posts,
                        "selected_posts": profile.selected_posts,
                        "focused_posts": profile.focused_posts,
                        "anchor_tag": profile.anchor_tag,
                        "tag_counts": dict(profile.tag_counts),
                        "appearance_tags": list(profile.appearance_tags),
                    },
                    target_variant,
                )
        if character_profiles:
            result = replace(
                result,
                outfit_profile_tags=tuple(dict.fromkeys(learned_outfit_tags)),
                appearance_profile_tags=tuple(
                    dict.fromkeys(learned_appearance_tags)
                ),
                character_profiles=tuple(character_profiles),
            )
        anchor_outfit_profiles = list(result.anchor_outfit_profiles)
        if result.status == "resolved" and result.named_outfit_tags:
            for anchor in anchors:
                if anchor.role not in {"outfit", "clothing"}:
                    continue
                candidate_keys = {
                    re.sub(r"\s+", "_", candidate.strip().lower())
                    for candidate in anchor.candidates
                }
                tag = next(
                    (
                        item
                        for item in result.named_outfit_tags
                        if item.lower() in candidate_keys
                    ),
                    "",
                )
                if not tag:
                    continue
                variant = self._outfit_variant(
                    f"{anchor.source_text} {anchor.description}"
                )
                self.remember_named_outfit(anchor.source_text, tag, variant)
                profile = await asyncio.to_thread(
                    fetch_variant_outfit_profile,
                    tag,
                    outfit_kind=variant,
                    timeout=min(2.5, timeout),
                    user_agent=(
                        self._str("danbooru_tag_user_agent", DEFAULT_USER_AGENT).strip()
                        or DEFAULT_USER_AGENT
                    ),
                    cache=self._cache,
                    donmai_base_urls=self._base_urls(),
                )
                if not profile.tags:
                    anchor_outfit_profiles.append((anchor.anchor_id, tag, (), variant))
                    continue
                anchor_outfit_profiles.append(
                    (anchor.anchor_id, tag, profile.tags, variant)
                )
                self.remember_outfit_summary(
                    anchor.source_text,
                    (tag,),
                    profile.tags,
                    {
                        "sample_mode": profile.sample_mode,
                        "total_posts": profile.total_posts,
                        "selected_posts": profile.selected_posts,
                        "focused_posts": profile.focused_posts,
                        "anchor_tag": profile.anchor_tag,
                        "tag_counts": dict(profile.tag_counts),
                    },
                    variant,
                )
        if anchor_outfit_profiles:
            result = replace(
                result,
                anchor_outfit_profiles=tuple(dict.fromkeys(anchor_outfit_profiles)),
            )
        if result.status != "resolved" or not result.outfit_source_tags:
            return result
        user_agent = (
            self._str("danbooru_tag_user_agent", DEFAULT_USER_AGENT).strip()
            or DEFAULT_USER_AGENT
        )
        profiles: list[str] = []
        scoped_profiles: list[tuple[str, str, tuple[str, ...], str]] = []
        source_anchors = [
            anchor for anchor in anchors if anchor.role == "outfit_source"
        ]
        used_anchor_ids: set[str] = set()
        for source_tag in result.outfit_source_tags:
            source_key = source_tag.lower()
            source_anchor = next(
                (
                    anchor
                    for anchor in source_anchors
                    if anchor.anchor_id not in used_anchor_ids
                    and any(
                        candidate.lower() == source_key
                        for candidate in anchor.candidates
                    )
                ),
                None,
            )
            if source_anchor is None:
                source_anchor = next(
                    (
                        anchor
                        for anchor in source_anchors
                        if anchor.anchor_id not in used_anchor_ids
                    ),
                    None,
                )
            if source_anchor is not None:
                used_anchor_ids.add(source_anchor.anchor_id)
            source_alias = (
                source_anchor.source_text
                if source_anchor is not None
                else source_tag.split("_(", 1)[0].replace("_", " ")
            )
            qualifier_text = (
                f"{source_anchor.source_text} {source_anchor.description}"
                if source_anchor is not None
                else ""
            )
            # Seasonal wording belongs to the outfit-source identity too.  A
            # winter lookup must not replace the character/source's default or
            # summer profile merely because the canonical source tag is equal.
            qualifier = self._outfit_variant(qualifier_text)
            stored_record = self._stored_profile_for_source_tag(source_tag, qualifier)
            stored_outfit, _stored_appearance = (
                self._stored_profile_components(stored_record)
                if stored_record is not None
                else ((), ())
            )
            profile = None
            if stored_record is None or self._stored_profile_refresh_needed(stored_record):
                profile = await asyncio.to_thread(
                    fetch_variant_outfit_profile,
                    source_tag,
                    outfit_kind=qualifier,
                    timeout=min(2.5, timeout),
                    user_agent=user_agent,
                    cache=self._cache,
                    donmai_base_urls=self._base_urls(),
                )
            effective_outfit = stored_outfit if stored_record is not None else profile.tags
            for tag in effective_outfit:
                if tag not in profiles:
                    profiles.append(tag)
            scoped_profiles.append(
                (source_alias, source_tag, effective_outfit, qualifier)
            )
            if source_anchor is not None:
                anchor_outfit_profiles.append(
                    (source_anchor.anchor_id, source_tag, effective_outfit, qualifier)
                )
            if profile is not None:
                profile_evidence = {
                    "sample_mode": profile.sample_mode,
                    "total_posts": profile.total_posts,
                    "selected_posts": profile.selected_posts,
                    "focused_posts": profile.focused_posts,
                    "anchor_tag": profile.anchor_tag,
                    "tag_counts": dict(profile.tag_counts),
                }
                self.remember_outfit_summary(
                    source_alias,
                    (source_tag,),
                    profile.tags,
                    profile_evidence,
                    qualifier,
                )
        resolved = replace(
            result,
            outfit_profile_tags=tuple(profiles[:48]),
            source_outfit_profiles=tuple(scoped_profiles),
            anchor_outfit_profiles=tuple(dict.fromkeys(anchor_outfit_profiles)),
        )
        return resolved

    async def resolve(
        self,
        *,
        llm_content: str,
        user_prompt: str,
        fixed_character: bool,
        candidate_hints: tuple[str, ...] = (),
    ) -> str:
        """Resolve likely character core tags in LLM-generated content.

        Args:
            llm_content: Tags returned by the LLM.
            user_prompt: Original user prompt.
            fixed_character: Whether a fixed character is already selected.
            candidate_hints: Optional evidence candidates proposed from the
                original user request.

        Returns:
            Tags with resolved or inserted core tags when lookup succeeds.
        """
        outcome = await self.resolve_detailed(
            llm_content=llm_content,
            user_prompt=user_prompt,
            fixed_character=fixed_character,
            candidate_hints=candidate_hints,
        )
        return outcome.text

    async def resolve_detailed(
        self,
        *,
        llm_content: str,
        user_prompt: str,
        fixed_character: bool,
        candidate_hints: tuple[str, ...] = (),
    ) -> DanbooruResolveOutcome:
        """Resolve character tags and return request-scoped diagnostics.

        Args:
            llm_content: Tags returned by the prompt LLM.
            user_prompt: Original user request.
            fixed_character: Whether local fixed-character tags own identity.
            candidate_hints: Evidence candidates proposed by the LLM.

        Returns:
            Resolved text, canonical identity anchors, and resolution status.
        """
        if (
            not llm_content
            or not self._bool("danbooru_core_tag_lookup_enabled", False)
        ):
            return DanbooruResolveOutcome(text=llm_content)
        requested = character_resolution_requested(
            llm_content,
            user_prompt=user_prompt if not fixed_character else "",
            candidate_hints=candidate_hints,
        )
        explicit_request = bool(
            not fixed_character
            and character_resolution_requested("", user_prompt=user_prompt)
        )
        timeout = max(
            1.0,
            min(self._float("danbooru_tag_lookup_timeout", 6.0), 20.0),
        )
        max_candidates = max(1, min(self._int("danbooru_tag_max_candidates", 6), 16))
        user_agent = (
            self._str("danbooru_tag_user_agent", DEFAULT_USER_AGENT).strip()
            or DEFAULT_USER_AGENT
        )
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    resolve_core_tags,
                    llm_content,
                    user_prompt=user_prompt,
                    allow_insert=not fixed_character,
                    candidate_hints=tuple(candidate_hints),
                    max_candidates=max_candidates,
                    timeout=min(2.0, timeout),
                    donmai_base_urls=self._base_urls(),
                    user_agent=user_agent,
                    cache=self._cache,
                ),
                timeout=timeout,
            )
        except TimeoutError:
            self.logger.warning(
                "[comfyui_agent] danbooru core tag lookup exceeded total %.1fs budget",
                timeout,
            )
            return DanbooruResolveOutcome(
                text=llm_content,
                status="source_unavailable" if requested else "not_requested",
                candidate_hints=tuple(candidate_hints),
                explicit_request=explicit_request,
            )
        except Exception as exc:
            self.logger.warning(
                "[comfyui_agent] danbooru core tag lookup failed: %s", exc
            )
            return DanbooruResolveOutcome(
                text=llm_content,
                status="source_unavailable" if requested else "not_requested",
                candidate_hints=tuple(candidate_hints),
                explicit_request=explicit_request,
            )
        for old, new, count, source in result.replacements:
            self.logger.info(
                "[comfyui_agent] danbooru core tag resolved: %s -> %s post_count=%s source=%s",
                old,
                new,
                count,
                source,
            )
        for new, count, source in result.inserted:
            self.logger.info(
                "[comfyui_agent] danbooru core tag inserted: %s post_count=%s source=%s",
                new,
                count,
                source,
            )
        for name, count, source in result.verified:
            self.logger.info(
                "[comfyui_agent] danbooru character tag verified: %s post_count=%s source=%s",
                name,
                count,
                source,
            )
        if result.status == "unresolved":
            self.logger.warning(
                "[comfyui_agent] named character tag unresolved candidates=%s",
                ",".join(result.candidate_hints) or "none",
            )
        return DanbooruResolveOutcome(
            text=result.text,
            status=result.status,
            canonical_tag=result.canonical_tag,
            identity_tags=result.identity_tags,
            candidate_hints=result.candidate_hints,
            evidence=result.evidence,
            explicit_request=result.explicit_request,
        )
