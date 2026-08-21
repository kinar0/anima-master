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

    @staticmethod
    def _outfit_variant(value: Any) -> str:
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
        return "default"

    @classmethod
    def _normalize_outfit_variant(cls, value: Any, *hints: Any) -> str:
        explicit = str(value or "").strip().lower()
        if explicit in {"default", "summer", "winter", "stage"}:
            return explicit
        return cls._outfit_variant(" ".join(str(hint or "") for hint in hints))

    @classmethod
    def _outfit_profile_key(cls, source_text: str, variant: str) -> str:
        source = cls._profile_alias_key(source_text)
        return source if variant == "default" else f"{source}::{variant}"

    @classmethod
    def _alias_fits_named_outfit(
        cls, alias: str, canonical_tag: str, variant: str
    ) -> bool:
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
        if variant in {"summer", "winter", "stage"}:
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
        """Build stable trigger aliases without treating them as separate sets."""
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
                and cls._alias_fits_named_outfit(alias, canonical_tag, variant)
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
        candidates = cls._named_outfit_aliases(
            profile_key, canonical_tag, aliases, variant=variant
        )
        human = [item for item in candidates if item not in generated]
        if not human:
            return cls._profile_alias_key(profile_key) or canonical_tag
        # Preserve the editor's explicit alias order.  Re-sorting by shortest
        # label made the visible primary alias (and therefore the persisted
        # profile key) change again immediately after a successful save.
        return human[0]

    def _profile_data(self) -> dict[str, Any]:
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
        key = self._outfit_profile_key(source_text, variant)
        profile = self._profile_data().get("profiles", {}).get(key)
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

    def outfit_source_refresh_needed(self, source_text: str) -> bool:
        """Refresh only absent, legacy, or week-old character outfit evidence."""
        variant = self._outfit_variant(source_text)
        key = self._outfit_profile_key(source_text, variant)
        profile = self._profile_data().get("profiles", {}).get(key)
        if not isinstance(profile, dict):
            return True
        evidence = profile.get("evidence")
        if not isinstance(evidence, dict) or evidence.get("algorithm_version") != 4:
            return True
        try:
            updated_at = float(evidence.get("updated_at") or 0.0)
        except (TypeError, ValueError):
            return True
        return time.time() - updated_at >= 7 * 86400

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
        safe: list[str] = []
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
        """Return every persisted independent outfit-set explicitly named in a request."""
        text = str(user_prompt or "").lower()
        requested_variant = self._outfit_variant(text)
        matched: list[str] = list(
            dict.fromkeys(
                tag
                for alias, tag in self._configured_named_outfits().items()
                if alias.lower() in text or tag.replace("_", " ") in text
            )
        )
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
            if variant != requested_variant:
                continue
            aliases = self._safe_learned_outfit_aliases(
                str(alias), profile, canonical, variant, trusted_uniforms
            )
            derived_matches = (
                canonical in text or canonical.replace("_", " ") in text
            )
            if not derived_matches and not any(value in text for value in aliases):
                continue
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
        return SemanticLookupResult(
            confirmed_tags=tuple(matched),
            named_outfit_tags=tuple(matched),
            outfit_profile_tags=tuple(profile_tags_for_request),
            status="profile_cache",
        )

    def _configured_named_outfits(self) -> dict[str, str]:
        """Parse user-defined alias=canonical_tag mappings from plugin config."""
        return self._configured_mappings("danbooru_named_outfit_mappings")

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
                if alias in text
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
        alias = re.sub(r"\s+", " ", str(value or "").strip())
        if not alias or len(alias) > 80 or any(ord(char) < 32 for char in alias):
            raise WardrobeValidationError(f"{label}为空、过长或含控制字符。")
        return alias

    def wardrobe_snapshot(self) -> dict[str, Any]:
        """Return editable outfit profiles, named sets, and noun translations."""
        profiles = self._profile_data().get("profiles", {})
        outfits: list[dict[str, Any]] = []
        learned_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
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
                learned = learned_by_identity.setdefault(
                    (canonical, variant),
                    {"profileKey": str(key), "aliases": []},
                )
                learned["aliases"].extend(cleaned_aliases)
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
                    "qualifier": self._normalize_outfit_variant(
                        raw.get("qualifier"), str(key), aliases
                    ),
                    "evidence": {
                        "sampleMode": str(evidence.get("sample_mode") or ""),
                        "sampleCount": int(evidence.get("focused_posts") or 0),
                        "updatedAt": float(evidence.get("updated_at") or 0),
                    },
                }
            )
        configured_by_identity: dict[tuple[str, str], list[str]] = {}
        for alias, tag in self._configured_named_outfits().items():
            variant = self._outfit_variant(alias)
            configured_by_identity.setdefault((tag, variant), []).append(alias)
        configured_sets: list[dict[str, Any]] = []
        for (tag, variant), configured_aliases in configured_by_identity.items():
            learned_aliases = learned_by_identity.get(
                (tag, variant), {}
            ).get("aliases", [])
            aliases = self._named_outfit_aliases(
                configured_aliases[0],
                tag,
                configured_aliases,
                learned_aliases,
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
        configured_identities = set(configured_by_identity)
        learned_sets = [
            {
                "alias": preferred,
                "aliases": self._named_outfit_aliases(
                    preferred, tag, item["aliases"], variant=variant
                ),
                "tag": tag,
                "variant": variant,
                "origin": "learned",
                "profileKey": item["profileKey"],
            }
            for (tag, variant), item in learned_by_identity.items()
            for preferred in [
                self._preferred_named_outfit_alias(
                    item["profileKey"], item["aliases"], tag, variant
                )
            ]
            if (tag, variant) not in configured_identities
        ]
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
            if key in seen_keys:
                raise WardrobeValidationError(f"重复的服装档案：{key}")
            seen_keys.add(key)
            aliases = [self._clean_alias(value, label="档案别名") for value in item.get("aliases", [])]
            source_tags = self._clean_tag_list(item.get("sourceTags", []), limit=20)
            tags = self._clean_tag_list(item.get("tags", []), limit=80)
            qualifier = self._normalize_outfit_variant(
                item.get("qualifier"), key, aliases
            )
            old = old_profiles.get(key) if isinstance(old_profiles.get(key), dict) else {}
            record: dict[str, Any] = {
                "kind": "character_outfit",
                "qualifier": qualifier,
                "aliases": list(dict.fromkeys(aliases or [key])),
                "source_tags": source_tags,
                "copyright_tags": list(old.get("copyright_tags", [])),
                "outfit_tags": tags,
            }
            if isinstance(old.get("evidence"), dict):
                record["evidence"] = old["evidence"]
            new_profiles[key] = record

        configured_sets: list[str] = []
        learned_by_key: dict[str, dict[str, Any]] = {}
        seen_set_aliases: set[str] = set()
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
                if alias_key in seen_set_aliases:
                    raise WardrobeValidationError(f"重复的服装套组名称：{alias}")
                seen_set_aliases.add(alias_key)
            if item.get("origin") == "learned" and item.get("profileKey"):
                profile_key = self._profile_alias_key(aliases[0])
                if profile_key in new_profiles:
                    profile_key = f"{profile_key}::named"
                learned = learned_by_key.setdefault(
                    profile_key,
                    {
                        "kind": "named_outfit",
                        "variant": variant,
                        "aliases": [],
                        "source_tags": [],
                        "copyright_tags": [],
                        "outfit_tags": [tag],
                    },
                )
                if learned["outfit_tags"] != [tag] or learned["variant"] != variant:
                    raise WardrobeValidationError("同一学习套组的 canonical tag 不一致。")
                learned["aliases"].extend(aliases)
            else:
                configured_sets.extend(f"{alias}={tag}" for alias in aliases)
        for profile_key, learned in learned_by_key.items():
            tag = learned["outfit_tags"][0]
            learned["aliases"] = self._named_outfit_aliases(
                learned["aliases"][0],
                tag,
                learned["aliases"],
                variant=learned["variant"],
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
        """Persist a validated source and its reusable outfit summary."""
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
        profiles[key] = {
            "kind": "character_outfit",
            "qualifier": qualifier_key,
            "aliases": [
                source_text,
                *(
                    [f"{source_text}的演出服", f"{source_text} stage outfit"]
                    if qualifier_key == "stage"
                    else []
                ),
            ],
            "source_tags": list(dict.fromkeys(source_tags)),
            "copyright_tags": copyright_tags,
            "outfit_tags": list(dict.fromkeys(outfit_tags)),
        }
        if evidence:
            profiles[key]["evidence"] = {
                **evidence,
                "algorithm_version": 4,
                "updated_at": time.time(),
            }
        self._save_profile_data()

    def remember_named_outfit(
        self, alias: str, canonical_tag: str, variant: str | None = None
    ) -> None:
        """Persist one screened Danbooru tag representing a complete named outfit set."""
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
        matching_keys = [
            str(profile_key)
            for profile_key, profile in profiles.items()
            if isinstance(profile, dict)
            and profile.get("kind") == "named_outfit"
            and self._named_profile_variant(str(profile_key), profile) == variant_key
            and tag
            in {
                str(item).strip().lower()
                for item in profile.get("outfit_tags", [])
            }
        ]
        target_key = next(
            (
                profile_key
                for profile_key in matching_keys
                if self._alias_fits_named_outfit(profile_key, tag, variant_key)
            ),
            key,
        )
        existing_aliases: list[str] = []
        removed_duplicate = False
        for profile_key in matching_keys:
            profile = profiles.get(profile_key, {})
            existing_aliases.extend(profile.get("aliases", []))
            if profile_key != target_key:
                profiles.pop(profile_key, None)
                removed_duplicate = True
        new_profile = {
            "kind": "named_outfit",
            "variant": variant_key,
            "aliases": self._named_outfit_aliases(
                key, tag, existing_aliases, variant=variant_key
            ),
            "source_tags": [],
            "copyright_tags": [],
            "outfit_tags": [tag],
        }
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
                if variant not in {"summer", "winter"}:
                    continue
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
                    continue
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
        for source_tag in result.outfit_source_tags[:2]:
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
            profile = await asyncio.to_thread(
                fetch_variant_outfit_profile,
                source_tag,
                outfit_kind=qualifier,
                timeout=min(2.5, timeout),
                user_agent=user_agent,
                cache=self._cache,
                donmai_base_urls=self._base_urls(),
            )
            for tag in profile.tags:
                if tag not in profiles:
                    profiles.append(tag)
            scoped_profiles.append(
                (source_alias, source_tag, profile.tags, qualifier)
            )
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
