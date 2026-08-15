from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

try:
    from .danbooru_tags import (
        DEFAULT_DONMAI_BASE_URLS,
        DEFAULT_USER_AGENT,
        character_resolution_requested,
        fetch_variant_outfit_tags,
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
        fetch_variant_outfit_tags,
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
        self._profile_cache_path = profile_cache_path
        self._profile_cache_data: dict[str, Any] | None = None

    @staticmethod
    def _profile_alias_key(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

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
        key = self._profile_alias_key(source_text)
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

    def remember_outfit_summary(
        self,
        source_text: str,
        source_tags: tuple[str, ...],
        outfit_tags: tuple[str, ...],
    ) -> None:
        """Persist a validated source and its reusable outfit summary."""
        key = self._profile_alias_key(source_text)
        if not key or not source_tags:
            return
        copyright_tags: list[str] = []
        for source_tag in source_tags:
            scoped = re.fullmatch(r".+_\(([^)]+)\)", source_tag)
            if scoped and scoped.group(1) not in copyright_tags:
                copyright_tags.append(scoped.group(1))
        profiles = self._profile_data().setdefault("profiles", {})
        profiles[key] = {
            "source_tags": list(dict.fromkeys(source_tags)),
            "copyright_tags": copyright_tags,
            "outfit_tags": list(dict.fromkeys(outfit_tags)),
        }
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
        if cli_path is None or not anchors:
            return SemanticLookupResult(anchors=anchors, status="not_available")
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
        if result.status != "resolved" or not result.outfit_source_tags:
            return result
        user_agent = (
            self._str("danbooru_tag_user_agent", DEFAULT_USER_AGENT).strip()
            or DEFAULT_USER_AGENT
        )
        profiles: list[str] = []
        for source_tag in result.outfit_source_tags[:2]:
            tags = await asyncio.to_thread(
                fetch_variant_outfit_tags,
                source_tag,
                timeout=min(2.5, timeout),
                user_agent=user_agent,
                cache=self._cache,
                donmai_base_urls=self._base_urls(),
            )
            for tag in tags:
                if tag not in profiles:
                    profiles.append(tag)
        resolved = replace(result, outfit_profile_tags=tuple(profiles[:24]))
        source_alias = next(
            (
                anchor.source_text
                for anchor in anchors
                if anchor.role == "outfit_source" and anchor.source_text
            ),
            "",
        )
        if source_alias:
            self.remember_outfit_summary(
                source_alias,
                resolved.outfit_source_tags,
                resolved.outfit_profile_tags,
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
