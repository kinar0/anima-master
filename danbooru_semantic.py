from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SEMANTIC_PLAN_SYSTEM_PROMPT = (
    "You extract semantic lookup anchors for a local Danbooru index. "
    "Return valid JSON only. Never claim that a candidate is verified."
)


_ALLOWED_GROUPS = {
    "character",
    "series",
    "appearance",
    "expression",
    "pose",
    "action",
    "clothing",
    "outfit",
    "accessory",
    "prop",
    "scene",
    "background",
    "lighting",
}
_ALLOWED_ROLES = {
    "target_character",
    "outfit_source",
    "copyright",
    "appearance",
    "expression",
    "pose",
    "action",
    "clothing",
    "outfit",
    "accessory",
    "prop",
    "scene",
    "lighting",
}


@dataclass(frozen=True)
class SemanticAnchor:
    """One bounded semantic concept proposed by the prompt-planning LLM."""

    anchor_id: str
    role: str
    group: str
    source_text: str
    description: str
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class SemanticLookupResult:
    """Locally validated semantic anchors for one image request."""

    confirmed_tags: tuple[str, ...] = ()
    outfit_source_tags: tuple[str, ...] = ()
    outfit_profile_tags: tuple[str, ...] = ()
    appearance_profile_tags: tuple[str, ...] = ()
    named_outfit_tags: tuple[str, ...] = ()
    source_outfit_profiles: tuple[
        tuple[str, str, tuple[str, ...], str], ...
    ] = ()
    missing_descriptions: tuple[str, ...] = ()
    candidate_tags: tuple[str, ...] = ()
    anchors: tuple[SemanticAnchor, ...] = ()
    status: str = "not_available"

    def prompt_context(
        self,
        *,
        include_outfit_source_anchor: bool = True,
        effective_outfit_tags: tuple[str, ...] = (),
        removed_outfit_tags: tuple[str, ...] = (),
    ) -> str:
        """Render verified evidence for the final prompt-writing LLM."""
        if self.status == "not_available" or (
            not self.anchors
            and not self.confirmed_tags
            and not self.outfit_source_tags
            and not self.outfit_profile_tags
            and not self.appearance_profile_tags
            and not self.named_outfit_tags
            and not self.source_outfit_profiles
            and not effective_outfit_tags
        ):
            return ""
        lines = [
            "Local Danbooru validation (authoritative for hard tags):",
            "Use confirmed tags exactly. Do not transliterate missing concepts into invented tags.",
        ]
        confirmed = tuple(
            tag
            for tag in self.confirmed_tags
            if include_outfit_source_anchor or tag not in self.outfit_source_tags
        )
        if confirmed:
            lines.append("confirmed hard tags: " + ", ".join(confirmed))
        if self.outfit_source_tags:
            if include_outfit_source_anchor:
                lines.append(
                    "costume/cosplay source anchors (not extra visible people): "
                    + ", ".join(self.outfit_source_tags)
                )
                if not self.outfit_profile_tags:
                    lines.append(
                        "The source anchor must still appear in Tags. It is sufficient "
                        "on its own: do not invent or infer clothing attributes when no "
                        "post-derived outfit profile is available."
                    )
            else:
                lines.append(
                    "costume source identity (context only; do not emit as a hard tag): "
                    + ", ".join(self.outfit_source_tags)
                )
        visible_outfit_tags = effective_outfit_tags or self.outfit_profile_tags
        if visible_outfit_tags:
            lines.append(
                "authoritative visible outfit tags for this request: "
                + ", ".join(visible_outfit_tags)
            )
            lines.append(
                "Do not add clothing outside this authoritative list. Explicit user "
                "changes have already been applied to it."
            )
        if self.appearance_profile_tags:
            lines.append(
                "authoritative recurring appearance tags for this character: "
                + ", ".join(self.appearance_profile_tags)
            )
        if self.source_outfit_profiles:
            lines.append("Character-scoped outfit-source profiles (never mix them):")
            for alias, source_tag, tags, qualifier in self.source_outfit_profiles:
                label = f"{alias} [{qualifier}]" if qualifier != "default" else alias
                lines.append(
                    f"- {label} / {source_tag}: " + ", ".join(tags)
                )
        if self.named_outfit_tags:
            lines.append(
                "user-requested named outfit-set hard tags: "
                + ", ".join(self.named_outfit_tags)
            )
            lines.append(
                "Keep each named outfit set on the character explicitly wearing it."
            )
            lines.append(
                "A named outfit-set tag is complete on its own. Do not infer or add "
                "component garments such as shirt, jacket, skirt, pleated_skirt, "
                "necktie, hosiery, or shoes unless the user explicitly modifies them "
                "or an authoritative visible outfit profile lists them."
            )
        if removed_outfit_tags:
            lines.append(
                "outfit tags explicitly replaced or removed by the user; never restore: "
                + ", ".join(removed_outfit_tags)
            )
        if self.missing_descriptions:
            lines.append(
                "unresolved concepts; express these only in Nltags: "
                + "; ".join(self.missing_descriptions)
            )
        return "\n".join(lines)

    def outfit_tags_for_source(self, source_text: str) -> tuple[str, ...]:
        """Return only the profile belonging to one requested costume source."""
        key = re.sub(r"\s+", " ", str(source_text or "").strip().lower())
        for alias, source_tag, tags, _qualifier in self.source_outfit_profiles:
            alias_key = re.sub(r"\s+", " ", alias.strip().lower())
            tag_key = source_tag.split("_(", 1)[0].replace("_", " ").lower()
            if key and (key in alias_key or alias_key in key or key == tag_key):
                return tags
        return ()


def build_semantic_plan_prompt(user_prompt: str) -> str:
    """Ask the LLM for search hints without letting it certify tags."""
    return (
        "Analyze the image request into a small set of Danbooru lookup anchors. "
        "Separate visible target characters from character/persona tags used only "
        "as outfit or cosplay sources. Also include explicitly requested works, "
        "clothing, accessories, props, poses, actions, expressions, and scenes "
        "when their exact Danbooru spelling may matter.\n\n"
        "Return JSON only with this schema:\n"
        '{"anchors":[{"id":"target_1","role":"target_character",'
        '"group":"character","source_text":"exact phrase from request",'
        '"description":"short English visible meaning",'
        '"candidates":["canonical_tag_guess"]}]}\n'
        "Allowed roles: target_character, outfit_source, copyright, appearance, "
        "expression, pose, action, clothing, outfit, accessory, prop, scene, lighting. "
        "Allowed groups are the same except target_character/outfit_source use "
        "character and copyright uses series. Use at most 12 anchors and at most "
        "3 candidates per anchor. Candidates are untrusted lookup hints, not "
        "answers. Use lowercase Danbooru spelling with underscores. Do not turn "
        "an outfit_source into a visible target character. Use role outfit for a "
        "complete independently named clothing set such as a specific school uniform, "
        "ceremonial outfit, or named costume; use clothing for individual garments. "
        "Keep explicit seasonal modifiers such as summer/winter or 夏季/冬季 in "
        "source_text and description even when both variants share one canonical tag. "
        "Wearable props or accessories mentioned beside a uniform (for example cat-paw "
        "gloves, masks, bags, or animal accessories) must remain separate accessory/prop "
        "anchors and must never become aliases of the uniform. "
        "For a proper-name outfit such as XX school uniform, source_text must retain "
        "the complete proper name from the request and candidates must target that "
        "specific set; never shorten it to school_uniform or replace it with another "
        "school's uniform. "
        "Omit ordinary prose "
        "that does not need a hard tag.\n\n"
        f"User request: {user_prompt}"
    )


def parse_semantic_plan(raw: str, user_prompt: str) -> tuple[SemanticAnchor, ...]:
    """Parse and tightly bound an untrusted semantic-plan response."""
    text = re.sub(r"^```(?:json)?\s*", "", str(raw or "").strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return ()
    items = data.get("anchors") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return ()
    anchors: list[SemanticAnchor] = []
    for index, item in enumerate(items[:12]):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        group = str(item.get("group") or "").strip().lower()
        source_text = str(item.get("source_text") or "").strip()
        if role == "outfit" and source_text in {"校服", "制服"}:
            # Some planners preserve the outfit role but discard the proper-name
            # modifier. Recover the complete phrase after an explicit wear verb so
            # 月之森校服 cannot later be cached globally under the alias "校服".
            specialized = re.search(
                r"(?:穿着|身穿|身着|换上|换成)\s*[\"“']?"
                r"([^，。；;、\s\"”']{2,24}(?:校服|制服))",
                user_prompt,
                flags=re.I,
            )
            if specialized:
                source_text = specialized.group(1)
        description = re.sub(r"\s+", " ", str(item.get("description") or "").strip())
        if role not in _ALLOWED_ROLES or group not in _ALLOWED_GROUPS:
            continue
        if role in {"target_character", "outfit_source"} and group != "character":
            continue
        if role == "copyright" and group != "series":
            continue
        if not source_text or (
            source_text not in user_prompt
            and source_text.lower() not in user_prompt.lower()
        ):
            continue
        if not description or len(description) > 180:
            description = source_text
        candidates: list[str] = []
        raw_candidates = item.get("candidates")
        if not isinstance(raw_candidates, list):
            continue
        for value in raw_candidates[:3]:
            candidate = re.sub(r"\s+", "_", str(value or "").strip().lower())
            if not re.fullmatch(r"[a-z0-9_.'():!\-]{2,120}", candidate):
                continue
            if candidate not in candidates:
                candidates.append(candidate)
        if not candidates:
            continue
        anchor_id = re.sub(
            r"[^a-z0-9_-]+", "_", str(item.get("id") or f"anchor_{index}").lower()
        ).strip("_") or f"anchor_{index}"
        anchors.append(
            SemanticAnchor(
                anchor_id=anchor_id,
                role=role,
                group=group,
                source_text=source_text,
                description=description,
                candidates=tuple(candidates),
            )
        )
    return tuple(anchors)


def extract_parenthesized_character_aliases(user_prompt: str) -> tuple[SemanticAnchor, ...]:
    """Extract explicit English character aliases paired with a Chinese label.

    Args:
        user_prompt: The original image request.

    Returns:
        Character lookup anchors that do not depend on an LLM recognizing the
        parenthesized alias convention.
    """
    prompt = str(user_prompt or "")
    matches: list[tuple[str, str]] = []
    chinese_label = r"[\u4e00-\u9fff][\u4e00-\u9fffA-Za-z0-9 _.-]{1,40}"
    english_alias = r"[A-Za-z][A-Za-z0-9 _.'-]{1,78}"
    for match in re.finditer(
        rf"(?P<label>{chinese_label})\s*[（(]\s*(?P<alias>{english_alias})\s*[）)]",
        prompt,
    ):
        matches.append((match.group("label").strip(), match.group("alias").strip()))
    for match in re.finditer(
        rf"(?P<alias>{english_alias})\s*[（(]\s*(?P<label>{chinese_label})\s*[）)]",
        prompt,
    ):
        matches.append((match.group("label").strip(), match.group("alias").strip()))

    anchors: list[SemanticAnchor] = []
    seen_aliases: set[str] = set()
    for label, alias in matches:
        candidate = re.sub(r"\s+", "_", alias.lower()).strip("_")
        if not re.fullmatch(r"[a-z0-9_.'()-]{2,120}", candidate):
            continue
        if candidate in seen_aliases:
            continue
        seen_aliases.add(candidate)
        anchors.append(
            SemanticAnchor(
                anchor_id=f"parenthesized_character_alias_{len(anchors) + 1}",
                role="target_character",
                group="character",
                source_text=alias,
                description=f"Explicit character alias for {label}",
                candidates=(candidate,),
            )
        )
    return tuple(anchors)


def extract_parenthesized_copyright_aliases(user_prompt: str) -> tuple[SemanticAnchor, ...]:
    """Extract explicit English work titles paired with a Chinese book-title mark.

    Args:
        user_prompt: The original image request.

    Returns:
        Copyright lookup anchors independent of the semantic-planning LLM.
    """
    anchors: list[SemanticAnchor] = []
    prompt = str(user_prompt or "")
    for match in re.finditer(
        r"《(?P<title>[^》]{1,80})》\s*[（(]\s*"
        r"(?P<alias>[A-Za-z][A-Za-z0-9 _.'!: -]{1,78})\s*[）)]",
        prompt,
    ):
        title = match.group("title").strip()
        alias = match.group("alias").strip()
        candidate = re.sub(r"\s+", "_", alias.lower()).strip("_")
        if not re.fullmatch(r"[a-z0-9_.'():!\-]{2,120}", candidate):
            continue
        anchors.append(
            SemanticAnchor(
                anchor_id=f"parenthesized_copyright_alias_{len(anchors) + 1}",
                role="copyright",
                group="series",
                source_text=alias,
                description=f"Explicit work title for {title}",
                candidates=(candidate,),
            )
        )
    return tuple(anchors)


def resolve_local_cli_path(configured_path: str = "") -> Path | None:
    """Resolve one explicit/well-known local CLI path without filesystem search."""
    candidates: list[Path] = []
    if str(configured_path or "").strip():
        candidates.append(Path(str(configured_path).strip()).expanduser())
    skills_root = str(os.environ.get("COMFYUI_GOOD_ANIMA_SKILLS_DIR") or "").strip()
    if skills_root:
        candidates.append(Path(skills_root) / "danbooru-tags" / "bin" / "danbooru-tags.exe")
    candidates.append(
        Path.home() / ".codex" / "skills" / "danbooru-tags" / "bin" / "danbooru-tags.exe"
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _run_cli_batch(
    queries: list[dict[str, Any]],
    *,
    cli_path: Path,
    timeout: float,
    exact_only: bool = False,
) -> dict[str, Any] | None:
    """Run one bounded local CLI batch and return decoded JSON."""
    payload = json.dumps({"queries": queries}, ensure_ascii=False)
    batch_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", delete=False
        ) as handle:
            handle.write(payload)
            batch_path = handle.name
        command = [
            str(cli_path),
            "--batch-file",
            batch_path,
            "--batch-workers",
            "4",
            "--for-prompt",
            "--json",
            "--compact",
        ]
        if exact_only:
            command.extend(("--match-mode", "exact"))
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1.0, min(float(timeout), 20.0)),
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        if batch_path:
            try:
                Path(batch_path).unlink(missing_ok=True)
            except OSError:
                pass
    if completed.returncode != 0:
        return None
    try:
        data = json.loads(completed.stdout)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def lookup_semantic_anchors(
    anchors: tuple[SemanticAnchor, ...],
    *,
    cli_path: Path,
    timeout: float = 8.0,
) -> SemanticLookupResult:
    """Validate all proposed candidates with one local batch invocation."""
    if not anchors:
        return SemanticLookupResult(status="empty_plan")
    queries: list[dict[str, Any]] = []
    query_map: dict[str, tuple[int, int, bool, str]] = {}
    for anchor_index, anchor in enumerate(anchors):
        primary_candidates = list(anchor.candidates[:2])
        raw_phrase = str(anchor.source_text or "").strip()
        fallback_phrase = ""
        if re.fullmatch(r"[A-Za-z0-9_.'():!\- ]{2,80}", raw_phrase):
            fallback_phrase = raw_phrase
        query_candidates = list(primary_candidates)
        if fallback_phrase and re.sub(r"\s+", "_", fallback_phrase.lower()) not in {
            re.sub(r"\s+", "_", item.lower()) for item in query_candidates
        }:
            query_candidates.append(fallback_phrase)
        for candidate_index, candidate in enumerate(query_candidates[:3]):
            query_id = f"a{anchor_index}_c{candidate_index}"
            candidate_key = re.sub(r"\s+", "_", candidate.lower()).strip("_")
            source_key = re.sub(r"\s+", "_", raw_phrase.lower()).strip("_")
            # A raw source-word query needs intent-filtered candidate handling
            # even when the planner happened to include that same word among
            # its primary guesses.
            is_fallback = bool(source_key and candidate_key == source_key)
            query_map[query_id] = (
                anchor_index,
                candidate_index,
                is_fallback,
                candidate,
            )
            queries.append(
                {
                    "id": query_id,
                    "group": anchor.group,
                    "keyword": candidate,
                    "limit": 5 if anchor.group in {"character", "series"} else 10,
                }
            )
    data = _run_cli_batch(queries, cli_path=cli_path, timeout=timeout)
    if data is None:
        return SemanticLookupResult(anchors=anchors, status="tool_failed")
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, dict):
        return SemanticLookupResult(anchors=anchors, status="tool_failed")

    confirmed_by_anchor: dict[int, str] = {}
    inferred_series_tags: list[str] = []
    candidates: list[str] = []
    fallback_candidates: dict[int, list[tuple[str, int]]] = {}
    named_outfit_by_anchor: dict[int, str] = {}

    def is_specific_school_uniform(anchor: SemanticAnchor) -> bool:
        source = re.sub(r"[\s\"“”'‘’]+", "", anchor.source_text).lower()
        if not re.search(r"(?:校服|制服|schooluniform)$", source, flags=re.I):
            return False
        modifier = re.sub(
            r"(?:校服|制服|schooluniform)$", "", source, flags=re.I
        ).strip()
        return len(modifier) >= 2

    def outfit_tag_has_required_specificity(
        anchor: SemanticAnchor, tag: str
    ) -> bool:
        tag_key = re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")
        if is_specific_school_uniform(anchor) and tag_key in {
            "school_uniform",
            "uniform",
        }:
            return False
        return True

    def is_named_outfit_anchor(anchor: SemanticAnchor, tag: str) -> bool:
        tag_key = re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")
        return outfit_tag_has_required_specificity(anchor, tag) and (
            anchor.role == "outfit" or bool(
            anchor.role == "clothing"
            and tag_key.endswith("_uniform")
            and re.search(r"(?:校服|制服|uniform)", anchor.source_text, flags=re.I)
            )
        )

    for query_id, result in results.items():
        mapping = query_map.get(str(query_id))
        if not isinstance(result, dict):
            continue
        if mapping is None:
            continue
        anchor_index, candidate_index, is_fallback, query_phrase = mapping
        confirmed = result.get("confirmed_tags")
        if isinstance(confirmed, dict):
            records = [
                record
                for values in confirmed.values()
                if isinstance(values, list)
                for record in values
                if isinstance(record, dict)
            ]
            if records and (
                anchor_index not in confirmed_by_anchor
                or candidate_index == 0
            ):
                tag = next(
                    (
                        str(record.get("tag") or "").strip()
                        for record in records
                        if outfit_tag_has_required_specificity(
                            anchors[anchor_index],
                            str(record.get("tag") or "").strip(),
                        )
                    ),
                    "",
                )
                if tag:
                    confirmed_by_anchor[anchor_index] = tag
                    if is_named_outfit_anchor(anchors[anchor_index], tag):
                        named_outfit_by_anchor[anchor_index] = tag
        raw_candidates = result.get("candidate_tags")
        if isinstance(raw_candidates, dict):
            for values in raw_candidates.values():
                if not isinstance(values, list):
                    continue
                for record in values[:3]:
                    if not isinstance(record, dict):
                        continue
                    tag = str(record.get("tag") or "").strip()
                    if tag and tag not in candidates:
                        candidates.append(tag)
                    anchor = anchors[anchor_index]
                    tag_key = re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")
                    query_key = re.sub(
                        r"[^a-z0-9]+", "_", query_phrase.lower()
                    ).strip("_")
                    try:
                        count = int(record.get("count") or 0)
                    except (TypeError, ValueError):
                        count = 0
                    match_layer = str(record.get("match_layer") or "").strip()
                    source_category = str(
                        record.get("source_category") or record.get("category") or ""
                    ).strip()
                    # The local index classifies many canonical clothing-set tags as
                    # general tags.  They therefore arrive as candidates even when
                    # their spelling exactly matches the planner's explicit lookup.
                    # Promote only a high-evidence, user-requested complete ensemble;
                    # ordinary garments and fuzzy candidates remain non-authoritative.
                    if (
                        is_named_outfit_anchor(anchor, tag)
                        and tag_key == query_key
                        and match_layer == "group_general_fallback"
                        and source_category == "general"
                        and count >= 10
                    ):
                        confirmed_by_anchor[anchor_index] = tag
                        named_outfit_by_anchor[anchor_index] = tag
                        continue
                    if not tag or not is_fallback:
                        continue
                    if anchor.group != "character":
                        continue
                    if tag_key != query_key and not tag.lower().startswith(
                        query_key + "_("
                    ):
                        continue
                    values = fallback_candidates.setdefault(anchor_index, [])
                    if (tag, count) not in values:
                        values.append((tag, count))

    confirmed_copyright_tags = {
        tag
        for index, tag in confirmed_by_anchor.items()
        if anchors[index].role == "copyright"
    }
    selected_fallbacks: dict[int, tuple[str, int]] = {}
    for index, values in fallback_candidates.items():
        compatible = []
        for tag, count in values:
            scoped = re.fullmatch(r".+_\(([^)]+)\)", tag)
            scope = scoped.group(1) if scoped else ""
            if any(
                copyright == scope
                or copyright.startswith(scope + "_")
                or scope.startswith(copyright + "_")
                for copyright in confirmed_copyright_tags
            ):
                compatible.append((tag, count))
        if confirmed_copyright_tags:
            selected = compatible
        elif len(values) == 1:
            selected = values
        else:
            continue
        selected_fallbacks[index] = max(selected, key=lambda item: item[1])

    unresolved_fallbacks = {
        index: value[0]
        for index, value in selected_fallbacks.items()
        if index not in confirmed_by_anchor
    }
    fallback_queries = [
            {
                "id": f"fallback_{index}",
                "group": anchors[index].group,
                "keyword": tag,
                "limit": 5,
            }
            for index, tag in unresolved_fallbacks.items()
        ]
    series_candidates: list[str] = []
    for tag in (*confirmed_by_anchor.values(), *unresolved_fallbacks.values()):
        scoped = re.fullmatch(r".+_\(([^)]+)\)", tag)
        if scoped and scoped.group(1) not in series_candidates:
            series_candidates.append(scoped.group(1))
    fallback_queries.extend(
        {
            "id": f"series_{index}",
            "group": "series",
            "keyword": series,
            "limit": 5,
        }
        for index, series in enumerate(series_candidates)
    )
    if fallback_queries:
        fallback_data = _run_cli_batch(
            fallback_queries,
            cli_path=cli_path,
            timeout=timeout,
            exact_only=True,
        )
        fallback_results = (
            fallback_data.get("results") if isinstance(fallback_data, dict) else None
        )
        if isinstance(fallback_results, dict):
            for index, selected_tag in unresolved_fallbacks.items():
                result = fallback_results.get(f"fallback_{index}")
                confirmed = result.get("confirmed_tags") if isinstance(result, dict) else None
                records = [
                    record
                    for values in confirmed.values()
                    if isinstance(confirmed, dict) and isinstance(values, list)
                    for record in values
                    if isinstance(record, dict)
                ] if isinstance(confirmed, dict) else []
                if any(str(record.get("tag") or "") == selected_tag for record in records):
                    confirmed_by_anchor[index] = selected_tag
            for index, series in enumerate(series_candidates):
                result = fallback_results.get(f"series_{index}")
                confirmed = result.get("confirmed_tags") if isinstance(result, dict) else None
                if not isinstance(confirmed, dict):
                    continue
                for values in confirmed.values():
                    if not isinstance(values, list):
                        continue
                    for record in values:
                        if not isinstance(record, dict):
                            continue
                        tag = str(record.get("tag") or "").strip()
                        if tag == series and tag not in inferred_series_tags:
                            inferred_series_tags.append(tag)

    confirmed_tags: list[str] = []
    source_tags: list[str] = []
    missing: list[str] = []
    role_priority = {
        "target_character": 0,
        "outfit_source": 1,
        "copyright": 2,
        "clothing": 3,
        "outfit": 3,
        "appearance": 4,
        "accessory": 5,
        "prop": 6,
        "expression": 7,
        "pose": 8,
        "action": 8,
        "scene": 9,
        "lighting": 10,
    }
    ordered_indices = sorted(
        range(len(anchors)),
        key=lambda index: (role_priority.get(anchors[index].role, 50), index),
    )
    series_inserted = False
    for index in ordered_indices:
        anchor = anchors[index]
        tag = confirmed_by_anchor.get(index, "")
        if tag:
            if tag not in confirmed_tags:
                confirmed_tags.append(tag)
            if anchor.role == "outfit_source" and tag not in source_tags:
                source_tags.append(tag)
                for series_tag in inferred_series_tags:
                    if series_tag not in confirmed_tags:
                        confirmed_tags.append(series_tag)
                series_inserted = True
        elif anchor.description not in missing:
            missing.append(anchor.description)
    if not series_inserted:
        for tag in inferred_series_tags:
            if tag not in confirmed_tags:
                confirmed_tags.append(tag)
    return SemanticLookupResult(
        confirmed_tags=tuple(confirmed_tags),
        outfit_source_tags=tuple(source_tags),
        named_outfit_tags=tuple(
            dict.fromkeys(named_outfit_by_anchor.values())
        ),
        missing_descriptions=tuple(missing),
        candidate_tags=tuple(candidates[:12]),
        anchors=anchors,
        status="resolved",
    )


def merge_semantic_results(
    *results: SemanticLookupResult | None,
) -> SemanticLookupResult:
    """Merge cache and request lookup evidence without letting either shadow the other."""
    present = tuple(result for result in results if result is not None)
    if not present:
        return SemanticLookupResult()

    def merged(field: str) -> tuple[Any, ...]:
        return tuple(
            dict.fromkeys(
                item
                for result in present
                for item in getattr(result, field, ())
            )
        )

    statuses = tuple(result.status for result in present if result.status)
    if "resolved" in statuses:
        status = "resolved"
    elif "profile_cache" in statuses:
        status = "profile_cache"
    else:
        status = statuses[-1] if statuses else "not_available"
    return SemanticLookupResult(
        confirmed_tags=merged("confirmed_tags"),
        outfit_source_tags=merged("outfit_source_tags"),
        outfit_profile_tags=merged("outfit_profile_tags"),
        appearance_profile_tags=merged("appearance_profile_tags"),
        named_outfit_tags=merged("named_outfit_tags"),
        source_outfit_profiles=merged("source_outfit_profiles"),
        missing_descriptions=merged("missing_descriptions"),
        candidate_tags=merged("candidate_tags"),
        anchors=merged("anchors"),
        status=status,
    )
