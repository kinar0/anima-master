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
    missing_descriptions: tuple[str, ...] = ()
    candidate_tags: tuple[str, ...] = ()
    anchors: tuple[SemanticAnchor, ...] = ()
    status: str = "not_available"

    def prompt_context(self) -> str:
        """Render verified evidence for the final prompt-writing LLM."""
        if self.status == "not_available" or not self.anchors:
            return ""
        lines = [
            "Local Danbooru validation (authoritative for hard tags):",
            "Use confirmed tags exactly. Do not transliterate missing concepts into invented tags.",
        ]
        if self.confirmed_tags:
            lines.append("confirmed hard tags: " + ", ".join(self.confirmed_tags))
        if self.outfit_source_tags:
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
        if self.outfit_profile_tags:
            lines.append(
                "visible outfit tags extracted from source posts: "
                + ", ".join(self.outfit_profile_tags)
            )
        if self.missing_descriptions:
            lines.append(
                "unresolved concepts; express these only in Nltags: "
                + "; ".join(self.missing_descriptions)
            )
        return "\n".join(lines)


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
        "expression, pose, action, clothing, accessory, prop, scene, lighting. "
        "Allowed groups are the same except target_character/outfit_source use "
        "character and copyright uses series. Use at most 12 anchors and at most "
        "3 candidates per anchor. Candidates are untrusted lookup hints, not "
        "answers. Use lowercase Danbooru spelling with underscores. Do not turn "
        "an outfit_source into a visible target character. Omit ordinary prose "
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
    fallback_candidates: dict[int, tuple[str, int]] = {}
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
                tag = str(records[0].get("tag") or "").strip()
                if tag:
                    confirmed_by_anchor[anchor_index] = tag
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
                    if not tag or not is_fallback:
                        continue
                    anchor = anchors[anchor_index]
                    if anchor.group != "character":
                        continue
                    query_key = re.sub(r"[^a-z0-9]+", "_", query_phrase.lower()).strip("_")
                    tag_key = tag.lower()
                    if tag_key != query_key and not tag_key.startswith(query_key + "_("):
                        continue
                    try:
                        count = int(record.get("count") or 0)
                    except (TypeError, ValueError):
                        count = 0
                    previous = fallback_candidates.get(anchor_index)
                    if previous is None or count > previous[1]:
                        fallback_candidates[anchor_index] = (tag, count)

    unresolved_fallbacks = {
        index: value[0]
        for index, value in fallback_candidates.items()
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
        missing_descriptions=tuple(missing),
        candidate_tags=tuple(candidates[:12]),
        anchors=anchors,
        status="resolved",
    )
