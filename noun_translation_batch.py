"""Offline, review-first Chinese aliases for the local Anima tag catalog.

The batch is deliberately separate from image generation and never edits the
live ``danbooru_term_mappings`` configuration.
"""

from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


try:
    from .mapping_syntax import parse_mapping_entry
except ImportError:  # Direct script-style imports.
    from mapping_syntax import parse_mapping_entry


_CHINESE = re.compile(r"[\u3400-\u9fff]")
_TAG = re.compile(r"[a-z0-9_.'():!\-]{2,120}\Z")


def select_general_tags(source: Path, *, minimum_posts: int = 1000,
                        limit: int | None = None) -> list[dict[str, Any]]:
    """Select all eligible general tags by popularity, optionally limiting results."""
    if not source.is_file():
        raise FileNotFoundError(f"Anima CSV 不存在：{source}")
    stat = source.stat()
    return [dict(row) for row in _select_general_tags_cached(
        str(source.resolve()), stat.st_mtime_ns, stat.st_size, minimum_posts, limit)]


@lru_cache(maxsize=4)
def _select_general_tags_cached(source_path: str, mtime: int, size: int,
                                minimum_posts: int, limit: int | None):
    source = Path(source_path)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 3 or row[1] != "0":
                continue
            tag = row[0].strip()
            try:
                count = int(row[2])
            except ValueError:
                continue
            if count < minimum_posts or not _TAG.fullmatch(tag) or tag in seen:
                continue
            seen.add(tag)
            selected.append({"tag": tag, "post_count": count})
    selected.sort(key=lambda item: (-item["post_count"], item["tag"]))
    return tuple(selected[:limit])


def translation_prompt(batch: list[dict[str, Any]]) -> str:
    """Ask for short literal aliases; keep the canonical tag as the ID."""
    tags = [item["tag"] for item in batch]
    return (
        "将以下 Anima/Danbooru 通用标签译为用户可能输入的简体中文名词或短语。"
        "别名必须与原标签的含义等价：不得把上位概念译成一种子类，"
        "也不得把具体标签译成泛称。例如 skirt 不能译为短裙，"
        "underwear 不能译为内裤，dress 不能译为裙子。"
        "保留大小、颜色、部位、人数和动作等限定；不要凭英文别名扩展含义。"
        "有歧义、不适合直接按中文子串触发、或无法准确翻译的标签，aliases 留空。"
        "每个标签最多给 2 个自然中文别名，不要输出泛化词或完整句子。"
        "只返回 JSON 数组，逐项格式为 {\"tag\":\"原标签\",\"aliases\":[\"中文别名\"]}。"
        "必须包含全部输入标签，顺序一致，不要 Markdown。\n"
        + json.dumps(tags, ensure_ascii=False)
    )


def _completion_text(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()
    for name in ("completion_text", "text", "content"):
        value = getattr(response, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def parse_translation(response: str, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reject incomplete or malformed replies instead of silently skipping tags."""
    body = response.strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body).strip()
    rows = json.loads(body)
    if not isinstance(rows, list) or len(rows) != len(batch):
        raise ValueError("翻译响应数量与请求不一致")
    result: list[dict[str, Any]] = []
    for source, row in zip(batch, rows):
        if not isinstance(row, dict) or row.get("tag") != source["tag"]:
            raise ValueError("翻译响应的 canonical tag 或顺序不一致")
        aliases = row.get("aliases")
        if not isinstance(aliases, list) or len(aliases) > 2:
            raise ValueError("中文别名格式无效")
        clean: list[str] = []
        for alias in aliases:
            if not isinstance(alias, str):
                raise ValueError("中文别名必须是字符串")
            alias = re.sub(r"\s+", " ", alias.strip())
            if (not _CHINESE.search(alias) or len(alias) > 30
                    or any(char in alias for char in "=|\n\r\t")):
                raise ValueError(f"中文别名无效：{alias}")
            if alias not in clean:
                clean.append(alias)
        result.append({**source, "aliases": clean})
    return result


def review_rows(rows: list[dict[str, Any]], *,
                existing_mappings: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Flag unsafe aliases, including collisions with the live manual lexicon."""
    owners: dict[str, set[str]] = {}
    for row in rows:
        for alias in row["aliases"]:
            owners.setdefault(alias.casefold(), set()).add(row["tag"])
    for alias, tag in (existing_mappings or {}).items():
        owners.setdefault(alias.casefold(), set()).add(tag)
    reviewed = []
    # Traverse literal substrings instead of comparing every alias to the whole
    # lexicon. Keep both containment directions and the exact old flag meaning.
    trie: dict = {}
    related: dict[str, set[str]] = {alias: set() for alias in owners}
    for alias in owners:
        branch = trie
        for character in alias:
            branch = branch.setdefault(character, {})
        branch[None] = alias
    for longer in owners:
        for start in range(len(longer)):
            branch = trie
            for character in longer[start:]:
                if character not in branch:
                    break
                branch = branch[character]
                shorter = branch.get(None)
                if shorter is not None and shorter != longer:
                    related[longer].update(owners[shorter])
                    related[shorter].update(owners[longer])
    for row in rows:
        flags = []
        for alias in row["aliases"]:
            if len(alias) < 2:
                flags.append(f"过短:{alias}")
            if len(owners[alias.casefold()]) > 1:
                flags.append(f"别名冲突:{alias}")
            if related[alias.casefold()] - {row["tag"]}:
                flags.append(f"与其他标签的触发词有包含关系:{alias}")
        reviewed.append({**row, "review_flags": flags})
    return reviewed


def parse_existing_mappings(entries: Any) -> dict[str, str]:
    """Read current user mappings for review without changing their authority."""
    if isinstance(entries, dict):
        return {str(alias).strip().casefold(): str(tag).strip().lower()
                for alias, tag in entries.items() if str(alias).strip()}
    mappings: dict[str, str] = {}
    if isinstance(entries, (list, tuple)):
        for entry in entries:
            pair = parse_mapping_entry(entry)
            if pair:
                mappings[pair[0].casefold()] = pair[1].lower()
    return mappings


def save_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def verification_prompt(batch: list[dict[str, Any]],
                        existing_mappings: dict[str, str]) -> str:
    """Request an independent semantic check of already translated aliases."""
    return (
        "独立审核这些 Danbooru 标签的中文别名。只保留与原 canonical tag 完全等价、"
        "适合在用户中文生图描述中直接触发的别名。上位词、子类词、含义有歧义的词必须删除；"
        "例如不能把裙子当 dress、短裙当 skirt、内裤当 underwear。"
        "不要新增或改写别名。参考已有用户映射，同一中文别名不得指向不同 tag。"
        "只返回 JSON 数组，顺序与输入相同，每项仅含 tag 和 approved_aliases；"
        "approved_aliases 必须是该项 aliases 的子集，可为空。不要 Markdown。\n"
        "已有映射：" + json.dumps(existing_mappings, ensure_ascii=False) + "\n"
        "待审核：" + json.dumps(
            [{"tag": row["tag"], "aliases": row["aliases"]} for row in batch],
            ensure_ascii=False,
        )
    )


def parse_verification(response: str,
                       batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    body = response.strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body).strip()
    rows = json.loads(body)
    if not isinstance(rows, list) or len(rows) != len(batch):
        raise ValueError("审核响应数量与请求不一致")
    verified = []
    for source, row in zip(batch, rows):
        if not isinstance(row, dict) or row.get("tag") != source["tag"]:
            raise ValueError("审核响应的 canonical tag 或顺序不一致")
        approved = row.get("approved_aliases")
        if (not isinstance(approved, list)
                or any(not isinstance(alias, str) or alias not in source["aliases"]
                       for alias in approved)):
            raise ValueError("审核模型增加了未经初译的别名")
        verified.append({"tag": source["tag"],
                         "approved_aliases": list(dict.fromkeys(approved))})
    return verified


async def verify_with_astrbot(context: Any, provider_id: str,
                              translated_path: Path, verified_path: Path,
                              existing_mappings: dict[str, str], *,
                              batch_size: int = 25,
                              max_batches: int = 4) -> dict[str, int]:
    """Independently verify all pending translated rows in bounded batches."""
    if not provider_id.strip():
        raise ValueError("请指定审核 provider ID")
    if batch_size < 1 or batch_size > 25 or max_batches < 1 or max_batches > 4:
        raise ValueError("审核批次超出安全范围")
    translated = json.loads(translated_path.read_text(encoding="utf-8"))
    verified = json.loads(verified_path.read_text(encoding="utf-8")) if verified_path.exists() else []
    if not isinstance(translated, list) or not isinstance(verified, list):
        raise ValueError("翻译或审核断点格式无效")
    translated_by_tag = {row["tag"]: row for row in translated}
    completed: dict[str, dict[str, Any]] = {}
    for row in verified:
        tag = row.get("tag") if isinstance(row, dict) else None
        if tag not in translated_by_tag or not set(row.get("approved_aliases", [])) <= set(
            translated_by_tag[tag]["aliases"]
        ):
            raise ValueError("审核断点与初译结果不一致")
        completed[tag] = row
    pending = [row for row in translated if row["tag"] not in completed]
    previously_verified = len(completed)
    calls = 0
    input_tokens = 0
    cached_tokens = 0
    output_tokens = 0
    for offset in range(0, min(len(pending), batch_size * max_batches), batch_size):
        batch = pending[offset:offset + batch_size]
        reply = await context.llm_generate(
            chat_provider_id=provider_id,
            prompt=verification_prompt(batch, existing_mappings),
            system_prompt="你是独立的 Danbooru 中文别名审核员，只输出指定 JSON。",
            max_tokens=1800,
            thinking={"type": "disabled"},
        )
        parsed = parse_verification(_completion_text(reply), batch)
        usage = getattr(reply, "usage", None)
        input_tokens += int(getattr(usage, "input_other", 0) or 0)
        cached_tokens += int(getattr(usage, "input_cached", 0) or 0)
        output_tokens += int(getattr(usage, "output", 0) or 0)
        completed.update({row["tag"]: row for row in parsed})
        save_checkpoint(verified_path, [completed[row["tag"]] for row in translated
                                        if row["tag"] in completed])
        calls += 1
    return {"calls": calls, "verified": len(completed),
            "newly_verified": len(completed) - previously_verified,
            "remaining": len(translated) - len(completed),
            "input_tokens": input_tokens, "cached_tokens": cached_tokens,
            "output_tokens": output_tokens}


def build_import_plan(translated: list[dict[str, Any]],
                      verified: list[dict[str, Any]],
                      existing_mappings: dict[str, str]) -> dict[str, Any]:
    """Keep only independently approved, conflict-free aliases for import."""
    source = {row["tag"]: set(row["aliases"]) for row in translated}
    proposals: dict[str, set[str]] = {}
    for row in verified:
        tag = row.get("tag") if isinstance(row, dict) else None
        approved = row.get("approved_aliases") if isinstance(row, dict) else None
        if tag not in source or not isinstance(approved, list):
            raise ValueError("审核结果缺少有效来源")
        for alias in approved:
            if not isinstance(alias, str) or alias not in source[tag]:
                raise ValueError("审核结果包含未经初译的别名")
            proposals.setdefault(alias.casefold(), set()).add(tag)
    additions: dict[str, str] = {}
    skipped: dict[str, str] = {}
    existing: dict[str, str] = {}
    for alias, tags in proposals.items():
        if (len(alias) < 2 or len(alias) > 30 or not _CHINESE.search(alias)
                or any(char in alias for char in "=|\n\r\t")):
            skipped[alias] = "触发词过短、过长或无中文"
        elif len(tags) != 1:
            skipped[alias] = "多个 canonical tag 争用同一别名"
        elif alias in existing_mappings:
            if existing_mappings[alias] in tags:
                existing[alias] = existing_mappings[alias]
            else:
                skipped[alias] = "与手工映射冲突"
        else:
            additions[alias] = next(iter(tags))
    return {"additions": additions, "skipped": skipped, "existing": existing}


def translation_review_rows(translated: list[dict[str, Any]],
                            verified: list[dict[str, Any]],
                            existing_mappings: dict[str, str],
                            plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe each alias against the current lexicon, not import history.

    A row can have both importable and unresolved aliases. Advisory substring
    flags are not import failures; an exact mapping already in the lexicon is
    resolved even when an old model checkpoint rejected it.
    """
    approved_by_tag = {row["tag"]: row["approved_aliases"] for row in verified}
    result = []
    for row in translated:
        tag = row["tag"]
        approved = approved_by_tag.get(tag)
        details = []
        for alias in row["aliases"]:
            key = alias.casefold()
            current = existing_mappings.get(key)
            if current == tag:
                state, reason = "existing", "已在词库"
            elif approved is None:
                state, reason = "pending", "待模型复核"
            elif current is not None:
                state, reason = "conflict", f"与现有映射冲突：当前指向 {current}"
            elif alias not in approved:
                state, reason = "rejected", "模型复核未通过，未导入"
            elif plan["additions"].get(key) == tag:
                state, reason = "ready", "可导入"
            else:
                state, reason = "conflict", plan["skipped"][key]
            details.append({"alias": alias, "state": state, "reason": reason})
        reasons = [f"{item['alias']}：{item['reason']}" for item in details
                   if item["state"] in {"conflict", "rejected"}]
        if approved is not None and not row["aliases"]:
            reasons.append("初译未提供中文别名，未导入")
        states = {item["state"] for item in details}
        result.append({
            **row, "approved_aliases": approved, "alias_statuses": details,
            "needs_manual_review": bool(reasons), "manual_review_reasons": reasons,
            "ready": "ready" in states,
            "in_lexicon": bool(details) and states == {"existing"},
        })
    return result


async def translate_with_astrbot(context: Any, provider_id: str,
                                 candidates: list[dict[str, Any]],
                                 checkpoint: Path, *, batch_size: int = 25,
                                 max_batches: int = 2) -> dict[str, int]:
    """Run a bounded, resumable batch through AstrBot's configured provider."""
    if not provider_id.strip():
        raise ValueError("请先配置名词翻译专用 provider ID")
    if batch_size < 1 or batch_size > 25 or max_batches < 1 or max_batches > 10:
        raise ValueError("批次大小或批次数超出安全范围")
    existing = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else []
    if not isinstance(existing, list):
        raise ValueError("翻译断点文件格式无效")
    candidate_by_tag = {row["tag"]: row for row in candidates}
    completed: dict[str, dict[str, Any]] = {}
    for row in existing:
        if not isinstance(row, dict) or row.get("tag") not in candidate_by_tag:
            raise ValueError("翻译断点与当前候选清单不匹配")
        completed[row["tag"]] = row
    pending = [row for row in candidates if row["tag"] not in completed]
    previously_translated = len(completed)
    calls = 0
    input_tokens = 0
    cached_tokens = 0
    output_tokens = 0
    for offset in range(0, min(len(pending), batch_size * max_batches), batch_size):
        batch = pending[offset:offset + batch_size]
        reply = await context.llm_generate(
            chat_provider_id=provider_id,
            prompt=translation_prompt(batch),
            system_prompt="你是谨慎的 Danbooru 标签中文词典编纂员。只输出指定 JSON。",
            max_tokens=1800,
            thinking={"type": "disabled"},
        )
        parsed = parse_translation(_completion_text(reply), batch)
        usage = getattr(reply, "usage", None)
        input_tokens += int(getattr(usage, "input_other", 0) or 0)
        cached_tokens += int(getattr(usage, "input_cached", 0) or 0)
        output_tokens += int(getattr(usage, "output", 0) or 0)
        completed.update({row["tag"]: row for row in parsed})
        save_checkpoint(checkpoint, [completed[row["tag"]] for row in candidates
                                     if row["tag"] in completed])
        calls += 1
    return {"calls": calls, "completed": len(completed),
            "newly_translated": len(completed) - previously_translated,
            "remaining": len(candidates) - len(completed),
            "input_tokens": input_tokens, "cached_tokens": cached_tokens,
            "output_tokens": output_tokens}
