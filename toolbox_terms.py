"""Offline Toolbox vocabulary import into the existing noun mapping setting.

No model calls, network requests, profile changes, or runtime vocabulary source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

try:
    from .mapping_syntax import parse_mapping_entry
except ImportError:  # Direct script-style imports.
    from mapping_syntax import parse_mapping_entry

MAX_TERMS = 50000
TAG = re.compile(r"[a-z0-9_.'():!\-]{2,120}")
HAN = re.compile(r"[\u4e00-\u9fff]")
FIELDS = ("zhFull", "zhShort", "zhNick", "zh")


def extract_terms(data: dict) -> dict:
    """Keep general-tag aliases with one unambiguous canonical owner."""
    if not isinstance(data, dict) or not isinstance(data.get("general"), list):
        raise ValueError("Toolbox general 标签索引格式无效")
    owners: dict[str, set[str]] = defaultdict(set)
    invalid_records = []

    def reject(index, field, value, reason):
        invalid_records.append({"entry": index, "field": field,
                                "value": str(value)[:160], "reason": reason})

    for index, entry in enumerate(data["general"], 1):
        if not isinstance(entry, dict):
            reject(index, "entry", entry, "entry_not_object")
            continue
        tag = str(entry.get("tag", "")).strip().lower().replace(" ", "_")
        if not TAG.fullmatch(tag):
            reject(index, "tag", entry.get("tag"), "invalid_canonical_tag")
            continue
        for field in FIELDS:
            aliases = entry.get(field, [])
            if not isinstance(aliases, list):
                reject(index, field, type(aliases).__name__, "aliases_not_array")
                continue
            for raw in aliases:
                if not isinstance(raw, str):
                    reject(index, field, raw, "alias_not_text")
                    continue
                alias = " ".join(raw.split()).lower()
                reason = (
                    "alias_length_out_of_range" if not 2 <= len(alias) <= 80 else
                    "alias_without_han" if not HAN.search(alias) else
                    "alias_contains_separator" if any(c in raw for c in "=|") else
                    "alias_contains_control" if any(ord(c) < 32 for c in raw) else ""
                )
                if reason:
                    reject(index, field, raw, reason)
                    continue
                owners[alias].add(tag)
    return {
        "candidates": {a: next(iter(ts)) for a, ts in sorted(owners.items()) if len(ts) == 1},
        "ambiguous": {a: sorted(ts) for a, ts in sorted(owners.items()) if len(ts) > 1},
        "invalid_occurrences": len(invalid_records), "invalid_records": invalid_records,
    }


def mapping_rows(config: dict) -> tuple[dict, str]:
    """Locate the actual setting while preserving unrelated grouped config."""
    if "danbooru_term_mappings" in config:
        return config, "danbooru_term_mappings"
    groups = [v for v in config.values() if isinstance(v, dict)
              and "danbooru_term_mappings" in v]
    if len(groups) != 1:
        raise ValueError("无法唯一定位现有名词翻译配置")
    return groups[0], "danbooru_term_mappings"


def existing_terms(rows: list) -> dict[str, str]:
    """Reject malformed or conflicting existing data rather than overwrite it."""
    if not isinstance(rows, list):
        raise ValueError("现有名词翻译不是列表")
    result = {}
    for row in rows:
        pair = parse_mapping_entry(row)
        if pair is None:
            raise ValueError("现有词条格式无效，未写入任何配置")
        alias = pair[0].lower()
        tag = pair[1].lower().replace(" ", "_")
        if alias in result and result[alias] != tag:
            raise ValueError("现有词库存在冲突别名，未写入任何配置")
        result[alias] = tag
    return result


def merge_terms(rows: list, extracted: dict) -> dict:
    current = existing_terms(rows)
    additions, already, conflicts = {}, {}, {}
    for alias, tag in extracted["candidates"].items():
        if alias not in current:
            additions[alias] = tag
        elif current[alias] == tag:
            already[alias] = tag
        else:
            conflicts[alias] = {"existing": current[alias], "candidate": tag}
    merged = [*rows, *(f"{a}={t}" for a, t in additions.items())]
    if len(merged) > MAX_TERMS:
        raise ValueError(f"合并后 {len(merged)} 条超过 {MAX_TERMS} 上限，未写入任何配置")
    return {"additions": additions, "already": already, "conflicts": conflicts,
            "total": len(merged), "rows": merged}


def atomic_json(path: Path, value: dict, *, expected_bytes: bytes | None = None) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temp = path.with_name(path.name + ".toolbox.tmp")
    try:
        # Keep the journal digest identical to actual bytes on Windows too.
        temp.write_bytes(text.encode("utf-8"))
        if expected_bytes is not None and path.read_bytes() != expected_bytes:
            raise ValueError("配置已变更，请重新预览合并")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def run_import(source: Path, config_path: Path, report: Path, *, apply: bool = False) -> dict:
    source_bytes = source.read_bytes()
    config_bytes = config_path.read_bytes()
    extracted = extract_terms(json.loads(source_bytes.decode("utf-8-sig")))
    config = json.loads(config_bytes.decode("utf-8-sig"))
    target, key = mapping_rows(config)
    rows = target[key]
    merged = merge_terms(rows, extracted)
    target[key] = merged["rows"]
    expected_after = (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    target[key] = rows
    manifest = {
        "source": str(source.resolve()), "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_version": "0.9.8", "rules_version": 1, "mapping_parser_version": 2, "applied": False,
        "config_path": str(config_path.resolve()),
        "config_sha256_before": hashlib.sha256(config_bytes).hexdigest(),
        "config_sha256_after": hashlib.sha256(expected_after).hexdigest(),
        "previous_rows": rows, "additions": merged["additions"],
        "already": merged["already"], "conflicts": merged["conflicts"],
        "ambiguous": extracted["ambiguous"], "invalid_occurrences": extracted["invalid_occurrences"],
        "invalid_records": extracted["invalid_records"], "total": merged["total"],
    }
    # Snapshot only the vocabulary; never copy keys or other private settings.
    report.parent.mkdir(parents=True, exist_ok=True)
    if report.exists() and any(json.loads(report.read_text(encoding="utf-8")).get(key)
                               for key in ("applied", "apply_pending")):
        raise ValueError("请使用新的报告文件名，避免覆盖已应用导入的恢复清单")
    atomic_json(report, manifest)
    if apply and merged["additions"]:
        if config_path.read_bytes() != config_bytes:
            raise ValueError("配置已变更，请重新预览合并")
        # Persist the recovery journal before writing the external setting. If
        # the final report write fails, restore can still use this pending journal.
        manifest["apply_pending"] = True
        atomic_json(report, manifest)
        target[key] = merged["rows"]
        try:
            atomic_json(config_path, config, expected_bytes=config_bytes)
        except ValueError:
            manifest["apply_pending"] = False
            atomic_json(report, manifest)
            raise
        manifest["applied"] = True
        manifest["apply_pending"] = False
        atomic_json(report, manifest)
    return manifest


def restore_import(config_path: Path, manifest: dict) -> int:
    """Remove only unchanged imported aliases; preserve later user edits."""
    if not (manifest.get("applied") or manifest.get("apply_pending")):
        raise ValueError("该清单没有已应用的导入")
    if Path(manifest["config_path"]).resolve() != config_path.resolve():
        raise ValueError("恢复配置路径与导入清单不一致")
    raw = config_path.read_bytes()
    if (not manifest.get("applied") and manifest.get("apply_pending")
            and hashlib.sha256(raw).hexdigest() != manifest.get("config_sha256_after")):
        raise ValueError("待应用清单无法确认配置已写入；请保留清单核对，不撤销手工词库")
    config = json.loads(raw.decode("utf-8-sig"))
    target, key = mapping_rows(config)
    kept = []
    for row in target[key]:
        parsed = existing_terms([row])
        alias, tag = next(iter(parsed.items()))
        if manifest["additions"].get(alias) != tag:
            kept.append(row)
    count = len(target[key]) - len(kept)
    if config_path.read_bytes() != raw:
        raise ValueError("配置已变更，请重新恢复")
    target[key] = kept
    atomic_json(config_path, config, expected_bytes=raw)
    return count


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", type=Path)
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--report", type=Path, required=True)
    cli.add_argument("--apply", action="store_true")
    cli.add_argument("--restore", action="store_true")
    args = cli.parse_args()
    if args.restore:
        print(json.dumps({"restored": restore_import(args.config, json.loads(args.report.read_text(encoding="utf-8")))}))
    else:
        if args.source is None:
            cli.error("--source is required for import")
        result = run_import(args.source, args.config, args.report, apply=args.apply)
        print(json.dumps({k: len(result[k]) for k in ("additions", "already", "conflicts", "ambiguous")}
                         | {"total": result["total"], "applied": result["applied"]}))
