"""Validated local edits; the original Toolbox files remain immutable."""
from __future__ import annotations

import json
from pathlib import Path
import re

OVERLAY_FILE = "local_overrides.json"
TAG = re.compile(r"[a-z0-9][a-z0-9_():!'.\-/+]*\Z")


def _text(value, limit):
    if (not isinstance(value, str) or len(value) > limit
            or any(ord(c) < 32 for c in value)):
        raise ValueError("文字格式不正确或超过长度限制")
    return value.strip()


def validate_entry(value):
    if not isinstance(value, dict) or set(value) - {
        "aliases", "definition", "trap", "contrasts", "categories", "disabled"
    }:
        raise ValueError("词条字段无效")
    result = {}
    for key, limit in (("aliases", 80), ("categories", 100)):
        rows = value.get(key, [])
        if not isinstance(rows, list) or len(rows) > 60:
            raise ValueError("别名或分类最多 60 项")
        result[key] = list(dict.fromkeys(_text(x, limit) for x in rows if x != ""))
        if any(not x for x in result[key]):
            raise ValueError("别名和分类不能只含空白")
    result["definition"] = _text(value.get("definition", ""), 100)
    result["trap"] = _text(value.get("trap", ""), 100)
    contrasts = value.get("contrasts", [])
    if not isinstance(contrasts, list) or len(contrasts) > 8:
        raise ValueError("易混辨析最多 8 项")
    result["contrasts"] = []
    for row in contrasts:
        if (not isinstance(row, list) or len(row) != 2
                or not isinstance(row[0], str)):
            raise ValueError("易混词需要名称和区别说明")
        name = _text(row[0], 100)
        if not name:
            raise ValueError("易混词名称不能为空")
        # Upstream uses both canonical tags and readable comparison names.
        # Keep those labels intact; generation retains its own tag validator.
        result["contrasts"].append([name, _text(row[1], 100)])
    if type(value.get("disabled", False)) is not bool:
        raise ValueError("停用状态无效")
    result["disabled"] = value.get("disabled", False)
    return result


def read_overlay(directory: Path):
    path = directory / OVERLAY_FILE
    raw = path.read_bytes() if path.exists() else b'{"version":1,"entries":{}}'
    if len(raw) > 4_000_000:
        raise ValueError("本地维护记录过大")
    data = json.loads(raw)
    if (not isinstance(data, dict) or data.get("version") != 1
            or not isinstance(data.get("entries"), dict) or len(data["entries"]) > 10000):
        raise ValueError("本地维护记录格式无效")
    entries = {}
    for tag, row in data["entries"].items():
        if not isinstance(tag, str) or len(tag) > 160 or not TAG.fullmatch(tag):
            raise ValueError("本地维护 tag 无效")
        entries[tag] = validate_entry(row)
    return entries, raw
