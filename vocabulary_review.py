"""Local vocabulary maintenance catalog. Never changes generation mappings."""
from __future__ import annotations

from copy import deepcopy
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import threading
from typing import Any

STATES = {"pending", "keep_current", "proposed", "modified", "deferred", "rejected"}


def _key(value: str) -> str:
    return " ".join(value.strip().casefold().split())


class VocabularyReview:
    """Paginated, searchable evidence and revision-checked review notes."""

    def __init__(self, catalog_path: Path, decisions_path: Path):
        self.catalog_path = catalog_path
        self.decisions_path = decisions_path
        self._catalog_signature = None
        self._catalog: dict[str, Any] = {"records": [], "source": {}}
        self._catalog_digest = ""
        self._lock = threading.RLock()

    def _read_catalog(self) -> dict[str, Any]:
        signature = (self.catalog_path.stat().st_mtime_ns, self.catalog_path.stat().st_size)
        if signature != self._catalog_signature:
            raw = self.catalog_path.read_bytes()
            value = json.loads(raw)
            if not isinstance(value, dict) or not isinstance(value.get("records"), list):
                raise ValueError("候选资料格式无效")
            self._catalog = value
            self._catalog_digest = hashlib.sha256(raw).hexdigest()
            self._catalog_signature = signature
        return self._catalog

    def _notes(self) -> tuple[dict[str, Any], str]:
        raw = self.decisions_path.read_bytes() if self.decisions_path.exists() else b"{}"
        notes = json.loads(raw)
        if not isinstance(notes, dict):
            raise ValueError("人工审查记录格式无效")
        revision = hashlib.sha256(self._catalog_digest.encode()+raw).hexdigest()
        return notes, revision

    def snapshot(self, *, existing: dict[str, str], query: str = "", state: str = "all",
                 page: int = 1, page_size: int = 30) -> dict[str, Any]:
        with self._lock:
            catalog = self._read_catalog()
            notes, revision = self._notes()
            if state not in STATES | {"all", "issues", "focus"}:
                raise ValueError("审查筛选无效")
            page_size = min(100, max(1, int(page_size)))
            query = _key(str(query)[:160])
            records = []
            # Search the maintenance evidence only; similarity is never a
            # generation trigger, nor a semantic confidence score.
            for record in catalog["records"]:
                confirmed = record.get("confirmed_context", [])
                default = {"state": "keep_current" if confirmed else "pending", "aliases": [],
                           "note": "；".join(i["reason"] for i in confirmed)}
                decision = notes.get(record["tag"], default)
                if state in STATES and decision["state"] != state:
                    continue
                if state == "issues" and not record.get("issues"):
                    continue
                if state == "focus" and not record.get("focus"):
                    continue
                terms = [record["tag"], record.get("definition", ""), *record.get("categories", []),
                         *(a["alias"] for a in record["aliases"]), *decision.get("aliases", [])]
                normalized = [_key(t) for t in terms]
                score = 0
                reason = ""
                if query:
                    if query in normalized:
                        score, reason = 3, "名称或别名完全匹配"
                    elif any(query in t for t in normalized):
                        score, reason = 2, "名称、别名或说明包含搜索词"
                    elif len(query) >= 3:
                        score = max((SequenceMatcher(None, query, t).ratio() for t in normalized if abs(len(t)-len(query)) <= 3), default=0)
                        if score < .72:
                            continue
                        reason = "字面相近候选，请核对含义"
                    else:
                        continue
                records.append((score, record, decision, reason))
            if query:
                records.sort(key=lambda x: (-x[0], not x[1].get("focus"), x[1]["tag"]))
            total = len(records)
            pages = max(1, (total+page_size-1)//page_size)
            page = max(1, min(int(page), pages))
            output = []
            for _, record, decision, reason in records[(page-1)*page_size:page*page_size]:
                item = deepcopy(record)
                item["decision"] = deepcopy(decision)
                item["search_reason"] = reason
                item["current_aliases"] = [a for a, tag in existing.items() if tag == record["tag"]]
                for alias in item["aliases"]:
                    current = existing.get(_key(alias["alias"]))
                    alias["current_tag"] = current
                    alias["mapping_state"] = "existing" if current == record["tag"] else "conflict" if current else "candidate"
                output.append(item)
            return {"revision": revision, "source": catalog.get("source", {}), "rows": output,
                    "total": total, "catalog_total": len(catalog["records"]), "page": page,
                    "page_size": page_size, "pages": pages, "applied": False}

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._read_catalog()
            notes, revision = self._notes()
            if not isinstance(payload, dict) or payload.get("base_revision") != revision:
                raise ValueError("审查记录已变化，请刷新后重试")
            tag = payload.get("tag")
            state = payload.get("state")
            if tag not in {r["tag"] for r in self._catalog["records"]} or state not in STATES:
                raise ValueError("标签或审查状态无效")
            aliases = payload.get("aliases", [])
            note = payload.get("note", "")
            if (not isinstance(aliases, list) or len(aliases) > 30
                    or any(not isinstance(a, str) or not 2 <= len(a.strip()) <= 80
                           or any(c in a for c in "=|\r\n\t") for a in aliases)
                    or not isinstance(note, str) or len(note) > 2000):
                raise ValueError("别名或备注格式无效")
            aliases = list(dict.fromkeys(a.strip() for a in aliases))
            if state in {"proposed", "modified"} and not aliases:
                raise ValueError("请先选择或填写待验收别名")
            notes[tag] = {"state": state, "aliases": aliases, "note": note.strip()}
            self.decisions_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.decisions_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(notes, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            temporary.replace(self.decisions_path)
            _, new_revision = self._notes()
            return {"revision": new_revision, "decision": notes[tag], "applied": False,
                    "message": "审查记录已保存；未写入生图词库，采纳项仍需验收"}
