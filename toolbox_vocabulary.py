"""Bounded, local Toolbox vocabulary evidence. No model/network/wardrobe authority.

Scoring is adapted from llm-nai-toolbox 0.9.8 (lyf, MIT), whose search.ts
attributes koishi-plugin-reforge. See third_party/llm_nai_toolbox/NOTICE.md.
Data is read from the configured local copy, never imported into term mappings.
"""
from __future__ import annotations

import asyncio
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cmp_to_key
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import Any

try:
    from .term_match_context import TermMatchContext
    from .toolbox_overrides import OVERLAY_FILE, read_overlay
except ImportError:
    from term_match_context import TermMatchContext
    from toolbox_overrides import OVERLAY_FILE, read_overlay

_PAIRS = (
    "戦战絵绘桜樱剣剑沢泽様样竜龙鉄铁気气焼烧転转悪恶霊灵氷冰浜滨撃击歩步弾弹糸丝戯戏辺边隠隐"
    "対对広广変变図图薬药雑杂壊坏歳岁総总弁辩栄荣脳脑縁缘麺面覚觉豊丰軽轻芸艺涙泪蛍萤帰归駅驿"
    "継继駆驱錬炼黙默砕碎専专拡扩齢龄権权譲让択择歴历摂摄猟猎営営挙举賛赞悩恼訳译従从厳严聴听拠据庁厅顕显吒咤乾干"
)
_FOLD = str.maketrans(dict(zip(_PAIRS[::2], _PAIRS[1::2])))
_NORMAL = re.compile(r"[\s_\-()（）【】\[\]·・、。，]")
_TAG = re.compile(r"[a-z0-9][a-z0-9_():!'.\-/+]*\Z", re.I)
_FIELDS = ("zh", "zhFull", "zhShort", "zhNick", "ja", "en", "other")
_FILES = ("tags_index_v2.json", "tag_gloss.json", "tag_browse.json", "tag_deprecated.json")


def normalize(text: str) -> str:
    return _NORMAL.sub("", text.lower()).translate(_FOLD)


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" or "\u3040" <= c <= "\u30ff" for c in text)


def _units(text: str) -> str:
    """JS string length/substring operate on UTF-16 units, unlike Python."""
    raw = text.encode("utf-16-le", errors="surrogatepass")
    return "".join(chr(raw[i] | raw[i + 1] << 8) for i in range(0, len(raw), 2))


def _grams(text: str) -> set[str]:
    s = _units(text)
    if len(s) < 2:
        return {s} if s else set()
    sizes = (2, 3) if has_cjk(text) or len(s) < 3 else (3,)
    return {s[i:i+n] for n in sizes for i in range(len(s)-n+1)}


def _edit(a: str, b: str) -> int:
    row = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        new = [i]
        for j, cb in enumerate(b, 1):
            new.append(min(new[-1]+1, row[j]+1, row[j-1]+(ca != cb)))
        row = new
    return row[-1]


def similarity(query: str, target: str) -> float:
    nq, nt = normalize(query), normalize(target)
    if not nq or not nt:
        return 0.0
    if nq == nt:
        return 1.0
    uq, ut = _units(nq), _units(nt)
    if nt.find(nq) >= 0:
        cover = len(uq)/len(ut)
        if len(uq) >= (2 if has_cjk(nq) else 3) and cover >= .15:
            return .9 + .1*cover
    elif nq.find(nt) >= 0:
        return .6 + .3*len(ut)/len(uq)
    max_len = max(len(uq), len(ut))
    cjk = has_cjk(nq) or has_cjk(nt)
    if not cjk and abs(len(uq)-len(ut)) <= 2 and max_len >= 3:
        distance = _edit(uq, ut)
        score = 1.0 - distance/max_len*.4
        if distance <= min(2, math.ceil(max_len*.3)) and score > .65:
            return score
    bq = {uq[i:i+2] for i in range(len(uq)-1)}
    bt = {ut[i:i+2] for i in range(len(ut)-1)}
    bigram = len(bq & bt)/len(bq | bt) if bq and bt else 0
    if cjk:
        cq, ct = set(nq), set(nt)
        chars = len(cq & ct)/len(cq | ct)
        return min(.84, (chars*.7 + bigram*.3)*1.15)
    if bq and bt:
        return bigram*.65
    cq, ct = set(nq), set(nt)
    return len(cq & ct)/len(cq | ct)*.5


def match_entry(query: str, entry: dict[str, Any]) -> tuple[float, str]:
    best, field = similarity(query, entry["tag"]), "tag"
    structured = any(entry.get(k) for k in ("zhFull", "zhShort", "zhNick"))
    for key in _FIELDS:
        for name in entry.get(key, ()):
            if structured and key == "zh" and any(name in entry.get(k, ()) for k in ("zhFull", "zhShort", "zhNick")):
                continue
            score = similarity(query, name)
            if structured and has_cjk(name):
                length = len(_units(name))
                if key == "zhFull":
                    score = max(0, score-.03) if length == 1 else min(1, score+(.01 if length == 2 else .02))
                elif key == "zhShort":
                    score = max(0, score-({1:.15, 2:.05, 3:.02}.get(length, 0)))
                elif key == "zhNick":
                    score = max(0, score-.01)
            if score > best:
                best, field = score, key
    if field == "tag" and 0 < best < 1:
        best = min(1, best+.03)
    return best, field


def _strings(value: Any) -> tuple[str, ...]:
    return tuple(x for x in value if isinstance(x, str) and x) if isinstance(value, list) else ()


def _clean(value: str) -> str:
    # Data is a reference, never a template or an executable instruction.
    return re.sub(r"[\r\n\t<>]", " ", value).strip()


def _general_rows(path: Path) -> list[Any]:
    """Decode top-level categories one at a time; don't retain artist/identity data.

    Use the standard JSON decoder for keys and values, including escaped strings;
    this is not a regex extraction of a nested JSON array.
    """
    source = path.read_text(encoding="utf-8-sig")
    decoder = json.JSONDecoder()
    def space(pos: int) -> int:
        while pos < len(source) and source[pos] in " \t\r\n":
            pos += 1
        return pos
    pos = space(0)
    if source[pos:pos+1] != "{":
        raise ValueError("invalid_general_index")
    pos = space(pos+1)
    general = None
    while source[pos:pos+1] != "}":
        key,pos = decoder.raw_decode(source,pos)
        pos = space(pos)
        if not isinstance(key,str) or source[pos:pos+1] != ":":
            raise ValueError("invalid_general_index")
        value,pos = decoder.raw_decode(source,space(pos+1))
        if key == "general":
            general = value
        del value
        pos = space(pos)
        if source[pos:pos+1] == "}":
            break
        if source[pos:pos+1] != ",":
            raise ValueError("invalid_general_index")
        pos = space(pos+1)
        if source[pos:pos+1] == "}":
            raise ValueError("invalid_general_index")
    if space(pos+1) != len(source) or not isinstance(general,list):
        raise ValueError("invalid_general_index")
    return general


@dataclass(frozen=True)
class VocabularyPacket:
    rows: str = ""
    diagnostics: dict[str, Any] | None = None


class ToolboxVocabulary:
    """One lazy snapshot per pipeline; load/search happens in a worker thread."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._signature: tuple[Any, ...] | None = None
        self._data: dict[str, Any] = {}
        self._failure: str = ""

    def _load(self, directory: Path) -> dict[str, Any]:
        signatures = []
        for name in (*_FILES, OVERLAY_FILE):
            p = directory/name
            try:
                st = p.stat()
                signatures.append((name, st.st_mtime_ns, st.st_size))
            except FileNotFoundError:
                signatures.append((name, None, None))
        signature = (str(directory.resolve()), *signatures)
        if signature == self._signature:
            if self._failure:
                raise ValueError(self._failure)
            return self._data
        self._signature, self._data, self._failure = signature, {}, ""
        started = time.perf_counter()
        try:
            raw = _general_rows(directory/_FILES[0])
            entries, by_tag, index = [], {}, defaultdict(lambda: array("I"))
            for row in raw:
                if not isinstance(row, dict) or not isinstance(row.get("tag"), str) or not _TAG.fullmatch(row["tag"]):
                    continue
                tag = row["tag"].lower()
                if tag in by_tag:
                    continue
                count = row.get("count", 0)
                count = max(0, count) if isinstance(count, (int, float)) and math.isfinite(count) else 0
                entry = {"tag":tag, "count":count, **{k:_strings(row.get(k)) for k in _FIELDS}}
                entries.append(entry)
                by_tag[tag] = entry
            gloss, categories, deprecated = {}, defaultdict(list), {}
            for name in _FILES[1:]:
                path = directory/name
                if not path.exists():
                    continue
                value = json.loads(path.read_text(encoding="utf-8-sig"))
                if not isinstance(value, dict):
                    raise ValueError("invalid_optional_dictionary")
                if name == "tag_gloss.json":
                    for tag, row in value.items():
                        if tag in by_tag and isinstance(row, dict) and isinstance(row.get("g"), str):
                            gloss[tag] = {"g":_clean(row["g"]), "trap":_clean(row.get("trap", "")) if isinstance(row.get("trap"), str) else "",
                                          "vs":tuple((x[0], _clean(x[1])) for x in row.get("vs", []) if isinstance(x, list) and len(x)>1 and isinstance(x[0], str) and isinstance(x[1], str)) if isinstance(row.get("vs"), list) else ()}
                elif name == "tag_browse.json":
                    for category, rows in value.items():
                        if category == "_toc" or not isinstance(rows, list):
                            continue
                        for row in rows:
                            if isinstance(row, dict) and row.get("t") in by_tag:
                                tag = row["t"]
                                categories[tag].append(category)
                                if tag not in gloss and isinstance(row.get("g"), str):
                                    gloss[tag] = {"g":_clean(row["g"]), "trap":"", "vs":()}
                else:
                    deprecated = {t:_clean(v) for t,v in value.items() if t in by_tag and isinstance(v, str)}
            overrides, _ = read_overlay(directory)
            # Base evidence remains available to the editor for inspection/restore.
            base = {e['tag']: e for e in entries}
            base_gloss = dict(gloss)
            base_categories = {t:list(c) for t,c in categories.items()}
            effective, disabled = [], set()
            for tag in dict.fromkeys((*base, *overrides)):
                entry = dict(base.get(tag, {"tag":tag, "count":0, **{k:() for k in _FIELDS}}))
                edit = overrides.get(tag)
                if edit is not None:
                    # Edited aliases replace the Chinese trigger set, including
                    # structured upstream aliases. Other languages stay intact.
                    base_aliases = {s for k in ('zhFull', 'zhShort', 'zhNick', 'zh') for s in entry[k]}
                    if set(edit['aliases']) != base_aliases:
                        entry.update(zh=tuple(edit['aliases']), zhFull=(), zhShort=(), zhNick=())
                    gloss[tag] = {'g':edit['definition'], 'trap':edit['trap'],
                                  'vs':tuple(tuple(x) for x in edit['contrasts'])}
                    categories[tag] = list(edit['categories'])
                    if edit['disabled']:
                        disabled.add(tag)
                effective.append(entry)
            entries = [e for e in effective if e['tag'] not in disabled]
            by_tag = {e['tag']:e for e in entries}
            for number, entry in enumerate(entries):
                grams = set()
                for name in (entry['tag'], *(s for k in _FIELDS for s in entry[k])):
                    grams.update(_grams(normalize(name)))
                for gram in grams:
                    index[gram].append(number)
            self._data = {"entries":entries, "by_tag":by_tag, "index":dict(index), "gloss":gloss,
                          "categories":dict(categories), "deprecated":deprecated,
                          "catalog":effective, "base":base, "base_gloss":base_gloss,
                          "base_categories":base_categories, "overrides":overrides,
                          "disabled":disabled,
                          "load_ms":round((time.perf_counter()-started)*1000, 2)}
            return self._data
        except Exception as exc:
            # Cache failure by signature too; do not repeatedly parse a broken 29MB file.
            self._failure = type(exc).__name__
            raise

    @staticmethod
    def _search(data: dict[str, Any], query: str, limit: int = 2) -> tuple[list[dict[str, Any]], bool]:
        nq = normalize(query)
        if not nq:
            return [], False
        counts: Counter[int] = Counter()
        grams = _grams(nq)
        for gram in grams:
            counts.update(data["index"].get(gram, ()))
        threshold = max(1, math.ceil(len(grams)*.2))
        indices = [i for i,n in counts.items() if n >= threshold]
        clipped = len(indices)>1000
        # Request-time bound is an intentional adaptation of upstream searchOne.
        indices.sort(key=lambda i:(-counts[i], -data["entries"][i]["count"], i))
        scored = []
        for i in indices[:1000]:
            entry = data["entries"][i]
            score, field = match_entry(query, entry)
            if score >= .5:
                scored.append({"tag":entry["tag"], "score":math.floor(score*100+.5)/100,
                               "field":field, "count":entry["count"]})
        length = len(_units(nq))
        weight = .05 if length <= 2 else .03
        gap = .10 + max(0, 3-length)*.03
        def compare(a: dict[str, Any], b: dict[str, Any]) -> int:
            diff = b["score"]-a["score"]
            if abs(diff) < gap:
                diff += (math.log10(b["count"]+1)-math.log10(a["count"]+1))*weight
            return (diff>0)-(diff<0)
        scored.sort(key=cmp_to_key(compare))
        strong = [s for s in scored if s["score"]>=.85]
        return (strong[:limit] if strong else scored[:1]), clipped

    def search(self, directory: str | Path, query: str, limit: int = 2) -> list[dict[str, Any]]:
        """Local inspection API. It grants no request or identity authority."""
        with self._lock:
            rows, _ = self._search(self._load(Path(directory)), query, max(1, min(limit, 8)))
            return rows

    def browse(self, directory: str | Path, category: str, keyword: str = "", limit: int = 8) -> list[dict[str, str]]:
        """Bounded category inspection; keyword ranks but never erases the category.

        This API does not inject a category directory or full pages into a writer.
        The wiki/alias search field `m` is deliberately not copied or rendered.
        """
        with self._lock:
            data = self._load(Path(directory))
            tags = [tag for tag,cats in data["categories"].items() if category in cats and tag in data['by_tag']]
            q = keyword.casefold().strip()
            tags.sort(key=lambda tag:(
                not (q and q in (tag+" "+data["gloss"].get(tag,{}).get("g","")).casefold()),
                -data["by_tag"][tag]["count"], tag,
            ))
            return [{"tag":t,"definition":data["gloss"].get(t,{}).get("g","")}
                    for t in tags[:max(0,min(16,limit))]]

    async def prepare(self, *, directory: str, mode: str, text: str,
                      mappings: tuple[tuple[str,str], ...] = (), phrases: tuple[str,...] = (),
                      max_chars: int = 1200) -> VocabularyPacket:
        if mode not in {"observe", "reference"}:
            return VocabularyPacket(diagnostics={"mode":"off"})
        if not directory.strip():
            return VocabularyPacket(diagnostics={"mode":mode, "state":"unconfigured"})
        try:
            return await asyncio.to_thread(self._prepare, directory, mode, text, mappings, phrases, max_chars)
        except Exception as exc:
            return VocabularyPacket(diagnostics={"mode":mode,"state":"unavailable","error_type":type(exc).__name__})

    def _prepare(self, directory: str, mode: str, text: str, mappings: tuple[tuple[str,str], ...],
                 phrases: tuple[str,...], max_chars: int) -> VocabularyPacket:
        started = time.perf_counter()
        with self._lock:
            try:
                data = self._load(Path(directory).expanduser())
            except Exception as exc:
                return VocabularyPacket(diagnostics={"mode":mode, "state":"unavailable", "error_type":type(exc).__name__})
            context = TermMatchContext(text)
            queries: list[tuple[str, int, int, str | None]] = []
            def add(phrase: str, canonical: str | None = None) -> None:
                if not isinstance(phrase, str) or not 2 <= len(phrase.strip()) <= 48:
                    return
                phrase = phrase.strip()
                for m in re.finditer(re.escape(phrase), text, re.I):
                    a,b = m.span()
                    # Conservative: uncertain mixed negative/literal clauses stay with the writer.
                    if context.suppressed(a,b) or any(a<y and x<b for x,y in context.blocked):
                        continue
                    if not has_cjk(phrase) and ((a and text[a-1].isalnum()) or (b<len(text) and text[b].isalnum())):
                        continue
                    item = (phrase, a,b, canonical)
                    if item not in queries:
                        queries.append(item)
                    break
            for alias, tag in mappings:
                add(alias, tag)
            for phrase in phrases:
                add(phrase)
            for m in re.finditer(r"[^，,。；;！!？?\n：:]+", text[:2000]):
                add(m.group().strip())
            # Longest overlapping fragments win; a second independent occurrence is preserved.
            selected = []
            for item in queries:
                if item[3] is not None or not any(item[1]>=q[1] and item[2]<=q[2] for q in selected):
                    selected.append(item)
            rows, seen_tags, details = [], set(), []
            mapped_tags = {tag for _alias,tag in mappings}
            bounded_chars = max(0, min(2400, int(max_chars)))
            # Reserve > the actual normal/NAI wrapper cost when there are no existing mappings.
            row_budget = max(0, bounded_chars-600)
            clipped_searches = 0
            weak_without_literal = 0
            for query,a,b,canonical in selected[:8]:
                if canonical is not None:
                    candidates = ([{"tag":canonical,"score":1.0,"field":"configured_mapping"}] if canonical in data["by_tag"] else [])
                else:
                    candidates, clipped = self._search(data, query)
                    clipped_searches += clipped
                for candidate in candidates:
                    tag = candidate["tag"]
                    if tag in seen_tags or (canonical is None and tag in mapped_tags):
                        continue
                    entry = data["by_tag"][tag]
                    mentions = []
                    for name in (tag, *(s for k in _FIELDS for s in entry[k])):
                        for match in re.finditer(re.escape(name).replace("_", "[ _]"), text[a:b], re.I):
                            start,end = a+match.start(), a+match.end()
                            if not has_cjk(name) and ((start and text[start-1].isalnum()) or (end<len(text) and text[end].isalnum())):
                                continue
                            mentions.append((a+match.start(), a+match.end()))
                    if mentions:
                        # An absence concept can itself start with 'no/没穿'.
                        # Only an outside negation/literal context suppresses it.
                        if all(context.suppressed(start,end) for start,end in mentions):
                            continue
                    elif any(a<y and x<b for x,y in context.negative):
                        continue
                    elif candidate["score"]<.85:
                        # Upstream's one weak lexical winner is not semantic
                        # evidence. Without a literal local name, abstain instead
                        # of suggesting one meaning for an unseen paraphrase.
                        weak_without_literal += 1
                        continue
                    g = data["gloss"].get(tag, {})
                    meaning = g.get("g", "")
                    # Known manual mappings need explanations only for an actual trap/contrast.
                    if canonical and not g.get("trap") and not g.get("vs") and tag not in data["deprecated"]:
                        continue
                    if not meaning or len(meaning)>100 or candidate["score"]<.72:
                        continue
                    detail = f"- {_clean(query)} => {tag} — {meaning}"
                    if canonical:
                        detail = f"  {tag} 释义：{meaning}"
                    if g.get("trap"):
                        if len(g["trap"])>100:
                            continue
                        detail += f"；易误读：{g['trap']}"
                    if g.get("vs"):
                        other, contrast = g["vs"][0]
                        if len(contrast)>100 or not _TAG.fullmatch(other):
                            continue
                        detail += f"；与 {other} 区别：{contrast}"
                    if tag in data["deprecated"]:
                        reason = data["deprecated"][tag]
                        if len(reason)>100:
                            continue
                        detail += f"；已废弃：{reason}"
                    if len("\n".join((*rows, detail)))+1>row_budget:
                        continue
                    rows.append(detail)
                    seen_tags.add(tag)
                    details.append({"tag":tag,"score":candidate["score"],"field":candidate["field"]})
            rendered = "\n".join(rows)
            diagnostics = {"mode":mode,"state":"ready","queries":min(8,len(selected)),
                           "entries":len(data["entries"]),"candidates":details,"rows_chars":len(rendered),
                           "added_chars_upper_bound":len(rendered)+600 if rendered else 0,
                           "max_chars":bounded_chars,"load_ms":data["load_ms"],
                           "elapsed_ms":round((time.perf_counter()-started)*1000,2),
                           "clipped_searches":clipped_searches,
                           "weak_without_literal":weak_without_literal}
            return VocabularyPacket(rows=rendered if mode=="reference" else "", diagnostics=diagnostics)
