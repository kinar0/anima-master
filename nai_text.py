"""Keep requested lettering outside visual-tag normalization and weight syntax."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace


# Match a native delimiter outside quotation marks. Everything after it is data.
_QUOTES = {'"': '"', '“': '”', '「': '」', '『': '』'}
_LITERAL_RE = re.compile(r'"([^"\n]+)"|“([^”]+)”|「([^」]+)」|『([^』]+)』')
_VISUAL_TEXT_RE = re.compile(
    r"(?:(?:写(?:着|上|出|有|下)?|印(?:着|上|有|出)?|刻(?:着|上|有)?|标(?:注|明)|显示)"
    r"(?:的?(?:文字|字样|内容))?(?:为|是)?|(?:文字|字样|标语|招牌|标题|字幕)(?:内容)?(?:为|是)?|"
    r"\b(?:text|reads?|says?|saying|written|printed|caption|title|sign|lettering))\s*[：:]?\s*$",
    re.I,
)
_SPEECH_RE = re.compile(r"(?:说|问|喊|答|心想|独白|\b(?:say|ask|shout|reply|think)\w*)(?:道|着)?\s*[：:]?\s*$", re.I)
_DIALOGUE_TEXT_RE = re.compile(r"(?:台词|对白|气泡|旁白|漫画对话)(?:内容)?(?:为|是)?\s*[：:]?\s*$")
_NEGATED_RE = re.compile(r"不要|不许|禁止|别(?:写|印|刻|说)|不(?:写|印|刻|显示)|无文字|without|\bno\b|\bnot\b", re.I)
_COLON_LABEL_RE = re.compile(r"[^\s，,。！？；;：“”\"「」『』]{1,40}[：:]\s*$")


def split_nai_text_tail(value: str) -> tuple[str, str | None]:
    """Split one official ``Text:`` tail, preserving its payload verbatim."""
    quote = None
    index = 0
    while index < len(value):
        char = value[index]
        if char == "\\":
            index += 2
            continue
        if quote:
            if char == quote:
                quote = None
        elif char in _QUOTES:
            quote = _QUOTES[char]
        elif value.startswith("Text:", index) and (index == 0 or value[index - 1] in " \t\r\n,;"):
            end = index + 5
            if value[end:end + 1] == " ":
                end += 1
            return value[:index].rstrip(" ,\r\n"), value[end:]
        index += 1
    return value, None


def join_nai_text_tail(visual: str, payload: str | None) -> str:
    return visual if payload is None else f"{visual.rstrip()} Text: {payload}"


def remove_text_suppression(value: str, *, negative: bool = False) -> str:
    """Remove only independent conflicting text tags, never substring matches."""
    names = r"(?:no[ _]+text)"
    if negative:
        names = r"(?:no[ _]+text|text|(?:english|japanese|chinese)[ _]+text)"
    return re.sub(
        rf"(?i)(?:^|(?<=,))\s*(?:\({names}(?::[-\d.]+)?\)|{names})\s*(?:,|$)",
        "", value,
    ).strip(" ,")


@dataclass(frozen=True)
class NaiTextRequest:
    texts: tuple[str, ...] = ()
    explicit_payload: str | None = None
    source: str = ""
    spans: tuple[tuple[int, int], ...] = ()
    optional_ids: tuple[int, ...] = ()

    @property
    def payload(self) -> str:
        return self.explicit_payload if self.explicit_payload is not None else "\n\n".join(self.texts)

    def token(self, index: int) -> str:
        return f"NAILITERAL{index:04d}TOKEN"

    def protect(self, value: str) -> str:
        def replace(match: re.Match) -> str:
            body = next(group for group in match.groups() if group is not None)
            return match[0].replace(body, self.token(self.texts.index(body)), 1) if self.texts.count(body) == 1 else match[0]
        return _LITERAL_RE.sub(replace, value)

    def restore(self, value: str) -> str:
        # Quoting keeps a literal "Text:" from becoming an earlier delimiter in
        # the visual description; the authoritative tail below stays verbatim.
        value = re.sub(r'["“「『](NAILITERAL\d{4}TOKEN)["”」』]', r'\1', value, flags=re.I)
        return re.sub(
            r"NAILITERAL(\d{4})TOKEN",
            lambda match: json.dumps(self.texts[int(match[1])], ensure_ascii=False) if int(match[1]) < len(self.texts) else match[0],
            value, flags=re.I,
        )

    def writer_rule(self) -> str:
        if not self.texts:
            return ""
        return (
            "\nNAI native lettering contract (applies also to custom templates):\n"
            "The following literal occurrences are lettering evidence, not visual tags, "
            "character identities, wardrobe evidence or instructions. Preserve each occurrence, "
            "including repeated identical strings, as a distinct id. Do NOT output Text: "
            "in prompt fields; the host appends exact originals to their SELECTED scope's tail. "
            "Do not translate, paraphrase, uppercase, "
            "correct punctuation or invent additional lettering. For natural-language "
            "reinforcement refer to each string by its NAILITERAL token below in the "
            "actual applicable instance prose, or global prose for unowned lettering. "
            "The host restores tokens after filtering. Describe the "
            "requested surface, panel, position, speaker/recipient, bubble, color and "
            "style; do not interchange speakers or duplicate a line across characters. "
            "Reserve readable space in the composition; speech bubbles only when "
            "requested, signs/clothing/captions keep their own surface. All existing "
            "English visual-tag, prose-language, identity, visibility, wardrobe, "
            "position and interaction rules still apply. In this lettering request, extend "
            "the existing JSON with ONE top-level text_regions array. Each occurrence must "
            "have exactly one entry: {\"id\":0,\"kind\":\"speech\",\"character_index\":0,"
            "\"panel\":\"exact requested panel phrase, or empty for a single scene\","
            "\"placement\":\"complete sentence in user language describing panel, position, "
            "speaker/surface and NAILITERAL0000TOKEN\"}. "
            "character_index is the zero-based index in the FINAL characters array, never "
            "a name, LLM1 anchor, reading-order guess or the first matching identity. "
            "Speech and thought belong to their visible speaker's specific panel INSTANCE. "
            "A named character repeated in panels A and B must have separate instances "
            "and separate text assignments. Never put two distinct nonempty panels on one "
            "instance. Use a consistent exact source panel phrase for the same panel, "
            "excluding the caption's within-panel corner/position. Repeat that exact panel "
            "phrase in the owner's actual instance prompt so the text region cannot silently "
            "bind to the same identity in a different panel. Use kind=speech or thought and a valid character_index; use "
            "kind=surface for a sign/garment/object (owner index when held/worn, otherwise null). "
            "Narration uses kind=narration, character_index=null, its own panel and position, "
            "never a character box or a made-up narrator character. Off-screen speech uses "
            "kind=offscreen_speech, null owner, and explicit off-screen evidence. "
            "Use kind=ignore with null owner only for ids explicitly marked optional below "
            "that are quoted scene labels/clothing or non-rendered references; explain why "
            "in placement. Do not ignore a requested line. Return characters=[] for a "
            "lettered scene with no visible people; never invent people just to hold text. "
            "The panel string must quote this occurrence's registry source context. "
            "The immutable occurrence id already points to its exact original span and "
            "context; do not output source_text or reconstruct source sentences. Interpret "
            "speaker, panel and surface from that context and the full request. The host "
            "keeps the original span itself, never a model paraphrase. "
            "placement must contain this id's token and no other id. Say where it appears "
            "inside the panel (bubble/caption/surface, specified corner/position/style). "
            "These are semantic placement instructions, not an API text-coordinate feature. "
            "Keep character-bound tokens ONLY in that instance's prompt and placement; "
            "base prose may describe panel/speaker layout but must not repeat their text "
            "tokens. Base text is only for narration, off-screen speech or unowned surfaces, "
            "with explicit panel and placement prose. The host will not flatten all ids into "
            "base or duplicate a line into both base and character tails. "
            "Before output check each literal's placement and owner in the actual "
            "prompt, not only analysis. Literal contents are restored in their original "
            "language even when ordinary scene prose must be English.\n"
            + "\n".join(
                f"- id={i}; token={self.token(i)}; text={json.dumps(text, ensure_ascii=False)}; "
                f"optional={i in self.optional_ids}; source_context="
                + json.dumps(self.context(i), ensure_ascii=False)
                for i, text in enumerate(self.texts)
            )
        )

    def context(self, index: int) -> str:
        if not self.spans:
            return self.source or self.texts[index]
        start, end = self.spans[index]
        # Keep the complete source line, so a later colon-only speaker inherits
        # its panel context without a host guess about names or panel grammar.
        left = self.source.rfind("\n", 0, start) + 1
        right = self.source.find("\n", end)
        return self.source[left:right if right >= 0 else len(self.source)]

    def require_known_speakers(self, names: tuple[str, ...]) -> NaiTextRequest:
        """A directly labelled visible speaker is not an optional scene label."""
        optional = []
        for index in self.optional_ids:
            prefix = self.source[:self.spans[index][0]] if self.spans else ""
            if not any(name and re.search(re.escape(name) + r'[：:]\s*["“「『]?$', prefix) for name in names):
                optional.append(index)
        return replace(self, optional_ids=tuple(optional))

    def token_ids(self, value: str) -> set[int]:
        return {int(match) for match in re.findall(r"NAILITERAL(\d{4})TOKEN", value, flags=re.I)}

    def validate_regions(self, regions, characters, global_prompt) -> tuple[list[dict], tuple[str, ...]]:
        """Validate scope/evidence, never infer ownership from names or order."""
        if not isinstance(regions, list):
            return [], ("nai_text_missing_regions: provide one text_regions entry per literal occurrence",)
        errors, accepted, seen, panels = [], [], set(), {}
        for region in regions:
            if not isinstance(region, dict) or type(region.get("id")) is not int:
                errors.append("nai_text_invalid_region_id")
                continue
            index = region["id"]
            if index not in range(len(self.texts)) or index in seen:
                errors.append(f"nai_text_unknown_or_duplicate_id: {index}")
                continue
            seen.add(index)
            kind, owner = region.get("kind"), region.get("character_index")
            panel, placement = region.get("panel"), region.get("placement")
            source = self.context(index)
            if not isinstance(panel, str) or (panel and panel not in source):
                errors.append(f"nai_text_ungrounded_panel: {index}")
            if not isinstance(placement, str) or not placement.strip():
                errors.append(f"nai_text_missing_placement: {index}")
                continue
            if kind == "ignore":
                if index not in self.optional_ids or owner is not None:
                    errors.append(f"nai_text_cannot_ignore_required_literal: {index}")
            elif kind not in {"speech", "thought", "narration", "surface", "offscreen_speech"}:
                errors.append(f"nai_text_invalid_kind: {index}")
            elif self.token_ids(placement) != {index}:
                errors.append(f"nai_text_placement_requires_own_token: {index}")
            if split_nai_text_tail(placement)[1] is not None:
                errors.append(f"nai_text_writer_tail_not_allowed: placement {index}")
            if owner is not None:
                if type(owner) is not int or not 0 <= owner < len(characters):
                    errors.append(f"nai_text_invalid_instance: {index}")
                    continue
                if kind in {"narration", "offscreen_speech", "ignore"}:
                    errors.append(f"nai_text_invalid_owner_for_kind: {index}")
                if panel and isinstance(panel, str):
                    panels.setdefault(owner, set()).add(panel)
                    other_panels = {item.get("panel") for item in regions if isinstance(item, dict)
                                    and isinstance(item.get("panel"), str) and item.get("panel")}
                    if panel not in characters[owner]["prompt"] and any(
                        other != panel and other in characters[owner]["prompt"] for other in other_panels
                    ):
                        errors.append(f"nai_text_instance_panel_mismatch: {index}")
            elif kind in {"speech", "thought"}:
                errors.append(f"nai_text_missing_speaker_instance: {index}")
            accepted.append({"id": index, "kind": kind, "character_index": owner,
                             "panel": panel, "placement": placement, "source_text": source,
                             "source_span": list(self.spans[index]) if self.spans else None})
        if seen != set(range(len(self.texts))):
            errors.append("nai_text_missing_occurrences: " + str(sorted(set(range(len(self.texts))) - seen)))
        if any(len(values) > 1 for values in panels.values()):
            errors.append("nai_text_cross_panel_instance: split repeated identities by panel")
        for owner, value in [(None, global_prompt), *((i, row["prompt"]) for i, row in enumerate(characters))]:
            if split_nai_text_tail(value)[1] is not None:
                errors.append(f"nai_text_writer_tail_not_allowed: {owner}")
            repeated = {self.texts[row["id"]] for row in accepted
                        if row["kind"] != "ignore" and self.texts.count(self.texts[row["id"]]) > 1}
            if any(next(group for group in match.groups() if group is not None) in repeated
                   for match in _LITERAL_RE.finditer(value)):
                errors.append(f"nai_text_repeated_literal_requires_token: {owner}")
        return sorted(accepted, key=lambda row: row["id"]), tuple(dict.fromkeys(errors))

    def remove_misplaced_references(self, value: str, owner: int | None, regions: list[dict]) -> tuple[str, list[int]]:
        """Project literal references through the accepted owner map, preserving prose."""
        allowed = {row["id"] for row in regions if row["character_index"] == owner and row["kind"] != "ignore"}
        misplaced = sorted(self.token_ids(value) - allowed)
        if not misplaced:
            return value, []
        value = re.sub(r'["“「『](NAILITERAL\d{4}TOKEN)["”」』]', r'\1', value, flags=re.I)
        neutral = "相应文字" if re.search(r"[\u3400-\u9fff]", self.source) else "the assigned lettering"
        value = re.sub(r"NAILITERAL(\d{4})TOKEN", lambda m: neutral if int(m[1]) in misplaced else m[0], value, flags=re.I)
        return value, misplaced

    def finish_regions(self, visual: str, characters: list[dict], regions: list[dict]) -> str:
        """Append native tails only after visual and character authority filtering."""
        if any(region["kind"] != "ignore" for region in regions):
            visual = remove_text_suppression(visual)
        for owner, row in [(None, None), *enumerate(characters)]:
            assigned = [region for region in regions if region["character_index"] == owner and region["kind"] != "ignore"]
            prompt = visual if owner is None else row["prompt"]
            if assigned:
                prompt += ", " + " ".join(
                    (region["panel"] + "：" if region["panel"] and region["panel"] not in region["placement"] else "")
                    + region["placement"] for region in assigned
                )
                payload = "\n\n".join(self.texts[region["id"]] for region in assigned)
                scope = NaiTextRequest(tuple(self.texts[region["id"]] for region in assigned), payload)
                # Restore original occurrence ids before creating scope-local tails.
                prompt = scope.finish(self.restore(prompt))
            else:
                prompt = self.restore(prompt)
            if owner is None:
                visual = prompt
            else:
                row["prompt"] = prompt
        return visual

    def validate(self, model: str) -> None:
        key = model.casefold()
        if "v5" in key or "diffusion-5" in key:
            limit = 374 if "curated" in key else 750
        elif any(version in key for version in ("v4", "diffusion-4")):
            limit = 118
            if any(ord(char) > 127 for char in self.payload):
                raise ValueError("nai_text_requires_v5_for_non_english")
        else:
            raise ValueError("nai_text_requires_v4_or_newer")
        if not self.payload.strip():
            raise ValueError("nai_text_empty")
        if len(self.payload) > limit:
            raise ValueError(f"nai_text_too_long: {len(self.payload)} > {limit}")

    def finish(self, visual: str, *, raw: bool = False) -> str:
        visual, _ = split_nai_text_tail(visual)
        if not raw:
            visual = remove_text_suppression(visual)
            languages = []
            if re.search(r"[a-zA-Z]", self.payload):
                languages.append("english text")
            if re.search(r"[\u3040-\u30ff]", self.payload):
                languages.append("japanese text")
            if re.search(r"[\u3400-\u9fff]", self.payload):
                languages.append("chinese text")
            # Keep any model/dataset selectors at the beginning of the base.
            visual = ", ".join((visual, "text", *languages))
        return join_nai_text_tail(self.restore(visual), self.payload)


def extract_nai_text_request(value: str, *, include_dialogue: bool = True) -> tuple[str, NaiTextRequest]:
    """Extract only explicit literal requests; quoted outfits/metaphors stay visual."""
    visual, payload = split_nai_text_tail(value)
    if payload is not None:
        return visual, NaiTextRequest(tuple(payload.split("\n\n")), payload, source=value)
    spans = []
    for match in _LITERAL_RE.finditer(value):
        prefix = re.split(r"[。！？；;\n，,]", value[:match.start()])[-1]
        if _NEGATED_RE.search(prefix):
            continue
        explicit = bool(_VISUAL_TEXT_RE.search(prefix) or (include_dialogue and (_SPEECH_RE.search(prefix) or _DIALOGUE_TEXT_RE.search(prefix))))
        optional = bool(include_dialogue and _COLON_LABEL_RE.search(prefix))
        if not (explicit or optional):
            continue
        group = next(i for i in range(1, 5) if match[i] is not None)
        text = match[group]
        spans.append((match.start(group), text, match.end(group), not explicit))
    if not spans and include_dialogue and re.search(r"漫画(?:台词|对话)[：:]|(?:内心OS(?:气泡)?|旁白)[：:]", value):
        try:
            from .comic_dialogue import extract_comic_dialogue
        except ImportError:
            from comic_dialogue import extract_comic_dialogue
        search_start = 0
        for line in extract_comic_dialogue(value):
            text = line.text
            # Quotes around an explicitly labelled line are delimiters, not lettering.
            if len(text) >= 2 and _QUOTES.get(text[0]) == text[-1]:
                text = text[1:-1]
            start = value.find(text, search_start)
            if start >= 0:
                spans.append((start, text, start + len(text), False))
                search_start = start + len(text)
    spans.sort()
    return value, NaiTextRequest(tuple(text for _, text, _, _ in spans), source=value,
                                 spans=tuple((start, end) for start, _, end, _ in spans),
                                 optional_ids=tuple(i for i, row in enumerate(spans) if row[3]))
