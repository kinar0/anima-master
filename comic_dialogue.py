"""Optional, model-independent comic dialogue extraction and lettering."""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

_LABEL_RE = re.compile(
    r"(?:(?<!\S)|(?<=[：:。！？；]))(?P<label>[\u4e00-\u9fffA-Za-z0-9_]{1,16})"
    r"(?:（(?P<note>[^）]{0,60})）)?[：:]"
)
_START_LABELS = frozenset({"漫画对话", "漫画台词", "内心OS", "内心OS气泡", "旁白", "电视旁白"})
_HEADING_LABELS = frozenset({"漫画对话", "漫画台词"})
_VISUAL_SECTION_RE = re.compile(r"正向提示词[：:]")
_QUOTED_VERB_RE = re.compile(
    r"(?<![\u4e00-\u9fffA-Za-z0-9_])"
    r"(?P<speaker>[\u4e00-\u9fff]{1,8}?|[A-Z][A-Za-z0-9_]{0,15}?)"
    r"(?:（[^）]{0,60}）)?"
    r"(?P<verb>喃喃自语|低声说|轻声说|喊道|说道|回答|心想|说|问|喊|答)[：:]?"
    r"[“「](?P<text>[^”」]+)[”」]"
)
_QUOTED_COLON_RE = re.compile(
    r"(?<![\u4e00-\u9fffA-Za-z0-9_])"
    r"(?P<speaker>[\u4e00-\u9fff]{1,8}|[A-Z][A-Za-z0-9_]{0,15})"
    r"(?:（[^）]{0,60}）)?[：:]"
    r"[“「](?P<text>[^”」]+)[”」]"
)


@dataclass(frozen=True)
class ComicLine:
    kind: str
    speaker: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class ComicDialogueError(ValueError):
    """An explicit comic script could not be safely delivered."""


def extract_comic_dialogue(prompt: str) -> tuple[ComicLine, ...]:
    """Copy labelled or clearly attributed quoted speech without rewriting."""
    source = str(prompt or "")
    visual_start = _VISUAL_SECTION_RE.search(source)
    script = source[: visual_start.start()] if visual_start else source
    labels = list(_LABEL_RE.finditer(script))
    first = next(
        (index for index, match in enumerate(labels)
         if match.group("label") in _START_LABELS),
        None,
    )
    if first is None:
        quoted: list[ComicLine] = []
        verb_matches = list(_QUOTED_VERB_RE.finditer(script))
        colon_matches = [
            match for match in _QUOTED_COLON_RE.finditer(script)
            if not any(
                match.start() < verb.end() and match.end() > verb.start()
                for verb in verb_matches
            )
        ]
        for match in sorted((*verb_matches, *colon_matches), key=lambda item: item.start()):
            speaker = match.group("speaker")
            kind = "thought" if match.re is _QUOTED_VERB_RE and match.group("verb") == "心想" else "speech"
            if speaker in {"旁白", "电视旁白"}:
                kind = "narration"
            quoted.append(
                ComicLine(kind=kind, speaker=speaker, text=match.group("text"))
            )
        return tuple(quoted)
    lines: list[ComicLine] = []
    for index in range(first, len(labels)):
        match = labels[index]
        label = match.group("label")
        if label in {"剧情原文", "分镜", "负向提示词"}:
            break
        end = labels[index + 1].start() if index + 1 < len(labels) else len(script)
        content = script[match.end() : end].strip()
        if not content or (label in _HEADING_LABELS and not match.group("note")):
            continue
        note = str(match.group("note") or "").strip()
        if label.startswith("内心OS"):
            kind = "thought"
            speaker = note
        elif "旁白" in label:
            kind = "narration"
            speaker = label if label != "旁白" else ""
        elif label in _HEADING_LABELS:
            kind = "speech"
            # A parenthetical stage direction is not necessarily a speaker name.
            speaker = note if len(note) <= 6 else ""
        else:
            kind = "speech"
            speaker = label
        # Every recovered line must be a literal span of the user's request.
        if content not in source:
            raise ComicDialogueError("dialogue span is not in the original request")
        lines.append(ComicLine(kind=kind, speaker=speaker, text=content))
    if not lines:
        raise ComicDialogueError("comic dialogue marker has no recoverable text")
    return tuple(lines)


def _font_path() -> Path:
    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ComicDialogueError("Chinese font is unavailable for comic lettering")


def _wrap_text(draw, text: str, font, max_width: int) -> list[str]:
    """Wrap by rendered width while preserving every visible source character."""
    wrapped: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for char in paragraph:
            candidate = current + char
            if current and draw.textlength(candidate, font=font) > max_width:
                wrapped.append(current)
                current = char
            else:
                current = candidate
        wrapped.append(current)
    return wrapped


def _bubble_layout(draw, lines, body_font, label_size, line_height, padding, width):
    layout = []
    for line in lines:
        wrapped = _wrap_text(draw, line.text, body_font, width - 2 * padding)
        bubble_height = 2 * padding + len(wrapped) * line_height
        if line.speaker:
            bubble_height += label_size + 8
        layout.append((line, wrapped, bubble_height))
    return layout


def _draw_bubble(
    draw, item, *, x, y, width, padding, body_font, label_font, label_size,
    line_height, stroke,
):
    line, wrapped, bubble_height = item
    fill = "#ffffff" if line.kind == "speech" else "#eef3f8"
    draw.rounded_rectangle(
        (x, y, x + width, y + bubble_height),
        radius=max(18, width // 20),
        fill=fill,
        outline="#242424",
        width=stroke,
    )
    text_y = y + padding
    if line.speaker:
        draw.text((x + padding, text_y), line.speaker, font=label_font, fill="#33475b")
        text_y += label_size + 8
    for row in wrapped:
        draw.text((x + padding, text_y), row, font=body_font, fill="#151515")
        text_y += line_height


def _in_frame_placements(frame, layout, bubble_width, margin, gap):
    """Find compact, low-detail regions in the generated frame."""
    from PIL import ImageFilter, ImageStat

    width, height = frame.size
    if len(layout) > 4 or sum(item[2] for item in layout) > height * 0.65:
        return None
    edges = frame.convert("L").filter(ImageFilter.FIND_EDGES)
    placements = []
    occupied = []
    for index, item in enumerate(layout):
        bubble_height = item[2]
        if bubble_height > height * 0.32:
            return None
        # In a two-panel comic, dialogue order normally follows left then right.
        x = margin if index % 2 == 0 else width - margin - bubble_width
        candidates = []
        for fraction in (0.025, 0.10, 0.18, 0.28, 0.40, 0.52):
            y = max(margin, int(height * fraction))
            if y + bubble_height > height * 0.70:
                continue
            box = (x, y, x + bubble_width, y + bubble_height)
            if any(
                x < previous[2] + gap
                and x + bubble_width + gap > previous[0]
                and y < previous[3] + gap
                and y + bubble_height + gap > previous[1]
                for previous in occupied
            ):
                continue
            detail = ImageStat.Stat(edges.crop(box)).mean[0]
            candidates.append((detail + fraction * 8, box))
        if not candidates:
            return None
        _, box = min(candidates, key=lambda item: item[0])
        placements.append((box[0], box[1]))
        occupied.append(box)
    return placements


def letter_comic_image(
    source_path: Path,
    lines: tuple[ComicLine, ...],
    output_dir: Path,
    *,
    task_id: str,
    index: int,
) -> Path:
    """Place exact text in the frame when possible, else add a caption area."""
    if not lines:
        raise ComicDialogueError("no comic dialogue to render")
    # Pillow remains optional for all ordinary generation paths.
    from PIL import Image, ImageDraw, ImageFont

    source_path = Path(source_path)
    with Image.open(source_path) as original:
        frame = original.convert("RGB")
    width, height = frame.size
    if width < 256 or height < 256:
        raise ComicDialogueError("generated image is too small for comic lettering")
    font_path = _font_path()
    body_size = max(22, min(38, width // 29))
    label_size = max(18, body_size - 5)
    body_font = ImageFont.truetype(str(font_path), body_size)
    label_font = ImageFont.truetype(str(font_path), label_size)
    margin = max(24, width // 24)
    padding = max(18, width // 48)
    probe = ImageDraw.Draw(frame)
    line_height = body_size + max(8, body_size // 3)
    gap = max(14, width // 70)
    bubble_width = (
        min(width - 2 * margin, int(width * 0.62))
        if len(lines) == 1
        else (width - 3 * margin) // 2
    )
    frame_layout = _bubble_layout(
        probe, lines, body_font, label_size, line_height, padding, bubble_width
    )
    placements = _in_frame_placements(frame, frame_layout, bubble_width, margin, gap)
    if placements is not None:
        page = frame.copy()
        draw = ImageDraw.Draw(page)
        for item, (x, y) in zip(frame_layout, placements):
            _draw_bubble(
                draw, item, x=x, y=y, width=bubble_width, padding=padding,
                body_font=body_font, label_font=label_font, label_size=label_size,
                line_height=line_height, stroke=max(2, width // 400),
            )
    else:
        full_width = width - 2 * margin
        layout = _bubble_layout(
            probe, lines, body_font, label_size, line_height, padding, full_width
        )
        caption_height = margin * 2 + sum(item[2] for item in layout) + gap * (len(layout) - 1)
        if caption_height > height * 3:
            raise ComicDialogueError("comic dialogue exceeds the supported page height")
        page = Image.new("RGB", (width, height + caption_height), "#f7f5f0")
        page.paste(frame, (0, caption_height))
        draw = ImageDraw.Draw(page)
        y = margin
        for item in layout:
            _draw_bubble(
                draw, item, x=margin, y=y, width=full_width, padding=padding,
                body_font=body_font, label_font=label_font, label_size=label_size,
                line_height=line_height, stroke=max(2, width // 400),
            )
            y += item[2] + gap
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_task_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(task_id))[:64] or "comic"
    destination = output_dir / f"{safe_task_id}_{index}_comic.png"
    temporary = output_dir / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        page.save(temporary, format="PNG")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
