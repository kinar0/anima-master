from __future__ import annotations

import re

DEFAULT_PORTRAIT = "default_portrait"
EXPLICIT_SCENE = "explicit_scene"
DEFAULT_PORTRAIT_MARKER = "background_mode_default_portrait"
EXPLICIT_SCENE_MARKER = "background_mode_explicit_scene"

_FRAMING_TAGS = {
    "ass focus",
    "bust",
    "breast focus",
    "close-up",
    "cowboy shot",
    "cropped torso",
    "eye focus",
    "face",
    "face focus",
    "feet focus",
    "foot focus",
    "hair focus",
    "hand focus",
    "head out of frame",
    "headshot",
    "leg focus",
    "lower body",
    "mouth focus",
    "navel focus",
    "portrait",
    "thigh focus",
    "upper body",
}

_CHINESE_SCENE_MARKERS = (
    "室内",
    "室外",
    "户外",
    "房间",
    "卧室",
    "客厅",
    "教室",
    "厨房",
    "浴室",
    "和室",
    "庭院",
    "街道",
    "城市",
    "海边",
    "沙滩",
    "森林",
    "公园",
    "花园",
    "车站",
    "舞台",
    "餐厅",
    "咖啡馆",
    "办公室",
    "图书馆",
    "神社",
    "寺庙",
    "城堡",
    "天空",
    "雪景",
    "雨景",
    "夜景",
)

_ENGLISH_SCENE_RE = re.compile(
    r"\b(?:indoors?|outdoors?|bedroom|living room|classroom|kitchen|bathroom|"
    r"street|city|beach|forest|park|garden|station|stage|restaurant|cafe|"
    r"office|library|shrine|temple|castle|nightscape)\b",
    flags=re.I,
)


def _positive_background_text(text: str) -> str:
    """Remove common negative background requests before intent detection."""
    positive = re.sub(
        r"(?:不要|无需|不需要|没有|无)"
        r"(?:任何|特定|具体|复杂|简单|简约|简洁|纯白色?|白色)?"
        r"(?:的)?(?:背景|场景)",
        "",
        str(text or ""),
    )
    positive = re.sub(r"(?:不要|无需|不需要|没有|无)白底", "", positive)
    return re.sub(
        r"\b(?:no|without)\s+(?:a\s+)?"
        r"(?:(?:white|simple|complex)\s+)?background\b",
        "",
        positive,
        flags=re.I,
    )


def user_requests_explicit_background(text: str) -> bool:
    """Return whether the user's own text positively requests a scene/background."""
    raw = str(text or "").strip()
    if not raw:
        return False
    positive = _positive_background_text(raw)
    if "背景" in positive or "场景" in positive:
        return True
    if any(marker in positive for marker in _CHINESE_SCENE_MARKERS):
        return True
    return bool(
        _ENGLISH_SCENE_RE.search(positive)
        or re.search(r"\b(?:in|inside|at|on)\s+(?:an?\s+|the\s+)?\w+", positive, re.I)
    )


def strip_unrequested_default_background_tags(text: str, user_text: str) -> str:
    """Remove white/simple background tags invented despite an explicit scene."""
    request = _positive_background_text(user_text)
    keep_white = bool(
        re.search(r"(?:纯白|白色|白底)(?:的)?背景|\bwhite background\b", request, re.I)
    )
    keep_simple = bool(
        re.search(r"(?:简单|简约|简洁)(?:的)?背景|\bsimple background\b", request, re.I)
    )
    kept: list[str] = []
    for part in str(text or "").split(","):
        tag = part.strip()
        normalized = tag.lower().replace("_", " ")
        if normalized == "white background" and not keep_white:
            continue
        if normalized == "simple background" and not keep_simple:
            continue
        if tag:
            kept.append(tag)
    return ", ".join(kept)


def enforce_user_background_intent(
    text: str, background_mode: str, user_text: str
) -> tuple[str, str, bool]:
    """Let an explicit user scene override an incorrect LLM portrait decision."""
    if not user_requests_explicit_background(user_text):
        return str(text or ""), background_mode, False
    return (
        strip_unrequested_default_background_tags(text, user_text),
        EXPLICIT_SCENE,
        background_mode != EXPLICIT_SCENE,
    )


def extract_background_mode(text: str) -> tuple[str, str]:
    """Remove the LLM control marker and return its background decision.

    Args:
        text: Raw comma-separated LLM output.

    Returns:
        A pair containing cleaned tags and the normalized background mode. The
        mode is empty when the LLM omitted or contradicted the control marker.
    """
    raw = str(text or "").strip()
    found = {
        marker
        for marker in (DEFAULT_PORTRAIT_MARKER, EXPLICIT_SCENE_MARKER)
        if re.search(rf"(?<![\w]){re.escape(marker)}(?![\w])", raw, flags=re.I)
    }
    cleaned = raw
    for marker in (DEFAULT_PORTRAIT_MARKER, EXPLICIT_SCENE_MARKER):
        cleaned = re.sub(
            rf"\s*,?\s*(?<![\w]){re.escape(marker)}(?![\w])\s*,?\s*",
            ", ",
            cleaned,
            flags=re.I,
        )
    cleaned = re.sub(r"(?:\s*,\s*){2,}", ", ", cleaned).strip(" ,")
    if found == {DEFAULT_PORTRAIT_MARKER}:
        return cleaned, DEFAULT_PORTRAIT
    if found == {EXPLICIT_SCENE_MARKER}:
        return cleaned, EXPLICIT_SCENE
    return cleaned, ""


def apply_default_portrait_tags(
    text: str, *, include_full_body: bool = True
) -> str:
    """Ensure a clean white-background character illustration tag set.

    Args:
        text: Cleaned content tags produced by the normal prompt pipeline.

    Returns:
        Tags with compatible portrait framing and the required simple white
        background. Existing explicit framing is preserved.
    """
    tags = [part.strip() for part in str(text or "").split(",") if part.strip()]
    normalized = {tag.lower().replace("_", " ") for tag in tags}
    additions: list[str] = []
    if (
        include_full_body
        and not normalized.intersection(_FRAMING_TAGS)
        and "full body" not in normalized
    ):
        additions.append("full body")
    if "centered" not in normalized:
        additions.append("centered")
    for tag in ("simple background", "white background"):
        if tag not in normalized:
            additions.append(tag)
    return ", ".join((*tags, *additions))
